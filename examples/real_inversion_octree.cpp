// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file real_inversion_octree.cpp
/// @brief 3D MT inversion driver using Octree FV backend.
///
/// Usage:
///   mpirun -np <N> real_inversion_octree --edi-dir <path> [options]
///
/// Loads EDI files, builds octree mesh with local refinement,
/// runs NLCG or GN-CG inversion, exports VTK per iteration.

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/forward/forward_solver_fv.h"
#include "maple3dmt/inversion/inversion_fv.h"
#include "maple3dmt/inversion/regularization_octree.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/io/edi_io.h"
#include "maple3dmt/io/vtk_export_octree.h"
#include "maple3dmt/mesh/dem.h"   // ALOSDem (DEM loading & interpolation)
#include "maple3dmt/utils/logger.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <mpi.h>

using namespace maple3dmt;
using Clock = std::chrono::steady_clock;

namespace {

void print_usage() {
    std::cout <<
R"(Usage:
  mpirun -np <N> real_inversion_octree --edi-dir <path> [options]

Required:
  --edi-dir <path>          EDI file directory

I/O options:
  --dem <path>              DEM file (ASCII xyz or GeoTIFF)
  --bathymetry <path>       Bathymetry DEM (GEBCO/ETOPO) for ocean regions.
                            Merged with --dem: land from primary, ocean floor
                            from bathymetry. Required for coastal surveys.
  --resume                  Resume from last iteration in output-dir

Mesh options:
  --domain-size <km>        Domain half-width in km (default: auto from skin depth)
  --min-level <n>           Minimum octree level (default: 3)
  --max-level <n>           Maximum octree level (default: 7)
  --station-refine <n>      Refinement level near stations (default: max-level)

Model options:
  --sigma <val>             Starting conductivity (S/m, default: 0.01)
  --sigma-ocean <val>       Seawater conductivity (S/m, default: 3.3)

Inversion options:
  --solver <type>           nlcg, lbfgs, or gn-cg (default: nlcg)
  --niter <n>               Max iterations (default: 50)
  --target-rms <val>        Target RMS misfit (default: 1.0)
  --lambda <val>            Initial lambda (default: 10.0)
  --lambda-dec <val>        Lambda decrease factor (default: 0.6)
  --lambda-strategy <type>  plateau (default) or ratio
  --plateau-tol <val>       |ΔRMS/RMS| threshold for plateau detection (default: 0.02)
  --plateau-patience <n>    Consecutive slow iters before lambda decrease (default: 2)
  --plateau-dec <val>       Lambda multiplier on plateau (default: 0.5)
  --cg-maxiter <n>          GN inner CG max iterations (default: 20)

Data selection:
  --error-floor <pct>       Impedance error floor (%, default: 5)
  --fmin <val>              Min frequency Hz (default: data min)
  --fmax <val>              Max frequency Hz (default: data max)

Regularisation:
  --alpha-s <val>           Smallness weight (default: 1.0)
  --alpha-x <val>           X-smoothing (default: 1.0)
  --alpha-y <val>           Y-smoothing (default: 1.0)
  --alpha-z <val>           Z-smoothing (default: 0.5)

Mesh (advanced):
  --max-depth <km>          Depth of interest limit in km (default: auto from skin depth)
                            Below this depth, mesh becomes very coarse.

Solver:
  --bicgstab-tol <val>      BiCGStab tolerance (default: 1e-7)
  --bicgstab-maxiter <n>    BiCGStab max iterations (default: 5000)

Output:
  --output-dir <path>       Output directory (default: output_octree)
  --vtk-interval <n>        VTK export every N iterations (default: 1)
)" << std::endl;
}

std::string get_arg(int argc, char* argv[], const std::string& flag, const std::string& def = "") {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag)
            return argv[i + 1];
    }
    return def;
}

bool has_flag(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    utils::Logger::instance().set_rank(rank);

    if (argc < 2 || has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        if (rank == 0) print_usage();
        MPI_Finalize();
        return 0;
    }

    // ---------------------------------------------------------------
    // Parse arguments
    // ---------------------------------------------------------------
    std::string edi_dir        = get_arg(argc, argv, "--edi-dir");
    std::string dem_path_str   = get_arg(argc, argv, "--dem");
    std::string bathy_path_str = get_arg(argc, argv, "--bathymetry");
    std::string output_dir     = get_arg(argc, argv, "--output-dir", "output_octree");
    bool        do_resume      = has_flag(argc, argv, "--resume");
    std::string solver_type    = get_arg(argc, argv, "--solver", "nlcg");

    Real sigma0        = std::stod(get_arg(argc, argv, "--sigma", "0.01"));
    Real sigma_ocean   = std::stod(get_arg(argc, argv, "--sigma-ocean", "3.3"));
    int  niter         = std::stoi(get_arg(argc, argv, "--niter", "50"));
    Real target_rms    = std::stod(get_arg(argc, argv, "--target-rms", "1.0"));
    Real lambda0       = std::stod(get_arg(argc, argv, "--lambda", "10.0"));
    Real lambda_dec    = std::stod(get_arg(argc, argv, "--lambda-dec", "0.6"));
    std::string lambda_strat = get_arg(argc, argv, "--lambda-strategy", "plateau");
    Real plateau_tol   = std::stod(get_arg(argc, argv, "--plateau-tol", "0.02"));
    int  plateau_pat   = std::stoi(get_arg(argc, argv, "--plateau-patience", "2"));
    Real plateau_dec   = std::stod(get_arg(argc, argv, "--plateau-dec", "0.5"));
    Real error_floor   = std::stod(get_arg(argc, argv, "--error-floor", "5")) / 100.0;

    int  min_level     = std::stoi(get_arg(argc, argv, "--min-level", "3"));
    int  max_level     = std::stoi(get_arg(argc, argv, "--max-level", "7"));
    int  sta_refine    = std::stoi(get_arg(argc, argv, "--station-refine",
                                           std::to_string(max_level)));
    Real max_depth_km  = std::stod(get_arg(argc, argv, "--max-depth", "0"));

    Real alpha_s = std::stod(get_arg(argc, argv, "--alpha-s", "1.0"));
    Real alpha_x = std::stod(get_arg(argc, argv, "--alpha-x", "1.0"));
    Real alpha_y = std::stod(get_arg(argc, argv, "--alpha-y", "1.0"));
    Real alpha_z = std::stod(get_arg(argc, argv, "--alpha-z", "0.5"));

    Real bicgstab_tol  = std::stod(get_arg(argc, argv, "--bicgstab-tol", "1e-7"));
    int  bicgstab_max  = std::stoi(get_arg(argc, argv, "--bicgstab-maxiter", "5000"));
    bool use_divcorr   = (get_arg(argc, argv, "--no-divcorr") == "") ? true : false;
    int  vtk_interval  = std::stoi(get_arg(argc, argv, "--vtk-interval", "1"));
    int  cg_maxiter    = std::stoi(get_arg(argc, argv, "--cg-maxiter", "20"));

    Real fmin_select = std::stod(get_arg(argc, argv, "--fmin", "0"));
    Real fmax_select = std::stod(get_arg(argc, argv, "--fmax", "0"));

    if (edi_dir.empty()) {
        if (rank == 0) {
            std::cerr << "ERROR: --edi-dir is required\n";
            print_usage();
        }
        MPI_Finalize();
        return 1;
    }

    // ---------------------------------------------------------------
    // 1. Load EDI data
    // ---------------------------------------------------------------
    if (rank == 0)
        std::cout << "\n=== Octree FV 3D MT Inversion ===" << std::endl;

    data::MTData observed;
    io::load_edi_directory(edi_dir, observed);

    if (observed.num_stations() == 0) {
        if (rank == 0) std::cerr << "ERROR: No EDI files loaded from " << edi_dir << "\n";
        MPI_Finalize();
        return 1;
    }

    // Apply error floor
    if (error_floor > 0)
        observed.apply_error_floor(error_floor, 0.02);

    int ns = observed.num_stations();
    int nf_data = observed.num_frequencies();

    // Frequency selection: rebuild MTData with only selected frequencies
    if (fmin_select > 0 || fmax_select > 0) {
        Real flo = (fmin_select > 0) ? fmin_select : 0.0;
        Real fhi = (fmax_select > 0) ? fmax_select : 1e30;

        // Identify valid frequency indices
        std::vector<int> keep_fi;
        RealVec keep_freqs;
        for (int fi = 0; fi < nf_data; ++fi) {
            Real f = observed.frequencies()[fi];
            if (f >= flo && f <= fhi) {
                keep_fi.push_back(fi);
                keep_freqs.push_back(f);
            }
        }

        if (static_cast<int>(keep_fi.size()) < nf_data) {
            data::MTData filtered;
            filtered.set_frequencies(keep_freqs);
            for (int si = 0; si < ns; ++si) {
                filtered.add_station(observed.station(si));
                for (int ki = 0; ki < static_cast<int>(keep_fi.size()); ++ki) {
                    filtered.set_observed(si, ki, observed.observed(si, keep_fi[ki]));
                }
            }
            int dropped = nf_data - static_cast<int>(keep_fi.size());
            observed = std::move(filtered);
            ns = observed.num_stations();
            nf_data = observed.num_frequencies();

            if (rank == 0)
                std::cout << "  Frequency filter: kept " << nf_data
                          << " freqs in [" << flo << ", " << fhi << "] Hz (dropped "
                          << dropped << ")" << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << "  Loaded EDI files: "
                  << ns << " stations, " << nf_data << " frequencies" << std::endl;
        std::cout << "  Freq range: " << observed.frequencies().front()
                  << " — " << observed.frequencies().back() << " Hz" << std::endl;
    }

    // ---------------------------------------------------------------
    // 1b. Convert lat/lon to local meters (centroid-centered)
    // ---------------------------------------------------------------
    // Find centroid of all stations in geographic coords
    Real lat0 = 0, lon0 = 0;
    for (int s = 0; s < ns; ++s) {
        lat0 += observed.station(s).lat;
        lon0 += observed.station(s).lon;
    }
    lat0 /= ns;
    lon0 /= ns;

    // Approximate meter-per-degree at centroid latitude
    Real m_per_deg_lat = 110540.0;
    Real m_per_deg_lon = 111320.0 * std::cos(lat0 * constants::PI / 180.0);

    std::vector<std::array<Real,3>> station_xyz(ns);
    for (int s = 0; s < ns; ++s) {
        const auto& st = observed.station(s);
        Real x = (st.lon - lon0) * m_per_deg_lon;
        Real y = (st.lat - lat0) * m_per_deg_lat;
        Real z = st.z;  // elevation (0 if not set)
        station_xyz[s] = {x, y, z};

        // Write local coordinates back to MTData so that forward solver
        // find_stations_() and VTK export use the correct positions.
        observed.station_mut(s).x = x;
        observed.station_mut(s).y = y;
        // Keep st.z (elevation) — already set from EDI ELEV field
    }

    if (rank == 0) {
        std::cout << "  Centroid: lat=" << lat0 << " lon=" << lon0 << std::endl;
        // Station spread
        Real sx_min = station_xyz[0][0], sx_max = station_xyz[0][0];
        Real sy_min = station_xyz[0][1], sy_max = station_xyz[0][1];
        for (int s = 1; s < ns; ++s) {
            sx_min = std::min(sx_min, station_xyz[s][0]);
            sx_max = std::max(sx_max, station_xyz[s][0]);
            sy_min = std::min(sy_min, station_xyz[s][1]);
            sy_max = std::max(sy_max, station_xyz[s][1]);
        }
        std::cout << "  Station spread: x=[" << sx_min/1000 << ", " << sx_max/1000
                  << "] km, y=[" << sy_min/1000 << ", " << sy_max/1000 << "] km" << std::endl;
    }

    // ---------------------------------------------------------------
    // 2. Build octree mesh
    // ---------------------------------------------------------------
    // Domain size: max of (5 × skin depth, 3 × station extent)
    Real fmin_data = *std::min_element(observed.frequencies().begin(),
                                        observed.frequencies().end());
    Real delta_max = std::sqrt(2.0 / (constants::TWOPI * fmin_data *
                                       constants::MU0 * sigma0));

    // Station extent
    Real sx_range = 0, sy_range = 0;
    for (int s = 0; s < ns; ++s) {
        sx_range = std::max(sx_range, std::abs(station_xyz[s][0]));
        sy_range = std::max(sy_range, std::abs(station_xyz[s][1]));
    }
    Real sta_extent = std::max(sx_range, sy_range);

    // Station bounding box (for domain sizing and skin-depth refinement)
    Real sx_min = station_xyz[0][0], sx_max = station_xyz[0][0];
    Real sy_min = station_xyz[0][1], sy_max = station_xyz[0][1];
    for (int s = 1; s < ns; ++s) {
        sx_min = std::min(sx_min, station_xyz[s][0]);
        sx_max = std::max(sx_max, station_xyz[s][0]);
        sy_min = std::min(sy_min, station_xyz[s][1]);
        sy_max = std::max(sy_max, station_xyz[s][1]);
    }
    Real sta_spread_x = sx_max - sx_min;
    Real sta_spread_y = sy_max - sy_min;
    Real sta_spread = std::max(sta_spread_x, sta_spread_y);

    // Minimum station spacing (for finest cell size reference)
    Real min_sta_spacing = 1e30;
    for (int i = 0; i < ns; ++i) {
        for (int j = i+1; j < ns; ++j) {
            Real dx = station_xyz[i][0] - station_xyz[j][0];
            Real dy = station_xyz[i][1] - station_xyz[j][1];
            Real dist = std::sqrt(dx*dx + dy*dy);
            if (dist > 100) min_sta_spacing = std::min(min_sta_spacing, dist);
        }
    }
    if (min_sta_spacing > 1e20) min_sta_spacing = 1000;

    // Auto domain sizing strategy (ModEM-style):
    //   Horizontal: station extent + 2×skin_depth padding on each side
    //   Depth (z<0): 3×skin_depth (sensitivity decays exponentially)
    //   Air (z>0): 1×skin_depth (enough for boundary conditions)
    //
    // Key principle: domain should be large enough for BC to be valid,
    // but NOT so large that coarse cells dominate.
    Real domain_from_stations = sta_extent + 2.0 * delta_max;
    Real domain_from_skin     = 3.0 * delta_max;
    Real domain_half = std::max(domain_from_stations, domain_from_skin);
    // Must contain all stations with comfortable padding
    domain_half = std::max(domain_half, sta_extent + 1.5 * delta_max);

    // Z-direction: asymmetric — deeper into earth, less air
    Real z_depth = 3.0 * delta_max;  // 3 skin depths below surface
    Real z_air   = delta_max;         // 1 skin depth above surface

    std::string domain_str = get_arg(argc, argv, "--domain-size", "0");
    if (std::stod(domain_str) > 0)
        domain_half = std::stod(domain_str) * 1000.0;  // km to m

    if (rank == 0) {
        std::cout << "  Skin depth (fmin=" << fmin_data << " Hz): "
                  << delta_max/1000.0 << " km" << std::endl;
        std::cout << "  Station spread: " << sta_spread/1000.0 << " km" << std::endl;
        std::cout << "  Min station spacing: " << min_sta_spacing/1000.0 << " km" << std::endl;
        std::cout << "  Finest cell at L" << max_level << ": "
                  << 2.0*domain_half/std::pow(2.0, max_level)/1000.0 << " km" << std::endl;
        std::cout << "  Domain: ±" << domain_half/1000.0 << " km (h), "
                  << -z_depth/1000.0 << " to +" << z_air/1000.0 << " km (z)" << std::endl;
    }

    octree::RefinementParams mesh_params;
    mesh_params.domain_x_min = -domain_half;  mesh_params.domain_x_max = domain_half;
    mesh_params.domain_y_min = -domain_half;  mesh_params.domain_y_max = domain_half;
    mesh_params.domain_z_min = -z_depth;      mesh_params.domain_z_max = z_air;
    mesh_params.min_level = min_level;
    mesh_params.max_level = max_level;
    // Station refinement radius: enough to capture the impedance (E and H)
    // at each station, but not so large that L_max cells fill huge volumes.
    // Strategy: max of (3× finest cell, min station spacing, 2 km floor)
    Real cell_finest = 2.0 * domain_half / std::pow(2.0, max_level);
    Real sta_refine_r = std::max({3.0 * cell_finest, min_sta_spacing, 2000.0});
    // Cap at 0.1× skin depth to avoid excessive refinement
    sta_refine_r = std::min(sta_refine_r, delta_max * 0.1);
    mesh_params.station_refine_radius = sta_refine_r;
    if (rank == 0)
        std::cout << "  Station refine radius: " << sta_refine_r/1000.0 << " km" << std::endl;
    mesh_params.station_refine_level = sta_refine;
    mesh_params.sigma_bg = sigma0;
    // Depth-of-interest limit: below this, mesh becomes very coarse
    if (max_depth_km > 0) {
        mesh_params.max_interest_depth = max_depth_km * 1000.0;  // km → m
        if (rank == 0)
            std::cout << "  Max depth of interest: " << max_depth_km << " km" << std::endl;
    }
    mesh_params.replicate_mesh = (nproc > 1);  // MPI freq-parallel: all ranks own full mesh
    // Station bounds for skin-depth refinement
    mesh_params.station_x_min = sx_min;
    mesh_params.station_x_max = sx_max;
    mesh_params.station_y_min = sy_min;
    mesh_params.station_y_max = sy_max;

    auto t0 = Clock::now();

    // Load DEM if provided
    std::unique_ptr<mesh::ALOSDem> dem_ptr;
    if (!dem_path_str.empty()) {
        dem_ptr = std::make_unique<mesh::ALOSDem>();
        fs::path dem_path(dem_path_str);
        try {
            std::string dem_ext = dem_path.extension().string();
            if (dem_ext == ".tif" || dem_ext == ".tiff") {
                // Prefer ASCII sidecar if available
                fs::path ascii_dem = fs::path(dem_path).replace_extension(".txt");
                if (fs::exists(ascii_dem)) {
                    if (rank == 0)
                        std::cout << "  Using ASCII DEM: " << ascii_dem.string() << "\n";
                    dem_ptr->load_ascii(ascii_dem);
                } else {
                    dem_ptr->load_geotiff(dem_path);
                }
            } else {
                dem_ptr->load_ascii(dem_path);
            }
            if (rank == 0)
                std::cout << "  DEM loaded: " << dem_ptr->n_lon << "x"
                          << dem_ptr->n_lat << " grid" << std::endl;

            // Check DEM coverage against station bounding box
            if (rank == 0) {
                Real slon_min = 1e30, slon_max = -1e30;
                Real slat_min = 1e30, slat_max = -1e30;
                for (int s = 0; s < ns; ++s) {
                    slon_min = std::min(slon_min, observed.station(s).lon);
                    slon_max = std::max(slon_max, observed.station(s).lon);
                    slat_min = std::min(slat_min, observed.station(s).lat);
                    slat_max = std::max(slat_max, observed.station(s).lat);
                }
                // Add small margin (0.01 deg ≈ 1km)
                dem_ptr->check_coverage(slon_min - 0.01, slon_max + 0.01,
                                        slat_min - 0.01, slat_max + 0.01);
            }
        } catch (const std::exception& e) {
            if (rank == 0)
                std::cerr << "  DEM load failed: " << e.what()
                          << "\n  Proceeding without terrain.\n";
            dem_ptr.reset();
        }
    }

    // Load bathymetry DEM (GEBCO/ETOPO) and merge with primary
    if (!bathy_path_str.empty() && dem_ptr) {
        auto bathy_ptr = std::make_shared<mesh::ALOSDem>();
        fs::path bathy_path(bathy_path_str);
        try {
            std::string bext = bathy_path.extension().string();
            if (bext == ".tif" || bext == ".tiff") {
                fs::path ascii_b = fs::path(bathy_path).replace_extension(".txt");
                if (fs::exists(ascii_b))
                    bathy_ptr->load_ascii(ascii_b);
                else
                    bathy_ptr->load_geotiff(bathy_path);
            } else {
                bathy_ptr->load_ascii(bathy_path);
            }
            if (rank == 0)
                std::cout << "  Bathymetry DEM loaded: " << bathy_ptr->n_lon << "x"
                          << bathy_ptr->n_lat << " grid" << std::endl;

            dem_ptr->set_bathymetry(bathy_ptr);

            // Diagnose land-ocean transition quality
            if (rank == 0) {
                Real slon_min = 1e30, slon_max = -1e30;
                Real slat_min = 1e30, slat_max = -1e30;
                for (int s = 0; s < ns; ++s) {
                    slon_min = std::min(slon_min, observed.station(s).lon);
                    slon_max = std::max(slon_max, observed.station(s).lon);
                    slat_min = std::min(slat_min, observed.station(s).lat);
                    slat_max = std::max(slat_max, observed.station(s).lat);
                }
                Real margin = 0.05;  // ~5 km margin for coastline sampling
                dem_ptr->diagnose_coastline(
                    slon_min - margin, slon_max + margin,
                    slat_min - margin, slat_max + margin);
            }
        } catch (const std::exception& e) {
            if (rank == 0)
                std::cerr << "  Bathymetry load failed: " << e.what()
                          << "\n  Proceeding without ocean bathymetry.\n";
        }
    } else if (!bathy_path_str.empty() && !dem_ptr) {
        // Bathymetry specified but no primary DEM — use bathymetry as primary
        dem_ptr = std::make_unique<mesh::ALOSDem>();
        fs::path bathy_path(bathy_path_str);
        try {
            std::string bext = bathy_path.extension().string();
            if (bext == ".tif" || bext == ".tiff")
                dem_ptr->load_geotiff(bathy_path);
            else
                dem_ptr->load_ascii(bathy_path);
            if (rank == 0)
                std::cout << "  Using bathymetry DEM as primary: "
                          << dem_ptr->n_lon << "x" << dem_ptr->n_lat << " grid" << std::endl;
        } catch (const std::exception& e) {
            if (rank == 0)
                std::cerr << "  Bathymetry load failed: " << e.what() << "\n";
            dem_ptr.reset();
        }
    }

    octree::OctreeMesh mesh;
    mesh.setup(mesh_params, station_xyz, observed.frequencies(), MPI_COMM_WORLD);

    if (dem_ptr) {
        // Capture DEM + coordinate transform for terrain function.
        // DEM uses geographic coords (lon, lat); mesh uses local meters.
        // Convert mesh (x, y) back to (lon, lat) using the same centroid.
        auto* dem_raw = dem_ptr.get();
        mesh.set_terrain([dem_raw, lon0, lat0, m_per_deg_lon, m_per_deg_lat]
                         (Real x, Real y) -> Real {
            Real lon = lon0 + x / m_per_deg_lon;
            Real lat = lat0 + y / m_per_deg_lat;
            return dem_raw->interpolate(lon, lat);
        });
    } else {
        mesh.set_terrain([](Real, Real) { return 0.0; });
    }

    // Validate: all stations must be in Earth cells, not Air.
    // If DEM is inaccurate or missing, some stations may end up in Air,
    // which makes forward modelling invalid. Fix by reclassifying.
    int n_air_stations = mesh.validate_stations_earth(station_xyz);
    if (n_air_stations > 0 && rank == 0) {
        std::cerr << "  WARNING: " << n_air_stations << " stations were in AIR cells.\n"
                  << "  Cells reclassified to EARTH. Check DEM coverage.\n";
    }

    mesh.build_staggered_grid();

    int nc = mesh.staggered().num_cells();
    int ne = mesh.staggered().num_edges();

    auto t1 = Clock::now();
    Real mesh_time = std::chrono::duration<Real>(t1 - t0).count();

    if (rank == 0) {
        std::cout << "  Mesh: " << nc << " cells, " << ne << " DOFs"
                  << " (" << mesh_time << "s)" << std::endl;
    }

    // ---------------------------------------------------------------
    // 3. Setup conductivity model
    // ---------------------------------------------------------------
    model::ConductivityModel model;
    model.init_3d(nc, sigma0);

    auto& log_sigma = model.params();
    int n_ocean_cells = 0;
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) == octree::CellType::AIR) {
            log_sigma[c] = std::log(1e-8);
        } else if (mesh.cell_type(c) == octree::CellType::OCEAN) {
            log_sigma[c] = std::log(sigma_ocean);
            ++n_ocean_cells;
        }
        // EARTH cells keep sigma0 (background)
    }
    model.invalidate_cache();

    if (rank == 0 && n_ocean_cells > 0) {
        std::cout << "  Ocean cells: " << n_ocean_cells
                  << " (σ=" << sigma_ocean << " S/m)" << std::endl;
    }

    // ---------------------------------------------------------------
    // 4. Setup forward solver
    // ---------------------------------------------------------------
    forward::ForwardParamsFV fwd_params;
    fwd_params.bicgstab_tol = bicgstab_tol;
    fwd_params.bicgstab_maxiter = bicgstab_max;
    fwd_params.print_level = 0;
    fwd_params.div_correction = use_divcorr;

    forward::ForwardSolverFV fwd;
    fwd.setup(mesh, fwd_params);

    if (rank == 0) {
        std::cout << "  Forward solver: BiCGStab+SSOR"
                  << " tol=" << bicgstab_tol
                  << " maxiter=" << bicgstab_max
                  << " DivCorr=" << (use_divcorr ? "ON" : "OFF")
                  << std::endl;
    }

    // ---------------------------------------------------------------
    // 5. Setup regularisation
    // ---------------------------------------------------------------
    inversion::RegParamsOctree reg_params;
    reg_params.alpha_s = alpha_s;
    reg_params.alpha_x = alpha_x;
    reg_params.alpha_y = alpha_y;
    reg_params.alpha_z = alpha_z;

    inversion::RegularizationOctree reg;
    reg.setup(mesh, reg_params);

    // Free face_neighbors_ — no longer needed after StaggeredGrid + Reg setup.
    // Saves ~144 bytes/cell = ~8.6 MB for 60K cells, ~86 MB for 600K cells.
    mesh.release_face_neighbors();

    // ---------------------------------------------------------------
    // 6. Setup inversion
    // ---------------------------------------------------------------
    inversion::InversionParamsFV inv_params;
    if (solver_type == "gn-cg")
        inv_params.solver = inversion::InversionParamsFV::Solver::GN_CG;
    else if (solver_type == "lbfgs")
        inv_params.solver = inversion::InversionParamsFV::Solver::LBFGS;
    else
        inv_params.solver = inversion::InversionParamsFV::Solver::NLCG;
    inv_params.max_iterations = niter;
    inv_params.target_rms = target_rms;
    inv_params.lambda_init = lambda0;
    inv_params.lambda_decrease = lambda_dec;
    inv_params.cg_max_iter = cg_maxiter;

    // Lambda strategy
    if (lambda_strat == "ratio")
        inv_params.lambda_strategy = inversion::InversionParamsFV::LambdaStrategy::RATIO;
    else
        inv_params.lambda_strategy = inversion::InversionParamsFV::LambdaStrategy::PLATEAU;
    inv_params.plateau_tol = plateau_tol;
    inv_params.plateau_patience = plateau_pat;
    inv_params.plateau_decrease = plateau_dec;

    inversion::InversionFV inv;
    inv.setup(model, observed, fwd, reg, inv_params);

    if (rank == 0) {
        std::cout << "\n  Solver: " << solver_type << std::endl;
        std::cout << "  Active params: " << reg.n_active() << std::endl;
        std::cout << "  Lambda_0: " << lambda0 << std::endl;
    }

    // ---------------------------------------------------------------
    // 7. Export initial model
    // ---------------------------------------------------------------
    // Compute reasonable slice resolution based on station extent
    Real slice_res = std::max(500.0, sta_extent * 0.02);  // ~2% of station extent
    io::OctreeExportParams export_params;
    export_params.auto_slice_interval = 5000;   // 5 km between slices
    export_params.slice_dx = slice_res;
    export_params.slice_dy = slice_res;

    // Initial model: VTU only (no slow slices)
    {
        io::OctreeExportParams init_params = export_params;
        init_params.auto_slice_interval = 0;  // disable slices for initial model
        io::export_octree_all(mesh, model, observed, output_dir, init_params, 0);
    }

    // ---------------------------------------------------------------
    // 8. Inversion loop
    // ---------------------------------------------------------------
    if (rank == 0)
        std::cout << "\n--- Inversion loop ---\n" << std::endl;

    auto t_inv_start = Clock::now();
    auto t_last_iter = t_inv_start;

    inv.set_iteration_callback(
        [&](int iter, const inversion::IterationLogFV& entry) {
            auto t_now = Clock::now();
            Real iter_time = std::chrono::duration<Real>(t_now - t_last_iter).count();
            t_last_iter = t_now;

            if (rank == 0) {
                std::cout << "  Iter " << entry.iteration
                          << "  Φ=" << std::scientific << std::setprecision(4) << entry.objective
                          << "  RMS=" << std::fixed << std::setprecision(3) << entry.rms
                          << "  λ=" << std::scientific << std::setprecision(2) << entry.lambda
                          << "  time=" << std::fixed << std::setprecision(1) << iter_time << "s"
                          << std::endl;
            }

            // Export VTK (VTU only per iteration — slices only at final)
            if (iter % vtk_interval == 0) {
                io::OctreeExportParams iter_params = export_params;
                iter_params.export_depth_slices = false;
                iter_params.export_stations_csv = false;
                io::export_octree_all(mesh, model, observed, output_dir, iter_params, iter);
            }

            // Export data fit CSV every iteration (rank 0 only)
            // In freq-parallel mode, each rank only has predicted data for its
            // assigned frequencies. Allreduce to collect all predictions on rank 0.
            //
            // BUG FIX: Only pack frequencies assigned to this rank into the
            // allreduce buffer. Previously, ALL frequencies were packed, including
            // stale allreduced values from the previous callback. Since MPI_SUM
            // is used, each non-assigned frequency's stale value (already the sum
            // of all ranks) was summed again across all ranks, causing exponential
            // growth: Z_displayed ≈ nproc^iter × Z_true.
            if (nproc > 1 && inv.is_freq_parallel()) {
                int ns_loc = observed.num_stations();
                int nf_loc = observed.num_frequencies();
                int buf_size = ns_loc * nf_loc * 8;  // 4 Z components × 2 (re/im)

                // Build set of this rank's assigned frequencies
                const auto& my_freqs = inv.freq_manager().my_frequency_indices();
                std::vector<bool> is_my_freq(nf_loc, false);
                for (int fi : my_freqs) is_my_freq[fi] = true;

                std::vector<double> local_buf(buf_size, 0.0);
                // Pack ONLY this rank's frequencies — others stay zero
                for (int s = 0; s < ns_loc; ++s) {
                    for (int f = 0; f < nf_loc; ++f) {
                        if (!is_my_freq[f]) continue;  // leave as zero
                        const auto& p = observed.predicted(s, f);
                        int base = (s * nf_loc + f) * 8;
                        local_buf[base + 0] = p.Zxx.value.real();
                        local_buf[base + 1] = p.Zxx.value.imag();
                        local_buf[base + 2] = p.Zxy.value.real();
                        local_buf[base + 3] = p.Zxy.value.imag();
                        local_buf[base + 4] = p.Zyx.value.real();
                        local_buf[base + 5] = p.Zyx.value.imag();
                        local_buf[base + 6] = p.Zyy.value.real();
                        local_buf[base + 7] = p.Zyy.value.imag();
                    }
                }
                std::vector<double> global_buf(buf_size, 0.0);
                MPI_Allreduce(local_buf.data(), global_buf.data(), buf_size,
                              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                // Unpack to a temporary — do NOT write back to observed.predicted
                // to avoid contaminating the next iteration's forward solve.
                // Instead, create a temporary MTData copy for CSV export.
                for (int s = 0; s < ns_loc; ++s) {
                    for (int f = 0; f < nf_loc; ++f) {
                        int base = (s * nf_loc + f) * 8;
                        data::MTResponse resp;
                        resp.Zxx.value = Complex(global_buf[base+0], global_buf[base+1]);
                        resp.Zxy.value = Complex(global_buf[base+2], global_buf[base+3]);
                        resp.Zyx.value = Complex(global_buf[base+4], global_buf[base+5]);
                        resp.Zyy.value = Complex(global_buf[base+6], global_buf[base+7]);
                        observed.set_predicted(s, f, resp);
                    }
                }
            } else if (nproc > 1) {
                // Non-freq-parallel MPI: all ranks have same predicted data,
                // no allreduce needed (rank 0 can export directly).
            }
            if (rank == 0) {
                io::export_data_fit_csv(observed,
                                        output_dir + "/data_fit_iter" +
                                        std::to_string(iter) + ".csv",
                                        iter);

                // Save inversion state for resume
                std::vector<std::pair<int, Real>> rms_hist;
                for (const auto& h : inv.history())
                    rms_hist.push_back({h.iteration, h.rms});
                io::save_inversion_state(
                    output_dir + "/inversion_state.json",
                    entry.iteration, entry.lambda, entry.rms, rms_hist);
            }
        });

    // Resume from previous run if requested
    if (do_resume) {
        io::InversionState state;
        std::string state_path = output_dir + "/inversion_state.json";
        if (io::load_inversion_state(state_path, state) && state.last_iteration > 0) {
            // Find the latest model VTU
            std::string vtu_path = output_dir + "/model_iter" +
                                   std::to_string(state.last_iteration) + ".vtu";
            if (fs::exists(vtu_path)) {
                RealVec saved_sigma = io::load_conductivity_from_vtu(vtu_path);
                if (static_cast<int>(saved_sigma.size()) == nc) {
                    // Restore model: write log(sigma) into params
                    for (int c = 0; c < nc; ++c) {
                        Real s = std::max(saved_sigma[c], Real(1e-30));
                        model.params()[c] = std::log(s);
                    }
                    model.invalidate_cache();

                    inv.resume_from(state.last_iteration, state.lambda);

                    if (rank == 0) {
                        std::cout << "\n  === RESUME from iter " << state.last_iteration
                                  << " (RMS=" << state.rms
                                  << ", lambda=" << state.lambda << ") ===" << std::endl;
                    }
                } else {
                    if (rank == 0)
                        std::cerr << "  Resume: cell count mismatch ("
                                  << saved_sigma.size() << " vs " << nc
                                  << "), starting fresh.\n";
                }
            } else {
                if (rank == 0)
                    std::cerr << "  Resume: " << vtu_path << " not found, starting fresh.\n";
            }
        } else {
            if (rank == 0)
                std::cout << "  Resume: no inversion_state.json found, starting fresh.\n";
        }
    }

    inv.run();

    // Export final model + data fit
    {
        const auto& hist = inv.history();
        if (!hist.empty()) {
            int final_iter = hist.back().iteration;
            io::export_octree_all(mesh, model, observed, output_dir, export_params,
                                  final_iter);
            if (rank == 0)
                io::export_data_fit_csv(observed,
                                    output_dir + "/data_fit_iter" +
                                    std::to_string(final_iter) + ".csv",
                                    final_iter);
        }
    }

    auto t_inv_end = Clock::now();
    Real total_time = std::chrono::duration<Real>(t_inv_end - t_inv_start).count();

    if (rank == 0) {
        std::cout << "\n  Total inversion time: " << total_time << "s" << std::endl;
        std::cout << "\n=== Done ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
