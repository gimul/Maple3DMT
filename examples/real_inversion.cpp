// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file real_inversion.cpp
/// @brief 3D MT inversion driver for real EDI field data.
///
/// Usage:
///   mpirun -np 12 real_inversion --edi-dir <path> [options]
///
/// Loads EDI files, builds 3D terrain-conforming mesh, runs CG-based
/// Gauss-Newton inversion, and exports results per iteration (VTK, slices).

#include "maple3dmt/inversion/inversion_3d.h"
#include "maple3dmt/mesh/hex_mesh_3d.h"
#include "maple3dmt/io/vtk_export_3d.h"
#include "maple3dmt/io/edi_io.h"
#include "maple3dmt/utils/logger.h"
#include "maple3dmt/utils/memory.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <mfem.hpp>

using namespace maple3dmt;
using Clock = std::chrono::steady_clock;

namespace {

void print_usage() {
    std::cout <<
R"(Usage:
  mpirun -np <N> real_inversion <edi_dir> [options]
  mpirun -np <N> real_inversion --edi-dir <path> [options]

Required:
  <edi_dir>               EDI file directory (positional or --edi-dir)

Mesh options:
  --dem <path>            DEM file (ASCII: lon lat elev)
  --h-surface <m>         Surface horizontal element size (0 = auto, default)
  --h-surface-z <m>       Surface vertical element size (0 = h_surface/5)
  --refine <n>            Local refinement near stations (default: 1)
  --air-start <m>         First air layer thickness (default: 500m)
  --growth-air <r>        Air layer growth rate (default: 2.0)

Inversion options:
  --solver <type>         Solver type: nlcg (default) or gn-cg
  --sigma <val>           Starting model conductivity (S/m, default: 0.01)
  --niter <n>             Max iterations (default: 50)
  --target-rms <val>      Target RMS misfit (default: 1.0)
  --lambda <val>          Initial regularisation parameter (default: 10.0)
  --lambda-dec <val>      Lambda decrease factor per iteration (default: 0.8)
  --cg-maxiter <n>        Max inner FGMRES iterations per GN step (default: 20)
  --cg-tol <val>          Inner FGMRES tolerance (default: 0.1)
  --cg-kdim <n>           Inner FGMRES Krylov dim (default: 20)

Data selection:
  --freq <f1,f2,...>      Comma-separated frequencies (Hz)
  --nfreq <n>             Number of log-spaced frequencies (default: all)
  --fmin <val>            Minimum frequency (Hz, default: data min)
  --fmax <val>            Maximum frequency (Hz, default: data max)
  --error-floor <pct>     Impedance error floor (%, default: 5)

Regularisation:
  --alpha-s <val>         Smoothness weight (default: 1.0)
  --alpha-x <val>         Horizontal smoothing (default: 1.0)
  --alpha-z <val>         Vertical smoothing (default: 1.0)

Solver options:
  --order <n>             FE element order (default: 1)
  --precond <name>        Preconditioner (default: AMS)
                          AMS   = DoubledAMS (optimal for edge elements)
                          AMG   = BoomerAMG (fallback)
                          DIAG  = diagonal scaling (baseline)

FGMRES:
  --gmres-kdim <n>        Krylov subspace dim (default: 150)
  --gmres-tol <val>       Relative tolerance (default: 2e-3)
  --gmres-maxiter <n>     Max iterations (default: 500)
  --gmres-print <n>       Print: -1=auto, 0=silent, 1=per-iter (default: -1)

Adjoint solver:
  --adjoint-tol <val>     Adjoint FGMRES tolerance (default: 0.1)
  --adjoint-maxiter <n>   Adjoint FGMRES max iterations (default: 500)
  --adjoint-kdim <n>      Adjoint Krylov dim (default: 500)
  --adjoint-div-corr      Enable divergence correction (default: ON)
  --no-adjoint-div-corr   Disable divergence correction

AMS preconditioner tuning:
  --ams-vcycles <n>       Max V-cycles at high freq (default: 3)
  --ams-omega-mid <val>   2 V-cycle threshold (default: 0.1)
  --ams-omega-high <val>  Max V-cycle threshold (default: 1.0)
  --ams-smooth-type <n>   0=Jacobi 2=l1-symGS 6=symGS (default: 2)
  --ams-smooth-sweeps <n> Sweeps per level (default: 2)

MPI parallelism:
  --spatial-procs <n>     Spatial procs per freq group (0=auto, default: 0)
                          Auto: maximizes frequency parallelism.
                          e.g. 64 procs, 81 freqs → 64 groups × 1 spatial

Output:
  --output <dir>          Output directory (default: inversion_output)
  --slice-depths <d1,...> Comma-separated depth slice levels (m)
  --slice-interval <m>    Auto depth slice every N metres (default: 2000)
  --profile <x0,y0,x1,y1>  Add vertical profile slice

General:
  --resume <dir>          Resume from checkpoint directory
  --help                  Show this message
)";
}

std::vector<Real> parse_csv_reals(const std::string& s) {
    std::vector<Real> vals;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (!token.empty()) vals.push_back(std::stod(token));
    }
    return vals;
}

std::vector<Real> log_spaced_freqs(Real fmin, Real fmax, int n) {
    std::vector<Real> freqs;
    if (n <= 0) return freqs;
    if (n == 1) { freqs.push_back(std::sqrt(fmin * fmax)); return freqs; }
    Real lmin = std::log10(fmin), lmax = std::log10(fmax);
    for (int i = 0; i < n; ++i) {
        freqs.push_back(std::pow(10.0, lmin + i * (lmax - lmin) / (n - 1)));
    }
    return freqs;
}

Real snap_to_data_freq(Real target, const RealVec& available) {
    Real best = available[0];
    Real best_d = std::abs(std::log10(target) - std::log10(best));
    for (size_t i = 1; i < available.size(); ++i) {
        Real d = std::abs(std::log10(target) - std::log10(available[i]));
        if (d < best_d) { best = available[i]; best_d = d; }
    }
    return best;
}

std::string fmt_elapsed(double sec) {
    if (sec < 60) return std::to_string(static_cast<int>(sec)) + " s";
    int m = static_cast<int>(sec) / 60;
    int s = static_cast<int>(sec) % 60;
    return std::to_string(m) + " min " + std::to_string(s) + " s";
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    mfem::Mpi::Init(argc, argv);
    int rank = mfem::Mpi::WorldRank();
    int nprocs = mfem::Mpi::WorldSize();

    // =================================================================
    // Parse arguments
    // =================================================================
    fs::path edi_dir;
    fs::path dem_path;
    fs::path resume_dir;
    std::string solver_type = "nlcg";  // default: NLCG (ModEM-style, --solver gn-cg for 2nd-order)
    Real sigma_bg        = 0.01;
    int  max_iter        = 50;
    Real target_rms      = 1.0;
    Real lambda_init     = 10.0;
    Real lambda_dec      = 0.8;
    int  cg_maxiter      = 20;
    int  cg_kdim         = 20;
    Real cg_tol          = 0.1;
    int  fe_order        = 1;
    Real h_surface       = 0;
    Real h_surface_z     = 0;     // 0 = auto (h_surface / 5)
    Real h_air_start     = 500;   // first air layer thickness (m)
    Real growth_air      = 2.0;   // air layer growth rate
    int  refine          = 1;
    Real error_floor_pct = 5.0;
    Real alpha_s         = 1.0;
    Real alpha_x         = 1.0;
    Real alpha_z         = 1.0;
    int  nfreq           = -1;   // -1 = all
    Real fmin_user       = 0;
    Real fmax_user       = 0;
    std::string output_dir = "inversion_output";
    std::vector<Real> user_freqs;
    std::vector<Real> slice_depths;
    Real slice_interval  = 2000;
    std::vector<std::array<Real,4>> profile_coords;  // (x0,y0,x1,y1)
    // Solver backend: COMPLEX=BiCGStab+ILU(0), ITERATIVE=FGMRES+AMS
    std::string solver_backend_ = "ITERATIVE";  // AMS needed for unstructured Nédélec
    std::string precond_str = "AMS";
    int  gmres_kdim_     = 30;    // Inner PCG(AMS) → ~10-15 outer iter, 30 sufficient
    Real gmres_tol_      = 2e-3;
    int  gmres_maxiter_  = 500;
    int  gmres_print_    = -1;
    // AMS preconditioner tuning
    int  ams_max_vcycles_   = 3;
    Real ams_omega_mid_     = 0.1;
    Real ams_omega_high_    = 1.0;
    int  ams_smooth_type_   = 6;    // hybrid-symGS/SSOR
    int  ams_smooth_sweeps_ = 3;
    // Adjoint solver (iterative + DivCorr)
    Real adjoint_tol_       = 0.1;
    int  adjoint_maxiter_   = 500;
    int  adjoint_kdim_      = 500;
    bool adjoint_div_corr_  = false;  // OFF by default: DivCorr outer loop proven harmful with AMS+CCGD
    bool adjoint_use_jacobi_ = true;   // true=Jacobi (fast), false=AMS
    int  adjoint_inner_iter_ = 50;     // Inner FGMRES iters per outer DivCorr step
    int  adjoint_outer_iter_ = 20;     // Max outer DivCorr iterations
    bool ccgd_enabled_       = true;
    Real ccgd_tau_           = 0.0;    // 0=auto
    // Frequency parallelism (2-level MPI)
    int  spatial_procs_      = 0;      // 0=auto (maximize freq-parallel)
    bool no_skin_mesh_       = false;  // --no-skin-mesh: disable skin-depth z-optimization

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : "";
        };
        if      (arg == "--edi-dir")       edi_dir = next();
        else if (arg == "--dem")           dem_path = next();
        else if (arg == "--solver")        solver_type = next();
        else if (arg == "--sigma")         sigma_bg = std::stod(next());
        else if (arg == "--niter")         max_iter = std::stoi(next());
        else if (arg == "--target-rms")    target_rms = std::stod(next());
        else if (arg == "--lambda")        lambda_init = std::stod(next());
        else if (arg == "--lambda-dec")    lambda_dec = std::stod(next());
        else if (arg == "--cg-maxiter")    cg_maxiter = std::stoi(next());
        else if (arg == "--cg-tol")        cg_tol = std::stod(next());
        else if (arg == "--cg-kdim")       cg_kdim = std::stoi(next());
        else if (arg == "--order")         fe_order = std::stoi(next());
        else if (arg == "--h-surface")     h_surface = std::stod(next());
        else if (arg == "--h-surface-z")   h_surface_z = std::stod(next());
        else if (arg == "--air-start")     h_air_start = std::stod(next());
        else if (arg == "--growth-air")    growth_air = std::stod(next());
        else if (arg == "--refine")        refine = std::stoi(next());
        else if (arg == "--error-floor")   error_floor_pct = std::stod(next());
        else if (arg == "--alpha-s")       alpha_s = std::stod(next());
        else if (arg == "--alpha-x")       alpha_x = std::stod(next());
        else if (arg == "--alpha-z")       alpha_z = std::stod(next());
        else if (arg == "--nfreq")         nfreq = std::stoi(next());
        else if (arg == "--fmin" || arg == "--freq-min")   fmin_user = std::stod(next());
        else if (arg == "--fmax" || arg == "--freq-max")   fmax_user = std::stod(next());
        else if (arg == "--freq")          user_freqs = parse_csv_reals(next());
        else if (arg == "--output")        output_dir = next();
        else if (arg == "--slice-depths")  slice_depths = parse_csv_reals(next());
        else if (arg == "--slice-interval") slice_interval = std::stod(next());
        else if (arg == "--resume")        resume_dir = next();
        else if (arg == "--precond")       precond_str = next();
        else if (arg == "--gmres-kdim")    gmres_kdim_ = std::stoi(next());
        else if (arg == "--gmres-tol")     gmres_tol_ = std::stod(next());
        else if (arg == "--gmres-maxiter") gmres_maxiter_ = std::stoi(next());
        else if (arg == "--adjoint-tol")       adjoint_tol_ = std::stod(next());
        else if (arg == "--adjoint-maxiter")   adjoint_maxiter_ = std::stoi(next());
        else if (arg == "--adjoint-kdim")      adjoint_kdim_ = std::stoi(next());
        else if (arg == "--adjoint-div-corr")  adjoint_div_corr_ = true;
        else if (arg == "--no-adjoint-div-corr") adjoint_div_corr_ = false;
        else if (arg == "--adjoint-prec") {
            std::string v = next();
            adjoint_use_jacobi_ = (v != "AMS" && v != "ams");
        }
        else if (arg == "--adjoint-inner-iter") adjoint_inner_iter_ = std::stoi(next());
        else if (arg == "--adjoint-outer-iter") adjoint_outer_iter_ = std::stoi(next());
        else if (arg == "--ccgd-tau")           ccgd_tau_ = std::stod(next());
        else if (arg == "--no-ccgd")            ccgd_enabled_ = false;
        else if (arg == "--ams-vcycles")       ams_max_vcycles_ = std::stoi(next());
        else if (arg == "--ams-omega-mid")     ams_omega_mid_ = std::stod(next());
        else if (arg == "--ams-omega-high")    ams_omega_high_ = std::stod(next());
        else if (arg == "--ams-smooth-type")   ams_smooth_type_ = std::stoi(next());
        else if (arg == "--ams-smooth-sweeps") ams_smooth_sweeps_ = std::stoi(next());
        else if (arg == "--solver-backend")   solver_backend_ = next();
        else if (arg == "--gmres-print")      gmres_print_ = std::stoi(next());
        else if (arg == "--spatial-procs")    spatial_procs_ = std::stoi(next());
        else if (arg == "--no-skin-mesh")     no_skin_mesh_ = true;
        else if (arg == "--profile") {
            auto vals = parse_csv_reals(next());
            if (vals.size() == 4) {
                profile_coords.push_back({vals[0], vals[1], vals[2], vals[3]});
            } else if (rank == 0) {
                std::cerr << "WARNING: --profile requires x0,y0,x1,y1\n";
            }
        }
        else if (arg == "--help" || arg == "-h") {
            if (rank == 0) print_usage();
            mfem::Mpi::Finalize();
            return 0;
        }
        else if (arg[0] != '-' && edi_dir.empty()) {
            // Positional argument: EDI directory
            edi_dir = arg;
        }
    }

    if (edi_dir.empty()) {
        if (rank == 0) {
            std::cerr << "ERROR: --edi-dir is required.\n";
            print_usage();
        }
        mfem::Mpi::Finalize();
        return 1;
    }

    auto t_total = Clock::now();

    // =================================================================
    // 1. Load EDI files
    // =================================================================
    data::MTData all_data;
    io::load_edi_directory(edi_dir, all_data);
    int ns = all_data.num_stations();
    int nf_all = all_data.num_frequencies();

    if (rank == 0) {
        double total_mem = utils::total_memory_gb();
        double avail_mem = utils::available_memory_gb();
        std::cout << "\n================================================\n"
                  << " 3D MT Inversion (Real Field Data)\n"
                  << "================================================\n"
                  << "  MPI processes:   " << nprocs << "\n"
                  << "  System memory:   " << std::fixed << std::setprecision(1)
                  << total_mem << " GB total, "
                  << avail_mem << " GB available\n"
                  << "  EDI directory:   " << edi_dir.string() << "\n"
                  << "  Stations:        " << ns << "\n"
                  << "  All frequencies: " << nf_all << "\n"
                  << "  Starting sigma:  " << sigma_bg << " S/m ("
                  << 1.0/sigma_bg << " Ohm.m)\n"
                  << "  FE order:        " << fe_order << "\n"
                  << "  Error floor:     " << error_floor_pct << " %\n"
                  << "  Solver:          ITERATIVE (FGMRES + AMS)\n"
                  << std::endl;
    }

    // =================================================================
    // 2. Select frequencies
    // =================================================================
    const auto& all_freqs = all_data.frequencies();
    Real fmin = fmin_user > 0 ? fmin_user
        : *std::min_element(all_freqs.begin(), all_freqs.end());
    Real fmax = fmax_user > 0 ? fmax_user
        : *std::max_element(all_freqs.begin(), all_freqs.end());

    std::vector<Real> sel_freqs;
    if (!user_freqs.empty()) {
        for (Real f : user_freqs)
            sel_freqs.push_back(snap_to_data_freq(f, all_freqs));
    } else if (nfreq > 0) {
        auto target = log_spaced_freqs(fmin, fmax, nfreq);
        for (Real f : target)
            sel_freqs.push_back(snap_to_data_freq(f, all_freqs));
    } else {
        // Use all frequencies within [fmin, fmax]
        for (Real f : all_freqs) {
            if (f >= fmin * 0.99 && f <= fmax * 1.01)
                sel_freqs.push_back(f);
        }
    }

    // Sort and remove duplicates
    std::sort(sel_freqs.begin(), sel_freqs.end());
    sel_freqs.erase(std::unique(sel_freqs.begin(), sel_freqs.end()),
                     sel_freqs.end());
    int nf = static_cast<int>(sel_freqs.size());

    if (rank == 0) {
        std::cout << "  Selected frequencies (" << nf << "):\n";
        for (int i = 0; i < nf; ++i) {
            Real f = sel_freqs[i];
            std::cout << "    [" << std::setw(2) << i << "] f="
                      << std::setw(12) << std::setprecision(6) << f
                      << " Hz  (T=" << std::setprecision(4)
                      << 1.0/f << " s)\n";
        }
        std::cout << std::endl;
    }

    // Build filtered MTData
    data::MTData observed, predicted;
    for (int s = 0; s < ns; ++s) {
        observed.add_station(all_data.station(s));
        predicted.add_station(all_data.station(s));
    }
    observed.set_frequencies(RealVec(sel_freqs.begin(), sel_freqs.end()));
    predicted.set_frequencies(RealVec(sel_freqs.begin(), sel_freqs.end()));

    // Copy observed impedance data
    for (int si = 0; si < ns; ++si) {
        for (int fi = 0; fi < nf; ++fi) {
            for (int ai = 0; ai < nf_all; ++ai) {
                if (std::abs(all_freqs[ai] - sel_freqs[fi]) < 1e-10) {
                    observed.set_observed(si, fi, all_data.observed(si, ai));
                    break;
                }
            }
        }
    }

    // Apply error floor
    observed.apply_error_floor(error_floor_pct / 100.0, 0.03);

    if (rank == 0) {
        std::cout << "  Active data points: " << observed.num_active_data()
                  << "\n" << std::endl;
    }

    // =================================================================
    // 3. Build mesh
    // =================================================================
    auto t_mesh = Clock::now();

    std::vector<mesh::Station3D> stations =
        mesh::stations_from_mt_data(observed);

    // Use skin-depth-aware mesh generation if frequencies are available
    // and --no-skin-mesh was NOT specified.
    auto mesh_params = (!no_skin_mesh_ && !sel_freqs.empty())
        ? mesh::auto_mesh_params(stations, RealVec(sel_freqs.begin(), sel_freqs.end()),
                                 1.0 / sigma_bg,   // rho = 1/σ_bg
                                 10.0)              // safety: f_design = f_min/10
        : mesh::auto_mesh_params(stations);

    if (h_surface > 0) {
        mesh_params.h_surface_x = h_surface;
        mesh_params.h_surface_y = h_surface;
    }
    if (h_surface_z > 0) {
        mesh_params.h_surface_z = h_surface_z;
    } else if (h_surface > 0) {
        mesh_params.h_surface_z = h_surface / 5.0;  // auto: h_surface/5
    }
    mesh_params.h_air_start = h_air_start;
    mesh_params.growth_air  = growth_air;
    mesh_params.refine_near_stations = refine;

    // Load DEM if provided
    std::unique_ptr<mesh::ALOSDem> dem_ptr;
    if (!dem_path.empty()) {
        dem_ptr = std::make_unique<mesh::ALOSDem>();
        std::string dem_ext = dem_path.extension().string();
        if (dem_ext == ".tif" || dem_ext == ".tiff") {
            // GeoTIFF: try ASCII fallback (.txt) first
            fs::path ascii_dem = fs::path(dem_path).replace_extension(".txt");
            if (fs::exists(ascii_dem)) {
                if (rank == 0)
                    std::cout << "  Using ASCII DEM: " << ascii_dem.string() << "\n";
                dem_ptr->load_ascii(ascii_dem);
            } else {
                try {
                    dem_ptr->load_geotiff(dem_path);
                } catch (const std::exception& e) {
                    if (rank == 0)
                        std::cerr << "  DEM load failed: " << e.what()
                                  << "\n  Proceeding without terrain.\n";
                    dem_ptr.reset();
                }
            }
        } else {
            dem_ptr->load_ascii(dem_path);
        }
        mesh_params.use_terrain = (dem_ptr != nullptr);
        if (dem_ptr && rank == 0) {
            std::cout << "  DEM loaded: " << dem_ptr->n_lon << "x"
                      << dem_ptr->n_lat << std::endl;
        }
    }

    mesh::HexMeshGenerator3D generator;
    // Disable non-conforming h-refinement: MFEM 4.9 hex NCMesh + ParMesh
    // causes GetSharedFaceTransformationsByLocalIndex assertion failure.
    // Instead, use finer base resolution (--h-surface) for station detail.
    if (mesh_params.refine_near_stations > 0 && rank == 0) {
        std::cout << "  NOTE: h-refinement disabled (MFEM 4.9 hex NCMesh "
                  << "limitation).\n"
                  << "  Use --h-surface to control resolution instead.\n";
    }
    mesh_params.refine_near_stations = 0;
    auto serial_mesh = generator.generate(mesh_params, stations,
                                           dem_ptr.get());
    int ne_serial = serial_mesh->GetNE();

    if (rank == 0) {
        double dt = std::chrono::duration<double>(Clock::now() - t_mesh).count();
        std::cout << "  Mesh: " << ne_serial << " elements  ["
                  << fmt_elapsed(dt) << "]"
                  << (mesh_params.skin_depth_mesh ? "  [skin-depth optimized]" : "")
                  << "\n";
    }

    // Create conforming ParMesh (no NCMesh — avoids MFEM 4.9 hex bug)
    mfem::ParMesh pmesh(MPI_COMM_WORLD, *serial_mesh);
    serial_mesh.reset();

    int ne_local = pmesh.GetNE();
    int ne_global = 0;
    MPI_Allreduce(&ne_local, &ne_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    {
        long long est_dof = 6LL * ne_global;
        // Iterative solver memory: ~80 bytes/DOF (Krylov vectors + preconditioner)
        // kdim=50: 2*50*2*N*8 bytes + AMS setup ≈ 80 bytes/DOF
        double bytes_per_dof = 80.0 * (gmres_kdim_ / 50.0);
        double est_mem_gb = est_dof * 2.0 * bytes_per_dof / 1e9;
        double sys_mem_gb = utils::total_memory_gb();

        if (rank == 0) {
            std::cout << "  ParMesh: " << ne_global << " elements on "
                      << nprocs << " procs\n"
                      << "  Est. DOF: ~" << est_dof / 1000000 << "M  "
                      << "est. memory: ~" << std::fixed << std::setprecision(1)
                      << est_mem_gb << " GB (iterative)\n" << std::endl;
        }

        if (est_mem_gb > sys_mem_gb * 0.80) {
            if (rank == 0) {
                std::cout << "  *** WARNING: Estimated memory (" << est_mem_gb
                          << " GB) > 80% of RAM (" << sys_mem_gb << " GB).\n"
                          << "  *** Consider: --h-surface " << std::max(1000.0, h_surface * 1.5)
                          << " or --gmres-kdim " << gmres_kdim_ / 2
                          << " to reduce memory ***\n\n";
            }
        }
    }

    // =================================================================
    // 3b. Attribute diagnostics (verify earth/air classification)
    // =================================================================
    {
        int n_earth_local = 0, n_air_local = 0, n_other_local = 0;
        for (int e = 0; e < pmesh.GetNE(); ++e) {
            int attr = pmesh.GetAttribute(e);
            if (attr == 1) ++n_earth_local;
            else if (attr == 2) ++n_air_local;
            else ++n_other_local;
        }
        int counts[3] = {n_earth_local, n_air_local, n_other_local};
        int global_counts[3] = {0, 0, 0};
        MPI_Allreduce(counts, global_counts, 3, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "  Attributes: earth=" << global_counts[0]
                      << " air=" << global_counts[1]
                      << " other=" << global_counts[2] << "\n";
            if (global_counts[0] == 0) {
                std::cerr << "  *** ERROR: No earth elements (attr=1) found! "
                          << "Check mesh generation. ***\n";
            }
        }
    }

    // =================================================================
    // 4. Initial model (uniform halfspace)
    // =================================================================
    model::ConductivityModel model;
    model.init_3d(pmesh.GetNE(), sigma_bg);

    // =================================================================
    // 5. Forward solver
    // =================================================================
    forward::ForwardParams3D fwd_params;
    fwd_params.fe_order = fe_order;

    // Parse solver backend string → enum
    {
        auto upper = [](std::string s) {
            for (auto& c : s) c = std::toupper(c);
            return s;
        };
        std::string be = upper(solver_backend_);
        if (be == "COMPLEX" || be == "BICGSTAB" || be == "ILU") {
            fwd_params.backend = forward::SolverBackend::COMPLEX_BICGSTAB;
            if (rank == 0)
                std::cout << "  Solver backend:  COMPLEX (BiCGStab + ILU(0))\n";
        } else if (be == "ITERATIVE" || be == "FGMRES" || be == "AMS") {
            fwd_params.backend = forward::SolverBackend::ITERATIVE;
            if (rank == 0)
                std::cout << "  Solver backend:  ITERATIVE (FGMRES + AMS)\n";
        } else {
            fwd_params.backend = forward::SolverBackend::COMPLEX_BICGSTAB;
            if (rank == 0)
                std::cout << "  Solver backend:  COMPLEX (default)\n";
        }
    }
    // Parse preconditioner type
    {
        auto upper = [](std::string s) {
            for (auto& c : s) c = std::toupper(c);
            return s;
        };
        std::string ps = upper(precond_str);
        if      (ps == "AMG")   fwd_params.precond = forward::PrecondType::AMG;
        else if (ps == "AMS")   fwd_params.precond = forward::PrecondType::AMS;
        else if (ps == "DIAG")  fwd_params.precond = forward::PrecondType::DIAG;
        else {
            if (rank == 0)
                std::cerr << "WARNING: unknown precond '" << precond_str
                          << "', using AMS\n";
            fwd_params.precond = forward::PrecondType::AMS;
        }
        if (rank == 0) {
            std::cout << "  Preconditioner:  " << precond_str << "\n";
            if (ps == "AMS") {
                std::cout << "  AMS V-cycles:    max=" << ams_max_vcycles_
                          << " (ω_mid=" << ams_omega_mid_
                          << " ω_high=" << ams_omega_high_ << ")\n"
                          << "  AMS smoother:    type=" << ams_smooth_type_
                          << " sweeps=" << ams_smooth_sweeps_ << "\n";
            }
        }
    }
    // Iterative solver settings (no MUMPS)
    fwd_params.gmres_kdim       = gmres_kdim_;
    fwd_params.gmres_tol        = gmres_tol_;
    fwd_params.gmres_maxiter    = gmres_maxiter_;
    fwd_params.gmres_print      = gmres_print_;
    fwd_params.adjoint_direct   = false;  // always iterative
    fwd_params.adjoint_tol      = adjoint_tol_;
    fwd_params.adjoint_maxiter  = adjoint_maxiter_;
    fwd_params.adjoint_kdim     = adjoint_kdim_;
    fwd_params.adjoint_div_corr    = adjoint_div_corr_;
    fwd_params.adjoint_use_jacobi  = adjoint_use_jacobi_;
    fwd_params.adjoint_inner_iter  = adjoint_inner_iter_;
    fwd_params.adjoint_outer_iter  = adjoint_outer_iter_;
    fwd_params.ccgd_enabled        = ccgd_enabled_;
    fwd_params.ccgd_tau            = ccgd_tau_;
    fwd_params.ams_max_vcycles  = ams_max_vcycles_;
    fwd_params.ams_omega_mid    = ams_omega_mid_;
    fwd_params.ams_omega_high   = ams_omega_high_;
    fwd_params.ams_smooth_type  = ams_smooth_type_;
    fwd_params.ams_smooth_sweeps = ams_smooth_sweeps_;

    forward::ForwardSolver3D fwd;
    fwd.setup(pmesh, model, fwd_params);

    // =================================================================
    // 6. Regularisation
    // =================================================================
    regularization::RegParams reg_params;
    reg_params.alpha_s = alpha_s;
    reg_params.alpha_x = alpha_x;
    reg_params.alpha_z = alpha_z;
    reg_params.alpha_r = 0.0;

    regularization::Regularization reg;
    reg.setup_3d(pmesh, reg_params);

    if (rank == 0) {
        std::cout << "  Regularisation: " << reg.n_active()
                  << " active parameters\n"
                  << "    alpha_s=" << alpha_s
                  << "  alpha_x=" << alpha_x
                  << "  alpha_z=" << alpha_z << "\n" << std::endl;
    }

    // =================================================================
    // 7. VTK export setup
    // =================================================================
    fs::path out_path(output_dir);
    fs::create_directories(out_path);
    fs::create_directories(out_path / "iterations");
    fs::create_directories(out_path / "slices");

    io::ExportParams export_params;
    export_params.export_vtk = true;
    export_params.export_vtu_parallel = false;
    export_params.export_slices = true;
    export_params.export_station_csv = true;
    export_params.export_station_geojson = true;
    export_params.auto_slice_interval = slice_interval;

    if (!slice_depths.empty()) {
        export_params.slice_depths = slice_depths;
        export_params.auto_slice_interval = 0;
    }

    for (size_t p = 0; p < profile_coords.size(); ++p) {
        const auto& c = profile_coords[p];
        export_params.profile_slices.push_back(
            {"profile_" + std::to_string(p), c[0], c[1], c[2], c[3], 200});
    }

    // Export initial model
    if (rank == 0) {
        io::export_model_vtk(pmesh, model, out_path / "model_initial.vtk", 0);
        io::export_stations_csv(observed, out_path / "stations.csv");
        io::export_stations_geojson(observed, out_path / "stations.geojson");
        std::cout << "  Exported initial model + stations\n" << std::endl;
    }

    // =================================================================
    // 8. Save run configuration
    // =================================================================
    if (rank == 0) {
        std::ofstream cfg(out_path / "run_config.txt");
        cfg << "# Maple3DMT Inversion Configuration\n"
            << "# " << __DATE__ << " " << __TIME__ << "\n\n"
            << "edi_dir        = " << edi_dir.string() << "\n"
            << "dem            = " << dem_path.string() << "\n"
            << "nprocs         = " << nprocs << "\n"
            << "stations       = " << ns << "\n"
            << "frequencies    = " << nf << "\n"
            << "elements       = " << ne_global << "\n"
            << "sigma_bg       = " << sigma_bg << "\n"
            << "fe_order       = " << fe_order << "\n"
            << "solver         = " << solver_type << "\n"
            << "backend        = " << solver_backend_ << "\n"
            << "error_floor    = " << error_floor_pct << " %\n"
            << "max_iter       = " << max_iter << "\n"
            << "target_rms     = " << target_rms << "\n"
            << "lambda_init    = " << lambda_init << "\n"
            << "lambda_dec     = " << lambda_dec << "\n"
            << "adjoint_tol    = " << adjoint_tol_ << "\n"
            << "adjoint_maxiter= " << adjoint_maxiter_ << "\n"
            << "adjoint_div_corr = " << (adjoint_div_corr_ ? "ON" : "OFF") << "\n"
            << "alpha_s        = " << alpha_s << "\n"
            << "alpha_x        = " << alpha_x << "\n"
            << "alpha_z        = " << alpha_z << "\n"
            << "h_surface      = " << h_surface << "\n"
            << "h_air_start    = " << h_air_start << " m\n"
            << "growth_air     = " << growth_air << "\n"
            << "refine         = " << refine << "\n"
            << "system_ram_gb  = " << std::fixed << std::setprecision(1)
            << utils::total_memory_gb() << "\n";
        cfg.close();
    }

    // =================================================================
    // 9. Run inversion
    // =================================================================
    inversion::InversionParams3D inv_params;
    if (solver_type == "gn-cg" || solver_type == "GN-CG") {
        inv_params.solver = inversion::InversionSolver::GN_CG;
    } else {
        inv_params.solver = inversion::InversionSolver::NLCG;
    }
    inv_params.max_iterations    = max_iter;
    inv_params.target_rms        = target_rms;
    inv_params.lambda_init       = lambda_init;
    inv_params.lambda_decrease   = lambda_dec;
    inv_params.cg_max_iter       = cg_maxiter;
    inv_params.cg_kdim           = cg_kdim;
    inv_params.cg_tolerance      = cg_tol;
    inv_params.cg_adaptive_tol   = true;
    inv_params.linesearch_max    = 8;
    inv_params.linesearch_beta   = 0.5;
    inv_params.save_checkpoints  = true;
    inv_params.checkpoint_every  = 1;
    inv_params.checkpoint_dir    = out_path / "checkpoints";
    inv_params.release_factor_between_freqs = true;
    inv_params.freq_parallel_spatial_procs = spatial_procs_;

    inversion::Inversion3D inversion;
    inversion.setup(pmesh, model, observed, fwd, reg, inv_params);

    // Resume from checkpoint if requested
    if (!resume_dir.empty()) {
        if (rank == 0) {
            std::cout << "  Resuming from: " << resume_dir.string() << "\n\n";
        }
        inversion.resume(resume_dir);
    }

    // Frequency progress callback: clean \r-based progress bar.
    // FGMRES output is auto-suppressed (gmres_print=-1) when this callback is set,
    // so \r overwriting works cleanly.
    inversion.set_freq_progress_callback(
        [&](int fi, int nf, Real fhz, const std::string& phase) {
            if (rank != 0) return;
            int pct = ((fi + 1) * 100) / nf;
            int bar_width = 30;
            int filled = ((fi + 1) * bar_width) / nf;
            std::string bar(filled, '=');
            if (filled < bar_width) {
                bar += std::string(bar_width - filled, ' ');
            }
            std::cout << "\r  " << phase << ": [" << bar << "] "
                      << std::setw(3) << pct << "% "
                      << "(" << (fi + 1) << "/" << nf << ") "
                      << std::fixed << std::setprecision(4) << fhz << " Hz"
                      << "    " << std::flush;
            if (fi == nf - 1) std::cout << std::endl;
        });

    // Per-iteration callback: export model + print progress
    inversion.set_iteration_callback(
        [&](int iter, const inversion::IterationLog3D& log) {
            if (rank == 0) {
                // Export model at this iteration
                io::export_model_vtk(pmesh, model,
                    out_path / "iterations" /
                    ("model_iter_" + std::to_string(iter) + ".vtk"),
                    iter);

                // Append to convergence log
                std::ofstream conv(out_path / "convergence.csv",
                    iter == 1 ? std::ios::out : std::ios::app);
                if (iter == 1) {
                    conv << "iter,rms,objective,data_misfit,model_norm,"
                         << "lambda,step_length,cg_iters,peak_mem_gb\n";
                }
                conv << iter << ","
                     << std::setprecision(8) << log.rms << ","
                     << log.objective << ","
                     << log.data_misfit << ","
                     << log.model_norm << ","
                     << log.lambda << ","
                     << log.step_length << ","
                     << log.cg_iterations << ","
                     << log.peak_memory_gb << "\n";
                conv.close();
            }
        });

    if (rank == 0) {
        std::cout << "================================================\n"
                  << " Starting " << solver_type << " Inversion\n"
                  << "  max_iter=" << max_iter
                  << "  target_rms=" << target_rms
                  << "  lambda=" << lambda_init << "\n"
                  << "  Adjoint: iterative"
                  << " tol=" << adjoint_tol_
                  << " maxiter=" << adjoint_maxiter_
                  << " kdim=" << adjoint_kdim_
                  << (adjoint_div_corr_ ? " +DivCorr" : "") << "\n"
                  << "================================================\n"
                  << std::endl;
    }

    auto t_inv = Clock::now();
    inversion.run();
    double dt_inv = std::chrono::duration<double>(Clock::now() - t_inv).count();

    // =================================================================
    // 10. Export final results
    // =================================================================
    if (rank == 0) {
        std::cout << "\n--- Exporting final results ---\n";
    }

    // Final model VTK + slices
    io::export_all(pmesh, model, observed, export_params, out_path, -1);

    // Write final predicted responses
    if (rank == 0) {
        std::ofstream csv(out_path / "final_responses.csv");
        csv << "station,freq_Hz,period_s,"
            << "obs_rhoXY,obs_phiXY,obs_rhoYX,obs_phiYX,"
            << "pred_rhoXY,pred_phiXY,pred_rhoYX,pred_phiYX,"
            << "obs_ZxyRe,obs_ZxyIm,obs_ZyxRe,obs_ZyxIm,"
            << "pred_ZxyRe,pred_ZxyIm,pred_ZyxRe,pred_ZyxIm\n";

        for (int si = 0; si < ns; ++si) {
            for (int fi = 0; fi < nf; ++fi) {
                Real f = sel_freqs[fi];
                Real omega = constants::TWOPI * f;

                const auto& obs  = observed.observed(si, fi);
                const auto& pred = observed.predicted(si, fi);

                auto rho_phi = [&](Complex Z) -> std::pair<Real,Real> {
                    Real rho = std::norm(Z) / (omega * constants::MU0);
                    Real phi = std::atan2(Z.imag(), Z.real()) * 180.0 / constants::PI;
                    return {rho, phi};
                };

                auto [orXY, opXY] = rho_phi(obs.Zxy.value);
                auto [orYX, opYX] = rho_phi(obs.Zyx.value);
                auto [prXY, ppXY] = rho_phi(pred.Zxy.value);
                auto [prYX, ppYX] = rho_phi(pred.Zyx.value);

                csv << observed.station(si).name << ","
                    << std::setprecision(8) << f << "," << 1.0/f << ","
                    << std::setprecision(6)
                    << orXY << "," << opXY << ","
                    << orYX << "," << opYX << ","
                    << prXY << "," << ppXY << ","
                    << prYX << "," << ppYX << ","
                    << obs.Zxy.value.real() << "," << obs.Zxy.value.imag() << ","
                    << obs.Zyx.value.real() << "," << obs.Zyx.value.imag() << ","
                    << pred.Zxy.value.real() << "," << pred.Zxy.value.imag() << ","
                    << pred.Zyx.value.real() << "," << pred.Zyx.value.imag() << "\n";
            }
        }
        csv.close();
    }

    // =================================================================
    // 11. Print summary
    // =================================================================
    if (rank == 0) {
        double dt_total = std::chrono::duration<double>(
            Clock::now() - t_total).count();

        const auto& history = inversion.history();

        std::cout << "\n================================================\n"
                  << " Inversion Summary\n"
                  << "================================================\n"
                  << "  Stations:        " << ns << "\n"
                  << "  Frequencies:     " << nf << "\n"
                  << "  Elements:        " << ne_global << "\n"
                  << "  GN iterations:   " << history.size() << "\n";

        if (!history.empty()) {
            std::cout << "  Initial RMS:     "
                      << std::setprecision(4) << history.front().rms << "\n"
                      << "  Final RMS:       " << history.back().rms << "\n"
                      << "  Final lambda:    " << history.back().lambda << "\n";
        }

        std::cout << "  Inversion time:  " << fmt_elapsed(dt_inv) << "\n"
                  << "  Total time:      " << fmt_elapsed(dt_total) << "\n"
                  << "  Output:          " << out_path.string() << "/\n"
                  << std::endl;

        // Convergence table
        std::cout << "  Convergence:\n"
                  << "  " << std::setw(5) << "Iter"
                  << std::setw(10) << "RMS"
                  << std::setw(14) << "Objective"
                  << std::setw(12) << "Lambda"
                  << std::setw(8) << "CG"
                  << std::setw(10) << "Step" << "\n"
                  << "  " << std::string(59, '-') << "\n";

        for (const auto& h : history) {
            std::cout << "  " << std::setw(5) << h.iteration
                      << std::setw(10) << std::setprecision(4) << h.rms
                      << std::setw(14) << std::setprecision(6) << h.objective
                      << std::setw(12) << std::setprecision(4) << h.lambda
                      << std::setw(8) << h.cg_iterations
                      << std::setw(10) << std::setprecision(4) << h.step_length
                      << "\n";
        }

        std::cout << "\n=== Inversion Complete ===\n" << std::endl;
    }

    mfem::Mpi::Finalize();
    return 0;
}
