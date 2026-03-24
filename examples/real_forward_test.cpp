// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file real_forward_test.cpp
/// @brief Forward modelling with real EDI data and terrain-conforming mesh.
///
/// Usage:
///   mpirun -np 12 real_forward_test --edi-dir <path> [options]
///
/// Loads EDI files, builds 3D terrain mesh, runs forward solver for
/// selected frequencies, and exports predicted responses + VTK model.

#include "maple3dmt/forward/forward_solver_3d.h"
#include "maple3dmt/mesh/hex_mesh_3d.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/io/edi_io.h"
#include "maple3dmt/io/vtk_export_3d.h"
#include "maple3dmt/utils/logger.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>
#include <mfem.hpp>

using namespace maple3dmt;
using Clock = std::chrono::steady_clock;

namespace {

void print_usage() {
    std::cout <<
R"(Usage:
  mpirun -np <N> real_forward_test --edi-dir <path> [options]

Options:
  --edi-dir <path>    EDI file directory (REQUIRED)
  --dem <path>        DEM file (ASCII: lon lat elev)
  --sigma <val>       Background conductivity (S/m, default: 0.01)
  --freq <f1,f2,...>  Comma-separated frequencies to test (Hz)
                      (default: pick 3 log-spaced from data range)
  --nfreq <n>         Number of log-spaced test frequencies (default: 3)
  --order <val>       FE element order (default: 1)
  --output <dir>      Output directory (default: real_fwd_output)
  --h-surface <m>     Surface element size (0 = auto, default)
  --refine <n>        Local refinement near stations (default: 1)
  --blr-tol <val>     MUMPS BLR tolerance (default: 1e-6, lower=more accurate)
  --error-floor <pct> Impedance error floor (%, default: 5)
  --help              Show this message
)";
}

/// Parse comma-separated frequency list: "0.01,0.1,1.0,10.0"
std::vector<Real> parse_freq_list(const std::string& s) {
    std::vector<Real> freqs;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (!token.empty()) freqs.push_back(std::stod(token));
    }
    return freqs;
}

/// Pick n log-spaced frequencies from [fmin, fmax].
std::vector<Real> log_spaced_freqs(Real fmin, Real fmax, int n) {
    std::vector<Real> freqs;
    if (n <= 0) return freqs;
    if (n == 1) {
        freqs.push_back(std::sqrt(fmin * fmax));  // geometric mean
        return freqs;
    }
    Real log_min = std::log10(fmin);
    Real log_max = std::log10(fmax);
    for (int i = 0; i < n; ++i) {
        Real log_f = log_min + i * (log_max - log_min) / (n - 1);
        freqs.push_back(std::pow(10.0, log_f));
    }
    return freqs;
}

/// Find closest available frequency in data.
Real snap_to_data_freq(Real target, const RealVec& available) {
    Real best = available[0];
    Real best_diff = std::abs(std::log10(target) - std::log10(best));
    for (size_t i = 1; i < available.size(); ++i) {
        Real diff = std::abs(std::log10(target) - std::log10(available[i]));
        if (diff < best_diff) {
            best = available[i];
            best_diff = diff;
        }
    }
    return best;
}

/// Format elapsed time.
std::string fmt_elapsed(double seconds) {
    if (seconds < 60) return std::to_string(static_cast<int>(seconds)) + " s";
    int m = static_cast<int>(seconds) / 60;
    int s = static_cast<int>(seconds) % 60;
    return std::to_string(m) + " min " + std::to_string(s) + " s";
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    mfem::Mpi::Init(argc, argv);
    int rank = mfem::Mpi::WorldRank();
    int nprocs = mfem::Mpi::WorldSize();

    // ===== Parse arguments =====
    fs::path edi_dir;
    fs::path dem_path;
    Real sigma_bg = 0.01;
    std::vector<Real> user_freqs;
    int nfreq = 3;
    int fe_order = 1;
    std::string output_dir = "real_fwd_output";
    Real h_surface = 0;  // auto
    Real h_surface_z = 0;  // 0 = h_surface/5
    int refine = 1;
    Real error_floor_pct = 5.0;
    Real blr_tol = 1e-6;  // relaxed for memory savings (vs 1e-10 for accuracy)

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--edi-dir" && i + 1 < argc) {
            edi_dir = argv[++i];
        } else if (arg == "--dem" && i + 1 < argc) {
            dem_path = argv[++i];
        } else if (arg == "--sigma" && i + 1 < argc) {
            sigma_bg = std::stod(argv[++i]);
        } else if (arg == "--freq" && i + 1 < argc) {
            user_freqs = parse_freq_list(argv[++i]);
        } else if (arg == "--nfreq" && i + 1 < argc) {
            nfreq = std::stoi(argv[++i]);
        } else if (arg == "--order" && i + 1 < argc) {
            fe_order = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--h-surface" && i + 1 < argc) {
            h_surface = std::stod(argv[++i]);
        } else if (arg == "--h-surface-z" && i + 1 < argc) {
            h_surface_z = std::stod(argv[++i]);
        } else if (arg == "--refine" && i + 1 < argc) {
            refine = std::stoi(argv[++i]);
        } else if (arg == "--blr-tol" && i + 1 < argc) {
            blr_tol = std::stod(argv[++i]);
        } else if (arg == "--error-floor" && i + 1 < argc) {
            error_floor_pct = std::stod(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            if (rank == 0) print_usage();
            mfem::Mpi::Finalize();
            return 0;
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

    // ===== 1. Load EDI files =====
    data::MTData all_data;
    io::load_edi_directory(edi_dir, all_data);
    int ns = all_data.num_stations();
    int nf_all = all_data.num_frequencies();

    if (rank == 0) {
        std::cout << "\n========================================\n"
                  << " 3D MT Forward Modelling (Real Data)\n"
                  << "========================================\n"
                  << "  MPI processes: " << nprocs << "\n"
                  << "  EDI directory: " << edi_dir.string() << "\n"
                  << "  Stations:      " << ns << "\n"
                  << "  Frequencies:   " << nf_all << " total\n"
                  << "  sigma_bg:      " << sigma_bg << " S/m ("
                  << 1.0/sigma_bg << " Ohm.m)\n"
                  << "  FE order:      " << fe_order << "\n"
                  << "  BLR tolerance: " << blr_tol << "\n"
                  << "  Error floor:   " << error_floor_pct << " %\n"
                  << std::endl;
    }

    // ===== 2. Select test frequencies =====
    const auto& all_freqs = all_data.frequencies();
    Real fmin = *std::min_element(all_freqs.begin(), all_freqs.end());
    Real fmax = *std::max_element(all_freqs.begin(), all_freqs.end());

    std::vector<Real> test_freqs;
    if (!user_freqs.empty()) {
        // Snap user-specified frequencies to nearest available
        for (Real f : user_freqs) {
            test_freqs.push_back(snap_to_data_freq(f, all_freqs));
        }
    } else {
        // Pick nfreq log-spaced from data range
        auto target = log_spaced_freqs(fmin, fmax, nfreq);
        for (Real f : target) {
            test_freqs.push_back(snap_to_data_freq(f, all_freqs));
        }
    }

    // Remove duplicates
    std::sort(test_freqs.begin(), test_freqs.end());
    test_freqs.erase(std::unique(test_freqs.begin(), test_freqs.end()),
                      test_freqs.end());

    if (rank == 0) {
        std::cout << "  Test frequencies (" << test_freqs.size() << "):\n";
        for (Real f : test_freqs) {
            std::cout << "    f = " << std::setprecision(6) << f
                      << " Hz  (T = " << std::setprecision(4)
                      << 1.0/f << " s)\n";
        }
        std::cout << std::endl;
    }

    // Build filtered MTData with selected frequencies
    data::MTData observed, predicted;
    for (int s = 0; s < ns; ++s) {
        observed.add_station(all_data.station(s));
        predicted.add_station(all_data.station(s));
    }
    observed.set_frequencies(RealVec(test_freqs.begin(), test_freqs.end()));
    predicted.set_frequencies(RealVec(test_freqs.begin(), test_freqs.end()));

    // Copy observed data for selected frequencies
    for (int si = 0; si < ns; ++si) {
        for (int fi = 0; fi < static_cast<int>(test_freqs.size()); ++fi) {
            // Find this freq in all_data
            for (int ai = 0; ai < nf_all; ++ai) {
                if (std::abs(all_freqs[ai] - test_freqs[fi]) < 1e-10) {
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
                  << std::endl;
    }

    // ===== 3. Build mesh =====
    auto t_mesh = Clock::now();

    std::vector<mesh::Station3D> stations =
        mesh::stations_from_mt_data(observed);

    auto mesh_params = mesh::auto_mesh_params(stations);
    if (h_surface > 0) {
        mesh_params.h_surface_x = h_surface;
        mesh_params.h_surface_y = h_surface;
        if (h_surface_z > 0) {
            mesh_params.h_surface_z = h_surface_z;
        } else {
            mesh_params.h_surface_z = h_surface / 5.0;
        }
    }
    mesh_params.refine_near_stations = refine;

    // Load DEM
    std::unique_ptr<mesh::ALOSDem> dem_ptr;
    if (!dem_path.empty()) {
        dem_ptr = std::make_unique<mesh::ALOSDem>();
        dem_ptr->load_ascii(dem_path);
        mesh_params.use_terrain = true;
        if (rank == 0) {
            std::cout << "  DEM loaded: " << dem_ptr->n_lon << "x"
                      << dem_ptr->n_lat << std::endl;
        }
    }

    mesh::HexMeshGenerator3D generator;
    // Disable non-conforming h-refinement (MFEM 4.9 hex NCMesh limitation)
    mesh_params.refine_near_stations = 0;
    auto serial_mesh = generator.generate(mesh_params, stations,
                                           dem_ptr.get());
    int ne_serial = serial_mesh->GetNE();

    if (rank == 0) {
        double dt = std::chrono::duration<double>(Clock::now() - t_mesh).count();
        std::cout << "\n  Mesh: " << ne_serial << " elements  ["
                  << fmt_elapsed(dt) << "]\n";
    }

    // Create conforming ParMesh (no NCMesh)
    mfem::ParMesh pmesh(MPI_COMM_WORLD, *serial_mesh);
    serial_mesh.reset();

    // h-refinement disabled; kept for future MFEM versions
    if (false && !stations.empty()) {
        mesh::HexMeshGenerator3D::refine_near_stations_parallel(
            pmesh, stations, 1,
            mesh_params.h_surface_x, mesh_params.h_surface_y);
    }
    int ne_local = pmesh.GetNE();
    int ne_global = 0;
    MPI_Allreduce(&ne_local, &ne_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        // Rough memory estimate: Nedelec DOFs ~ 3*ne, complex block → 6*ne real DOFs
        // MUMPS BLR: ~100-300 bytes/DOF depending on BLR tol and sparsity
        long long est_dof = 6LL * ne_global;
        double est_mem_low_gb = est_dof * 100.0 / 1e9;   // BLR tol=1e-6
        double est_mem_high_gb = est_dof * 300.0 / 1e9;   // BLR tol=1e-10
        std::cout << "  ParMesh: " << ne_global << " elements on "
                  << nprocs << " procs (" << ne_global / nprocs
                  << " per proc avg)\n"
                  << "  Est. DOF: ~" << est_dof / 1000000 << "M  "
                  << "MUMPS peak memory: ~"
                  << std::setprecision(1) << std::fixed
                  << est_mem_low_gb << "-" << est_mem_high_gb << " GB\n"
                  << std::endl;
        if (est_mem_high_gb > 60.0) {
            std::cout << "  *** WARNING: May exceed 64 GB RAM! ***\n"
                      << "  *** Use --h-surface to coarsen mesh or reduce --refine ***\n"
                      << std::endl;
        }
    }

    // ===== 4. Uniform halfspace model =====
    model::ConductivityModel model;
    model.init_3d(pmesh.GetNE(), sigma_bg);

    // ===== 5. Export initial model =====
    fs::path out_path(output_dir);
    fs::create_directories(out_path);

    if (rank == 0) {
        io::export_model_vtk(pmesh, model, out_path / "model_halfspace.vtk");
        io::export_stations_csv(observed, out_path / "stations.csv");
        io::export_stations_geojson(observed, out_path / "stations.geojson");
        std::cout << "  Exported: model_halfspace.vtk, stations.csv\n"
                  << std::endl;
    }

    // ===== 6. Forward solver =====
    forward::ForwardParams3D fwd_params;
    fwd_params.fe_order = fe_order;
    fwd_params.backend = forward::SolverBackend::ITERATIVE;
    fwd_params.blr_tolerance = blr_tol;

    forward::ForwardSolver3D solver;
    solver.setup(pmesh, model, fwd_params);

    if (rank == 0) {
        std::cout << "--- Forward Solve ---\n";
    }

    // Solve frequency by frequency (memory-friendly)
    for (int fi = 0; fi < static_cast<int>(test_freqs.size()); ++fi) {
        auto t_freq = Clock::now();

        solver.compute_single_frequency(fi, observed, predicted);
        solver.release_factorization();

        double dt = std::chrono::duration<double>(Clock::now() - t_freq).count();

        if (rank == 0) {
            Real f = test_freqs[fi];
            std::cout << "  f=" << std::setw(12) << std::setprecision(6) << f
                      << " Hz (T=" << std::setw(10) << std::setprecision(4)
                      << 1.0/f << " s)  " << fmt_elapsed(dt) << "\n";
        }
    }

    // ===== 7. Report results =====
    if (rank == 0) {
        std::cout << "\n--- Forward Results ---\n";

        // Write response CSV
        std::ofstream csv(out_path / "forward_responses.csv");
        csv << "station,freq_Hz,period_s,"
            << "rhoXY,phaseXY,rhoYX,phaseYX,"
            << "ZxyRe,ZxyIm,ZyxRe,ZyxIm,"
            << "ZxxRe,ZxxIm,ZyyRe,ZyyIm\n";

        for (int si = 0; si < ns; ++si) {
            for (int fi = 0; fi < static_cast<int>(test_freqs.size()); ++fi) {
                const auto& resp = predicted.predicted(si, fi);
                Complex Zxy = resp.Zxy.value;
                Complex Zyx = resp.Zyx.value;
                Complex Zxx = resp.Zxx.value;
                Complex Zyy = resp.Zyy.value;
                Real f = test_freqs[fi];
                Real omega = constants::TWOPI * f;

                // Apparent resistivity: rho_a = |Z|^2 / (omega * mu0)
                Real rho_xy = std::norm(Zxy) / (omega * constants::MU0);
                Real rho_yx = std::norm(Zyx) / (omega * constants::MU0);
                Real phi_xy = std::atan2(Zxy.imag(), Zxy.real()) * 180.0 / constants::PI;
                Real phi_yx = std::atan2(Zyx.imag(), Zyx.real()) * 180.0 / constants::PI;

                csv << all_data.station(si).name << ","
                    << std::setprecision(8) << f << ","
                    << 1.0/f << ","
                    << std::setprecision(6)
                    << rho_xy << "," << phi_xy << ","
                    << rho_yx << "," << phi_yx << ","
                    << Zxy.real() << "," << Zxy.imag() << ","
                    << Zyx.real() << "," << Zyx.imag() << ","
                    << Zxx.real() << "," << Zxx.imag() << ","
                    << Zyy.real() << "," << Zyy.imag() << "\n";
            }
        }
        csv.close();

        // Summary: average app.rho per frequency (should ≈ 1/sigma_bg for halfspace)
        Real expected_rho = 1.0 / sigma_bg;
        std::cout << "\n  Expected rho_a = " << expected_rho
                  << " Ohm.m (uniform halfspace)\n\n";
        std::cout << std::setw(14) << "Freq (Hz)"
                  << std::setw(12) << "Period (s)"
                  << std::setw(14) << "rho_XY avg"
                  << std::setw(14) << "rho_YX avg"
                  << std::setw(12) << "phi_XY"
                  << std::setw(12) << "phi_YX" << "\n";
        std::cout << std::string(88, '-') << "\n";

        for (int fi = 0; fi < static_cast<int>(test_freqs.size()); ++fi) {
            Real f = test_freqs[fi];
            Real omega = constants::TWOPI * f;
            Real sum_rho_xy = 0, sum_rho_yx = 0;
            Real sum_phi_xy = 0, sum_phi_yx = 0;

            for (int si = 0; si < ns; ++si) {
                const auto& resp = predicted.predicted(si, fi);
                sum_rho_xy += std::norm(resp.Zxy.value) / (omega * constants::MU0);
                sum_rho_yx += std::norm(resp.Zyx.value) / (omega * constants::MU0);
                sum_phi_xy += std::atan2(resp.Zxy.value.imag(),
                                          resp.Zxy.value.real()) * 180.0 / constants::PI;
                sum_phi_yx += std::atan2(resp.Zyx.value.imag(),
                                          resp.Zyx.value.real()) * 180.0 / constants::PI;
            }
            std::cout << std::setw(14) << std::setprecision(6) << f
                      << std::setw(12) << std::setprecision(4) << 1.0/f
                      << std::setw(14) << std::setprecision(2) << sum_rho_xy / ns
                      << std::setw(14) << sum_rho_yx / ns
                      << std::setw(12) << sum_phi_xy / ns
                      << std::setw(12) << sum_phi_yx / ns << "\n";
        }

        double dt_total = std::chrono::duration<double>(
            Clock::now() - t_total).count();
        std::cout << "\n  Total time: " << fmt_elapsed(dt_total) << "\n";
        std::cout << "  Output: " << out_path.string() << "/\n";
        std::cout << "\n=== Forward Test Complete ===\n" << std::endl;
    }

    solver.release_factorization();
    solver.mem_profiler().report(MPI_COMM_WORLD);
    mfem::Mpi::Finalize();
    return 0;
}
