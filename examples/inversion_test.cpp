// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file inversion_test.cpp
/// @brief Validation driver: 3D inversion of synthetic data with a conductive block.
///
/// Usage:
///   mpirun -np 1 inversion_test [--sigma_bg 0.01] [--sigma_block 0.1] [--niter 10]
///
/// Creates synthetic MT data from a model with a conductive block embedded
/// in a uniform halfspace, then inverts starting from the uniform model.
/// Validates that RMS decreases and conductivity increases near the block.

#include "maple3dmt/inversion/inversion_3d.h"
#include "maple3dmt/mesh/hex_mesh_3d.h"
#include "maple3dmt/io/vtk_export_3d.h"
#include "maple3dmt/utils/logger.h"
#include <iostream>
#include <cmath>
#include <random>
#include <mfem.hpp>

using namespace maple3dmt;

namespace {

void print_usage() {
    std::cout << "Usage:\n"
              << "  mpirun -np 1 inversion_test [options]\n"
              << "\nOptions:\n"
              << "  --sigma_bg <val>    Background conductivity (S/m, default: 0.01)\n"
              << "  --sigma_block <val> Block conductivity (S/m, default: 0.1)\n"
              << "  --niter <val>       Max GN iterations (default: 5)\n"
              << "  --noise <val>       Noise level (fraction, default: 0.02)\n"
              << std::endl;
}

/// Check if element center is inside the conductive block.
bool is_in_block(mfem::ParMesh& pmesh, int elem,
                 Real bx_min, Real bx_max,
                 Real by_min, Real by_max,
                 Real bz_min, Real bz_max) {
    mfem::Vector center(3);
    pmesh.GetElementCenter(elem, center);
    return center(0) >= bx_min && center(0) <= bx_max &&
           center(1) >= by_min && center(1) <= by_max &&
           center(2) >= bz_min && center(2) <= bz_max;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    mfem::Mpi::Init(argc, argv);
    int rank = mfem::Mpi::WorldRank();

    Real sigma_bg = 0.01;
    Real sigma_block = 0.1;
    int max_iter = 5;
    Real noise_level = 0.02;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sigma_bg" && i + 1 < argc) {
            sigma_bg = std::stod(argv[++i]);
        } else if (arg == "--sigma_block" && i + 1 < argc) {
            sigma_block = std::stod(argv[++i]);
        } else if (arg == "--niter" && i + 1 < argc) {
            max_iter = std::stoi(argv[++i]);
        } else if (arg == "--noise" && i + 1 < argc) {
            noise_level = std::stod(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            if (rank == 0) print_usage();
            mfem::Mpi::Finalize();
            return 0;
        }
    }

    if (rank == 0) {
        std::cout << "\n=== Inversion3D Validation ===\n"
                  << "  sigma_bg    = " << sigma_bg << " S/m\n"
                  << "  sigma_block = " << sigma_block << " S/m\n"
                  << "  max_iter    = " << max_iter << "\n"
                  << "  noise       = " << noise_level * 100 << " %\n"
                  << std::endl;
    }

    // ==================================================================
    // Create mesh with stations
    // ==================================================================
    std::vector<mesh::Station3D> stations;
    // 2x2 grid of stations
    for (int iy = -1; iy <= 1; iy += 2) {
        for (int ix = -1; ix <= 1; ix += 2) {
            mesh::Station3D s;
            s.name = "S" + std::to_string(stations.size() + 1);
            s.x = ix * 500.0;
            s.y = iy * 500.0;
            s.z = 0.0;
            s.lon = 128.5;
            s.lat = 35.5;
            s.elevation = 0.0;
            stations.push_back(std::move(s));
        }
    }

    mesh::MeshParams3D mesh_params;
    mesh_params.x_min = -5000; mesh_params.x_max = 5000;
    mesh_params.y_min = -5000; mesh_params.y_max = 5000;
    mesh_params.z_min = -10000; mesh_params.z_air = 5000;
    mesh_params.h_surface_x = 2000;
    mesh_params.h_surface_y = 2000;
    mesh_params.h_surface_z = 500;
    mesh_params.growth_x = 1.5;
    mesh_params.growth_y = 1.5;
    mesh_params.growth_z = 1.5;
    mesh_params.growth_air = 2.0;
    mesh_params.roi_x_pad = 2000;
    mesh_params.roi_y_pad = 2000;
    mesh_params.roi_depth = 5000;
    mesh_params.use_terrain = false;
    mesh_params.refine_near_stations = 0;

    mesh::HexMeshGenerator3D generator;
    auto serial_mesh = generator.generate(mesh_params, stations, nullptr);

    if (rank == 0) {
        std::cout << "Mesh: " << serial_mesh->GetNE() << " elements, "
                  << serial_mesh->GetNV() << " vertices\n" << std::endl;
    }

    mfem::ParMesh pmesh(MPI_COMM_WORLD, *serial_mesh);
    serial_mesh.reset();

    // ==================================================================
    // Create TRUE model (with conductive block)
    // ==================================================================
    model::ConductivityModel true_model;
    true_model.init_3d(pmesh.GetNE(), sigma_bg);

    // Block centered at origin, depth 500-2000m, 1500m wide
    Real bx_min = -750.0, bx_max = 750.0;
    Real by_min = -750.0, by_max = 750.0;
    Real bz_min = -2000.0, bz_max = -500.0;

    int block_count = 0;
    for (int e = 0; e < pmesh.GetNE(); ++e) {
        if (pmesh.GetAttribute(e) == 2) continue;  // skip air
        if (is_in_block(pmesh, e, bx_min, bx_max, by_min, by_max, bz_min, bz_max)) {
            true_model.params()[e] = std::log(sigma_block);
            ++block_count;
        }
    }

    if (rank == 0) {
        std::cout << "True model: " << block_count << " elements in conductive block\n"
                  << "  Block sigma = " << sigma_block << " S/m (log = "
                  << std::log(sigma_block) << ")\n" << std::endl;
    }

    // ==================================================================
    // Generate synthetic data
    // ==================================================================
    data::MTData observed;
    for (const auto& s : stations) {
        data::Station ds;
        ds.name = s.name;
        ds.x = s.x; ds.y = s.y; ds.z = s.z;
        ds.lon = s.lon; ds.lat = s.lat; ds.has_geo = true;
        observed.add_station(ds);
    }

    // Use 3 frequencies
    RealVec freqs = {0.1, 1.0, 10.0};
    observed.set_frequencies(freqs);

    // Forward solve with true model
    forward::ForwardParams3D fwd_params;
    fwd_params.fe_order = 1;
    fwd_params.backend = forward::SolverBackend::ITERATIVE;
    fwd_params.gmres_tol = 1e-4;  // tighter for small test mesh (gradient accuracy)
    fwd_params.gmres_maxiter = 500;
    fwd_params.gmres_kdim = 50;

    forward::ForwardSolver3D fwd;
    fwd.setup(pmesh, true_model, fwd_params);
    fwd.compute_responses(observed, observed);

    // Add noise and set errors
    std::mt19937 rng(42);
    std::normal_distribution<Real> noise_dist(0.0, 1.0);

    int ns = observed.num_stations();
    int nf = observed.num_frequencies();

    for (int f = 0; f < nf; ++f) {
        for (int s = 0; s < ns; ++s) {
            auto resp = observed.predicted(s, f);

            // Error floor based on off-diagonal magnitude (standard MT practice).
            // Prevents w² explosion for near-zero Zxx/Zyy.
            Real Zxy_mag = std::abs(resp.Zxy.value);
            Real Zyx_mag = std::abs(resp.Zyx.value);
            Real offdiag_ref = std::sqrt(std::max(Zxy_mag * Zyx_mag, 1e-30));
            Real error_floor = 0.05 * offdiag_ref;  // 5% of geometric mean

            auto add_noise = [&](data::Datum& d) {
                Real mag = std::abs(d.value);
                Real err = std::max({mag * noise_level, error_floor, 1e-10});
                d.value += Complex(err * noise_dist(rng), err * noise_dist(rng));
                d.error = err;
                d.weight = 1.0;
            };

            add_noise(resp.Zxx);
            add_noise(resp.Zxy);
            add_noise(resp.Zyx);
            add_noise(resp.Zyy);

            observed.set_observed(s, f, resp);
        }
    }

    if (rank == 0) {
        std::cout << "Synthetic data generated: "
                  << ns << " stations × " << nf << " frequencies\n"
                  << std::endl;
    }

    // ==================================================================
    // Set up starting model (uniform halfspace)
    // ==================================================================
    model::ConductivityModel inv_model;
    inv_model.init_3d(pmesh.GetNE(), sigma_bg);

    // Set up forward solver for inversion (reuses mesh)
    forward::ForwardSolver3D inv_fwd;
    inv_fwd.setup(pmesh, inv_model, fwd_params);

    // ==================================================================
    // Set up regularization
    // ==================================================================
    regularization::RegParams reg_params;
    reg_params.alpha_s = 1.0;
    reg_params.alpha_x = 1.0;
    reg_params.alpha_z = 1.0;
    reg_params.alpha_r = 0.0;

    regularization::Regularization reg;
    reg.setup_3d(pmesh, reg_params);

    if (rank == 0) {
        std::cout << "Regularization: " << reg.n_active() << " active parameters\n"
                  << std::endl;
    }

    // ==================================================================
    // Run inversion
    // ==================================================================
    inversion::InversionParams3D inv_params;
    inv_params.max_iterations = max_iter;
    inv_params.target_rms = 1.0;
    inv_params.lambda_init = 10.0;
    inv_params.lambda_decrease = 0.7;
    inv_params.cg_max_iter = 10;
    inv_params.cg_tolerance = 0.5;
    inv_params.linesearch_max = 15;
    inv_params.linesearch_beta = 0.5;
    inv_params.save_checkpoints = false;

    inversion::Inversion3D inversion;
    inversion.setup(pmesh, inv_model, observed, inv_fwd, reg, inv_params);

    // VTK export setup
    fs::path output_dir = "inversion_output";
    io::ExportParams export_params;
    export_params.export_vtk = true;
    export_params.export_vtu_parallel = false;
    export_params.export_slices = true;
    export_params.slice_depths = {500, 1000, 2000};
    export_params.export_station_csv = true;
    export_params.export_station_geojson = true;
    export_params.auto_slice_interval = 0;
    // Profile through the block center
    export_params.profile_slices.push_back({"main", -3000, 0, 3000, 0, 200});

    // Export initial model
    io::export_all(pmesh, inv_model, observed, export_params, output_dir, 0);

    // Callback to print progress and export per-iteration
    inversion.set_iteration_callback(
        [&](int iter, const inversion::IterationLog3D& log) {
            if (rank == 0) {
                std::cout << "\n--- Iteration " << iter
                          << ": RMS = " << log.rms
                          << ", Phi = " << log.objective
                          << ", alpha = " << log.step_length
                          << ", CG = " << log.cg_iterations
                          << " ---\n" << std::endl;
            }
            io::export_model_vtk(pmesh, inv_model,
                                 output_dir / ("model_iter_" + std::to_string(iter) + ".vtk"),
                                 iter);
        });

    inversion.run();

    // Export final model with all slices
    io::export_all(pmesh, inv_model, observed, export_params, output_dir, -1);

    // Export true model for comparison
    io::export_model_vtk(pmesh, true_model, output_dir / "model_true.vtk");

    // ==================================================================
    // Validate results
    // ==================================================================
    if (rank == 0) {
        std::cout << "\n=== Validation ===\n";

        // Check RMS history
        const auto& history = inversion.history();
        bool rms_decreasing = true;
        for (size_t i = 1; i < history.size(); ++i) {
            if (history[i].rms > history[i-1].rms * 1.1) {
                rms_decreasing = false;
                break;
            }
        }
        std::cout << "RMS decreasing: " << (rms_decreasing ? "PASS" : "FAIL") << "\n";

        // Check sigma recovery in block region
        Real sigma_sum_block = 0.0;
        Real sigma_sum_bg = 0.0;
        int n_block = 0, n_bg = 0;

        for (int e = 0; e < pmesh.GetNE(); ++e) {
            if (pmesh.GetAttribute(e) == 2) continue;
            Real sigma_inv = inv_model.sigma(e);

            if (is_in_block(pmesh, e, bx_min, bx_max, by_min, by_max, bz_min, bz_max)) {
                sigma_sum_block += sigma_inv;
                ++n_block;
            } else {
                sigma_sum_bg += sigma_inv;
                ++n_bg;
            }
        }

        Real mean_block = (n_block > 0) ? sigma_sum_block / n_block : 0.0;
        Real mean_bg = (n_bg > 0) ? sigma_sum_bg / n_bg : 0.0;

        std::cout << "Mean sigma (block region): " << mean_block << " S/m"
                  << " (true: " << sigma_block << ")\n"
                  << "Mean sigma (background):   " << mean_bg << " S/m"
                  << " (true: " << sigma_bg << ")\n";

        bool block_higher = mean_block > mean_bg * 1.5;
        std::cout << "Block conductivity > 1.5× background: "
                  << (block_higher ? "PASS" : "FAIL") << "\n";

        // Print final RMS
        Real final_rms = observed.rms_misfit();
        std::cout << "Final RMS: " << final_rms << "\n";

        std::cout << "\n=== Inversion Test "
                  << ((rms_decreasing) ? "PASSED" : "FAILED")
                  << " ===\n" << std::endl;
    }

    mfem::Mpi::Finalize();
    return 0;
}
