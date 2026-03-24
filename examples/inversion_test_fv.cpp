// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file inversion_test_fv.cpp
/// @brief Synthetic COMMEMI 3D-1 inversion test using Octree FV backend.
///
/// Workflow:
///   1. Build octree mesh with conductive block anomaly
///   2. Forward solve with true model → synthetic observed data
///   3. Add Gaussian noise (5% of |Z|)
///   4. Start inversion from uniform halfspace
///   5. Run NLCG for 10 iterations
///   6. Validate: RMS decreases, block conductivity recovered
///
/// This is a self-contained integration test for the full inversion pipeline:
///   ForwardSolverFV → adjoint → gradient → NLCG → line search → model update

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/octree/staggered_grid.h"
#include "maple3dmt/octree/operators.h"
#include "maple3dmt/forward/forward_solver_fv.h"
#include "maple3dmt/inversion/inversion_fv.h"
#include "maple3dmt/inversion/regularization_octree.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/io/vtk_export_octree.h"
#include "maple3dmt/utils/logger.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include <mpi.h>

using namespace maple3dmt;
using Clock = std::chrono::steady_clock;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    utils::Logger::instance().set_rank(rank);

    if (rank == 0) {
        std::cout << "\n======================================================" << std::endl;
        std::cout << "  Synthetic Inversion Test: COMMEMI 3D-1 (Octree FV)" << std::endl;
        std::cout << "======================================================" << std::endl;
    }

    auto t_start = Clock::now();

    // ---------------------------------------------------------------
    // 1. True model parameters
    // ---------------------------------------------------------------
    const Real sigma_bg    = 0.01;   // 100 Ohm-m halfspace
    const Real sigma_block = 0.1;    // 10 Ohm-m block
    const Real block_xmin  = -5000,  block_xmax = 5000;
    const Real block_ymin  = -5000,  block_ymax = 5000;
    const Real block_zmin  = -5000,  block_zmax = 0;    // surface to 5 km depth

    // 2 frequencies: shallow (block) + deep (halfspace)
    std::vector<Real> frequencies = {0.01, 1.0};
    int nf = static_cast<int>(frequencies.size());

    // 5 stations: center + edges + far field
    std::vector<std::array<Real,3>> station_xyz = {
        { 0.0,     0.0, 0.0},   // center of block
        { 5000.0,  0.0, 0.0},   // block edge (+x)
        {10000.0,  0.0, 0.0},   // outside block
        { 0.0,  5000.0, 0.0},   // block edge (+y)
        {20000.0,  0.0, 0.0},   // far field
    };
    std::vector<std::string> station_names = {
        "center", "edge_x", "outside", "edge_y", "far"
    };
    int ns = static_cast<int>(station_xyz.size());

    if (rank == 0) {
        std::cout << "\n  True model: 10 Ohm-m block in 100 Ohm-m halfspace" << std::endl;
        std::cout << "  Block: [" << block_xmin/1000 << "," << block_xmax/1000
                  << "] x [" << block_ymin/1000 << "," << block_ymax/1000
                  << "] x [" << block_zmin/1000 << "," << block_zmax/1000
                  << "] km" << std::endl;
        std::cout << "  Stations: " << ns << std::endl;
        std::cout << "  Frequencies: " << nf << " (";
        for (int f = 0; f < nf; ++f) {
            if (f > 0) std::cout << ", ";
            std::cout << frequencies[f] << " Hz";
        }
        std::cout << ")" << std::endl;
    }

    // ---------------------------------------------------------------
    // 2. Build octree mesh (L5-L7 for speed)
    // ---------------------------------------------------------------
    octree::RefinementParams mesh_params;
    mesh_params.domain_x_min = -200000;  mesh_params.domain_x_max = 200000;
    mesh_params.domain_y_min = -200000;  mesh_params.domain_y_max = 200000;
    mesh_params.domain_z_min = -200000;  mesh_params.domain_z_max = 56000;

    mesh_params.min_level = 5;
    mesh_params.max_level = 7;
    mesh_params.station_refine_radius = 25000;
    mesh_params.station_refine_level = 7;
    mesh_params.sigma_bg = sigma_bg;
    mesh_params.replicate_mesh = true;  // All ranks own full mesh (freq-parallel)

    // Refine around the block
    octree::RefineRegion block_region;
    block_region.x_min = block_xmin;  block_region.x_max = block_xmax;
    block_region.y_min = block_ymin;  block_region.y_max = block_ymax;
    block_region.z_min = block_zmin;  block_region.z_max = block_zmax;
    block_region.level = 7;
    block_region.padding = 5000;
    mesh_params.refine_regions.push_back(block_region);

    octree::OctreeMesh mesh;
    mesh.setup(mesh_params, station_xyz, {frequencies.front()}, MPI_COMM_WORLD);
    mesh.set_terrain([](Real, Real) { return 0.0; });
    mesh.build_staggered_grid();

    int nc = mesh.staggered().num_cells();
    int ne = mesh.staggered().num_edges();

    if (rank == 0) {
        std::cout << "\n  Mesh: " << nc << " cells, " << ne << " edge DOFs" << std::endl;
        std::cout << "  Levels: " << mesh_params.min_level << "-"
                  << mesh_params.max_level << std::endl;
    }

    // ---------------------------------------------------------------
    // 3. Setup TRUE conductivity model (with block)
    // ---------------------------------------------------------------
    model::ConductivityModel true_model;
    true_model.init_3d(nc, sigma_bg);

    auto& true_log_sigma = true_model.params();
    int n_block_cells = 0;
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH) {
            true_log_sigma[c] = std::log(1e-8);  // air
            continue;
        }
        Real cx, cy, cz;
        mesh.cell_center(c, cx, cy, cz);
        if (cx >= block_xmin && cx <= block_xmax &&
            cy >= block_ymin && cy <= block_ymax &&
            cz >= block_zmin && cz <= block_zmax) {
            true_log_sigma[c] = std::log(sigma_block);
            ++n_block_cells;
        }
    }
    true_model.invalidate_cache();

    if (rank == 0)
        std::cout << "  Block cells: " << n_block_cells << std::endl;

    // ---------------------------------------------------------------
    // 4. Forward solve with true model → synthetic data
    // ---------------------------------------------------------------
    forward::ForwardParamsFV fwd_params;
    fwd_params.bicgstab_tol = 1e-8;
    fwd_params.bicgstab_maxiter = 10000;
    fwd_params.print_level = 1;
    fwd_params.div_correction = true;
    fwd_params.scattered_field = false;
    fwd_params.air_z_threshold = 1.0;
    fwd_params.air_bc_iterations = 1;

    forward::ForwardSolverFV fwd;
    fwd.setup(mesh, fwd_params);
    fwd.update_sigma(true_model);

    // Setup MT data container (observed = placeholder 1D Z for impedance extraction)
    data::MTData observed;
    for (int s = 0; s < ns; ++s) {
        data::Station st;
        st.name = station_names[s];
        st.x = station_xyz[s][0];
        st.y = station_xyz[s][1];
        st.z = station_xyz[s][2];
        observed.add_station(st);
    }
    observed.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));

    // Initialize observed with 1D halfspace Z (will be overwritten by forward solve)
    for (int f = 0; f < nf; ++f) {
        Real omega = constants::TWOPI * frequencies[f];
        Complex iwmu(0, omega * constants::MU0);
        Complex Z_1d = std::sqrt(iwmu / Complex(sigma_bg, 0));
        for (int s = 0; s < ns; ++s) {
            data::MTResponse resp;
            resp.Zxy.value = Z_1d;
            resp.Zxy.error = 0.05 * std::abs(Z_1d);
            resp.Zxy.weight = 1.0;
            resp.Zyx.value = -Z_1d;
            resp.Zyx.error = 0.05 * std::abs(Z_1d);
            resp.Zyx.weight = 1.0;
            observed.set_observed(s, f, resp);
        }
    }

    if (rank == 0)
        std::cout << "\n--- Forward solve (true model) ---" << std::endl;

    data::MTData synthetic;
    for (int s = 0; s < ns; ++s) {
        data::Station st;
        st.name = station_names[s];
        st.x = station_xyz[s][0];
        st.y = station_xyz[s][1];
        st.z = station_xyz[s][2];
        synthetic.add_station(st);
    }
    synthetic.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));

    fwd.compute_responses(observed, synthetic);

    // ---------------------------------------------------------------
    // 5. Copy synthetic predictions → observed data + add noise
    // ---------------------------------------------------------------
    if (rank == 0)
        std::cout << "\n--- Synthetic data generation ---" << std::endl;

    std::mt19937 rng(42);  // fixed seed for reproducibility
    std::normal_distribution<Real> noise(0.0, 1.0);

    Real noise_floor_pct = 0.05;  // 5% noise

    for (int f = 0; f < nf; ++f) {
        Real omega = constants::TWOPI * frequencies[f];
        for (int s = 0; s < ns; ++s) {
            const auto& pred = synthetic.predicted(s, f);
            data::MTResponse obs_resp;

            // Zxy: add noise to real and imaginary parts
            {
                Real err = noise_floor_pct * std::abs(pred.Zxy.value);
                if (err < 1e-20) err = 1e-20;
                Complex noisy = pred.Zxy.value +
                    Complex(noise(rng) * err, noise(rng) * err);
                obs_resp.Zxy.value = noisy;
                obs_resp.Zxy.error = err;
                obs_resp.Zxy.weight = 1.0;
            }
            // Zyx
            {
                Real err = noise_floor_pct * std::abs(pred.Zyx.value);
                if (err < 1e-20) err = 1e-20;
                Complex noisy = pred.Zyx.value +
                    Complex(noise(rng) * err, noise(rng) * err);
                obs_resp.Zyx.value = noisy;
                obs_resp.Zyx.error = err;
                obs_resp.Zyx.weight = 1.0;
            }

            observed.set_observed(s, f, obs_resp);
        }
    }

    // Print synthetic observed data summary
    if (rank == 0) {
        std::cout << "\n  Synthetic observed data (with 5% noise):\n" << std::endl;
        std::cout << "  " << std::setw(10) << "station"
                  << "  " << std::setw(8) << "freq"
                  << "  " << std::setw(10) << "rho_xy"
                  << "  " << std::setw(10) << "phi_xy"
                  << "  " << std::setw(10) << "rho_yx"
                  << "  " << std::setw(10) << "phi_yx" << std::endl;
        std::cout << "  " << std::string(62, '-') << std::endl;

        for (int f = 0; f < nf; ++f) {
            Real omega = constants::TWOPI * frequencies[f];
            for (int s = 0; s < ns; ++s) {
                const auto& o = observed.observed(s, f);
                Complex Zxy = o.Zxy.value, Zyx = o.Zyx.value;
                Real rho_xy = std::norm(Zxy) / (omega * constants::MU0);
                Real phi_xy = std::arg(Zxy) * 180.0 / constants::PI;
                Real rho_yx = std::norm(Zyx) / (omega * constants::MU0);
                Real phi_yx = std::arg(Zyx) * 180.0 / constants::PI;

                char fbuf[16];
                snprintf(fbuf, sizeof(fbuf), "%.2g Hz", frequencies[f]);
                std::cout << "  " << std::setw(10) << station_names[s]
                          << "  " << std::setw(8) << fbuf
                          << "  " << std::fixed << std::setprecision(1)
                          << std::setw(10) << rho_xy
                          << "  " << std::setw(10) << phi_xy
                          << "  " << std::setw(10) << rho_yx
                          << "  " << std::setw(10) << phi_yx << std::endl;
            }
        }
    }

    // ---------------------------------------------------------------
    // 6. Setup starting model (uniform halfspace)
    // ---------------------------------------------------------------
    model::ConductivityModel inv_model;
    inv_model.init_3d(nc, sigma_bg);

    // Set air cells
    auto& inv_log_sigma = inv_model.params();
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH)
            inv_log_sigma[c] = std::log(1e-8);
    }
    inv_model.invalidate_cache();

    // ---------------------------------------------------------------
    // 7. Setup forward solver for inversion (lower tolerance for speed)
    // ---------------------------------------------------------------
    forward::ForwardParamsFV inv_fwd_params;
    inv_fwd_params.bicgstab_tol = 1e-7;
    inv_fwd_params.bicgstab_maxiter = 5000;
    inv_fwd_params.print_level = 0;  // quiet during inversion
    inv_fwd_params.div_correction = true;
    inv_fwd_params.scattered_field = false;
    inv_fwd_params.air_z_threshold = 1.0;
    inv_fwd_params.air_bc_iterations = 1;

    forward::ForwardSolverFV inv_fwd;
    inv_fwd.setup(mesh, inv_fwd_params);

    // ---------------------------------------------------------------
    // 8. Setup regularisation
    // ---------------------------------------------------------------
    inversion::RegParamsOctree reg_params;
    reg_params.alpha_s = 1e-4;   // smallness
    reg_params.alpha_x = 1.0;
    reg_params.alpha_y = 1.0;
    reg_params.alpha_z = 0.5;

    inversion::RegularizationOctree reg;
    reg.setup(mesh, reg_params);

    // Reference model = starting halfspace
    RealVec ref_model = inv_log_sigma;
    reg.set_reference_model(ref_model);

    if (rank == 0) {
        std::cout << "\n  Active parameters: " << reg.n_active() << std::endl;
    }

    // ---------------------------------------------------------------
    // 9. Setup inversion (NLCG)
    // ---------------------------------------------------------------
    inversion::InversionParamsFV inv_params;
    inv_params.solver = inversion::InversionParamsFV::Solver::NLCG;
    inv_params.max_iterations = 10;
    inv_params.target_rms = 1.0;
    inv_params.lambda_init = 10.0;
    inv_params.lambda_decrease = 0.7;
    inv_params.linesearch_max = 6;
    inv_params.linesearch_startdm = 20.0;
    inv_params.save_checkpoints = false;

    inversion::InversionFV inv;
    inv.setup(inv_model, observed, inv_fwd, reg, inv_params);

    // ---------------------------------------------------------------
    // 10. Run inversion
    // ---------------------------------------------------------------
    if (rank == 0)
        std::cout << "\n--- Inversion (NLCG, max " << inv_params.max_iterations
                  << " iter) ---\n" << std::endl;

    auto t_inv_start = Clock::now();
    auto t_last = t_inv_start;

    inv.set_iteration_callback(
        [&](int iter, const inversion::IterationLogFV& entry) {
            auto t_now = Clock::now();
            Real dt = std::chrono::duration<Real>(t_now - t_last).count();
            t_last = t_now;

            if (rank == 0) {
                std::cout << "  Iter " << std::setw(2) << entry.iteration
                          << "  Phi=" << std::scientific << std::setprecision(4)
                          << entry.objective
                          << "  RMS=" << std::fixed << std::setprecision(3)
                          << entry.rms
                          << "  lambda=" << std::scientific << std::setprecision(2)
                          << entry.lambda
                          << "  step=" << std::fixed << std::setprecision(4)
                          << entry.step_length
                          << "  (" << std::setprecision(1) << dt << "s)"
                          << std::endl;
            }
        });

    inv.run();

    auto t_inv_end = Clock::now();
    Real inv_time = std::chrono::duration<Real>(t_inv_end - t_inv_start).count();

    // ---------------------------------------------------------------
    // 11. Validation
    // ---------------------------------------------------------------
    if (rank == 0) {
        std::cout << "\n======================================================" << std::endl;
        std::cout << "  Validation" << std::endl;
        std::cout << "======================================================" << std::endl;
    }

    const auto& history = inv.history();

    // 11a. RMS should decrease
    bool rms_decreasing = true;
    if (history.size() >= 2) {
        Real rms_first = history.front().rms;
        Real rms_last  = history.back().rms;
        rms_decreasing = (rms_last < rms_first);
    }

    if (rank == 0) {
        std::cout << "\n  Convergence history:" << std::endl;
        for (const auto& h : history) {
            std::cout << "    Iter " << h.iteration
                      << "  RMS=" << std::fixed << std::setprecision(3) << h.rms
                      << "  Phi=" << std::scientific << std::setprecision(4) << h.objective
                      << "  lambda=" << h.lambda << std::endl;
        }
    }

    // 11b. Check conductivity recovery in block region
    Real sigma_block_avg = 0.0;
    Real sigma_bg_avg = 0.0;
    int n_blk = 0, n_bg = 0;

    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH) continue;

        Real cx, cy, cz;
        mesh.cell_center(c, cx, cy, cz);
        Real sigma_inv = inv_model.sigma(c);

        // Inside block (core, not edges)
        if (std::abs(cx) < 3000 && std::abs(cy) < 3000 &&
            cz > block_zmin + 1000 && cz < block_zmax - 500) {
            sigma_block_avg += sigma_inv;
            ++n_blk;
        }
        // Background (far from block)
        else if (std::abs(cx) > 15000 && cz < -1000 && cz > -10000) {
            sigma_bg_avg += sigma_inv;
            ++n_bg;
        }
    }

    if (n_blk > 0) sigma_block_avg /= n_blk;
    if (n_bg > 0)  sigma_bg_avg /= n_bg;

    Real rho_block_inv = (sigma_block_avg > 0) ? 1.0 / sigma_block_avg : 0;
    Real rho_bg_inv    = (sigma_bg_avg > 0)    ? 1.0 / sigma_bg_avg : 0;

    if (rank == 0) {
        std::cout << "\n  Model recovery:" << std::endl;
        std::cout << "    Block core: sigma_avg = " << std::scientific << std::setprecision(4)
                  << sigma_block_avg << " S/m (rho = " << std::fixed << std::setprecision(1)
                  << rho_block_inv << " Ohm-m)" << std::endl;
        std::cout << "    True block: sigma = " << sigma_block << " S/m (rho = "
                  << 1.0/sigma_block << " Ohm-m)" << std::endl;
        std::cout << "    Background: sigma_avg = " << std::scientific << std::setprecision(4)
                  << sigma_bg_avg << " S/m (rho = " << std::fixed << std::setprecision(1)
                  << rho_bg_inv << " Ohm-m)" << std::endl;
        std::cout << "    True bg:    sigma = " << sigma_bg << " S/m (rho = "
                  << 1.0/sigma_bg << " Ohm-m)" << std::endl;
    }

    // 11c. Summary pass/fail
    bool block_recovered = (sigma_block_avg > sigma_bg * 2.0);  // at least 2x more conductive
    bool bg_preserved = (std::abs(sigma_bg_avg - sigma_bg) / sigma_bg < 0.5);  // within 50%

    Real rms_final = history.empty() ? 99.0 : history.back().rms;
    Real rms_initial = history.empty() ? 99.0 : history.front().rms;

    if (rank == 0) {
        std::cout << "\n  Test results:" << std::endl;
        std::cout << "    [" << (rms_decreasing ? "PASS" : "FAIL")
                  << "] RMS decreasing: " << std::fixed << std::setprecision(3)
                  << rms_initial << " -> " << rms_final << std::endl;
        std::cout << "    [" << (block_recovered ? "PASS" : "FAIL")
                  << "] Block conductivity recovered (sigma_block > 2x sigma_bg): "
                  << std::scientific << sigma_block_avg << " vs " << sigma_bg << std::endl;
        std::cout << "    [" << (bg_preserved ? "PASS" : "FAIL")
                  << "] Background preserved (within 50%): "
                  << std::scientific << sigma_bg_avg << " vs " << sigma_bg << std::endl;

        bool all_pass = rms_decreasing && block_recovered && bg_preserved;
        std::cout << "\n  Overall: " << (all_pass ? "ALL TESTS PASS" : "SOME TESTS FAILED")
                  << std::endl;

        auto t_total = Clock::now();
        Real total_time = std::chrono::duration<Real>(t_total - t_start).count();
        std::cout << "\n  Inversion time: " << std::fixed << std::setprecision(1)
                  << inv_time << "s" << std::endl;
        std::cout << "  Total time: " << total_time << "s" << std::endl;
    }

    // ---------------------------------------------------------------
    // 12. Export results (skip in scalability mode via env var)
    // ---------------------------------------------------------------
    const char* skip_export = std::getenv("SKIP_VTK_EXPORT");
    if (!skip_export) {
        io::OctreeExportParams export_params;
        export_params.auto_slice_interval = 2000;
        export_params.slice_dx = 500;
        export_params.slice_dy = 500;

        io::export_octree_all(mesh, inv_model, observed,
                              "output_inversion_test", export_params,
                              history.empty() ? 0 : history.back().iteration);
    }

    if (rank == 0) {
        std::cout << "\n  Results exported to: output_inversion_test/" << std::endl;
        std::cout << "\n=== Inversion Test Complete ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
