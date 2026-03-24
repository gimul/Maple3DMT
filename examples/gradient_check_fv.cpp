// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file gradient_check_fv.cpp
/// @brief Finite-difference gradient verification for Octree FV inversion.
///
/// Creates a simple 1D halfspace with a conductivity anomaly, then compares
/// the adjoint gradient with central finite differences for selected cells.
///
/// Usage: mpirun -np 1 gradient_check_fv

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/octree/staggered_grid.h"
#include "maple3dmt/octree/operators.h"
#include "maple3dmt/forward/forward_solver_fv.h"
#include "maple3dmt/inversion/inversion_fv.h"
#include "maple3dmt/inversion/regularization_octree.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/utils/logger.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

using namespace maple3dmt;

// namespace maple3dmt { namespace forward { void set_sensitivity_diag_cell(int c); } }

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    utils::Logger::instance().set_rank(rank);

    // ---- 1. Build octree mesh ----
    octree::RefinementParams mesh_params;
    mesh_params.domain_x_min = -100000;  mesh_params.domain_x_max = 100000;
    mesh_params.domain_y_min = -100000;  mesh_params.domain_y_max = 100000;
    mesh_params.domain_z_min = -100000;  mesh_params.domain_z_max = 50000;
    mesh_params.min_level = 3;
    mesh_params.max_level = 4;  // coarse for speed
    mesh_params.station_refine_radius = 30000;
    mesh_params.station_refine_level = 4;
    mesh_params.sigma_bg = 0.01;

    // Two stations
    std::vector<std::array<Real,3>> stations = {
        {0.0, 0.0, 0.0},
        {5000.0, 0.0, 0.0}
    };
    RealVec freqs = {1.0};  // single frequency for speed

    octree::OctreeMesh mesh;
    mesh.setup(mesh_params, stations, freqs, MPI_COMM_WORLD);
    mesh.set_terrain([](Real, Real) { return 0.0; });
    mesh.build_staggered_grid();

    int nc = mesh.staggered().num_cells();
    std::cout << "\n=== Gradient Check (Octree FV) ===" << std::endl;
    std::cout << "  Cells: " << nc << std::endl;
    std::cout << "  Edges: " << mesh.staggered().num_edges() << std::endl;

    // ---- 2. Setup conductivity model ----
    // Background: 0.01 S/m = ln(0.01) = -4.605
    Real sigma_bg = 0.01;
    Real ln_sigma_bg = std::log(sigma_bg);
    RealVec log_sigma(nc, ln_sigma_bg);

    // Add a block anomaly: 0.1 S/m in center region
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH) {
            log_sigma[c] = std::log(1e-8);  // air
            continue;
        }
        Real cx, cy, cz;
        mesh.cell_center(c, cx, cy, cz);
        if (std::abs(cx) < 10000 && std::abs(cy) < 10000 &&
            cz > -10000 && cz < -1000) {
            log_sigma[c] = std::log(0.1);
        }
    }

    model::ConductivityModel model;
    model.init_3d(nc, sigma_bg);
    // Override with our log_sigma values
    model.params() = log_sigma;
    model.invalidate_cache();

    // ---- 3. Setup forward solver ----
    // Parse --divcorr flag (default: ON, matching new adjoint DivCorr)
    bool use_divcorr = true;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-divcorr") use_divcorr = false;
        if (arg == "--divcorr")    use_divcorr = true;
    }

    forward::ForwardParamsFV fwd_params;
    fwd_params.bicgstab_tol = 1e-8;
    fwd_params.bicgstab_maxiter = 3000;
    fwd_params.print_level = 0;
    fwd_params.div_correction = use_divcorr;
    std::cout << "  DivCorr: " << (use_divcorr ? "ON" : "OFF") << std::endl;

    forward::ForwardSolverFV fwd;
    fwd.setup(mesh, fwd_params);

    // Set sigma from model
    auto update_fwd_sigma = [&]() {
        fwd.update_sigma(model);
    };
    update_fwd_sigma();

    // ---- 4. Setup synthetic data ----
    data::MTData observed;
    observed.set_frequencies(freqs);
    for (size_t i = 0; i < stations.size(); ++i) {
        data::Station sta;
        sta.name = "STA" + std::to_string(i);
        sta.x = stations[i][0];
        sta.y = stations[i][1];
        observed.add_station(sta);
    }

    // Initialize observed data (generate synthetic)
    data::MTData predicted = observed;
    fwd.compute_responses(observed, predicted);

    // Copy predicted to observed, then add noise as "target"
    for (int f = 0; f < observed.num_frequencies(); ++f) {
        for (int s = 0; s < observed.num_stations(); ++s) {
            auto resp = predicted.predicted(s, f);
            // Set as observed with 5% error floor
            Real err_floor = 0.05;
            auto set_obs = [&](data::Datum& d, const data::Datum& p) {
                d.value = p.value * Complex(1.02, 0.01);  // slight perturbation
                d.error = std::abs(p.value) * err_floor;
                if (d.error < 1e-10) d.error = 1e-10;
                d.weight = 1.0;
            };
            data::MTResponse obs_resp;
            set_obs(obs_resp.Zxy, resp.Zxy);
            set_obs(obs_resp.Zyx, resp.Zyx);
            set_obs(obs_resp.Zxx, resp.Zxx);
            set_obs(obs_resp.Zyy, resp.Zyy);
            observed.set_observed(s, f, obs_resp);
        }
    }

    // ---- 5. Setup regularization ----
    inversion::RegParamsOctree reg_params;
    reg_params.alpha_s = 1.0;
    reg_params.alpha_x = 1.0;
    reg_params.alpha_y = 1.0;
    reg_params.alpha_z = 1.0;

    inversion::RegularizationOctree reg;
    reg.setup(mesh, reg_params);

    // Reference model
    RealVec ref_active(reg.n_active(), ln_sigma_bg);
    reg.set_reference_model(ref_active);

    // ---- 6. Setup inversion (for gradient computation) ----
    Real fd_lambda = 1.0;  // must match FD lambda below

    inversion::InversionParamsFV inv_params;
    inv_params.max_iterations = 1;
    inv_params.lambda_init = fd_lambda;

    inversion::InversionFV inv;
    inv.setup(model, observed, fwd, reg, inv_params);

    // ---- 7. Compute adjoint gradient ----
    std::cout << "\n  Computing adjoint gradient..." << std::endl;

    // First: forward solve to fill predicted
    fwd.compute_responses(observed, predicted);

    // Get adjoint gradient
    RealVec grad_adj = inv.gradient();
    std::cout << "  Adjoint gradient computed (" << grad_adj.size() << " active params)"
              << std::endl;

    // ---- 8. Finite difference gradient check ----
    std::cout << "\n  Computing finite-difference gradient for selected cells..."
              << std::endl;

    Real eps = 1e-4;  // FD perturbation in ln(σ)

    // Select cells for gradient check:
    // (a) Anomaly cells (σ ≠ σ_bg) → nonzero regularization gradient
    // (b) Cells near stations but NOT the station cell itself → nonzero data gradient
    // Station cells are excluded: nearest-cell interpolation makes adjoint singular there.
    std::vector<int> check_cells;
    const auto& a2g = reg.active_to_global();

    // Find station cell IDs (to exclude)
    std::vector<int> station_cell_ids;
    for (const auto& sta : stations) {
        int best = -1; Real best_d = 1e30;
        for (int c = 0; c < nc; ++c) {
            Real cx, cy, cz; mesh.cell_center(c, cx, cy, cz);
            Real d = std::sqrt((cx-sta[0])*(cx-sta[0]) + (cy-sta[1])*(cy-sta[1]) + cz*cz);
            if (d < best_d) { best_d = d; best = c; }
        }
        if (best >= 0) station_cell_ids.push_back(best);
    }

    // (a) Anomaly cells (away from stations)
    for (int j = 0; j < reg.n_active() && check_cells.size() < 5; ++j) {
        int g = a2g[j];
        Real cx, cy, cz; mesh.cell_center(g, cx, cy, cz);
        if (cz >= 0) continue;
        // Must be in anomaly region
        if (std::abs(log_sigma[g] - ln_sigma_bg) < 0.01) continue;
        // Not a station cell
        bool is_sta = false;
        for (int sc : station_cell_ids) if (g == sc) { is_sta = true; break; }
        if (is_sta) continue;
        check_cells.push_back(j);
    }

    // (b) Background cells near anomaly boundary
    for (int j = 0; j < reg.n_active() && check_cells.size() < 10; ++j) {
        int g = a2g[j];
        Real cx, cy, cz; mesh.cell_center(g, cx, cy, cz);
        if (cz >= 0) continue;
        // Background cell near anomaly
        if (std::abs(log_sigma[g] - ln_sigma_bg) > 0.01) continue;
        if (std::abs(cx) > 30000 || std::abs(cy) > 30000) continue;
        if (cz < -20000) continue;
        bool is_sta = false;
        for (int sc : station_cell_ids) if (g == sc) { is_sta = true; break; }
        if (is_sta) continue;
        check_cells.push_back(j);
    }

    std::cout << "  Selected " << check_cells.size() << " cells for gradient check"
              << std::endl;

    std::cout << "\n  " << std::setw(6) << "Cell"
              << std::setw(14) << "g_adj"
              << std::setw(14) << "g_fd"
              << std::setw(14) << "rel_err"
              << std::setw(8) << "Status"
              << std::endl;
    std::cout << "  " << std::string(56, '-') << std::endl;

    // lambda must match fd_lambda used in inversion setup
    bool all_pass = true;

    for (int active_idx : check_cells) {
        int g = a2g[active_idx];
        auto& params = model.params();
        Real orig = params[g];

        // Φ(m + ε)
        params[g] = orig + eps;
        model.invalidate_cache();
        update_fwd_sigma();

        fwd.compute_responses(observed, predicted);
        Real phi_plus = 0.0;
        for (int f = 0; f < observed.num_frequencies(); ++f) {
            for (int s = 0; s < observed.num_stations(); ++s) {
                const auto& obs = observed.observed(s, f);
                const auto& pred = predicted.predicted(s, f);
                auto add = [&](const data::Datum& o, const data::Datum& p) {
                    if (o.weight <= 0.0 || o.error <= 0.0) return;
                    Real w = 1.0 / o.error;
                    phi_plus += 0.5 * w * w * std::norm(o.value - p.value);
                };
                add(obs.Zxx, pred.Zxx);
                add(obs.Zxy, pred.Zxy);
                add(obs.Zyx, pred.Zyx);
                add(obs.Zyy, pred.Zyy);
            }
        }
        phi_plus += fd_lambda * reg.evaluate(model);

        // Φ(m - ε)
        params[g] = orig - eps;
        model.invalidate_cache();
        update_fwd_sigma();

        fwd.compute_responses(observed, predicted);
        Real phi_minus = 0.0;
        for (int f = 0; f < observed.num_frequencies(); ++f) {
            for (int s = 0; s < observed.num_stations(); ++s) {
                const auto& obs = observed.observed(s, f);
                const auto& pred = predicted.predicted(s, f);
                auto add = [&](const data::Datum& o, const data::Datum& p) {
                    if (o.weight <= 0.0 || o.error <= 0.0) return;
                    Real w = 1.0 / o.error;
                    phi_minus += 0.5 * w * w * std::norm(o.value - p.value);
                };
                add(obs.Zxx, pred.Zxx);
                add(obs.Zxy, pred.Zxy);
                add(obs.Zyx, pred.Zyx);
                add(obs.Zyy, pred.Zyy);
            }
        }
        phi_minus += fd_lambda * reg.evaluate(model);

        // Central FD gradient
        Real g_fd = (phi_plus - phi_minus) / (2.0 * eps);

        // Restore
        params[g] = orig;
        model.invalidate_cache();

        // Compare
        Real g_adj_val = grad_adj[active_idx];
        // Use relative error when gradients are significant,
        // absolute error when both are near zero
        Real abs_err = std::abs(g_adj_val - g_fd);
        Real scale = std::max(std::abs(g_fd), std::abs(g_adj_val));
        Real rel_err = (scale > 1e-6) ? abs_err / scale : abs_err;

        bool pass = (scale > 1e-6) ? (rel_err < 0.1) : (abs_err < 1e-4);
        if (!pass) all_pass = false;

        std::cout << "  " << std::setw(6) << g
                  << std::setw(14) << std::scientific << std::setprecision(4) << g_adj_val
                  << std::setw(14) << g_fd
                  << std::setw(14) << std::fixed << std::setprecision(4) << rel_err
                  << std::setw(8) << (pass ? "PASS" : "FAIL")
                  << std::endl;
    }

    // Restore sigma
    update_fwd_sigma();

    std::cout << "\n  Gradient check: " << (all_pass ? "ALL PASS" : "SOME FAILED")
              << std::endl;
    std::cout << "\n=== Gradient Check Complete ===" << std::endl;

    MPI_Finalize();
    return all_pass ? 0 : 1;
}
