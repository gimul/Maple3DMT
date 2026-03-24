// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file commemi_3d2_inversion.cpp
/// @brief COMMEMI 3D-2 synthetic inversion benchmark for publication.
///
/// Model: Two blocks in a 3-layer earth.
///   Layer 1 (host): 10 Ωm, z ∈ [0, −3 km]
///   Layer 2:       100 Ωm, z ∈ [−3, −10 km]
///   Layer 3:       0.1 Ωm, z ∈ [−10 km, −∞]  (conductive basement)
///
///   Block A (conductive): 1 Ωm, x∈[−5,5] y∈[2,8] z∈[−3,−5] km
///   Block B (resistive):  100 Ωm, x∈[−5,5] y∈[−8,−2] z∈[−3,−5] km
///
/// Reference: Zhdanov et al. (1997), COMMEMI.

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
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include <sys/stat.h>
#include <mpi.h>

using namespace maple3dmt;
using Clock = std::chrono::steady_clock;

static void mkdir_p(const std::string& path) {
    ::mkdir(path.c_str(), 0755);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    utils::Logger::instance().set_rank(rank);

    const std::string output_dir = "output_commemi_3d2_uniform";
    if (rank == 0) mkdir_p(output_dir);

    if (rank == 0) {
        std::cout << "\n======================================================"
                  << "\n  COMMEMI 3D-2 Synthetic Inversion (Octree FV)"
                  << "\n  Two blocks in 3-layer earth"
                  << "\n======================================================"
                  << std::endl;
    }

    auto t_start = Clock::now();

    // ---------------------------------------------------------------
    // 1. True model: 3-layer earth + two blocks
    // ---------------------------------------------------------------
    // Layered background
    const Real sigma_L1 = 0.1;    // 10 Ωm, 0 to −3 km
    const Real sigma_L2 = 0.01;   // 100 Ωm, −3 to −10 km
    const Real sigma_L3 = 10.0;   // 0.1 Ωm, below −10 km
    const Real z_L1_bot = -3000;
    const Real z_L2_bot = -10000;

    // Block A (conductive): 1 Ωm
    const Real sigma_A = 1.0;
    const Real Ax0 = -5000, Ax1 = 5000;
    const Real Ay0 =  2000, Ay1 = 8000;
    const Real Az0 = -5000, Az1 = -3000;

    // Block B (resistive): 100 Ωm
    const Real sigma_B = 0.01;
    const Real Bx0 = -5000, Bx1 = 5000;
    const Real By0 = -8000, By1 = -2000;
    const Real Bz0 = -5000, Bz1 = -3000;

    // 5 frequencies
    std::vector<Real> frequencies = {0.01, 0.032, 0.1, 0.32, 1.0};
    int nf = static_cast<int>(frequencies.size());

    // ---------------------------------------------------------------
    // 2. Areal station grid (typical 3-D MT survey layout)
    // ---------------------------------------------------------------
    std::vector<std::array<Real,3>> station_xyz;
    std::vector<std::string> station_names;

    // 2-D grid: −15 to +15 km, 3 km spacing → 11×11 = 121 stations
    const Real sta_min = -15000, sta_max = 15000, sta_dx = 3000;
    for (Real y = sta_min; y <= sta_max + 1; y += sta_dx) {
        for (Real x = sta_min; x <= sta_max + 1; x += sta_dx) {
            station_xyz.push_back({x, y, 0.0});
            char name[64];
            snprintf(name, sizeof(name), "S%+.0f_%+.0f", x / 1000, y / 1000);
            station_names.push_back(name);
        }
    }

    int ns = static_cast<int>(station_xyz.size());

    if (rank == 0) {
        std::cout << "\n  Layered earth: 10/100/0.1 Ωm at 0/−3/−10 km"
                  << "\n  Block A (conductive 1 Ωm): y∈[2,8] km, z∈[−3,−5] km"
                  << "\n  Block B (resistive 100 Ωm): y∈[−8,−2] km, z∈[−3,−5] km"
                  << "\n  Stations: " << ns << " (areal grid, 2 km spacing)"
                  << "\n  Frequencies: " << nf << std::endl;
    }

    // ---------------------------------------------------------------
    // 3. Build octree mesh
    // ---------------------------------------------------------------
    octree::RefinementParams mesh_params;
    mesh_params.domain_x_min = -200000;  mesh_params.domain_x_max = 200000;
    mesh_params.domain_y_min = -200000;  mesh_params.domain_y_max = 200000;
    mesh_params.domain_z_min = -200000;  mesh_params.domain_z_max = 56000;

    mesh_params.min_level = 5;
    mesh_params.max_level = 7;
    mesh_params.station_refine_radius = 25000;
    mesh_params.station_refine_level = 7;
    mesh_params.sigma_bg = sigma_L1;
    mesh_params.replicate_mesh = true;

    // Refine around both blocks
    octree::RefineRegion regA;
    regA.x_min = Ax0; regA.x_max = Ax1;
    regA.y_min = Ay0; regA.y_max = Ay1;
    regA.z_min = Az0; regA.z_max = Az1;
    regA.level = 7; regA.padding = 3000;
    mesh_params.refine_regions.push_back(regA);

    octree::RefineRegion regB;
    regB.x_min = Bx0; regB.x_max = Bx1;
    regB.y_min = By0; regB.y_max = By1;
    regB.z_min = Bz0; regB.z_max = Bz1;
    regB.level = 7; regB.padding = 3000;
    mesh_params.refine_regions.push_back(regB);

    octree::OctreeMesh mesh;
    mesh.setup(mesh_params, station_xyz, {frequencies.front()}, MPI_COMM_WORLD);
    mesh.set_terrain([](Real, Real) { return 0.0; });
    mesh.build_staggered_grid();

    int nc = mesh.staggered().num_cells();
    int ne = mesh.staggered().num_edges();

    if (rank == 0)
        std::cout << "\n  Mesh: " << nc << " cells, " << ne << " edge DOFs" << std::endl;

    // ---------------------------------------------------------------
    // 4. True conductivity model (layered + blocks)
    // ---------------------------------------------------------------
    model::ConductivityModel true_model;
    true_model.init_3d(nc, sigma_L1);

    auto& true_ls = true_model.params();
    int nA = 0, nB = 0;
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH) {
            true_ls[c] = std::log(1e-8);
            continue;
        }
        Real cx, cy, cz;
        mesh.cell_center(c, cx, cy, cz);

        // Layered background
        Real sigma_layer = sigma_L1;
        if (cz < z_L2_bot) sigma_layer = sigma_L3;
        else if (cz < z_L1_bot) sigma_layer = sigma_L2;
        true_ls[c] = std::log(sigma_layer);

        // Block A override
        if (cx >= Ax0 && cx <= Ax1 && cy >= Ay0 && cy <= Ay1 &&
            cz >= Az0 && cz <= Az1) {
            true_ls[c] = std::log(sigma_A);
            ++nA;
        }
        // Block B override
        if (cx >= Bx0 && cx <= Bx1 && cy >= By0 && cy <= By1 &&
            cz >= Bz0 && cz <= Bz1) {
            true_ls[c] = std::log(sigma_B);
            ++nB;
        }
    }
    true_model.invalidate_cache();

    if (rank == 0) {
        std::cout << "  Block A cells: " << nA << ", Block B cells: " << nB << std::endl;
        io::export_octree_vtu(mesh, true_model, output_dir + "/model_true.vtu");
    }

    // ---------------------------------------------------------------
    // 5. Forward solve → synthetic data
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

    // 1D layered Z for initial observed (placeholder)
    for (int f = 0; f < nf; ++f) {
        Real omega = constants::TWOPI * frequencies[f];
        Complex iwmu(0, omega * constants::MU0);
        Complex Z_1d = std::sqrt(iwmu / Complex(sigma_L1, 0));
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

    if (rank == 0) std::cout << "\n--- Forward solve (true model) ---" << std::endl;

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
    // 6. Add 5% noise → observed
    // ---------------------------------------------------------------
    std::mt19937 rng(123);
    std::normal_distribution<Real> ndist(0.0, 1.0);

    for (int f = 0; f < nf; ++f) {
        for (int s = 0; s < ns; ++s) {
            const auto& pred = synthetic.predicted(s, f);
            data::MTResponse obs;
            auto add_n = [&](const data::Datum& p) -> data::Datum {
                data::Datum d;
                Real err = 0.05 * std::abs(p.value);
                if (err < 1e-20) err = 1e-20;
                d.value = p.value + Complex(ndist(rng) * err, ndist(rng) * err);
                d.error = err;
                d.weight = 1.0;
                return d;
            };
            obs.Zxy = add_n(pred.Zxy);
            obs.Zyx = add_n(pred.Zyx);
            observed.set_observed(s, f, obs);
        }
    }

    if (rank == 0) {
        io::export_stations_csv(observed, output_dir + "/stations.csv");
    }

    // ---------------------------------------------------------------
    // 7. Starting model = uniform halfspace (10 Ωm = 0.1 S/m)
    // ---------------------------------------------------------------
    const Real sigma_uniform = 0.1;  // 10 Ωm halfspace
    model::ConductivityModel inv_model;
    inv_model.init_3d(nc, sigma_uniform);

    auto& inv_ls = inv_model.params();
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH) {
            inv_ls[c] = std::log(1e-8);
            continue;
        }
        inv_ls[c] = std::log(sigma_uniform);
    }
    inv_model.invalidate_cache();

    // ---------------------------------------------------------------
    // 8. Inversion forward solver
    // ---------------------------------------------------------------
    forward::ForwardParamsFV inv_fwd_params;
    inv_fwd_params.bicgstab_tol = 1e-7;
    inv_fwd_params.bicgstab_maxiter = 5000;
    inv_fwd_params.print_level = 0;
    inv_fwd_params.div_correction = true;
    inv_fwd_params.scattered_field = false;
    inv_fwd_params.air_z_threshold = 1.0;
    inv_fwd_params.air_bc_iterations = 1;

    forward::ForwardSolverFV inv_fwd;
    inv_fwd.setup(mesh, inv_fwd_params);

    // ---------------------------------------------------------------
    // 9. Regularisation
    // ---------------------------------------------------------------
    inversion::RegParamsOctree reg_params;
    reg_params.alpha_s = 1e-4;
    reg_params.alpha_x = 1.0;
    reg_params.alpha_y = 1.0;
    reg_params.alpha_z = 0.5;

    inversion::RegularizationOctree reg;
    reg.setup(mesh, reg_params);
    reg.set_reference_model(inv_ls);  // reference = uniform 10 Ωm

    // ---------------------------------------------------------------
    // 10. Inversion (NLCG, 50 iterations, uniform start)
    // ---------------------------------------------------------------
    const int max_iter = 50;

    inversion::InversionParamsFV inv_params;
    inv_params.solver = inversion::InversionParamsFV::Solver::NLCG;
    inv_params.max_iterations = max_iter;
    inv_params.target_rms = 1.0;
    inv_params.lambda_init = 10.0;
    inv_params.lambda_decrease = 0.7;
    inv_params.linesearch_max = 6;
    inv_params.linesearch_startdm = 20.0;
    inv_params.save_checkpoints = false;

    inversion::InversionFV inv;
    inv.setup(inv_model, observed, inv_fwd, reg, inv_params);

    std::ofstream fconv;
    if (rank == 0) {
        fconv.open(output_dir + "/convergence.csv");
        fconv << "iteration,objective,data_misfit,model_norm,rms,lambda,step_length,cg_iters,time_s\n";
    }

    auto t_inv_start = Clock::now();
    auto t_last = t_inv_start;

    if (rank == 0)
        std::cout << "\n--- Inversion (NLCG, max " << max_iter << " iter) ---\n" << std::endl;

    inv.set_iteration_callback(
        [&](int iter, const inversion::IterationLogFV& entry) {
            auto t_now = Clock::now();
            Real dt = std::chrono::duration<Real>(t_now - t_last).count();
            Real elapsed = std::chrono::duration<Real>(t_now - t_inv_start).count();
            t_last = t_now;

            if (rank == 0) {
                std::cout << "  Iter " << std::setw(2) << entry.iteration
                          << "  Φ=" << std::scientific << std::setprecision(4) << entry.objective
                          << "  RMS=" << std::fixed << std::setprecision(3) << entry.rms
                          << "  λ=" << std::scientific << std::setprecision(2) << entry.lambda
                          << "  (" << std::fixed << std::setprecision(1) << dt << "s)"
                          << std::endl;

                fconv << entry.iteration << ","
                      << std::scientific << std::setprecision(6)
                      << entry.objective << "," << entry.data_misfit << ","
                      << entry.model_norm << ","
                      << std::fixed << std::setprecision(4) << entry.rms << ","
                      << std::scientific << entry.lambda << ","
                      << std::fixed << std::setprecision(6) << entry.step_length << ","
                      << entry.cg_iterations << ","
                      << std::fixed << std::setprecision(1) << elapsed << "\n";
                fconv.flush();

                io::export_octree_vtu(mesh, inv_model,
                                      output_dir + "/model_iter" +
                                      std::to_string(iter) + ".vtu", iter);
                io::export_data_fit_csv(observed,
                                        output_dir + "/data_fit_iter" +
                                        std::to_string(iter) + ".csv", iter);
            }
        });

    inv.run();

    auto t_inv_end = Clock::now();
    Real inv_time = std::chrono::duration<Real>(t_inv_end - t_inv_start).count();
    if (rank == 0) fconv.close();

    // ---------------------------------------------------------------
    // 11. Export profiles + validation
    // ---------------------------------------------------------------
    if (rank == 0) {
        io::export_data_fit_csv(observed, output_dir + "/data_fit_final.csv");

        // XZ profile (y=0) and YZ profile (x=0)
        auto write_profile = [&](const char* fname, bool xz,
                                 const model::ConductivityModel& mdl) {
            std::ofstream fp(std::string(output_dir) + "/" + fname);
            fp << (xz ? "x_m" : "y_m") << ",z_m,log10_rho\n";
            for (int c = 0; c < nc; ++c) {
                if (mesh.cell_type(c) != octree::CellType::EARTH) continue;
                Real cx, cy, cz;
                mesh.cell_center(c, cx, cy, cz);
                Real perp = xz ? std::abs(cy) : std::abs(cx);
                if (perp > 2000) continue;
                Real sigma = mdl.sigma(c);
                Real coord = xz ? cx : cy;
                fp << coord << "," << cz << "," << std::log10(1.0 / sigma) << "\n";
            }
        };

        write_profile("profile_xz.csv", true, inv_model);
        write_profile("profile_yz.csv", false, inv_model);
        write_profile("profile_xz_true.csv", true, true_model);
        write_profile("profile_yz_true.csv", false, true_model);

        const auto& history = inv.history();
        Real rms_init  = history.empty() ? 99 : history.front().rms;
        Real rms_final = history.empty() ? 99 : history.back().rms;

        std::cout << "\n  RMS: " << std::fixed << std::setprecision(3)
                  << rms_init << " → " << rms_final
                  << "\n  Inversion time: " << std::setprecision(1) << inv_time << " s"
                  << "\n  Output: " << output_dir << "/"
                  << "\n=== COMMEMI 3D-2 Inversion Complete ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
