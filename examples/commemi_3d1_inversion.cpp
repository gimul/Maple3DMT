// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file commemi_3d1_inversion.cpp
/// @brief COMMEMI 3D-1A synthetic inversion benchmark for publication.
///
/// Model: 10 Ωm block (σ=0.1) in 100 Ωm halfspace (σ=0.01).
///   Block: |x|,|y| ≤ 5 km, z ∈ [0, −5 km].
///
/// Dense station coverage (X-profile + Y-profile + grid) and 5 frequencies
/// for publication-quality convergence, cross-section, and data-fitting figures.
///
/// Output (per iteration):
///   - model_iterN.vtu:       3D octree model
///   - data_fit_iterN.csv:    obs vs pred (rho_a, phase, misfit)
///   - convergence.csv:       RMS, Φ, λ, step
///   - depth slices at 1, 2, 3, 5, 8 km

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
#include <algorithm>
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

    const std::string output_dir = "output_commemi_3d1";
    if (rank == 0) mkdir_p(output_dir);

    if (rank == 0) {
        std::cout << "\n======================================================"
                  << "\n  COMMEMI 3D-1A Synthetic Inversion (Octree FV)"
                  << "\n  Publication benchmark — dense stations, 5 frequencies"
                  << "\n======================================================"
                  << std::endl;
    }

    auto t_start = Clock::now();

    // ---------------------------------------------------------------
    // 1. True model parameters
    // ---------------------------------------------------------------
    const Real sigma_bg    = 0.01;   // 100 Ωm halfspace
    const Real sigma_block = 0.1;    // 10 Ωm block
    const Real block_xmin  = -5000,  block_xmax = 5000;
    const Real block_ymin  = -5000,  block_ymax = 5000;
    const Real block_zmin  = -5000,  block_zmax = 0;    // surface to 5 km depth

    // 5 frequencies: 3 decades
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
        std::cout << "\n  True model: 10 Ωm block in 100 Ωm halfspace"
                  << "\n  Block: [−5,5]×[−5,5]×[0,−5] km"
                  << "\n  Stations: " << ns << " (areal grid, 2 km spacing)"
                  << "\n  Frequencies: " << nf << " (";
        for (int f = 0; f < nf; ++f) {
            if (f > 0) std::cout << ", ";
            std::cout << frequencies[f] << " Hz";
        }
        std::cout << ")\n  MPI ranks: " << nprocs << std::endl;
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
    mesh_params.sigma_bg = sigma_bg;
    mesh_params.replicate_mesh = true;

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
        std::cout << "\n  Mesh: " << nc << " cells, " << ne << " edge DOFs"
                  << "\n  Levels: " << mesh_params.min_level << "–"
                  << mesh_params.max_level << std::endl;
    }

    // ---------------------------------------------------------------
    // 4. True conductivity model
    // ---------------------------------------------------------------
    model::ConductivityModel true_model;
    true_model.init_3d(nc, sigma_bg);

    auto& true_log_sigma = true_model.params();
    int n_block_cells = 0;
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH) {
            true_log_sigma[c] = std::log(1e-8);
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

    // Export true model VTU
    if (rank == 0) {
        io::export_octree_vtu(mesh, true_model,
                              output_dir + "/model_true.vtu");
    }

    // ---------------------------------------------------------------
    // 5. Forward solve → synthetic observed data
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

    // Initialize observed data container with 1D values (placeholder)
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
    // 6. Add 5% Gaussian noise → observed data
    // ---------------------------------------------------------------
    if (rank == 0)
        std::cout << "\n--- Synthetic data generation (5% noise) ---" << std::endl;

    std::mt19937 rng(42);
    std::normal_distribution<Real> noise_dist(0.0, 1.0);
    Real noise_pct = 0.05;

    for (int f = 0; f < nf; ++f) {
        for (int s = 0; s < ns; ++s) {
            const auto& pred = synthetic.predicted(s, f);
            data::MTResponse obs_resp;

            auto add_noise = [&](const data::Datum& p) -> data::Datum {
                data::Datum d;
                Real err = noise_pct * std::abs(p.value);
                if (err < 1e-20) err = 1e-20;
                d.value = p.value + Complex(noise_dist(rng) * err,
                                            noise_dist(rng) * err);
                d.error = err;
                d.weight = 1.0;
                return d;
            };

            obs_resp.Zxy = add_noise(pred.Zxy);
            obs_resp.Zyx = add_noise(pred.Zyx);
            observed.set_observed(s, f, obs_resp);
        }
    }

    // Export clean data and noisy data for plotting
    if (rank == 0) {
        io::export_data_fit_csv(synthetic, output_dir + "/data_true.csv");
        io::export_stations_csv(observed, output_dir + "/stations.csv");

        // Export observed (noisy) data as separate CSV
        std::ofstream fobs(output_dir + "/data_observed.csv");
        fobs << "station,freq_hz,component,obs_re,obs_im,error,app_res_obs,phase_obs_deg\n";
        for (int f = 0; f < nf; ++f) {
            Real omega = constants::TWOPI * frequencies[f];
            for (int s = 0; s < ns; ++s) {
                const auto& o = observed.observed(s, f);
                for (const char* comp : {"Zxy", "Zyx"}) {
                    const data::Datum& d = (std::string(comp) == "Zxy") ? o.Zxy : o.Zyx;
                    if (d.weight <= 0) continue;
                    Real rho = std::norm(d.value) / (omega * constants::MU0);
                    Real phi = std::arg(d.value) * 180.0 / constants::PI;
                    fobs << station_names[s] << "," << frequencies[f] << ","
                         << comp << ","
                         << d.value.real() << "," << d.value.imag() << ","
                         << d.error << "," << rho << "," << phi << "\n";
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // 7. Starting model (uniform halfspace)
    // ---------------------------------------------------------------
    model::ConductivityModel inv_model;
    inv_model.init_3d(nc, sigma_bg);

    auto& inv_log_sigma = inv_model.params();
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH)
            inv_log_sigma[c] = std::log(1e-8);
    }
    inv_model.invalidate_cache();

    if (rank == 0) {
        io::export_octree_vtu(mesh, inv_model,
                              output_dir + "/model_iter0.vtu", 0);
    }

    // ---------------------------------------------------------------
    // 8. Forward solver for inversion
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
    reg.set_reference_model(inv_log_sigma);

    if (rank == 0)
        std::cout << "\n  Active parameters: " << reg.n_active() << std::endl;

    // ---------------------------------------------------------------
    // 10. Inversion (NLCG, 30 iterations)
    // ---------------------------------------------------------------
    const int max_iter = 30;

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

    // Convergence CSV
    std::ofstream fconv;
    if (rank == 0) {
        fconv.open(output_dir + "/convergence.csv");
        fconv << "iteration,objective,data_misfit,model_norm,rms,lambda,step_length,cg_iters,time_s\n";
    }

    // Export params
    io::OctreeExportParams export_params;
    export_params.export_depth_slices = true;
    export_params.slice_depths = {1000, 2000, 3000, 5000, 8000};
    export_params.slice_dx = 2000;  // coarser for speed during inversion
    export_params.slice_dy = 2000;
    export_params.export_stations_csv = false;  // already saved

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
                // Console
                std::cout << "  Iter " << std::setw(2) << entry.iteration
                          << "  Φ=" << std::scientific << std::setprecision(4)
                          << entry.objective
                          << "  RMS=" << std::fixed << std::setprecision(3) << entry.rms
                          << "  λ=" << std::scientific << std::setprecision(2) << entry.lambda
                          << "  step=" << std::fixed << std::setprecision(4) << entry.step_length
                          << "  (" << std::setprecision(1) << dt << "s)"
                          << std::endl;

                // Convergence CSV
                fconv << entry.iteration << ","
                      << std::scientific << std::setprecision(6) << entry.objective << ","
                      << entry.data_misfit << ","
                      << entry.model_norm << ","
                      << std::fixed << std::setprecision(4) << entry.rms << ","
                      << std::scientific << entry.lambda << ","
                      << std::fixed << std::setprecision(6) << entry.step_length << ","
                      << entry.cg_iterations << ","
                      << std::fixed << std::setprecision(1) << elapsed << "\n";
                fconv.flush();

                // Per-iteration VTU
                io::export_octree_vtu(mesh, inv_model,
                                      output_dir + "/model_iter" +
                                      std::to_string(iter) + ".vtu", iter);

                // Per-iteration data fit CSV
                io::export_data_fit_csv(observed,
                                        output_dir + "/data_fit_iter" +
                                        std::to_string(iter) + ".csv", iter);

                // Depth slices only at final iteration (slow export)
                // Intermediate slices skipped for speed
            }
        });

    inv.run();

    auto t_inv_end = Clock::now();
    Real inv_time = std::chrono::duration<Real>(t_inv_end - t_inv_start).count();

    if (rank == 0) fconv.close();

    // ---------------------------------------------------------------
    // 11. Final export + validation
    // ---------------------------------------------------------------
    if (rank == 0) {
        // Final data fit
        io::export_data_fit_csv(observed,
                                output_dir + "/data_fit_final.csv");

        // Final model VTU with all slices
        io::export_octree_all(mesh, inv_model, observed, output_dir,
                              export_params, -1);

        std::cout << "\n======================================================"
                  << "\n  Validation" << std::endl;
    }

    // Check conductivity recovery
    Real sigma_block_avg = 0, sigma_bg_avg = 0;
    int n_blk = 0, n_bg = 0;

    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH) continue;
        Real cx, cy, cz;
        mesh.cell_center(c, cx, cy, cz);
        Real sigma_inv = inv_model.sigma(c);

        if (std::abs(cx) < 3000 && std::abs(cy) < 3000 &&
            cz > block_zmin + 1000 && cz < block_zmax - 500) {
            sigma_block_avg += sigma_inv;
            ++n_blk;
        } else if (std::abs(cx) > 15000 && cz < -1000 && cz > -10000) {
            sigma_bg_avg += sigma_inv;
            ++n_bg;
        }
    }

    if (n_blk > 0) sigma_block_avg /= n_blk;
    if (n_bg > 0) sigma_bg_avg /= n_bg;

    const auto& history = inv.history();

    if (rank == 0) {
        Real rho_blk = (sigma_block_avg > 0) ? 1.0 / sigma_block_avg : 0;
        Real rho_bg  = (sigma_bg_avg > 0)    ? 1.0 / sigma_bg_avg : 0;

        std::cout << "\n  Block core: σ=" << std::scientific << std::setprecision(4)
                  << sigma_block_avg << " S/m (ρ=" << std::fixed << std::setprecision(1)
                  << rho_blk << " Ωm) [true: 10 Ωm]"
                  << "\n  Background: σ=" << std::scientific << std::setprecision(4)
                  << sigma_bg_avg << " S/m (ρ=" << std::fixed << std::setprecision(1)
                  << rho_bg << " Ωm) [true: 100 Ωm]" << std::endl;

        Real rms_init  = history.empty() ? 99 : history.front().rms;
        Real rms_final = history.empty() ? 99 : history.back().rms;
        std::cout << "\n  RMS: " << std::fixed << std::setprecision(3)
                  << rms_init << " → " << rms_final
                  << "\n  Inversion time: " << std::setprecision(1) << inv_time << " s"
                  << "\n  Total time: "
                  << std::chrono::duration<Real>(Clock::now() - t_start).count() << " s"
                  << std::endl;

        // Export XZ and YZ profile sections for figure script
        {
            std::ofstream fp(output_dir + "/profile_xz.csv");
            fp << "x_m,z_m,log10_rho\n";
            for (int c = 0; c < nc; ++c) {
                if (mesh.cell_type(c) != octree::CellType::EARTH) continue;
                Real cx, cy, cz;
                mesh.cell_center(c, cx, cy, cz);
                if (std::abs(cy) > 2000) continue;  // near y=0
                Real sigma = inv_model.sigma(c);
                fp << cx << "," << cz << "," << std::log10(1.0 / sigma) << "\n";
            }
        }
        {
            std::ofstream fp(output_dir + "/profile_yz.csv");
            fp << "y_m,z_m,log10_rho\n";
            for (int c = 0; c < nc; ++c) {
                if (mesh.cell_type(c) != octree::CellType::EARTH) continue;
                Real cx, cy, cz;
                mesh.cell_center(c, cx, cy, cz);
                if (std::abs(cx) > 2000) continue;  // near x=0
                Real sigma = inv_model.sigma(c);
                fp << cy << "," << cz << "," << std::log10(1.0 / sigma) << "\n";
            }
        }
        // True model profiles
        {
            std::ofstream fp(output_dir + "/profile_xz_true.csv");
            fp << "x_m,z_m,log10_rho\n";
            for (int c = 0; c < nc; ++c) {
                if (mesh.cell_type(c) != octree::CellType::EARTH) continue;
                Real cx, cy, cz;
                mesh.cell_center(c, cx, cy, cz);
                if (std::abs(cy) > 2000) continue;
                Real sigma = true_model.sigma(c);
                fp << cx << "," << cz << "," << std::log10(1.0 / sigma) << "\n";
            }
        }
        {
            std::ofstream fp(output_dir + "/profile_yz_true.csv");
            fp << "y_m,z_m,log10_rho\n";
            for (int c = 0; c < nc; ++c) {
                if (mesh.cell_type(c) != octree::CellType::EARTH) continue;
                Real cx, cy, cz;
                mesh.cell_center(c, cx, cy, cz);
                if (std::abs(cx) > 2000) continue;
                Real sigma = true_model.sigma(c);
                fp << cy << "," << cz << "," << std::log10(1.0 / sigma) << "\n";
            }
        }

        std::cout << "\n  Output directory: " << output_dir << "/"
                  << "\n=== COMMEMI 3D-1A Inversion Complete ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
