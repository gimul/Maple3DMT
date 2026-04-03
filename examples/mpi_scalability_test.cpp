// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file mpi_scalability_test.cpp
/// @brief MPI strong/weak scaling benchmark for Octree FV forward solver.
///
/// Measures wall time for forward solve (2 polarizations × N frequencies)
/// with varying MPI ranks. Output is a CSV for scalability plots.
///
/// Usage:
///   mpiexec -np 1 mpi_scalability_test [--nfreq 10] [--level 7]
///   mpiexec -np 2 mpi_scalability_test
///   mpiexec -np 4 mpi_scalability_test
///   ...
///
/// Designed for frequency-parallel mode (replicate_mesh=true).
/// Each rank solves a subset of frequencies on the full mesh.

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/octree/staggered_grid.h"
#include "maple3dmt/octree/operators.h"
#include "maple3dmt/forward/forward_solver_fv.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/utils/logger.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <mpi.h>

using namespace maple3dmt;
using Clock = std::chrono::steady_clock;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    utils::Logger::instance().set_rank(rank);

    // Parse args
    int nfreq = 10;
    int max_level = 7;
    bool include_inversion_step = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--nfreq") == 0 && i + 1 < argc)
            nfreq = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--level") == 0 && i + 1 < argc)
            max_level = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--with-adjoint") == 0)
            include_inversion_step = true;
    }

    if (rank == 0) {
        std::cout << "\n======================================================"
                  << "\n  MPI Scalability Test — Octree FV Forward Solver"
                  << "\n======================================================"
                  << "\n  MPI ranks: " << nprocs
                  << "\n  Frequencies: " << nfreq
                  << "\n  Max level: " << max_level
                  << "\n  Include adjoint: " << (include_inversion_step ? "yes" : "no")
                  << std::endl;
    }

    // ---------------------------------------------------------------
    // 1. COMMEMI 3D-1A model
    // ---------------------------------------------------------------
    const Real sigma_bg = 0.01, sigma_block = 0.1;

    // Generate frequencies logarithmically
    std::vector<Real> frequencies(nfreq);
    for (int i = 0; i < nfreq; ++i) {
        Real log_f = std::log10(0.001) +
            (std::log10(10.0) - std::log10(0.001)) * i / (nfreq - 1);
        frequencies[i] = std::pow(10.0, log_f);
    }

    // 21 stations along X-profile
    std::vector<std::array<Real,3>> station_xyz;
    std::vector<std::string> station_names;
    for (Real x = -20000; x <= 20001; x += 2000) {
        station_xyz.push_back({x, 0.0, 0.0});
        char name[32]; snprintf(name, sizeof(name), "X%+.0f", x / 1000);
        station_names.push_back(name);
    }
    int ns = static_cast<int>(station_xyz.size());

    // ---------------------------------------------------------------
    // 2. Build mesh
    // ---------------------------------------------------------------
    auto t_mesh_start = Clock::now();

    octree::RefinementParams mesh_params;
    mesh_params.domain_x_min = -200000;  mesh_params.domain_x_max = 200000;
    mesh_params.domain_y_min = -200000;  mesh_params.domain_y_max = 200000;
    mesh_params.domain_z_min = -200000;  mesh_params.domain_z_max = 56000;
    mesh_params.min_level = 5;
    mesh_params.max_level = max_level;
    mesh_params.station_refine_radius = 25000;
    mesh_params.station_refine_level = std::min(max_level, 7);
    mesh_params.sigma_bg = sigma_bg;
    mesh_params.replicate_mesh = true;

    octree::RefineRegion block_region;
    block_region.x_min = -5000;  block_region.x_max = 5000;
    block_region.y_min = -5000;  block_region.y_max = 5000;
    block_region.z_min = -5000;  block_region.z_max = 0;
    block_region.level = max_level;
    block_region.padding = 5000;
    mesh_params.refine_regions.push_back(block_region);

    octree::OctreeMesh mesh;
    mesh.setup(mesh_params, station_xyz, {frequencies.front()}, MPI_COMM_WORLD);
    mesh.set_terrain([](Real, Real) { return 0.0; });
    mesh.build_staggered_grid();

    Real t_mesh = std::chrono::duration<Real>(Clock::now() - t_mesh_start).count();

    int nc = mesh.staggered().num_cells();
    int ne = mesh.staggered().num_edges();

    if (rank == 0) {
        std::cout << "\n  Mesh: " << nc << " cells, " << ne << " edges"
                  << " (build: " << std::fixed << std::setprecision(1) << t_mesh << "s)"
                  << std::endl;
    }

    // ---------------------------------------------------------------
    // 3. Setup model
    // ---------------------------------------------------------------
    model::ConductivityModel model;
    model.init_3d(nc, sigma_bg);
    auto& log_sigma = model.params();
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH) {
            log_sigma[c] = std::log(1e-8);
            continue;
        }
        Real cx, cy, cz;
        mesh.cell_center(c, cx, cy, cz);
        if (std::abs(cx) <= 5000 && std::abs(cy) <= 5000 &&
            cz >= -5000 && cz <= 0) {
            log_sigma[c] = std::log(sigma_block);
        }
    }
    model.invalidate_cache();

    // ---------------------------------------------------------------
    // 4. Setup forward solver
    // ---------------------------------------------------------------
    forward::ForwardParamsFV fwd_params;
    fwd_params.bicgstab_tol = 1e-8;
    fwd_params.bicgstab_maxiter = 10000;
    fwd_params.print_level = (rank == 0) ? 1 : 0;
    fwd_params.div_correction = true;
    fwd_params.scattered_field = false;
    fwd_params.air_z_threshold = 1.0;
    fwd_params.air_bc_iterations = 1;

    forward::ForwardSolverFV fwd;
    fwd.setup(mesh, fwd_params);
    fwd.update_sigma(model);

    // MT data
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
    for (int f = 0; f < nfreq; ++f) {
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

    data::MTData predicted;
    for (int s = 0; s < ns; ++s) {
        data::Station st;
        st.name = station_names[s]; st.x = station_xyz[s][0];
        st.y = station_xyz[s][1]; st.z = station_xyz[s][2];
        predicted.add_station(st);
    }
    predicted.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));

    // ---------------------------------------------------------------
    // 5. Warmup solve (1 frequency)
    // ---------------------------------------------------------------
    if (rank == 0) std::cout << "\n--- Warmup ---" << std::endl;
    {
        data::MTData warmup_obs, warmup_pred;
        for (int s = 0; s < ns; ++s) {
            data::Station st; st.name = station_names[s];
            st.x = station_xyz[s][0]; st.y = station_xyz[s][1]; st.z = 0;
            warmup_obs.add_station(st);
            warmup_pred.add_station(st);
        }
        warmup_obs.set_frequencies({frequencies[nfreq / 2]});
        warmup_pred.set_frequencies({frequencies[nfreq / 2]});
        Real omega = constants::TWOPI * frequencies[nfreq / 2];
        Complex iwmu(0, omega * constants::MU0);
        Complex Z_1d = std::sqrt(iwmu / Complex(sigma_bg, 0));
        for (int s = 0; s < ns; ++s) {
            data::MTResponse r;
            r.Zxy.value = Z_1d; r.Zxy.error = 0.05 * std::abs(Z_1d); r.Zxy.weight = 1.0;
            r.Zyx.value = -Z_1d; r.Zyx.error = 0.05 * std::abs(Z_1d); r.Zyx.weight = 1.0;
            warmup_obs.set_observed(s, 0, r);
        }
        fwd.compute_responses(warmup_obs, warmup_pred);
    }

    // ---------------------------------------------------------------
    // 6. Timed forward solve (all frequencies)
    // ---------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) std::cout << "\n--- Forward solve (" << nfreq << " frequencies) ---" << std::endl;

    auto t_fwd_start = Clock::now();
    fwd.compute_responses(observed, predicted);
    MPI_Barrier(MPI_COMM_WORLD);
    Real t_fwd = std::chrono::duration<Real>(Clock::now() - t_fwd_start).count();

    // ---------------------------------------------------------------
    // 7. (Optional) Timed adjoint solve
    // ---------------------------------------------------------------
    Real t_adj = 0;
    if (include_inversion_step) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "\n--- Adjoint solve ---" << std::endl;

        auto t_adj_start = Clock::now();
        // Adjoint = forward with conjugate RHS, same solver
        // For timing, just re-run forward as proxy (same cost)
        fwd.compute_responses(observed, predicted);
        MPI_Barrier(MPI_COMM_WORLD);
        t_adj = std::chrono::duration<Real>(Clock::now() - t_adj_start).count();
    }

    // ---------------------------------------------------------------
    // 8. Collect per-rank timing
    // ---------------------------------------------------------------
    Real t_fwd_max, t_fwd_min, t_fwd_sum;
    MPI_Reduce(&t_fwd, &t_fwd_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_fwd, &t_fwd_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_fwd, &t_fwd_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // ---------------------------------------------------------------
    // 9. Output results
    // ---------------------------------------------------------------
    if (rank == 0) {
        Real t_ideal = t_fwd_max * nprocs;  // if 1 rank took max time
        Real speedup = t_fwd_sum / t_fwd_max;  // actual parallelism
        Real efficiency = speedup / nprocs * 100.0;

        std::cout << "\n======================================================"
                  << "\n  Results"
                  << "\n======================================================"
                  << "\n  Mesh:        " << nc << " cells, " << ne << " edges"
                  << "\n  Frequencies: " << nfreq
                  << "\n  MPI ranks:   " << nprocs
                  << "\n  Mesh build:  " << std::fixed << std::setprecision(1) << t_mesh << " s"
                  << "\n  Forward:     " << t_fwd_max << " s (wall)"
                  << "\n  Fwd min/max: " << t_fwd_min << " / " << t_fwd_max << " s"
                  << "\n  Speedup:     " << std::setprecision(2) << speedup << "x"
                  << "\n  Efficiency:  " << efficiency << "%"
                  << std::endl;

        if (include_inversion_step) {
            std::cout << "  Adjoint:     " << std::setprecision(1) << t_adj << " s" << std::endl;
        }

        // Append to CSV
        const char* csv_file = "scalability_results.csv";
        bool file_exists = false;
        {
            std::ifstream check(csv_file);
            file_exists = check.good();
        }

        std::ofstream csv(csv_file, std::ios::app);
        if (!file_exists) {
            csv << "nprocs,nfreq,max_level,n_cells,n_edges,t_mesh_s,t_forward_s,"
                << "t_fwd_min_s,t_fwd_max_s,speedup,efficiency_pct\n";
        }
        csv << nprocs << "," << nfreq << "," << max_level << ","
            << nc << "," << ne << ","
            << std::fixed << std::setprecision(2)
            << t_mesh << "," << t_fwd_max << ","
            << t_fwd_min << "," << t_fwd_max << ","
            << speedup << "," << efficiency << "\n";

        std::cout << "\n  Results appended to: " << csv_file << std::endl;
    }

    MPI_Finalize();
    return 0;
}
