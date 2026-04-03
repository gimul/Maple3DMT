// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file generate_data.cpp
/// @brief Generate synthetic EDI files for COMMEMI 3D-2 benchmark.
///
/// Model: 3-layer earth (10/100/0.1 Ohm-m at 0/-3/-10 km) with two blocks:
///   Block A (1 Ohm-m, conductive): x in [-5,5], y in [2,8], z in [-3,-5] km
///   Block B (100 Ohm-m, resistive): x in [-5,5], y in [-8,-2], z in [-3,-5] km
///   121 stations (11x11 areal grid, 3 km spacing), 5 frequencies.
///
/// Usage:
///   mpirun -np 5 generate_commemi_3d2_data [--output-dir edi]
///
/// Output:
///   edi/             121 synthetic EDI files with 5% Gaussian noise
///   true_model.vtu   True model for visualization

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/forward/forward_solver_fv.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/io/edi_io.h"
#include "maple3dmt/io/vtk_export_octree.h"
#include "maple3dmt/utils/logger.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <mpi.h>
#include <sys/stat.h>

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

    // Parse output directory
    std::string output_dir = "edi";
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--output-dir")
            output_dir = argv[i + 1];
    }

    if (rank == 0) {
        mkdir_p(output_dir);
        std::cout << "\n======================================================"
                  << "\n  COMMEMI 3D-2 — Synthetic Data Generation"
                  << "\n======================================================\n"
                  << std::endl;
    }

    auto t_start = Clock::now();

    // -----------------------------------------------------------------
    // 1. True model: 3-layer earth + two blocks
    // -----------------------------------------------------------------
    const Real sigma_L1 = 0.1;    // 10 Ohm-m,  0 to -3 km
    const Real sigma_L2 = 0.01;   // 100 Ohm-m, -3 to -10 km
    const Real sigma_L3 = 10.0;   // 0.1 Ohm-m, below -10 km
    const Real z_L1_bot = -3000;
    const Real z_L2_bot = -10000;

    // Block A (conductive): 1 Ohm-m
    const Real sigma_A = 1.0;
    const Real Ax0 = -5000, Ax1 = 5000;
    const Real Ay0 =  2000, Ay1 = 8000;
    const Real Az0 = -5000, Az1 = -3000;

    // Block B (resistive): 100 Ohm-m
    const Real sigma_B = 0.01;
    const Real Bx0 = -5000, Bx1 = 5000;
    const Real By0 = -8000, By1 = -2000;
    const Real Bz0 = -5000, Bz1 = -3000;

    // 5 frequencies
    std::vector<Real> frequencies = {0.01, 0.032, 0.1, 0.32, 1.0};
    int nf = static_cast<int>(frequencies.size());

    // -----------------------------------------------------------------
    // 2. Areal station grid: 11x11 = 121 stations, 3 km spacing
    // -----------------------------------------------------------------
    std::vector<std::array<Real,3>> station_xyz;
    std::vector<std::string> station_names;

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
        std::cout << "  3-layer earth: 10/100/0.1 Ohm-m at 0/-3/-10 km\n"
                  << "  Block A (1 Ohm-m): y in [2,8] km, z in [-3,-5] km\n"
                  << "  Block B (100 Ohm-m): y in [-8,-2] km, z in [-3,-5] km\n"
                  << "  Stations: " << ns << " (11x11 areal grid, 3 km spacing)\n"
                  << "  Frequencies: " << nf << " (0.01 - 1.0 Hz)\n"
                  << "  MPI ranks: " << nprocs << "\n" << std::endl;
    }

    // -----------------------------------------------------------------
    // 3. Build octree mesh
    // -----------------------------------------------------------------
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
    if (rank == 0)
        std::cout << "  Mesh: " << nc << " cells, "
                  << mesh.staggered().num_edges() << " edge DOFs\n" << std::endl;

    // -----------------------------------------------------------------
    // 4. True conductivity model (layered + blocks)
    // -----------------------------------------------------------------
    model::ConductivityModel true_model;
    true_model.init_3d(nc, sigma_L1);

    auto& true_ls = true_model.params();
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
        }
        // Block B override
        if (cx >= Bx0 && cx <= Bx1 && cy >= By0 && cy <= By1 &&
            cz >= Bz0 && cz <= Bz1) {
            true_ls[c] = std::log(sigma_B);
        }
    }
    true_model.invalidate_cache();

    // Export true model VTU
    if (rank == 0) {
        io::export_octree_vtu(mesh, true_model, "true_model.vtu");
        std::cout << "  Exported: true_model.vtu" << std::endl;
    }

    // -----------------------------------------------------------------
    // 5. Forward solve (true model)
    // -----------------------------------------------------------------
    forward::ForwardParamsFV fwd_params;
    fwd_params.bicgstab_tol = 1e-8;
    fwd_params.bicgstab_maxiter = 10000;
    fwd_params.print_level = 1;
    fwd_params.div_correction = true;
    fwd_params.scattered_field = false;
    fwd_params.air_z_threshold = 1.0;

    forward::ForwardSolverFV fwd;
    fwd.setup(mesh, fwd_params);
    fwd.update_sigma(true_model);

    // Initialize MTData with 1D placeholder
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

    // -----------------------------------------------------------------
    // 6. Add 5% Gaussian noise → observed data
    // -----------------------------------------------------------------
    if (rank == 0)
        std::cout << "\n--- Adding 5% Gaussian noise ---" << std::endl;

    std::mt19937 rng(123);
    std::normal_distribution<Real> noise_dist(0.0, 1.0);

    for (int f = 0; f < nf; ++f) {
        for (int s = 0; s < ns; ++s) {
            const auto& pred = synthetic.predicted(s, f);
            data::MTResponse obs_resp;

            auto add_noise = [&](const data::Datum& p) -> data::Datum {
                data::Datum d;
                Real err = 0.05 * std::abs(p.value);
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

    // -----------------------------------------------------------------
    // 7. Write EDI files
    // -----------------------------------------------------------------
    if (rank == 0) {
        io::save_edi_directory(output_dir, observed);

        auto elapsed = std::chrono::duration<Real>(Clock::now() - t_start).count();
        std::cout << "\n======================================================"
                  << "\n  Done! " << ns << " EDI files written to: " << output_dir << "/"
                  << "\n  Time: " << std::fixed << std::setprecision(1) << elapsed << " s"
                  << "\n======================================================"
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
