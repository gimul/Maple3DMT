// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file test_halfspace.cpp
/// @brief Quick diagnostic: uniform halfspace should give rho_a = 100 Ωm, phase = 45°.

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/octree/staggered_grid.h"
#include "maple3dmt/octree/operators.h"
#include "maple3dmt/forward/forward_solver_fv.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/utils/logger.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <mpi.h>

using namespace maple3dmt;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    utils::Logger::instance().set_rank(rank);

    std::cout << "\n=== Uniform Halfspace Test ===" << std::endl;
    std::cout << "  Expected: rho_a = 100 Ωm, phase = 45° at all stations/frequencies\n" << std::endl;

    const Real sigma_bg = 0.01;  // 100 Ωm

    std::vector<Real> frequencies = {0.01, 0.1, 1.0};

    // A few test stations
    std::vector<std::array<Real,3>> station_xyz;
    std::vector<std::string> station_names;
    for (Real x : {0.0, 5000.0, 10000.0, 20000.0}) {
        station_xyz.push_back({x, 0.0, 0.0});
        char name[32]; snprintf(name, sizeof(name), "X%.0f", x/1000);
        station_names.push_back(name);
    }

    // Mesh
    octree::RefinementParams mesh_params;
    mesh_params.domain_x_min = -200000;  mesh_params.domain_x_max = 200000;
    mesh_params.domain_y_min = -200000;  mesh_params.domain_y_max = 200000;
    mesh_params.domain_z_min = -200000;  mesh_params.domain_z_max = 56000;
    mesh_params.min_level = 5;
    mesh_params.max_level = 7;
    mesh_params.station_refine_radius = 25000;
    mesh_params.station_refine_level = 7;
    mesh_params.sigma_bg = sigma_bg;

    octree::OctreeMesh mesh;
    mesh.setup(mesh_params, station_xyz, {frequencies.front()}, MPI_COMM_WORLD);
    mesh.set_terrain([](Real, Real) { return 0.0; });
    mesh.build_staggered_grid();

    int nc = mesh.staggered().num_cells();
    int ne = mesh.staggered().num_edges();
    std::cout << "  Mesh: " << nc << " cells, " << ne << " edge DOFs" << std::endl;

    // Uniform conductivity (no block)
    model::ConductivityModel model;
    model.init_3d(nc, sigma_bg);
    auto& log_sigma = model.params();
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH)
            log_sigma[c] = std::log(1e-8);  // air
    }
    model.invalidate_cache();

    // Forward solver
    forward::ForwardParamsFV fwd_params;
    fwd_params.bicgstab_tol = 1e-8;
    fwd_params.bicgstab_maxiter = 10000;
    fwd_params.print_level = 1;
    fwd_params.div_correction = true;
    fwd_params.scattered_field = false;

    forward::ForwardSolverFV fwd;
    fwd.setup(mesh, fwd_params);
    fwd.update_sigma(model);

    // Data
    int ns = static_cast<int>(station_xyz.size());
    int nf = static_cast<int>(frequencies.size());
    data::MTData observed, predicted;
    for (int s = 0; s < ns; ++s) {
        data::Station st;
        st.name = station_names[s];
        st.x = station_xyz[s][0];
        st.y = station_xyz[s][1];
        st.z = station_xyz[s][2];
        observed.add_station(st);
        predicted.add_station(st);
    }
    observed.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));
    predicted.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));

    // Dummy observed
    for (int f = 0; f < nf; ++f)
        for (int s = 0; s < ns; ++s) {
            data::MTResponse resp;
            observed.set_observed(s, f, resp);
        }

    // Solve
    fwd.compute_responses(observed, predicted);

    // Print results
    std::cout << "\n  " << std::setw(8) << "station"
              << "  " << std::setw(8) << "freq"
              << "  " << std::setw(10) << "rho_xy"
              << "  " << std::setw(10) << "phi_xy"
              << "  " << std::setw(10) << "rho_yx"
              << "  " << std::setw(10) << "phi_yx"
              << "  " << std::setw(10) << "err_xy%"
              << std::endl;
    std::cout << "  " << std::string(72, '-') << std::endl;

    for (int f = 0; f < nf; ++f) {
        Real freq = frequencies[f];
        Real omega = constants::TWOPI * freq;
        for (int s = 0; s < ns; ++s) {
            const auto& p = predicted.predicted(s, f);
            Complex Zxy = p.Zxy.value, Zyx = p.Zyx.value;
            Real rho_xy = std::norm(Zxy) / (omega * constants::MU0);
            Real phi_xy = std::arg(Zxy) * 180.0 / constants::PI;
            Real rho_yx = std::norm(Zyx) / (omega * constants::MU0);
            Real phi_yx = std::arg(Zyx) * 180.0 / constants::PI;
            Real err_pct = (rho_xy - 100.0) / 100.0 * 100;

            std::cout << "  " << std::setw(8) << station_names[s]
                      << "  " << std::scientific << std::setprecision(1)
                      << std::setw(8) << freq << " Hz"
                      << "  " << std::fixed << std::setprecision(2)
                      << std::setw(10) << rho_xy
                      << "  " << std::setw(10) << phi_xy
                      << "  " << std::setw(10) << rho_yx
                      << "  " << std::setw(10) << phi_yx
                      << "  " << std::setw(10) << err_pct
                      << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "=== Halfspace Test Complete ===" << std::endl;
    MPI_Finalize();
    return 0;
}
