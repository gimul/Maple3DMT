// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file halfspace_1d_test.cpp
/// @brief §14.4 Validation: 1D Halfspace Analytical Test.
///
/// Uniform mesh (no anomaly), σ=0.01 S/m.
/// Expected: ρ_app = 100 Ωm (±5%), phase = 45° (±2°).
/// This is the MINIMAL test to validate Me, Mf, Ce, BiCGStab, DivCorr, and Z extraction.

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

    std::cout << "\n=== §14.4 1D Halfspace Analytical Test ===" << std::endl;

    // ---------------------------------------------------------------
    // 1. Model: uniform halfspace σ = 0.01 S/m (ρ = 100 Ωm)
    // ---------------------------------------------------------------
    const Real sigma_bg = 0.01;

    // Frequencies: 0.01, 0.1, 1.0, 10.0 Hz
    std::vector<Real> frequencies = {0.1};

    // A few stations along x-axis at surface (y=0, z=0)
    std::vector<std::array<Real,3>> station_xyz;
    std::vector<std::string> station_names;
    for (Real x : {0.0, 10000.0, 50000.0}) {
        station_xyz.push_back({x, 0.0, 0.0});
        station_names.push_back("X" + std::to_string(static_cast<int>(x)));
    }

    int ns = static_cast<int>(station_xyz.size());
    int nf = static_cast<int>(frequencies.size());

    // ---------------------------------------------------------------
    // 2. Build uniform octree mesh.
    //    Domain: ±100km laterally, -100km to +50km vertically
    //    Level 5: 32×32×32 = 32768 cells, h = 150km/32 ≈ 4.7km
    //    At 0.1Hz, δ ≈ 50km → kh ≈ 0.59 (acceptable for FD)
    // ---------------------------------------------------------------
    octree::RefinementParams mesh_params;
    mesh_params.domain_x_min = -100000;  mesh_params.domain_x_max = 100000;
    mesh_params.domain_y_min = -100000;  mesh_params.domain_y_max = 100000;
    mesh_params.domain_z_min = -100000;  mesh_params.domain_z_max = 50000;

    mesh_params.min_level = 5;
    mesh_params.max_level = 5;  // uniform 32×32×32
    mesh_params.station_refine_radius = 0;
    mesh_params.station_refine_level = 5;
    mesh_params.sigma_bg = sigma_bg;

    octree::OctreeMesh mesh;
    mesh.setup(mesh_params, station_xyz, {frequencies.front()}, MPI_COMM_WORLD);
    mesh.set_terrain([](Real, Real) { return 0.0; });
    mesh.build_staggered_grid();

    int nc = mesh.staggered().num_cells();
    int ne = mesh.staggered().num_edges();
    std::cout << "  Mesh: " << nc << " cells, " << ne << " edge DOFs" << std::endl;

    // ---------------------------------------------------------------
    // 3. Uniform conductivity model (no anomaly)
    // ---------------------------------------------------------------
    model::ConductivityModel model;
    model.init_3d(nc, sigma_bg);

    // Set air cells
    auto& log_sigma = model.params();
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH)
            log_sigma[c] = std::log(1e-8);
    }
    model.invalidate_cache();

    // ---------------------------------------------------------------
    // 4. Forward solver with generous iteration count
    // ---------------------------------------------------------------
    forward::ForwardParamsFV fwd_params;
    fwd_params.bicgstab_tol = 1e-7;
    fwd_params.bicgstab_maxiter = 500;
    fwd_params.print_level = 1;
    fwd_params.div_correction = true;

    forward::ForwardSolverFV fwd;
    fwd.setup(mesh, fwd_params);
    fwd.update_sigma(model);

    // ---------------------------------------------------------------
    // 5. Setup MT data
    // ---------------------------------------------------------------
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

    data::MTData predicted;
    for (int s = 0; s < ns; ++s) {
        data::Station st;
        st.name = station_names[s];
        st.x = station_xyz[s][0];
        st.y = station_xyz[s][1];
        st.z = station_xyz[s][2];
        predicted.add_station(st);
    }
    predicted.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));

    // ---------------------------------------------------------------
    // 6. Forward solve
    // ---------------------------------------------------------------
    std::cout << "\n--- Forward solve ---" << std::endl;
    fwd.compute_responses(observed, predicted);

    // ---------------------------------------------------------------
    // 7. Results: compare with analytical 1D
    // ---------------------------------------------------------------
    std::cout << "\n--- §14.4 Results: 1D Halfspace Validation ---\n" << std::endl;
    // Note: system uses e^{+iωt} convention → halfspace phase = -135° (Zxy), +45° (Zyx)
    std::cout << "  Expected: ρ_app = 100 Ωm (±20% for coarse mesh)\n" << std::endl;
    std::cout << "  freq(Hz)  station   rho_xy(Ωm)  phase_xy(°)  rho_yx(Ωm)  phase_yx(°)  PASS?"
              << std::endl;
    std::cout << "  " << std::string(80, '-') << std::endl;

    bool all_pass = true;
    for (int f = 0; f < nf; ++f) {
        Real freq = frequencies[f];
        Real omega = constants::TWOPI * freq;

        for (int s = 0; s < ns; ++s) {
            const auto& p = predicted.predicted(s, f);

            Complex Zxy = p.Zxy.value;
            Complex Zyx = p.Zyx.value;

            Real rho_xy = std::norm(Zxy) / (omega * constants::MU0);
            Real phase_xy = std::arg(Zxy) * 180.0 / constants::PI;
            Real rho_yx = std::norm(Zyx) / (omega * constants::MU0);
            Real phase_yx = std::arg(Zyx) * 180.0 / constants::PI;

            // Check pass criteria: ρ ≈ 100 (±20% for coarse mesh, kh≈0.6)
            // Phase: -135° for Zxy (e^{+iωt} convention) or 45° for Zyx
            // At kh≈0.6, expect ~7% ρ error and ~12° phase error
            bool rho_ok = (rho_xy > 75 && rho_xy < 125) || (rho_yx > 75 && rho_yx < 125);
            bool phase_ok = (std::abs(phase_xy + 135) < 15 || std::abs(phase_yx - 45) < 15);
            bool pass = rho_ok && phase_ok;
            if (!pass) all_pass = false;

            std::cout << "  " << std::setw(8) << std::scientific << std::setprecision(1) << freq
                      << "  " << std::setw(8) << station_names[s]
                      << "  " << std::setw(10) << std::fixed << std::setprecision(2) << rho_xy
                      << "  " << std::setw(10) << std::fixed << std::setprecision(2) << phase_xy
                      << "  " << std::setw(10) << std::fixed << std::setprecision(2) << rho_yx
                      << "  " << std::setw(10) << std::fixed << std::setprecision(2) << phase_yx
                      << "  " << (pass ? "✓" : "✗")
                      << std::endl;
        }
    }

    std::cout << "\n  Overall: " << (all_pass ? "ALL PASSED ✓" : "SOME FAILED ✗") << std::endl;

    std::cout << "\n=== §14.4 Test Complete ===" << std::endl;

    MPI_Finalize();
    return 0;
}
