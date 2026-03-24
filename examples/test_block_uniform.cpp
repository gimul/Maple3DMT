// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file test_block_uniform.cpp
/// @brief Diagnostic: COMMEMI-like block on UNIFORM mesh (no hanging faces).
///
/// Purpose: isolate whether the wrong rho at center is caused by:
///   (a) hanging face treatment (→ should be OK on uniform mesh)
///   (b) formulation issue (→ still wrong on uniform mesh)
///
/// Uses L6 uniform mesh (no AMR) with reduced domain for speed.

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
#include <complex>
#include <vector>
#include <mpi.h>

using namespace maple3dmt;

/// Compute 1D MT impedance for a layered model using Wait recursion.
Complex impedance_1d(Real freq, const std::vector<Real>& sigma,
                     const std::vector<Real>& thickness) {
    int n = static_cast<int>(sigma.size());
    Real omega = 2 * M_PI * freq;
    Complex iwmu(0, omega * constants::MU0);
    Complex k_n = std::sqrt(-iwmu * sigma[n-1]);
    Complex Z = iwmu / k_n;
    for (int i = n-2; i >= 0; --i) {
        Complex k_i = std::sqrt(-iwmu * sigma[i]);
        Complex Z_i = iwmu / k_i;
        Complex ik_d = Complex(0,1) * k_i * thickness[i];
        Complex th = std::tanh(ik_d);
        Z = Z_i * (Z + Z_i * th) / (Z_i + Z * th);
    }
    return Z;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    utils::Logger::instance().set_rank(rank);

    std::cout << "\n=== Block on Uniform Mesh (no hanging faces) ===" << std::endl;

    const Real sigma_bg    = 0.01;   // 100 Ωm halfspace
    const Real sigma_block = 0.1;    // 10 Ωm block

    // Block bounds (same as COMMEMI 3D-1)
    const Real block_half = 5000;
    const Real block_depth = 5000;

    // Only test 0.01 Hz (the problematic frequency)
    std::vector<Real> frequencies = {0.01};

    // Print analytical 1D reference
    for (Real f : frequencies) {
        Complex Z_1d = impedance_1d(f, {sigma_block, sigma_bg}, {block_depth});
        Real omega = 2 * M_PI * f;
        Real rho_1d = std::norm(Z_1d) / (omega * constants::MU0);
        std::cout << "  1D layer ref at " << f << " Hz: rho_a = " << rho_1d << " Ωm" << std::endl;
    }
    std::cout << "  COMMEMI 3D center at 0.01 Hz: rho ~ 55-85 Ωm (published)" << std::endl;

    // Station at center only
    std::vector<std::array<Real,3>> station_xyz = {{0.0, 0.0, 0.0}};

    // --- UNIFORM mesh: min_level = max_level = 6 ---
    // Domain: slightly reduced for speed but still > 1 skin depth
    // At 0.01 Hz in 100 Ωm: skin depth ≈ 159 km
    // Domain ±200km × ±200km × [-200km, 56km]
    // At L6: cell_x = 400000/64 = 6250m, cell_z = 256000/64 = 4000m
    octree::RefinementParams mesh_params;
    mesh_params.domain_x_min = -200000;  mesh_params.domain_x_max = 200000;
    mesh_params.domain_y_min = -200000;  mesh_params.domain_y_max = 200000;
    mesh_params.domain_z_min = -200000;  mesh_params.domain_z_max = 56000;
    mesh_params.min_level = 6;
    mesh_params.max_level = 6;  // UNIFORM: no AMR
    mesh_params.station_refine_radius = 0;  // No station refinement
    mesh_params.station_refine_level = 6;
    mesh_params.sigma_bg = sigma_bg;
    // NO refine_regions → pure uniform L6

    octree::OctreeMesh mesh;
    mesh.setup(mesh_params, station_xyz, {frequencies.front()}, MPI_COMM_WORLD);
    mesh.set_terrain([](Real, Real) { return 0.0; });
    mesh.build_staggered_grid();

    int nc = mesh.staggered().num_cells();
    int ne = mesh.staggered().num_edges();
    std::cout << "\n  Uniform L6 Mesh: " << nc << " cells, " << ne << " edge DOFs" << std::endl;

    // Assign block conductivity
    model::ConductivityModel model;
    model.init_3d(nc, sigma_bg);
    auto& log_sigma = model.params();
    int n_block = 0;
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH) {
            log_sigma[c] = std::log(1e-8);
            continue;
        }
        Real cx, cy, cz;
        mesh.cell_center(c, cx, cy, cz);
        if (std::abs(cx) <= block_half && std::abs(cy) <= block_half &&
            cz >= -block_depth && cz < 0) {
            log_sigma[c] = std::log(sigma_block);
            ++n_block;
        }
    }
    model.invalidate_cache();
    std::cout << "  Block cells: " << n_block << std::endl;

    // Forward solver
    forward::ForwardParamsFV fwd_params;
    fwd_params.bicgstab_tol = 1e-8;
    fwd_params.bicgstab_maxiter = 10000;
    fwd_params.print_level = 1;
    fwd_params.div_correction = false;
    fwd_params.scattered_field = false;

    forward::ForwardSolverFV fwd;
    fwd.setup(mesh, fwd_params);
    fwd.update_sigma(model);

    // Data
    int nf = static_cast<int>(frequencies.size());
    data::MTData observed, predicted;
    {
        data::Station st;
        st.name = "center";
        st.x = 0; st.y = 0; st.z = 0;
        observed.add_station(st);
        predicted.add_station(st);
    }
    observed.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));
    predicted.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));
    for (int f = 0; f < nf; ++f) {
        data::MTResponse resp;
        observed.set_observed(0, f, resp);
    }

    // Solve
    std::cout << "\n--- Forward solve (uniform L6, no hanging faces) ---" << std::endl;
    fwd.compute_responses(observed, predicted);

    // Results
    for (int f = 0; f < nf; ++f) {
        Real freq = frequencies[f];
        Real omega = constants::TWOPI * freq;
        const auto& p = predicted.predicted(0, f);

        Complex Zxy_num = p.Zxy.value;
        Complex Zyx_num = p.Zyx.value;
        Real rho_xy = std::norm(Zxy_num) / (omega * constants::MU0);
        Real phi_xy = std::arg(Zxy_num) * 180.0 / M_PI;
        Real rho_yx = std::norm(Zyx_num) / (omega * constants::MU0);
        Real phi_yx = std::arg(Zyx_num) * 180.0 / M_PI;

        Complex Z_1d = impedance_1d(freq, {sigma_block, sigma_bg}, {block_depth});
        Real rho_1d = std::norm(Z_1d) / (omega * constants::MU0);

        std::cout << "\n  Freq = " << freq << " Hz:" << std::endl;
        std::cout << "    rho_xy = " << rho_xy << " Ωm  (phase " << phi_xy << "°)" << std::endl;
        std::cout << "    rho_yx = " << rho_yx << " Ωm  (phase " << phi_yx << "°)" << std::endl;
        std::cout << "    1D ref = " << rho_1d << " Ωm" << std::endl;
        std::cout << "    COMMEMI published: 55-85 Ωm" << std::endl;
        std::cout << "    If rho >> 10 → formulation OK, hanging faces were the issue" << std::endl;
        std::cout << "    If rho ≈ 10 → formulation issue (not hanging faces)" << std::endl;
    }

    // Also test uniform halfspace on same mesh as sanity check
    std::cout << "\n--- Sanity check: uniform halfspace on same mesh ---" << std::endl;
    model::ConductivityModel model_hs;
    model_hs.init_3d(nc, sigma_bg);
    auto& ls_hs = model_hs.params();
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH)
            ls_hs[c] = std::log(1e-8);
    }
    model_hs.invalidate_cache();

    forward::ForwardSolverFV fwd_hs;
    fwd_hs.setup(mesh, fwd_params);
    fwd_hs.update_sigma(model_hs);

    data::MTData pred_hs;
    {
        data::Station st;
        st.name = "center";
        st.x = 0; st.y = 0; st.z = 0;
        pred_hs.add_station(st);
    }
    pred_hs.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));
    fwd_hs.compute_responses(observed, pred_hs);

    for (int f = 0; f < nf; ++f) {
        Real freq = frequencies[f];
        Real omega = constants::TWOPI * freq;
        const auto& p = pred_hs.predicted(0, f);
        Complex Zxy = p.Zxy.value;
        Real rho = std::norm(Zxy) / (omega * constants::MU0);
        Real phi = std::arg(Zxy) * 180.0 / M_PI;
        std::cout << "  Halfspace: rho_xy = " << rho << " Ωm  (expected 100, phase " << phi << "°)" << std::endl;
    }

    std::cout << "\n=== Uniform Mesh Block Test Complete ===" << std::endl;
    MPI_Finalize();
    return 0;
}
