// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file test_1d_layer.cpp
/// @brief 1D layered model test: 10 Ωm (0-5km) / 100 Ωm halfspace.
/// Verifies vertical wave propagation and impedance extraction.
///
/// Expected apparent resistivities (Wait recursion, 1D analytical):
///   0.01 Hz: ~24 Ωm (penetrates through layer)
///   0.1 Hz:  ~14 Ωm (mostly in layer)
///   1.0 Hz:  ~10 Ωm (shallow, block dominates)

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

    // Bottom halfspace impedance
    Complex k_n = std::sqrt(-iwmu * sigma[n-1]);
    Complex Z = iwmu / k_n;

    // Recurse upward
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

    std::cout << "\n=== 1D Layered Model Test ===" << std::endl;
    std::cout << "  Layer: 10 Ωm (σ=0.1) from 0 to -5000m" << std::endl;
    std::cout << "  Halfspace: 100 Ωm (σ=0.01) below -5000m" << std::endl;

    const Real sigma_layer = 0.1;   // 10 Ωm
    const Real sigma_hs    = 0.01;  // 100 Ωm
    const Real layer_depth = 5000;  // 5 km

    std::vector<Real> frequencies = {0.001, 0.01, 0.1, 1.0};

    // Print analytical 1D impedance
    std::cout << "\n  1D Analytical (Wait recursion):" << std::endl;
    std::cout << "  " << std::setw(10) << "freq"
              << "  " << std::setw(10) << "rho_a"
              << "  " << std::setw(10) << "phase" << std::endl;
    for (Real f : frequencies) {
        Complex Z = impedance_1d(f, {sigma_layer, sigma_hs}, {layer_depth});
        Real omega = 2 * M_PI * f;
        Real rho = std::norm(Z) / (omega * constants::MU0);
        Real phi = std::arg(Z) * 180.0 / M_PI;
        std::cout << "  " << std::scientific << std::setprecision(2)
                  << std::setw(10) << f
                  << "  " << std::fixed << std::setprecision(2)
                  << std::setw(10) << rho
                  << "  " << std::setw(10) << phi << std::endl;
    }

    // Single station at center
    std::vector<std::array<Real,3>> station_xyz = {{0.0, 0.0, 0.0}};

    // Mesh
    octree::RefinementParams mesh_params;
    mesh_params.domain_x_min = -200000;  mesh_params.domain_x_max = 200000;
    mesh_params.domain_y_min = -200000;  mesh_params.domain_y_max = 200000;
    mesh_params.domain_z_min = -200000;  mesh_params.domain_z_max = 56000;
    mesh_params.min_level = 5;
    mesh_params.max_level = 8;
    mesh_params.station_refine_radius = 25000;
    mesh_params.station_refine_level = 7;
    mesh_params.sigma_bg = sigma_hs;  // background = halfspace

    // Refine ONLY around the station/layer boundary (not the whole domain!)
    octree::RefineRegion layer_region;
    layer_region.x_min = -10000;  layer_region.x_max = 10000;
    layer_region.y_min = -10000;  layer_region.y_max = 10000;
    layer_region.z_min = -layer_depth;  layer_region.z_max = 0;
    layer_region.level = 8;
    layer_region.padding = 5000;
    mesh_params.refine_regions.push_back(layer_region);

    octree::OctreeMesh mesh;
    mesh.setup(mesh_params, station_xyz, {frequencies.front()}, MPI_COMM_WORLD);
    mesh.set_terrain([](Real, Real) { return 0.0; });
    mesh.build_staggered_grid();

    int nc = mesh.staggered().num_cells();
    int ne = mesh.staggered().num_edges();
    std::cout << "\n  Mesh: " << nc << " cells, " << ne << " edge DOFs" << std::endl;

    // 1D layered conductivity (no lateral variation)
    model::ConductivityModel model;
    model.init_3d(nc, sigma_hs);
    auto& log_sigma = model.params();
    int n_layer = 0;
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH) {
            log_sigma[c] = std::log(1e-8);
            continue;
        }
        Real cx, cy, cz;
        mesh.cell_center(c, cx, cy, cz);
        if (cz >= -layer_depth && cz < 0) {
            log_sigma[c] = std::log(sigma_layer);
            ++n_layer;
        }
    }
    model.invalidate_cache();
    std::cout << "  Layer cells: " << n_layer << std::endl;

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
    int ns = 1, nf = static_cast<int>(frequencies.size());
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
    std::cout << "\n--- Forward solve ---" << std::endl;
    fwd.compute_responses(observed, predicted);

    // Compare
    std::cout << "\n  " << std::setw(10) << "freq"
              << "  " << std::setw(10) << "rho_num"
              << "  " << std::setw(10) << "phi_num"
              << "  " << std::setw(10) << "rho_1d"
              << "  " << std::setw(10) << "phi_1d"
              << "  " << std::setw(10) << "err %"
              << std::endl;
    std::cout << "  " << std::string(65, '-') << std::endl;

    for (int f = 0; f < nf; ++f) {
        Real freq = frequencies[f];
        Real omega = constants::TWOPI * freq;

        const auto& p = predicted.predicted(0, f);
        Complex Zxy_num = p.Zxy.value;
        Real rho_num = std::norm(Zxy_num) / (omega * constants::MU0);
        Real phi_num = std::arg(Zxy_num) * 180.0 / M_PI;

        Complex Z_1d = impedance_1d(freq, {sigma_layer, sigma_hs}, {layer_depth});
        Real rho_1d = std::norm(Z_1d) / (omega * constants::MU0);
        Real phi_1d = std::arg(Z_1d) * 180.0 / M_PI;

        Real err = (rho_num - rho_1d) / rho_1d * 100;

        std::cout << "  " << std::scientific << std::setprecision(2)
                  << std::setw(10) << freq
                  << "  " << std::fixed << std::setprecision(2)
                  << std::setw(10) << rho_num
                  << "  " << std::setw(10) << phi_num
                  << "  " << std::setw(10) << rho_1d
                  << "  " << std::setw(10) << phi_1d
                  << "  " << std::setw(10) << err
                  << std::endl;
    }

    // Also test without div correction and with scattered field
    std::cout << "\n--- Without DivCorr ---" << std::endl;
    fwd_params.div_correction = false;
    fwd_params.scattered_field = false;
    forward::ForwardSolverFV fwd2;
    fwd2.setup(mesh, fwd_params);
    fwd2.update_sigma(model);

    data::MTData pred2;
    {
        data::Station st;
        st.name = "center";
        st.x = 0; st.y = 0; st.z = 0;
        pred2.add_station(st);
    }
    pred2.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));
    fwd2.compute_responses(observed, pred2);

    for (int f = 0; f < nf; ++f) {
        Real freq = frequencies[f];
        Real omega = constants::TWOPI * freq;
        const auto& p = pred2.predicted(0, f);
        Complex Zxy = p.Zxy.value;
        Real rho = std::norm(Zxy) / (omega * constants::MU0);
        Real phi = std::arg(Zxy) * 180.0 / M_PI;
        Complex Z_1d = impedance_1d(freq, {sigma_layer, sigma_hs}, {layer_depth});
        Real rho_1d = std::norm(Z_1d) / (omega * constants::MU0);
        Real err = (rho - rho_1d) / rho_1d * 100;
        std::cout << "  " << std::scientific << std::setprecision(2)
                  << std::setw(10) << freq
                  << "  " << std::fixed << std::setprecision(2)
                  << std::setw(10) << rho
                  << "  " << std::setw(10) << phi
                  << "  " << std::setw(10) << rho_1d
                  << "  " << std::setw(10) << err
                  << std::endl;
    }

    std::cout << "\n=== 1D Test Complete ===" << std::endl;
    MPI_Finalize();
    return 0;
}
