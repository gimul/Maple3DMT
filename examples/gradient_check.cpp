// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file gradient_check.cpp
/// @brief Finite-difference gradient verification for Maple3DMT.
///
/// Uses same setup as inversion_test (block anomaly, 4 stations, 1 freq).
/// Compares adjoint gradient with central FD for selected elements.
///
/// Usage: mpirun -np 1 gradient_check

#include "maple3dmt/inversion/inversion_3d.h"
#include "maple3dmt/mesh/hex_mesh_3d.h"
#include "maple3dmt/io/vtk_export_3d.h"
#include "maple3dmt/utils/logger.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mfem.hpp>

using namespace maple3dmt;

bool is_in_block(mfem::ParMesh& pmesh, int elem,
                 Real bx0, Real bx1, Real by0, Real by1, Real bz0, Real bz1) {
    mfem::Vector c(3);
    pmesh.GetElementCenter(elem, c);
    return c(0)>=bx0 && c(0)<=bx1 && c(1)>=by0 && c(1)<=by1 && c(2)>=bz0 && c(2)<=bz1;
}

int main(int argc, char* argv[]) {
    mfem::Mpi::Init(argc, argv);
    int rank = mfem::Mpi::WorldRank();

    // ── Mesh (same as inversion_test) ──
    std::vector<mesh::Station3D> stations;
    for (int iy = -1; iy <= 1; iy += 2)
        for (int ix = -1; ix <= 1; ix += 2) {
            mesh::Station3D s;
            s.name = "S" + std::to_string(stations.size()+1);
            s.x = ix*500.0; s.y = iy*500.0; s.z = 0;
            s.lon = 128.5; s.lat = 35.5; s.elevation = 0;
            stations.push_back(s);
        }

    mesh::MeshParams3D mp;
    // Tiny mesh for fast FGMRES convergence (gradient accuracy critical)
    mp.x_min=-5000; mp.x_max=5000; mp.y_min=-5000; mp.y_max=5000;
    mp.z_min=-10000; mp.z_air=5000;
    mp.h_surface_x=5000; mp.h_surface_y=5000; mp.h_surface_z=2000;
    mp.growth_x=2.0; mp.growth_y=2.0; mp.growth_z=2.0; mp.growth_air=3.0;
    mp.roi_x_pad=2000; mp.roi_y_pad=2000; mp.roi_depth=5000;
    mp.use_terrain=false; mp.refine_near_stations=0; mp.h_air_start=2000;

    mesh::HexMeshGenerator3D gen;
    auto smesh = gen.generate(mp, stations, nullptr);
    mfem::ParMesh pmesh(MPI_COMM_WORLD, *smesh);
    smesh.reset();

    int ne = pmesh.GetNE();
    if (rank==0) std::cout << "\n=== Gradient Check ===\n  Elements: " << ne << "\n";

    // ── True model (with block) ──
    Real sigma_bg = 0.01, sigma_block = 0.1;
    model::ConductivityModel true_model;
    true_model.init_3d(ne, sigma_bg);
    for (int e=0; e<ne; ++e)
        if (pmesh.GetAttribute(e)==1 &&
            is_in_block(pmesh, e, -750,750, -750,750, -2000,-500))
            true_model.params()[e] = std::log(sigma_block);

    // ── Observed data (true model + 2% noise) ──
    data::MTData observed;
    for (auto& s : stations) {
        data::Station ds;
        ds.name=s.name; ds.x=s.x; ds.y=s.y; ds.z=s.z;
        ds.lon=s.lon; ds.lat=s.lat; ds.has_geo=true;
        observed.add_station(ds);
    }
    RealVec freqs = {1.0};  // single frequency for speed
    observed.set_frequencies(freqs);

    forward::ForwardParams3D fp;
    fp.fe_order = 1;
    fp.backend = forward::SolverBackend::ITERATIVE;
    fp.gmres_tol = 1e-6;   // tight tolerance for gradient accuracy
    fp.gmres_maxiter = 1000;
    fp.gmres_kdim = 50;
    fp.gmres_print = -1;  // show final convergence info

    forward::ForwardSolver3D true_fwd;
    true_fwd.setup(pmesh, true_model, fp);
    true_fwd.compute_responses(observed, observed);

    // Add noise, set errors
    std::mt19937 rng(42);
    std::normal_distribution<Real> nd(0,1);
    for (int f=0; f<1; ++f)
        for (int s=0; s<4; ++s) {
            auto resp = observed.predicted(s,f);
            auto noise = [&](data::Datum& d) {
                Real mag = std::abs(d.value);
                Real err = std::max(mag*0.05, 1e-6);
                d.value += Complex(err*nd(rng), err*nd(rng));
                d.error = err;
                d.weight = 1.0;
            };
            // All 4 components for 3D
            Real Zxy_mag = std::abs(resp.Zxy.value);
            Real Zyx_mag = std::abs(resp.Zyx.value);
            Real offdiag_ref = std::sqrt(std::max(Zxy_mag * Zyx_mag, 1e-30));
            Real efloor = 0.05 * offdiag_ref;

            auto noise_all = [&](data::Datum& d) {
                Real mag = std::abs(d.value);
                Real err = std::max({mag*0.05, efloor, 1e-10});
                d.value += Complex(err*nd(rng), err*nd(rng));
                d.error = err;
                d.weight = 1.0;
            };
            noise_all(resp.Zxx); noise_all(resp.Zxy);
            noise_all(resp.Zyx); noise_all(resp.Zyy);
            observed.set_observed(s, f, resp);
        }

    // ── Starting model (uniform halfspace) ──
    model::ConductivityModel inv_model;
    inv_model.init_3d(ne, sigma_bg);

    forward::ForwardSolver3D inv_fwd;
    inv_fwd.setup(pmesh, inv_model, fp);

    regularization::RegParams rp;
    rp.alpha_s=1; rp.alpha_x=1; rp.alpha_z=1; rp.alpha_r=0;
    regularization::Regularization reg;
    reg.setup_3d(pmesh, rp);

    int n_active = reg.n_active();
    const auto& a2g = reg.active_to_global();

    // ── Setup inversion (to get gradient) ──
    inversion::InversionParams3D ip;
    ip.lambda_init = 10.0;
    ip.save_checkpoints = false;

    inversion::Inversion3D inv;
    inv.setup(pmesh, inv_model, observed, inv_fwd, reg, ip);

    // ── Step 1: Compute adjoint gradient ──
    if (rank==0) std::cout << "\n--- Computing adjoint gradient ---\n";
    RealVec adj_grad = inv.gradient();
    if (rank==0) {
        Real gnorm = 0;
        for (auto v : adj_grad) gnorm += v*v;
        gnorm = std::sqrt(gnorm);
        std::cout << "  ||g_adj|| = " << gnorm << " (n_active=" << n_active << ")\n";
        // Print first 5
        for (int j=0; j<std::min(5,n_active); ++j)
            std::cout << "  g_adj[" << j << "] (elem " << a2g[j] << ") = " << adj_grad[j] << "\n";
    }

    // ── Step 2: Compute objective at starting model ──
    // Forward solve already done inside gradient(), predicted is current.
    // Recompute to be safe.
    inv_fwd.compute_responses(observed, observed);
    Real phi0 = inv.objective();
    if (rank==0) std::cout << "\n  Phi(m0) = " << std::setprecision(12) << phi0 << "\n";

    // ── Step 3: FD gradient for selected elements ──
    // Use larger eps because iterative solver introduces noise at ~tol level
    Real eps = 5e-2;  // larger eps to overcome solver noise
    if (rank==0) std::cout << "\n--- Finite difference gradient (central, eps=" << eps << ") ---\n";

    // Test first 5 active elements + one in middle + one near end
    std::vector<int> test_j;
    for (int j=0; j<std::min(5,n_active); ++j) test_j.push_back(j);
    if (n_active > 10) test_j.push_back(n_active/2);
    if (n_active > 20) test_j.push_back(n_active-1);

    for (int j : test_j) {
        int e = a2g[j];
        Real orig = inv_model.params()[e];

        // Phi(m + eps)
        inv_model.params()[e] = orig + eps;
        inv_model.invalidate_cache();
        inv_fwd.compute_responses(observed, observed);
        Real phi_p = inv.objective();

        // Phi(m - eps)
        inv_model.params()[e] = orig - eps;
        inv_model.invalidate_cache();
        inv_fwd.compute_responses(observed, observed);
        Real phi_m = inv.objective();

        // Restore
        inv_model.params()[e] = orig;
        inv_model.invalidate_cache();

        Real fd = (phi_p - phi_m) / (2*eps);

        if (rank==0) {
            Real ratio = (std::abs(fd) > 1e-15) ? adj_grad[j] / fd : 0;
            std::cout << std::setprecision(8)
                      << "  j=" << j << " elem=" << e
                      << "  adj=" << std::setw(14) << adj_grad[j]
                      << "  fd=" << std::setw(14) << fd
                      << "  ratio=" << std::setw(8) << ratio
                      << "  phi+=" << phi_p << " phi-=" << phi_m
                      << "\n";
        }
    }

    // ── Step 4: Summary ──
    if (rank==0) {
        std::cout << "\n--- Interpretation ---\n"
                  << "  ratio ≈ 1.0: gradient is CORRECT\n"
                  << "  ratio ≈ -1.0: gradient has WRONG SIGN\n"
                  << "  ratio ≈ 0 or huge: gradient is WRONG\n"
                  << "  ratio varies wildly: numerical issues\n"
                  << std::endl;
    }

    mfem::Mpi::Finalize();
    return 0;
}
