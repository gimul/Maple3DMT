// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file adjoint_test.cpp
/// @brief Validate adjoint gradient against finite-difference.

#include "maple3dmt/forward/forward_solver_3d.h"
#include "maple3dmt/mesh/hex_mesh_3d.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/utils/logger.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <mfem.hpp>

using namespace maple3dmt;

namespace {

/// Element-wise constant coefficient (indexed by local element number).
class ElementCoefficient : public mfem::Coefficient {
    const mfem::Vector& values_;
public:
    ElementCoefficient(const mfem::Vector& v) : values_(v) {}
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint&) override {
        return values_(T.ElementNo);
    }
};

/// VectorCoefficient = scalar_coeff(x) * grid_function(x)
class ScaledGFVectorCoefficient : public mfem::VectorCoefficient {
    mfem::Coefficient& scalar_;
    mfem::ParGridFunction& gf_;
public:
    ScaledGFVectorCoefficient(mfem::Coefficient& s, mfem::ParGridFunction& g)
        : mfem::VectorCoefficient(g.FESpace()->GetMesh()->SpaceDimension()),
          scalar_(s), gf_(g) {}
    void Eval(mfem::Vector& V, mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override {
        Real s = scalar_.Eval(T, ip);
        gf_.GetVectorValue(T, ip, V);
        V *= s;
    }
};

Complex analytic_impedance(Real omega, Real sigma) {
    Real mag = std::sqrt(omega * constants::MU0 / sigma);
    return Complex(mag / std::sqrt(2.0), mag / std::sqrt(2.0));
}

Real compute_misfit(const data::MTData& observed,
                    const data::MTData& predicted,
                    const RealVec& data_weights) {
    Real phi = 0.0;
    int ns = observed.num_stations();
    for (int s = 0; s < ns; ++s) {
        const auto& obs = observed.observed(s, 0);
        const auto& pred = predicted.predicted(s, 0);
        int d = s * 8;
        auto add = [&](Complex r, int i) {
            phi += std::pow(r.real() * data_weights[d+i], 2);
            phi += std::pow(r.imag() * data_weights[d+i+1], 2);
        };
        add(obs.Zxx.value - pred.Zxx.value, 0);
        add(obs.Zxy.value - pred.Zxy.value, 2);
        add(obs.Zyx.value - pred.Zyx.value, 4);
        add(obs.Zyy.value - pred.Zyy.value, 6);
    }
    return 0.5 * phi;
}

Real forward_and_misfit(mfem::ParMesh& pmesh,
                        model::ConductivityModel& model,
                        const forward::ForwardParams3D& fwd_params,
                        const data::MTData& observed,
                        const RealVec& data_weights) {
    data::MTData predicted;
    for (int s = 0; s < observed.num_stations(); ++s)
        predicted.add_station(observed.station(s));
    predicted.set_frequencies(observed.frequencies());
    forward::ForwardSolver3D solver;
    solver.setup(pmesh, model, fwd_params);
    solver.compute_single_frequency(0, observed, predicted);
    solver.release_factorization();
    return compute_misfit(observed, predicted, data_weights);
}

/// Verify basis function evaluation at a point by comparing manual
/// summation Σ u_k N_k(x) with MFEM's GetVectorValue.
void verify_basis_functions(mfem::ParFiniteElementSpace* fes,
                            mfem::ParMesh* pmesh,
                            mfem::ParGridFunction& gf,
                            int elem,
                            const mfem::IntegrationPoint& ip,
                            const char* label) {
    mfem::ElementTransformation* T = pmesh->GetElementTransformation(elem);
    T->SetIntPoint(&ip);

    // MFEM's built-in evaluation
    mfem::Vector val_mfem(3);
    gf.GetVectorValue(*T, ip, val_mfem);

    // Manual evaluation via basis functions
    mfem::Array<int> dofs;
    fes->GetElementDofs(elem, dofs);
    const mfem::FiniteElement* fe = fes->GetFE(elem);
    int ndof = dofs.Size();
    mfem::DenseMatrix vshape(ndof, 3);
    fe->CalcVShape(*T, vshape);

    mfem::Vector val_manual(3);
    val_manual = 0.0;
    for (int k = 0; k < ndof; ++k) {
        int gdof = dofs[k];
        Real sign = (gdof >= 0) ? 1.0 : -1.0;
        Real u_k = gf(std::abs(gdof));
        for (int d = 0; d < 3; ++d) {
            val_manual(d) += u_k * sign * vshape(k, d);
        }
    }

    Real err = 0.0;
    for (int d = 0; d < 3; ++d)
        err += std::pow(val_mfem(d) - val_manual(d), 2);
    err = std::sqrt(err);
    Real norm = val_mfem.Norml2();

    std::cout << "  " << label << ": MFEM=(" << val_mfem(0) << "," << val_mfem(1)
              << ") Manual=(" << val_manual(0) << "," << val_manual(1)
              << ") err=" << err << " rel=" << (norm > 0 ? err/norm : err) << "\n";
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    mfem::Mpi::Init(argc, argv);
    int rank = mfem::Mpi::WorldRank();

    Real sigma_bg = 0.01;
    Real test_freq = 1.0;
    Real eps = 1e-4;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sigma" && i+1 < argc) sigma_bg = std::stod(argv[++i]);
        else if (arg == "--freq" && i+1 < argc) test_freq = std::stod(argv[++i]);
        else if (arg == "--eps" && i+1 < argc) eps = std::stod(argv[++i]);
    }

    Real omega = constants::TWOPI * test_freq;

    if (rank == 0) {
        std::cout << "\n=== Adjoint Gradient Validation ===\n"
                  << "  sigma=" << sigma_bg << "  freq=" << test_freq
                  << "  eps=" << eps << "\n\n";
    }

    // --- Mesh ---
    std::vector<mesh::Station3D> stations;
    for (int i = 0; i < 2; ++i) {
        mesh::Station3D s;
        s.name = "ST" + std::to_string(i+1);
        s.x = (i == 0) ? -500.0 : 500.0;
        s.y = 0.0; s.z = 0.0;
        s.lon = 128.5; s.lat = 35.5; s.elevation = 0.0;
        stations.push_back(std::move(s));
    }
    auto mp = mesh::auto_mesh_params(stations);
    mp.use_terrain = false; mp.refine_near_stations = 0;
    mesh::HexMeshGenerator3D gen;
    auto smesh = gen.generate(mp, stations, nullptr);
    if (rank == 0)
        std::cout << "Elements: " << smesh->GetNE() << "\n\n";

    mfem::ParMesh pmesh(MPI_COMM_WORLD, *smesh);
    smesh.reset();
    int ne = pmesh.GetNE();

    // --- Observed: analytic halfspace σ=0.1 ---
    Real sigma_obs = 10.0 * sigma_bg;
    Complex Z_obs = analytic_impedance(omega, sigma_obs);

    data::MTData observed, predicted;
    for (auto& s : stations) {
        data::Station ds;
        ds.name = s.name; ds.x = s.x; ds.y = s.y; ds.z = s.z;
        ds.lon = s.lon; ds.lat = s.lat; ds.has_geo = true;
        observed.add_station(ds);
        predicted.add_station(ds);
    }
    observed.set_frequencies({test_freq});
    predicted.set_frequencies({test_freq});

    int ns = observed.num_stations();
    for (int s = 0; s < ns; ++s) {
        data::MTResponse r;
        r.Zxy.value = Z_obs; r.Zyx.value = -Z_obs;
        observed.set_observed(s, 0, r);
    }

    RealVec data_weights(ns * 8, 1.0);

    // --- Forward at background model ---
    model::ConductivityModel model;
    model.init_3d(ne, sigma_bg);

    forward::ForwardParams3D fwd_params;
    fwd_params.fe_order = 1;
    fwd_params.backend = forward::SolverBackend::ITERATIVE;  // exact factorization (no BLR)
    fwd_params.blr_tolerance = 1e-10;

    forward::ForwardSolver3D solver;
    solver.setup(pmesh, model, fwd_params);
    solver.compute_single_frequency(0, observed, predicted);

    Real phi0 = compute_misfit(observed, predicted, data_weights);
    if (rank == 0) {
        auto& p = predicted.predicted(0, 0);
        std::cout << "Z_obs_xy  = (" << Z_obs.real() << ", " << Z_obs.imag() << ")\n"
                  << "Z_pred_xy = (" << p.Zxy.value.real() << ", " << p.Zxy.value.imag() << ")\n"
                  << "Phi = " << phi0 << "\n\n";
    }

    // --- Diagnostic: verify basis function evaluation at station 0 ---
    if (rank == 0) {
        std::cout << "=== Basis Function Verification (Station 0) ===\n";
        // Find station element (same as forward solver does internally)
        mfem::DenseMatrix pts(3, 1);
        pts(0,0) = stations[0].x;
        pts(1,0) = stations[0].y;
        pts(2,0) = stations[0].z - 0.1;
        mfem::Array<int> elem_ids;
        mfem::Array<mfem::IntegrationPoint> ips;
        pmesh.FindPoints(pts, elem_ids, ips);

        if (elem_ids[0] >= 0) {
            int elem = elem_ids[0];
            auto& ip = ips[0];
            std::cout << "  Station in element " << elem << "\n";

            auto* fes = solver.fespace();
            verify_basis_functions(fes, &pmesh, *solver.E1_real(), elem, ip, "E1_real");
            verify_basis_functions(fes, &pmesh, *solver.E1_imag(), elem, ip, "E1_imag");

            // Also verify curl evaluation
            mfem::ElementTransformation* T = pmesh.GetElementTransformation(elem);
            T->SetIntPoint(&ip);

            mfem::Vector curlE1r_mfem(3);
            solver.E1_real()->GetCurl(*T, curlE1r_mfem);

            // Manual curl evaluation
            mfem::Array<int> dofs;
            fes->GetElementDofs(elem, dofs);
            const mfem::FiniteElement* fe = fes->GetFE(elem);
            int ndof = dofs.Size();
            mfem::DenseMatrix curlshape(ndof, 3);
            fe->CalcCurlShape(ip, curlshape);

            const mfem::DenseMatrix& J = T->Jacobian();
            Real detJ = T->Weight();
            mfem::DenseMatrix curlshape_phys(ndof, 3);
            for (int k = 0; k < ndof; ++k) {
                for (int d = 0; d < 3; ++d) {
                    Real val = 0.0;
                    for (int dd = 0; dd < 3; ++dd)
                        val += J(d, dd) * curlshape(k, dd);
                    curlshape_phys(k, d) = val / detJ;
                }
            }

            mfem::Vector curlE1r_manual(3);
            curlE1r_manual = 0.0;
            for (int k = 0; k < ndof; ++k) {
                int gdof = dofs[k];
                Real sign = (gdof >= 0) ? 1.0 : -1.0;
                Real u_k = (*solver.E1_real())(std::abs(gdof));
                for (int d = 0; d < 3; ++d)
                    curlE1r_manual(d) += u_k * sign * curlshape_phys(k, d);
            }

            Real cerr = 0;
            for (int d = 0; d < 3; ++d)
                cerr += std::pow(curlE1r_mfem(d) - curlE1r_manual(d), 2);
            cerr = std::sqrt(cerr);
            std::cout << "  curl_E1r: MFEM=(" << curlE1r_mfem(0) << "," << curlE1r_mfem(1)
                      << ") Manual=(" << curlE1r_manual(0) << "," << curlE1r_manual(1)
                      << ") err=" << cerr << "\n";

            // Print element info
            mfem::Vector ctr(3);
            pmesh.GetElementCenter(elem, ctr);
            std::cout << "  Element center: (" << ctr(0) << "," << ctr(1) << "," << ctr(2) << ")\n";
            std::cout << "  detJ = " << detJ << "  ndof = " << ndof << "\n\n";
        }
    }

    // ==================================================================
    // BLOCK ORDERING DIAGNOSTIC: verify system_matrix_ convention
    // Try both orderings: [E_real;E_imag] and [E_imag;E_real]
    // The correct ordering should give A*x ≈ B (forward RHS)
    // ==================================================================
    if (rank == 0) {
        std::cout << "=== Block Ordering Diagnostic ===\n";
        int tdof = solver.fespace()->GetTrueVSize();
        int sys_size = 2 * tdof;

        // Get true DOF vectors from E1
        mfem::Vector e1r_tdof(tdof), e1i_tdof(tdof);
        solver.E1_real()->GetTrueDofs(e1r_tdof);
        solver.E1_imag()->GetTrueDofs(e1i_tdof);

        // Option A: [E_real; E_imag]  (what solve_forward_rhs assumes)
        mfem::Vector xA(sys_size);
        for (int i = 0; i < tdof; ++i) {
            xA(i)       = e1r_tdof(i);
            xA(i+tdof)  = e1i_tdof(i);
        }

        // Option B: [E_imag; E_real]  (swapped)
        mfem::Vector xB(sys_size);
        for (int i = 0; i < tdof; ++i) {
            xB(i)       = e1i_tdof(i);
            xB(i+tdof)  = e1r_tdof(i);
        }

        // Multiply by system matrix
        mfem::Vector AxA(sys_size), AxB(sys_size);
        solver.system_matrix()->Mult(xA, AxA);
        solver.system_matrix()->Mult(xB, AxB);

        // For the correct ordering, A*x should equal the forward RHS.
        // Since interior RHS ≈ 0 for uniform halfspace, |A*x| at interior should be small.
        // But essential DOF rows are identity, so A*x[ess] = x[ess].
        // Compare norms of A*x (smaller is better for interior DOFs).
        std::cout << "  |A * [Er;Ei]| = " << AxA.Norml2() << "\n"
                  << "  |A * [Ei;Er]| = " << AxB.Norml2() << "\n";

        // Better test: the forward RHS is B from FormLinearSystem which is nonzero
        // only at boundary contributions. We don't have B, but we can check which
        // ordering makes A*x closer to zero at specific interior DOFs.
        // Pick DOFs near the station element.
        mfem::Array<int> dofs;
        solver.fespace()->GetElementDofs(89398, dofs);
        Real normA = 0, normB = 0;
        for (int k = 0; k < dofs.Size(); ++k) {
            int td = solver.fespace()->GetLocalTDofNumber(std::abs(dofs[k]));
            if (td < 0) continue;
            // Check both blocks
            normA += AxA(td)*AxA(td) + AxA(td+tdof)*AxA(td+tdof);
            normB += AxB(td)*AxB(td) + AxB(td+tdof)*AxB(td+tdof);
        }
        std::cout << "  Interior DOFs (elem 89398): |A*[Er;Ei]| = " << std::sqrt(normA)
                  << "  |A*[Ei;Er]| = " << std::sqrt(normB) << "\n";

        // Check off-diagonal block sign: is A(0,1) block = +ωσM or -ωσM?
        // Compute A*[0;u_i] and check sign of block0 result at interior DOFs
        mfem::Vector x_imag_only(sys_size);
        x_imag_only = 0.0;
        for (int i = 0; i < tdof; ++i) x_imag_only(i + tdof) = e1i_tdof(i);
        mfem::Vector Ax_imag(sys_size);
        solver.system_matrix()->Mult(x_imag_only, Ax_imag);

        // A*[0; u_i] = [A_01 u_i; A_11 u_i]
        // If A_01 = +ωσM: block0 = ωσM u_i (positive for positive u_i)
        // If A_01 = -ωσM: block0 = -ωσM u_i (negative)
        // Sample a DOF from element 89398
        int sample_td = solver.fespace()->GetLocalTDofNumber(std::abs(dofs[0]));
        if (sample_td >= 0) {
            std::cout << "  Off-diag check: A*[0;u_i]_block0[" << sample_td << "] = "
                      << Ax_imag(sample_td) << "  u_i[" << sample_td << "] = "
                      << e1i_tdof(sample_td) << "\n";
            // If same sign → +ωσM (convention 1). If opposite → -ωσM (convention 2).
            Real ratio = Ax_imag(sample_td) / (e1i_tdof(sample_td) + 1e-30);
            std::cout << "  ratio A01*u_i/u_i = " << ratio
                      << " (>0 → +ωσM, <0 → -ωσM)\n";
        }
        std::cout << "\n";
    }

    // ==================================================================
    // ADJOINT GRADIENT
    // ==================================================================
    mfem::Vector adj_rhs1, adj_rhs2;
    solver.build_adjoint_rhs(0, observed, predicted, data_weights,
                             adj_rhs1, adj_rhs2);

    if (rank == 0) {
        std::cout << "|adj_rhs1| = " << adj_rhs1.Norml2()
                  << "  |adj_rhs2| = " << adj_rhs2.Norml2() << "\n";
    }

    auto* fes = solver.fespace();
    mfem::ParGridFunction l1r(fes), l1i(fes), l2r(fes), l2i(fes);
    solver.adjoint_solve(adj_rhs1, l1r, l1i);
    solver.adjoint_solve(adj_rhs2, l2r, l2i);

    if (rank == 0) {
        std::cout << "|lam1_r|=" << l1r.Norml2() << " |lam1_i|=" << l1i.Norml2()
                  << "  |lam2_r|=" << l2r.Norml2() << " |lam2_i|=" << l2i.Norml2() << "\n";
    }

    RealVec gp1(ne), gp2(ne), g_adj(ne);
    solver.compute_sensitivity(*solver.E1_real(), *solver.E1_imag(),
                               l1r, l1i, gp1);
    solver.compute_sensitivity(*solver.E2_real(), *solver.E2_imag(),
                               l2r, l2i, gp2);
    for (int e = 0; e < ne; ++e) g_adj[e] = gp1[e] + gp2[e];

    int test_e = -1;
    Real max_g = 0;
    for (int e = 0; e < ne; ++e) {
        if (pmesh.GetAttribute(e) == 2) continue;
        if (std::abs(g_adj[e]) > max_g) { max_g = std::abs(g_adj[e]); test_e = e; }
    }

    if (rank == 0) {
        mfem::Vector ctr(3);
        pmesh.GetElementCenter(test_e, ctr);
        std::cout << "\nMax |g_adj| = " << max_g << " at elem " << test_e
                  << " (z=" << ctr(2) << "m)\n\n";

        // Print gradient sums
        Real g_sum = std::accumulate(g_adj.begin(), g_adj.end(), 0.0);
        Real g_abs_sum = std::accumulate(g_adj.begin(), g_adj.end(), 0.0,
                                          [](Real a, Real b){ return a + std::abs(b); });
        std::cout << "Sum(g_adj) = " << g_sum << "  Sum(|g_adj|) = " << g_abs_sum << "\n";
    }

    // ==================================================================
    // J·v TEST: Forward perturbation for single element (no adjoint)
    // This isolates whether the bug is in Q^T/adjoint or forward chain.
    // ==================================================================
    if (rank == 0) std::cout << "=== J·v Verification (elem " << test_e << ") ===\n";
    {
        // Perturb element test_e: δ(log σ) = 1  →  δσ = σ_bg
        Real delta_sigma = sigma_bg;  // since d/d(log σ) * 1 = σ · 1

        // Build perturbation RHS: δA·E = -iωδσ M_e E
        // For both polarizations, then extract δZ at stations
        int tdof = solver.fespace()->GetTrueVSize();
        int sys_size = 2 * tdof;

        // For polarization p, the perturbation RHS in element test_e:
        // rhs_real[k] = ωδσ ∫_e N_k · E_imag dV   (from -iω·δσ → ω affects imag)
        // rhs_imag[k] = -ωδσ ∫_e N_k · E_real dV
        // Actually: -iωδσ M E = -iωδσ(E_r + iE_i) M = ωδσ E_i M - iωδσ E_r M
        // So RHS_real = ωδσ M E_i,  RHS_imag = -ωδσ M E_r

        Real omega_ds = omega * delta_sigma;

        for (int pol = 0; pol < 2; ++pol) {
            mfem::Vector pert_rhs(sys_size);
            pert_rhs = 0.0;

            const auto& E_r = (pol == 0) ? *solver.E1_real() : *solver.E2_real();
            const auto& E_i = (pol == 0) ? *solver.E1_imag() : *solver.E2_imag();

            // Integrate over the test element
            mfem::ElementTransformation* T = pmesh.GetElementTransformation(test_e);
            const mfem::FiniteElement* fe = solver.fespace()->GetFE(test_e);
            mfem::Array<int> dofs;
            solver.fespace()->GetElementDofs(test_e, dofs);
            int ndof_e = dofs.Size();

            int order_q = 2 * fe->GetOrder() + T->OrderW();
            const mfem::IntegrationRule& ir_q =
                mfem::IntRules.Get(fe->GetGeomType(), order_q);

            for (int q = 0; q < ir_q.GetNPoints(); ++q) {
                const mfem::IntegrationPoint& ip_q = ir_q.IntPoint(q);
                T->SetIntPoint(&ip_q);
                Real w = ip_q.weight * T->Weight();

                mfem::Vector Er(3), Ei(3);
                E_r.GetVectorValue(*T, ip_q, Er);
                E_i.GetVectorValue(*T, ip_q, Ei);

                mfem::DenseMatrix vshape(ndof_e, 3);
                fe->CalcVShape(*T, vshape);

                for (int k = 0; k < ndof_e; ++k) {
                    int gdof = dofs[k];
                    int tdof_k = solver.fespace()->GetLocalTDofNumber(std::abs(gdof));
                    if (tdof_k < 0) continue;
                    Real sign = (gdof >= 0) ? 1.0 : -1.0;

                    Real Nk_dot_Ei = 0, Nk_dot_Er = 0;
                    for (int d = 0; d < 3; ++d) {
                        Nk_dot_Ei += sign * vshape(k, d) * Ei(d);
                        Nk_dot_Er += sign * vshape(k, d) * Er(d);
                    }

                    // System matrix ordering [block0=u_r; block1=u_i]:
                    // -δA·E = [[0,-ωδσM],[ωδσM,0]] [u_r;u_i]
                    // block0 = -ωδσ M u_i = -ωδσ ∫ N_k · Ei dV  (Ei=u_i, true imag)
                    // block1 =  ωδσ M u_r =  ωδσ ∫ N_k · Er dV  (Er=u_r, true real)
                    pert_rhs(tdof_k)        += w * (-omega_ds) * Nk_dot_Ei;
                    pert_rhs(tdof_k + tdof) += w * omega_ds * Nk_dot_Er;
                }
            }

            if (rank == 0)
                std::cout << "  |pert_rhs_pol" << pol << "| = " << pert_rhs.Norml2() << "\n";

            // Solve A·δE = pert_rhs
            mfem::ParGridFunction dE_r(solver.fespace()), dE_i(solver.fespace());
            solver.solve_forward_rhs(pert_rhs, dE_r, dE_i);

            // Verify solve: compute |A*δX - pert_rhs| / |pert_rhs|
            if (rank == 0) {
                mfem::Vector dX_true(sys_size);
                dE_r.GetTrueDofs(dX_true);  // first half
                mfem::Vector dX_true_i(sys_size);
                dE_i.GetTrueDofs(dX_true_i);
                // Build full vector [dX_r; dX_i]
                mfem::Vector dX_full(sys_size);
                for (int ii = 0; ii < tdof; ++ii) {
                    dX_full(ii)        = dX_true(ii);
                    dX_full(ii + tdof) = dX_true_i(ii);
                }
                mfem::Vector AdX(sys_size);
                solver.system_matrix()->Mult(dX_full, AdX);
                AdX -= pert_rhs;
                std::cout << "  Solve residual pol" << pol << ": |A*dX - rhs|/|rhs| = "
                          << AdX.Norml2() / pert_rhs.Norml2() << "\n";
                std::cout << "  |dE_r|=" << dE_r.Norml2() << " |dE_i|=" << dE_i.Norml2() << "\n";
            }

            // Extract δZ at stations and compute directional derivative
            // dΦ/dp_j = Re[Σ_s Σ_ij (δZ_ij)^* · W² · r_ij]
            // where r = obs - pred (same as in build_adjoint_rhs)
            Real g_Jv_pol = 0.0;
            for (int s2 = 0; s2 < ns; ++s2) {
                // We need station location in mesh — reuse solver's internal data
                // Evaluate δE, δcurlE at station
                mfem::DenseMatrix pts2(3, 1);
                pts2(0,0) = observed.station(s2).x;
                pts2(1,0) = observed.station(s2).y;
                pts2(2,0) = observed.station(s2).z - 0.1;
                mfem::Array<int> eids2;
                mfem::Array<mfem::IntegrationPoint> ips2;
                pmesh.FindPoints(pts2, eids2, ips2);
                if (eids2[0] < 0) continue;

                int elem2 = eids2[0];
                auto& ip2 = ips2[0];
                mfem::ElementTransformation* T2 = pmesh.GetElementTransformation(elem2);
                T2->SetIntPoint(&ip2);

                mfem::Vector dEr(3), dEi(3);
                dE_r.GetVectorValue(*T2, ip2, dEr);
                dE_i.GetVectorValue(*T2, ip2, dEi);

                mfem::Vector dcurlEr(3), dcurlEi(3);
                dE_r.GetCurl(*T2, dcurlEr);
                dE_i.GetCurl(*T2, dcurlEi);

                // δH = curl(δE)/(iωμ₀) = (dcurl_i - i dcurl_r)/(ωμ₀)
                Real inv_wmu = 1.0 / (omega * constants::MU0);
                Complex dHx(dcurlEi(0)*inv_wmu, -dcurlEr(0)*inv_wmu);
                Complex dHy(dcurlEi(1)*inv_wmu, -dcurlEr(1)*inv_wmu);

                Complex dEx(dEr(0), dEi(0));
                Complex dEy(dEr(1), dEi(1));

                // Get station's E, H, Z (from the forward solve)
                // Re-evaluate
                mfem::Vector E1r(3), E1i(3), E2r(3), E2i(3);
                solver.E1_real()->GetVectorValue(*T2, ip2, E1r);
                solver.E1_imag()->GetVectorValue(*T2, ip2, E1i);
                solver.E2_real()->GetVectorValue(*T2, ip2, E2r);
                solver.E2_imag()->GetVectorValue(*T2, ip2, E2i);

                mfem::Vector cE1r(3), cE1i(3), cE2r(3), cE2i(3);
                solver.E1_real()->GetCurl(*T2, cE1r);
                solver.E1_imag()->GetCurl(*T2, cE1i);
                solver.E2_real()->GetCurl(*T2, cE2r);
                solver.E2_imag()->GetCurl(*T2, cE2i);

                Complex Hx1(cE1i(0)*inv_wmu, -cE1r(0)*inv_wmu);
                Complex Hy1(cE1i(1)*inv_wmu, -cE1r(1)*inv_wmu);
                Complex Hx2(cE2i(0)*inv_wmu, -cE2r(0)*inv_wmu);
                Complex Hy2(cE2i(1)*inv_wmu, -cE2r(1)*inv_wmu);
                Complex det_H = Hx1*Hy2 - Hx2*Hy1;
                if (std::abs(det_H) < 1e-30) continue;
                Complex inv_det = 1.0/det_H;

                Complex Ex1(E1r(0), E1i(0)), Ey1(E1r(1), E1i(1));
                Complex Ex2(E2r(0), E2i(0)), Ey2(E2r(1), E2i(1));

                Complex Zxx = (Ex1*Hy2 - Ex2*Hy1)*inv_det;
                Complex Zxy = (Ex2*Hx1 - Ex1*Hx2)*inv_det;
                Complex Zyx = (Ey1*Hy2 - Ey2*Hy1)*inv_det;
                Complex Zyy = (Ey2*Hx1 - Ey1*Hx2)*inv_det;

                // δZ = (δE_mat - Z · δH_mat) · H⁻¹
                // For pol: δE_mat has one column, δH_mat has one column
                // H⁻¹ rows: [[Hy2,-Hx2],[-Hy1,Hx1]] / det_H
                Complex Hi00 = Hy2*inv_det, Hi01 = -Hx2*inv_det;
                Complex Hi10 = -Hy1*inv_det, Hi11 = Hx1*inv_det;

                Complex Ax, Ay;
                Complex dZxx, dZxy, dZyx, dZyy;

                if (pol == 0) {
                    Ax = dEx - Zxx*dHx - Zxy*dHy;
                    Ay = dEy - Zyx*dHx - Zyy*dHy;
                    dZxx = Ax*Hi00; dZxy = Ax*Hi01;
                    dZyx = Ay*Hi00; dZyy = Ay*Hi01;
                } else {
                    Ax = dEx - Zxx*dHx - Zxy*dHy;
                    Ay = dEy - Zyx*dHx - Zyy*dHy;
                    dZxx = Ax*Hi10; dZxy = Ax*Hi11;
                    dZyx = Ay*Hi10; dZyy = Ay*Hi11;
                }

                if (rank == 0 && s2 == 0) {
                    std::cout << "  J·v pol" << pol << " sta0: dZxy=("
                              << dZxy.real() << "," << dZxy.imag()
                              << ") dZyx=(" << dZyx.real() << "," << dZyx.imag() << ")\n";
                    std::cout << "    dE=(" << dEx.real() << "," << dEx.imag()
                              << "),(" << dEy.real() << "," << dEy.imag()
                              << ") dH=(" << dHx.real() << "," << dHx.imag()
                              << "),(" << dHy.real() << "," << dHy.imag() << ")\n";
                    std::cout << "    Z used: Zyx=(" << Zyx.real() << "," << Zyx.imag()
                              << ") det_H=(" << det_H.real() << "," << det_H.imag() << ")\n";
                }

                // Residuals
                const auto& obs2 = observed.observed(s2, 0);
                const auto& pred2 = predicted.predicted(s2, 0);
                Complex r_xx = obs2.Zxx.value - pred2.Zxx.value;
                Complex r_xy = obs2.Zxy.value - pred2.Zxy.value;
                Complex r_yx = obs2.Zyx.value - pred2.Zyx.value;
                Complex r_yy = obs2.Zyy.value - pred2.Zyy.value;

                // Apply weights
                int db = s2 * 8;
                r_xx = Complex(r_xx.real()*data_weights[db]*data_weights[db],
                               r_xx.imag()*data_weights[db+1]*data_weights[db+1]);
                r_xy = Complex(r_xy.real()*data_weights[db+2]*data_weights[db+2],
                               r_xy.imag()*data_weights[db+3]*data_weights[db+3]);
                r_yx = Complex(r_yx.real()*data_weights[db+4]*data_weights[db+4],
                               r_yx.imag()*data_weights[db+5]*data_weights[db+5]);
                r_yy = Complex(r_yy.real()*data_weights[db+6]*data_weights[db+6],
                               r_yy.imag()*data_weights[db+7]*data_weights[db+7]);

                // dΦ/dp = Re[δZ^* · W² r]  (note: r already weighted)
                g_Jv_pol += (std::conj(dZxx)*r_xx + std::conj(dZxy)*r_xy +
                             std::conj(dZyx)*r_yx + std::conj(dZyy)*r_yy).real();
            }
            if (rank == 0)
                std::cout << "  g_Jv_pol" << pol << " = " << g_Jv_pol << "\n";
        }
    }

    // ==================================================================
    // CROSS-CHECK: pert_rhs · λ_sys = gradient (correct for non-symmetric A)
    // A^T λ = adj_rhs, so g_j = λ^T · pert_rhs = pert_rhs · λ_sys
    // ==================================================================
    if (rank == 0) {
        std::cout << "\n=== pert_rhs · λ Cross-Check (elem " << test_e << ") ===\n";

        int tdof_n = solver.fespace()->GetTrueVSize();
        int sys_size = 2 * tdof_n;
        Real omega_ds = omega * sigma_bg;

        Real g_cross_total = 0.0;
        for (int pol = 0; pol < 2; ++pol) {
            const auto& E_r = (pol == 0) ? *solver.E1_real() : *solver.E2_real();
            const auto& E_i = (pol == 0) ? *solver.E1_imag() : *solver.E2_imag();
            const auto& lam_r = (pol == 0) ? l1r : l2r;
            const auto& lam_i = (pol == 0) ? l1i : l2i;

            // Build perturbation RHS (current convention)
            mfem::Vector pert_rhs(sys_size);
            pert_rhs = 0.0;

            mfem::ElementTransformation* T = pmesh.GetElementTransformation(test_e);
            const mfem::FiniteElement* fe = solver.fespace()->GetFE(test_e);
            mfem::Array<int> dofs;
            solver.fespace()->GetElementDofs(test_e, dofs);
            int ndof_e = dofs.Size();
            int order_q = 2 * fe->GetOrder() + T->OrderW();
            const mfem::IntegrationRule& ir =
                mfem::IntRules.Get(fe->GetGeomType(), order_q);

            for (int q = 0; q < ir.GetNPoints(); ++q) {
                const mfem::IntegrationPoint& ip = ir.IntPoint(q);
                T->SetIntPoint(&ip);
                Real w = ip.weight * T->Weight();
                mfem::Vector Er(3), Ei(3);
                E_r.GetVectorValue(*T, ip, Er);
                E_i.GetVectorValue(*T, ip, Ei);
                mfem::DenseMatrix vshape(ndof_e, 3);
                fe->CalcVShape(*T, vshape);
                for (int k = 0; k < ndof_e; ++k) {
                    int gdof = dofs[k];
                    int tdof_k = solver.fespace()->GetLocalTDofNumber(std::abs(gdof));
                    if (tdof_k < 0) continue;
                    Real sign = (gdof >= 0) ? 1.0 : -1.0;
                    Real Nk_dot_Er = 0, Nk_dot_Ei = 0;
                    for (int d = 0; d < 3; ++d) {
                        Nk_dot_Er += sign * vshape(k, d) * Er(d);
                        Nk_dot_Ei += sign * vshape(k, d) * Ei(d);
                    }
                    pert_rhs(tdof_k)           += w * (-omega_ds) * Nk_dot_Er;
                    pert_rhs(tdof_k + tdof_n)  += w * omega_ds * Nk_dot_Ei;
                }
            }

            // Construct λ in system convention [λ_r; λ_i]
            mfem::Vector lr_tdof(tdof_n), li_tdof(tdof_n);
            lam_r.GetTrueDofs(lr_tdof);
            lam_i.GetTrueDofs(li_tdof);
            mfem::Vector lam_sys(sys_size);
            for (int i = 0; i < tdof_n; ++i) {
                lam_sys(i)           = lr_tdof(i);   // block0 = λ_r
                lam_sys(i + tdof_n)  = li_tdof(i);   // block1 = λ_i
            }

            // g_j = pert_rhs · λ_sys
            Real g_cross_pol = pert_rhs * lam_sys;
            g_cross_total += g_cross_pol;
            std::cout << "  pol" << pol << ": pert_rhs · λ = " << g_cross_pol << "\n";

            // Also try with flipped pert_rhs blocks (original convention)
            mfem::Vector pert_rhs_flip(sys_size);
            pert_rhs_flip = 0.0;
            for (int q = 0; q < ir.GetNPoints(); ++q) {
                const mfem::IntegrationPoint& ip = ir.IntPoint(q);
                T->SetIntPoint(&ip);
                Real w = ip.weight * T->Weight();
                mfem::Vector Er2(3), Ei2(3);
                E_r.GetVectorValue(*T, ip, Er2);
                E_i.GetVectorValue(*T, ip, Ei2);
                mfem::DenseMatrix vshape2(ndof_e, 3);
                fe->CalcVShape(*T, vshape2);
                for (int k = 0; k < ndof_e; ++k) {
                    int gdof = dofs[k];
                    int tdof_k = solver.fespace()->GetLocalTDofNumber(std::abs(gdof));
                    if (tdof_k < 0) continue;
                    Real sign = (gdof >= 0) ? 1.0 : -1.0;
                    Real Nk_dot_Er2 = 0, Nk_dot_Ei2 = 0;
                    for (int d = 0; d < 3; ++d) {
                        Nk_dot_Er2 += sign * vshape2(k, d) * Er2(d);
                        Nk_dot_Ei2 += sign * vshape2(k, d) * Ei2(d);
                    }
                    pert_rhs_flip(tdof_k)           += w * omega_ds * Nk_dot_Ei2;
                    pert_rhs_flip(tdof_k + tdof_n)  += w * (-omega_ds) * Nk_dot_Er2;
                }
            }
            Real g_flip_pol = pert_rhs_flip * lam_sys;
            std::cout << "  pol" << pol << ": pert_rhs_flip · λ = " << g_flip_pol << "\n";
        }
        std::cout << "  TOTAL pert_rhs·λ = " << g_cross_total
                  << "  (should match g_fd=" << -0.00407 << ")\n";
    }

    // ==================================================================
    // DIAGNOSTIC: MFEM-assembled RHS vs manual assembly
    // Build the perturbation RHS using MFEM's ParLinearForm and compare.
    // ==================================================================
    if (rank == 0) {
        std::cout << "\n=== MFEM RHS vs Manual RHS Diagnostic (elem " << test_e << ") ===\n";
    }
    {
        int tdof = solver.fespace()->GetTrueVSize();
        int sys_size = 2 * tdof;
        int vsize = solver.fespace()->GetVSize();
        Real delta_sigma = sigma_bg;  // δ(log σ) = 1 → δσ = σ

        // Coefficient: -ω·δσ on test_e, 0 elsewhere
        mfem::Vector coeff_neg_wds(ne);
        coeff_neg_wds = 0.0;
        coeff_neg_wds(test_e) = -omega * delta_sigma;
        ElementCoefficient neg_wds_coeff(coeff_neg_wds);

        mfem::Vector coeff_pos_wds(ne);
        coeff_pos_wds = 0.0;
        coeff_pos_wds(test_e) = omega * delta_sigma;
        ElementCoefficient pos_wds_coeff(coeff_pos_wds);

        for (int pol = 0; pol < 2; ++pol) {
            auto& E_r = (pol == 0) ? *solver.E1_real() : *solver.E2_real();
            auto& E_i = (pol == 0) ? *solver.E1_imag() : *solver.E2_imag();

            // Block0 RHS = -ωδσ ∫ N · E_i dV  (via MFEM's ParLinearForm)
            ScaledGFVectorCoefficient neg_wds_Ei(neg_wds_coeff, E_i);
            mfem::ParLinearForm lf0(solver.fespace());
            lf0.AddDomainIntegrator(
                new mfem::VectorFEDomainLFIntegrator(neg_wds_Ei));
            lf0.Assemble();
            mfem::Vector b0_true(tdof);
            lf0.ParallelAssemble(b0_true);

            // Block1 RHS = +ωδσ ∫ N · E_r dV
            ScaledGFVectorCoefficient pos_wds_Er(pos_wds_coeff, E_r);
            mfem::ParLinearForm lf1(solver.fespace());
            lf1.AddDomainIntegrator(
                new mfem::VectorFEDomainLFIntegrator(pos_wds_Er));
            lf1.Assemble();
            mfem::Vector b1_true(tdof);
            lf1.ParallelAssemble(b1_true);

            // Combine into system-sized vector
            mfem::Vector rhs_mfem(sys_size);
            for (int i = 0; i < tdof; ++i) {
                rhs_mfem(i)        = b0_true(i);
                rhs_mfem(i + tdof) = b1_true(i);
            }

            // Re-assemble manual pert_rhs for comparison
            mfem::Vector pert_rhs_man(sys_size);
            pert_rhs_man = 0.0;
            {
                Real omega_ds = omega * delta_sigma;
                mfem::ElementTransformation* T =
                    pmesh.GetElementTransformation(test_e);
                const mfem::FiniteElement* fe =
                    solver.fespace()->GetFE(test_e);
                mfem::Array<int> dofs;
                solver.fespace()->GetElementDofs(test_e, dofs);
                int ndof_e = dofs.Size();
                int order_q = 2 * fe->GetOrder() + T->OrderW();
                const mfem::IntegrationRule& ir =
                    mfem::IntRules.Get(fe->GetGeomType(), order_q);

                for (int q = 0; q < ir.GetNPoints(); ++q) {
                    const mfem::IntegrationPoint& ip = ir.IntPoint(q);
                    T->SetIntPoint(&ip);
                    Real w = ip.weight * T->Weight();

                    mfem::Vector Er(3), Ei(3);
                    E_r.GetVectorValue(*T, ip, Er);
                    E_i.GetVectorValue(*T, ip, Ei);

                    mfem::DenseMatrix vshape(ndof_e, 3);
                    fe->CalcVShape(*T, vshape);

                    for (int k = 0; k < ndof_e; ++k) {
                        int gdof = dofs[k];
                        int tdof_k = solver.fespace()->GetLocalTDofNumber(
                            std::abs(gdof));
                        if (tdof_k < 0) continue;
                        Real sign = (gdof >= 0) ? 1.0 : -1.0;

                        Real Nk_dot_Ei = 0, Nk_dot_Er = 0;
                        for (int d = 0; d < 3; ++d) {
                            Nk_dot_Ei += sign * vshape(k, d) * Ei(d);
                            Nk_dot_Er += sign * vshape(k, d) * Er(d);
                        }
                        pert_rhs_man(tdof_k) +=
                            w * (-omega_ds) * Nk_dot_Ei;
                        pert_rhs_man(tdof_k + tdof) +=
                            w * omega_ds * Nk_dot_Er;
                    }
                }
            }

            // Compare
            mfem::Vector rhs_diff(sys_size);
            for (int i = 0; i < sys_size; ++i)
                rhs_diff(i) = rhs_mfem(i) - pert_rhs_man(i);
            Real norm_mfem = rhs_mfem.Norml2();
            Real norm_man  = pert_rhs_man.Norml2();
            Real norm_diff = rhs_diff.Norml2();

            if (rank == 0) {
                std::cout << "  pol" << pol
                          << ": |rhs_mfem|=" << norm_mfem
                          << " |rhs_manual|=" << norm_man
                          << " |diff|=" << norm_diff
                          << " rel=" << norm_diff / (norm_mfem + 1e-30) << "\n";

                // Print a few non-zero entries for comparison
                int cnt = 0;
                for (int i = 0; i < sys_size && cnt < 5; ++i) {
                    if (std::abs(rhs_mfem(i)) > 1e-15 ||
                        std::abs(pert_rhs_man(i)) > 1e-15) {
                        std::cout << "    dof " << i
                                  << ": mfem=" << rhs_mfem(i)
                                  << " manual=" << pert_rhs_man(i)
                                  << " ratio=" << pert_rhs_man(i)/(rhs_mfem(i)+1e-30) << "\n";
                        ++cnt;
                    }
                }
            }

            // Solve with MFEM-assembled RHS and compare δE at station
            mfem::ParGridFunction dE_r_mfem(solver.fespace());
            mfem::ParGridFunction dE_i_mfem(solver.fespace());
            solver.solve_forward_rhs(rhs_mfem, dE_r_mfem, dE_i_mfem);

            mfem::ParGridFunction dE_r_man(solver.fespace());
            mfem::ParGridFunction dE_i_man(solver.fespace());
            solver.solve_forward_rhs(pert_rhs_man, dE_r_man, dE_i_man);

            // Evaluate δE at station 0
            if (rank == 0) {
                mfem::DenseMatrix pts_diag(3, 1);
                pts_diag(0,0) = observed.station(0).x;
                pts_diag(1,0) = observed.station(0).y;
                pts_diag(2,0) = observed.station(0).z - 0.1;
                mfem::Array<int> eids_d;
                mfem::Array<mfem::IntegrationPoint> ips_d;
                pmesh.FindPoints(pts_diag, eids_d, ips_d);
                if (eids_d[0] >= 0) {
                    auto* T_d = pmesh.GetElementTransformation(eids_d[0]);
                    T_d->SetIntPoint(&ips_d[0]);

                    mfem::Vector dEr_mfem(3), dEi_mfem(3);
                    dE_r_mfem.GetVectorValue(*T_d, ips_d[0], dEr_mfem);
                    dE_i_mfem.GetVectorValue(*T_d, ips_d[0], dEi_mfem);

                    mfem::Vector dEr_man(3), dEi_man(3);
                    dE_r_man.GetVectorValue(*T_d, ips_d[0], dEr_man);
                    dE_i_man.GetVectorValue(*T_d, ips_d[0], dEi_man);

                    std::cout << "  δE at sta0 (MFEM RHS): dEr=("
                              << dEr_mfem(0) << "," << dEr_mfem(1) << ","
                              << dEr_mfem(2) << ") dEi=("
                              << dEi_mfem(0) << "," << dEi_mfem(1) << ","
                              << dEi_mfem(2) << ")\n";
                    std::cout << "  δE at sta0 (manual):   dEr=("
                              << dEr_man(0) << "," << dEr_man(1) << ","
                              << dEr_man(2) << ") dEi=("
                              << dEi_man(0) << "," << dEi_man(1) << ","
                              << dEi_man(2) << ")\n";
                }
            }
        }
    }

    // ==================================================================
    // DIAGNOSTIC: Matrix-based RHS (build A_pert, compute -(A_pert-A)*E)
    // ==================================================================
    if (rank == 0)
        std::cout << "\n=== Matrix-based RHS Verification (elem " << test_e << ") ===\n";
    {
        int tdof = solver.fespace()->GetTrueVSize();
        int sys_size = 2 * tdof;
        int vsize = solver.fespace()->GetVSize();

        // Recompute essential DOFs
        mfem::Array<int> ess_bdr(pmesh.bdr_attributes.Max());
        ess_bdr = 1;
        mfem::Array<int> ess_tdofs;
        solver.fespace()->GetEssentialTrueDofs(ess_bdr, ess_tdofs);

        // Build perturbed system matrix with σ_pert at test_e
        // Use small ε for linearization comparison
        Real small_eps = 1e-6;
        Real sigma_pert_e = sigma_bg * std::exp(small_eps);

        auto conv = mfem::ComplexOperator::HERMITIAN;
        mfem::ParSesquilinearForm a_pert(solver.fespace(), conv);
        mfem::ConstantCoefficient inv_mu0(1.0 / constants::MU0);
        a_pert.AddDomainIntegrator(
            new mfem::CurlCurlIntegrator(inv_mu0), NULL);

        mfem::Vector neg_ws_pert(ne);
        for (int i = 0; i < ne; ++i) {
            int attr = pmesh.GetAttribute(i);
            Real sig = (i == test_e) ? sigma_pert_e :
                       (attr == 2) ? 1e-6 : sigma_bg;
            neg_ws_pert(i) = -omega * sig;
        }
        ElementCoefficient coeff_pert(neg_ws_pert);
        a_pert.AddDomainIntegrator(
            NULL, new mfem::VectorFEMassIntegrator(coeff_pert));
        a_pert.Assemble();

        mfem::Vector x0(2*vsize), b0(2*vsize);
        x0 = 0.0; b0 = 0.0;
        mfem::OperatorHandle A_pert_op;
        mfem::Vector X_pert, B_pert;
        a_pert.FormLinearSystem(ess_tdofs, x0, b0, A_pert_op, X_pert, B_pert);

        auto* cAp = dynamic_cast<mfem::ComplexHypreParMatrix*>(A_pert_op.Ptr());
        std::unique_ptr<mfem::HypreParMatrix> A_pert_mat(cAp->GetSystemMatrix());

        // Build u_sys for pol1 and pol2
        for (int pol = 0; pol < 2; ++pol) {
            auto& E_r = (pol == 0) ? *solver.E1_real() : *solver.E2_real();
            auto& E_i = (pol == 0) ? *solver.E1_imag() : *solver.E2_imag();

            mfem::Vector er_tdof(tdof), ei_tdof(tdof);
            E_r.GetTrueDofs(er_tdof);
            E_i.GetTrueDofs(ei_tdof);
            mfem::Vector u_sys(sys_size);
            for (int i = 0; i < tdof; ++i) {
                u_sys(i)        = er_tdof(i);
                u_sys(i + tdof) = ei_tdof(i);
            }

            // rhs_from_matrix = -(A_pert - A) * u / small_eps
            mfem::Vector Au(sys_size), Apu(sys_size);
            solver.system_matrix()->Mult(u_sys, Au);
            A_pert_mat->Mult(u_sys, Apu);

            mfem::Vector rhs_mat(sys_size);
            for (int i = 0; i < sys_size; ++i)
                rhs_mat(i) = -(Apu(i) - Au(i)) / small_eps;

            // Manual RHS for δ(log σ) = 1 (same as above)
            mfem::Vector rhs_man(sys_size);
            rhs_man = 0.0;
            {
                Real omega_ds = omega * sigma_bg;
                mfem::ElementTransformation* T =
                    pmesh.GetElementTransformation(test_e);
                const mfem::FiniteElement* fe =
                    solver.fespace()->GetFE(test_e);
                mfem::Array<int> dofs;
                solver.fespace()->GetElementDofs(test_e, dofs);
                int ndof_e = dofs.Size();
                int order_q = 2 * fe->GetOrder() + T->OrderW();
                const mfem::IntegrationRule& ir =
                    mfem::IntRules.Get(fe->GetGeomType(), order_q);

                for (int q = 0; q < ir.GetNPoints(); ++q) {
                    const mfem::IntegrationPoint& ip = ir.IntPoint(q);
                    T->SetIntPoint(&ip);
                    Real w = ip.weight * T->Weight();
                    mfem::Vector Er(3), Ei(3);
                    E_r.GetVectorValue(*T, ip, Er);
                    E_i.GetVectorValue(*T, ip, Ei);
                    mfem::DenseMatrix vshape(ndof_e, 3);
                    fe->CalcVShape(*T, vshape);

                    for (int k = 0; k < ndof_e; ++k) {
                        int gdof = dofs[k];
                        int tdof_k = solver.fespace()->GetLocalTDofNumber(
                            std::abs(gdof));
                        if (tdof_k < 0) continue;
                        Real sign = (gdof >= 0) ? 1.0 : -1.0;
                        Real Nk_Ei = 0, Nk_Er = 0;
                        for (int d = 0; d < 3; ++d) {
                            Nk_Ei += sign * vshape(k, d) * Ei(d);
                            Nk_Er += sign * vshape(k, d) * Er(d);
                        }
                        rhs_man(tdof_k) +=
                            w * (-omega_ds) * Nk_Ei;
                        rhs_man(tdof_k + tdof) +=
                            w * omega_ds * Nk_Er;
                    }
                }
            }

            mfem::Vector diff(sys_size);
            for (int i = 0; i < sys_size; ++i)
                diff(i) = rhs_mat(i) - rhs_man(i);

            if (rank == 0) {
                std::cout << "  pol" << pol
                          << ": |rhs_matrix|=" << rhs_mat.Norml2()
                          << " |rhs_manual|=" << rhs_man.Norml2()
                          << " |diff|=" << diff.Norml2()
                          << " rel=" << diff.Norml2()/(rhs_mat.Norml2()+1e-30)
                          << "\n";

                // Print first few non-zero entries
                int cnt = 0;
                for (int i = 0; i < sys_size && cnt < 5; ++i) {
                    if (std::abs(rhs_mat(i)) > 1e-15 ||
                        std::abs(rhs_man(i)) > 1e-15) {
                        std::cout << "    dof " << i
                                  << ": matrix=" << rhs_mat(i)
                                  << " manual=" << rhs_man(i)
                                  << " ratio=" << rhs_man(i)/(rhs_mat(i)+1e-30) << "\n";
                        ++cnt;
                    }
                }
            }

            // Solve with matrix-derived RHS and evaluate δE at station
            mfem::ParGridFunction dE_r_m(solver.fespace());
            mfem::ParGridFunction dE_i_m(solver.fespace());
            solver.solve_forward_rhs(rhs_mat, dE_r_m, dE_i_m);

            if (rank == 0) {
                mfem::DenseMatrix pts_diag(3, 1);
                pts_diag(0,0) = observed.station(0).x;
                pts_diag(1,0) = observed.station(0).y;
                pts_diag(2,0) = observed.station(0).z - 0.1;
                mfem::Array<int> eids_d;
                mfem::Array<mfem::IntegrationPoint> ips_d;
                pmesh.FindPoints(pts_diag, eids_d, ips_d);
                if (eids_d[0] >= 0) {
                    auto* T_d = pmesh.GetElementTransformation(eids_d[0]);
                    T_d->SetIntPoint(&ips_d[0]);
                    mfem::Vector dEr(3), dEi(3);
                    dE_r_m.GetVectorValue(*T_d, ips_d[0], dEr);
                    dE_i_m.GetVectorValue(*T_d, ips_d[0], dEi);
                    std::cout << "  δE at sta0 (matrix RHS): dEr=("
                              << dEr(0) << "," << dEr(1) << "," << dEr(2)
                              << ") dEi=(" << dEi(0) << "," << dEi(1) << ","
                              << dEi(2) << ")\n";
                }
            }
        }
    }

    // Also check test_e != 0 (element 0 determines sigma_bg in FD)
    if (rank == 0 && test_e == 0) {
        std::cout << "\n*** WARNING: test_e == 0! FD test will change sigma_bg ***\n";
    }

    solver.release_factorization();

    // ==================================================================
    // FD: single element — also extract Z for δZ comparison
    // ==================================================================
    model::ConductivityModel mpp, mpm;
    mpp.init_3d(ne, sigma_bg); mpp.params()[test_e] += eps; mpp.invalidate_cache();
    mpm.init_3d(ne, sigma_bg); mpm.params()[test_e] -= eps; mpm.invalidate_cache();

    // Forward solve for mpp (perturbed +)
    data::MTData pred_p, pred_m;
    for (int s = 0; s < ns; ++s) {
        pred_p.add_station(observed.station(s));
        pred_m.add_station(observed.station(s));
    }
    pred_p.set_frequencies(observed.frequencies());
    pred_m.set_frequencies(observed.frequencies());

    // Keep solvers alive to extract E fields for δE comparison
    forward::ForwardSolver3D sol_p, sol_m;
    sol_p.setup(pmesh, mpp, fwd_params);
    sol_p.compute_single_frequency(0, observed, pred_p);

    sol_m.setup(pmesh, mpm, fwd_params);
    sol_m.compute_single_frequency(0, observed, pred_m);

    // Extract δE at station 0 from FD
    if (rank == 0) {
        mfem::DenseMatrix pts_fd(3, 1);
        pts_fd(0,0) = observed.station(0).x;
        pts_fd(1,0) = observed.station(0).y;
        pts_fd(2,0) = observed.station(0).z - 0.1;
        mfem::Array<int> eids_fd;
        mfem::Array<mfem::IntegrationPoint> ips_fd;
        pmesh.FindPoints(pts_fd, eids_fd, ips_fd);
        if (eids_fd[0] >= 0) {
            int elem_fd = eids_fd[0];
            auto& ip_fd = ips_fd[0];
            auto* T_fd = pmesh.GetElementTransformation(elem_fd);
            T_fd->SetIntPoint(&ip_fd);

            for (int pol = 0; pol < 2; ++pol) {
                auto& Epr = (pol==0) ? *sol_p.E1_real() : *sol_p.E2_real();
                auto& Epi = (pol==0) ? *sol_p.E1_imag() : *sol_p.E2_imag();
                auto& Emr = (pol==0) ? *sol_m.E1_real() : *sol_m.E2_real();
                auto& Emi = (pol==0) ? *sol_m.E1_imag() : *sol_m.E2_imag();

                mfem::Vector epr(3), epi(3), emr(3), emi(3);
                Epr.GetVectorValue(*T_fd, ip_fd, epr);
                Epi.GetVectorValue(*T_fd, ip_fd, epi);
                Emr.GetVectorValue(*T_fd, ip_fd, emr);
                Emi.GetVectorValue(*T_fd, ip_fd, emi);

                // δE_fd = (E+ - E-)/(2ε) per d(log σ)
                mfem::Vector dEr_fd(3), dEi_fd(3);
                for (int d = 0; d < 3; ++d) {
                    dEr_fd(d) = (epr(d) - emr(d)) / (2.0 * eps);
                    dEi_fd(d) = (epi(d) - emi(d)) / (2.0 * eps);
                }
                std::cout << "  FD dE pol" << pol << ": dEr=("
                          << dEr_fd(0) << "," << dEr_fd(1) << "," << dEr_fd(2)
                          << ") dEi=(" << dEi_fd(0) << "," << dEi_fd(1) << "," << dEi_fd(2) << ")\n";

                // Also print curl(δE)
                mfem::Vector cpr(3), cpi(3), cmr(3), cmi(3);
                Epr.GetCurl(*T_fd, cpr); Epi.GetCurl(*T_fd, cpi);
                Emr.GetCurl(*T_fd, cmr); Emi.GetCurl(*T_fd, cmi);
                mfem::Vector dcr(3), dci(3);
                for (int d = 0; d < 3; ++d) {
                    dcr(d) = (cpr(d) - cmr(d)) / (2.0 * eps);
                    dci(d) = (cpi(d) - cmi(d)) / (2.0 * eps);
                }
                std::cout << "           dcurlEr=(" << dcr(0) << "," << dcr(1) << ","
                          << dcr(2) << ") dcurlEi=(" << dci(0) << "," << dci(1) << "," << dci(2) << ")\n";
            }
        }
    }

    sol_p.release_factorization();
    sol_m.release_factorization();

    Real phi_p = compute_misfit(observed, pred_p, data_weights);
    Real phi_m = compute_misfit(observed, pred_m, data_weights);
    Real g_fd = (phi_p - phi_m) / (2.0 * eps);

    if (rank == 0) {
        std::cout << "\n=== FD Diagnostics ===\n"
                  << "  Phi0=" << phi0 << "  Phi+=" << phi_p << "  Phi-=" << phi_m << "\n"
                  << "  dPhi = " << (phi_p - phi_m) << "  g_fd = " << g_fd << "\n";

        // Compare δZ_fd with J·v δZ at station 0
        auto& zp = pred_p.predicted(0, 0);
        auto& zm = pred_m.predicted(0, 0);
        auto& z0 = predicted.predicted(0, 0);

        Complex dZxy_fd = (zp.Zxy.value - zm.Zxy.value) / (2.0 * eps);
        Complex dZyx_fd = (zp.Zyx.value - zm.Zyx.value) / (2.0 * eps);
        Complex dZxx_fd = (zp.Zxx.value - zm.Zxx.value) / (2.0 * eps);
        Complex dZyy_fd = (zp.Zyy.value - zm.Zyy.value) / (2.0 * eps);

        std::cout << "  dZxy_fd = (" << dZxy_fd.real() << ", " << dZxy_fd.imag() << ")\n"
                  << "  dZyx_fd = (" << dZyx_fd.real() << ", " << dZyx_fd.imag() << ")\n"
                  << "  dZxx_fd = (" << dZxx_fd.real() << ", " << dZxx_fd.imag() << ")\n"
                  << "  dZyy_fd = (" << dZyy_fd.real() << ", " << dZyy_fd.imag() << ")\n";

        // Print ALL Z components for baseline model
        std::cout << "  Z0: Zxx=(" << z0.Zxx.value.real() << "," << z0.Zxx.value.imag()
                  << ") Zxy=(" << z0.Zxy.value.real() << "," << z0.Zxy.value.imag()
                  << ") Zyx=(" << z0.Zyx.value.real() << "," << z0.Zyx.value.imag()
                  << ") Zyy=(" << z0.Zyy.value.real() << "," << z0.Zyy.value.imag() << ")\n";
        std::cout << "  Z+: Zyx=(" << zp.Zyx.value.real() << "," << zp.Zyx.value.imag() << ")\n";
        std::cout << "  Z-: Zyx=(" << zm.Zyx.value.real() << "," << zm.Zyx.value.imag() << ")\n";

        // Per-component FD gradients (using only that component's misfit)
        auto comp_phi = [&](const data::MTData& pred, int s_idx) -> std::array<Real, 4> {
            const auto& o = observed.observed(s_idx, 0);
            const auto& p = pred.predicted(s_idx, 0);
            int d = s_idx * 8;
            auto sq = [&](Complex r, int i) {
                return std::pow(r.real()*data_weights[d+i],2) +
                       std::pow(r.imag()*data_weights[d+i+1],2);
            };
            return {0.5*sq(o.Zxx.value-p.Zxx.value, 0),
                    0.5*sq(o.Zxy.value-p.Zxy.value, 2),
                    0.5*sq(o.Zyx.value-p.Zyx.value, 4),
                    0.5*sq(o.Zyy.value-p.Zyy.value, 6)};
        };

        std::cout << "  Per-component FD (station 0):\n";
        auto phi0_c = comp_phi(predicted, 0);
        auto phip_c = comp_phi(pred_p, 0);
        auto phim_c = comp_phi(pred_m, 0);
        const char* names[] = {"Zxx", "Zxy", "Zyx", "Zyy"};
        Real g_fd_s0_total = 0;
        for (int c = 0; c < 4; ++c) {
            Real gf = (phip_c[c] - phim_c[c]) / (2*eps);
            g_fd_s0_total += gf;
            std::cout << "    " << names[c] << ": phi0=" << phi0_c[c]
                      << " g_fd=" << gf << "\n";
        }
        // Also station 1
        auto phi0_c1 = comp_phi(predicted, 1);
        auto phip_c1 = comp_phi(pred_p, 1);
        auto phim_c1 = comp_phi(pred_m, 1);
        Real g_fd_s1_total = 0;
        std::cout << "  Per-component FD (station 1):\n";
        for (int c = 0; c < 4; ++c) {
            Real gf = (phip_c1[c] - phim_c1[c]) / (2*eps);
            g_fd_s1_total += gf;
            std::cout << "    " << names[c] << ": phi0=" << phi0_c1[c]
                      << " g_fd=" << gf << "\n";
        }
        std::cout << "  g_fd sta0=" << g_fd_s0_total << " sta1=" << g_fd_s1_total
                  << " total=" << (g_fd_s0_total+g_fd_s1_total) << "\n";
    }

    // Skip uniform perturbation for now — focus on per-element diagnostic
    Real g_fd_uniform = 0.0;

    if (rank == 0) {
        Real g_sum = std::accumulate(g_adj.begin(), g_adj.end(), 0.0);

        std::cout << "\n=== Single Element (elem " << test_e << ") ===\n"
                  << "  Adjoint: " << g_adj[test_e] << "\n"
                  << "  FD:      " << g_fd << "\n"
                  << "  Ratio:   " << g_adj[test_e] / g_fd << "\n\n";

        std::cout << "=== Uniform Perturbation ===\n"
                  << "  Sum(g_adj): " << g_sum << "\n"
                  << "  FD uniform: " << g_fd_uniform << "\n"
                  << "  Ratio:      " << g_sum / g_fd_uniform << "\n\n";
    }

    mfem::Mpi::Finalize();
    return 0;
}
