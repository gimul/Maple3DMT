// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#include "maple3dmt/forward/forward_solver_fv.h"
#include "maple3dmt/forward/cocg.h"
#include "maple3dmt/forward/bicgstab.h"
#include "maple3dmt/forward/cocr.h"
#include "maple3dmt/forward/ssor.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/utils/logger.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>

#define LOG_INFO(msg)  MAPLE3DMT_LOG_INFO(msg)
#define LOG_DEBUG(msg) MAPLE3DMT_LOG_DEBUG(msg)

namespace maple3dmt {
namespace forward {


// =========================================================================
// Setup
// =========================================================================
void ForwardSolverFV::setup(octree::OctreeMesh& mesh,
                             const ForwardParamsFV& params) {
    mesh_   = &mesh;
    params_ = params;

    // Use mesh params σ_bg as the 1D background conductivity
    sigma_bg_ = mesh.params().sigma_bg;
    if (sigma_bg_ <= 0) sigma_bg_ = 0.01;  // fallback

    // Build staggered grid if not already done
    if (mesh.staggered().num_edges() == 0)
        mesh.build_staggered_grid();

    // Build discrete operators
    ops_.build(mesh.staggered());

    LOG_INFO("ForwardSolverFV::setup — edges=" + std::to_string(ops_.num_edges()) +
             " faces=" + std::to_string(ops_.num_faces()) +
             " cells=" + std::to_string(ops_.num_cells()));
}

void ForwardSolverFV::set_sigma(const RealVec& sigma) {
    sigma_ = sigma;
}

void ForwardSolverFV::update_sigma(const model::ConductivityModel& model) {
    int nc = ops_.num_cells();
    sigma_.resize(nc);
    for (int c = 0; c < nc; ++c)
        sigma_[c] = model.sigma(c);
}

// =========================================================================
// Compute all frequency responses
// =========================================================================
void ForwardSolverFV::compute_responses(const data::MTData& observed,
                                          data::MTData& predicted) {
    // Map stations to cells
    find_stations_(observed);

    const auto& freqs = observed.frequencies();
    int nf = static_cast<int>(freqs.size());

    LOG_INFO("ForwardSolverFV: solving " + std::to_string(nf) + " frequencies");

    for (int fi = 0; fi < nf; ++fi) {
        if (freq_cb_) freq_cb_(fi, nf, "Forward");
        solve_frequency(freqs[fi], fi, observed, predicted);
    }

    LOG_INFO("ForwardSolverFV: all frequencies done");
}

// =========================================================================
// Single frequency solve (both polarizations)
// =========================================================================
void ForwardSolverFV::solve_frequency(Real freq_hz, int freq_idx,
                                        const data::MTData& observed,
                                        data::MTData& predicted) {
    Real omega = constants::TWOPI * freq_hz;
    current_omega_ = omega;

    LOG_INFO("  freq[" + std::to_string(freq_idx) + "] = " +
             std::to_string(freq_hz) + " Hz, omega=" + std::to_string(omega));

    // 1. Assemble system: A = C^T Mf^{-1} C + iω Me(σ)
    assemble_system_(omega);
    if (params_.div_correction) {
        compute_bc_flags_();
    }

    int ne = ops_.num_edges();
    const auto& sg = mesh_->staggered();

    // ---------------------------------------------------------------
    // TOTAL FIELD formulation:
    //   Solve A * E_total = 0 with Dirichlet BCs on all domain boundaries.
    //   BC = 1D analytical primary field E_primary(z):
    //     Air  (z >= 0): E = 1 (constant, no decay in air)
    //     Earth (z < 0): E = exp(ik*z), k = sqrt(-iωμσ_bg)
    //   This ensures correct field values at boundaries for all frequencies.
    // ---------------------------------------------------------------

    Real x_min = mesh_->params().domain_x_min;
    Real x_max = mesh_->params().domain_x_max;
    Real y_min = mesh_->params().domain_y_min;
    Real y_max = mesh_->params().domain_y_max;
    Real z_min = mesh_->params().domain_z_min;
    Real z_max = mesh_->params().domain_z_max;
    Real boundary_tol = mesh_->cell_size(0) * 0.1;

    // 1D primary wavenumber: k = sqrt(-iωμσ) with Im(k) > 0
    // ik = i*k used for E(z) = exp(ik*z) for z < 0
    Complex ik_bg = std::sqrt(Complex(0, omega * constants::MU0 * sigma_bg_));
    // Ensure decay direction: Re(ik) < 0 so exp(ik*z) decays for z < 0
    // k = sqrt(ωμσ/2)(1-i), ik = sqrt(ωμσ/2)(i+1) → Re(ik) > 0
    // For z < 0: ik*z has Re < 0 → decays ✓

    // Compute 1D primary field at a given z.
    //   Earth (z < 0): E = exp(α·z) where α = sqrt(iωμ₀σ), Re(α) > 0 → decays
    //   Air   (z ≥ 0): E = 1 + α·z  (derivative-continuous at z=0)
    //
    // The air field is NOT constant! dE/dz = α at z=0 (from Faraday: H continuous).
    // Using E=1 in air forces dE/dz≈0, which drags the surface amplitude down by
    // a factor |1/(1+α·z_max)|. For z_max=56km, 0.01Hz, 100Ωm → factor ≈ 0.42.
    auto E_primary = [&](Real z) -> Complex {
        if (z >= 0.0) return Complex(1.0, 0.0) + ik_bg * z;  // air: E = 1 + αz
        return std::exp(ik_bg * z);                            // earth: E = exp(αz)
    };

    // Identify boundary edges and set BC values.
    //
    // CRITICAL: All edges in the AIR region (z > 0) are also fixed as Dirichlet
    // BCs using the 1D primary field.  The curl-curl system has a gradient null
    // space that is regularised by iωσMe in the earth (σ > 0) but NOT in air
    // (σ ≈ 0).  Without this, the iterative solver produces unphysical field
    // amplification in air (|E| >> 1), creating a large spurious dE/dz at the
    // surface that inflates H and underestimates ρ_a.
    //
    // This mirrors the ModEM approach where the computational domain for E
    // only extends through the earth, with E = E_primary at the surface.
    std::vector<bool> is_bc(ne, false);
    ComplexVec bc_val_pol1(ne, Complex(0,0));
    ComplexVec bc_val_pol2(ne, Complex(0,0));

    int n_bc = 0, n_bc_nonzero = 0;
    int n_air_bc = 0;
    // Air Dirichlet: fix all air edges (z > threshold) to 1D primary field.
    // This eliminates the gradient null space in air (where σ≈0, so iω·Me≈0
    // provides no regularization). Without this, the iterative solver produces
    // unphysical |E| >> 1 in air, contaminating the surface field.
    //
    // Trade-off: kills E_scattered in air. For models where the anomaly extends
    // significantly above the surface (large/shallow bodies at low freq), this
    // can bias ρ_a by 10-30%. Grad-Div stabilization would be the proper fix
    // but requires careful implementation (discrete GG^T ≠ gradient projector).
    Real air_z_threshold = params_.air_z_threshold;

    for (int e = 0; e < ne; ++e) {
        const auto& ei = sg.edge(e);
        Real ex = ei.x, ey = ei.y, ez = ei.z;
        int d = ei.direction;

        bool on_boundary = false;

        // All 6 domain faces
        if (ex <= x_min + boundary_tol || ex >= x_max - boundary_tol ||
            ey <= y_min + boundary_tol || ey >= y_max - boundary_tol ||
            ez <= z_min + boundary_tol || ez >= z_max - boundary_tol) {
            on_boundary = true;
        }

        // Air edges: fix to 1D primary field to eliminate null space modes.
        // Edges exactly at z=0 are kept as free DOFs (surface anomaly).
        bool in_air = (ez > air_z_threshold);

        if (on_boundary || in_air) {
            Complex Ep = E_primary(ez);
            if (d == 0) bc_val_pol1[e] = Ep;  // Ex polarization
            if (d == 1) bc_val_pol2[e] = Ep;  // Ey polarization
            // z-directed edges: Ez = 0 for plane wave (default)
            is_bc[e] = true;
            ++n_bc;
            if (in_air && !on_boundary) ++n_air_bc;
            if (std::abs(Ep) > 1e-20 && d < 2) ++n_bc_nonzero;
        }
    }

    LOG_INFO("  BC edges: " + std::to_string(n_bc) + " / " + std::to_string(ne) +
             " (" + std::to_string(n_bc_nonzero) + " nonzero, "
             + std::to_string(n_air_bc) + " air-fixed)");

    // ---------------------------------------------------------------
    // SimPEG-style dead edge elimination:
    //   Dead faces already zeroed in Ce (handle_hanging_faces_).
    //   Dead edges have zero mass (build_edge_mass).
    //   → A[dead,dead] = 0. Set identity row/col, post-solve interpolate.
    // ---------------------------------------------------------------
    const auto& hanging_pairs = ops_.hanging_pairs();
    std::vector<bool> is_dead(ne, false);
    for (const auto& hp : hanging_pairs) is_dead[hp.coarse] = true;

    int n_dead = static_cast<int>(hanging_pairs.size());
    if (n_dead > 0)
        LOG_INFO("  Dead edges: " + std::to_string(n_dead) + " (SimPEG-style elimination)");

    // Constrained = BC or dead
    std::vector<bool> is_constrained(ne, false);
    for (int i = 0; i < ne; ++i)
        is_constrained[i] = is_bc[i] || is_dead[i];

    // Modify A in place: identity rows/cols for constrained DOFs
    struct BCEntry { int row; int k; Complex orig_val; };
    std::vector<BCEntry> saved_entries;
    saved_entries.reserve(ne * 2);

    for (int i = 0; i < ne; ++i) {
        for (int k = A_.rowptr[i]; k < A_.rowptr[i+1]; ++k) {
            int j = A_.colidx[k];
            if (is_constrained[i] || is_constrained[j]) {
                saved_entries.push_back({i, k, A_.values[k]});
                if (is_constrained[i])
                    A_.values[k] = (j == i) ? Complex(1,0) : Complex(0,0);
                else
                    A_.values[k] = Complex(0,0);
            }
        }
    }

    // SSOR preconditioner (symmetric Gauss-Seidel, omega=1)
    // Much stronger than Jacobi: 2x nnz cost but typically 3-5x fewer iterations
    SSORPreconditioner ssor_pc;
    ssor_pc.setup(A_, 1.0);
    LOG_INFO("  SSOR preconditioner setup (omega=1.0)");

    // Build RHS
    auto build_rhs = [&](const ComplexVec& E_bc) -> ComplexVec {
        ComplexVec rhs(ne, Complex(0,0));
        for (const auto& se : saved_entries) {
            if (is_constrained[se.row]) continue;
            int j = A_.colidx[se.k];
            if (is_bc[j]) rhs[se.row] -= se.orig_val * E_bc[j];
        }
        for (int i = 0; i < ne; ++i) {
            if (is_bc[i]) rhs[i] = E_bc[i];
        }
        return rhs;
    };

    // Post-solve: interpolate dead edges from children
    auto interpolate_hanging = [&](ComplexVec& E) {
        for (const auto& hp : hanging_pairs)
            E[hp.coarse] = Complex(0.5) * (E[hp.child1] + E[hp.child2]);
    };

    // BiCGStab + SSOR inner solve
    // Note: SSOR is non-symmetric → cannot use COCG/COCR with left-preconditioning.
    //       BiCGStab handles non-symmetric preconditioned systems correctly.
    auto inner_solve = [&](const ComplexVec& rhs, ComplexVec& E_sol) {
        BiCGStabSolver solver;
        solver.set_tolerance(params_.bicgstab_tol);
        solver.set_max_iterations(params_.bicgstab_maxiter);
        solver.set_print_level(params_.print_level);

        solver.set_operator([&](const ComplexVec& in, ComplexVec& out) {
            A_.matvec(in, out);
        });

        solver.set_preconditioner(ssor_pc.callback());

        auto result = solver.solve(rhs, E_sol);
        if (n_dead > 0) interpolate_hanging(E_sol);

        { char rb[64]; snprintf(rb, sizeof(rb), "%.3e", result.residual);
        LOG_INFO("  BiCGStab+SSOR: " + std::to_string(result.iterations) + " iters, " +
                 "res=" + std::string(rb) +
                 (result.converged ? " [OK]" : " [FAIL: " + result.info + "]")); }
    };

    // Build initial guess using 1D primary field
    auto init_guess = [&](ComplexVec& E_sol, const ComplexVec& E_bc, int pol_dir) {
        E_sol.resize(ne);
        for (int i = 0; i < ne; ++i) {
            const auto& ei = sg.edge(i);
            if (ei.direction == pol_dir) E_sol[i] = E_primary(ei.z);
            else E_sol[i] = Complex(0,0);
        }
        for (int i = 0; i < ne; ++i) {
            if (is_bc[i]) E_sol[i] = E_bc[i];
        }
    };

    // Solve with optional scattered field + DivCorr
    auto solve_pol = [&](const ComplexVec& E_bc, ComplexVec& E_sol, int pol_dir,
                         const std::string& pol_name) {
        LOG_INFO("  Solving " + pol_name + "...");
        ComplexVec rhs_total = build_rhs(E_bc);

        // ---------------------------------------------------------------
        // Scattered field formulation:
        //   E_total = E_primary + E_secondary
        //   A_mod · E_total = rhs_total
        //   ⟹ A_mod · E_sec = rhs_total - A_mod · E_pri  (=: rhs_sec)
        //
        //   rhs_sec ≈ 0 where σ = σ_bg → source localized to anomaly
        //   E_sec BC = 0 (E_pri already satisfies BC)
        //   Initial guess = 0 → much better conditioned
        // ---------------------------------------------------------------
        if (params_.scattered_field) {
            // 1. Compute E_pri everywhere from 1D analytical solution
            ComplexVec E_pri(ne);
            for (int i = 0; i < ne; ++i) {
                const auto& ei = sg.edge(i);
                if (ei.direction == pol_dir) E_pri[i] = E_primary(ei.z);
                else E_pri[i] = Complex(0,0);
            }

            // 2. rhs_sec = rhs_total - A_mod · E_pri
            ComplexVec A_Epri(ne);
            A_.matvec(E_pri, A_Epri);
            ComplexVec rhs_sec(ne);
            Real rhs_sec_norm = 0;
            for (int i = 0; i < ne; ++i) {
                rhs_sec[i] = rhs_total[i] - A_Epri[i];
                rhs_sec_norm += std::norm(rhs_sec[i]);
            }
            rhs_sec_norm = std::sqrt(rhs_sec_norm);

            Real rhs_total_norm = std::sqrt(
                [&]() { Real s=0; for (auto& v : rhs_total) s += std::norm(v); return s; }());

            { char buf1[64], buf2[64];
            snprintf(buf1, sizeof(buf1), "%.3e", rhs_total_norm);
            snprintf(buf2, sizeof(buf2), "%.3e", rhs_sec_norm);
            LOG_INFO("    Scattered field: ||rhs_total||=" + std::string(buf1) +
                     " ||rhs_sec||=" + std::string(buf2) +
                     " ratio=" + std::to_string(rhs_sec_norm / (rhs_total_norm + 1e-30))); }

            // 3. Solve for E_sec with zero initial guess.
            //    Forward solve uses FULL BiCGStab (not interleaved DivCorr).
            //    Reason: forward starts from E_sec=0 (cold start), so BiCGStab
            //    needs many iterations to converge. Interrupting with DivCorr
            //    every 500 iters causes stagnation and wastes time.
            //    Interleaved DivCorr is only used for adjoint/perturbation
            //    where warm-start is effective (ModEM does the same).
            ComplexVec E_sec(ne, Complex(0,0));
            inner_solve(rhs_sec, E_sec);

            // 4. E_total = E_pri + E_sec
            E_sol.resize(ne);
            for (int i = 0; i < ne; ++i) E_sol[i] = E_pri[i] + E_sec[i];
            for (int i = 0; i < ne; ++i) {
                if (is_bc[i]) E_sol[i] = E_bc[i];
            }

            // 5. Post-hoc DivCorr: remove gradient null space from E_total.
            //    Quick Poisson cleanup (no full re-solve).
            if (params_.div_correction && divcorr_setup_done_) {
                divcorr_poisson_(E_sol, nullptr);
            }

        } else {
            // Total field formulation
            LOG_INFO("    RHS: ||rhs||=" + std::to_string(
                std::sqrt([&]() { Real s=0; for (auto& v : rhs_total) s += std::norm(v); return s; }())));

            init_guess(E_sol, E_bc, pol_dir);
            inner_solve(rhs_total, E_sol);
            for (int i = 0; i < ne; ++i) {
                if (is_bc[i]) E_sol[i] = rhs_total[i];
            }

            // Post-hoc DivCorr
            if (params_.div_correction && divcorr_setup_done_) {
                divcorr_poisson_(E_sol, nullptr);
            }
        }
    };

    // 2. Solve both polarizations (with optional air BC iteration)
    //
    // Two-pass air BC update:
    //   Pass 1: air BC = E_primary (standard, kills E_scattered in air)
    //   Pass 2+: update air BC using E_scattered from previous solution,
    //            extrapolated vertically (E_scat(z) ≈ E_scat(0) for z close to 0).
    //
    // This corrects the bias from killing E_scattered in air, while still
    // maintaining null space control via Dirichlet BCs.
    int n_air_passes = std::max(1, params_.air_bc_iterations);

    // Track which edges are "air BC" (not domain boundary)
    std::vector<bool> is_air_bc(ne, false);
    for (int e = 0; e < ne; ++e) {
        if (!is_bc[e]) continue;
        const auto& ei = sg.edge(e);
        Real ez = ei.z;
        // Air BC = BC edges that are NOT on the domain boundary
        bool on_boundary =
            (ei.x <= x_min + boundary_tol || ei.x >= x_max - boundary_tol ||
             ei.y <= y_min + boundary_tol || ei.y >= y_max - boundary_tol ||
             ez <= z_min + boundary_tol || ez >= z_max - boundary_tol);
        if (!on_boundary && ez > air_z_threshold) {
            is_air_bc[e] = true;
        }
    }

    for (int air_pass = 0; air_pass < n_air_passes; ++air_pass) {
        if (air_pass > 0) {
            // ---- Update air BC from previous solution ----
            // For each surface-adjacent air edge, find corresponding surface edges
            // and set BC to E_primary(z) + E_scattered_surface.
            //
            // E_scattered is extracted from the earth-side solution at z=0.
            // For simplicity, each air edge at (x,y,z) uses the E_scattered
            // from the nearest edges at z=0 in the same direction.
            //
            // Since air z-coordinates are at discrete octree levels, we
            // collect E_scattered per direction as a spatial average of surface edges.

            // Build spatial map: (discretized x, y, direction) → E_scattered
            // Each air edge looks up the surface edge directly below it.
            Real h_min = mesh_->cell_size(mesh_->params().max_level);
            Real snap = h_min * 0.3;  // quantization for coordinate matching
            Real surface_tol = h_min * 0.6;

            struct SurfKey {
                int64_t ix, iy; int dir;
                bool operator==(const SurfKey& o) const {
                    return ix == o.ix && iy == o.iy && dir == o.dir;
                }
            };
            struct SurfKeyHash {
                size_t operator()(const SurfKey& k) const {
                    size_t h = std::hash<int64_t>()(k.ix);
                    h ^= std::hash<int64_t>()(k.iy) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    h ^= std::hash<int>()(k.dir) + 0x9e3779b9 + (h << 6) + (h >> 2);
                    return h;
                }
            };
            std::unordered_map<SurfKey, std::pair<Complex, Complex>, SurfKeyHash> surf_map;

            int n_surf = 0;
            for (int e = 0; e < ne; ++e) {
                const auto& ei = sg.edge(e);
                if (ei.direction >= 2) continue;
                if (is_bc[e]) continue;
                if (std::abs(ei.z) > surface_tol) continue;

                int d = ei.direction;
                SurfKey key{static_cast<int64_t>(std::round(ei.x / snap)),
                            static_cast<int64_t>(std::round(ei.y / snap)), d};

                Complex E_scat_p1 = E1_[e] - ((d == 0) ? E_primary(ei.z) : Complex(0,0));
                Complex E_scat_p2 = E2_[e] - ((d == 1) ? E_primary(ei.z) : Complex(0,0));
                surf_map[key] = {E_scat_p1, E_scat_p2};
                ++n_surf;
            }

            // Update air BC edges using nearest surface E_scattered
            // Under-relaxation to prevent feedback divergence:
            //   BC = E_primary + relax * E_scat * decay(z)
            Real decay_scale = 5000.0;  // meters (~anomaly half-width)
            Real relax = 0.5;  // under-relaxation factor
            int n_updated = 0, n_matched = 0;
            Real max_scat = 0;

            for (int e = 0; e < ne; ++e) {
                if (!is_air_bc[e]) continue;
                const auto& ei = sg.edge(e);
                int d = ei.direction;
                if (d >= 2) continue;

                SurfKey key{static_cast<int64_t>(std::round(ei.x / snap)),
                            static_cast<int64_t>(std::round(ei.y / snap)), d};

                auto it = surf_map.find(key);
                if (it == surf_map.end()) {
                    ++n_updated;
                    continue;  // keep BC = E_primary (default)
                }

                Real z_above = ei.z;
                Real decay = std::exp(-z_above / decay_scale);
                Complex scale(relax * decay, 0);
                Complex E_scat_p1 = it->second.first * scale;
                Complex E_scat_p2 = it->second.second * scale;

                if (d == 0) bc_val_pol1[e] = E_primary(z_above) + E_scat_p1;
                if (d == 1) bc_val_pol2[e] = E_primary(z_above) + E_scat_p2;

                max_scat = std::max(max_scat, std::abs(E_scat_p1));
                max_scat = std::max(max_scat, std::abs(E_scat_p2));
                ++n_updated;
                ++n_matched;
            }

            { char b1[64]; snprintf(b1, sizeof(b1), "%.6e", max_scat);
              LOG_INFO("  Air BC pass " + std::to_string(air_pass) +
                       ": " + std::to_string(n_matched) + "/" + std::to_string(n_updated) +
                       " matched, max|E_scat|=" + std::string(b1) +
                       " (surf_map=" + std::to_string(n_surf) + ")"); }

            // No matrix reassembly needed: build_rhs captures bc_val_pol1/2
            // by reference. The system matrix A_ only depends on WHICH edges
            // are BCs (unchanged), not on the BC VALUES.
        }

        std::string pass_suffix = (n_air_passes > 1)
            ? " [pass " + std::to_string(air_pass + 1) + "/" + std::to_string(n_air_passes) + "]"
            : "";

        solve_pol(bc_val_pol1, E1_, 0, "pol1 (Ex)" + pass_suffix);
        solve_pol(bc_val_pol2, E2_, 1, "pol2 (Ey)" + pass_suffix);
    }

    // NOTE: Do NOT restore A_ here. The BC-modified A_ (identity rows for
    // air/boundary/dead edges) is needed by adjoint_solve() which runs after
    // compute_single_frequency() in the gradient computation.
    // A_ will be reassembled from scratch at the next frequency (assemble_system_),
    // or cleared by release().

    // 4. Extract impedance at stations
    extract_impedance_(freq_idx, observed, predicted);
}

// =========================================================================
// System assembly: A = C^T Mf^{-1} C + iω Me(σ)
// =========================================================================
void ForwardSolverFV::assemble_system_(Real omega) {
    ops_.assemble_system(omega, sigma_, A_);
    LOG_DEBUG("  System assembled: " + std::to_string(A_.nrows) + " x " +
              std::to_string(A_.ncols) + ", nnz=" + std::to_string(A_.nnz()));
}

// =========================================================================
// Compute and cache BC flags + DivCorr Laplacian for current sigma.
// Called once per frequency (from factorize_frequency or solve_frequency).
// =========================================================================
void ForwardSolverFV::compute_bc_flags_() {
    const auto& sg = mesh_->staggered();
    int ne = ops_.num_edges();
    int nn = ops_.num_nodes();

    // --- BC flags (same logic as solve_frequency) ---
    const auto& mp = mesh_->params();
    Real x_min = mp.domain_x_min, x_max = mp.domain_x_max;
    Real y_min = mp.domain_y_min, y_max = mp.domain_y_max;
    Real z_min = mp.domain_z_min, z_max = mp.domain_z_max;
    Real boundary_tol = 1.0;
    Real air_z = params_.air_z_threshold;

    cached_is_bc_.assign(ne, false);
    for (int e = 0; e < ne; ++e) {
        if (ops_.is_dead_edge(e)) continue;
        const auto& ei = sg.edge(e);
        if (ei.x <= x_min + boundary_tol || ei.x >= x_max - boundary_tol ||
            ei.y <= y_min + boundary_tol || ei.y >= y_max - boundary_tol ||
            ei.z <= z_min + boundary_tol || ei.z >= z_max - boundary_tol ||
            ei.z > air_z) {
            cached_is_bc_[e] = true;
        }
    }

    // --- Boundary nodes (for DivCorr Dirichlet φ=0) ---
    cached_bc_nodes_.assign(nn, false);
    const auto& G = ops_.gradient_node();
    for (int e = 0; e < ne; ++e) {
        if (cached_is_bc_[e]) {
            for (int k = G.rowptr[e]; k < G.rowptr[e + 1]; ++k)
                cached_bc_nodes_[G.colidx[k]] = true;
        }
    }

    // --- Node Laplacian L = G^T Me_σ G ---
    ops_.build_div_laplacian(sigma_, cached_L_);

    // Extract diagonal for Jacobi preconditioner
    cached_L_diag_.resize(nn);
    for (int i = 0; i < nn; ++i) {
        cached_L_diag_[i] = 1.0;
        if (cached_bc_nodes_[i]) continue;
        for (int k = cached_L_.rowptr[i]; k < cached_L_.rowptr[i + 1]; ++k) {
            if (cached_L_.colidx[k] == i) { cached_L_diag_[i] = cached_L_.values[k]; break; }
        }
    }

    // Apply Dirichlet BC to L: boundary rows → identity, boundary columns → 0
    for (int i = 0; i < nn; ++i) {
        if (!cached_bc_nodes_[i]) continue;
        for (int k = cached_L_.rowptr[i]; k < cached_L_.rowptr[i + 1]; ++k)
            cached_L_.values[k] = (cached_L_.colidx[k] == i) ? 1.0 : 0.0;
    }
    for (int i = 0; i < nn; ++i) {
        if (cached_bc_nodes_[i]) continue;
        for (int k = cached_L_.rowptr[i]; k < cached_L_.rowptr[i + 1]; ++k) {
            if (cached_bc_nodes_[cached_L_.colidx[k]])
                cached_L_.values[k] = 0.0;
        }
    }

    divcorr_setup_done_ = true;
    LOG_DEBUG("  DivCorr setup cached: " + std::to_string(nn) + " nodes");
}

// =========================================================================
// DivCorr Poisson cleanup (ModEM-style, with optional source divergence).
//
// Target:  div(σE) − phi0 → 0
//   where phi0 = G^T · rhs / (iω)  for adjoint/perturbation solves,
//         phi0 = 0                   for forward (BC-only source).
//
// Algorithm:  L·φ = divJ − phi0,  E -= G·φ
//   where L = G^T·Me(σ)·G (node Laplacian, cached).
//   Since C·G = 0 (de Rham), curl(E) is unchanged by the correction.
// =========================================================================
void ForwardSolverFV::divcorr_poisson_(ComplexVec& E, const ComplexVec* phi0) {
    if (!divcorr_setup_done_) {
        LOG_INFO("  DivCorr: setup not done, skipping");
        return;
    }

    int nn = ops_.num_nodes();
    int ne = ops_.num_edges();
    int max_divcorr = params_.div_corr_iters;
    Real div_tol = params_.div_corr_tol;
    Real div_norm_initial = -1;

    for (int iter = 0; iter < max_divcorr; ++iter) {
        // 1. Compute div(σE) on nodes
        ComplexVec divJ;
        ops_.compute_div_sigma_E(E, sigma_, divJ);

        // 2. Subtract source divergence: divJ -= phi0
        if (phi0) {
            for (int i = 0; i < nn; ++i)
                divJ[i] -= (*phi0)[i];
        }

        // 3. Zero boundary nodes (φ=0 Dirichlet)
        for (int i = 0; i < nn; ++i)
            if (cached_bc_nodes_[i]) divJ[i] = Complex(0, 0);

        Real div_norm = 0;
        for (int i = 0; i < nn; ++i) div_norm += std::norm(divJ[i]);
        div_norm = std::sqrt(div_norm);
        if (iter == 0) div_norm_initial = div_norm;

        Real rel_div = (div_norm_initial > 1e-30) ? div_norm / div_norm_initial : 0;
        { char b1[64], b2[64]; snprintf(b1, sizeof(b1), "%.3e", div_norm); snprintf(b2, sizeof(b2), "%.3e", rel_div);
        LOG_INFO("  DivCorr iter=" + std::to_string(iter) +
                 ": ||div(σE)" + (phi0 ? "-phi0" : "") + "||=" + std::string(b1) +
                 " rel=" + std::string(b2)); }

        if ((rel_div < div_tol && iter > 0) || div_norm < 1e-12) {
            LOG_INFO("  DivCorr converged at iter " + std::to_string(iter));
            break;
        }

        // 4. CG solve: L·φ = divJ  (Jacobi preconditioned)
        ComplexVec phi(nn, Complex(0, 0));
        {
            ComplexVec r = divJ;
            ComplexVec z(nn), p(nn), Ap(nn);

            for (int i = 0; i < nn; ++i)
                z[i] = (std::abs(cached_L_diag_[i]) > 1e-30) ? r[i] / cached_L_diag_[i] : r[i];
            p = z;

            Complex rz(0, 0);
            for (int i = 0; i < nn; ++i) rz += std::conj(r[i]) * z[i];

            int cg_max = std::max(500, nn);
            for (int cg = 0; cg < cg_max; ++cg) {
                // Ap = L * p
                for (int i = 0; i < nn; ++i) {
                    Ap[i] = Complex(0, 0);
                    for (int k = cached_L_.rowptr[i]; k < cached_L_.rowptr[i + 1]; ++k)
                        Ap[i] += cached_L_.values[k] * p[cached_L_.colidx[k]];
                }

                Complex pAp(0, 0);
                for (int i = 0; i < nn; ++i) pAp += std::conj(p[i]) * Ap[i];
                if (std::abs(pAp) < 1e-30) break;

                Complex alpha = rz / pAp;
                for (int i = 0; i < nn; ++i) {
                    phi[i] += alpha * p[i];
                    r[i] -= alpha * Ap[i];
                }

                Real r_norm = 0;
                for (int i = 0; i < nn; ++i) r_norm += std::norm(r[i]);
                if (std::sqrt(r_norm) < 1e-10 * div_norm) break;

                for (int i = 0; i < nn; ++i)
                    z[i] = (std::abs(cached_L_diag_[i]) > 1e-30) ? r[i] / cached_L_diag_[i] : r[i];

                Complex rz_new(0, 0);
                for (int i = 0; i < nn; ++i) rz_new += std::conj(r[i]) * z[i];
                Complex beta = rz_new / (rz + Complex(1e-30, 0));
                for (int i = 0; i < nn; ++i)
                    p[i] = z[i] + beta * p[i];
                rz = rz_new;
            }
        }

        // 5. Correct E: E -= G·φ
        ComplexVec grad_phi;
        ops_.apply_cell_gradient(phi, grad_phi);
        for (int i = 0; i < ne; ++i) {
            if (!cached_is_bc_[i])
                E[i] -= grad_phi[i];
        }
    }
}

// =========================================================================
// Solve system with BiCGStab + SSOR (adjoint/perturbation solves)
// =========================================================================
void ForwardSolverFV::solve_system_(const ComplexVec& rhs, ComplexVec& x) {
    int n = A_.nrows;

    // SSOR preconditioner (non-symmetric → use BiCGStab, not COCG/COCR)
    SSORPreconditioner ssor;
    ssor.setup(A_, 1.0);

    BiCGStabSolver solver;
    Real tol = (tol_override_ > 0) ? tol_override_ : params_.bicgstab_tol;
    solver.set_tolerance(tol);
    solver.set_max_iterations(params_.bicgstab_maxiter);
    solver.set_print_level(params_.print_level);

    solver.set_operator([&](const ComplexVec& in, ComplexVec& out) {
        A_.matvec(in, out);
    });

    solver.set_preconditioner(ssor.callback());

    auto result = solver.solve(rhs, x);

    // Detect divergence: if residual grew beyond initial, solution is garbage
    if (!result.converged && result.residual > 1.0) {
        LOG_INFO("  BiCGStab+SSOR: DIVERGED (res=" + std::to_string(result.residual) +
                 "), zeroing solution");
        std::fill(x.begin(), x.end(), Complex(0, 0));
    } else {
        char rb[64]; snprintf(rb, sizeof(rb), "%.3e", result.residual);
        LOG_INFO("  BiCGStab+SSOR: " + std::to_string(result.iterations) + " iters, " +
                 "res=" + std::string(rb) +
                 (result.converged ? " [OK]" : " [FAIL: " + result.info + "]"));
    }
}

// =========================================================================
// Interleaved BiCGStab + DivCorr outer loop (ModEM-style).
//
// ModEM (Egbert & Kelbert 2012, EMsolve3D.f90 line 264-346):
//   loop:
//     QMR(b, E, maxiter=IterPerDivCor)   ← short Krylov burst (warm start)
//     SdivCorr(E, phi0)                  ← remove gradient null space
//     if ||A·E - b|| / ||b|| < tol: break
//   end loop
//
// Why this is faster: BiCGStab stalls when the gradient null space
// component of the error (invisible to curl-curl) grows.  DivCorr
// removes it cheaply (Poisson solve), giving BiCGStab a clean restart.
// Typical: 5000 iter [FAIL] → 100×5 outer = 500 iter [OK].
//
// phi0 = G^T · rhs / (iω)  — source divergence for adjoint/perturbation.
// =========================================================================
void ForwardSolverFV::solve_with_divcorr_(const ComplexVec& rhs, ComplexVec& x,
                                            const ComplexVec* phi0,
                                            const char* label) {
    int ne = A_.nrows;
    Real tol = (tol_override_ > 0) ? tol_override_ : params_.bicgstab_tol;
    int iter_per_dc = params_.divcorr_iter_per_dc;
    int max_outer = params_.divcorr_outer_max;

    // Compute ||rhs|| for relative residual
    Real rhs_norm = 0;
    for (int i = 0; i < ne; ++i) rhs_norm += std::norm(rhs[i]);
    rhs_norm = std::sqrt(rhs_norm);
    if (rhs_norm < 1e-30) rhs_norm = 1.0;

    // SSOR preconditioner (reused across outer iterations)
    SSORPreconditioner ssor;
    ssor.setup(A_, 1.0);

    // Initial DivCorr before Krylov iterations (ModEM does this too)
    if (phi0) {
        divcorr_poisson_(x, phi0);
    }

    int total_iter = 0;
    bool converged = false;
    Real prev_rel_res = 1e30;  // for stagnation detection

    for (int outer = 0; outer < max_outer; ++outer) {
        // --- Short BiCGStab burst (warm start from current x) ---
        BiCGStabSolver solver;
        solver.set_tolerance(tol);
        solver.set_max_iterations(iter_per_dc);
        solver.set_print_level(0);  // quiet inner iterations

        solver.set_operator([&](const ComplexVec& in, ComplexVec& out) {
            A_.matvec(in, out);
        });
        solver.set_preconditioner(ssor.callback());

        auto result = solver.solve(rhs, x);
        total_iter += result.iterations;

        // Detect catastrophic divergence
        if (result.residual > 1.0) {
            LOG_INFO("  " + std::string(label) + " outer=" + std::to_string(outer) +
                     ": BiCGStab DIVERGED (res=" + std::to_string(result.residual) +
                     "), zeroing");
            std::fill(x.begin(), x.end(), Complex(0, 0));
            break;
        }

        // Check if already converged
        if (result.converged) {
            char rb[64]; snprintf(rb, sizeof(rb), "%.3e", result.residual);
            LOG_INFO("  " + std::string(label) + ": " + std::to_string(total_iter) +
                     " iters (" + std::to_string(outer + 1) + " outer), res=" +
                     std::string(rb) + " [OK]");
            converged = true;
            break;
        }

        // --- DivCorr cleanup ---
        divcorr_poisson_(x, phi0);

        // --- Check true residual: ||A·x - rhs|| / ||rhs|| ---
        ComplexVec Ax(ne);
        A_.matvec(x, Ax);
        Real res_norm = 0;
        for (int i = 0; i < ne; ++i) res_norm += std::norm(Ax[i] - rhs[i]);
        res_norm = std::sqrt(res_norm);
        Real rel_res = res_norm / rhs_norm;

        { char rb[64]; snprintf(rb, sizeof(rb), "%.3e", rel_res);
        LOG_INFO("  " + std::string(label) + " outer=" + std::to_string(outer) +
                 ": " + std::to_string(total_iter) + " total iters, rel_res=" +
                 std::string(rb)); }

        if (rel_res < tol) {
            LOG_INFO("  " + std::string(label) + ": converged after DivCorr at outer=" +
                     std::to_string(outer) + ", total " + std::to_string(total_iter) +
                     " iters [OK]");
            converged = true;
            break;
        }

        // --- Stagnation detection ---
        // If residual didn't improve by >5% since last DivCorr cycle,
        // the outer loop is wasting time — accept current solution.
        if (outer > 0 && rel_res > 0.95 * prev_rel_res) {
            char rb[64]; snprintf(rb, sizeof(rb), "%.3e", rel_res);
            LOG_INFO("  " + std::string(label) + ": stagnated at outer=" +
                     std::to_string(outer) + " (res=" + std::string(rb) +
                     ", prev=" + std::to_string(prev_rel_res) +
                     "), accepting [" + std::to_string(total_iter) + " iters]");
            break;
        }
        prev_rel_res = rel_res;
    }

    if (!converged) {
        // Compute final residual
        ComplexVec Ax(ne);
        A_.matvec(x, Ax);
        Real res_norm = 0;
        for (int i = 0; i < ne; ++i) res_norm += std::norm(Ax[i] - rhs[i]);
        Real rel_res = std::sqrt(res_norm) / rhs_norm;
        char rb[64]; snprintf(rb, sizeof(rb), "%.3e", rel_res);
        LOG_INFO("  " + std::string(label) + ": " + std::to_string(total_iter) +
                 " total iters, res=" + std::string(rb) + " [FAIL: max outer DivCorr]");
    }
}

// =========================================================================
// Compute phi0 = G^T · rhs / (iω) for adjoint/perturbation DivCorr.
// =========================================================================
static void compute_phi0(const octree::DiscreteOperators& ops,
                          const ComplexVec& rhs, Real omega,
                          ComplexVec& phi0) {
    int nn = ops.num_nodes();
    Complex iw(0, omega);
    const auto& G = ops.gradient_node();
    phi0.assign(nn, Complex(0, 0));
    for (int e = 0; e < ops.num_edges(); ++e) {
        if (!ops.is_dead_edge(e)) {
            for (int k = G.rowptr[e]; k < G.rowptr[e + 1]; ++k)
                phi0[G.colidx[k]] += G.values[k] * rhs[e];
        }
    }
    for (int i = 0; i < nn; ++i)
        phi0[i] /= iw;
}

// =========================================================================
// Adjoint solve: A^T = A (complex symmetric)
// Uses interleaved BiCGStab + DivCorr (ModEM-style) when div_correction=true.
// =========================================================================
void ForwardSolverFV::adjoint_solve(const ComplexVec& rhs, ComplexVec& lambda) {
    lambda.assign(A_.nrows, Complex(0, 0));

    if (params_.div_correction && divcorr_setup_done_ && current_omega_ > 0) {
        ComplexVec phi0;
        compute_phi0(ops_, rhs, current_omega_, phi0);
        solve_with_divcorr_(rhs, lambda, &phi0, "Adjoint+DC");
    } else {
        solve_system_(rhs, lambda);
    }
}

// =========================================================================
// Forward solve with arbitrary RHS (for J·v perturbation).
// Uses interleaved BiCGStab + DivCorr (ModEM-style) when div_correction=true.
// =========================================================================
void ForwardSolverFV::solve_rhs(const ComplexVec& rhs, ComplexVec& solution) {
    solution.assign(A_.nrows, Complex(0, 0));

    if (params_.div_correction && divcorr_setup_done_ && current_omega_ > 0) {
        ComplexVec phi0;
        compute_phi0(ops_, rhs, current_omega_, phi0);
        solve_with_divcorr_(rhs, solution, &phi0, "Perturb+DC");
    } else {
        solve_system_(rhs, solution);
    }
}

// =========================================================================
// Build perturbation RHS: δb = -iω δσ Me_unit E_bg
// =========================================================================
void ForwardSolverFV::build_perturbation_rhs(int polarization,
                                               const RealVec& delta_sigma,
                                               ComplexVec& pert_rhs) {
    int ne = ops_.num_edges();
    int nc = ops_.num_cells();
    pert_rhs.assign(ne, Complex(0, 0));

    const ComplexVec& E_bg = (polarization == 1) ? E1_ : E2_;
    Complex iw(0, current_omega_);

    const auto& sg = mesh_->staggered();
    // delta_sigma is in σ-space (not log-σ).
    // Perturbation RHS: δb[e] = -iω Σ_c ∂Me[e]/∂σ_c · δσ_c · E[e]
    // where ∂Me[e]/∂σ_c = L³ · σ_avg² / (n_adj · σ_c²)
    for (int c = 0; c < nc; ++c) {
        if (std::abs(delta_sigma[c]) < 1e-30) continue;
        Real sigma_c = sigma_[c];

        for (const auto& ce : sg.cell_edges()[c]) {
            int e = ce.edge_id;
            Real L = sg.edge(e).length;
            Real edge_vol = L * L * L;

            // Harmonic mean for this edge
            const auto& adj = sg.edge(e).adj_cells;
            Real sum_inv = 0.0;
            int n_adj = 0;
            for (int ac : adj) {
                if (ac >= 0 && ac < nc && sigma_[ac] > 1e-30) {
                    sum_inv += 1.0 / sigma_[ac];
                    ++n_adj;
                }
            }
            Real sigma_avg = (n_adj > 0 && sum_inv > 1e-30) ? n_adj / sum_inv : 1e-8;
            Real dMe = edge_vol * sigma_avg * sigma_avg / (n_adj * sigma_c * sigma_c);

            pert_rhs[e] -= iw * Complex(delta_sigma[c] * dMe, 0) * E_bg[e];
        }
    }
}

// =========================================================================
// Sensitivity: g_cell = Re(iω Σ_edges conj(λ_e) * E_e * vol_fraction)
// =========================================================================
void ForwardSolverFV::compute_sensitivity(const ComplexVec& E_bg,
                                            const ComplexVec& lambda,
                                            RealVec& sensitivity) {
    int nc = ops_.num_cells();
    sensitivity.assign(nc, 0.0);

    Complex iw(0, current_omega_);
    const auto& sg = mesh_->staggered();

    // Sensitivity formula: ∂Φ/∂(ln σ_c) = Re(iω Σ_{e adj c} λ_e · E_e · ∂Me[e]/∂(ln σ_c))
    //
    // The edge mass matrix uses ARITHMETIC mean (see build_edge_mass):
    //   Me[e] = σ_avg · A_dual[e] · L[e]
    //   where σ_avg = Σσ_j / n_adj (arithmetic mean of adjacent cells)
    //
    // Derivative of arithmetic mean w.r.t. σ_c: ∂σ_avg/∂σ_c = 1/n_adj
    // So: ∂Me/∂σ_c = A_dual · L / n_adj
    // Chain rule for ln σ: ∂Me/∂(ln σ_c) = σ_c · ∂Me/∂σ_c = σ_c · A_dual · L / n_adj
    //
    // For complex symmetric A (Option B adjoint): uses λ (not conj(λ)).
    //
    // Determine constrained edges (same logic as solve_frequency BC setup).
    // These have identity rows in A_, so their sensitivity contribution is zero.
    Real air_z_threshold = params_.air_z_threshold;
    const auto& mp = mesh_->params();
    Real x_min = mp.domain_x_min, x_max = mp.domain_x_max;
    Real y_min = mp.domain_y_min, y_max = mp.domain_y_max;
    Real z_min = mp.domain_z_min, z_max = mp.domain_z_max;
    Real boundary_tol = 1.0;

    std::vector<bool> is_constrained(sg.num_edges(), false);
    for (int e = 0; e < sg.num_edges(); ++e) {
        if (ops_.is_dead_edge(e)) { is_constrained[e] = true; continue; }
        const auto& ei = sg.edge(e);
        if (ei.x <= x_min + boundary_tol || ei.x >= x_max - boundary_tol ||
            ei.y <= y_min + boundary_tol || ei.y >= y_max - boundary_tol ||
            ei.z <= z_min + boundary_tol || ei.z >= z_max - boundary_tol ||
            ei.z > air_z_threshold) {
            is_constrained[e] = true;
        }
    }

    for (int c = 0; c < nc; ++c) {
        Complex s(0, 0);
        Real sigma_c = sigma_[c];

        for (const auto& ce : sg.cell_edges()[c]) {
            int e = ce.edge_id;

            // Skip constrained edges (dead, boundary, air BC): Me contribution fixed
            if (is_constrained[e]) continue;

            Real L = sg.edge(e).length;
            Real A_dual = ops_.edge_dual_area(e);

            // Count adjacent cells for this edge
            const auto& adj = sg.edge(e).adj_cells;
            int n_adj = 0;
            for (int ac : adj) {
                if (ac >= 0 && ac < nc) ++n_adj;
            }
            if (n_adj == 0) continue;

            // ∂Me[e]/∂(ln σ_c) = σ_c · A_dual · L / n_adj
            Real dMe_dlnsigma = sigma_c * A_dual * L / n_adj;

            s += lambda[e] * E_bg[e] * dMe_dlnsigma;
        }
        sensitivity[c] = (iw * s).real();
    }
}

// =========================================================================
// Set background fields (for GN-CG caching)
// =========================================================================
void ForwardSolverFV::set_background_fields(const ComplexVec& E1,
                                              const ComplexVec& E2) {
    E1_ = E1;
    E2_ = E2;
}

void ForwardSolverFV::release() {
    A_ = SparseMatC{};
    cached_L_ = SparseMatR{};
    cached_L_diag_.clear();
    cached_is_bc_.clear();
    cached_bc_nodes_.clear();
    divcorr_setup_done_ = false;
}

// =========================================================================
// Primary field: 1D layered medium analytical solution
// =========================================================================
void ForwardSolverFV::compute_primary_field_(Real omega, int pol,
                                               ComplexVec& E0) {
    int ne = ops_.num_edges();
    E0.assign(ne, Complex(0, 0));

    // 1D plane wave in a halfspace with σ_bg.
    // For a uniform halfspace: E(z) = E0 * exp(-k*z) for z < 0 (earth)
    // where k = sqrt(iωμσ) = (1+i)/δ, δ = sqrt(2/(ωμσ))
    //
    // In air (z > 0): E = E0 (constant)
    // Boundary at z = 0.
    //
    // For pol=1: Ex = E_primary, Ey = 0
    // For pol=2: Ex = 0, Ey = E_primary

    // Use the stored background σ (from mesh params) for the 1D primary field.
    // This must match the σ_bg used in build_rhs_scattered_.
    Real sigma_bg = sigma_bg_;

    // k = sqrt(i * omega * mu0 * sigma)
    Complex ik = std::sqrt(Complex(0, omega * constants::MU0 * sigma_bg));

    const auto& edges = mesh_->staggered().edges();
    for (int e = 0; e < ne; ++e) {
        const auto& ei = edges[e];
        int d = ei.direction;

        // Only the horizontal component matching the polarization
        if ((pol == 1 && d != 0) || (pol == 2 && d != 1)) continue;

        Real z = ei.z;  // edge midpoint z-coordinate

        if (z >= 0) {
            // Air: E = 1 + α·z (derivative-continuous with earth at z=0)
            E0[e] = Complex(1.0, 0.0) + ik * z;
        } else {
            // Earth: E = exp(α·z) where α = sqrt(iωμ₀σ), decays for z < 0
            E0[e] = std::exp(ik * z);
        }
    }
}

// =========================================================================
// Build scattered field RHS: b = -iω (σ - σ_bg) Me_unit E_primary
// =========================================================================
void ForwardSolverFV::build_rhs_scattered_(Real omega, int pol,
                                             const ComplexVec& E0,
                                             ComplexVec& rhs) {
    int ne = ops_.num_edges();
    rhs.assign(ne, Complex(0, 0));

    // Use stored background σ (must match compute_primary_field_)
    Real sigma_bg = sigma_bg_;

    Complex iw(0, omega);
    const auto& sg = mesh_->staggered();
    int nc = ops_.num_cells();

    // Compute the scattered field RHS per-edge:
    //   rhs[e] = -iω * (Me[e](σ) - Me[e](σ_bg_model)) * E0[e]
    //
    // Me[e] = σ_avg_harmonic * L³  where σ_avg = n / Σ(1/σ_j)
    //
    // The background model has σ_bg for EARTH cells and σ_air for AIR cells.
    // Me_bg_model[e] uses the same harmonic mean but with earth cells = σ_bg.
    // This ensures dMe = 0 at air/earth boundaries (no spurious source).

    int n_rhs_nonzero = 0;
    for (int e = 0; e < ne; ++e) {
        if (std::abs(E0[e]) < 1e-30) continue;

        const auto& adj = sg.edge(e).adj_cells;

        // Compute σ_avg from actual model
        Real sum_inv_actual = 0.0;
        Real sum_inv_bg = 0.0;
        int n_adj = 0;
        for (int ac : adj) {
            if (ac < 0 || ac >= nc) continue;
            Real sig_actual = sigma_[ac];
            Real sig_bg_cell = (mesh_->cell_type(ac) != octree::CellType::AIR)
                               ? sigma_bg : sigma_[ac];  // air/ocean cells: same in both models
            if (sig_actual > 1e-30) sum_inv_actual += 1.0 / sig_actual;
            if (sig_bg_cell > 1e-30) sum_inv_bg += 1.0 / sig_bg_cell;
            ++n_adj;
        }

        if (n_adj == 0) continue;
        Real sigma_avg_actual = (sum_inv_actual > 1e-30) ? n_adj / sum_inv_actual : 1e-8;
        Real sigma_avg_bg     = (sum_inv_bg > 1e-30) ? n_adj / sum_inv_bg : 1e-8;

        Real L = sg.edge(e).length;
        Real dMe = (sigma_avg_actual - sigma_avg_bg) * L * L * L;

        if (std::abs(dMe) < 1e-30) continue;

        rhs[e] -= iw * Complex(dMe, 0) * E0[e];
        ++n_rhs_nonzero;
    }

    // Debug info
    Real rhs_norm = 0;
    for (int e = 0; e < ne; ++e) rhs_norm += std::norm(rhs[e]);
    rhs_norm = std::sqrt(rhs_norm);
    LOG_INFO("  scatter RHS: sigma_bg=" + std::to_string(sigma_bg) +
             " n_rhs_edges=" + std::to_string(n_rhs_nonzero) +
             " |rhs|=" + std::to_string(rhs_norm));

    (void)pol;
}

// =========================================================================
// Divergence correction (ModEM-style, Egbert & Kelbert 2012)
//
// Outer loop interleaving BiCGStab + gradient cleanup:
//   for k = 1 to max_divcorr:
//     E ← BiCGStab(A, b, E_prev)       (warm start from previous E)
//     divJ[c] = div(σE)
//     φ ← CG_solve(L, divJ)            (cell Laplacian, Dirichlet BC)
//     E ← E - grad(φ)                  (remove gradient null space)
//     if ||divJ|| / ||divJ₀|| < tol: break
// =========================================================================
void ForwardSolverFV::divergence_correction_(
        ComplexVec& E, const ComplexVec& rhs,
        const std::vector<bool>& is_bc, Real omega,
        std::function<void(const ComplexVec&, ComplexVec&)> inner_solve) {
    int nn = ops_.num_nodes();  // φ lives on NODES (not cells)
    int ne = ops_.num_edges();

    // Build node-based Laplacian: L = G^T * Me_σ * G  (nodes × nodes)
    SparseMatR L;
    ops_.build_div_laplacian(sigma_, L);

    // Identify boundary nodes (φ = 0 Dirichlet).
    // A node is on boundary if any of its edges is a BC edge endpoint.
    // We enforce φ=0 by zeroing RHS and setting diagonal to 1 for boundary nodes.
    Real x_min = mesh_->params().domain_x_min;
    Real x_max = mesh_->params().domain_x_max;
    Real y_min = mesh_->params().domain_y_min;
    Real y_max = mesh_->params().domain_y_max;
    Real z_min = mesh_->params().domain_z_min;
    Real z_max = mesh_->params().domain_z_max;
    Real bnd_tol = mesh_->cell_size(0) * 0.1;

    // We can identify boundary nodes by checking if any coordinate is at domain boundary.
    // But we don't store node coordinates directly in forward solver.
    // Alternative: mark nodes that belong to BC edges.
    std::vector<bool> is_bc_node(nn, false);
    // Any node connected to a BC edge is a boundary node
    // (Conservative: ensures φ=0 at all boundary edges)
    for (int e = 0; e < ne; ++e) {
        if (is_bc[e]) {
            // Mark both endpoint nodes
            const auto& G = ops_.gradient_node();
            for (int k = G.rowptr[e]; k < G.rowptr[e + 1]; ++k)
                is_bc_node[G.colidx[k]] = true;
        }
    }
    int n_bc_nodes = 0;
    for (int i = 0; i < nn; ++i) if (is_bc_node[i]) ++n_bc_nodes;
    LOG_INFO("  DivCorr: " + std::to_string(nn) + " nodes, " +
             std::to_string(n_bc_nodes) + " boundary nodes");

    // Extract diagonal of L for Jacobi preconditioner
    RealVec L_diag(nn, 1.0);
    for (int i = 0; i < nn; ++i) {
        if (is_bc_node[i]) { L_diag[i] = 1.0; continue; }
        for (int k = L.rowptr[i]; k < L.rowptr[i + 1]; ++k) {
            if (L.colidx[k] == i) { L_diag[i] = L.values[k]; break; }
        }
    }

    // Modify L for Dirichlet BC: boundary rows → identity
    for (int i = 0; i < nn; ++i) {
        if (!is_bc_node[i]) continue;
        for (int k = L.rowptr[i]; k < L.rowptr[i + 1]; ++k) {
            L.values[k] = (L.colidx[k] == i) ? 1.0 : 0.0;
        }
    }
    // Also zero boundary columns in non-boundary rows
    for (int i = 0; i < nn; ++i) {
        if (is_bc_node[i]) continue;
        for (int k = L.rowptr[i]; k < L.rowptr[i + 1]; ++k) {
            if (is_bc_node[L.colidx[k]])
                L.values[k] = 0.0;
        }
    }

    int max_divcorr = params_.div_corr_iters;
    Real div_tol = params_.div_corr_tol;
    Real div_norm_initial = -1;

    // DivCorr expects E to already contain a valid (approximate) solution.
    // The caller is responsible for the initial A*E = rhs solve.
    //
    // Repeated Poisson correction (NO re-solve of full system).
    //
    //   div(σE) = div(σ * E_true) + div(σ * grad(φ)) = 0 + L*φ
    //   Solve L*φ = div(σE), then E -= G*φ removes the gradient component.
    //
    //   Since C*G = 0 (DeRham), the corrected E still satisfies A*E ≈ rhs
    //   with residual increase O(ωμσ) (small at low frequency where this matters).
    for (int iter = 0; iter < max_divcorr; ++iter) {
        // Compute div(σE) on nodes = G^T * Me_σ * E
        ComplexVec divJ;
        ops_.compute_div_sigma_E(E, sigma_, divJ);

        // Zero boundary node divergence (φ=0 there)
        for (int i = 0; i < nn; ++i)
            if (is_bc_node[i]) divJ[i] = Complex(0, 0);

        Real div_norm = 0;
        for (int i = 0; i < nn; ++i) div_norm += std::norm(divJ[i]);
        div_norm = std::sqrt(div_norm);

        if (iter == 0) div_norm_initial = div_norm;

        Real rel_div = (div_norm_initial > 1e-30) ? div_norm / div_norm_initial : 0;
        { char b1[64], b2[64]; snprintf(b1, sizeof(b1), "%.3e", div_norm); snprintf(b2, sizeof(b2), "%.3e", rel_div);
        LOG_INFO("  DivCorr iter=" + std::to_string(iter) +
                 ": ||div(σE)||=" + std::string(b1) + " rel=" + std::string(b2)); }

        // Converge if relative divergence is small enough,
        // or if absolute divergence is already tiny.
        if ((rel_div < div_tol && iter > 0) || div_norm < 1e-12) {
            LOG_INFO("  DivCorr converged at iter " + std::to_string(iter));
            break;
        }

        // Solve Poisson L*φ = divJ using CG+Jacobi
        ComplexVec phi(nn, Complex(0, 0));
        {
            ComplexVec r = divJ;
            ComplexVec z(nn), p(nn), Ap(nn);

            for (int i = 0; i < nn; ++i)
                z[i] = (std::abs(L_diag[i]) > 1e-30) ? r[i] / L_diag[i] : r[i];
            p = z;

            Complex rz(0, 0);
            for (int i = 0; i < nn; ++i) rz += std::conj(r[i]) * z[i];

            Real r_norm0 = div_norm;

            int max_cg = std::max(500, nn);
            for (int cg = 0; cg < max_cg; ++cg) {
                Ap.assign(nn, Complex(0, 0));
                for (int i = 0; i < nn; ++i) {
                    Complex s(0, 0);
                    for (int k = L.rowptr[i]; k < L.rowptr[i + 1]; ++k)
                        s += L.values[k] * p[L.colidx[k]];
                    Ap[i] = s;
                }

                Complex pAp(0, 0);
                for (int i = 0; i < nn; ++i) pAp += std::conj(p[i]) * Ap[i];
                if (std::abs(pAp) < 1e-30) break;

                Complex alpha = rz / pAp;
                for (int i = 0; i < nn; ++i) {
                    phi[i] += alpha * p[i];
                    r[i] -= alpha * Ap[i];
                }

                Real r_norm = 0;
                for (int i = 0; i < nn; ++i) r_norm += std::norm(r[i]);
                r_norm = std::sqrt(r_norm);

                if (cg == 0 || (cg+1) % 200 == 0 || r_norm / (r_norm0 + 1e-30) < 1e-10)
                    LOG_INFO("    Poisson CG iter=" + std::to_string(cg+1) +
                             " rel_res=" + std::to_string(r_norm/(r_norm0+1e-30)));
                if (r_norm / (r_norm0 + 1e-30) < 1e-10) break;

                for (int i = 0; i < nn; ++i)
                    z[i] = (std::abs(L_diag[i]) > 1e-30) ? r[i] / L_diag[i] : r[i];

                Complex rz_new(0, 0);
                for (int i = 0; i < nn; ++i) rz_new += std::conj(r[i]) * z[i];
                if (std::abs(rz) < 1e-30) break;

                Complex beta = rz_new / rz;
                rz = rz_new;
                for (int i = 0; i < nn; ++i) p[i] = z[i] + beta * p[i];
            }
        }

        // E -= G*φ (remove gradient component)
        ComplexVec grad_phi;
        ops_.apply_cell_gradient(phi, grad_phi);

        Real grad_norm = 0, E_norm = 0;
        for (int e = 0; e < ne; ++e) { grad_norm += std::norm(grad_phi[e]); E_norm += std::norm(E[e]); }
        Real corr_ratio = std::sqrt(grad_norm / (E_norm + 1e-30));
        { char b1[64]; snprintf(b1, sizeof(b1), "%.3e", corr_ratio);
        LOG_INFO("    ||G*φ||/||E||=" + std::string(b1)); }

        for (int e = 0; e < ne; ++e)
            E[e] -= grad_phi[e];

        // Re-enforce BCs
        for (int i = 0; i < ne; ++i) {
            if (is_bc[i]) E[i] = rhs[i];
        }
    }

    (void)omega;
}

// =========================================================================
// Station location mapping
// =========================================================================
void ForwardSolverFV::find_stations_(const data::MTData& observed) {
    if (stations_mapped_) return;

    station_cells_.clear();
    for (int si = 0; si < observed.num_stations(); ++si) {
        const auto& sta = observed.station(si);
        // Find the cell containing this station by nearest-center search
        Real sx = sta.x, sy = sta.y, sz = 0.0;  // stations at surface

        // Find the shallowest EARTH cell that contains or is nearest to the station.
        // For surface stations (z=0), we want the first earth cell below the surface
        // at the station's (x,y) position.
        //
        // Strategy: among all earth cells whose horizontal extent covers the station,
        // pick the one with the smallest |cz| (closest to surface from below).
        // If none covers the station horizontally, fall back to nearest-center.
        int best_cell = -1;
        Real best_cz_abs = 1e30;      // for shallowest-cell search
        Real best_dist_xy = 1e30;     // for fallback
        int fallback_cell = -1;
        Real fallback_dist = 1e30;

        for (int c = 0; c < mesh_->num_cells_local(); ++c) {
            if (mesh_->cell_type(c) == octree::CellType::AIR) continue;

            Real cx, cy, cz;
            mesh_->cell_center(c, cx, cy, cz);
            Real hdx, hdy, hdz;
            mesh_->cell_size_xyz(c, hdx, hdy, hdz);

            // Check if station is within the horizontal extent of this cell
            bool in_x = (sx >= cx - hdx/2 - 1.0) && (sx <= cx + hdx/2 + 1.0);
            bool in_y = (sy >= cy - hdy/2 - 1.0) && (sy <= cy + hdy/2 + 1.0);

            if (in_x && in_y && cz < 0) {
                // Earth/Ocean cell below surface containing station horizontally
                // Pick the shallowest one (smallest |cz|)
                if (std::abs(cz) < best_cz_abs) {
                    best_cz_abs = std::abs(cz);
                    best_cell = c;
                }
            }

            // Fallback: 3D distance
            Real d = std::sqrt((cx-sx)*(cx-sx) + (cy-sy)*(cy-sy) + (cz-sz)*(cz-sz));
            if (d < fallback_dist) {
                fallback_dist = d;
                fallback_cell = c;
            }
        }

        if (best_cell < 0) best_cell = fallback_cell;

        // Debug: for center station, list all candidate cells (disabled for perf)
        if (false && (si == 0 || (std::abs(sx) < 100 && std::abs(sy) < 100))) {
            int n_candidates = 0;
            for (int c = 0; c < mesh_->num_cells_local(); ++c) {
                if (mesh_->cell_type(c) == octree::CellType::AIR) continue;
                Real cx2, cy2, cz2;
                mesh_->cell_center(c, cx2, cy2, cz2);
                Real hdx2, hdy2, hdz2;
                mesh_->cell_size_xyz(c, hdx2, hdy2, hdz2);
                bool ix = (sx >= cx2-hdx2/2-1) && (sx <= cx2+hdx2/2+1);
                bool iy = (sy >= cy2-hdy2/2-1) && (sy <= cy2+hdy2/2+1);
                if (ix && iy && cz2 < 0 && std::abs(cz2) < 10000) {
                    if (n_candidates < 10) {
                        char buf[256];
                        snprintf(buf, sizeof(buf),
                                 "    candidate c=%d cx=%.1f cy=%.1f cz=%.1f hdz=%.1f level=%d",
                                 c, cx2, cy2, cz2, hdz2, mesh_->cell_level(c));
                        LOG_INFO(std::string(buf));
                    }
                    ++n_candidates;
                }
            }
            LOG_INFO("  Station " + std::to_string(si) + " total candidates within 10km: " +
                     std::to_string(n_candidates) + " best_cell=" + std::to_string(best_cell));
        }

        if (best_cell >= 0) {
            Real cx, cy, cz;
            mesh_->cell_center(best_cell, cx, cy, cz);
            LOG_DEBUG("  Station " + std::to_string(si) + " → cell " +
                      std::to_string(best_cell) + " cz=" + std::to_string(cz) +
                      " h=" + std::to_string(mesh_->cell_size(best_cell)));
        }

        station_cells_.push_back({best_cell, 1.0, 1.0, 1.0});
    }

    stations_mapped_ = true;
    LOG_INFO("  Mapped " + std::to_string(station_cells_.size()) + " stations to cells");
}

// =========================================================================
// Impedance extraction: Z from E1, E2 at station locations
// =========================================================================
void ForwardSolverFV::extract_impedance_(int freq_idx,
                                           const data::MTData& observed,
                                           data::MTData& predicted) {
    // At each station, extract Z from E1, E2 (two polarizations).
    //
    // Z matrix:  [Ex1 Ex2] = [Zxx Zxy] [Hx1 Hx2]
    //            [Ey1 Ey2]   [Zyx Zyy] [Hy1 Hy2]
    //
    // H-field is computed from the vertical derivative of E:
    //   curl(E) = -iωμ₀H  (Faraday's law)
    //   Hy = -(1/(iωμ₀)) * ∂Ex/∂z
    //   Hx =  (1/(iωμ₀)) * ∂Ey/∂z
    //
    // The vertical derivative is computed from edges at the top and
    // bottom of the station cell.

    const auto& sg = mesh_->staggered();
    Real omega = current_omega_;
    Complex iwmu(0, omega * constants::MU0);

    station_H_cache_.resize(station_cells_.size());

    for (size_t si = 0; si < station_cells_.size(); ++si) {
        int cell_id = station_cells_[si].cell_id;
        if (cell_id < 0) continue;

        Real cx, cy, cz;
        mesh_->cell_center(cell_id, cx, cy, cz);
        // Use actual z-dimension for vertical derivative (not min of all dims)
        Real cdx, cdy, cdz;
        mesh_->cell_size_xyz(cell_id, cdx, cdy, cdz);
        // cdx, cdy, cdz used for full curl computation below

        // =================================================================
        // Collect ALL edge values for this cell, separated by direction
        // and position (top/bot for x,y-edges; +x/-x, +y/-y for z-edges)
        // =================================================================
        Complex Ex1_top(0,0), Ex1_bot(0,0), Ex2_top(0,0), Ex2_bot(0,0);
        Complex Ey1_top(0,0), Ey1_bot(0,0), Ey2_top(0,0), Ey2_bot(0,0);
        Complex Ez1_px(0,0), Ez1_mx(0,0), Ez2_px(0,0), Ez2_mx(0,0);  // z-edges at +x/-x
        Complex Ez1_py(0,0), Ez1_my(0,0), Ez2_py(0,0), Ez2_my(0,0);  // z-edges at +y/-y
        int nx_top=0, nx_bot=0, ny_top=0, ny_bot=0;
        int nz_px=0, nz_mx=0, nz_py=0, nz_my=0;

        for (const auto& ce : sg.cell_edges()[cell_id]) {
            int e = ce.edge_id;
            const auto& ei = sg.edge(e);
            if (ei.direction == 0) {
                if (ei.z > cz) {
                    Ex1_top += E1_[e]; Ex2_top += E2_[e]; ++nx_top;
                } else {
                    Ex1_bot += E1_[e]; Ex2_bot += E2_[e]; ++nx_bot;
                }
            } else if (ei.direction == 1) {
                if (ei.z > cz) {
                    Ey1_top += E1_[e]; Ey2_top += E2_[e]; ++ny_top;
                } else {
                    Ey1_bot += E1_[e]; Ey2_bot += E2_[e]; ++ny_bot;
                }
            } else if (ei.direction == 2) {
                // z-directed edges: classify by x and y position relative to center
                if (ei.x > cx) {
                    Ez1_px += E1_[e]; Ez2_px += E2_[e]; ++nz_px;
                } else {
                    Ez1_mx += E1_[e]; Ez2_mx += E2_[e]; ++nz_mx;
                }
                if (ei.y > cy) {
                    Ez1_py += E1_[e]; Ez2_py += E2_[e]; ++nz_py;
                } else {
                    Ez1_my += E1_[e]; Ez2_my += E2_[e]; ++nz_my;
                }
            }
        }

        // Average per group
        if (nx_top > 0) { Ex1_top /= Complex(nx_top,0); Ex2_top /= Complex(nx_top,0); }
        if (nx_bot > 0) { Ex1_bot /= Complex(nx_bot,0); Ex2_bot /= Complex(nx_bot,0); }
        if (ny_top > 0) { Ey1_top /= Complex(ny_top,0); Ey2_top /= Complex(ny_top,0); }
        if (ny_bot > 0) { Ey1_bot /= Complex(ny_bot,0); Ey2_bot /= Complex(ny_bot,0); }
        if (nz_px > 0) { Ez1_px /= Complex(nz_px,0); Ez2_px /= Complex(nz_px,0); }
        if (nz_mx > 0) { Ez1_mx /= Complex(nz_mx,0); Ez2_mx /= Complex(nz_mx,0); }
        if (nz_py > 0) { Ez1_py /= Complex(nz_py,0); Ez2_py /= Complex(nz_py,0); }
        if (nz_my > 0) { Ez1_my /= Complex(nz_my,0); Ez2_my /= Complex(nz_my,0); }

        // E at cell CENTER = average of top and bottom edges.
        // For a staggered grid, E and H at the cell center are collocated,
        // giving correct Z for uniform and layered models.
        Complex Ex1 = (nx_top > 0 && nx_bot > 0) ? (Ex1_top + Ex1_bot) * Complex(0.5,0)
                                                   : ((nx_top > 0) ? Ex1_top : Ex1_bot);
        Complex Ey1 = (ny_top > 0 && ny_bot > 0) ? (Ey1_top + Ey1_bot) * Complex(0.5,0)
                                                   : ((ny_top > 0) ? Ey1_top : Ey1_bot);
        Complex Ex2 = (nx_top > 0 && nx_bot > 0) ? (Ex2_top + Ex2_bot) * Complex(0.5,0)
                                                   : ((nx_top > 0) ? Ex2_top : Ex2_bot);
        Complex Ey2 = (ny_top > 0 && ny_bot > 0) ? (Ey2_top + Ey2_bot) * Complex(0.5,0)
                                                   : ((ny_top > 0) ? Ey2_top : Ey2_bot);

        // =================================================================
        // Full curl(E) at cell center for H computation:
        //   curl(E)_x = dEz/dy - dEy/dz  →  Hx = -curl(E)_x / (iωμ₀)
        //   curl(E)_y = dEx/dz - dEz/dx  →  Hy = -curl(E)_y / (iωμ₀)
        //
        // Previously we only used dEx/dz and dEy/dz. For 3D models with
        // lateral σ variations, Ez is nonzero and dEz/dx, dEz/dy contribute
        // significantly (galvanic effect at block boundaries).
        // =================================================================

        // Vertical derivatives (dEx/dz, dEy/dz)
        Complex dEx1_dz(0,0), dEx2_dz(0,0), dEy1_dz(0,0), dEy2_dz(0,0);
        if (nx_top > 0 && nx_bot > 0) {
            dEx1_dz = (Ex1_top - Ex1_bot) / cdz;
            dEx2_dz = (Ex2_top - Ex2_bot) / cdz;
        }
        if (ny_top > 0 && ny_bot > 0) {
            dEy1_dz = (Ey1_top - Ey1_bot) / cdz;
            dEy2_dz = (Ey2_top - Ey2_bot) / cdz;
        }

        // Horizontal derivatives of Ez (dEz/dx, dEz/dy)
        Complex dEz1_dx(0,0), dEz2_dx(0,0), dEz1_dy(0,0), dEz2_dy(0,0);
        if (nz_px > 0 && nz_mx > 0) {
            dEz1_dx = (Ez1_px - Ez1_mx) / cdx;
            dEz2_dx = (Ez2_px - Ez2_mx) / cdx;
        }
        if (nz_py > 0 && nz_my > 0) {
            dEz1_dy = (Ez1_py - Ez1_my) / cdy;
            dEz2_dy = (Ez2_py - Ez2_my) / cdy;
        }

        // H from Faraday's law: curl(E) = iωμ₀H  (e^{-iωt} convention)
        //   Hx = (dEz/dy - dEy/dz) / (iωμ₀)
        //   Hy = (dEx/dz - dEz/dx) / (iωμ₀)
        //
        // Full Faraday curl for H:
        //   Hx = (dEz/dy - dEy/dz) / (iωμ₀)
        //   Hy = (dEx/dz - dEz/dx) / (iωμ₀)
        // The dEz terms are essential for 3D models with lateral σ contrasts
        // (galvanic charges create vertical E at interfaces).
        // Full Faraday curl for H — dEz terms essential for 3D galvanic effects.
        // Removing dEz gives ρ≈3 at center (81° phase) — completely wrong.
        Complex Hx1 = (dEz1_dy - dEy1_dz) / iwmu;
        Complex Hy1 = (dEx1_dz - dEz1_dx) / iwmu;
        Complex Hx2 = (dEz2_dy - dEy2_dz) / iwmu;
        Complex Hy2 = (dEx2_dz - dEz2_dx) / iwmu;

        Complex det_H = Hx1 * Hy2 - Hx2 * Hy1;

        // Debug output (disabled for performance — enable for debugging)
        bool is_center = (std::abs(cx) < 2000 && std::abs(cy) < 2000);
        if (si < 2 && freq_idx == 0) {
            char buf[512];
            snprintf(buf, sizeof(buf), "  DEBUG station %zu: cell=%d cdz=%.1f cz=%.1f nx_top=%d nx_bot=%d ny_top=%d ny_bot=%d nz_px=%d nz_mx=%d",
                     si, cell_id, cdz, cz, nx_top, nx_bot, ny_top, ny_bot, nz_px, nz_mx);
            LOG_INFO(std::string(buf));
            snprintf(buf, sizeof(buf), "    Ex1=(%.6e,%.6e) Ey2=(%.6e,%.6e)",
                     Ex1.real(), Ex1.imag(), Ey2.real(), Ey2.imag());
            LOG_INFO(std::string(buf));
            snprintf(buf, sizeof(buf), "    Hy1=(%.6e,%.6e) Hx2=(%.6e,%.6e)",
                     Hy1.real(), Hy1.imag(), Hx2.real(), Hx2.imag());
            LOG_INFO(std::string(buf));
            snprintf(buf, sizeof(buf), "    Ex1_top=(%.6e,%.6e) Ex1_bot=(%.6e,%.6e)",
                     Ex1_top.real(), Ex1_top.imag(), Ex1_bot.real(), Ex1_bot.imag());
            LOG_INFO(std::string(buf));
            snprintf(buf, sizeof(buf), "    dEx1_dz=(%.6e,%.6e) dEz1_dx=(%.6e,%.6e) dEz2_dy=(%.6e,%.6e)",
                     dEx1_dz.real(), dEx1_dz.imag(), dEz1_dx.real(), dEz1_dx.imag(),
                     dEz2_dy.real(), dEz2_dy.imag());
            LOG_INFO(std::string(buf));
            // Compute Zxy locally for debug
            Complex Zxy_dbg(0,0);
            if (std::abs(det_H) > 1e-30)
                Zxy_dbg = (Ex2 * Hx1 - Ex1 * Hx2) / det_H;
            snprintf(buf, sizeof(buf), "    |det_H|=%.6e  Zxy=(%.6e,%.6e) |Zxy|=%.6e",
                     std::abs(det_H), Zxy_dbg.real(), Zxy_dbg.imag(), std::abs(Zxy_dbg));
            LOG_INFO(std::string(buf));
            Real rho_xy_dbg = std::norm(Zxy_dbg) / (omega * constants::MU0);
            Real phase_xy_dbg = std::arg(Zxy_dbg) * 180.0 / constants::PI;
            snprintf(buf, sizeof(buf), "    rho_xy=%.4f phase_xy=%.2f",
                     rho_xy_dbg, phase_xy_dbg);
            LOG_INFO(std::string(buf));

            if (si == 0) {
                int n_nonzero_E1 = 0;
                double max_E1 = 0;
                for (int e = 0; e < static_cast<int>(E1_.size()); ++e) {
                    if (std::abs(E1_[e]) > 1e-20) ++n_nonzero_E1;
                    max_E1 = std::max(max_E1, std::abs(E1_[e]));
                }
                snprintf(buf, sizeof(buf), "    E1: %d nonzero, max=%.6e",
                         n_nonzero_E1, max_E1);
                LOG_INFO(std::string(buf));

                // Count cell edges
                int n_cell_edges = static_cast<int>(sg.cell_edges()[cell_id].size());
                snprintf(buf, sizeof(buf), "    cell_edges count=%d", n_cell_edges);
                LOG_INFO(std::string(buf));
            }
        }

        // Cache H, Z, and cell dimensions for adjoint RHS
        station_H_cache_[si].Hx1 = Hx1;
        station_H_cache_[si].Hy1 = Hy1;
        station_H_cache_[si].Hx2 = Hx2;
        station_H_cache_[si].Hy2 = Hy2;
        station_H_cache_[si].det_H = det_H;
        station_H_cache_[si].cdx = cdx;
        station_H_cache_[si].cdy = cdy;
        station_H_cache_[si].cdz = cdz;

        // Z = E * H^{-1}
        Complex Zxx, Zxy, Zyx, Zyy;
        if (std::abs(det_H) > 1e-30) {
            Zxx = (Ex1 * Hy2 - Ex2 * Hy1) / det_H;
            Zxy = (Ex2 * Hx1 - Ex1 * Hx2) / det_H;
            Zyx = (Ey1 * Hy2 - Ey2 * Hy1) / det_H;
            Zyy = (Ey2 * Hx1 - Ey1 * Hx2) / det_H;
        } else {
            // Fallback: 1D halfspace impedance using local σ
            Real sigma_avg = sigma_[cell_id];
            Complex Z_1d = std::sqrt(iwmu / Complex(sigma_avg, 0));
            Zxx = Complex(0, 0);
            Zxy = Z_1d;
            Zyx = -Z_1d;
            Zyy = Complex(0, 0);
        }

        // Cache Z for adjoint H-contribution
        station_H_cache_[si].Zxx = Zxx;
        station_H_cache_[si].Zxy = Zxy;
        station_H_cache_[si].Zyx = Zyx;
        station_H_cache_[si].Zyy = Zyy;

        data::MTResponse resp;
        resp.Zxy.value = Zxy;
        resp.Zyx.value = Zyx;
        resp.Zxx.value = Zxx;
        resp.Zyy.value = Zyy;
        predicted.set_predicted(static_cast<int>(si), freq_idx, resp);
    }
}

// =========================================================================
// IForwardSolver interface: compute_single_frequency
// =========================================================================
void ForwardSolverFV::compute_single_frequency(int freq_idx,
                                                 const data::MTData& observed,
                                                 data::MTData& predicted) {
    find_stations_(observed);
    Real freq_hz = observed.frequencies()[freq_idx];
    solve_frequency(freq_hz, freq_idx, observed, predicted);
}

// =========================================================================
// IForwardSolver interface: factorize_frequency (assemble only)
// =========================================================================
void ForwardSolverFV::factorize_frequency(Real freq_hz) {
    Real omega = constants::TWOPI * freq_hz;
    current_omega_ = omega;
    assemble_system_(omega);
    if (params_.div_correction) {
        compute_bc_flags_();
    }
}

// =========================================================================
// IForwardSolver interface: release_factorization
// =========================================================================
void ForwardSolverFV::release_factorization() {
    release();
}

// =========================================================================
// IForwardSolver interface: set_background_fields_complex
// =========================================================================
void ForwardSolverFV::set_background_fields_complex(const ComplexVec& E1,
                                                      const ComplexVec& E2) {
    set_background_fields(E1, E2);
}

// =========================================================================
// IForwardSolver interface: adjoint_solve_complex
// =========================================================================
void ForwardSolverFV::adjoint_solve_complex(const ComplexVec& rhs,
                                              ComplexVec& lambda) {
    adjoint_solve(rhs, lambda);
}

// =========================================================================
// IForwardSolver interface: compute_sensitivity_complex
// =========================================================================
void ForwardSolverFV::compute_sensitivity_complex(const ComplexVec& E_bg,
                                                    const ComplexVec& lambda,
                                                    RealVec& sensitivity) {
    compute_sensitivity(E_bg, lambda, sensitivity);
}

// =========================================================================
// IForwardSolver interface: build_perturbation_rhs_complex
// =========================================================================
void ForwardSolverFV::build_perturbation_rhs_complex(int polarization,
                                                       const RealVec& delta_sigma,
                                                       ComplexVec& pert_rhs) {
    build_perturbation_rhs(polarization, delta_sigma, pert_rhs);
}

// =========================================================================
// IForwardSolver interface: solve_rhs_complex
// =========================================================================
void ForwardSolverFV::solve_rhs_complex(const ComplexVec& rhs,
                                          ComplexVec& solution) {
    solve_rhs(rhs, solution);
}

// =========================================================================
// Build adjoint RHS from weighted residual
// =========================================================================
void ForwardSolverFV::build_adjoint_rhs_from_residual(
    int freq_idx,
    const std::vector<std::array<Complex,4>>& weighted_residual,
    ComplexVec& adj_rhs_pol1,
    ComplexVec& adj_rhs_pol2) {

    int ne = ops_.num_edges();
    adj_rhs_pol1.assign(ne, Complex(0, 0));
    adj_rhs_pol2.assign(ne, Complex(0, 0));

    const auto& sg = mesh_->staggered();

    // For each station, map the weighted impedance residual back to edge DOFs.
    // Q^T operator: full adjoint of Z = E · H^{-1}.
    //
    // The impedance perturbation is:
    //   δZ = (δE - Z · δH) · H^{-1}
    //
    // where δH = curl(δE) / (iωμ₀). The adjoint has TWO contributions:
    //
    // (1) E-contribution (direct): adj_E = H^{-T} · wr
    //     Distributes to x,y edges of the station cell.
    //
    // (2) H-contribution (via curl): adj_H = -H^{-T} · Z^T · wr
    //     Maps back through the curl transpose to edges contributing to H.
    //     Hx = (dEz/dy - dEy/dz) / (iωμ₀)
    //     Hy = (dEx/dz - dEz/dx) / (iωμ₀)
    //
    // Both contributions are essential for correct adjoint gradients.

    Real omega = current_omega_;
    Complex iwmu(0, omega * constants::MU0);

    for (size_t si = 0; si < station_cells_.size(); ++si) {
        int cell_id = station_cells_[si].cell_id;
        if (cell_id < 0) continue;

        const auto& wr = weighted_residual[si];
        // wr = {W²·conj(r_xx), W²·conj(r_xy), W²·conj(r_yx), W²·conj(r_yy)}

        const auto& hc = station_H_cache_[si];
        Complex det_H = hc.det_H;
        if (std::abs(det_H) < 1e-30) continue;

        // H^{-T}: [Hy2, -Hy1; -Hx2, Hx1] / det_H
        //
        // WR matrix (component × polarization):
        //   WR = [wr_xx  wr_xy]    wr[0] wr[1]
        //        [wr_yx  wr_yy]    wr[2] wr[3]
        //
        // (1) E-contribution: adj_E = WR · H^{-T}
        //   [adj_E]_{comp,pol} = Σ_j WR_{comp,j} · [H^{-T}]_{j,pol}
        //
        // For pol1 (column 1 of H^{-T} = [Hy2; -Hx2] / det_H):
        //   adj_E_x_pol1 = (wr_xx·Hy2 - wr_xy·Hx2) / det_H
        //   adj_E_y_pol1 = (wr_yx·Hy2 - wr_yy·Hx2) / det_H
        //
        // For pol2 (column 2 of H^{-T} = [-Hy1; Hx1] / det_H):
        //   adj_E_x_pol2 = (-wr_xx·Hy1 + wr_xy·Hx1) / det_H
        //   adj_E_y_pol2 = (-wr_yx·Hy1 + wr_yy·Hx1) / det_H

        Complex src_x1 = ( wr[0] * hc.Hy2 - wr[1] * hc.Hx2) / det_H;
        Complex src_y1 = ( wr[2] * hc.Hy2 - wr[3] * hc.Hx2) / det_H;
        Complex src_x2 = (-wr[0] * hc.Hy1 + wr[1] * hc.Hx1) / det_H;
        Complex src_y2 = (-wr[2] * hc.Hy1 + wr[3] * hc.Hx1) / det_H;

        // (2) H-contribution: adj_H = -Z^T · WR · H^{-T}
        //   We can reuse the E-contribution: adj_H = -Z^T · adj_E
        //   adj_H_{comp,pol} = -Σ_i [Z^T]_{comp,i} · [adj_E]_{i,pol}
        //                    = -Σ_i Z_{i,comp} · adj_E_{i,pol}
        Complex adj_Hx1 = -(hc.Zxx * src_x1 + hc.Zyx * src_y1);
        Complex adj_Hy1 = -(hc.Zxy * src_x1 + hc.Zyy * src_y1);
        Complex adj_Hx2 = -(hc.Zxx * src_x2 + hc.Zyx * src_y2);
        Complex adj_Hy2 = -(hc.Zxy * src_x2 + hc.Zyy * src_y2);

        // Classify edges by direction and position (same as extract_impedance_)
        Real cx, cy, cz;
        mesh_->cell_center(cell_id, cx, cy, cz);
        Real cdx = hc.cdx, cdy = hc.cdy, cdz = hc.cdz;

        // Count edges per group
        int nx_top=0, nx_bot=0, ny_top=0, ny_bot=0;
        int nz_px=0, nz_mx=0, nz_py=0, nz_my=0;
        for (const auto& ce : sg.cell_edges()[cell_id]) {
            const auto& ei = sg.edge(ce.edge_id);
            if (ei.direction == 0) {
                if (ei.z > cz) ++nx_top; else ++nx_bot;
            } else if (ei.direction == 1) {
                if (ei.z > cz) ++ny_top; else ++ny_bot;
            } else if (ei.direction == 2) {
                if (ei.x > cx) ++nz_px; else ++nz_mx;
                if (ei.y > cy) ++nz_py; else ++nz_my;
            }
        }

        // --- Distribute E-contribution ---
        // E at center = (E_top + E_bot) / 2, so adjoint weight = 0.5 per group
        Real w_xt = (nx_top > 0) ? 0.5 / nx_top : 0.0;
        Real w_xb = (nx_bot > 0) ? 0.5 / nx_bot : 0.0;
        Real w_yt = (ny_top > 0) ? 0.5 / ny_top : 0.0;
        Real w_yb = (ny_bot > 0) ? 0.5 / ny_bot : 0.0;
        // If only one group exists, it gets full weight
        if (nx_top == 0) w_xb = (nx_bot > 0) ? 1.0 / nx_bot : 0.0;
        if (nx_bot == 0) w_xt = (nx_top > 0) ? 1.0 / nx_top : 0.0;
        if (ny_top == 0) w_yb = (ny_bot > 0) ? 1.0 / ny_bot : 0.0;
        if (ny_bot == 0) w_yt = (ny_top > 0) ? 1.0 / ny_top : 0.0;

        // --- Compute H-contribution adjoint weights ---
        // Transpose of curl computation at station:
        //   Hx = (dEz/dy - dEy/dz) / iwmu
        //   Hy = (dEx/dz - dEz/dx) / iwmu
        //
        // Derivatives use: dEx/dz = (Ex_top - Ex_bot) / cdz, etc.
        //
        // adj_Hx → edges via transpose:
        //   adj_Ez_py += adj_Hx / (iwmu * cdy), adj_Ez_my -= ...
        //   adj_Ey_top -= adj_Hx / (iwmu * cdz), adj_Ey_bot += ...
        //
        // adj_Hy → edges:
        //   adj_Ex_top += adj_Hy / (iwmu * cdz), adj_Ex_bot -= ...
        //   adj_Ez_px -= adj_Hy / (iwmu * cdx), adj_Ez_mx += ...

        // Pre-compute H-adjoint per edge group
        // H-contribution validated via gradient check (all 10 cells PASS)
        Complex inv_iwmu_cdz = Complex(1,0) / (iwmu * cdz);
        Complex inv_iwmu_cdx = Complex(1,0) / (iwmu * cdx);
        Complex inv_iwmu_cdy = Complex(1,0) / (iwmu * cdy);

        // --- Distribute to edges ---
        for (const auto& ce : sg.cell_edges()[cell_id]) {
            int e = ce.edge_id;
            const auto& ei = sg.edge(e);

            if (ei.direction == 0) {  // x-edges
                bool is_top = (ei.z > cz);
                // E-contribution
                Real we = is_top ? w_xt : w_xb;
                adj_rhs_pol1[e] += src_x1 * we;
                adj_rhs_pol2[e] += src_x2 * we;

                // H-contribution from adj_Hy: dEx/dz = (Ex_top - Ex_bot)/cdz
                // adj_Ex_top += adj_Hy / (iwmu * cdz)
                // adj_Ex_bot -= adj_Hy / (iwmu * cdz)
                {
                if (is_top && nx_top > 0) {
                    Complex wh = inv_iwmu_cdz / Real(nx_top);
                    adj_rhs_pol1[e] += adj_Hy1 * wh;
                    adj_rhs_pol2[e] += adj_Hy2 * wh;
                } else if (!is_top && nx_bot > 0) {
                    Complex wh = -inv_iwmu_cdz / Real(nx_bot);
                    adj_rhs_pol1[e] += adj_Hy1 * wh;
                    adj_rhs_pol2[e] += adj_Hy2 * wh;
                }
                }

            } else if (ei.direction == 1) {  // y-edges
                bool is_top = (ei.z > cz);
                // E-contribution
                Real we = is_top ? w_yt : w_yb;
                adj_rhs_pol1[e] += src_y1 * we;
                adj_rhs_pol2[e] += src_y2 * we;

                // H-contribution from adj_Hx: -dEy/dz in Hx
                // adj_Ey_top -= adj_Hx / (iwmu * cdz)
                // adj_Ey_bot += adj_Hx / (iwmu * cdz)
                {
                if (is_top && ny_top > 0) {
                    Complex wh = -inv_iwmu_cdz / Real(ny_top);
                    adj_rhs_pol1[e] += adj_Hx1 * wh;
                    adj_rhs_pol2[e] += adj_Hx2 * wh;
                } else if (!is_top && ny_bot > 0) {
                    Complex wh = inv_iwmu_cdz / Real(ny_bot);
                    adj_rhs_pol1[e] += adj_Hx1 * wh;
                    adj_rhs_pol2[e] += adj_Hx2 * wh;
                }
                }

            } else if (ei.direction == 2) {  // z-edges
                {
                // H-contribution only (z-edges don't contribute to E_mat)
                // From adj_Hx: dEz/dy = (Ez_py - Ez_my) / cdy
                bool is_py = (ei.y > cy);
                if (is_py && nz_py > 0) {
                    Complex wh = inv_iwmu_cdy / Real(nz_py);
                    adj_rhs_pol1[e] += adj_Hx1 * wh;
                    adj_rhs_pol2[e] += adj_Hx2 * wh;
                } else if (!is_py && nz_my > 0) {
                    Complex wh = -inv_iwmu_cdy / Real(nz_my);
                    adj_rhs_pol1[e] += adj_Hx1 * wh;
                    adj_rhs_pol2[e] += adj_Hx2 * wh;
                }

                // From adj_Hy: -dEz/dx = -(Ez_px - Ez_mx) / cdx
                bool is_px = (ei.x > cx);
                if (is_px && nz_px > 0) {
                    Complex wh = -inv_iwmu_cdx / Real(nz_px);
                    adj_rhs_pol1[e] += adj_Hy1 * wh;
                    adj_rhs_pol2[e] += adj_Hy2 * wh;
                } else if (!is_px && nz_mx > 0) {
                    Complex wh = inv_iwmu_cdx / Real(nz_mx);
                    adj_rhs_pol1[e] += adj_Hy1 * wh;
                    adj_rhs_pol2[e] += adj_Hy2 * wh;
                }
                }
            }
        }
    }

    (void)freq_idx;
}

// =========================================================================
// Extract delta impedance from perturbation field
// =========================================================================
void ForwardSolverFV::extract_delta_impedance_complex(
    const ComplexVec& dE, int polarization,
    std::vector<std::array<Complex,4>>& delta_Z) {

    delta_Z.resize(station_cells_.size());
    const auto& sg = mesh_->staggered();

    for (size_t si = 0; si < station_cells_.size(); ++si) {
        int cell_id = station_cells_[si].cell_id;
        delta_Z[si] = {Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0)};
        if (cell_id < 0) continue;

        // Average δE at station cell edges
        Complex dEx(0,0), dEy(0,0);
        int nx = 0, ny = 0;
        for (const auto& ce : sg.cell_edges()[cell_id]) {
            int e = ce.edge_id;
            const auto& ei = sg.edge(e);
            if (ei.direction == 0) { dEx += dE[e]; ++nx; }
            else if (ei.direction == 1) { dEy += dE[e]; ++ny; }
        }
        if (nx > 0) dEx /= Complex(nx, 0);
        if (ny > 0) dEy /= Complex(ny, 0);

        const auto& hc = station_H_cache_[si];
        if (std::abs(hc.det_H) < 1e-30) continue;

        // δZ = δE * H^{-1}
        // H^{-1} = [Hy2, -Hx2; -Hy1, Hx1] / det
        Complex inv_det = Complex(1, 0) / hc.det_H;

        if (polarization == 1) {
            // Pol1 contributes to column 1 of Z = E * H^{-1}
            // δZ_xx += δEx * Hy2 / det,  δZ_xy += δEx * (-Hx2) / det
            // δZ_yx += δEy * Hy2 / det,  δZ_yy += δEy * (-Hx2) / det
            delta_Z[si][0] = dEx * hc.Hy2 * inv_det;   // δZxx
            delta_Z[si][1] = dEx * (-hc.Hx2) * inv_det; // δZxy
            delta_Z[si][2] = dEy * hc.Hy2 * inv_det;   // δZyx
            delta_Z[si][3] = dEy * (-hc.Hx2) * inv_det; // δZyy
        } else {
            // Pol2 contributes to column 2 of Z = E * H^{-1}
            delta_Z[si][0] = dEx * (-hc.Hy1) * inv_det;
            delta_Z[si][1] = dEx * hc.Hx1 * inv_det;
            delta_Z[si][2] = dEy * (-hc.Hy1) * inv_det;
            delta_Z[si][3] = dEy * hc.Hx1 * inv_det;
        }
    }
}

} // namespace forward
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
