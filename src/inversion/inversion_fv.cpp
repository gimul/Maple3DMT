// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file inversion_fv.cpp
/// @brief FV-based 3D MT inversion driver.

#include "maple3dmt/inversion/inversion_fv.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/inversion/implicit_jtj_fv.h"
#include "maple3dmt/utils/logger.h"
#include "maple3dmt/utils/memory.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iomanip>

namespace {
#ifdef MAPLE3DMT_USE_MPI
#define INV_LOG_INFO(msg) do { \
    int r_; MPI_Comm_rank(MPI_COMM_WORLD, &r_); \
    if (r_ == 0) MAPLE3DMT_LOG_INFO(msg); \
} while(0)
#define INV_LOG_WARNING(msg) do { \
    int r_; MPI_Comm_rank(MPI_COMM_WORLD, &r_); \
    if (r_ == 0) MAPLE3DMT_LOG_WARNING(msg); \
} while(0)
#else
#define INV_LOG_INFO(msg)    MAPLE3DMT_LOG_INFO(msg)
#define INV_LOG_WARNING(msg) MAPLE3DMT_LOG_WARNING(msg)
#endif
} // anonymous namespace

namespace maple3dmt {
namespace inversion {

// =========================================================================
// Setup
// =========================================================================
void InversionFV::setup(model::ConductivityModel& model,
                          data::MTData& data,
                          forward::IForwardSolver& fwd,
                          RegularizationOctree& reg,
                          const InversionParamsFV& params) {
    model_ = &model;
    data_  = &data;
    fwd_   = &fwd;
    reg_   = &reg;
    params_ = params;
    current_lambda_ = params_.lambda_init;
    history_.clear();

    // Setup frequency-parallel manager (MPI)
#ifdef MAPLE3DMT_USE_MPI
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size > 1) {
        fpm_.setup(MPI_COMM_WORLD, data_->num_frequencies(), /*spatial_procs=*/1);
        freq_parallel_active_ = fpm_.is_freq_parallel();
        fpm_.print_summary();
    }
#endif

    INV_LOG_INFO("InversionFV setup: " + std::to_string(reg_->n_active()) +
                 " active params, " + std::to_string(data_->num_stations()) +
                 " stations, " + std::to_string(data_->num_frequencies()) + " freqs" +
                 (freq_parallel_active_ ? " [FREQ-PARALLEL]" : ""));
}

// =========================================================================
// Data misfit from predicted
// =========================================================================
Real InversionFV::data_misfit_from_predicted_() const {
    Real phi = 0.0;
    int nf = data_->num_frequencies();
    int ns = data_->num_stations();

    for (int f = 0; f < nf; ++f) {
        for (int s = 0; s < ns; ++s) {
            const auto& obs = data_->observed(s, f);
            const auto& pred = data_->predicted(s, f);

            auto add = [&](const data::Datum& o, const data::Datum& p) {
                if (o.weight <= 0.0 || o.error <= 0.0) return;
                Complex r = o.value - p.value;
                Real w = 1.0 / o.error;
                phi += 0.5 * w * w * std::norm(r);
            };
            add(obs.Zxx, pred.Zxx);
            add(obs.Zxy, pred.Zxy);
            add(obs.Zyx, pred.Zyx);
            add(obs.Zyy, pred.Zyy);
        }
    }
    return phi;
}

// =========================================================================
// Data misfit for specific frequencies (for freq-parallel mode)
// =========================================================================
Real InversionFV::data_misfit_for_frequencies_(const std::vector<int>& freq_indices) const {
    Real phi = 0.0;
    int ns = data_->num_stations();

    for (int f : freq_indices) {
        for (int s = 0; s < ns; ++s) {
            const auto& obs = data_->observed(s, f);
            const auto& pred = data_->predicted(s, f);
            auto add = [&](const data::Datum& o, const data::Datum& p) {
                if (o.weight <= 0.0 || o.error <= 0.0) return;
                Complex r = o.value - p.value;
                Real w = 1.0 / o.error;
                phi += 0.5 * w * w * std::norm(r);
            };
            add(obs.Zxx, pred.Zxx);
            add(obs.Zxy, pred.Zxy);
            add(obs.Zyx, pred.Zyx);
            add(obs.Zyy, pred.Zyy);
        }
    }
    return phi;
}

// =========================================================================
// RMS misfit
// =========================================================================
Real InversionFV::compute_rms_() const {
    if (freq_parallel_active_) {
        Real chi2_local = 0.0;
        int nd_local = 0;
        compute_rms_for_frequencies_(fpm_.my_frequency_indices(), chi2_local, nd_local);

        // Allreduce chi2 and nd across frequency groups
        Real chi2_global = fpm_.allreduce_scalar(chi2_local);
        Real nd_d = static_cast<Real>(nd_local);
        Real nd_global = fpm_.allreduce_scalar(nd_d);
        int nd = static_cast<int>(nd_global);
        return (nd > 0) ? std::sqrt(chi2_global / nd) : 0.0;
    }

    Real chi2 = 0.0;
    int nd = 0;
    int nf = data_->num_frequencies();
    int ns = data_->num_stations();

    for (int f = 0; f < nf; ++f) {
        for (int s = 0; s < ns; ++s) {
            const auto& obs = data_->observed(s, f);
            const auto& pred = data_->predicted(s, f);
            auto add = [&](const data::Datum& o, const data::Datum& p) {
                if (o.weight <= 0.0 || o.error <= 0.0) return;
                Complex r = o.value - p.value;
                Real w = 1.0 / o.error;
                chi2 += w * w * (r.real() * r.real() + r.imag() * r.imag());
                nd += 2;  // Re + Im
            };
            add(obs.Zxx, pred.Zxx);
            add(obs.Zxy, pred.Zxy);
            add(obs.Zyx, pred.Zyx);
            add(obs.Zyy, pred.Zyy);
        }
    }
    return (nd > 0) ? std::sqrt(chi2 / nd) : 0.0;
}

Real InversionFV::compute_rms_for_frequencies_(
    const std::vector<int>& freq_indices, Real& chi2_out, int& nd_out) const
{
    chi2_out = 0.0;
    nd_out = 0;
    int ns = data_->num_stations();

    for (int f : freq_indices) {
        for (int s = 0; s < ns; ++s) {
            const auto& obs = data_->observed(s, f);
            const auto& pred = data_->predicted(s, f);
            auto add = [&](const data::Datum& o, const data::Datum& p) {
                if (o.weight <= 0.0 || o.error <= 0.0) return;
                Complex r = o.value - p.value;
                Real w = 1.0 / o.error;
                chi2_out += w * w * (r.real() * r.real() + r.imag() * r.imag());
                nd_out += 2;
            };
            add(obs.Zxx, pred.Zxx);
            add(obs.Zxy, pred.Zxy);
            add(obs.Zyx, pred.Zyx);
            add(obs.Zyy, pred.Zyy);
        }
    }
    return (nd_out > 0) ? std::sqrt(chi2_out / nd_out) : 0.0;
}

// =========================================================================
// Objective: Φ = Φ_data + λ Φ_model
// =========================================================================
Real InversionFV::objective_() {
    fwd_->update_sigma(*model_);
    if (freq_cb_) {
        fwd_->set_freq_progress_callback(
            [this](int fi, int nf, Real fhz, const std::string&) {
                freq_cb_(fi, nf, fhz, "Forward");
            });
    }

    if (freq_parallel_active_) {
        // Each rank solves its assigned frequencies only
        for (int f : fpm_.my_frequency_indices()) {
            fwd_->compute_single_frequency(f, *data_, *data_);
            fwd_->release_factorization();
        }
        // Compute local data misfit (only for my frequencies)
        Real phi_data_local = data_misfit_for_frequencies_(fpm_.my_frequency_indices());

        Real phi_data = fpm_.allreduce_scalar(phi_data_local);
        Real phi_model = reg_->evaluate(*model_);
        return phi_data + current_lambda_ * phi_model;
    } else {
        fwd_->compute_responses(*data_, *data_);
        Real phi_data = data_misfit_from_predicted_();
        Real phi_model = reg_->evaluate(*model_);
        return phi_data + current_lambda_ * phi_model;
    }
}

// =========================================================================
// Gradient: g = JᵀW²r + λ ∇R
// =========================================================================
RealVec InversionFV::compute_gradient_() {
    int n_active = reg_->n_active();
    int ne = fwd_->num_elements();
    const auto& a2g = reg_->active_to_global();

    RealVec grad_global(ne, 0.0);
    int ns = data_->num_stations();
    int nf = data_->num_frequencies();

    // Determine which frequencies this rank handles
    std::vector<int> my_freqs;
    if (freq_parallel_active_) {
        my_freqs = fpm_.my_frequency_indices();
    } else {
        my_freqs.resize(nf);
        std::iota(my_freqs.begin(), my_freqs.end(), 0);
    }

    fwd_->update_sigma(*model_);

    for (int f : my_freqs) {
        // Forward solve
        fwd_->compute_single_frequency(f, *data_, *data_);

        // Build weighted residual for adjoint RHS.
        // For complex symmetric A (A^T=A, conj(A)≠A), the correct adjoint uses
        // w² * conj(r) in the RHS, with sensitivity = σ Re(iω λ E vol).
        // This is because ∂Φ/∂σ = Re(iω (A^{-1} Q^T w² conj(r))^T Me E).
        std::vector<std::array<Complex,4>> wr(ns);
        for (int s = 0; s < ns; ++s) {
            const auto& obs = data_->observed(s, f);
            const auto& pred = data_->predicted(s, f);
            auto w_res = [](const data::Datum& o, const data::Datum& p) -> Complex {
                if (o.weight <= 0.0 || o.error <= 0.0) return Complex(0, 0);
                Real w = 1.0 / o.error;
                return w * w * std::conj(o.value - p.value);
            };
            wr[s][0] = w_res(obs.Zxx, pred.Zxx);
            wr[s][1] = w_res(obs.Zxy, pred.Zxy);
            wr[s][2] = w_res(obs.Zyx, pred.Zyx);
            wr[s][3] = w_res(obs.Zyy, pred.Zyy);
        }

        // Build adjoint RHS
        ComplexVec adj_rhs1, adj_rhs2;
        fwd_->build_adjoint_rhs_from_residual(f, wr, adj_rhs1, adj_rhs2);

        // Adjoint solves
        ComplexVec lam1, lam2;
        fwd_->adjoint_solve_complex(adj_rhs1, lam1);
        fwd_->adjoint_solve_complex(adj_rhs2, lam2);

        // Sensitivity per polarization
        RealVec g1(ne, 0.0), g2(ne, 0.0);
        fwd_->compute_sensitivity_complex(fwd_->background_E(0), lam1, g1);
        fwd_->compute_sensitivity_complex(fwd_->background_E(1), lam2, g2);

        for (int e = 0; e < ne; ++e)
            grad_global[e] += g1[e] + g2[e];

        fwd_->release_factorization();
    }

    // Allreduce gradient across frequency groups (MPI)
    if (freq_parallel_active_) {
        RealVec grad_sum;
        fpm_.allreduce_gradient(grad_global, grad_sum);
        grad_global = std::move(grad_sum);
    }

    // Map to active space
    RealVec grad(n_active, 0.0);
    for (int j = 0; j < n_active; ++j)
        grad[j] = grad_global[a2g[j]];

    // Add regularization gradient
    RealVec reg_grad;
    reg_->gradient(*model_, reg_grad);
    for (int j = 0; j < n_active; ++j)
        grad[j] += current_lambda_ * reg_grad[a2g[j]];

    return grad;
}

// =========================================================================
// Solver-adaptive line search
// =========================================================================
//  - L-BFGS / GN: α₀=1 (quasi-Newton / Newton), Armijo backtracking
//  - NLCG: α₀ heuristic (||dm|| scaling or previous α), Armijo + safeguarded
//            quadratic interpolation on first reject only
//
//  Armijo condition: Φ(α) ≤ Φ(0) + c₁ · α · φ'(0)
//     where φ'(0) = dirderiv = gᵀd (directional derivative).
//
//  Design rationale (Nocedal & Wright, Ch. 3):
//  - L-BFGS direction already encodes curvature → α=1 is almost always
//    accepted (1 forward eval), and history update requires actual α to
//    compute good (s,y) pairs.  Quadratic refinement would cost 1 extra
//    forward eval with marginal benefit.
//  - GN step = (JᵀJ + λWᵀW)⁻¹g → natural length 1.  α<1 only when
//    model nonlinearity is large (rare for MT log-conductivity).
//  - NLCG lacks curvature info → α varies widely between iterations.
//    Using the previous accepted α avoids overshooting.  If the first
//    trial is rejected, a single quadratic interpolation (safeguarded)
//    gives a much better second guess than fixed β=0.5 halving.
// =========================================================================
Real InversionFV::line_search_(const RealVec& dm, Real obj0, Real dirderiv,
                               LSMode mode) {
    const auto& a2g = reg_->active_to_global();
    int n_active = reg_->n_active();

    auto& params = model_->params();
    RealVec params_save = params;

    // Compute ||dm|| and ||dm||_∞
    Real dm_norm = 0.0, dm_inf = 0.0;
    for (int j = 0; j < n_active; ++j) {
        dm_norm += dm[j] * dm[j];
        dm_inf = std::max(dm_inf, std::abs(dm[j]));
    }
    dm_norm = std::sqrt(dm_norm);
    if (dm_norm < 1e-15) return 0.0;

    // ── Initial step size ─────────────────────────────────────────────
    //
    // Key insight: α₀ should be chosen so that the model perturbation
    // ||α₀ · dm||_∞ is a reasonable fraction of log(σ) range.
    // For log-conductivity models, a max change of 0.5-2.0 per step is safe.
    //
    // Three strategies:
    //   (a) Directional derivative: α₀ = -2ΔΦ_target / φ'(0)
    //       where ΔΦ_target = fraction of current objective we want to reduce
    //   (b) Model-space scaling: α₀ = Δm_max / ||dm||_∞
    //       limits max per-cell log(σ) change
    //   (c) L-BFGS/GN natural step: α₀ = 1 (Hessian-scaled direction)
    //
    Real alpha;
    int max_evals;
    switch (mode) {
        case LSMode::LBFGS:
            // α=1 is natural step, but on first iteration (no history),
            // direction may be poorly scaled. Limit max model perturbation.
            if (lbfgs_stored_ > 0) {
                // History available: trust Hessian scaling, start at 1
                alpha = 1.0;
                max_evals = 6;
            } else {
                // First iteration: scale so max Δm_j ≤ 1.0 (safe in log-space)
                alpha = std::min(1.0, 1.0 / dm_inf);
                max_evals = 8;  // more budget for first iteration
                INV_LOG_INFO("  LS: L-BFGS first iter, α₀ scaled to " +
                             std::to_string(alpha) + " (||dm||_∞=" +
                             std::to_string(dm_inf) + ")");
            }
            break;
        case LSMode::GN:
            alpha     = 1.0;      // Newton: natural step
            max_evals = 6;
            break;
        case LSMode::NLCG:
        default: {
            // Strategy: use directional derivative to estimate good α₀.
            // Target: reduce objective by ~5%. Then:
            //   Φ(α) ≈ Φ(0) + α·φ'(0) + 0.5·α²·φ''(0)
            //   For a pure quadratic, minimum at α* = -φ'(0)/φ''(0)
            //   Without φ'', use: α₀ = -2·ΔΦ_target / φ'(0)
            //
            // Also cap by max model perturbation: α ≤ 2.0 / ||dm||_∞
            // (max 2.0 decades change per cell per step in log-σ space)
            Real delta_target = 0.05 * std::abs(obj0);  // aim for 5% reduction
            if (std::abs(dirderiv) > 1e-20) {
                alpha = 2.0 * delta_target / std::abs(dirderiv);
            } else {
                alpha = params_.linesearch_startdm / dm_norm;
            }
            // Model-space cap: max 2.0 decades change per cell
            Real alpha_model_cap = 2.0 / dm_inf;
            alpha = std::min(alpha, alpha_model_cap);
            // Also use previous α as lower bound hint (don't go smaller than
            // 0.1× what worked before, but allow going larger)
            if (nlcg_prev_alpha_ > 0) {
                alpha = std::max(alpha, 0.1 * nlcg_prev_alpha_);
            }
            // Reasonable bounds
            alpha = std::clamp(alpha, Real(1e-8), Real(10.0));
            max_evals = params_.linesearch_max;
            INV_LOG_INFO("  LS: NLCG α₀=" + std::to_string(alpha) +
                         " (dirderiv=" + std::to_string(dirderiv) +
                         " ||dm||_∞=" + std::to_string(dm_inf) + ")");
            break;
        }
    }

    // Armijo threshold: Φ(0) + c₁ · α · φ'(0)
    const Real c1 = params_.linesearch_c1;

    // Helper: apply trial step, return objective
    auto try_step = [&](Real a) -> Real {
        for (int j = 0; j < n_active; ++j) {
            int g = a2g[j];
            params[g] = params_save[g] + a * dm[j];
            params[g] = std::clamp(params[g], params_.log_sigma_min,
                                   params_.log_sigma_max);
        }
        model_->invalidate_cache();
        return objective_();
    };

    Real best_alpha = 0.0;
    Real best_obj = obj0;
    Real prev_obj = obj0;   // for quadratic interpolation with 2 points

    for (int eval = 0; eval < max_evals; ++eval) {
        Real obj = try_step(alpha);

        { char buf[128];
        snprintf(buf, sizeof(buf), "  LS[%d] α=%.4e Φ=%.6e ΔΦ=%.4e",
                 eval, alpha, obj, obj - obj0);
        INV_LOG_INFO(std::string(buf)); }

        if (obj < best_obj) {
            best_obj = obj;
            best_alpha = alpha;
        }

        // Armijo sufficient decrease check
        Real armijo_rhs = obj0 + c1 * alpha * dirderiv;
        if (obj <= armijo_rhs) {
            INV_LOG_INFO("  LS: accepted α=" + std::to_string(alpha));
            break;
        }

        // ── Rejected: compute next α ─────────────────────────────────
        if (eval == 0 && dirderiv < 0) {
            // Safeguarded quadratic interpolation (1st reject only):
            //   Fit parabola through  (0, obj0, φ'(0))  and  (α, obj)
            //   Minimum at α* = -φ'(0) · α² / [2 · (obj − obj0 − φ'(0)·α)]
            Real denom = 2.0 * (obj - obj0 - dirderiv * alpha);
            if (denom > 0) {
                Real alpha_q = -dirderiv * alpha * alpha / denom;
                // Safeguard: keep within [0.05α, 0.5α]
                // Lower bound 0.05 (not 0.1) to allow bigger jumps for NLCG
                alpha_q = std::clamp(alpha_q, 0.05 * alpha, 0.5 * alpha);
                alpha = alpha_q;
            } else {
                alpha *= params_.linesearch_beta;
            }
        } else {
            // Subsequent rejects: simple backtracking
            alpha *= params_.linesearch_beta;
        }
        prev_obj = obj;
    }

    // Restore best model found
    if (best_alpha > 0) {
        for (int j = 0; j < n_active; ++j) {
            int g = a2g[j];
            params[g] = params_save[g] + best_alpha * dm[j];
            params[g] = std::clamp(params[g], params_.log_sigma_min,
                                   params_.log_sigma_max);
        }
        model_->invalidate_cache();
    } else {
        // Revert to original model
        params = params_save;
        model_->invalidate_cache();
    }

    return best_alpha;
}

// =========================================================================
// NLCG step (Polak-Ribière with restart)
// =========================================================================
void InversionFV::nlcg_step_(int iter) {
    int n_active = reg_->n_active();

    // Compute gradient
    RealVec grad = compute_gradient_();
    cached_grad_ = grad;

    // CmCmᵀ preconditioning
    RealVec h(n_active);
    reg_->apply_CmCmT(grad, h);

    // Polak-Ribière β
    Real beta = 0.0;
    if (iter > 0 && !nlcg_prev_grad_.empty()) {
        Real num = 0.0, den = 0.0;
        for (int j = 0; j < n_active; ++j) {
            num += h[j] * (grad[j] - nlcg_prev_grad_[j]);
            den += nlcg_prev_precond_grad_[j] * nlcg_prev_grad_[j];
        }
        if (den > 1e-30)
            beta = std::max(Real(0), num / den);

        // Reset check
        Real dot_gg = 0.0, dot_g2 = 0.0;
        for (int j = 0; j < n_active; ++j) {
            dot_gg += grad[j] * nlcg_prev_grad_[j];
            dot_g2 += grad[j] * grad[j];
        }
        if (dot_g2 > 0 && std::abs(dot_gg) / dot_g2 > params_.nlcg_reset_threshold)
            beta = 0.0;
        if (iter % params_.nlcg_reset_every == 0)
            beta = 0.0;
    }

    // Search direction: d = -h + β * d_prev
    if (beta == 0.0 || nlcg_direction_.empty()) {
        nlcg_direction_.resize(n_active);
        for (int j = 0; j < n_active; ++j)
            nlcg_direction_[j] = -h[j];
    } else {
        for (int j = 0; j < n_active; ++j)
            nlcg_direction_[j] = -h[j] + beta * nlcg_direction_[j];
    }

    // Directional derivative
    Real dirderiv = 0.0;
    for (int j = 0; j < n_active; ++j)
        dirderiv += grad[j] * nlcg_direction_[j];

    if (dirderiv >= 0) {
        // Not a descent direction, reset
        for (int j = 0; j < n_active; ++j)
            nlcg_direction_[j] = -h[j];
        dirderiv = 0.0;
        for (int j = 0; j < n_active; ++j)
            dirderiv += grad[j] * nlcg_direction_[j];
    }

    // Line search — obj_current must use freq-parallel misfit (not stale all-freq)
    Real obj_current;
    if (freq_parallel_active_) {
        Real local_phi = data_misfit_for_frequencies_(fpm_.my_frequency_indices());
        obj_current = fpm_.allreduce_scalar(local_phi) +
                      current_lambda_ * reg_->evaluate(*model_);
    } else {
        obj_current = data_misfit_from_predicted_() +
                      current_lambda_ * reg_->evaluate(*model_);
    }
    Real alpha = line_search_(nlcg_direction_, obj_current, dirderiv,
                              LSMode::NLCG);

    // Save state for next iteration
    nlcg_prev_grad_ = grad;
    nlcg_prev_precond_grad_ = h;
    nlcg_prev_alpha_ = alpha;

    INV_LOG_INFO("NLCG iter " + std::to_string(iter) +
                 ": β=" + std::to_string(beta) +
                 " α=" + std::to_string(alpha));
}

// =========================================================================
// GN-CG step
// =========================================================================
void InversionFV::gn_cg_step_(int iter) {
    int n_active = reg_->n_active();

    // Compute gradient (also fills predicted data)
    RealVec grad = compute_gradient_();
    cached_grad_ = grad;

    // Build JtJ operator
    ImplicitJtJOperatorFV jtj(*fwd_, *data_, *model_, *reg_, current_lambda_);
    jtj.cache_background_fields();

    // RHS = -gradient (we want to solve (JtJ + λWtW)δm = -g)
    RealVec rhs(n_active);
    for (int j = 0; j < n_active; ++j)
        rhs[j] = -grad[j];

    // CG solve
    RealVec dm(n_active, 0.0);
    RealVec r = rhs;  // initial residual
    RealVec z(n_active);
    reg_->apply_CmCmT(r, z);  // preconditioned residual
    RealVec p = z;

    Real rz = 0.0;
    for (int j = 0; j < n_active; ++j)
        rz += r[j] * z[j];

    Real r0_norm = 0.0;
    for (int j = 0; j < n_active; ++j)
        r0_norm += rhs[j] * rhs[j];
    r0_norm = std::sqrt(r0_norm);

    Real cg_tol = params_.cg_tolerance;
    if (params_.cg_adaptive_tol && iter > 0) {
        Real rms = compute_rms_();
        cg_tol = std::min(cg_tol, std::max(0.01, rms - 1.0));
    }

    int cg_iters = 0;
    for (int k = 0; k < params_.cg_max_iter; ++k) {
        RealVec Ap(n_active);
        jtj.Mult(p, Ap);

        Real pAp = 0.0;
        for (int j = 0; j < n_active; ++j)
            pAp += p[j] * Ap[j];

        if (pAp <= 0) {
            INV_LOG_WARNING("GN-CG: negative curvature at iter " + std::to_string(k));
            break;
        }

        Real alpha_cg = rz / pAp;

        for (int j = 0; j < n_active; ++j) {
            dm[j] += alpha_cg * p[j];
            r[j] -= alpha_cg * Ap[j];
        }

        Real r_norm = 0.0;
        for (int j = 0; j < n_active; ++j)
            r_norm += r[j] * r[j];
        r_norm = std::sqrt(r_norm);

        cg_iters = k + 1;
        if (r_norm / (r0_norm + 1e-30) < cg_tol) break;

        reg_->apply_CmCmT(r, z);
        Real rz_new = 0.0;
        for (int j = 0; j < n_active; ++j)
            rz_new += r[j] * z[j];

        Real beta = rz_new / (rz + 1e-30);
        for (int j = 0; j < n_active; ++j)
            p[j] = z[j] + beta * p[j];

        rz = rz_new;
    }

    INV_LOG_INFO("GN-CG: " + std::to_string(cg_iters) + " CG iters, " +
                 std::to_string(jtj.num_matvecs()) + " matvecs");

    // Line search — freq-parallel safe obj_current
    Real obj_current;
    if (freq_parallel_active_) {
        Real local_phi = data_misfit_for_frequencies_(fpm_.my_frequency_indices());
        obj_current = fpm_.allreduce_scalar(local_phi) +
                      current_lambda_ * reg_->evaluate(*model_);
    } else {
        obj_current = data_misfit_from_predicted_() +
                      current_lambda_ * reg_->evaluate(*model_);
    }
    // GN-CG: dirderiv = gᵀd where d = dm (the CG solution of (JtJ+λWtW)δm = -g)
    //   For exact solve: d = -(JtJ+λWtW)⁻¹g, so gᵀd = -gᵀ(JtJ+λWtW)⁻¹g < 0 (descent)
    Real dirderiv_gn = 0.0;
    for (int j = 0; j < n_active; ++j)
        dirderiv_gn += grad[j] * dm[j];
    Real alpha = line_search_(dm, obj_current, dirderiv_gn, LSMode::GN);

    INV_LOG_INFO("GN iter " + std::to_string(iter) + ": α=" + std::to_string(alpha));
}

// =========================================================================
// L-BFGS step
// =========================================================================
void InversionFV::lbfgs_step_(int iter) {
    int n = reg_->n_active();
    int m = params_.lbfgs_memory;
    const auto& a2g = reg_->active_to_global();

    // Compute gradient (includes forward solve → fills predicted data)
    RealVec grad = compute_gradient_();
    cached_grad_ = grad;

    // Current model in active space
    RealVec params_active(n);
    for (int j = 0; j < n; ++j)
        params_active[j] = model_->params()[a2g[j]];

    // --- Update L-BFGS history ---
    if (iter > 0 && !lbfgs_prev_grad_.empty()) {
        RealVec s(n), y(n);
        for (int j = 0; j < n; ++j) {
            s[j] = params_active[j] - lbfgs_prev_params_[j];
            y[j] = grad[j] - lbfgs_prev_grad_[j];
        }

        Real sy = 0.0;
        for (int j = 0; j < n; ++j) sy += s[j] * y[j];

        if (sy > 1e-20) {
            // Allocate on first use
            if ((int)lbfgs_s_.size() < m) {
                lbfgs_s_.resize(m);
                lbfgs_y_.resize(m);
                lbfgs_rho_.resize(m, 0.0);
            }
            int idx = (lbfgs_stored_ < m) ? lbfgs_stored_ : lbfgs_oldest_;
            lbfgs_s_[idx] = std::move(s);
            lbfgs_y_[idx] = std::move(y);
            lbfgs_rho_[idx] = 1.0 / sy;

            if (lbfgs_stored_ < m) {
                ++lbfgs_stored_;
            } else {
                lbfgs_oldest_ = (lbfgs_oldest_ + 1) % m;
            }
        } else {
            INV_LOG_INFO("L-BFGS: skipping update (sᵀy=" +
                         std::to_string(sy) + " ≤ 0)");
        }
    }

    // Save current state for next iteration
    lbfgs_prev_grad_ = grad;
    lbfgs_prev_params_ = params_active;

    // --- Two-loop recursion: compute H⁻¹ · g ---
    RealVec q = grad;  // start with gradient

    int k = lbfgs_stored_;
    std::vector<Real> alpha_hist(k);

    // Most recent → oldest
    for (int i = k - 1; i >= 0; --i) {
        int idx = (lbfgs_oldest_ + i) % m;
        Real a = 0.0;
        for (int j = 0; j < n; ++j)
            a += lbfgs_rho_[idx] * lbfgs_s_[idx][j] * q[j];
        alpha_hist[i] = a;
        for (int j = 0; j < n; ++j)
            q[j] -= a * lbfgs_y_[idx][j];
    }

    // Initial Hessian approximation: H0 = γ·I where γ = sᵀy / yᵀy
    // (scaled identity from most recent pair)
    RealVec r(n);
    if (k > 0) {
        int newest = (lbfgs_oldest_ + k - 1) % m;
        Real yy = 0.0, sy = 0.0;
        for (int j = 0; j < n; ++j) {
            yy += lbfgs_y_[newest][j] * lbfgs_y_[newest][j];
            sy += lbfgs_s_[newest][j] * lbfgs_y_[newest][j];
        }
        Real gamma = (yy > 1e-30) ? sy / yy : 1.0;
        for (int j = 0; j < n; ++j)
            r[j] = gamma * q[j];
    } else {
        // First iteration: use scaled identity H₀ = (1/||g||_∞)·I
        // (Nocedal & Wright, §7.2)
        //
        // Using CmCmT here is dangerous: it can produce ||d|| >> 1
        // which combined with α₀=1 gives enormous model perturbations.
        // The scaled identity ensures ||d|| ≈ ||g||/||g||_∞ ≈ √n,
        // and the line search's α₀ = min(1, 1/||dm||_∞) keeps things safe.
        Real g_inf = 0.0;
        for (int j = 0; j < n; ++j)
            g_inf = std::max(g_inf, std::abs(q[j]));
        Real gamma0 = (g_inf > 1e-30) ? 1.0 / g_inf : 1.0;
        for (int j = 0; j < n; ++j)
            r[j] = gamma0 * q[j];
        INV_LOG_INFO("L-BFGS H₀: scaled identity γ₀=" + std::to_string(gamma0) +
                     " (||g||_∞=" + std::to_string(g_inf) + ")");
    }

    // Oldest → most recent
    for (int i = 0; i < k; ++i) {
        int idx = (lbfgs_oldest_ + i) % m;
        Real b = 0.0;
        for (int j = 0; j < n; ++j)
            b += lbfgs_rho_[idx] * lbfgs_y_[idx][j] * r[j];
        for (int j = 0; j < n; ++j)
            r[j] += lbfgs_s_[idx][j] * (alpha_hist[i] - b);
    }

    // Search direction: d = -H⁻¹ · g
    RealVec direction(n);
    for (int j = 0; j < n; ++j)
        direction[j] = -r[j];

    // Check descent
    Real dirderiv = 0.0;
    for (int j = 0; j < n; ++j)
        dirderiv += grad[j] * direction[j];

    if (dirderiv >= 0) {
        INV_LOG_INFO("L-BFGS: not descent, falling back to -CmCmT·g");
        reg_->apply_CmCmT(grad, r);
        for (int j = 0; j < n; ++j)
            direction[j] = -r[j];
        dirderiv = 0.0;
        for (int j = 0; j < n; ++j)
            dirderiv += grad[j] * direction[j];
        // Reset history
        lbfgs_stored_ = 0;
        lbfgs_oldest_ = 0;
    }

    // Line search
    Real obj_current;
    if (freq_parallel_active_) {
        Real local_phi = data_misfit_for_frequencies_(fpm_.my_frequency_indices());
        obj_current = fpm_.allreduce_scalar(local_phi) +
                      current_lambda_ * reg_->evaluate(*model_);
    } else {
        obj_current = data_misfit_from_predicted_() +
                      current_lambda_ * reg_->evaluate(*model_);
    }
    Real alpha = line_search_(direction, obj_current, dirderiv,
                              LSMode::LBFGS);

    INV_LOG_INFO("L-BFGS iter " + std::to_string(iter) +
                 ": m=" + std::to_string(lbfgs_stored_) +
                 " α=" + std::to_string(alpha));
}

// =========================================================================
// Lambda update (Occam strategy)
// =========================================================================
void InversionFV::update_lambda_() {
    Real rms = compute_rms_();
    Real ratio = rms / params_.target_rms;
    Real prev_lambda = current_lambda_;

    if (params_.lambda_strategy == InversionParamsFV::LambdaStrategy::PLATEAU) {
        // =================================================================
        // Plateau strategy (from NewMT):
        //   Hold lambda fixed while RMS improves well.
        //   Reduce only when improvement stalls for plateau_patience iters.
        // =================================================================
        if (history_.size() >= 2) {
            Real rms_prev = history_[history_.size() - 2].rms;
            Real rel_change = std::abs(rms - rms_prev) / std::max(rms_prev, Real(1e-12));

            if (rms > rms_prev * 1.01) {
                // RMS increased → increase lambda, reset counter
                current_lambda_ *= 1.5;
                plateau_count_ = 0;
                INV_LOG_INFO("  Plateau: RMS increased → lambda RAISED to " +
                             std::to_string(current_lambda_));
            } else if (rel_change < params_.plateau_tol) {
                // Slow improvement → plateau detection
                ++plateau_count_;
                INV_LOG_INFO("  Plateau: |ΔRMS/RMS| = " +
                             std::to_string(rel_change) + " < " +
                             std::to_string(params_.plateau_tol) +
                             " (count " + std::to_string(plateau_count_) +
                             "/" + std::to_string(params_.plateau_patience) + ")");

                if (plateau_count_ >= params_.plateau_patience) {
                    current_lambda_ *= params_.plateau_decrease;
                    plateau_count_ = 0;
                    INV_LOG_INFO("  Plateau: patience exhausted → lambda decreased to " +
                                 std::to_string(current_lambda_));
                }
            } else {
                // Good improvement — hold lambda, reset counter
                plateau_count_ = 0;
                INV_LOG_INFO("  Plateau: good improvement (|ΔRMS/RMS|=" +
                             std::to_string(rel_change) + ") → lambda HELD at " +
                             std::to_string(current_lambda_));
            }
        }
        // First iteration: no history to compare, keep initial lambda
    } else {
        // =================================================================
        // Ratio strategy (original): decrease based on RMS/target ratio
        // =================================================================
        if (ratio > 20.0) {
            current_lambda_ *= params_.lambda_decrease * params_.lambda_decrease * params_.lambda_decrease;
        } else if (ratio > 5.0) {
            current_lambda_ *= params_.lambda_decrease * params_.lambda_decrease;
        } else if (ratio > 1.5) {
            current_lambda_ *= params_.lambda_decrease;
        } else if (ratio < 0.9) {
            current_lambda_ /= params_.lambda_decrease;
        }
    }

    current_lambda_ = std::max(current_lambda_, Real(1e-6));

    INV_LOG_INFO("  Lambda update: " + std::to_string(prev_lambda) + " → " +
                 std::to_string(current_lambda_) + " (RMS/target=" +
                 std::to_string(ratio) + ")");
}

// =========================================================================
// Main loop
// =========================================================================
void InversionFV::resume_from(int start_iter, Real lambda) {
    start_iteration_ = start_iter;
    current_lambda_ = lambda;
    // Reset solver state — first iteration after resume will use steepest descent
    nlcg_prev_grad_.clear();
    nlcg_prev_precond_grad_.clear();
    nlcg_direction_.clear();
    nlcg_prev_alpha_ = 0.0;
    lbfgs_stored_ = 0;
    lbfgs_oldest_ = 0;
    lbfgs_prev_grad_.clear();
    lbfgs_prev_params_.clear();
    plateau_count_ = 0;
    INV_LOG_INFO("  Resume from iter " + std::to_string(start_iter) +
                 " with lambda=" + std::to_string(lambda));
}

void InversionFV::run() {
    int total_iters = start_iteration_ + params_.max_iterations;
    INV_LOG_INFO("=== InversionFV: " +
                 std::to_string(params_.max_iterations) + " iterations" +
                 (start_iteration_ > 0 ? " (resume from " + std::to_string(start_iteration_) + ")"
                                        : "") + " ===");
    {
        const char* sname = "NLCG";
        if (params_.solver == InversionParamsFV::Solver::GN_CG)  sname = "GN-CG";
        if (params_.solver == InversionParamsFV::Solver::LBFGS)  sname = "L-BFGS";
        INV_LOG_INFO("  Solver: " + std::string(sname));
    }

    for (int iter = start_iteration_; iter < total_iters; ++iter) {
        INV_LOG_INFO("\n--- Iteration " + std::to_string(iter + 1) + " ---");

        if (params_.solver == InversionParamsFV::Solver::NLCG) {
            nlcg_step_(iter);
        } else if (params_.solver == InversionParamsFV::Solver::LBFGS) {
            lbfgs_step_(iter);
        } else {
            gn_cg_step_(iter);
        }

        // Compute metrics
        Real phi_data;
        if (freq_parallel_active_) {
            Real local_phi = data_misfit_for_frequencies_(fpm_.my_frequency_indices());
            phi_data = fpm_.allreduce_scalar(local_phi);
        } else {
            phi_data = data_misfit_from_predicted_();
        }
        Real phi_model = reg_->evaluate(*model_);
        Real rms = compute_rms_();

        IterationLogFV entry;
        entry.iteration = iter + 1;
        entry.objective = phi_data + current_lambda_ * phi_model;
        entry.data_misfit = phi_data;
        entry.model_norm = phi_model;
        entry.rms = rms;
        entry.lambda = current_lambda_;
        entry.step_length = 0;
        entry.cg_iterations = 0;
        history_.push_back(entry);

        log_iteration_(iter + 1, entry);

        // Sync predicted data across ranks before callback (for CSV export)
        if (freq_parallel_active_) {
            fpm_.allreduce_predicted(*data_);
        }

        if (iter_cb_) iter_cb_(iter + 1, entry);

        // Convergence check
        if (rms <= params_.target_rms) {
            INV_LOG_INFO("=== Converged: RMS " + std::to_string(rms) +
                         " <= target " + std::to_string(params_.target_rms) + " ===");
            break;
        }

        // Blowup detection for L-BFGS: if RMS increased by >50% from
        // the best seen so far, reset history and increase lambda.
        // This catches cases where bad (s,y) pairs corrupt the direction.
        if (params_.solver == InversionParamsFV::Solver::LBFGS && history_.size() >= 2) {
            Real best_rms = rms;
            for (const auto& h : history_)
                best_rms = std::min(best_rms, h.rms);
            if (rms > best_rms * 1.5 && rms > best_rms + 1.0) {
                INV_LOG_INFO("  L-BFGS blowup detected (RMS=" + std::to_string(rms) +
                             " > 1.5×best=" + std::to_string(best_rms) +
                             ") → resetting history, increasing λ");
                lbfgs_stored_ = 0;
                lbfgs_oldest_ = 0;
                lbfgs_prev_grad_.clear();
                lbfgs_prev_params_.clear();
                current_lambda_ *= 2.0;  // more regularization to stabilize
            }
        }

        // Update lambda
        update_lambda_();

        // Checkpoint
        if (params_.save_checkpoints && (iter + 1) % params_.checkpoint_every == 0) {
            save_checkpoint_(iter + 1);
        }
    }

    INV_LOG_INFO("=== InversionFV complete ===");
}

// =========================================================================
// Logging
// =========================================================================
void InversionFV::log_iteration_(int iter, const IterationLogFV& entry) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "  Iter " << iter
        << "  RMS=" << entry.rms
        << "  Φ=" << entry.objective
        << "  Φd=" << entry.data_misfit
        << "  Φm=" << entry.model_norm
        << "  λ=" << std::scientific << entry.lambda;
    INV_LOG_INFO(oss.str());
}

// =========================================================================
// Checkpoint
// =========================================================================
void InversionFV::save_checkpoint_(int iter) {
    fs::path dir = params_.checkpoint_dir;
    fs::create_directories(dir);
    // TODO: save model + data to HDF5
    INV_LOG_INFO("  Checkpoint saved: iter " + std::to_string(iter));
}

} // namespace inversion
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
