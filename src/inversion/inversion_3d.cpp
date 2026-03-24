// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file inversion_3d.cpp
/// @brief 3D MT inversion using CG-based Gauss-Newton.

#include "maple3dmt/inversion/inversion_3d.h"
#include "maple3dmt/utils/logger.h"
#include "maple3dmt/utils/memory.h"
#include "maple3dmt/utils/memory_profiler.h"
#include <cmath>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace {

// Rank-0 only logging macros for inversion (avoids 12× duplicate MPI output)
#ifdef MAPLE3DMT_USE_MPI
#define INV_LOG_INFO(msg) do { \
    int r_; MPI_Comm_rank(MPI_COMM_WORLD, &r_); \
    if (r_ == 0) MAPLE3DMT_LOG_INFO(msg); \
} while(0)
#define INV_LOG_DEBUG(msg) do { \
    int r_; MPI_Comm_rank(MPI_COMM_WORLD, &r_); \
    if (r_ == 0) MAPLE3DMT_LOG_DEBUG(msg); \
} while(0)
#define INV_LOG_WARNING(msg) do { \
    int r_; MPI_Comm_rank(MPI_COMM_WORLD, &r_); \
    if (r_ == 0) MAPLE3DMT_LOG_WARNING(msg); \
} while(0)
#else
#define INV_LOG_INFO(msg)    MAPLE3DMT_LOG_INFO(msg)
#define INV_LOG_DEBUG(msg)   MAPLE3DMT_LOG_DEBUG(msg)
#define INV_LOG_WARNING(msg) MAPLE3DMT_LOG_WARNING(msg)
#endif

// Dynamic memory limit: 85% of physical RAM (computed at startup)
size_t compute_memory_limit() {
    double total = maple3dmt::utils::total_memory_gb();
    size_t limit = static_cast<size_t>(total * 0.85 * 1024.0 * 1024.0 * 1024.0);
    return limit;
}
} // anonymous namespace

namespace maple3dmt {
namespace inversion {

// =========================================================================
// Inversion3D::setup()
// =========================================================================
void Inversion3D::setup(mfem::ParMesh& mesh,
                        model::ConductivityModel& model,
                        data::MTData& data,
                        forward::ForwardSolver3D& fwd,
                        regularization::Regularization& reg,
                        const InversionParams3D& params) {
    mesh_   = &mesh;
    model_  = &model;
    data_   = &data;
    fwd_    = &fwd;
    reg_    = &reg;
    params_ = params;
    current_lambda_ = params_.lambda_init;
    history_.clear();

#ifdef MAPLE3DMT_USE_MPI
    comm_ = mesh_->GetComm();
#endif

    // ── Initialize 2-level frequency parallelism ──
    freq_parallel_.setup(
#ifdef MAPLE3DMT_USE_MPI
        comm_,
#endif
        data_->num_frequencies(),
        params_.freq_parallel_spatial_procs);
    freq_parallel_.print_summary();

    INV_LOG_INFO("Inversion3D setup: " +
                   std::to_string(reg_->n_active()) + " active params, " +
                   std::to_string(data_->num_stations()) + " stations, " +
                   std::to_string(data_->num_frequencies()) + " frequencies" +
                   (freq_parallel_.is_freq_parallel()
                    ? " (" + std::to_string(freq_parallel_.num_freq_groups()) + " freq groups)"
                    : ""));
}

// =========================================================================
// data_misfit_from_predicted_() — compute Φ_data from current predicted
// =========================================================================
Real Inversion3D::data_misfit_from_predicted_() const {
    // Φ_data = ½ Σ w² |r|²
    // Φ_data = ½ Σ w²|r|² where r = obs - pred, w = 1/error.
    // ∂Φ/∂Z_pred = -w²·r (the ½ and 2 from |·|² cancel).
    // Adjoint RHS uses w²·r → gradient = adjoint sensitivity.
    Real phi_data = 0.0;
    int nf = data_->num_frequencies();
    int ns = data_->num_stations();

    for (int f = 0; f < nf; ++f) {
        for (int s = 0; s < ns; ++s) {
            const auto& obs = data_->observed(s, f);
            const auto& pred = data_->predicted(s, f);

            auto add_misfit = [&](const data::Datum& o, const data::Datum& p) {
                if (o.weight <= 0.0 || o.error <= 0.0) return;
                Complex r = o.value - p.value;
                Real w = 1.0 / o.error;
                phi_data += 0.5 * w * w * (r.real() * r.real() + r.imag() * r.imag());
            };

            add_misfit(obs.Zxx, pred.Zxx);
            add_misfit(obs.Zxy, pred.Zxy);
            add_misfit(obs.Zyx, pred.Zyx);
            add_misfit(obs.Zyy, pred.Zyy);
        }
    }
    return phi_data;
}

// =========================================================================
// objective_() — compute Φ = Φ_data + λ Φ_model
// =========================================================================
Real Inversion3D::objective_() {
    // Set up frequency progress for this phase
    if (freq_progress_cb_) {
        fwd_->set_freq_progress_callback(
            [this](int fi, int nf, Real fhz, const std::string&) {
                freq_progress_cb_(fi, nf, fhz, "Forward");
            });
    }
    // Forward solve all frequencies
    fwd_->compute_responses(*data_, *data_);

    Real phi_data = data_misfit_from_predicted_();
    Real phi_model = reg_->evaluate(*model_);

    return phi_data + current_lambda_ * phi_model;
}

// =========================================================================
// compute_gradient_adjoint_() — ∂Φ_data/∂m + λ ∇Φ_model = ∇Φ
// =========================================================================
RealVec Inversion3D::compute_gradient_adjoint_() {
    int n_active = reg_->n_active();
    int ne = model_->num_elements();
    const auto& a2g = reg_->active_to_global();

    RealVec grad_global(ne, 0.0);

    // Build data weights: 1/error for each datum
    int ns = data_->num_stations();
    int nf = data_->num_frequencies();
    int data_per_station = 8;  // Re/Im for Zxx,Zxy,Zyx,Zyy
    RealVec data_weights(ns * data_per_station, 0.0);

    // Pre-allocate work vectors outside frequency loop
    auto fespace = fwd_->fespace();
    mfem::ParGridFunction lam1_r(fespace), lam1_i(fespace);
    mfem::ParGridFunction lam2_r(fespace), lam2_i(fespace);
    RealVec g1(ne), g2(ne);

    // ── Frequency loop ──
    // IMPORTANT: Frequency parallelism is DISABLED here because ParMesh uses
    // the GLOBAL MPI communicator.  MFEM collective operations (ParallelAssemble,
    // HypreParMatrix construction) require ALL ranks in the communicator to
    // participate simultaneously.  If different frequency groups enter
    // assemble_and_factorize_ at different times, the collective ops deadlock.
    //
    // To enable true frequency parallelism, each group would need its own
    // ParMesh/ParFiniteElementSpace built on spatial_comm_ (sub-communicator).
    // For now, all ranks process all frequencies sequentially (same as
    // compute_responses).
    if (freq_parallel_.is_freq_parallel()) {
        INV_LOG_WARNING("Freq-parallelism disabled in gradient: ParMesh uses "
                        "global comm → MFEM collective ops require all ranks");
    }
    std::vector<int> all_freqs(nf);
    std::iota(all_freqs.begin(), all_freqs.end(), 0);
    const auto& my_freqs = all_freqs;
    int n_my_freqs = nf;

    for (int fi = 0; fi < n_my_freqs; ++fi) {
        int f = my_freqs[fi];

        // Recompute weights per frequency (error may vary)
        for (int s = 0; s < ns; ++s) {
            const auto& obs = data_->observed(s, f);
            int base = s * data_per_station;
            auto set_w = [&](int offset, const data::Datum& d) {
                data_weights[base + offset] = (d.weight > 0 && d.error > 0)
                    ? 1.0 / d.error : 0.0;
            };
            set_w(0, obs.Zxx); set_w(1, obs.Zxx);
            set_w(2, obs.Zxy); set_w(3, obs.Zxy);
            set_w(4, obs.Zyx); set_w(5, obs.Zyx);
            set_w(6, obs.Zyy); set_w(7, obs.Zyy);
        }

        // Forward solve this frequency (factorization retained)
        INV_LOG_DEBUG("[PROGRESS] phase=Gradient_Forward freq=" +
                  std::to_string(fi + 1) + "/" + std::to_string(n_my_freqs) +
                  " (global " + std::to_string(f + 1) + "/" + std::to_string(nf) + ")");
        fwd_->compute_single_frequency(f, *data_, *data_);

        // Build adjoint RHS
        mfem::Vector adj_rhs1, adj_rhs2;
        fwd_->build_adjoint_rhs(f, *data_, *data_, data_weights, adj_rhs1, adj_rhs2);

        // Adjoint solves (reuse pre-allocated grid functions)
        INV_LOG_DEBUG("[PROGRESS] phase=Gradient_Adjoint freq=" +
                  std::to_string(fi + 1) + "/" + std::to_string(n_my_freqs));
        lam1_r = 0.0; lam1_i = 0.0;
        lam2_r = 0.0; lam2_i = 0.0;

        fwd_->adjoint_solve(adj_rhs1, lam1_r, lam1_i);
        // Seed: use Pol1 adjoint as initial guess for Pol2 adjoint
        fwd_->adjoint_solve(adj_rhs2, lam2_r, lam2_i, &lam1_r, &lam1_i);

        // Sensitivity for each polarization
        // compute_sensitivity already returns ∂Φ/∂(ln σ_e):
        //   g_e = ωσ_e ∫(λ_i·E_r - λ_r·E_i) dV
        // The σ_e factor IS the chain rule d(σ)/d(ln σ) = σ.
        // Do NOT multiply by σ again.
        fwd_->compute_sensitivity(*fwd_->E1_real(), *fwd_->E1_imag(),
                                  lam1_r, lam1_i, g1);
        fwd_->compute_sensitivity(*fwd_->E2_real(), *fwd_->E2_imag(),
                                  lam2_r, lam2_i, g2);

        // Gradient accumulation: sign verified by gradient_check (FD vs adjoint).
        for (int e = 0; e < ne; ++e) {
            grad_global[e] += g1[e] + g2[e];
        }

        fwd_->release_factorization();

        INV_LOG_DEBUG("[PROGRESS] phase=Gradient freq=" +
                  std::to_string(fi + 1) + "/" + std::to_string(n_my_freqs) + " done");
    }

    // ── Frequency-parallel gradient allreduce ──
    // Skipped: all ranks now process all frequencies (no frequency distribution)
    // because ParMesh uses global communicator.
    // if (freq_parallel_.is_freq_parallel()) {
    //     RealVec grad_sum;
    //     freq_parallel_.allreduce_gradient(grad_global, grad_sum);
    //     grad_global = std::move(grad_sum);
    // }

    // Map to active space
    RealVec grad_active(n_active, 0.0);
    for (int j = 0; j < n_active; ++j) {
        grad_active[j] = grad_global[a2g[j]];
    }

    // Add regularization gradient: λ ∇R
    RealVec reg_grad;
    reg_->gradient(*model_, reg_grad);
    for (int j = 0; j < n_active; ++j) {
        grad_active[j] += current_lambda_ * reg_grad[a2g[j]];
    }

    return grad_active;
}

// =========================================================================
// line_search_() — Quadratic interpolation + backtracking fallback
// =========================================================================
// Strategy:
//   1. Try α₀ (GN: start at 1.0, NLCG: startdm/||d||)
//   2. If Φ(α₀) increased and dirderiv available: quadratic interpolation
//   3. If still not accepted: continue halving (backtracking) up to max evals
// =========================================================================
Real Inversion3D::line_search_(const RealVec& dm, Real obj_current,
                                Real dirderiv) {
    const auto& a2g = reg_->active_to_global();
    int n_active = reg_->n_active();

    // Save original model
    auto& params = model_->params();
    RealVec params_save = params;

    // ---- Compute ||dm|| and ||dm||_inf ----
    Real dm_norm2 = 0.0;
    Real dm_inf = 0.0;
    for (int j = 0; j < n_active; ++j) {
        dm_norm2 += dm[j] * dm[j];
        dm_inf = std::max(dm_inf, std::abs(dm[j]));
    }
#ifdef MAPLE3DMT_USE_MPI
    {
        Real dm_norm2_global, dm_inf_global;
        MPI_Allreduce(&dm_norm2, &dm_norm2_global, 1, MPI_DOUBLE, MPI_SUM, comm_);
        MPI_Allreduce(&dm_inf, &dm_inf_global, 1, MPI_DOUBLE, MPI_MAX, comm_);
        dm_norm2 = dm_norm2_global;
        dm_inf = dm_inf_global;
    }
#endif
    Real dm_norm = std::sqrt(dm_norm2);

    // ---- Initial step size ----
    Real alpha0;
    if (params_.solver == InversionSolver::GN_CG) {
        alpha0 = (nlcg_prev_alpha_ > 0) ? nlcg_prev_alpha_ * 1.01 : 1.0;
    } else {
        if (nlcg_prev_alpha_ > 0) {
            alpha0 = nlcg_prev_alpha_ * 1.01;
        } else {
            alpha0 = (dm_norm > 1e-30) ? params_.linesearch_startdm / dm_norm : 1.0;
        }
    }

    // Safety: clamp so max step in log-space ≤ max_step_log
    constexpr Real max_step_log = 1.0;
    if (dm_inf > 1e-30 && alpha0 * dm_inf > max_step_log) {
        alpha0 = max_step_log / dm_inf;
    }

    INV_LOG_INFO("  LS: alpha0=" + std::to_string(alpha0) +
                   " ||dm||=" + std::to_string(dm_norm) +
                   " ||dm||_inf=" + std::to_string(dm_inf) +
                   " dirderiv=" + std::to_string(dirderiv) +
                   " obj_cur=" + std::to_string(obj_current));

    // ---- Helper: apply update + clamp to bounds ----
    auto apply_and_clamp = [&](Real a) {
        params = params_save;
        for (int j = 0; j < n_active; ++j) {
            Real val = params_save[a2g[j]] + a * dm[j];
            val = std::max(val, params_.log_sigma_min);
            val = std::min(val, params_.log_sigma_max);
            params[a2g[j]] = val;
        }
        model_->invalidate_cache();
    };

    Real noise_tol = std::max(1e-3 * std::abs(obj_current), 1.0);
    int max_evals = params_.linesearch_max;

    // Track best result across all evaluations
    Real best_alpha = 0.0;
    Real best_obj = obj_current;
    int eval_count = 0;

    // ---- Step 1: Evaluate f(α₀) ----
    apply_and_clamp(alpha0);
    Real obj_alpha0 = objective_();
    Real diff0 = obj_alpha0 - obj_current;
    eval_count++;

    INV_LOG_INFO("  LS[0] alpha=" + std::to_string(alpha0) +
                   "  Phi=" + std::to_string(obj_alpha0) +
                   "  diff=" + std::to_string(diff0));

    if (obj_alpha0 < best_obj) { best_alpha = alpha0; best_obj = obj_alpha0; }

    if (diff0 < noise_tol) {
        INV_LOG_INFO("  LS: alpha=" + std::to_string(alpha0) +
                     " accepted (step 0, diff=" + std::to_string(diff0) + ")");
        nlcg_prev_alpha_ = alpha0;
        return alpha0;
    }

    // ---- Step 2: Quadratic interpolation (if dirderiv available) ----
    Real alpha_next = -1.0;
    bool used_quadratic = false;

    if (dirderiv < -1e-30) {
        Real c = (obj_alpha0 - obj_current - dirderiv * alpha0) / (alpha0 * alpha0);
        if (c > 1e-30) {
            alpha_next = -dirderiv / (2.0 * c);
            // Safeguard: keep in [0.05·α₀, 0.5·α₀]
            alpha_next = std::max(alpha_next, 0.05 * alpha0);
            alpha_next = std::min(alpha_next, 0.5 * alpha0);
            used_quadratic = true;
        }
    }

    if (alpha_next < 0.0) {
        alpha_next = alpha0 * 0.5;  // fallback: halve
    }

    // ---- Backtracking loop (quadratic first, then halving) ----
    Real alpha = alpha_next;
    while (eval_count < max_evals) {
        if (alpha * dm_inf < 1e-12) break;  // step too small

        apply_and_clamp(alpha);
        Real obj_alpha = objective_();
        Real diff = obj_alpha - obj_current;
        eval_count++;

        INV_LOG_INFO("  LS[" + std::to_string(eval_count - 1) + "] alpha=" +
                     std::to_string(alpha) + "  Phi=" + std::to_string(obj_alpha) +
                     "  diff=" + std::to_string(diff) +
                     (used_quadratic ? "  (quadratic)" : ""));

        if (obj_alpha < best_obj) { best_alpha = alpha; best_obj = obj_alpha; }

        if (diff < noise_tol) {
            INV_LOG_INFO("  LS: alpha=" + std::to_string(alpha) +
                         " accepted (evals=" + std::to_string(eval_count) +
                         ", diff=" + std::to_string(diff) + ")");
            nlcg_prev_alpha_ = alpha;
            return alpha;
        }

        // Next step: always halve from here
        used_quadratic = false;
        alpha *= 0.5;
    }

    // ---- Accept best if it's a meaningful decrease ----
    if (best_obj < obj_current + noise_tol) {
        apply_and_clamp(best_alpha);
        objective_();  // recompute predicted for consistency
        INV_LOG_INFO("  LS: accepting best alpha=" + std::to_string(best_alpha) +
                     " (best_diff=" + std::to_string(best_obj - obj_current) +
                     ", evals=" + std::to_string(eval_count) + ")");
        nlcg_prev_alpha_ = best_alpha;
        return best_alpha;
    }

    // ---- Failed — restore ----
    params = params_save;
    model_->invalidate_cache();

    INV_LOG_WARNING("LS failed (" + std::to_string(eval_count) +
                    " evals): alpha0=" + std::to_string(alpha0) +
                    " → restored original model"
                    " (best_diff=" + std::to_string(best_obj - obj_current) + ")");

    nlcg_prev_alpha_ = 0;
    return 0.0;
}

// =========================================================================
// update_lambda_()
// =========================================================================
void Inversion3D::update_lambda_() {
    if (history_.size() < 2) {
        current_lambda_ *= params_.lambda_decrease;
        return;
    }

    Real rms_prev = history_[history_.size() - 2].rms;
    Real rms_curr = history_.back().rms;
    Real alpha = history_.back().step_length;

    if (alpha == 0.0) {
        // LS failed → INCREASE lambda (more regularization = smoother gradient)
        consecutive_ls_fail_++;
        if (consecutive_ls_fail_ >= 3) {
            // 3 consecutive failures → strong increase
            current_lambda_ *= 5.0;
            INV_LOG_INFO("  Lambda ×5 (3 consecutive LS failures): " +
                        std::to_string(current_lambda_));
        } else {
            current_lambda_ *= 2.0;
            INV_LOG_INFO("  Lambda ×2 (LS failure): " +
                        std::to_string(current_lambda_));
        }
    } else {
        consecutive_ls_fail_ = 0;
        Real rms_change = (rms_prev - rms_curr) / rms_prev;

        if (rms_change > 0.02) {
            // Good progress (>2% RMS reduction) → decrease lambda
            current_lambda_ *= params_.lambda_decrease;
        } else if (rms_change > 0.001) {
            // Moderate progress → gentle decrease
            current_lambda_ *= std::sqrt(params_.lambda_decrease);
        } else {
            // Stagnation: converged at current λ → decrease to allow more data fit
            // Standard Occam strategy: always reduce λ until RMS ≈ target
            current_lambda_ *= params_.lambda_decrease;
            INV_LOG_INFO("  Lambda decreased (stagnation at RMS=" +
                        std::to_string(rms_curr) + "): " +
                        std::to_string(current_lambda_));
        }
    }

    // Clamp lambda to reasonable range
    current_lambda_ = std::max(current_lambda_, 1e-6);
    current_lambda_ = std::min(current_lambda_, 1e6);
}

// =========================================================================
// save_checkpoint_()
// =========================================================================
void Inversion3D::save_checkpoint_(int iter) {
    if (!params_.save_checkpoints) return;

    // Only rank 0 writes the checkpoint file to avoid race conditions.
    int rank = 0;
#ifdef MAPLE3DMT_USE_MPI
    MPI_Comm_rank(comm_, &rank);
#endif
    if (rank != 0) return;

    fs::create_directories(params_.checkpoint_dir);
    fs::path path = params_.checkpoint_dir /
        ("iter_" + std::to_string(iter) + ".dat");

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        INV_LOG_WARNING("Cannot write checkpoint: " + path.string());
        return;
    }

    // Write iteration number and lambda
    out.write(reinterpret_cast<const char*>(&iter), sizeof(int));
    out.write(reinterpret_cast<const char*>(&current_lambda_), sizeof(Real));

    // Write model parameters
    const auto& p = model_->params();
    int np = static_cast<int>(p.size());
    out.write(reinterpret_cast<const char*>(&np), sizeof(int));
    out.write(reinterpret_cast<const char*>(p.data()), np * sizeof(Real));

    INV_LOG_DEBUG("Checkpoint saved: " + path.string());
}

// =========================================================================
// load_checkpoint_()
// =========================================================================
void Inversion3D::load_checkpoint_(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot read checkpoint: " + path.string());
    }

    int iter;
    in.read(reinterpret_cast<char*>(&iter), sizeof(int));
    in.read(reinterpret_cast<char*>(&current_lambda_), sizeof(Real));

    int np;
    in.read(reinterpret_cast<char*>(&np), sizeof(int));
    auto& p = model_->params();
    if (np != static_cast<int>(p.size())) {
        throw std::runtime_error("Checkpoint parameter count mismatch");
    }
    in.read(reinterpret_cast<char*>(p.data()), np * sizeof(Real));
    model_->invalidate_cache();

    INV_LOG_DEBUG("Checkpoint loaded: iter=" + std::to_string(iter) +
                   ", lambda=" + std::to_string(current_lambda_));
}

// =========================================================================
// log_iteration_()
// =========================================================================
void Inversion3D::log_iteration_(int iter, const IterationLog3D& entry) {
    // One-line summary for monitoring
    char buf[256];
    snprintf(buf, sizeof(buf),
             "Iter %d | RMS=%.4f  Obj=%.2f  Lambda=%.4f  Step=%.2e  CG=%d  Mem=%.1fGB",
             iter, entry.rms, entry.objective, entry.lambda,
             entry.step_length, entry.cg_iterations, entry.peak_memory_gb);
    INV_LOG_INFO(std::string(buf));
    // Detailed breakdown for debug
    INV_LOG_DEBUG(
        "  Data misfit: " + std::to_string(entry.data_misfit) +
        "  Model norm: " + std::to_string(entry.model_norm));
}

// =========================================================================
// resume()
// =========================================================================
void Inversion3D::resume(const fs::path& checkpoint_path) {
    load_checkpoint_(checkpoint_path);
    run();
}

// =========================================================================
// ImplicitJtJOperator
// =========================================================================
ImplicitJtJOperator::ImplicitJtJOperator(
    forward::ForwardSolver3D& fwd,
    const data::MTData& data,
    const model::ConductivityModel& model,
    const regularization::Regularization& reg,
    Real lambda)
    : mfem::Operator(reg.n_active())
    , fwd_(&fwd)
    , data_(&data)
    , model_(&model)
    , reg_(&reg)
    , lambda_(lambda)
{
}

void ImplicitJtJOperator::cache_background_fields() {
    int nf = data_->num_frequencies();
    bg_fields_cache_.resize(nf);

    INV_LOG_INFO("Caching background E fields for " + std::to_string(nf) + " frequencies...");

    for (int f = 0; f < nf; ++f) {
        // Full forward solve for this frequency (computes E1, E2)
        fwd_->compute_single_frequency(f, *data_, const_cast<data::MTData&>(*data_));

        // Cache the E fields as true-DOF vectors (compact, mesh-independent size)
        fwd_->E1_real()->GetTrueDofs(bg_fields_cache_[f].E1_r);
        fwd_->E1_imag()->GetTrueDofs(bg_fields_cache_[f].E1_i);
        fwd_->E2_real()->GetTrueDofs(bg_fields_cache_[f].E2_r);
        fwd_->E2_imag()->GetTrueDofs(bg_fields_cache_[f].E2_i);

        // Release factorization to free memory (will re-factorize in Mult)
        fwd_->release_factorization();

        INV_LOG_DEBUG("  Cached freq " + std::to_string(f + 1) + "/" +
                      std::to_string(nf) + ": " +
                      std::to_string(data_->frequencies()[f]) + " Hz");
    }

    INV_LOG_INFO("Background field caching complete.");
}

void ImplicitJtJOperator::ensure_work_vectors_() const {
    if (work_allocated_) return;

    int ne = model_->num_elements();
    int ns = data_->num_stations();
    int data_per_station = 8;

    work_delta_sigma_.resize(ne, 0.0);
    work_jtjx_global_.resize(ne, 0.0);
    work_data_weights_.resize(ns * data_per_station, 0.0);
    work_g1_.resize(ne);
    work_g2_.resize(ne);
    work_x_full_.resize(ne, 0.0);

    auto fespace = fwd_->fespace();
    work_lam1_r_ = std::make_unique<mfem::ParGridFunction>(fespace);
    work_lam1_i_ = std::make_unique<mfem::ParGridFunction>(fespace);
    work_lam2_r_ = std::make_unique<mfem::ParGridFunction>(fespace);
    work_lam2_i_ = std::make_unique<mfem::ParGridFunction>(fespace);
    work_dE_r_ = std::make_unique<mfem::ParGridFunction>(fespace);
    work_dE_i_ = std::make_unique<mfem::ParGridFunction>(fespace);

    work_allocated_ = true;
}

void ImplicitJtJOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const {
    ++num_matvecs_;
    ensure_work_vectors_();

    int n_active = reg_->n_active();
    int ne = model_->num_elements();
    const auto& a2g = reg_->active_to_global();
    int ns = data_->num_stations();
    int nf = data_->num_frequencies();
    int data_per_station = 8;

    // Active → global: delta_sigma[e] = sigma[e] * x[j] (log-σ chain rule)
    std::fill(work_delta_sigma_.begin(), work_delta_sigma_.end(), 0.0);
    for (int j = 0; j < n_active; ++j) {
        int e = a2g[j];
        work_delta_sigma_[e] = model_->sigma(e) * x(j);
    }

    // Accumulate J^T W² J x over all frequencies
    std::fill(work_jtjx_global_.begin(), work_jtjx_global_.end(), 0.0);

    for (int f = 0; f < nf; ++f) {
        // Use cached background fields (no re-solve!) + factorize only.
        // Background E fields were cached once before the FGMRES loop.
        Real freq_hz = data_->frequencies()[f];
        fwd_->factorize_frequency(freq_hz);

        // Restore cached E fields into forward solver
        if (!bg_fields_cache_.empty()) {
            auto fespace = fwd_->fespace();
            mfem::ParGridFunction E1_r(fespace), E1_i(fespace),
                                   E2_r(fespace), E2_i(fespace);
            E1_r.SetFromTrueDofs(bg_fields_cache_[f].E1_r);
            E1_i.SetFromTrueDofs(bg_fields_cache_[f].E1_i);
            E2_r.SetFromTrueDofs(bg_fields_cache_[f].E2_r);
            E2_i.SetFromTrueDofs(bg_fields_cache_[f].E2_i);
            fwd_->set_background_fields(E1_r, E1_i, E2_r, E2_i);
        }

        // Build data weights for this frequency
        std::fill(work_data_weights_.begin(), work_data_weights_.end(), 0.0);
        for (int s = 0; s < ns; ++s) {
            const auto& obs = data_->observed(s, f);
            int base = s * data_per_station;
            auto set_w = [&](int offset, const data::Datum& d) {
                work_data_weights_[base + offset] = (d.weight > 0 && d.error > 0)
                    ? 1.0 / d.error : 0.0;
            };
            set_w(0, obs.Zxx); set_w(1, obs.Zxx);
            set_w(2, obs.Zxy); set_w(3, obs.Zxy);
            set_w(4, obs.Zyx); set_w(5, obs.Zyx);
            set_w(6, obs.Zyy); set_w(7, obs.Zyy);
        }

        // --- J·x step: for each polarization ---
        std::vector<std::array<Complex,4>> total_delta_Z(ns,
            {Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0)});

        for (int pol = 0; pol < 2; ++pol) {
            mfem::Vector pert_rhs;
            fwd_->build_perturbation_rhs(pol, work_delta_sigma_, pert_rhs);

            *work_dE_r_ = 0.0;
            *work_dE_i_ = 0.0;
            fwd_->solve_forward_rhs(pert_rhs, *work_dE_r_, *work_dE_i_);

            std::vector<std::array<Complex,4>> dZ;
            fwd_->extract_delta_impedance(*work_dE_r_, *work_dE_i_, pol, dZ);

            for (int s = 0; s < ns; ++s) {
                for (int c = 0; c < 4; ++c) {
                    total_delta_Z[s][c] += dZ[s][c];
                }
            }
        }

        // Apply W² to δZ to get weighted residual
        std::vector<std::array<Complex,4>> weighted_dZ(ns);
        for (int s = 0; s < ns; ++s) {
            int base = s * data_per_station;
            for (int c = 0; c < 4; ++c) {
                Real wr = work_data_weights_[base + 2*c];
                Real wi = work_data_weights_[base + 2*c + 1];
                weighted_dZ[s][c] = Complex(
                    total_delta_Z[s][c].real() * wr * wr,
                    total_delta_Z[s][c].imag() * wi * wi);
            }
        }

        // --- J^T step: adjoint solve from weighted δZ ---
        // adj_rhs = Q^T · W² · δZ (no negation — sensitivity formula
        // already accounts for the adjoint identity sign).
        mfem::Vector adj_rhs1, adj_rhs2;
        fwd_->build_adjoint_rhs_from_residual(f, weighted_dZ, adj_rhs1, adj_rhs2);

        *work_lam1_r_ = 0.0; *work_lam1_i_ = 0.0;
        *work_lam2_r_ = 0.0; *work_lam2_i_ = 0.0;

        fwd_->adjoint_solve(adj_rhs1, *work_lam1_r_, *work_lam1_i_);
        // Seed: use Pol1 adjoint as initial guess for Pol2
        fwd_->adjoint_solve(adj_rhs2, *work_lam2_r_, *work_lam2_i_,
                            work_lam1_r_.get(), work_lam1_i_.get());

        fwd_->compute_sensitivity(*fwd_->E1_real(), *fwd_->E1_imag(),
                                  *work_lam1_r_, *work_lam1_i_, work_g1_);
        fwd_->compute_sensitivity(*fwd_->E2_real(), *fwd_->E2_imag(),
                                  *work_lam2_r_, *work_lam2_i_, work_g2_);

        for (int e = 0; e < ne; ++e) {
            work_jtjx_global_[e] += work_g1_[e] + work_g2_[e];
        }

        fwd_->release_factorization();
    }

    // Map to active space
    y.SetSize(n_active);
    for (int j = 0; j < n_active; ++j) {
        y(j) = work_jtjx_global_[a2g[j]];
    }

    // Add regularization Hessian: y += λ · 2α_s · WtW · x
    // The factor 2α_s comes from ∂²Φ_model/∂m² where Φ_model = α_s m^T WtW m.
    // apply_WtW returns WtW·x (unscaled), so we scale by 2·α_s here.
    std::fill(work_x_full_.begin(), work_x_full_.end(), 0.0);
    for (int j = 0; j < n_active; ++j) {
        work_x_full_[a2g[j]] = x(j);
    }
    RealVec wtw_result;
    reg_->apply_WtW(work_x_full_, wtw_result);
    Real hess_scale = 2.0 * reg_->alpha_s();
    for (int j = 0; j < n_active; ++j) {
        y(j) += lambda_ * hess_scale * wtw_result[a2g[j]];
    }
}

// =========================================================================
// gauss_newton_cg_step_()
// =========================================================================
void Inversion3D::gauss_newton_cg_step_(int iter) {
    int n_active = reg_->n_active();

    // ── Stagnation check: reuse cached gradient, skip forward+adjoint ──
    // ∇Φ_data (= JᵀR) is independent of λ.  When the previous iteration
    // stagnated (CG=0 or tiny alpha), recomputing forward+adjoint gives
    // the same gradient.  We only need to re-solve CG with the new λ.
    bool reuse_gradient = false;
    if (!cached_grad_.empty() && history_.size() >= 2) {
        const auto& prev = history_.back();
        if (prev.cg_iterations == 0 || prev.step_length < 1e-3) {
            reuse_gradient = true;
            INV_LOG_INFO("  Reusing cached gradient (stagnation at prev λ="
                         + std::to_string(prev.lambda) + ", new λ="
                         + std::to_string(current_lambda_) + ")");
        }
    }

    RealVec grad;
    if (reuse_gradient) {
        grad = cached_grad_;   // no forward+adjoint needed
    } else {
        INV_LOG_DEBUG("Computing gradient (adjoint)...");
        grad = compute_gradient_adjoint_();
    }

    // Gradient norm (MPI reduction for distributed model)
    Real grad_norm2 = 0.0;
    for (int j = 0; j < n_active; ++j) grad_norm2 += grad[j] * grad[j];
#ifdef MAPLE3DMT_USE_MPI
    {
        Real grad_norm2_global;
        MPI_Allreduce(&grad_norm2, &grad_norm2_global, 1,
                       MPI_DOUBLE, MPI_SUM, comm_);
        grad_norm2 = grad_norm2_global;
    }
#endif
    Real grad_norm = std::sqrt(grad_norm2);
    INV_LOG_DEBUG("  ||grad||=" + std::to_string(grad_norm));

    if (grad_norm < 1e-15) {
        INV_LOG_DEBUG("Gradient near zero — converged");
        return;
    }

    // Cache gradient for potential reuse in next iteration
    cached_grad_ = grad;

    // Set up implicit operator
    ImplicitJtJOperator jtj_op(*fwd_, *data_, *model_, *reg_, current_lambda_);

    // Cache background E fields ONCE before the FGMRES loop.
    // This eliminates the catastrophic bug where compute_single_frequency()
    // was called inside every FGMRES iteration, causing:
    //   (a) ~6min wasted per iteration on redundant forward solves
    //   (b) Operator inconsistency (iterative solver returns slightly different
    //       E fields each time) → CG/FGMRES divergence
    // With caching: Mult() only needs factorize + perturbation/adjoint solves.
    // Always cache: gradient computation releases factorizations per-freq,
    // so E fields are gone.  One extra forward pass (all freqs) is needed,
    // but this replaces N_fgmres × N_freq forward solves inside Mult().
    // TODO: optimize by caching during gradient computation to avoid this pass.
    jtj_op.cache_background_fields();

    // FGMRES solve: (JtJ + λ WtW) δm = -g
    mfem::Vector rhs(n_active);
    for (int j = 0; j < n_active; ++j) {
        rhs(j) = -grad[j];
    }

    // Eisenstat-Walker adaptive CG tolerance
    Real cg_tol = params_.cg_tolerance;
    if (params_.cg_adaptive_tol && !history_.empty()) {
        Real rms_prev = history_.back().rms;
        Real rms_curr = data_->rms_misfit();
        if (rms_prev > 0.0) {
            cg_tol = std::min(params_.cg_tolerance, 0.5 * rms_curr / rms_prev);
            cg_tol = std::max(cg_tol, 1e-4);
        }
    }

    INV_LOG_DEBUG("FGMRES solve: tol=" + std::to_string(cg_tol) +
                   ", max_iter=" + std::to_string(params_.cg_max_iter) +
                   ", kdim=" + std::to_string(params_.cg_kdim));

    // Use FGMRES instead of CG: JᵀJ operator is not exactly symmetric
    // because iterative adjoint solves introduce asymmetry.
    // FGMRES handles non-symmetric operators robustly.
#ifdef MAPLE3DMT_USE_MPI
    mfem::FGMRESSolver fgmres(comm_);
#else
    mfem::FGMRESSolver fgmres;
#endif
    fgmres.SetOperator(jtj_op);
    fgmres.SetRelTol(cg_tol);
    fgmres.SetAbsTol(1e-15);
    fgmres.SetMaxIter(params_.cg_max_iter);
    fgmres.SetKDim(params_.cg_kdim);
    fgmres.SetPrintLevel(1);

    mfem::Vector dm(n_active);
    dm = 0.0;
    fgmres.Mult(rhs, dm);

    int cg_iters = jtj_op.num_matvecs();
    INV_LOG_DEBUG("FGMRES converged in " + std::to_string(cg_iters) + " iterations");

    // Current objective — predicted data already computed by
    // compute_gradient_adjoint_() via compute_single_frequency() per freq.
    Real obj_current = data_misfit_from_predicted_() +
                       current_lambda_ * reg_->evaluate(*model_);
    INV_LOG_DEBUG("  obj_current (from gradient) = " + std::to_string(obj_current));

    // Directional derivative: f'(0) = grad · dm  (should be < 0)
    Real dirderiv = 0.0;
    for (int j = 0; j < n_active; ++j) dirderiv += grad[j] * dm(j);
#ifdef MAPLE3DMT_USE_MPI
    {
        Real dd_global;
        MPI_Allreduce(&dirderiv, &dd_global, 1, MPI_DOUBLE, MPI_SUM, comm_);
        dirderiv = dd_global;
    }
#endif
    INV_LOG_DEBUG("  dirderiv (g·dm) = " + std::to_string(dirderiv));

    // Line search with quadratic interpolation
    RealVec dm_vec(n_active);
    for (int j = 0; j < n_active; ++j) dm_vec[j] = dm(j);

    Real alpha = line_search_(dm_vec, obj_current, dirderiv);

    // Log iteration — values AFTER line search (model already updated)
    Real rms = data_->rms_misfit();
    Real phi_model = reg_->evaluate(*model_);
    Real phi_data_post = data_misfit_from_predicted_();

    IterationLog3D entry;
    entry.iteration = iter;
    entry.objective = phi_data_post + current_lambda_ * phi_model;
    entry.data_misfit = phi_data_post;
    entry.model_norm = phi_model;
    entry.rms = rms;
    entry.lambda = current_lambda_;
    entry.step_length = alpha;
    entry.cg_iterations = cg_iters;
    entry.peak_memory_gb = utils::current_rss_gb();

    history_.push_back(entry);
    log_iteration_(iter, entry);

    if (iter_callback_) {
        iter_callback_(iter, entry);
    }
}

// =========================================================================
// run()
// =========================================================================
// =========================================================================
// NLCG step (Polak-Ribière with automatic restart)
// =========================================================================
void Inversion3D::nlcg_step_(int iter) {
    int ne = model_->num_elements();
    int n_active = reg_->n_active();
    const auto& a2g = reg_->active_to_global();

    // 1. Compute gradient: g = ∂Φ_data/∂m + λ ∂Φ_model/∂m = ∇Φ
    //    (forward + adjoint solves happen inside, with factorize/release per freq)
    //    Returns active-space vector (size n_active), NOT global.
    RealVec grad = compute_gradient_adjoint_();

    // grad is already in active space — copy directly (no a2g mapping!)
    mfem::Vector g_active(n_active);
    for (int j = 0; j < n_active; ++j) {
        g_active(j) = grad[j];
    }

    // 1b. Apply CmCm^T preconditioning (ModEM-style model covariance smoothing).
    // Preconditioned gradient h = CmCm^T * g reduces dynamic range so that
    // line search step sizes become O(1-100) instead of O(1e-5).
    mfem::Vector h_active(n_active);
    reg_->apply_CmCmT(g_active, h_active);

    // Inner product g·h (for preconditioned NLCG β computation)
    Real g_dot_h = 0.0;
    for (int j = 0; j < n_active; ++j) {
        g_dot_h += g_active(j) * h_active(j);
    }
#ifdef MAPLE3DMT_USE_MPI
    {
        Real tmp;
        MPI_Allreduce(&g_dot_h, &tmp, 1, MPI_DOUBLE, MPI_SUM, comm_);
        g_dot_h = tmp;
    }
#endif

    Real g_norm2 = g_dot_h;  // preconditioned norm for logging

    // 2. Compute β (Preconditioned Polak-Ribière)
    //    β = g_new · (h_new - h_old) / (g_old · h_old)
    //    where h = CmCm^T * g  (preconditioned gradient)
    Real beta = 0.0;
    if (iter > 0 && !nlcg_prev_grad_.empty() && !nlcg_prev_precond_grad_.empty()) {
        // g_new · h_old
        Real g_new_dot_h_old = 0.0;
        for (int j = 0; j < n_active; ++j) {
            g_new_dot_h_old += g_active(j) * nlcg_prev_precond_grad_[j];
        }
#ifdef MAPLE3DMT_USE_MPI
        {
            Real tmp;
            MPI_Allreduce(&g_new_dot_h_old, &tmp, 1, MPI_DOUBLE, MPI_SUM, comm_);
            g_new_dot_h_old = tmp;
        }
#endif
        // g_old · h_old
        Real g_old_dot_h_old = 0.0;
        for (int j = 0; j < n_active; ++j) {
            g_old_dot_h_old += nlcg_prev_grad_[j] * nlcg_prev_precond_grad_[j];
        }
#ifdef MAPLE3DMT_USE_MPI
        {
            Real tmp;
            MPI_Allreduce(&g_old_dot_h_old, &tmp, 1, MPI_DOUBLE, MPI_SUM, comm_);
            g_old_dot_h_old = tmp;
        }
#endif
        if (g_old_dot_h_old > 1e-30) {
            // Preconditioned Polak-Ribière: β = (g·h - g·h_old) / (g_old·h_old)
            beta = (g_dot_h - g_new_dot_h_old) / g_old_dot_h_old;
            beta = std::max(beta, 0.0);  // PR+: enforce β ≥ 0
        }

        // Restart conditions (using preconditioned inner products)
        Real overlap = std::abs(g_new_dot_h_old) / g_dot_h;
        if (overlap > params_.nlcg_reset_threshold) {
            INV_LOG_DEBUG("  NLCG restart: overlap=" +
                           std::to_string(overlap) + " > " +
                           std::to_string(params_.nlcg_reset_threshold));
            beta = 0.0;
        }
        if (iter % params_.nlcg_reset_every == 0) {
            INV_LOG_DEBUG("  NLCG periodic restart at iter " +
                           std::to_string(iter));
            beta = 0.0;
        }
    }

    INV_LOG_DEBUG("[PROGRESS] phase=NLCG_Direction beta=" + std::to_string(beta));

    // 3. Search direction: d = -h + β * d_prev  (preconditioned NLCG)
    //    Uses preconditioned gradient h, NOT raw gradient g.
    if (nlcg_direction_.empty() || (int)nlcg_direction_.size() != n_active) {
        nlcg_direction_.resize(n_active, 0.0);
    }
    for (int j = 0; j < n_active; ++j) {
        nlcg_direction_[j] = -h_active(j) + beta * nlcg_direction_[j];
    }

    // Direction norm for diagnostics
    {
        Real d_inf = 0.0;
        for (int j = 0; j < n_active; ++j) {
            d_inf = std::max(d_inf, std::abs(nlcg_direction_[j]));
        }
#ifdef MAPLE3DMT_USE_MPI
        Real d_inf_global;
        MPI_Allreduce(&d_inf, &d_inf_global, 1, MPI_DOUBLE, MPI_MAX, comm_);
        d_inf = d_inf_global;
#endif
        INV_LOG_DEBUG("  ||grad||=" + std::to_string(std::sqrt(g_norm2)) +
                       "  beta=" + std::to_string(beta) +
                       "  ||d||_inf=" + std::to_string(d_inf));
    }

    // 4. Current objective — use predicted data already computed by
    //    compute_gradient_adjoint_() which calls compute_single_frequency()
    //    per frequency, each storing predicted impedance via set_predicted().
    //    No need for a redundant full forward pass.
    Real phi_data_cur = data_misfit_from_predicted_();
    Real phi_model_cur = reg_->evaluate(*model_);
    Real obj_current = phi_data_cur + current_lambda_ * phi_model_cur;
    INV_LOG_DEBUG("  obj_current (from gradient) = " + std::to_string(obj_current));

    // Gradient direction check: g·d should be < 0 for a descent direction
    Real g_dot_d = 0.0;
    for (int j = 0; j < n_active; ++j) {
        g_dot_d += g_active(j) * nlcg_direction_[j];
    }
#ifdef MAPLE3DMT_USE_MPI
    {
        Real g_dot_d_global;
        MPI_Allreduce(&g_dot_d, &g_dot_d_global, 1, MPI_DOUBLE, MPI_SUM, comm_);
        g_dot_d = g_dot_d_global;
    }
#endif
    INV_LOG_INFO("  g.d=" + std::to_string(g_dot_d) +
                 (g_dot_d >= 0 ? " WARN:ASCENT" : " OK:descent"));

    // 5. Line search with quadratic interpolation (pass dirderiv = g·d)
    Real alpha = line_search_(nlcg_direction_, obj_current, g_dot_d);

    INV_LOG_DEBUG("[PROGRESS] phase=Line_Search alpha=" + std::to_string(alpha));

    // Handle line search result
    Real rms, phi_model, phi_data_post;
    if (alpha == 0.0) {
        // LS failed: model was restored by line_search_, but predicted data is stale
        // (from last LS trial). Re-compute forward to restore consistent predicted data.
        INV_LOG_WARNING("  LS failed → recomputing forward + CG restart next iteration");
        nlcg_direction_.assign(n_active, 0.0);  // force β=0 (restart)
        fwd_->compute_responses(*data_, *data_);
        phi_data_post = data_misfit_from_predicted_();
        phi_model = phi_model_cur;
        rms = data_->rms_misfit();
    } else {
        // line_search_ updated model and left predicted data from its last
        // successful objective_() call → consistent.
        model_->invalidate_cache();
        rms = data_->rms_misfit();
        phi_model = reg_->evaluate(*model_);
        phi_data_post = data_misfit_from_predicted_();
    }

    IterationLog3D entry;
    entry.iteration = iter;
    entry.objective = phi_data_post + current_lambda_ * phi_model;
    entry.data_misfit = phi_data_post;
    entry.model_norm = phi_model;
    entry.rms = rms;
    entry.lambda = current_lambda_;
    entry.step_length = alpha;
    entry.cg_iterations = 0;  // NLCG: no inner CG
    entry.peak_memory_gb = utils::current_rss_gb();

    history_.push_back(entry);
    log_iteration_(iter, entry);

    if (iter_callback_) {
        iter_callback_(iter, entry);
    }

    // 7. Save gradient + preconditioned gradient for next iteration's β
    nlcg_prev_grad_.resize(n_active);
    nlcg_prev_precond_grad_.resize(n_active);
    for (int j = 0; j < n_active; ++j) {
        nlcg_prev_grad_[j] = g_active(j);
        nlcg_prev_precond_grad_[j] = h_active(j);
    }
}

// =========================================================================
// Main inversion loop with solver selection
// =========================================================================
void Inversion3D::run() {
    const char* solver_name =
        (params_.solver == InversionSolver::NLCG) ? "NLCG" : "GN-CG";

    INV_LOG_INFO("Starting 3D inversion (" + std::string(solver_name) +
                   "): max_iter=" + std::to_string(params_.max_iterations) +
                   ", target_rms=" + std::to_string(params_.target_rms));

    MemoryProfiler mem_prof;
    mem_prof.snap("inversion_start");

    // Set up frequency progress for forward phases
    if (freq_progress_cb_) {
        fwd_->set_freq_progress_callback(
            [this](int fi, int nf, Real fhz, const std::string&) {
                freq_progress_cb_(fi, nf, fhz, "Forward");
            });
    }

    // Initial forward solve
    fwd_->compute_responses(*data_, *data_);
    Real rms0 = data_->rms_misfit();
    INV_LOG_INFO("Initial RMS: " + std::to_string(rms0));
    mem_prof.snap("after_initial_forward");

    for (int iter = 0; iter < params_.max_iterations; ++iter) {
        INV_LOG_DEBUG("------ " + std::string(solver_name) +
                       " Iteration " + std::to_string(iter) + " ------");

        // Memory guard
        {
            static const size_t mem_limit = compute_memory_limit();
            double rss_gb = utils::current_rss_gb();
            double avail_gb = utils::available_memory_gb();
            double limit_gb = static_cast<double>(mem_limit) / (1024.0*1024.0*1024.0);

            INV_LOG_DEBUG("Memory: RSS=" + utils::fmt_mem_gb(rss_gb) +
                           " avail=" + utils::fmt_mem_gb(avail_gb));

            if (MemoryProfiler::check_limit(mem_limit, comm_)) {
                INV_LOG_WARNING("Memory limit exceeded! Saving checkpoint.");
                if (params_.save_checkpoints) {
                    try { save_checkpoint_(iter); }
                    catch (...) {}
                }
                break;
            }
        }

        // Solver dispatch
        if (params_.solver == InversionSolver::NLCG) {
            nlcg_step_(iter);
        } else {
            gauss_newton_cg_step_(iter);
        }

        mem_prof.snap("iter_" + std::to_string(iter));

        // Check convergence — use values already computed and logged
        // by nlcg_step_ or gauss_newton_cg_step_ (avoids redundant forward solve)
        Real rms = history_.empty() ? data_->rms_misfit() : history_.back().rms;
        Real obj = history_.empty() ? 0.0 : history_.back().objective;
        INV_LOG_INFO("[PROGRESS] phase=Iteration iter=" +
                       std::to_string(iter + 1) + "/" +
                       std::to_string(params_.max_iterations) +
                       " rms=" + std::to_string(rms) +
                       " obj=" + std::to_string(obj) +
                       " lambda=" + std::to_string(current_lambda_));

        if (rms <= params_.target_rms) {
            INV_LOG_INFO("Target RMS reached: " + std::to_string(rms));
            break;
        }

        // Stagnation detection with recovery strategy
        if (history_.size() >= 4) {
            bool stagnated = true;
            for (size_t k = history_.size() - 3; k < history_.size(); ++k) {
                Real r_prev = history_[k - 1].rms;
                Real r_curr = history_[k].rms;
                if (r_prev > 0.0 &&
                    std::abs(r_prev - r_curr) / r_prev > 1e-3) {
                    stagnated = false;
                    break;
                }
            }
            if (stagnated) {
                stagnation_count_++;
                if (stagnation_count_ >= 3) {
                    // 3rd stagnation → truly stuck, stop
                    INV_LOG_WARNING("Stagnation x3 → stopping.");
                    break;
                }
                // Recovery: halve lambda + force steepest descent restart
                current_lambda_ *= 0.5;
                nlcg_direction_.clear();   // force β=0 next iter
                nlcg_prev_grad_.clear();
                nlcg_prev_alpha_ = 0;
                INV_LOG_INFO("Stagnation #" + std::to_string(stagnation_count_) +
                            " → lambda/=2 (" + std::to_string(current_lambda_) +
                            "), NLCG restart");
                continue;  // skip normal lambda update
            } else {
                stagnation_count_ = 0;  // reset on progress
            }
        }

        // Update regularization parameter
        update_lambda_();

        // Checkpoint
        if (params_.save_checkpoints &&
            (iter + 1) % params_.checkpoint_every == 0) {
            save_checkpoint_(iter);
        }
    }

    mem_prof.snap("inversion_complete");
    mem_prof.report(comm_);

    INV_LOG_INFO("Inversion completed. Final RMS: " +
                   std::to_string(data_->rms_misfit()));
}

} // namespace inversion
} // namespace maple3dmt
