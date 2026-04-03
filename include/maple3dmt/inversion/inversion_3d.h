// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file inversion_3d.h
/// @brief 3D MT inversion driver using CG-based Gauss-Newton.
///
/// Key differences from 2.5D (NewMT):
///   - Jacobian is NEVER stored explicitly (matrix-free)
///   - Normal equations solved via CG (not dense inverse)
///   - Each CG iteration = 1 forward + 1 adjoint solve
///   - Memory scales as O(N_dof), not O(N_data × N_params)
///
/// Minimises:
///   Φ(m) = ||W_d (d_obs - d_pred(m))||² + λ ||W_m (m - m_ref)||²
///
/// Using Gauss-Newton with CG inner solve:
///   (J^T W_d^T W_d J + λ W_m^T W_m) δm = J^T W_d^T W_d r - λ W_m^T W_m (m - m_ref)
///   where each CG matvec computes (JtJ + λ WtW)·p via implicit J·p and J^T·(J·p)

#include "maple3dmt/common.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/data/static_shift.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/regularization/regularization.h"
#include "maple3dmt/forward/forward_solver_3d.h"
#include "maple3dmt/utils/freq_parallel.h"
#include <functional>
#include <memory>
#ifdef MAPLE3DMT_USE_MPI
#include <mpi.h>
#endif

namespace maple3dmt {
namespace inversion {

/// Solver type for 3D inversion.
enum class InversionSolver {
    GN_CG,   ///< Gauss-Newton with CG inner solver (original)
    NLCG,    ///< Nonlinear Conjugate Gradient (recommended for large meshes)
};

/// Inversion configuration for 3D.
struct InversionParams3D {
    InversionSolver solver = InversionSolver::NLCG;  // default: NLCG

    int    max_iterations    = 50;
    Real   target_rms        = 1.0;
    Real   lambda_init       = 10.0;
    Real   lambda_decrease   = 0.8;

    // GN-CG inner solver (only used when solver == GN_CG)
    // Uses FGMRES (not CG) because iterative adjoint breaks JᵀJ symmetry
    int    cg_max_iter       = 30;
    int    cg_kdim           = 20;     // FGMRES Krylov subspace dimension
    Real   cg_tolerance      = 0.1;
    bool   cg_adaptive_tol   = true;

    // NLCG parameters
    Real   nlcg_reset_threshold = 0.1;  // reset β if |g_new·g_old|/|g_new|² > this
    int    nlcg_reset_every     = 10;   // force reset every N iterations

    // Line search (used by both)
    int    linesearch_max    = 10;      // quadratic interp + backtracking fallback
    Real   linesearch_beta   = 0.5;
    Real   linesearch_alpha0 = 1.0;     // initial step length (legacy, overridden by startdm)
    Real   linesearch_startdm = 20.0;   // Initial step: α₀ = startdm / ||d|| (ModEM default)

    // Static shift (reused from 2.5D)
    bool   enable_static_shift = false;
    Real   shift_beta          = 1.0;
    Real   shift_s_max         = 2.0;
    std::vector<std::string> shift_stations;

    // Model parameter bounds (log-space).
    // log(σ) is clamped to [log_sigma_min, log_sigma_max] after each model update.
    // Defaults: σ ∈ [1e-6, 1e+3] S/m → log(σ) ∈ [-13.8, 6.9]
    Real   log_sigma_min       = -13.8;   // = ln(1e-6)
    Real   log_sigma_max       =   6.9;   // = ln(1e+3)

    // Phase-only mode
    bool   phase_only          = false;

    // Checkpoint
    bool   save_checkpoints    = true;
    int    checkpoint_every    = 1;
    fs::path checkpoint_dir    = "checkpoints";

    // Memory management
    bool   release_factor_between_freqs = true;

    // Frequency parallelism (2-level MPI)
    int    freq_parallel_spatial_procs = 0;  // 0=auto, >0=fixed spatial procs per group
};

/// Iteration log entry.
struct IterationLog3D {
    int  iteration;
    Real objective;
    Real data_misfit;
    Real model_norm;
    Real rms;
    Real lambda;
    Real step_length;
    int  cg_iterations;     // how many CG iters this GN step used
    Real peak_memory_gb;    // peak memory usage during this iteration
};

/// Implicit JtJ operator for CG.
///
/// Computes y = (J^T W_d^2 J + λ W_m^T W_m) x
/// without ever forming J explicitly.
///
/// Each application requires:
///   1. Jx: for each freq/source, solve A·δE = -δA·E (forward-like)
///   2. JtJx: for each freq/source, solve A^T·λ = Q^T·W^2·(Jx) (adjoint)
///   3. Add λ W_m^T W_m x
class ImplicitJtJOperator : public mfem::Operator {
public:
    ImplicitJtJOperator(forward::ForwardSolver3D& fwd,
                        const data::MTData& data,
                        const model::ConductivityModel& model,
                        const regularization::Regularization& reg,
                        Real lambda);

    /// Apply: y = (JtJ + λ WtW) x
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /// Track number of matvecs (= number of CG iterations).
    int num_matvecs() const { return num_matvecs_; }

    /// Cache background E fields for all frequencies.
    /// Must be called ONCE before the FGMRES loop begins.
    /// After this, Mult() uses cached fields instead of re-solving forward.
    void cache_background_fields();

    /// Check if background fields are cached.
    bool has_cached_fields() const { return !bg_fields_cache_.empty(); }

private:
    forward::ForwardSolver3D* fwd_;
    const data::MTData* data_;
    const model::ConductivityModel* model_;
    const regularization::Regularization* reg_;
    Real lambda_;
    mutable int num_matvecs_ = 0;

    // Pre-allocated work vectors (avoid repeated allocation in Mult)
    mutable RealVec work_delta_sigma_;
    mutable RealVec work_jtjx_global_;
    mutable RealVec work_data_weights_;
    mutable RealVec work_g1_, work_g2_;
    mutable RealVec work_x_full_, work_wtw_;

    // Cached ParGridFunctions (allocated once on first Mult call)
    mutable std::unique_ptr<mfem::ParGridFunction> work_lam1_r_, work_lam1_i_;
    mutable std::unique_ptr<mfem::ParGridFunction> work_lam2_r_, work_lam2_i_;
    mutable std::unique_ptr<mfem::ParGridFunction> work_dE_r_, work_dE_i_;
    mutable bool work_allocated_ = false;

    /// Per-frequency cached background E fields.
    /// Populated once by cache_background_fields(), used in every Mult() call.
    /// This eliminates the need to re-solve forward in each FGMRES iteration.
    struct FreqFieldCache {
        mfem::Vector E1_r, E1_i;  // Pol1 background field (true DOF vectors)
        mfem::Vector E2_r, E2_i;  // Pol2 background field (true DOF vectors)
    };
    std::vector<FreqFieldCache> bg_fields_cache_;

    void ensure_work_vectors_() const;
};

/// 3D Inversion driver.
class Inversion3D {
public:
    Inversion3D() = default;

    /// Configure the inversion.
    void setup(mfem::ParMesh& mesh,
               model::ConductivityModel& model,
               data::MTData& data,
               forward::ForwardSolver3D& fwd,
               regularization::Regularization& reg,
               const InversionParams3D& params);

    /// Callback invoked after each iteration.
    using IterCallback = std::function<void(int, const IterationLog3D&)>;
    void set_iteration_callback(IterCallback cb) { iter_callback_ = std::move(cb); }

    /// Frequency progress callback: (freq_idx, total_freq, freq_hz, phase_name).
    using FreqProgressCallback =
        std::function<void(int, int, Real, const std::string&)>;
    void set_freq_progress_callback(FreqProgressCallback cb) {
        freq_progress_cb_ = std::move(cb);
    }

    /// Run the inversion loop.
    void run();

    /// Resume from checkpoint.
    void resume(const fs::path& checkpoint_path);

    /// Access history.
    const std::vector<IterationLog3D>& history() const { return history_; }

    /// Get final model.
    const model::ConductivityModel& final_model() const { return *model_; }

    /// Compute gradient (public for gradient check tests).
    RealVec gradient() { return compute_gradient_adjoint_(); }

    /// Compute objective (public for gradient check tests).
    Real objective() { return objective_(); }

private:
    mfem::ParMesh*                     mesh_   = nullptr;
    model::ConductivityModel*          model_  = nullptr;
    data::MTData*                      data_   = nullptr;
    forward::ForwardSolver3D*          fwd_    = nullptr;
    regularization::Regularization*    reg_    = nullptr;
    InversionParams3D                  params_;
    std::vector<IterationLog3D>        history_;
    Real current_lambda_ = 10.0;
    int  consecutive_ls_fail_ = 0;
    int  stagnation_count_ = 0;
    RealVec cached_grad_;              // reuse gradient when λ changes (skip fwd+adj)
    IterCallback iter_callback_;
    FreqProgressCallback freq_progress_cb_;

    std::unique_ptr<data::StaticShiftParams> shift_;

    /// 2-level MPI: frequency groups × spatial decomposition.
    /// Initialized in run() if freq_parallel_spatial_procs > 0 or auto.
    utils::FreqParallelManager freq_parallel_;

#ifdef MAPLE3DMT_USE_MPI
    MPI_Comm comm_ = MPI_COMM_WORLD;
#endif

    /// One Gauss-Newton step with CG inner solve.
    void gauss_newton_cg_step_(int iter);

    /// One NLCG step (Polak-Ribière with restart).
    void nlcg_step_(int iter);

    /// Compute gradient: g = J^T W_d^2 (d_pred - d_obs) + λ W_m^T W_m (m - m_ref)
    RealVec compute_gradient_adjoint_();

    /// Line search with quadratic interpolation.
    /// @param dirderiv  directional derivative g·dm (< 0 for descent). 0 = unknown.
    Real line_search_(const RealVec& dm, Real obj_current, Real dirderiv = 0.0);

    /// Compute objective function value (full forward solve + regularization).
    Real objective_();

    /// Compute data misfit only from current predicted data (no forward solve).
    Real data_misfit_from_predicted_() const;

    void save_checkpoint_(int iter);
    void load_checkpoint_(const fs::path& path);
    void update_lambda_();
    void log_iteration_(int iter, const IterationLog3D& entry);

    // NLCG state
    RealVec nlcg_prev_grad_;
    RealVec nlcg_prev_precond_grad_;  // h_prev = CmCm^T * g_prev
    RealVec nlcg_direction_;
    Real    nlcg_prev_alpha_ = 0.0;  // previous successful step size (0 = use startdm/||d||)
};

} // namespace inversion
} // namespace maple3dmt
