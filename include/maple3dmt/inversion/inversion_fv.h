// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file inversion_fv.h
/// @brief 3D MT inversion driver for Octree FV backend.
///
/// Same algorithm as Inversion3D (NLCG / GN-CG, quadratic line search,
/// Occam lambda strategy) but using:
///   - ForwardSolverFV (IForwardSolver interface)
///   - RegularizationOctree
///   - ImplicitJtJOperatorFV
///   - ComplexVec instead of mfem::Vector
///
/// Key advantage: A^T = A → true CG for GN inner solve (not FGMRES).

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/data/static_shift.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/forward/iforward_solver.h"
#include "maple3dmt/inversion/regularization_octree.h"
#include "maple3dmt/utils/freq_parallel.h"
#include <functional>

#ifdef MAPLE3DMT_USE_MPI
#include <mpi.h>
#endif

namespace maple3dmt {
namespace inversion {

/// Inversion parameters (shared with InversionParams3D where applicable).
struct InversionParamsFV {
    enum class Solver { GN_CG, NLCG, LBFGS };
    Solver solver = Solver::NLCG;

    int    max_iterations    = 50;
    Real   target_rms        = 1.0;
    Real   lambda_init       = 10.0;
    Real   lambda_decrease   = 0.6;

    // Lambda strategy: "ratio" (current, aggressive) or "plateau" (NewMT-style)
    enum class LambdaStrategy { RATIO, PLATEAU };
    LambdaStrategy lambda_strategy = LambdaStrategy::PLATEAU;

    // Plateau strategy parameters (used when lambda_strategy == PLATEAU):
    //   Hold lambda fixed while RMS improves by >= plateau_tol per iteration.
    //   After plateau_patience consecutive slow iterations, reduce by plateau_decrease.
    Real   plateau_tol       = 0.02;   // |ΔRMS/RMS| threshold for "slow improvement"
    int    plateau_patience   = 2;      // consecutive slow iters before decrease
    Real   plateau_decrease   = 0.5;    // lambda multiplier on plateau detection

    // GN-CG inner solver (true CG, not FGMRES — A^T=A!)
    int    cg_max_iter       = 30;
    Real   cg_tolerance      = 0.1;
    bool   cg_adaptive_tol   = true;

    // NLCG
    Real   nlcg_reset_threshold = 0.1;
    int    nlcg_reset_every     = 10;

    // L-BFGS
    int    lbfgs_memory         = 7;   // number of stored (s,y) pairs

    // Line search
    int    linesearch_max       = 6;    // max forward evals
    Real   linesearch_beta      = 0.5;  // backtracking factor
    Real   linesearch_startdm   = 20.0; // initial step for NLCG (||dm*α||)
    Real   linesearch_c1        = 1e-4; // Armijo sufficient decrease param

    // Model bounds (log-space)
    Real   log_sigma_min = -13.8;
    Real   log_sigma_max =   6.9;

    // Static shift
    bool   enable_static_shift = false;

    // Checkpoint
    bool   save_checkpoints = true;
    int    checkpoint_every  = 1;
    fs::path checkpoint_dir  = "checkpoints";
};

/// Iteration log entry.
struct IterationLogFV {
    int  iteration;
    Real objective;
    Real data_misfit;
    Real model_norm;
    Real rms;
    Real lambda;
    Real step_length;
    int  cg_iterations;
};

/// FV-based 3D MT inversion driver.
class InversionFV {
public:
    InversionFV() = default;

    /// Configure.
    void setup(model::ConductivityModel& model,
               data::MTData& data,
               forward::IForwardSolver& fwd,
               RegularizationOctree& reg,
               const InversionParamsFV& params);

    using IterCallback = std::function<void(int, const IterationLogFV&)>;
    void set_iteration_callback(IterCallback cb) { iter_cb_ = std::move(cb); }

    using FreqProgressCB = std::function<void(int, int, Real, const std::string&)>;
    void set_freq_progress_callback(FreqProgressCB cb) { freq_cb_ = std::move(cb); }

    /// Run inversion loop.
    void run();

    /// Resume from a given iteration (sets starting iter and lambda).
    /// Call after setup() but before run().
    void resume_from(int start_iter, Real lambda);

    /// Access history.
    const std::vector<IterationLogFV>& history() const { return history_; }

    /// Access frequency-parallel info (for CSV export in callbacks).
    bool is_freq_parallel() const { return freq_parallel_active_; }
    const utils::FreqParallelManager& freq_manager() const { return fpm_; }

    /// Compute gradient (public for gradient check).
    RealVec gradient() { return compute_gradient_(); }

    /// Compute objective (public for gradient check).
    Real objective() { return objective_(); }

private:
    model::ConductivityModel*  model_ = nullptr;
    data::MTData*              data_  = nullptr;
    forward::IForwardSolver*   fwd_   = nullptr;
    RegularizationOctree*      reg_   = nullptr;
    InversionParamsFV          params_;
    std::vector<IterationLogFV> history_;
    Real current_lambda_ = 10.0;
    int  start_iteration_ = 0;     // resume offset (0 = fresh start)
    int  stagnation_count_ = 0;
    int  plateau_count_ = 0;       // consecutive slow-improvement iterations
    RealVec cached_grad_;

    IterCallback iter_cb_;
    FreqProgressCB freq_cb_;

    // Frequency-parallel manager (MPI)
    utils::FreqParallelManager fpm_;
    bool freq_parallel_active_ = false;

#ifdef MAPLE3DMT_USE_MPI
    MPI_Comm comm_ = MPI_COMM_WORLD;
#endif

    void nlcg_step_(int iter);
    void gn_cg_step_(int iter);
    void lbfgs_step_(int iter);
    RealVec compute_gradient_();
    /// Line search strategies per solver type.
    enum class LSMode { NLCG, LBFGS, GN };
    Real line_search_(const RealVec& dm, Real obj_current, Real dirderiv,
                      LSMode mode);
    Real objective_();
    Real data_misfit_from_predicted_() const;
    Real data_misfit_for_frequencies_(const std::vector<int>& freq_indices) const;
    Real compute_rms_() const;
    Real compute_rms_for_frequencies_(const std::vector<int>& freq_indices,
                                      Real& chi2_out, int& nd_out) const;
    void update_lambda_();
    void save_checkpoint_(int iter);
    void log_iteration_(int iter, const IterationLogFV& entry);

    // NLCG state
    RealVec nlcg_prev_grad_;
    RealVec nlcg_prev_precond_grad_;
    RealVec nlcg_direction_;
    Real    nlcg_prev_alpha_ = 0.0;

    // L-BFGS state: circular buffer of (s, y) pairs
    // s_k = x_{k+1} - x_k,  y_k = g_{k+1} - g_k
    std::vector<RealVec> lbfgs_s_;    // model differences
    std::vector<RealVec> lbfgs_y_;    // gradient differences
    std::vector<Real>    lbfgs_rho_;  // 1 / (y_k · s_k)
    RealVec lbfgs_prev_grad_;
    RealVec lbfgs_prev_params_;       // previous active-space model
    int lbfgs_stored_ = 0;            // number of pairs stored
    int lbfgs_oldest_ = 0;            // circular buffer index
};

} // namespace inversion
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
