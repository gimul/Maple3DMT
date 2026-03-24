// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file qmr.h
/// @brief QMR (Quasi-Minimal Residual) solver for complex symmetric systems.
///
/// Exploits A = A^T (complex symmetric, NOT Hermitian) to reduce cost:
///   - Lanczos with A^T = A → single matvec per iteration
///   - Smooth convergence curve (no BiCGStab-style stalling)
///   - Work vectors: 6 (vs BiCGStab's 8)
///
/// Based on: Freund & Nachtigal (1991), "QMR: a quasi-minimal residual
///   method for non-Hermitian linear systems", adapted for complex symmetric.
///
/// ModEM uses this solver for the same reason: A = C^T Mf^{-1} C + iωMeσ
/// is complex symmetric, and QMR gives much smoother convergence than BiCGStab.

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include <functional>
#include <cmath>
#include <string>

namespace maple3dmt {
namespace forward {

/// Result of a QMR solve.
struct QMRResult {
    int    iterations = 0;
    Real   residual   = 0.0;
    bool   converged  = false;
    std::string info;
};

/// QMR solver for complex symmetric N×N systems with optional preconditioner.
///
/// Solves: A x = b  where A = A^T (complex symmetric).
///
/// The Lanczos process for complex symmetric A uses:
///   w_{n+1} = A v_n - alpha_n v_n - beta_n v_{n-1}
/// with A^T = A, so only ONE matvec per iteration (vs 2 for general QMR).
///
/// The QMR approach minimizes ||r_n|| over a quasi-optimal subspace,
/// giving smooth monotone-like residual decrease (unlike BiCGStab).
class QMRSolver {
public:
    using MatVec = std::function<void(const ComplexVec&, ComplexVec&)>;

    void set_operator(MatVec op) { matvec_ = std::move(op); }
    void set_preconditioner(MatVec pc) { precond_ = std::move(pc); has_precond_ = true; }

    void set_tolerance(Real tol) { tol_ = tol; }
    void set_max_iterations(int maxiter) { maxiter_ = maxiter; }
    void set_print_level(int level) { print_level_ = level; }

    /// Solve A*x = b. x is initial guess on input, solution on output.
    QMRResult solve(const ComplexVec& b, ComplexVec& x) const;

private:
    MatVec matvec_;
    MatVec precond_;
    bool   has_precond_ = false;
    Real   tol_         = 1e-7;
    int    maxiter_     = 2000;
    int    print_level_ = 0;

    // Vector operations (same interface as BiCGStab for consistency)
    static Complex sym_dot(const ComplexVec& a, const ComplexVec& b_vec) {
        // For complex symmetric: use unconjugated dot product v^T w (NOT v^H w)
        // This is the correct bilinear form for A = A^T
        Complex s(0, 0);
        for (size_t i = 0; i < a.size(); ++i)
            s += a[i] * b_vec[i];
        return s;
    }
    static Real norm(const ComplexVec& v) {
        Real s = 0;
        for (auto& vi : v) s += std::norm(vi);  // |vi|^2
        return std::sqrt(s);
    }
};

} // namespace forward
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
