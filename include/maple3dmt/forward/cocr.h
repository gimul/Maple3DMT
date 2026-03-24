// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file cocr.h
/// @brief COCR (Conjugate-Orthogonal Conjugate Residual) for complex symmetric systems.
///
/// For A = A^T (complex symmetric). Minimizes residual norm, unlike COCG which
/// minimizes error in the A-bilinear form.
///
/// Advantages over COCG:
///   - Near-monotone residual decrease
///   - Breakdown far less likely: denominator is (Ap)^T(Ap) vs COCG's p^T(Ap)
///   - Same cost: 1 matvec per iteration
///
/// Reference: Sogabe & Zhang (2007), "A COCR method for solving complex
///   symmetric linear systems."

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include <functional>
#include <cmath>
#include <string>

namespace maple3dmt {
namespace forward {

struct COCRResult {
    int    iterations = 0;
    Real   residual   = 0.0;
    bool   converged  = false;
    std::string info;
};

/// COCR solver for complex symmetric N×N systems.
///
/// Algorithm (preconditioned COCR):
///   r_0 = b - A x_0;  z_0 = M^{-1} r_0;  p_0 = z_0
///   Az_0 = A z_0;  Ap_0 = Az_0
///   for k = 0, 1, 2, ...
///     alpha_k = (Az_k^T r_k) / (Ap_k^T Ap_k)      ← unconjugated
///     x_{k+1} = x_k + alpha_k p_k
///     r_{k+1} = r_k - alpha_k Ap_k
///     z_{k+1} = M^{-1} r_{k+1}
///     Az_{k+1} = A z_{k+1}                          ← 1 matvec
///     beta_k  = (Az_{k+1}^T r_{k+1}) / (Az_k^T r_k)
///     p_{k+1} = z_{k+1} + beta_k p_k
///     Ap_{k+1} = Az_{k+1} + beta_k Ap_k             ← no extra matvec
class COCRSolver {
public:
    using MatVec = std::function<void(const ComplexVec&, ComplexVec&)>;

    void set_operator(MatVec op) { matvec_ = std::move(op); }
    void set_preconditioner(MatVec pc) { precond_ = std::move(pc); has_precond_ = true; }

    void set_tolerance(Real tol) { tol_ = tol; }
    void set_max_iterations(int maxiter) { maxiter_ = maxiter; }
    void set_print_level(int level) { print_level_ = level; }

    COCRResult solve(const ComplexVec& b, ComplexVec& x) const;

private:
    MatVec matvec_;
    MatVec precond_;
    bool   has_precond_ = false;
    Real   tol_         = 1e-7;
    int    maxiter_     = 2000;
    int    print_level_ = 0;

    /// Unconjugated bilinear form: (u,v) = u^T v
    static Complex bdot(const ComplexVec& a, const ComplexVec& b_vec) {
        Complex s(0, 0);
        for (size_t i = 0; i < a.size(); ++i)
            s += a[i] * b_vec[i];
        return s;
    }

    static Real norm(const ComplexVec& v) {
        Real s = 0;
        for (auto& vi : v) s += std::norm(vi);
        return std::sqrt(s);
    }
};

} // namespace forward
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
