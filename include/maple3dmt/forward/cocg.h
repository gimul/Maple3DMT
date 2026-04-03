// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file cocg.h
/// @brief COCG (Conjugate-Orthogonal Conjugate Gradient) for complex symmetric systems.
///
/// For A = A^T (complex symmetric, NOT Hermitian), uses unconjugated bilinear
/// form (u,v) = u^T v instead of Hermitian inner product u^H v.
///
/// Key property: A = A^T → (Au,v) = u^T A^T v = u^T A v = (u, Av)
/// so the bilinear form is preserved by A, enabling CG-like iteration.
///
/// Cost: 1 matvec per iteration (vs BiCGStab's 2).
/// Convergence: smooth, CG-like (vs BiCGStab's erratic).
///
/// Reference: van der Vorst & Melissen (1990), "A Petrov-Galerkin type method
///   for solving Ax=b, where A is symmetric complex."

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include <functional>
#include <cmath>
#include <string>

namespace maple3dmt {
namespace forward {

/// Result of a COCG solve.
struct COCGResult {
    int    iterations = 0;
    Real   residual   = 0.0;
    bool   converged  = false;
    std::string info;
};

/// COCG solver for complex symmetric N×N systems.
///
/// Algorithm (preconditioned COCG):
///   r_0 = b - A x_0
///   z_0 = M^{-1} r_0
///   p_0 = z_0
///   for k = 0, 1, 2, ...
///     alpha_k = (r_k^T z_k) / (p_k^T A p_k)     ← unconjugated!
///     x_{k+1} = x_k + alpha_k p_k
///     r_{k+1} = r_k - alpha_k A p_k
///     z_{k+1} = M^{-1} r_{k+1}
///     beta_k  = (r_{k+1}^T z_{k+1}) / (r_k^T z_k)
///     p_{k+1} = z_{k+1} + beta_k p_k
///
/// Note: alpha, beta are COMPLEX (not real as in standard CG).
/// Breakdown: (p^T A p) = 0 is possible but rare in practice.
class COCGSolver {
public:
    using MatVec = std::function<void(const ComplexVec&, ComplexVec&)>;

    void set_operator(MatVec op) { matvec_ = std::move(op); }
    void set_preconditioner(MatVec pc) { precond_ = std::move(pc); has_precond_ = true; }

    void set_tolerance(Real tol) { tol_ = tol; }
    void set_max_iterations(int maxiter) { maxiter_ = maxiter; }
    void set_print_level(int level) { print_level_ = level; }

    /// Solve A*x = b. x is initial guess on input, solution on output.
    COCGResult solve(const ComplexVec& b, ComplexVec& x) const;

private:
    MatVec matvec_;
    MatVec precond_;
    bool   has_precond_ = false;
    Real   tol_         = 1e-7;
    int    maxiter_     = 2000;
    int    print_level_ = 0;

    /// Unconjugated bilinear form: (u,v) = u^T v = Σ u_i * v_i
    static Complex bdot(const ComplexVec& a, const ComplexVec& b_vec) {
        Complex s(0, 0);
        for (size_t i = 0; i < a.size(); ++i)
            s += a[i] * b_vec[i];  // NO conjugate!
        return s;
    }

    /// Standard Hermitian norm: ||v|| = sqrt(v^H v)
    static Real norm(const ComplexVec& v) {
        Real s = 0;
        for (auto& vi : v) s += std::norm(vi);  // |vi|^2
        return std::sqrt(s);
    }
};

} // namespace forward
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
