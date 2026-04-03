// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file bicgstab.h
/// @brief BiCGStab iterative solver for complex sparse systems.
///
/// Self-contained implementation (no PETSc/MFEM dependency).
/// Used by ForwardSolverFV for solving A*E = b where A is complex symmetric.

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include <functional>
#include <cmath>
#include <string>

namespace maple3dmt {
namespace forward {

/// Result of a BiCGStab solve.
struct BiCGStabResult {
    int    iterations = 0;
    Real   residual   = 0.0;
    bool   converged  = false;
    std::string info;
};

/// BiCGStab solver for complex N×N systems with optional preconditioner.
///
/// Solves: A x = b
/// Preconditioner M ≈ A, applied as M^{-1} (left preconditioning).
///
/// Usage:
///   BiCGStabSolver solver;
///   solver.set_operator([&](const ComplexVec& in, ComplexVec& out) {
///       A.matvec(in, out);
///   });
///   solver.set_preconditioner([&](const ComplexVec& in, ComplexVec& out) {
///       // M^{-1} * in → out  (e.g., Jacobi: out[i] = in[i] / A[i][i])
///       for (int i = 0; i < n; ++i) out[i] = in[i] / diag[i];
///   });
///   auto result = solver.solve(b, x);
class BiCGStabSolver {
public:
    using MatVec = std::function<void(const ComplexVec&, ComplexVec&)>;

    void set_operator(MatVec op) { matvec_ = std::move(op); }
    void set_preconditioner(MatVec pc) { precond_ = std::move(pc); has_precond_ = true; }

    void set_tolerance(Real tol) { tol_ = tol; }
    void set_max_iterations(int maxiter) { maxiter_ = maxiter; }
    void set_print_level(int level) { print_level_ = level; }

    /// Solve A*x = b. x is initial guess on input, solution on output.
    BiCGStabResult solve(const ComplexVec& b, ComplexVec& x) const;

private:
    MatVec matvec_;
    MatVec precond_;
    bool   has_precond_ = false;
    Real   tol_         = 1e-7;
    int    maxiter_     = 2000;
    int    print_level_ = 0;

    // Vector operations
    static Complex dot(const ComplexVec& a, const ComplexVec& b_vec) {
        Complex s(0, 0);
        for (size_t i = 0; i < a.size(); ++i)
            s += std::conj(a[i]) * b_vec[i];
        return s;
    }
    static Real norm(const ComplexVec& v) {
        Real s = 0;
        for (auto& vi : v) s += std::norm(vi);  // |vi|^2
        return std::sqrt(s);
    }
    static void axpy(Complex a, const ComplexVec& x_vec, ComplexVec& y) {
        for (size_t i = 0; i < y.size(); ++i)
            y[i] += a * x_vec[i];
    }
};

} // namespace forward
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
