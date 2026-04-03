// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#include "maple3dmt/forward/cocg.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/utils/logger.h"
#include <cstdio>

namespace maple3dmt {
namespace forward {

COCGResult COCGSolver::solve(const ComplexVec& b, ComplexVec& x) const {
    COCGResult result;
    const int n = static_cast<int>(b.size());

    if (!matvec_) {
        result.info = "No operator set";
        return result;
    }

    // r = b - A*x
    ComplexVec Ap(n);
    matvec_(x, Ap);

    ComplexVec r(n);
    for (int i = 0; i < n; ++i) r[i] = b[i] - Ap[i];

    Real bnorm = norm(b);
    if (bnorm < 1e-30) bnorm = 1.0;

    Real rnorm = norm(r);
    if (rnorm / bnorm < tol_) {
        result.converged = true;
        result.residual = rnorm / bnorm;
        result.iterations = 0;
        result.info = "Already converged";
        return result;
    }

    // z = M^{-1} r
    ComplexVec z(n);
    if (has_precond_) {
        precond_(r, z);
    } else {
        z = r;
    }

    // p = z
    ComplexVec p = z;

    // rz = r^T z (unconjugated bilinear form)
    Complex rz = bdot(r, z);

    for (int iter = 1; iter <= maxiter_; ++iter) {
        // Ap = A * p
        matvec_(p, Ap);

        // pAp = p^T A p (unconjugated)
        Complex pAp = bdot(p, Ap);

        // Breakdown check
        if (std::abs(pAp) < 1e-30) {
            result.info = "COCG breakdown: p^T A p = 0 at iter " + std::to_string(iter);
            result.iterations = iter;
            result.residual = norm(r) / bnorm;
            return result;
        }

        Complex alpha = rz / pAp;

        // x = x + alpha * p
        for (int i = 0; i < n; ++i) x[i] += alpha * p[i];

        // r = r - alpha * Ap
        for (int i = 0; i < n; ++i) r[i] -= alpha * Ap[i];

        rnorm = norm(r);  // Hermitian norm for convergence check
        Real rel_res = rnorm / bnorm;

        if (print_level_ >= 2 || (print_level_ >= 1 && iter % 50 == 0)) {
            char buf[64]; snprintf(buf, sizeof(buf), "%.3e", rel_res);
            MAPLE3DMT_LOG_INFO("  COCG iter=" + std::to_string(iter) +
                              " rel_res=" + std::string(buf));
        }

        if (rel_res < tol_) {
            result.converged = true;
            result.residual = rel_res;
            result.iterations = iter;
            result.info = "Converged";
            return result;
        }

        // z = M^{-1} r
        if (has_precond_) {
            precond_(r, z);
        } else {
            z = r;
        }

        // rz_new = r^T z (unconjugated)
        Complex rz_new = bdot(r, z);

        // Breakdown check
        if (std::abs(rz) < 1e-30) {
            result.info = "COCG breakdown: r^T z = 0 at iter " + std::to_string(iter);
            result.iterations = iter;
            result.residual = rel_res;
            return result;
        }

        Complex beta = rz_new / rz;
        rz = rz_new;

        // p = z + beta * p
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    }

    result.iterations = maxiter_;
    result.residual = norm(r) / bnorm;
    result.info = "Max iterations reached";
    return result;
}

} // namespace forward
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
