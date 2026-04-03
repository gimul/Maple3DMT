// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#include "maple3dmt/forward/cocr.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/utils/logger.h"
#include <cstdio>

namespace maple3dmt {
namespace forward {

COCRResult COCRSolver::solve(const ComplexVec& b, ComplexVec& x) const {
    COCRResult result;
    const int n = static_cast<int>(b.size());

    if (!matvec_) {
        result.info = "No operator set";
        return result;
    }

    // r_0 = b - A x_0
    ComplexVec Ax(n);
    matvec_(x, Ax);

    ComplexVec r(n);
    for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];

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

    // z_0 = M^{-1} r_0
    ComplexVec z(n);
    if (has_precond_) {
        precond_(r, z);
    } else {
        z = r;
    }

    // p_0 = z_0
    ComplexVec p = z;

    // Az_0 = A z_0
    ComplexVec Az(n);
    matvec_(z, Az);

    // Ap_0 = Az_0
    ComplexVec Ap = Az;

    // rAz = (Az)^T r  (unconjugated bilinear form)
    Complex rAz = bdot(Az, r);

    for (int iter = 1; iter <= maxiter_; ++iter) {
        // alpha = (Az^T r) / (Ap^T Ap)
        Complex ApAp = bdot(Ap, Ap);

        if (std::abs(ApAp) < 1e-30) {
            result.info = "COCR breakdown: (Ap)^T(Ap) = 0 at iter " + std::to_string(iter);
            result.iterations = iter;
            result.residual = norm(r) / bnorm;
            return result;
        }

        Complex alpha = rAz / ApAp;

        // x_{k+1} = x_k + alpha * p_k
        for (int i = 0; i < n; ++i) x[i] += alpha * p[i];

        // r_{k+1} = r_k - alpha * Ap_k
        for (int i = 0; i < n; ++i) r[i] -= alpha * Ap[i];

        rnorm = norm(r);
        Real rel_res = rnorm / bnorm;

        if (print_level_ >= 2 || (print_level_ >= 1 && iter % 50 == 0)) {
            char buf[64]; snprintf(buf, sizeof(buf), "%.3e", rel_res);
            MAPLE3DMT_LOG_INFO("  COCR iter=" + std::to_string(iter) +
                              " rel_res=" + std::string(buf));
        }

        if (rel_res < tol_) {
            result.converged = true;
            result.residual = rel_res;
            result.iterations = iter;
            result.info = "Converged";
            return result;
        }

        // z_{k+1} = M^{-1} r_{k+1}
        if (has_precond_) {
            precond_(r, z);
        } else {
            z = r;
        }

        // Az_{k+1} = A z_{k+1}   ← the 1 matvec per iteration
        matvec_(z, Az);

        // rAz_new = (Az_{k+1})^T r_{k+1}
        Complex rAz_new = bdot(Az, r);

        if (std::abs(rAz) < 1e-30) {
            result.info = "COCR breakdown: (Az)^T r = 0 at iter " + std::to_string(iter);
            result.iterations = iter;
            result.residual = rel_res;
            return result;
        }

        Complex beta = rAz_new / rAz;
        rAz = rAz_new;

        // p_{k+1} = z_{k+1} + beta * p_k
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];

        // Ap_{k+1} = Az_{k+1} + beta * Ap_k   ← no extra matvec!
        for (int i = 0; i < n; ++i) Ap[i] = Az[i] + beta * Ap[i];
    }

    result.iterations = maxiter_;
    result.residual = norm(r) / bnorm;
    result.info = "Max iterations reached";
    return result;
}

} // namespace forward
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
