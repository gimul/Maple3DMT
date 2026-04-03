// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#include "maple3dmt/forward/bicgstab.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/utils/logger.h"
#include <sstream>
#include <iomanip>

namespace maple3dmt {
namespace forward {

BiCGStabResult BiCGStabSolver::solve(const ComplexVec& b, ComplexVec& x) const {
    BiCGStabResult result;
    const int n = static_cast<int>(b.size());

    if (!matvec_) {
        result.info = "No operator set";
        return result;
    }

    // r = b - A*x
    ComplexVec r(n), r0(n);
    ComplexVec p(n), v(n), s(n), t(n);
    ComplexVec p_hat(n), s_hat(n);  // preconditioned vectors

    // Initial residual: r = b - A*x
    matvec_(x, v);  // v = A*x
    for (int i = 0; i < n; ++i) r[i] = b[i] - v[i];

    // r0 = r (shadow residual, fixed throughout)
    r0 = r;

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

    Complex rho_old(1, 0), alpha(1, 0), omega(1, 0);

    for (int iter = 1; iter <= maxiter_; ++iter) {
        Complex rho = dot(r0, r);

        if (std::abs(rho) < 1e-30) {
            result.info = "BiCGStab breakdown: rho=0 at iter " + std::to_string(iter);
            result.iterations = iter;
            result.residual = norm(r) / bnorm;
            return result;
        }

        if (iter == 1) {
            p = r;
        } else {
            Complex beta = (rho / rho_old) * (alpha / omega);
            // p = r + beta * (p - omega * v)
            for (int i = 0; i < n; ++i)
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // Apply preconditioner: p_hat = M^{-1} * p
        if (has_precond_) {
            precond_(p, p_hat);
        } else {
            p_hat = p;
        }

        // v = A * p_hat
        matvec_(p_hat, v);

        Complex r0v = dot(r0, v);
        if (std::abs(r0v) < 1e-30) {
            result.info = "BiCGStab breakdown: (r0,v)=0 at iter " + std::to_string(iter);
            result.iterations = iter;
            result.residual = norm(r) / bnorm;
            return result;
        }

        alpha = rho / r0v;

        // s = r - alpha * v
        for (int i = 0; i < n; ++i)
            s[i] = r[i] - alpha * v[i];

        // Check if s is small enough to skip second half
        Real snorm = norm(s);
        if (snorm / bnorm < tol_) {
            // x = x + alpha * p_hat
            axpy(alpha, p_hat, x);
            result.converged = true;
            result.residual = snorm / bnorm;
            result.iterations = iter;
            result.info = "Converged (early)";
            return result;
        }

        // Apply preconditioner: s_hat = M^{-1} * s
        if (has_precond_) {
            precond_(s, s_hat);
        } else {
            s_hat = s;
        }

        // t = A * s_hat
        matvec_(s_hat, t);

        // omega = (t, s) / (t, t)
        Complex tt = dot(t, t);
        if (std::abs(tt) < 1e-30) {
            result.info = "BiCGStab breakdown: (t,t)=0 at iter " + std::to_string(iter);
            result.iterations = iter;
            result.residual = norm(s) / bnorm;
            // Still update x
            axpy(alpha, p_hat, x);
            return result;
        }
        omega = dot(t, s) / tt;

        // x = x + alpha * p_hat + omega * s_hat
        for (int i = 0; i < n; ++i)
            x[i] += alpha * p_hat[i] + omega * s_hat[i];

        // r = s - omega * t
        for (int i = 0; i < n; ++i)
            r[i] = s[i] - omega * t[i];

        rnorm = norm(r);
        Real rel_res = rnorm / bnorm;

        if (print_level_ >= 2 || (print_level_ >= 1 && iter % 50 == 0)) {
            char buf[64]; snprintf(buf, sizeof(buf), "%.3e", rel_res);
            MAPLE3DMT_LOG_INFO("  BiCGStab iter=" + std::to_string(iter) +
                              " rel_res=" + std::string(buf));
        }

        if (rel_res < tol_) {
            result.converged = true;
            result.residual = rel_res;
            result.iterations = iter;
            result.info = "Converged";
            return result;
        }

        if (std::abs(omega) < 1e-30) {
            result.info = "BiCGStab breakdown: omega=0 at iter " + std::to_string(iter);
            result.iterations = iter;
            result.residual = rel_res;
            return result;
        }

        rho_old = rho;
    }

    result.iterations = maxiter_;
    result.residual = norm(r) / bnorm;
    result.info = "Max iterations reached";
    return result;
}

} // namespace forward
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
