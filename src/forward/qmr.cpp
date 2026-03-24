// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#include "maple3dmt/forward/qmr.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/utils/logger.h"
#include <cstdio>

namespace maple3dmt {
namespace forward {

/// QMR for complex symmetric systems (A = A^T).
///
/// Algorithm: Symmetric Lanczos + QR factorization of tridiagonal.
///
/// Lanczos for A = A^T:
///   beta_{n+1} * v_{n+1} = A * v_n - alpha_n * v_n - beta_n * v_{n-1}
///   alpha_n = v_n^T * A * v_n  (unconjugated, bilinear form)
///   beta_{n+1} = ||w||  where w = A*v_n - alpha*v_n - beta*v_{n-1}
///   (sign of beta chosen so v_{n+1} = w / beta_{n+1})
///
/// Key: only ONE matvec per iteration (A^T = A → no separate transpose multiply).
///
/// The QMR part applies Givens rotations to the tridiagonal to solve
/// the quasi-minimum residual problem in the Krylov subspace.
///
/// Reference: Freund (1992), "Conjugate gradient-type methods for linear
///   systems with complex symmetric coefficient matrices."
///   Also: Barrett et al., "Templates for the Solution of Linear Systems."
QMRResult QMRSolver::solve(const ComplexVec& b, ComplexVec& x) const {
    QMRResult result;
    const int n = static_cast<int>(b.size());

    if (!matvec_) {
        result.info = "No operator set";
        return result;
    }

    // Initial residual: r = b - A*x
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

    // === Symmetric Lanczos + QMR ===
    //
    // We maintain:
    //   v_prev, v_curr : Lanczos vectors
    //   d_prev, d_curr : direction vectors for solution update
    //   Givens rotation state for the tridiagonal QR factorization

    ComplexVec v_prev(n, Complex(0, 0));
    ComplexVec v_curr(n);
    ComplexVec v_next(n);
    ComplexVec Av(n);

    // Preconditioned initial: v_curr = M^{-1} r / ||M^{-1} r||
    // (left preconditioning: solve M z = r, then use z)
    ComplexVec z(n);
    if (has_precond_) {
        precond_(r, z);
    } else {
        z = r;
    }

    Real beta = norm(z);
    if (beta < 1e-30) {
        result.info = "Zero initial residual after preconditioning";
        result.residual = rnorm / bnorm;
        return result;
    }

    // v_1 = z / beta
    for (int i = 0; i < n; ++i) v_curr[i] = z[i] / beta;

    // Direction vectors for the QMR update
    ComplexVec d_prev(n, Complex(0, 0));
    ComplexVec d_curr(n, Complex(0, 0));

    // QR factorization state (2x2 Givens rotation)
    // The tridiagonal T has: T[k,k] = alpha_k, T[k,k-1] = T[k-1,k] = beta_k
    Complex c_prev(1, 0), s_prev(0, 0);  // previous Givens rotation
    Complex c_curr(1, 0), s_curr(0, 0);  // current Givens rotation

    // Residual norm tracking via QR factorization
    Real rho_prev = beta;  // ||r_0|| in Lanczos basis
    Complex eta = Complex(beta, 0);  // running residual factor

    Real beta_prev = 0;

    for (int iter = 1; iter <= maxiter_; ++iter) {
        // --- Lanczos step ---
        // w = A * v_curr
        matvec_(v_curr, Av);

        // If preconditioned: z = M^{-1} * A * v
        if (has_precond_) {
            precond_(Av, z);
        } else {
            z = Av;
        }

        // alpha = v_curr^T * z  (unconjugated for complex symmetric)
        Complex alpha = sym_dot(v_curr, z);

        // w = z - alpha * v_curr - beta * v_prev
        for (int i = 0; i < n; ++i)
            v_next[i] = z[i] - alpha * v_curr[i] - Complex(beta, 0) * v_prev[i];

        Real beta_next = norm(v_next);

        // Breakdown check
        if (beta_next < 1e-30) {
            // Lucky breakdown — exact solution in Krylov subspace
            // Still need to do the final QMR update
            // Apply current column [beta; alpha; 0] of tridiagonal
            Complex h1 = Complex(beta, 0);  // super-diagonal from previous
            Complex h2 = alpha;              // diagonal
            Complex h3(0, 0);                // sub-diagonal (zero for last)

            // Apply previous Givens rotation
            Complex temp = c_prev * h1 + s_prev * h2;
            h2 = -std::conj(s_prev) * h1 + std::conj(c_prev) * h2;
            h1 = temp;

            // New direction
            ComplexVec d_new(n);
            for (int i = 0; i < n; ++i)
                d_new[i] = (v_curr[i] - h1 * d_prev[i]) / h2;

            // Update solution
            for (int i = 0; i < n; ++i)
                x[i] += eta * d_new[i];

            // Recompute true residual
            matvec_(x, Ax);
            for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
            rnorm = norm(r);

            result.converged = (rnorm / bnorm < tol_);
            result.residual = rnorm / bnorm;
            result.iterations = iter;
            result.info = result.converged ? "Converged (breakdown)" : "Lanczos breakdown";
            return result;
        }

        // Normalize v_next
        for (int i = 0; i < n; ++i) v_next[i] /= beta_next;

        // --- QMR: Apply Givens rotations to eliminate sub-diagonal ---
        // Current column of tridiagonal: [beta; alpha; beta_next]
        // (beta is the super-diagonal from previous step)
        Complex h1 = Complex(beta, 0);
        Complex h2 = alpha;
        Complex h3 = Complex(beta_next, 0);

        // Apply previous Givens rotation to [h1; h2]
        {
            Complex temp = c_prev * h1 + s_prev * h2;
            h2 = -std::conj(s_prev) * h1 + std::conj(c_prev) * h2;
            h1 = temp;
        }

        // Compute new Givens rotation to zero out h3
        // [c  s] [h2] = [r]
        // [-s* c*] [h3]   [0]
        Real abs_h2 = std::abs(h2);
        Real abs_h3 = std::abs(h3);
        Real denom = std::sqrt(abs_h2 * abs_h2 + abs_h3 * abs_h3);

        if (denom < 1e-30) {
            result.info = "QMR breakdown: zero pivot at iter " + std::to_string(iter);
            result.iterations = iter;
            // Compute true residual
            matvec_(x, Ax);
            for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
            result.residual = norm(r) / bnorm;
            return result;
        }

        Complex c_new = h2 / Complex(denom, 0);
        Complex s_new = h3 / Complex(denom, 0);
        Complex r_diag = Complex(denom, 0);  // the diagonal after rotation

        // Update direction vector: d = (v - h1*d_prev) / r_diag
        ComplexVec d_new(n);
        for (int i = 0; i < n; ++i)
            d_new[i] = (v_curr[i] - h1 * d_prev[i]) / r_diag;

        // Update solution: x += eta * c_new * d_new
        Complex update_coeff = eta * std::conj(c_new);
        for (int i = 0; i < n; ++i)
            x[i] += update_coeff * d_new[i];

        // Update residual estimate: eta = -s_new * eta (for next iteration)
        eta = -std::conj(s_new) * eta;
        Real res_estimate = std::abs(eta) / bnorm;

        // Print progress
        if (print_level_ >= 2 || (print_level_ >= 1 && iter % 50 == 0)) {
            char buf[64]; snprintf(buf, sizeof(buf), "%.3e", res_estimate);
            MAPLE3DMT_LOG_INFO("  QMR iter=" + std::to_string(iter) +
                              " rel_res=" + std::string(buf));
        }

        // Convergence check (use estimate; verify with true residual periodically)
        if (res_estimate < tol_) {
            // Verify with true residual
            matvec_(x, Ax);
            for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
            rnorm = norm(r);
            Real true_rel = rnorm / bnorm;

            if (true_rel < tol_ * 10) {  // allow small margin
                result.converged = true;
                result.residual = true_rel;
                result.iterations = iter;
                result.info = "Converged";
                return result;
            }
            // If estimate was optimistic, continue
        }

        // Shift for next iteration
        v_prev = v_curr;
        v_curr = v_next;
        d_prev = d_curr;
        d_curr = d_new;
        c_prev = c_new;
        s_prev = s_new;
        beta = beta_next;
    }

    // Did not converge — compute true residual
    matvec_(x, Ax);
    for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
    result.residual = norm(r) / bnorm;
    result.iterations = maxiter_;
    result.info = "Max iterations reached";
    return result;
}

} // namespace forward
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
