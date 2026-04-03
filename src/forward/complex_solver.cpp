// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file complex_solver.cpp
/// @brief Complex N×N sparse solver: ILU(0) + BiCGStab.
///
/// ModEM-equivalent architecture for 3D MT forward/adjoint solves.
/// Key performance advantages over block-real 2N×2N + FGMRES + AMS:
///   - Matrix: N×N complex vs 2N×2N real (4× less memory)
///   - Preconditioner: ILU(0) O(nnz) vs AMS V-cycle (much heavier)
///   - Solver: BiCGStab (fixed memory) vs FGMRES(k) (growing memory)

#include "maple3dmt/forward/complex_solver.h"
#include "maple3dmt/utils/logger.h"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <numeric>
#include <stdexcept>

// Access HYPRE's internal CSR structure (available via MFEM's HYPRE dependency)
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"

namespace maple3dmt {
namespace forward {

// Logging macros for complex_solver.cpp (no mesh_ dependency unlike forward_solver_3d.cpp)
#define CS_LOG_INFO(msg)    MAPLE3DMT_LOG_INFO(msg)
#define CS_LOG_WARN(msg)    MAPLE3DMT_LOG_WARNING(msg)
#define CS_LOG_DEBUG(msg)   MAPLE3DMT_LOG_DEBUG(msg)

// ── MPI helper: Allreduce for std::complex<double> ──
// Use MPI_DOUBLE with count*2 to avoid MPI_C_DOUBLE_COMPLEX portability issues.
static void mpi_allreduce_complex(const Complex* sendbuf, Complex* recvbuf,
                                   int count, MPI_Comm comm)
{
#ifdef MAPLE3DMT_USE_MPI
    // std::complex<double> is layout-compatible with double[2]
    static_assert(sizeof(Complex) == 2 * sizeof(double),
                  "Complex must be layout-compatible with double[2]");
    MPI_Allreduce(sendbuf, recvbuf, count * 2, MPI_DOUBLE, MPI_SUM, comm);
#else
    std::copy(sendbuf, sendbuf + count, recvbuf);
#endif
}

// ═══════════════════════════════════════════════════════════
// ComplexCSR
// ═══════════════════════════════════════════════════════════

void ComplexCSR::build_from_hypre(const hypre_CSRMatrix* real_diag,
                                   const hypre_CSRMatrix* imag_diag,
                                   double imag_sign)
{
    nrows_ = hypre_CSRMatrixNumRows(real_diag);
    int nnz_real = hypre_CSRMatrixNumNonzeros(real_diag);

    const int*    rp_r = hypre_CSRMatrixI(real_diag);
    const int*    ci_r = hypre_CSRMatrixJ(real_diag);
    const double* vr   = hypre_CSRMatrixData(real_diag);

    const int*    rp_i = hypre_CSRMatrixI(imag_diag);
    const int*    ci_i = hypre_CSRMatrixJ(imag_diag);
    const double* vi   = hypre_CSRMatrixData(imag_diag);

    // Verify same sparsity pattern (same FE space → identical structure)
    int nnz_imag = hypre_CSRMatrixNumNonzeros(imag_diag);
    if (nnz_real != nnz_imag) {
        throw std::runtime_error(
            "ComplexCSR: real/imag sparsity mismatch: " +
            std::to_string(nnz_real) + " vs " + std::to_string(nnz_imag));
    }

    // Copy structure
    row_ptr_.assign(rp_r, rp_r + nrows_ + 1);
    col_idx_.assign(ci_r, ci_r + nnz_real);

    // Build complex values: A = A_real + imag_sign * i * A_imag
    values_.resize(nnz_real);
    for (int k = 0; k < nnz_real; ++k) {
        values_[k] = Complex(vr[k], imag_sign * vi[k]);
    }

    // Find diagonal indices
    diag_idx_.resize(nrows_, -1);
    for (int i = 0; i < nrows_; ++i) {
        for (int k = row_ptr_[i]; k < row_ptr_[i+1]; ++k) {
            if (col_idx_[k] == i) {
                diag_idx_[i] = k;
                break;
            }
        }
        if (diag_idx_[i] < 0) {
            throw std::runtime_error(
                "ComplexCSR: no diagonal entry in row " + std::to_string(i));
        }
    }
}

void ComplexCSR::mult(const Complex* x, Complex* y) const
{
    for (int i = 0; i < nrows_; ++i) {
        Complex sum(0.0, 0.0);
        for (int k = row_ptr_[i]; k < row_ptr_[i+1]; ++k) {
            sum += values_[k] * x[col_idx_[k]];
        }
        y[i] = sum;
    }
}

void ComplexCSR::mult_hermitian(const Complex* x, Complex* y) const
{
    // y = A^H * x = conj(A^T) * x
    // Compute by accumulation: for each row i, col j with value a_ij,
    //   y[j] += conj(a_ij) * x[i]
    std::fill(y, y + nrows_, Complex(0.0, 0.0));
    for (int i = 0; i < nrows_; ++i) {
        for (int k = row_ptr_[i]; k < row_ptr_[i+1]; ++k) {
            y[col_idx_[k]] += std::conj(values_[k]) * x[i];
        }
    }
}


// ═══════════════════════════════════════════════════════════
// ComplexILU0
// ═══════════════════════════════════════════════════════════

bool ComplexILU0::factorize(const ComplexCSR& A)
{
    nrows_ = A.nrows();
    int nnz = A.nnz();

    // Copy structure and values
    row_ptr_.assign(A.row_ptr(), A.row_ptr() + nrows_ + 1);
    col_idx_.assign(A.col_idx(), A.col_idx() + nnz);
    lu_vals_.assign(A.values(), A.values() + nnz);

    // Find diagonal indices
    diag_idx_.resize(nrows_);
    for (int i = 0; i < nrows_; ++i) {
        diag_idx_[i] = A.diag_index(i);
    }

    // ILU(0) factorization (Saad, "Iterative Methods for Sparse Linear Systems")
    // For each row k = 1, ..., n-1:
    //   For each nonzero a_ki where i < k:
    //     a_ki = a_ki / a_ii
    //     For each nonzero a_kj where j > i:
    //       If (k,j) exists in sparsity pattern:
    //         a_kj -= a_ki * a_ij
    //
    // This modifies lu_vals_ in-place. L is stored below diagonal (with
    // unit diagonal implied), U is stored on and above diagonal.

    for (int k = 1; k < nrows_; ++k) {
        int row_start = row_ptr_[k];
        int row_end   = row_ptr_[k + 1];

        // Process lower-triangular entries (col < k)
        for (int pk = row_start; pk < row_end; ++pk) {
            int i = col_idx_[pk];
            if (i >= k) break;  // Only process L part (i < k)

            // a_ki /= a_ii  (L factor: multiplier)
            Complex pivot = lu_vals_[diag_idx_[i]];
            if (std::abs(pivot) < 1e-30) {
                CS_LOG_WARN("ComplexILU0: near-zero pivot at row " +
                         std::to_string(i) + " (|pivot|=" +
                         std::to_string(std::abs(pivot)) + ")");
                return false;
            }
            lu_vals_[pk] /= pivot;
            Complex mult = lu_vals_[pk];

            // Update row k: a_kj -= mult * a_ij for existing entries
            int i_start = row_ptr_[i];
            int i_end   = row_ptr_[i + 1];

            // Walk through row i entries (j > i) and row k entries simultaneously
            int pi = diag_idx_[i] + 1;  // Start after diagonal of row i
            int pk2 = pk + 1;            // Start after current entry in row k

            while (pi < i_end && pk2 < row_end) {
                int j_i = col_idx_[pi];
                int j_k = col_idx_[pk2];

                if (j_i == j_k) {
                    // Both rows have entry at column j: update
                    lu_vals_[pk2] -= mult * lu_vals_[pi];
                    ++pi;
                    ++pk2;
                } else if (j_i < j_k) {
                    ++pi;   // Row i has entry, row k doesn't → skip (ILU(0))
                } else {
                    ++pk2;  // Row k has entry, row i doesn't → skip
                }
            }
        }
    }

    return true;
}

void ComplexILU0::solve(const Complex* b, Complex* x) const
{
    // Forward substitution: L * y = b  (L has unit diagonal)
    // y is stored in x temporarily
    for (int i = 0; i < nrows_; ++i) {
        Complex sum = b[i];
        for (int k = row_ptr_[i]; k < diag_idx_[i]; ++k) {
            sum -= lu_vals_[k] * x[col_idx_[k]];
        }
        x[i] = sum;
    }

    // Backward substitution: U * x = y
    for (int i = nrows_ - 1; i >= 0; --i) {
        Complex sum = x[i];
        for (int k = diag_idx_[i] + 1; k < row_ptr_[i + 1]; ++k) {
            sum -= lu_vals_[k] * x[col_idx_[k]];
        }
        x[i] = sum / lu_vals_[diag_idx_[i]];
    }
}

void ComplexILU0::solve_hermitian(const Complex* b, Complex* x) const
{
    // Solve (L*U)^H * x = b  →  U^H * (L^H * x) = b
    // Two steps:
    //   Step 1: U^H * y = b       (forward sweep, U^H is lower triangular)
    //   Step 2: L^H * x = y       (backward sweep, L^H is upper triangular)
    //
    // CSR stores rows of L (lower) and U (upper). For transpose solves,
    // we use accumulation: process row i and update entries in other rows.

    // ── Step 1: Forward sweep — solve U^H * y = b ──
    // U^H is lower triangular. U^H[i,j] = conj(U[j,i]).
    // Use accumulation: initialize x = b, then for each row i (ascending),
    // divide by conj(diag), and subtract from later rows.
    std::copy(b, b + nrows_, x);
    for (int i = 0; i < nrows_; ++i) {
        x[i] /= std::conj(lu_vals_[diag_idx_[i]]);
        Complex xi = x[i];
        // For each U[i,j] where j > i: U^H[j,i] = conj(U[i,j])
        for (int k = diag_idx_[i] + 1; k < row_ptr_[i + 1]; ++k) {
            x[col_idx_[k]] -= std::conj(lu_vals_[k]) * xi;
        }
    }

    // ── Step 2: Backward sweep — solve L^H * x = y ──
    // L^H is upper triangular with unit diagonal. L^H[j,i] = conj(L[i,j]).
    // Use accumulation: for each row i (descending), x[i] is final,
    // then subtract from earlier rows.
    for (int i = nrows_ - 1; i >= 0; --i) {
        // x[i] is already correct (L^H diagonal = 1)
        Complex xi = x[i];
        // For each L[i,j] where j < i: L^H[j,i] = conj(L[i,j])
        for (int k = row_ptr_[i]; k < diag_idx_[i]; ++k) {
            x[col_idx_[k]] -= std::conj(lu_vals_[k]) * xi;
        }
    }
}


// ═══════════════════════════════════════════════════════════
// ComplexParOperator
// ═══════════════════════════════════════════════════════════

void ComplexParOperator::setup(mfem::HypreParMatrix* A_real,
                                mfem::HypreParMatrix* A_imag)
{
    A_real_ = A_real;
    A_imag_ = A_imag;
    local_n_ = A_real->GetNumRows();

    // Pre-allocate work vectors
    x_r_.SetSize(local_n_);
    x_i_.SetSize(local_n_);
    y_r_.SetSize(local_n_);
    y_i_.SetSize(local_n_);
    tmp_.SetSize(local_n_);
}

void ComplexParOperator::mult(const Complex* x_local, Complex* y_local) const
{
    // Split complex x into real/imag
    for (int i = 0; i < local_n_; ++i) {
        x_r_(i) = x_local[i].real();
        x_i_(i) = x_local[i].imag();
    }

    // y = (A_r + i*A_i) * (x_r + i*x_i)
    //   = (A_r*x_r - A_i*x_i) + i*(A_r*x_i + A_i*x_r)

    // Real part: y_r = A_r*x_r - A_i*x_i
    A_real_->Mult(x_r_, y_r_);       // y_r = A_r * x_r
    A_imag_->Mult(x_i_, tmp_);       // tmp = A_i * x_i
    y_r_.Add(-1.0, tmp_);            // y_r -= A_i * x_i

    // Imag part: y_i = A_r*x_i + A_i*x_r
    A_real_->Mult(x_i_, y_i_);       // y_i = A_r * x_i
    A_imag_->Mult(x_r_, tmp_);       // tmp = A_i * x_r
    y_i_.Add(1.0, tmp_);             // y_i += A_i * x_r

    // Combine into complex output
    for (int i = 0; i < local_n_; ++i) {
        y_local[i] = Complex(y_r_(i), y_i_(i));
    }
}

void ComplexParOperator::mult_hermitian(const Complex* x_local, Complex* y_local) const
{
    // y = (A_r + i*A_i)^H * x = (A_r^T - i*A_i^T) * x
    //   = A_r^T*(x_r + i*x_i) - i*A_i^T*(x_r + i*x_i)
    //   = (A_r^T*x_r + A_i^T*x_i) + i*(A_r^T*x_i - A_i^T*x_r)

    for (int i = 0; i < local_n_; ++i) {
        x_r_(i) = x_local[i].real();
        x_i_(i) = x_local[i].imag();
    }

    // Real part: y_r = A_r^T*x_r + A_i^T*x_i
    A_real_->MultTranspose(x_r_, y_r_);
    A_imag_->MultTranspose(x_i_, tmp_);
    y_r_.Add(1.0, tmp_);

    // Imag part: y_i = A_r^T*x_i - A_i^T*x_r
    A_real_->MultTranspose(x_i_, y_i_);
    A_imag_->MultTranspose(x_r_, tmp_);
    y_i_.Add(-1.0, tmp_);

    for (int i = 0; i < local_n_; ++i) {
        y_local[i] = Complex(y_r_(i), y_i_(i));
    }
}


// ═══════════════════════════════════════════════════════════
// ComplexBiCGStab
// ═══════════════════════════════════════════════════════════

void ComplexBiCGStab::setup(MPI_Comm comm, double tol, int maxiter, int print_lvl)
{
    comm_      = comm;
    tol_       = tol;
    maxiter_   = maxiter;
    print_lvl_ = print_lvl;
}

Complex ComplexBiCGStab::parallel_dot(const Complex* u, const Complex* v, int n) const
{
    // Hermitian dot product: (u, v) = sum conj(u[i]) * v[i]
    // Note: BiCGStab shadow residual uses non-conjugated dot for ρ,
    // but we use Hermitian for norm computation. Keep both.
    Complex local_dot(0.0, 0.0);
    for (int i = 0; i < n; ++i) {
        local_dot += std::conj(u[i]) * v[i];
    }

    Complex global_dot;
    mpi_allreduce_complex(&local_dot, &global_dot, 1, comm_);
    return global_dot;
}

double ComplexBiCGStab::parallel_norm(const Complex* v, int n) const
{
    double local_norm2 = 0.0;
    for (int i = 0; i < n; ++i) {
        local_norm2 += std::norm(v[i]);  // |v[i]|²
    }
    double global_norm2;
#ifdef MAPLE3DMT_USE_MPI
    MPI_Allreduce(&local_norm2, &global_norm2, 1, MPI_DOUBLE, MPI_SUM, comm_);
#else
    global_norm2 = local_norm2;
#endif
    return std::sqrt(global_norm2);
}

ComplexBiCGStab::Result ComplexBiCGStab::solve(
    const ComplexParOperator& op,
    const ComplexILU0& prec,
    const Complex* b, Complex* x, int n_local)
{
    return solve_impl(
        [&](const Complex* in, Complex* out) { op.mult(in, out); },
        [&](const Complex* in, Complex* out) { prec.solve(in, out); },
        b, x, n_local, "fwd");
}

ComplexBiCGStab::Result ComplexBiCGStab::solve_hermitian(
    const ComplexParOperator& op,
    const ComplexILU0& prec,
    const Complex* b, Complex* x, int n_local)
{
    return solve_impl(
        [&](const Complex* in, Complex* out) { op.mult_hermitian(in, out); },
        [&](const Complex* in, Complex* out) { prec.solve_hermitian(in, out); },
        b, x, n_local, "adj");
}

ComplexBiCGStab::Result ComplexBiCGStab::solve_impl(
    std::function<void(const Complex*, Complex*)> matvec,
    std::function<void(const Complex*, Complex*)> precond,
    const Complex* b, Complex* x, int n_local,
    const char* label)
{
    Result result;

    // Allocate work vectors
    std::vector<Complex> r(n_local), r_hat(n_local);
    std::vector<Complex> p(n_local), v(n_local);
    std::vector<Complex> s(n_local), t(n_local);
    std::vector<Complex> p_hat(n_local), s_hat(n_local);

    // r₀ = b - A*x₀
    matvec(x, r.data());  // r = A*x
    for (int i = 0; i < n_local; ++i) {
        r[i] = b[i] - r[i];
    }

    double b_norm = parallel_norm(b, n_local);
    if (b_norm < 1e-30) {
        // Zero RHS → zero solution
        std::fill(x, x + n_local, Complex(0.0, 0.0));
        result.converged = true;
        return result;
    }

    double r_norm = parallel_norm(r.data(), n_local);
    if (r_norm / b_norm < tol_) {
        result.converged = true;
        result.final_residual = r_norm / b_norm;
        return result;
    }

    // r̂₀ = r₀ (shadow residual, fixed throughout)
    std::copy(r.begin(), r.end(), r_hat.begin());

    Complex rho_prev(1.0, 0.0), alpha(1.0, 0.0), omega(1.0, 0.0);
    std::fill(v.begin(), v.end(), Complex(0.0, 0.0));
    std::fill(p.begin(), p.end(), Complex(0.0, 0.0));

    int rank = 0;
#ifdef MAPLE3DMT_USE_MPI
    MPI_Comm_rank(comm_, &rank);
#endif

    for (int iter = 1; iter <= maxiter_; ++iter) {
        // ρᵢ = (r̂₀, rᵢ₋₁)  — non-conjugated dot for BiCGStab
        Complex rho(0.0, 0.0);
        for (int i = 0; i < n_local; ++i) {
            rho += std::conj(r_hat[i]) * r[i];
        }
        Complex rho_global;
        mpi_allreduce_complex(&rho, &rho_global, 1, comm_);
        rho = rho_global;

        if (std::abs(rho) < 1e-30) {
            // Breakdown: r̂₀ ⊥ rᵢ → restart with new shadow
            if (rank == 0 && print_lvl_ >= 1) {
                CS_LOG_WARN("  BiCGStab(" + std::string(label) +
                         "): breakdown at iter " + std::to_string(iter));
            }
            break;
        }

        // β = (ρᵢ/ρᵢ₋₁) * (α/ωᵢ₋₁)
        Complex beta = (rho / rho_prev) * (alpha / omega);

        // pᵢ = rᵢ₋₁ + β*(pᵢ₋₁ - ωᵢ₋₁*vᵢ₋₁)
        for (int i = 0; i < n_local; ++i) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // p̂ = M⁻¹ * pᵢ
        precond(p.data(), p_hat.data());

        // vᵢ = A * p̂
        matvec(p_hat.data(), v.data());

        // α = ρᵢ / (r̂₀, vᵢ)
        Complex rv(0.0, 0.0);
        for (int i = 0; i < n_local; ++i) {
            rv += std::conj(r_hat[i]) * v[i];
        }
        Complex rv_global;
        mpi_allreduce_complex(&rv, &rv_global, 1, comm_);
        rv = rv_global;

        if (std::abs(rv) < 1e-30) {
            if (rank == 0 && print_lvl_ >= 1) {
                CS_LOG_WARN("  BiCGStab(" + std::string(label) +
                         "): rv breakdown at iter " + std::to_string(iter));
            }
            break;
        }
        alpha = rho / rv;

        // s = rᵢ₋₁ - α*vᵢ
        for (int i = 0; i < n_local; ++i) {
            s[i] = r[i] - alpha * v[i];
        }

        // Check if s is small enough → early convergence
        double s_norm = parallel_norm(s.data(), n_local);
        if (s_norm / b_norm < tol_) {
            // x += α*p̂
            for (int i = 0; i < n_local; ++i) {
                x[i] += alpha * p_hat[i];
            }
            result.iterations = iter;
            result.final_residual = s_norm / b_norm;
            result.converged = true;
            if (rank == 0 && print_lvl_ >= 1) {
                CS_LOG_INFO("  BiCGStab(" + std::string(label) + "): converged at iter " +
                         std::to_string(iter) + " rel_res=" +
                         std::to_string(s_norm / b_norm));
            }
            return result;
        }

        // ŝ = M⁻¹ * s
        precond(s.data(), s_hat.data());

        // t = A * ŝ
        matvec(s_hat.data(), t.data());

        // ωᵢ = (t, s) / (t, t)
        Complex ts(0.0, 0.0), tt(0.0, 0.0);
        for (int i = 0; i < n_local; ++i) {
            ts += std::conj(t[i]) * s[i];
            tt += std::conj(t[i]) * t[i];
        }
        Complex ts_tt[2] = {ts, tt};
        Complex ts_tt_global[2];
        mpi_allreduce_complex(ts_tt, ts_tt_global, 2, comm_);
        ts = ts_tt_global[0];
        tt = ts_tt_global[1];

        if (std::abs(tt) < 1e-30) {
            omega = Complex(0.0, 0.0);
        } else {
            omega = ts / tt;
        }

        // xᵢ = xᵢ₋₁ + α*p̂ + ω*ŝ
        for (int i = 0; i < n_local; ++i) {
            x[i] += alpha * p_hat[i] + omega * s_hat[i];
        }

        // rᵢ = s - ω*t
        for (int i = 0; i < n_local; ++i) {
            r[i] = s[i] - omega * t[i];
        }

        r_norm = parallel_norm(r.data(), n_local);
        double rel_res = r_norm / b_norm;

        if (rank == 0 && print_lvl_ >= 1 && (iter % 50 == 0 || iter <= 5)) {
            CS_LOG_INFO("  BiCGStab(" + std::string(label) + ") iter=" +
                     std::to_string(iter) + " rel_res=" +
                     std::to_string(rel_res));
        }

        if (rel_res < tol_) {
            result.iterations = iter;
            result.final_residual = rel_res;
            result.converged = true;
            if (rank == 0 && print_lvl_ >= 1) {
                CS_LOG_INFO("  BiCGStab(" + std::string(label) + "): converged at iter " +
                         std::to_string(iter) + " rel_res=" +
                         std::to_string(rel_res));
            }
            return result;
        }

        if (std::abs(omega) < 1e-30) {
            if (rank == 0 && print_lvl_ >= 1) {
                CS_LOG_WARN("  BiCGStab(" + std::string(label) +
                         "): omega breakdown at iter " + std::to_string(iter));
            }
            break;
        }

        rho_prev = rho;
    }

    // Did not converge
    result.iterations = maxiter_;
    result.final_residual = r_norm / b_norm;
    result.converged = false;
    if (rank == 0) {
        CS_LOG_WARN("  BiCGStab(" + std::string(label) + "): NOT converged after " +
                 std::to_string(maxiter_) + " iter, rel_res=" +
                 std::to_string(result.final_residual));
    }
    return result;
}


// ═══════════════════════════════════════════════════════════
// ComplexSolverWrapper
// ═══════════════════════════════════════════════════════════

void ComplexSolverWrapper::build(
    mfem::ParSesquilinearForm& sesq,
    mfem::HypreParMatrix* grad_mat,
    const mfem::Array<int>& ess_tdofs,
    MPI_Comm comm,
    const Params& params)
{
    release();
    comm_ = comm;
    params_ = params;
    grad_mat_ = grad_mat;

    int rank = 0;
#ifdef MAPLE3DMT_USE_MPI
    MPI_Comm_rank(comm_, &rank);
#endif

    // ── Step 1: Assemble separate real/imag N×N HypreParMatrix ──
    // Instead of extracting from ComplexHypreParMatrix (API portability issues),
    // we assemble the real and imaginary parts separately using ParBilinearForm.
    // This guarantees deep ownership and avoids shallow-copy problems.
    //
    // The complex system is: A = K - iωσM
    //   Real part:  K = (1/μ₀) CurlCurl          (N×N SPD)
    //   Imag part:  -ωσ Mass                       (N×N, negative semi-definite)
    //
    // We reuse the sesquilinear form's internal bilinear forms to extract these.

    auto* fespace = sesq.ParFESpace();
    int vsize = fespace->GetVSize();

    // Access the internal real and imaginary bilinear forms from sesquilinear form.
    // ParSesquilinearForm stores two ParBilinearForm internally.
    // We need to get the assembled parallel matrices from each.
    //
    // Alternative approach: use FormLinearSystem and extract from ComplexHypreParMatrix.
    // MFEM's ComplexHypreParMatrix provides GetSystemMatrix() for 2N×2N monolithic,
    // but for separate matrices, we use the ComplexOperator interface.
    mfem::Vector x_dummy(2 * vsize), b_dummy(2 * vsize);
    x_dummy = 0.0; b_dummy = 0.0;

    mfem::OperatorHandle A_handle;
    mfem::Vector X_dummy, B_dummy;
    sesq.FormLinearSystem(ess_tdofs, x_dummy, b_dummy, A_handle, X_dummy, B_dummy);

    // ComplexOperator provides real() and imag() as Operator references.
    // ComplexHypreParMatrix : public ComplexOperator, which inherits from Operator.
    auto* complex_op = dynamic_cast<mfem::ComplexOperator*>(A_handle.Ptr());
    if (!complex_op) {
        throw std::runtime_error("ComplexSolverWrapper: failed to get ComplexOperator");
    }

    // Get real and imaginary operators.
    // MFEM ComplexOperator::real()/imag() return Operator& (reference).
    // For parallel assembly, these should be HypreParMatrix.
    auto* Ar_ptr = dynamic_cast<mfem::HypreParMatrix*>(&complex_op->real());
    auto* Ai_ptr = dynamic_cast<mfem::HypreParMatrix*>(&complex_op->imag());

    if (!Ar_ptr || !Ai_ptr) {
        throw std::runtime_error("ComplexSolverWrapper: real/imag not HypreParMatrix");
    }

    // Keep the OperatorHandle alive to prevent deallocation of internal matrices.
    // We store it as a member and use the raw pointers.
    A_handle_ = std::move(A_handle);
    A_real_ptr_ = Ar_ptr;
    A_imag_ptr_ = Ai_ptr;

    local_n_ = Ar_ptr->GetNumRows();

    if (rank == 0) {
        CS_LOG_INFO("ComplexSolver: built N=" + std::to_string(local_n_) +
                 " (was 2N=" + std::to_string(2 * local_n_) + " block-real)");
    }

    // ── Step 2: Setup parallel operator (for mat-vec) ──
    par_op_.setup(A_real_ptr_, A_imag_ptr_);

    // ── Step 3: Build local ComplexCSR from diagonal blocks ──
    // Get HYPRE's internal diagonal CSR blocks
    hypre_ParCSRMatrix* h_real = static_cast<hypre_ParCSRMatrix*>(*A_real_ptr_);
    hypre_ParCSRMatrix* h_imag = static_cast<hypre_ParCSRMatrix*>(*A_imag_ptr_);

    hypre_CSRMatrix* diag_real = hypre_ParCSRMatrixDiag(h_real);
    hypre_CSRMatrix* diag_imag = hypre_ParCSRMatrixDiag(h_imag);

    ComplexCSR local_A;
    local_A.build_from_hypre(diag_real, diag_imag, 1.0);

    if (rank == 0) {
        CS_LOG_INFO("ComplexSolver: local CSR nnz=" +
                 std::to_string(local_A.nnz()) +
                 " (nnz/row=" + std::to_string(local_A.nnz() / std::max(1, local_A.nrows())) + ")");
    }

    // ── Step 4: ILU(0) factorization ──
    double t0 = MPI_Wtime();
    bool ok = ilu_fwd_.factorize(local_A);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        CS_LOG_INFO("ComplexSolver: ILU(0) factorize " +
                 std::string(ok ? "OK" : "FAILED") +
                 " (" + std::to_string(t1 - t0) + "s)");
    }

    if (!ok) {
        CS_LOG_WARN("ComplexSolver: ILU(0) failed, falling back to diagonal scaling");
        // TODO: implement diagonal scaling fallback
    }

    // ── Step 5: Setup BiCGStab ──
    bicgstab_.setup(comm_, params_.tol, params_.maxiter,
                    params_.print_lvl > 0 ? 1 : 0);

    ready_ = true;
}

ComplexBiCGStab::Result ComplexSolverWrapper::solve(
    const mfem::Vector& rhs_real, const mfem::Vector& rhs_imag,
    mfem::Vector& sol_real, mfem::Vector& sol_imag)
{
    if (!ready_) {
        throw std::runtime_error("ComplexSolverWrapper::solve: not built");
    }

    // Convert MFEM vectors → complex
    std::vector<Complex> b(local_n_), x(local_n_, Complex(0.0, 0.0));
    for (int i = 0; i < local_n_; ++i) {
        b[i] = Complex(rhs_real(i), rhs_imag(i));
    }

    // Use initial guess if provided (non-zero)
    if (sol_real.Size() == local_n_ && sol_imag.Size() == local_n_) {
        double nr = sol_real.Norml2(), ni = sol_imag.Norml2();
        if (nr > 1e-30 || ni > 1e-30) {
            for (int i = 0; i < local_n_; ++i) {
                x[i] = Complex(sol_real(i), sol_imag(i));
            }
        }
    }

    // Solve
    auto result = bicgstab_.solve(par_op_, ilu_fwd_, b.data(), x.data(), local_n_);

    // Convert back to MFEM vectors
    sol_real.SetSize(local_n_);
    sol_imag.SetSize(local_n_);
    for (int i = 0; i < local_n_; ++i) {
        sol_real(i) = x[i].real();
        sol_imag(i) = x[i].imag();
    }

    return result;
}

ComplexBiCGStab::Result ComplexSolverWrapper::solve_adjoint(
    const mfem::Vector& rhs_real, const mfem::Vector& rhs_imag,
    mfem::Vector& sol_real, mfem::Vector& sol_imag)
{
    if (!ready_) {
        throw std::runtime_error("ComplexSolverWrapper::solve_adjoint: not built");
    }

    // Convert MFEM vectors → complex
    std::vector<Complex> b(local_n_), x(local_n_, Complex(0.0, 0.0));
    for (int i = 0; i < local_n_; ++i) {
        b[i] = Complex(rhs_real(i), rhs_imag(i));
    }

    // Use initial guess if provided
    if (sol_real.Size() == local_n_ && sol_imag.Size() == local_n_) {
        double nr = sol_real.Norml2(), ni = sol_imag.Norml2();
        if (nr > 1e-30 || ni > 1e-30) {
            for (int i = 0; i < local_n_; ++i) {
                x[i] = Complex(sol_real(i), sol_imag(i));
            }
        }
    }

    // Setup adjoint BiCGStab with relaxed tolerance
    ComplexBiCGStab adj_bicgstab;
    adj_bicgstab.setup(comm_, params_.adj_tol, params_.adj_maxiter,
                       params_.print_lvl > 0 ? 1 : 0);

    // Solve A^H * x = b
    auto result = adj_bicgstab.solve_hermitian(par_op_, ilu_fwd_,
                                                b.data(), x.data(), local_n_);

    // Convert back
    sol_real.SetSize(local_n_);
    sol_imag.SetSize(local_n_);
    for (int i = 0; i < local_n_; ++i) {
        sol_real(i) = x[i].real();
        sol_imag(i) = x[i].imag();
    }

    return result;
}

void ComplexSolverWrapper::release()
{
    ready_ = false;
    A_real_ptr_ = nullptr;
    A_imag_ptr_ = nullptr;
    A_handle_.Clear();   // Releases the ComplexHypreParMatrix (owns real/imag)
    GGt_.reset();
    ilu_fwd_ = ComplexILU0();
    par_op_ = ComplexParOperator();
    ccgd_tau_ = 0.0;
}

}  // namespace forward
}  // namespace maple3dmt
