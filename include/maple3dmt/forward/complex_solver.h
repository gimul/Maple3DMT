// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file complex_solver.h
/// @brief Complex N×N sparse matrix, ILU(0) preconditioner, and BiCGStab solver.
///
/// Replaces the block-real 2N×2N FGMRES+AMS approach with complex N×N BiCGStab+ILU(0).
/// This matches ModEM's architecture:
///   - Complex arithmetic: N×N instead of 2N×2N (4× less memory, better cache)
///   - ILU(0): O(nnz) per application (vs AMS V-cycle which is much heavier)
///   - BiCGStab: no Gram-Schmidt (vs FGMRES which requires orthogonalization)
///
/// Usage:
///   1. Build ComplexCSR from MFEM's separate real/imag HypreParMatrix diag blocks
///   2. Factorize ILU(0) on local diagonal block (block-Jacobi for MPI)
///   3. BiCGStab with parallel mat-vec (via MFEM) + local ILU(0) preconditioner

#include "maple3dmt/common.h"
#include <mfem.hpp>
#include <complex>
#include <vector>
#include <functional>
#ifdef MAPLE3DMT_USE_MPI
#include <mpi.h>
#endif

namespace maple3dmt {
namespace forward {

using Complex = std::complex<double>;

// ─────────────────────────────────────────────────────────
// ComplexCSR: Local complex sparse matrix in CSR format
// ─────────────────────────────────────────────────────────

/// @brief Local (single-process) complex sparse matrix in CSR format.
///
/// Built from two HYPRE CSR matrices (real and imaginary parts).
/// The sparsity pattern must be identical (same FE space, same integrators).
class ComplexCSR {
public:
    ComplexCSR() = default;

    /// Build from two hypre_CSRMatrix (diagonal blocks of HypreParMatrix).
    /// @param real_diag  Diagonal block of real-part HypreParMatrix
    /// @param imag_diag  Diagonal block of imag-part HypreParMatrix
    /// @param imag_sign  Sign for imaginary part (+1 or -1, for A vs A^H)
    void build_from_hypre(const hypre_CSRMatrix* real_diag,
                          const hypre_CSRMatrix* imag_diag,
                          double imag_sign = 1.0);

    /// y = A * x  (complex mat-vec, local only)
    void mult(const Complex* x, Complex* y) const;

    /// y = A^H * x  (conjugate-transpose mat-vec, local only)
    void mult_hermitian(const Complex* x, Complex* y) const;

    int nrows() const { return nrows_; }
    int nnz()   const { return static_cast<int>(values_.size()); }

    const int*     row_ptr()  const { return row_ptr_.data(); }
    const int*     col_idx()  const { return col_idx_.data(); }
    const Complex* values()   const { return values_.data(); }
    Complex*       values()         { return values_.data(); }

    /// Get pointer to diagonal element for row i (or nullptr if not found)
    int diag_index(int row) const { return diag_idx_[row]; }

private:
    int nrows_ = 0;
    std::vector<int>     row_ptr_;    // size nrows+1
    std::vector<int>     col_idx_;    // size nnz
    std::vector<Complex> values_;     // size nnz
    std::vector<int>     diag_idx_;   // index of diagonal element in each row
};


// ─────────────────────────────────────────────────────────
// ComplexILU0: ILU(0) factorization for complex CSR
// ─────────────────────────────────────────────────────────

/// @brief ILU(0) factorization of a complex sparse matrix.
///
/// Stores L and U factors in-place (same sparsity as original A).
/// L has unit diagonal (not stored). U diagonal is stored.
///
/// For parallel use: apply to diagonal block of each MPI rank (block-Jacobi).
/// This is equivalent to ModEM's DILU (Distributed ILU).
class ComplexILU0 {
public:
    ComplexILU0() = default;

    /// Compute ILU(0) factorization of A.
    /// @param A  Complex CSR matrix (local diagonal block)
    /// @return true if factorization succeeded (no zero pivots)
    bool factorize(const ComplexCSR& A);

    /// Solve (L*U) * x = b  via forward/backward substitution.
    /// @param b  RHS vector (size = A.nrows)
    /// @param x  Solution vector (size = A.nrows)
    void solve(const Complex* b, Complex* x) const;

    /// Solve (L*U)^H * x = b  for adjoint preconditioning.
    void solve_hermitian(const Complex* b, Complex* x) const;

    int nrows() const { return nrows_; }

private:
    int nrows_ = 0;
    std::vector<int>     row_ptr_;    // CSR row pointers (same as A)
    std::vector<int>     col_idx_;    // CSR column indices (same as A)
    std::vector<Complex> lu_vals_;    // L\U combined values
    std::vector<int>     diag_idx_;   // index of diagonal in each row
};


// ─────────────────────────────────────────────────────────
// ComplexParOperator: Parallel complex mat-vec via MFEM
// ─────────────────────────────────────────────────────────

/// @brief Parallel complex matrix-vector product using MFEM's HypreParMatrix.
///
/// Wraps two HypreParMatrix (real, imag parts) and performs:
///   y = (A_r + i*A_i) * x
/// where x, y are complex N-vectors stored as std::complex<double>*.
///
/// Internally converts to/from MFEM's [real;imag] layout and uses
/// HypreParMatrix::Mult for MPI communication.
class ComplexParOperator {
public:
    ComplexParOperator() = default;

    /// @param A_real  Real part of system matrix (N×N, parallel)
    /// @param A_imag  Imaginary part of system matrix (N×N, parallel)
    void setup(mfem::HypreParMatrix* A_real, mfem::HypreParMatrix* A_imag);

    /// y = (A_r + i*A_i) * x   (parallel, handles MPI communication)
    void mult(const Complex* x_local, Complex* y_local) const;

    /// y = (A_r + i*A_i)^H * x  (parallel, conjugate transpose)
    void mult_hermitian(const Complex* x_local, Complex* y_local) const;

    int local_size() const { return local_n_; }

    mfem::HypreParMatrix* real_part() { return A_real_; }
    mfem::HypreParMatrix* imag_part() { return A_imag_; }

private:
    mfem::HypreParMatrix* A_real_ = nullptr;
    mfem::HypreParMatrix* A_imag_ = nullptr;
    int local_n_ = 0;

    // Work vectors for MFEM mat-vec (avoid repeated allocation)
    mutable mfem::Vector x_r_, x_i_, y_r_, y_i_, tmp_;
};


// ─────────────────────────────────────────────────────────
// ComplexBiCGStab: Parallel BiCGStab solver
// ─────────────────────────────────────────────────────────

/// @brief Parallel BiCGStab solver for complex non-Hermitian systems.
///
/// Solves A*x = b where A is complex N×N (unsymmetric).
/// Uses block-Jacobi ILU(0) as preconditioner.
///
/// BiCGStab advantages over FGMRES:
///   - No Gram-Schmidt orthogonalization (O(1) per iter vs O(k) for FGMRES(k))
///   - Fixed memory: 8 vectors vs k+1 for FGMRES(k)
///   - 2 mat-vecs per iteration (vs 1 for FGMRES, but no restart needed)
///
/// Reference: van der Vorst, 1992. "Bi-CGSTAB: A fast and smoothly
///            converging variant of Bi-CG for the solution of
///            nonsymmetric linear systems"
class ComplexBiCGStab {
public:
    struct Result {
        int    iterations = 0;
        double final_residual = 0.0;
        bool   converged = false;
    };

    ComplexBiCGStab() = default;

    /// @param comm      MPI communicator
    /// @param tol       Relative residual tolerance
    /// @param maxiter   Maximum iterations
    /// @param print_lvl Print level: 0=silent, 1=per-iteration, 2=verbose
    void setup(MPI_Comm comm, double tol, int maxiter, int print_lvl = 0);

    /// Solve A*x = b using BiCGStab with ILU(0) preconditioner.
    ///
    /// @param op       Parallel complex operator (for mat-vec)
    /// @param prec     ILU(0) preconditioner (local, block-Jacobi)
    /// @param b        RHS vector (complex, local portion)
    /// @param x        Initial guess / solution (complex, local portion)
    /// @param n_local  Local vector size (N on this rank)
    /// @return         Convergence info
    Result solve(const ComplexParOperator& op,
                 const ComplexILU0& prec,
                 const Complex* b, Complex* x,
                 int n_local);

    /// Solve A^H*x = b for adjoint (uses hermitian mat-vec + ILU^H precond).
    Result solve_hermitian(const ComplexParOperator& op,
                           const ComplexILU0& prec,
                           const Complex* b, Complex* x,
                           int n_local);

private:
    MPI_Comm comm_ = MPI_COMM_WORLD;
    double   tol_  = 1e-3;
    int      maxiter_ = 500;
    int      print_lvl_ = 0;

    /// Parallel complex dot product: (u, v) = sum_i conj(u[i]) * v[i]
    Complex parallel_dot(const Complex* u, const Complex* v, int n) const;

    /// Parallel L2 norm
    double parallel_norm(const Complex* v, int n) const;

    /// Core BiCGStab loop (templated for forward/adjoint)
    Result solve_impl(
        std::function<void(const Complex*, Complex*)> matvec,
        std::function<void(const Complex*, Complex*)> precond,
        const Complex* b, Complex* x, int n_local,
        const char* label);
};


// ─────────────────────────────────────────────────────────
// ComplexSolverWrapper: Integrates into ForwardSolver3D
// ─────────────────────────────────────────────────────────

/// @brief High-level wrapper that manages complex solver lifecycle.
///
/// Replaces the block-real 2N×2N FGMRES+AMS path in ForwardSolver3D.
/// Per-frequency workflow:
///   1. build() — extract real/imag matrices, build ComplexCSR, factorize ILU(0)
///   2. solve() — BiCGStab forward solve
///   3. solve_adjoint() — BiCGStab adjoint solve (A^H)
///   4. release() — free matrices and factorization
class ComplexSolverWrapper {
public:
    struct Params {
        double tol       = 2e-3;    // Forward tolerance
        int    maxiter   = 2000;    // Forward max iterations
        double adj_tol   = 0.1;     // Adjoint tolerance (relaxed)
        int    adj_maxiter = 500;   // Adjoint max iterations
        int    print_lvl = 0;       // 0=silent, 1=per-iter
        bool   ccgd_enabled = true; // CCGD for adjoint
        double ccgd_tau  = 0.0;     // 0=auto
    };

    ComplexSolverWrapper() = default;

    /// Build complex solver for current frequency.
    /// @param sesq     Assembled sesquilinear form (provides ComplexHypreParMatrix)
    /// @param grad_mat Discrete gradient for CCGD (N_edge × N_node)
    /// @param ess_tdofs Essential boundary DOF list
    /// @param comm     MPI communicator
    void build(mfem::ParSesquilinearForm& sesq,
               mfem::HypreParMatrix* grad_mat,
               const mfem::Array<int>& ess_tdofs,
               MPI_Comm comm,
               const Params& params);

    /// Forward solve: A*x = b (complex)
    /// @param rhs_real  Real part of RHS (true DOF vector)
    /// @param rhs_imag  Imaginary part of RHS (true DOF vector)
    /// @param sol_real  Real part of solution (output)
    /// @param sol_imag  Imaginary part of solution (output)
    ComplexBiCGStab::Result solve(
        const mfem::Vector& rhs_real, const mfem::Vector& rhs_imag,
        mfem::Vector& sol_real, mfem::Vector& sol_imag);

    /// Adjoint solve: A^H*x = b
    ComplexBiCGStab::Result solve_adjoint(
        const mfem::Vector& rhs_real, const mfem::Vector& rhs_imag,
        mfem::Vector& sol_real, mfem::Vector& sol_imag);

    /// Release all matrices and factorizations for current frequency.
    void release();

    bool ready() const { return ready_; }

private:
    bool ready_ = false;
    Params params_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
    int local_n_ = 0;

    // Parallel operator (for mat-vec with MPI communication)
    ComplexParOperator par_op_;

    // Local ILU(0) preconditioner (block-Jacobi)
    // Forward uses ilu_fwd_.solve(), adjoint uses ilu_fwd_.solve_hermitian()
    ComplexILU0 ilu_fwd_;

    // BiCGStab solver
    ComplexBiCGStab bicgstab_;

    // OperatorHandle keeps ComplexHypreParMatrix alive (owns real/imag matrices)
    mfem::OperatorHandle A_handle_;
    mfem::HypreParMatrix* A_real_ptr_ = nullptr;  // borrowed from A_handle_
    mfem::HypreParMatrix* A_imag_ptr_ = nullptr;  // borrowed from A_handle_

    // CCGD support
    mfem::HypreParMatrix* grad_mat_ = nullptr;  // borrowed
    std::unique_ptr<mfem::HypreParMatrix> GGt_;
    double ccgd_tau_ = 0.0;
};

}  // namespace forward
}  // namespace maple3dmt
