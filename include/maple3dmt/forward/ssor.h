// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file ssor.h
/// @brief SSOR (Symmetric Successive Over-Relaxation) preconditioner for CSR.
///
/// For complex symmetric systems A = A^T (curl-curl + mass),
/// SSOR with omega=1 reduces to symmetric Gauss-Seidel:
///   M = (D + L) D^{-1} (D + U)
/// Applied as: M^{-1} r = (D+U)^{-1} D (D+L)^{-1} r
///   1. Forward sweep:  (D + omega*L) y = omega * r
///   2. Diagonal scale:  z = D * y
///   3. Backward sweep: (D + omega*U) x = z

#include "maple3dmt/common.h"
#include <functional>
#include <cmath>

namespace maple3dmt {
namespace forward {

/// SSOR preconditioner for CSRMatrix<Complex>.
/// Pre-extracts diagonal for efficiency. Supports relaxation parameter omega.
class SSORPreconditioner {
public:
    /// Setup from CSR matrix. Call once after assembly.
    void setup(const SparseMatC& A, Real omega = 1.0) {
        n_ = A.nrows;
        omega_ = omega;
        rowptr_ = &A.rowptr;
        colidx_ = &A.colidx;
        values_ = &A.values;

        // Extract diagonal and precompute inverse
        diag_.resize(n_);
        inv_diag_.resize(n_);
        for (int i = 0; i < n_; ++i) {
            Complex d(1, 0);  // fallback
            for (int k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) {
                if (A.colidx[k] == i) { d = A.values[k]; break; }
            }
            diag_[i] = d;
            inv_diag_[i] = (std::abs(d) > 1e-30) ? Complex(1, 0) / d : Complex(1, 0);
        }

        // Workspace
        y_.resize(n_);
    }

    /// Apply M^{-1} r -> out.
    /// Cost: 2 * nnz multiplications (same as 2 matvecs).
    void apply(const ComplexVec& r, ComplexVec& out) const {
        out.resize(n_);

        // Step 1: Forward sweep — solve (D + omega*L) y = omega * r
        //   y_i = (omega * r_i - omega * sum_{j<i} a_ij * y_j) / d_ii
        for (int i = 0; i < n_; ++i) {
            Complex sum(0, 0);
            for (int k = (*rowptr_)[i]; k < (*rowptr_)[i + 1]; ++k) {
                int j = (*colidx_)[k];
                if (j < i) sum += (*values_)[k] * y_[j];
            }
            y_[i] = omega_ * (r[i] - sum) * inv_diag_[i];
        }

        // Step 2+3: Diagonal scale + backward sweep
        //   Solve (D + omega*U) x = D * y
        //   x_i = (d_ii * y_i - omega * sum_{j>i} a_ij * x_j) / d_ii
        //       = y_i - omega * sum_{j>i} a_ij * x_j / d_ii
        for (int i = n_ - 1; i >= 0; --i) {
            Complex sum(0, 0);
            for (int k = (*rowptr_)[i]; k < (*rowptr_)[i + 1]; ++k) {
                int j = (*colidx_)[k];
                if (j > i) sum += (*values_)[k] * out[j];
            }
            out[i] = y_[i] - omega_ * sum * inv_diag_[i];
        }
    }

    /// Return as std::function callback for BiCGStab/COCG.
    std::function<void(const ComplexVec&, ComplexVec&)> callback() {
        return [this](const ComplexVec& in, ComplexVec& out) {
            this->apply(in, out);
        };
    }

private:
    int n_ = 0;
    Real omega_ = 1.0;
    ComplexVec diag_;
    ComplexVec inv_diag_;
    mutable ComplexVec y_;  // workspace for forward sweep

    // Non-owning pointers to CSR data (valid as long as A_ lives)
    const std::vector<int>* rowptr_ = nullptr;
    const std::vector<int>* colidx_ = nullptr;
    const std::vector<Complex>* values_ = nullptr;
};

} // namespace forward
} // namespace maple3dmt
