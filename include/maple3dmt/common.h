// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file common.h
/// @brief Common type aliases, constants, and macros for NewMT.

#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <cassert>
#include <cmath>
#include <algorithm>

#ifdef MAPLE3DMT_USE_MPI
#include <mpi.h>
#endif

namespace maple3dmt {

// -------------------------------------------------------------------------
// Scalar types
// -------------------------------------------------------------------------
using Real    = double;
using Complex = std::complex<double>;

// -------------------------------------------------------------------------
// Convenience aliases
// -------------------------------------------------------------------------
using RealVec    = std::vector<Real>;
using ComplexVec = std::vector<Complex>;

namespace fs = std::filesystem;

// -------------------------------------------------------------------------
// Sparse matrix (CSR format) for Octree FV
// -------------------------------------------------------------------------
/// Compressed Sparse Row matrix for complex or real entries.
template<typename T>
struct CSRMatrix {
    int nrows = 0;
    int ncols = 0;
    std::vector<int> rowptr;   // size nrows+1
    std::vector<int> colidx;   // size nnz
    std::vector<T>   values;   // size nnz

    int nnz() const { return static_cast<int>(values.size()); }

    /// y = A * x
    void matvec(const std::vector<T>& x, std::vector<T>& y) const {
        assert(static_cast<int>(x.size()) == ncols);
        y.assign(nrows, T(0));
        for (int i = 0; i < nrows; ++i) {
            T sum{};
            for (int k = rowptr[i]; k < rowptr[i + 1]; ++k)
                sum += values[k] * x[colidx[k]];
            y[i] = sum;
        }
    }

    /// y = A^T * x  (transpose matvec)
    void matvec_transpose(const std::vector<T>& x, std::vector<T>& y) const {
        assert(static_cast<int>(x.size()) == nrows);
        y.assign(ncols, T(0));
        for (int i = 0; i < nrows; ++i) {
            for (int k = rowptr[i]; k < rowptr[i + 1]; ++k)
                y[colidx[k]] += values[k] * x[i];
        }
    }
};

using SparseMatR = CSRMatrix<Real>;       // real sparse
using SparseMatC = CSRMatrix<Complex>;    // complex sparse

/// Diagonal matrix (stored as vector).
template<typename T>
struct DiagMatrix {
    std::vector<T> diag;
    int size() const { return static_cast<int>(diag.size()); }

    /// y = D * x
    void matvec(const std::vector<T>& x, std::vector<T>& y) const {
        int n = size();
        y.resize(n);
        for (int i = 0; i < n; ++i) y[i] = diag[i] * x[i];
    }

    /// y = D^{-1} * x
    void solve(const std::vector<T>& x, std::vector<T>& y) const {
        int n = size();
        y.resize(n);
        for (int i = 0; i < n; ++i) y[i] = x[i] / diag[i];
    }
};

using DiagMatR = DiagMatrix<Real>;
using DiagMatC = DiagMatrix<Complex>;

// -------------------------------------------------------------------------
// Physical constants
// -------------------------------------------------------------------------
namespace constants {
    inline constexpr Real MU0   = 4.0e-7 * 3.14159265358979323846;  // vacuum permeability
    inline constexpr Real PI    = 3.14159265358979323846;
    inline constexpr Real TWOPI = 2.0 * PI;
}

// -------------------------------------------------------------------------
// Version
// -------------------------------------------------------------------------
inline constexpr int VERSION_MAJOR = 0;
inline constexpr int VERSION_MINOR = 1;
inline constexpr int VERSION_PATCH = 0;

inline std::string version_string() {
    return std::to_string(VERSION_MAJOR) + "." +
           std::to_string(VERSION_MINOR) + "." +
           std::to_string(VERSION_PATCH);
}

} // namespace maple3dmt
