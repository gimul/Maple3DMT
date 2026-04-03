// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file mumps_solver_ext.cpp
/// @brief Extended MUMPS solver with OOC and full ICNTL control.

#include "maple3dmt/forward/mumps_solver_ext.h"
#include "maple3dmt/utils/logger.h"
#include "maple3dmt/utils/memory.h"

#ifdef MFEM_USE_MUMPS

#include <algorithm>
#include <cstring>
#include <numeric>

// Fortran-style 1-indexed access macros for MUMPS C arrays
#define ICNTL(i)  icntl[(i)-1]
#define CNTL(i)   cntl[(i)-1]
#define INFOG(i)  infog[(i)-1]
#define INFO(i)   info[(i)-1]
#define RINFOG(i) rinfog[(i)-1]

namespace maple3dmt {
namespace forward {

// =========================================================================
//  Construction / Destruction
// =========================================================================

MUMPSSolverExt::MUMPSSolverExt(MPI_Comm comm, const MUMPSConfig& config)
    : comm_(comm), config_(config)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nprocs_);
    std::memset(&id_, 0, sizeof(id_));
    init_mumps_();
}

MUMPSSolverExt::~MUMPSSolverExt() {
    if (initialized_) {
        id_.job = -2;  // terminate
        dmumps_c(&id_);
    }
}

void MUMPSSolverExt::init_mumps_() {
    id_.comm_fortran = MPI_Comm_c2f(comm_);
    id_.par  = 1;   // host participates
    id_.sym  = config_.unsymmetric ? 0 : 2;
    id_.job  = -1;  // initialize
    dmumps_c(&id_);

    if (id_.INFOG(1) != 0) {
        MAPLE3DMT_LOG_ERROR("MUMPS init failed: INFOG(1)=" +
                          std::to_string(id_.INFOG(1)));
        return;
    }

    initialized_ = true;
    set_icntl_params_();
}

void MUMPSSolverExt::set_icntl_params_() {
    // Output level
    id_.ICNTL(1) = (config_.print_level >= 1) ? 6 : 0;  // error stream
    id_.ICNTL(2) = (config_.print_level >= 2) ? 6 : 0;  // diagnostic
    id_.ICNTL(3) = (config_.print_level >= 2) ? 6 : 0;  // global info
    id_.ICNTL(4) = config_.print_level;

    // Distributed assembled matrix input
    id_.ICNTL(5)  = 0;   // assembled format
    id_.ICNTL(18) = 3;   // distributed matrix input

    // Ordering: automatic
    id_.ICNTL(7) = 7;    // automatic ordering

    // Workspace relaxation (default 20%, increase for safety)
    id_.ICNTL(14) = config_.mem_relax_pct;

    // Out-of-core
    if (config_.ooc_enabled) {
        id_.ICNTL(22) = 1;
        // Copy OOC directory into MUMPS structure
        std::strncpy(id_.ooc_tmpdir, config_.ooc_tmpdir.c_str(),
                     sizeof(id_.ooc_tmpdir) - 1);
        id_.ooc_tmpdir[sizeof(id_.ooc_tmpdir) - 1] = '\0';
        if (rank_ == 0) {
            MAPLE3DMT_LOG_INFO("MUMPS OOC enabled, tmpdir=" + config_.ooc_tmpdir);
        }
    }

    // Maximum memory per process (MB), 0 = let MUMPS decide
    if (config_.max_mem_mb > 0) {
        id_.ICNTL(23) = config_.max_mem_mb;
    }

    // BLR compression
    if (config_.blr_enabled) {
        id_.ICNTL(35) = 1;              // Enable BLR
        id_.CNTL(7)   = config_.blr_tolerance;
    }

    // Centralized RHS and solution (simplest approach)
    id_.ICNTL(20) = 0;   // centralized dense RHS on host
    id_.ICNTL(21) = 0;   // centralized solution on host
}

// =========================================================================
//  HypreParMatrix → MUMPS distributed COO
// =========================================================================

void MUMPSSolverExt::convert_hyprepar_to_coo_(const mfem::HypreParMatrix& A) {
    n_global_ = static_cast<MUMPS_INT>(A.GetGlobalNumRows());

    // Get local CSR data from hypre
    hypre_ParCSRMatrix* par_csr =
        const_cast<hypre_ParCSRMatrix*>(
            static_cast<const hypre_ParCSRMatrix*>(A));
    hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag(par_csr);
    hypre_CSRMatrix* offd = hypre_ParCSRMatrixOffd(par_csr);
    HYPRE_BigInt* col_map_offd = hypre_ParCSRMatrixColMapOffd(par_csr);

    int nrows_local = hypre_CSRMatrixNumRows(diag);
    int nnz_diag = hypre_CSRMatrixNumNonzeros(diag);
    int nnz_offd = hypre_CSRMatrixNumNonzeros(offd);
    int nnz_local = nnz_diag + nnz_offd;

    const HYPRE_BigInt* row_starts = A.RowPart();
    row_start_ = static_cast<int>(row_starts[0]);
    local_size_ = nrows_local;

    irn_loc_.resize(nnz_local);
    jcn_loc_.resize(nnz_local);
    a_loc_.resize(nnz_local);

    // Fill diagonal block
    auto* diag_i = hypre_CSRMatrixI(diag);
    auto* diag_j = hypre_CSRMatrixJ(diag);
    auto* diag_a = hypre_CSRMatrixData(diag);

    int idx = 0;
    for (int i = 0; i < nrows_local; ++i) {
        MUMPS_INT global_row = static_cast<MUMPS_INT>(row_start_ + i + 1);
        for (int k = diag_i[i]; k < diag_i[i+1]; ++k) {
            irn_loc_[idx] = global_row;
            jcn_loc_[idx] = static_cast<MUMPS_INT>(row_start_ + diag_j[k] + 1);
            a_loc_[idx]   = diag_a[k];
            ++idx;
        }
    }

    // Fill off-diagonal block
    auto* offd_i = hypre_CSRMatrixI(offd);
    auto* offd_j = hypre_CSRMatrixJ(offd);
    auto* offd_a = hypre_CSRMatrixData(offd);

    for (int i = 0; i < nrows_local; ++i) {
        MUMPS_INT global_row = static_cast<MUMPS_INT>(row_start_ + i + 1);
        for (int k = offd_i[i]; k < offd_i[i+1]; ++k) {
            irn_loc_[idx] = global_row;
            jcn_loc_[idx] = static_cast<MUMPS_INT>(col_map_offd[offd_j[k]] + 1);
            a_loc_[idx]   = offd_a[k];
            ++idx;
        }
    }
}

// =========================================================================
//  SetOperator — analyze + factorize
// =========================================================================

void MUMPSSolverExt::SetOperator(const mfem::Operator& op) {
    auto* A = dynamic_cast<const mfem::HypreParMatrix*>(&op);
    MFEM_VERIFY(A, "MUMPSSolverExt requires HypreParMatrix");

    double rss_before = utils::current_rss_gb();

    convert_hyprepar_to_coo_(*A);

    // Set matrix data
    id_.n       = n_global_;
    id_.nz_loc  = static_cast<MUMPS_INT>(a_loc_.size());
    id_.irn_loc = irn_loc_.data();
    id_.jcn_loc = jcn_loc_.data();
    id_.a_loc   = a_loc_.data();

    // Analysis phase
    id_.job = 1;
    dmumps_c(&id_);
    if (id_.INFOG(1) < 0) {
        MAPLE3DMT_LOG_ERROR("MUMPS analysis failed: INFOG(1)=" +
                          std::to_string(id_.INFOG(1)) +
                          " INFOG(2)=" + std::to_string(id_.INFOG(2)));
        return;
    }

    if (rank_ == 0) {
        // INFOG(17): estimated factorization memory (MB, total)
        // INFOG(16): estimated factorization memory (MB, max/proc)
        MAPLE3DMT_LOG_INFO("MUMPS analysis: est. factor mem = " +
                         std::to_string(id_.INFOG(17)) + " MB (total), " +
                         std::to_string(id_.INFOG(16)) + " MB (max/proc)");
    }

    // ── CRITICAL: abort BEFORE factorization if memory would exceed system ──
    // MUMPS factorization (job=2) may call MPI_Abort internally on OOM,
    // bypassing C++ exception handling.  We must check INFOG(17) from the
    // analysis phase and throw BEFORE attempting factorization.
    double avail_gb = utils::available_memory_gb();
    double total_gb = utils::total_memory_gb();
    double est_gb   = id_.INFOG(17) / 1024.0;  // INFOG(17) is in MB

    if (rank_ == 0) {
        MAPLE3DMT_LOG_INFO("MUMPS analysis estimate: " + utils::fmt_mem_gb(est_gb) +
                         " needed, " + utils::fmt_mem_gb(avail_gb) + " available, " +
                         utils::fmt_mem_gb(total_gb) + " total");
    }

    // MUMPS INFOG(17) is the DENSE factorization estimate (before BLR compression).
    // When BLR is enabled, actual memory is typically 5-10× less.
    // Apply estimated BLR compression factor before checking limits.
    double effective_est_gb = est_gb;
    if (config_.blr_enabled) {
        // BLR compression factor depends on tolerance:
        //   tol=1e-2 → ~10-20× compression, use 10× conservatively
        //   tol=1e-3 → ~5-10× compression, use 5×
        //   tol=1e-6 → ~2-3× compression
        double blr_factor = (config_.blr_tolerance >= 5e-3) ? 10.0 :
                            (config_.blr_tolerance >= 1e-4) ? 5.0 : 2.0;
        effective_est_gb = est_gb / blr_factor;
        if (rank_ == 0) {
            MAPLE3DMT_LOG_INFO("BLR compression: dense est=" + utils::fmt_mem_gb(est_gb) +
                             " → BLR est=" + utils::fmt_mem_gb(effective_est_gb) +
                             " (÷" + std::to_string(static_cast<int>(blr_factor)) + ")");
        }
    }

    // Memory check before factorization.
    // When OOC is enabled, MUMPS stores factors on disk — in-core memory
    // is only a fraction of total. Skip hard limit in OOC mode.
    // When BLR is enabled, the analysis estimate is for dense factors;
    // actual BLR memory is already accounted for in effective_est_gb.
    bool mem_exceeded = false;
    if (config_.ooc_enabled) {
        // OOC mode: MUMPS manages disk I/O, in-core needs ~10-20% of total.
        // Let MUMPS try; if it fails, the exception handler catches it.
        if (rank_ == 0) {
            MAPLE3DMT_LOG_INFO("OOC enabled: proceeding with factorization "
                             "(est=" + utils::fmt_mem_gb(effective_est_gb) +
                             ", OOC will use disk for overflow)");
        }
    } else if (effective_est_gb > total_gb * 0.90) {
        // No OOC, and estimated > 90% of total RAM — won't fit
        mem_exceeded = true;
        if (rank_ == 0) {
            MAPLE3DMT_LOG_WARNING("MUMPS estimated " + utils::fmt_mem_gb(effective_est_gb) +
                             " exceeds 90% of total RAM (" +
                             utils::fmt_mem_gb(total_gb) +
                             ") — aborting (try --ooc or --backend ITERATIVE)");
        }
    } else if (effective_est_gb > avail_gb * 0.90) {
        // No OOC, and exceeds available memory
        mem_exceeded = true;
        if (rank_ == 0) {
            MAPLE3DMT_LOG_WARNING("MUMPS estimated " + utils::fmt_mem_gb(effective_est_gb) +
                             " exceeds available memory (" +
                             utils::fmt_mem_gb(avail_gb) +
                             ") — aborting (try --ooc or --backend ITERATIVE)");
        }
    }

    // Broadcast decision to all ranks
    int abort_flag = mem_exceeded ? 1 : 0;
    MPI_Bcast(&abort_flag, 1, MPI_INT, 0, comm_);

    if (abort_flag) {
        // Release COO before throwing (they won't be needed)
        release_coo_arrays_();
        throw std::runtime_error(
            "MUMPS analysis: factorization would require ~" +
            std::to_string(static_cast<int>(est_gb)) +
            " GB but only " + std::to_string(static_cast<int>(avail_gb)) +
            " GB available. Use iterative solver (--backend HYBRID or ITERATIVE).");
    }

    // Factorization phase — safe to proceed
    id_.job = 2;
    dmumps_c(&id_);
    if (id_.INFOG(1) < 0) {
        release_coo_arrays_();
        throw std::runtime_error(
            "MUMPS factorization failed: INFOG(1)=" +
            std::to_string(id_.INFOG(1)) +
            " INFOG(2)=" + std::to_string(id_.INFOG(2)) +
            (id_.INFOG(1) == -9 || id_.INFOG(1) == -8
                ? " (out of memory)" : ""));
    }

    factorized_ = true;
    width = height = n_global_;

    // Release COO arrays — MUMPS no longer needs them after factorization
    release_coo_arrays_();

    double rss_after = utils::current_rss_gb();
    if (rank_ == 0) {
        MAPLE3DMT_LOG_INFO("MUMPS factorized: mem delta = " +
                         utils::fmt_mem_gb(rss_after - rss_before) +
                         " (RSS: " + utils::fmt_mem_gb(rss_after) + ")");
    }
}

// =========================================================================
//  Solve  (centralized RHS/solution on rank 0)
// =========================================================================

void MUMPSSolverExt::ensure_mult_buffers_() const {
    if (mult_buffers_ready_) return;

    int N = n_global_;

    recv_counts_.resize(nprocs_);
    displs_.resize(nprocs_);
    MPI_Allgather(&local_size_, 1, MPI_INT,
                  recv_counts_.data(), 1, MPI_INT, comm_);
    displs_[0] = 0;
    for (int i = 1; i < nprocs_; ++i) {
        displs_[i] = displs_[i-1] + recv_counts_[i-1];
    }

    if (rank_ == 0) {
        rhs_global_buf_.resize(N);
    }
    x_local_buf_.resize(local_size_);
    y_local_buf_.resize(local_size_);

    mult_buffers_ready_ = true;
}

void MUMPSSolverExt::Mult(const mfem::Vector& x, mfem::Vector& y) const {
    MFEM_VERIFY(factorized_, "MUMPS not factorized");

    ensure_mult_buffers_();

    auto& id = const_cast<DMUMPS_STRUC_C&>(id_);
    int N = n_global_;

    // Extract local portion of x
    for (int i = 0; i < local_size_; ++i) {
        x_local_buf_[i] = x(row_start_ + i);
    }

    // Gather RHS to rank 0
    MPI_Gatherv(x_local_buf_.data(), local_size_, MPI_DOUBLE,
                rhs_global_buf_.data(), recv_counts_.data(), displs_.data(),
                MPI_DOUBLE, 0, comm_);

    // Solve on rank 0
    if (rank_ == 0) {
        id.rhs  = rhs_global_buf_.data();
        id.nrhs = 1;
        id.lrhs = N;
    }

    id.job = 3;
    dmumps_c(&id);

    if (id.INFOG(1) < 0) {
        MAPLE3DMT_LOG_ERROR("MUMPS solve failed: INFOG(1)=" +
                          std::to_string(id.INFOG(1)));
    }

    // Solution is in rhs_global_buf_ on rank 0 (MUMPS overwrites RHS with solution)
    // Scatter back to all ranks
    y.SetSize(x.Size());
    y = 0.0;

    MPI_Scatterv(rank_ == 0 ? rhs_global_buf_.data() : nullptr,
                 recv_counts_.data(), displs_.data(), MPI_DOUBLE,
                 y_local_buf_.data(), local_size_, MPI_DOUBLE,
                 0, comm_);

    for (int i = 0; i < local_size_; ++i) {
        y(row_start_ + i) = y_local_buf_[i];
    }
}

// =========================================================================
//  Transpose solve: A^T y = x  (ICNTL(9)=2, same factorization)
// =========================================================================
void MUMPSSolverExt::MultTranspose(const mfem::Vector& x, mfem::Vector& y) const {
    MFEM_VERIFY(factorized_, "MUMPS not factorized");

    ensure_mult_buffers_();

    auto& id = const_cast<DMUMPS_STRUC_C&>(id_);
    int N = n_global_;

    // Extract local portion of x
    for (int i = 0; i < local_size_; ++i) {
        x_local_buf_[i] = x(row_start_ + i);
    }

    // Gather RHS to rank 0
    MPI_Gatherv(x_local_buf_.data(), local_size_, MPI_DOUBLE,
                rhs_global_buf_.data(), recv_counts_.data(), displs_.data(),
                MPI_DOUBLE, 0, comm_);

    // Solve A^T y = x on rank 0
    if (rank_ == 0) {
        id.rhs  = rhs_global_buf_.data();
        id.nrhs = 1;
        id.lrhs = N;
    }

    // ICNTL(9)=2 → solve A^T x = b instead of A x = b
    int saved_icntl9 = id.ICNTL(9);
    id.ICNTL(9) = 2;
    id.job = 3;
    dmumps_c(&id);
    id.ICNTL(9) = saved_icntl9;  // restore

    if (id.INFOG(1) < 0) {
        MAPLE3DMT_LOG_ERROR("MUMPS transpose solve failed: INFOG(1)=" +
                          std::to_string(id.INFOG(1)));
    }

    // Scatter solution back
    y.SetSize(x.Size());
    y = 0.0;

    MPI_Scatterv(rank_ == 0 ? rhs_global_buf_.data() : nullptr,
                 recv_counts_.data(), displs_.data(), MPI_DOUBLE,
                 y_local_buf_.data(), local_size_, MPI_DOUBLE,
                 0, comm_);

    for (int i = 0; i < local_size_; ++i) {
        y(row_start_ + i) = y_local_buf_[i];
    }
}

// =========================================================================
//  Utilities
// =========================================================================

void MUMPSSolverExt::release_coo_arrays_() {
    size_t freed_bytes = (irn_loc_.capacity() + jcn_loc_.capacity()) * sizeof(MUMPS_INT)
                       + a_loc_.capacity() * sizeof(double);

    // Swap with empty vectors to actually release memory (clear() doesn't shrink)
    { std::vector<MUMPS_INT>().swap(irn_loc_); }
    { std::vector<MUMPS_INT>().swap(jcn_loc_); }
    { std::vector<double>().swap(a_loc_); }

    // Null out MUMPS pointers (solve phase doesn't access these)
    id_.irn_loc = nullptr;
    id_.jcn_loc = nullptr;
    id_.a_loc   = nullptr;
    id_.nz_loc  = 0;

    if (rank_ == 0 && freed_bytes > 0) {
        MAPLE3DMT_LOG_INFO("Released COO arrays: " +
                         std::to_string(freed_bytes / (1024*1024)) + " MB freed");
    }
}

double MUMPSSolverExt::factorization_memory_mb() const {
    return static_cast<double>(id_.INFOG(17));
}

double MUMPSSolverExt::factorization_peak_mb() const {
    return static_cast<double>(id_.INFOG(16));
}

void MUMPSSolverExt::SetICNTL(int idx, int val) {
    id_.ICNTL(idx) = val;
}

void MUMPSSolverExt::SetCNTL(int idx, double val) {
    id_.CNTL(idx) = val;
}

} // namespace forward
} // namespace maple3dmt

#endif // MFEM_USE_MUMPS
