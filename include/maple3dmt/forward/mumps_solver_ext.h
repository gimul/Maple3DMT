// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file mumps_solver_ext.h
/// @brief Extended MUMPS solver with OOC, memory control, ICNTL access.
///
/// Wraps the MUMPS C interface directly (not through MFEM's MUMPSSolver)
/// to provide full control over ICNTL/CNTL parameters, including:
///   - Out-of-core (OOC) factorization  (ICNTL(22)=1)
///   - Memory workspace relaxation      (ICNTL(14))
///   - Maximum memory per process        (ICNTL(23))
///   - BLR compression                   (ICNTL(35), CNTL(7))
///   - Memory usage readback             (INFOG(16), INFOG(17))

#pragma once

#include "maple3dmt/common.h"
#include <mfem.hpp>
#include <memory>
#include <string>

#ifdef MFEM_USE_MUMPS
#include "dmumps_c.h"
#endif

#ifdef MAPLE3DMT_USE_MPI
#include <mpi.h>
#endif

namespace maple3dmt {
namespace forward {

/// MUMPS solver configuration
struct MUMPSConfig {
    bool   blr_enabled     = true;     // Use BLR compression
    double blr_tolerance   = 1e-10;    // CNTL(7)
    bool   ooc_enabled     = false;    // Out-of-core factorization
    std::string ooc_tmpdir = "/tmp";   // OOC temporary directory
    int    max_mem_mb      = 0;        // ICNTL(23), 0=auto
    int    mem_relax_pct   = 50;       // ICNTL(14), workspace relaxation %
    int    print_level     = 0;        // ICNTL(4), 0=silent
    bool   unsymmetric     = true;     // SYM=0 (unsymmetric)
};

#ifdef MFEM_USE_MUMPS

/// Extended MUMPS solver with full ICNTL/CNTL access.
///
/// Takes an mfem::HypreParMatrix, converts to distributed COO format,
/// and drives MUMPS directly via the C interface.
class MUMPSSolverExt : public mfem::Solver {
public:
    explicit MUMPSSolverExt(MPI_Comm comm, const MUMPSConfig& config = {});
    ~MUMPSSolverExt();

    // mfem::Solver interface
    void SetOperator(const mfem::Operator& op) override;
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

    /// Transpose solve: A^T y = x.  Uses MUMPS ICNTL(9)=2.
    /// Same factorization as Mult, no extra cost.
    void MultTranspose(const mfem::Vector& x, mfem::Vector& y) const override;

    /// Memory used by factorization (all procs total, MB)
    double factorization_memory_mb() const;

    /// Memory used by factorization (max single proc, MB)
    double factorization_peak_mb() const;

    /// Set ICNTL value directly (for advanced users)
    void SetICNTL(int idx, int val);

    /// Set CNTL value directly
    void SetCNTL(int idx, double val);

private:
    MPI_Comm comm_;
    int rank_, nprocs_;
    MUMPSConfig config_;

    DMUMPS_STRUC_C id_;
    bool initialized_ = false;
    bool factorized_  = false;

    // Distributed COO storage (local portion)
    std::vector<MUMPS_INT> irn_loc_, jcn_loc_;
    std::vector<double>    a_loc_;
    MUMPS_INT n_global_ = 0;

    // RHS/solution handling
    int row_start_ = 0;
    int local_size_ = 0;

    // Pre-allocated buffers for Mult() (avoid per-call allocation)
    mutable std::vector<double> rhs_global_buf_;
    mutable std::vector<double> x_local_buf_;
    mutable std::vector<double> y_local_buf_;
    mutable std::vector<int> recv_counts_;
    mutable std::vector<int> displs_;
    mutable bool mult_buffers_ready_ = false;
    void ensure_mult_buffers_() const;

    void init_mumps_();
    void convert_hyprepar_to_coo_(const mfem::HypreParMatrix& A);
    void set_icntl_params_();

    /// Release COO arrays after factorization (no longer needed for solve).
    /// Saves ~16 bytes per nonzero (irn + jcn + a).
    void release_coo_arrays_();
};

#endif // MFEM_USE_MUMPS

} // namespace forward
} // namespace maple3dmt
