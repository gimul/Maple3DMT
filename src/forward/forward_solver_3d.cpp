// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file forward_solver_3d.cpp
/// @brief 3D MT forward solver using Nedelec edge elements.
///        Supports MUMPS BLR (direct), HypreAMS+FGMRES (iterative), and hybrid.

#include "maple3dmt/forward/forward_solver_3d.h"
#ifdef MFEM_USE_MUMPS
#include "maple3dmt/forward/mumps_solver_ext.h"
#endif
#include "maple3dmt/utils/logger.h"
#include "maple3dmt/utils/memory.h"
#include "maple3dmt/utils/memory_profiler.h"
#include <mfem/fem/complex_fem.hpp>
#include <HYPRE_parcsr_ls.h>   // HYPRE AMS C API (AMSCreate, AMSSetup, AMSSolve)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace maple3dmt {
namespace forward {

// Helper: get MPI rank (0 if no MPI)
static int my_rank(mfem::ParMesh* m) {
    int r = 0;
#ifdef MAPLE3DMT_USE_MPI
    if (m) MPI_Comm_rank(m->GetComm(), &r);
#endif
    return r;
}
// Rank-0-only logging macros
#define LOG0_INFO(msg)  do { if (my_rank(mesh_) == 0) MAPLE3DMT_LOG_INFO(msg); } while(0)
#define LOG0_DEBUG(msg) do { if (my_rank(mesh_) == 0) MAPLE3DMT_LOG_DEBUG(msg); } while(0)
#define LOG0_WARN(msg)  do { if (my_rank(mesh_) == 0) MAPLE3DMT_LOG_WARNING(msg); } while(0)

/// Element-wise constant coefficient (indexed by local element number).
class ElementCoefficient : public mfem::Coefficient {
    const mfem::Vector& values_;
public:
    ElementCoefficient(const mfem::Vector& v) : values_(v) {}
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint&) override {
        return values_(T.ElementNo);
    }
};

// =========================================================================
// DoubledAMS — PRESB-optimal AMS preconditioner for 2N×2N block system
// =========================================================================
/// PRESB + block-diagonal AMS with explicit beta Poisson matrix.
///
/// For A = [K, -B; B, K], P = diag(S, S) where S = K + |ωσ|M.
/// Each S^{-1} approximated by HYPRE AMS V-cycle.
///
/// Three approaches for beta Poisson and their trade-offs:
///   (a) BetaPoissonMatrix(NULL):  fast setup, NO beta correction → 82 iter (BAD)
///   (b) Not set (HYPRE auto):     HYPRE computes G^T·A·G → 6+ min setup (SLOW)
///   (c) Explicit beta Poisson:    build H1 form ∫|ωσ|∇u·∇v → fast + correct (THIS)
///
/// PRESB block-diagonal preconditioner with inner PCG(AMS).
///
/// P = diag(S, S) where S = K + |ωσ|M (N×N SPD on edge space).
/// Each S^{-1} approximated by PCG with AMS V-cycle preconditioner.
/// Inner PCG converges in ~5-15 iterations to tol=1e-2 (~1% accuracy).
///
/// MFEM HypreAMS wrapper provides G, coords automatically from fespace.
/// Key optimization: pre-computed G^T·S·G is injected into HYPRE handle
/// BEFORE SetOperator(), preventing HYPRE's slow auto-computation (6+ min).
///
/// Ref: EMFEM (github.com/emfem/emfem), HYPRE ex15.c, MFEM ex25p
class BlockDiagAMS_PRESB : public mfem::Solver {
    std::unique_ptr<mfem::HypreAMS> ams_;
    std::unique_ptr<mfem::HyprePCG> pcg_;
    int N_;
    mutable std::unique_ptr<mfem::HypreParVector> b_work_, x_work_;

public:
    /// @param S_nxn         N×N SPD: K + |ωσ|M (edge space, BCs eliminated)
    /// @param beta_poisson  G^T·S·G on H1 space (pre-computed), or nullptr = let HYPRE auto-compute
    /// @param edge_fespace  Edge FE space (MFEM HypreAMS builds G, coords from this)
    /// @param inner_tol     Inner PCG convergence tolerance (1e-2 = 1% accuracy)
    /// @param inner_maxiter Max inner PCG iterations
    /// @param smooth_type   HYPRE smoother: 2=l1-hybrid-GS (MFEM default), 6=hybrid-symGS
    /// @param smooth_sweeps Smoother sweeps per AMS level
    BlockDiagAMS_PRESB(mfem::HypreParMatrix& S_nxn,
                       mfem::HypreParMatrix* beta_poisson,
                       mfem::ParFiniteElementSpace* edge_fespace,
                       double inner_tol = 1e-2,
                       int inner_maxiter = 20,
                       int smooth_type = 2,
                       int smooth_sweeps = 1)
        : N_(S_nxn.Height())
    {
        this->height = 2 * N_;
        this->width  = 2 * N_;

        // 1. Create MFEM HypreAMS (deferred setup — G, coords built automatically)
        ams_ = std::make_unique<mfem::HypreAMS>(edge_fespace);

        // 2. Get HYPRE handle BEFORE setup to inject beta Poisson and smoother
        HYPRE_Solver ams_h = (HYPRE_Solver)(*ams_);

        if (beta_poisson) {
            // Pre-computed G^T·S·G → HYPRE skips internal G^T·A·G (saves 6+ min)
            HYPRE_AMSSetBetaPoissonMatrix(ams_h,
                (HYPRE_ParCSRMatrix)(hypre_ParCSRMatrix*) *beta_poisson);
        }
        // If beta_poisson is nullptr, MFEM default: HYPRE auto-computes G^T·A·G

        // Override MFEM default smoother settings if needed
        HYPRE_AMSSetSmoothingOptions(ams_h, smooth_type, smooth_sweeps, 1.0, 1.0);

        // 3. Trigger AMS setup (uses pre-set beta Poisson, skips auto-computation)
        ams_->SetOperator(S_nxn);
        HYPRE_ClearAllErrors();

        // 4. Inner PCG with AMS as preconditioner (EMFEM approach)
        pcg_ = std::make_unique<mfem::HyprePCG>(S_nxn.GetComm());
        pcg_->SetPreconditioner(*ams_);
        pcg_->SetTol(inner_tol);
        pcg_->SetMaxIter(inner_maxiter);
        pcg_->SetPrintLevel(0);
        pcg_->SetOperator(S_nxn);

        // Work vectors
        MPI_Comm comm = S_nxn.GetComm();
        HYPRE_BigInt gr = S_nxn.GetGlobalNumRows();
        HYPRE_BigInt* rs = S_nxn.RowPart();
        b_work_.reset(new mfem::HypreParVector(comm, gr, rs));
        x_work_.reset(new mfem::HypreParVector(comm, gr, rs));
    }

    void SetOperator(const mfem::Operator&) override {}

    void Mult(const mfem::Vector& b, mfem::Vector& x) const override {
        const double* bd = b.GetData();
        double* xd = x.GetData();

        // Block 1: real part (first N entries)
        std::memcpy(b_work_->GetData(), bd, N_ * sizeof(double));
        *x_work_ = 0.0;
        pcg_->Mult(*b_work_, *x_work_);
        std::memcpy(xd, x_work_->GetData(), N_ * sizeof(double));

        // Block 2: imaginary part (next N entries)
        std::memcpy(b_work_->GetData(), bd + N_, N_ * sizeof(double));
        *x_work_ = 0.0;
        pcg_->Mult(*b_work_, *x_work_);
        std::memcpy(xd + N_, x_work_->GetData(), N_ * sizeof(double));
    }

    ~BlockDiagAMS_PRESB() = default;
};

// =========================================================================
// DivCorrectedPreconditioner — AMS + divergence correction (ModEM approach)
// =========================================================================
/// Wraps an existing preconditioner (typically BlockDiagAMS_PRESB) and applies
/// divergence correction after each preconditioning step.
///
/// Problem: Edge element (Nédélec) systems have a curl-curl null space
/// consisting of gradient fields ∇φ.  AMS handles this for smooth RHS,
/// but point-source adjoint RHS (delta functions at stations) excites
/// these gradient modes, causing AMS to stagnate at rel_res ~ 0.6.
///
/// Solution (following ModEM, Egbert & Kelbert 2012):
///   1. Apply inner preconditioner:  z = M^{-1} r
///   2. Project out gradient component:
///      φ = L^{-1} G^T z     where L = G^T G (scalar Laplacian on H1)
///      z ← z - G φ
///
/// For the 2N×2N block-real system [K,-B;B,K], the correction is applied
/// independently to each N-block (real and imaginary parts).
///
/// The scalar Laplacian solve (L φ = G^T z) is cheap: GMRES + BoomerAMG
/// on the H1 space (GMRES, not CG: G^T G is semi-definite with constant
/// null space, so CG fails with "not positive definite").
class DivCorrectedPreconditioner : public mfem::Solver {
    mfem::Solver* inner_prec_;              // not owned (existing AMS)
    mfem::HypreParMatrix* G_;              // discrete gradient (N_edge × N_node), not owned
    std::unique_ptr<mfem::HypreParMatrix> GtG_;   // scalar Laplacian G^T G
    std::unique_ptr<mfem::GMRESSolver> gmres_L_;   // GMRES for Laplacian (not CG: G^TG is singular)
    std::unique_ptr<mfem::HypreBoomerAMG> amg_;    // AMG preconditioner for Laplacian
    int nd_size_;   // edge DOF count per block (N)
    int h1_size_;   // node DOF count
    MPI_Comm comm_;

    // Work vectors (avoid per-call allocation)
    mutable mfem::Vector z_block_, Gt_z_, phi_;

public:
    /// @param inner   Existing preconditioner (e.g. DoubledAMS) — not owned
    /// @param G       Discrete gradient matrix (N_edge × N_node) — not owned
    /// @param comm    MPI communicator
    /// @param div_tol Tolerance for scalar Laplacian solve
    /// @param div_maxiter Max iterations for Laplacian
    DivCorrectedPreconditioner(mfem::Solver* inner,
                                mfem::HypreParMatrix* G,
                                MPI_Comm comm,
                                double div_tol = 1e-6,
                                int div_maxiter = 200)
        : inner_prec_(inner), G_(G), comm_(comm)
    {
        // Dimensions
        nd_size_ = G->Height();  // edge DOFs (local)
        h1_size_ = G->Width();   // node DOFs (local)

        // Build scalar Laplacian: L = G^T G
        // NOTE: G^T G is symmetric positive SEMI-definite — it has a constant
        // null space because G·(const) = 0.  CG fails with "not positive
        // definite" when roundoff introduces null-space components in G^T z.
        // Solution: use GMRES instead of CG.  GMRES handles singular/near-
        // singular symmetric systems without the positive-definiteness check.
        // Since this is a small H1 auxiliary solve inside the preconditioner,
        // the performance difference vs CG is negligible.
        {
            auto Gt = std::unique_ptr<mfem::HypreParMatrix>(G->Transpose());
            GtG_.reset(mfem::ParMult(Gt.get(), G));
        }

        // AMG for scalar Laplacian (handles near-singular Poisson well)
        amg_ = std::make_unique<mfem::HypreBoomerAMG>(*GtG_);
        amg_->SetPrintLevel(0);

        // GMRES solver (robust for semi-definite L = G^T G)
        gmres_L_ = std::make_unique<mfem::GMRESSolver>(comm);
        gmres_L_->SetOperator(*GtG_);
        gmres_L_->SetPreconditioner(*amg_);
        gmres_L_->SetRelTol(div_tol);
        gmres_L_->SetMaxIter(div_maxiter);
        gmres_L_->SetKDim(50);  // small Krylov dim (H1 problem converges fast)
        gmres_L_->SetPrintLevel(-1);

        // Allocate work vectors
        z_block_.SetSize(nd_size_);
        Gt_z_.SetSize(h1_size_);
        phi_.SetSize(h1_size_);

        // Set Solver dimensions (2N × 2N, same as inner preconditioner)
        this->height = 2 * nd_size_;
        this->width  = 2 * nd_size_;
    }

    void SetOperator(const mfem::Operator&) override { /* no-op */ }

    void Mult(const mfem::Vector& b, mfem::Vector& x) const override {
        // Step 1: Apply inner preconditioner (if available)
        if (inner_prec_) {
            inner_prec_->Mult(b, x);
        } else {
            x = b;  // identity (used when DivCorr is applied externally)
        }

        // Step 2: Divergence correction on each block independently
        div_correct_(x.GetData());             // Block 0 (real): x[0..N-1]
        div_correct_(x.GetData() + nd_size_);  // Block 1 (imag): x[N..2N-1]
    }

    /// Apply divergence correction externally to a single N-block (ModEM-style).
    /// Called from adjoint outer loop, NOT from inside preconditioner.
    void div_correct_external(double* z_data, int size) const {
        if (size != nd_size_) return;  // safety check
        div_correct_(z_data);
    }

    ~DivCorrectedPreconditioner() = default;

private:
    /// Apply divergence correction in-place to a single N-block.
    /// z ← z - G (G^T G)^{-1} G^T z
    void div_correct_(double* z_data) const {
        // Wrap z_data as Vector (no copy)
        mfem::Vector z_vec(z_data, nd_size_);

        // Compute G^T z
        G_->MultTranspose(z_vec, Gt_z_);

        // Solve L φ = G^T z  where L = G^T G
        phi_ = 0.0;
        gmres_L_->Mult(Gt_z_, phi_);

        // z ← z - G φ
        G_->Mult(-1.0, phi_, 1.0, z_vec);
    }
};

// =========================================================================
// CCGDOperator — wraps A + τ·GGt without modifying A (BC-safe)
// =========================================================================
/// Computes y = A*x + τ*diag(GGt,GGt)*x per block, then restores BCs.
/// Uses N×N GGt applied independently to real and imag blocks,
/// avoiding HypreParMatrixFromBlocks (which produces corrupted matrices
/// in some MFEM/HYPRE configurations).
class CCGDOperator : public mfem::Operator {
    mfem::Operator* A_;             // original 2N×2N system (with BCs), not owned
    mfem::HypreParMatrix* GGt_;    // N×N grad-div term, not owned
    double tau_;
    int N_;                         // block size (= TrueVSize of ND space)
    mfem::Array<int> ess_2N_;      // essential DOFs in 2N system
    mutable mfem::Vector tmp_block_; // N-sized temp
public:
    CCGDOperator(mfem::Operator* A, mfem::HypreParMatrix* GGt, double tau,
                 const mfem::Array<int>& ess_tdof, int tdof_per_block)
        : Operator(A->Height(), A->Width()),
          A_(A), GGt_(GGt), tau_(tau), N_(tdof_per_block),
          tmp_block_(tdof_per_block)
    {
        // Build 2N essential DOF list
        ess_2N_.SetSize(2 * ess_tdof.Size());
        for (int i = 0; i < ess_tdof.Size(); i++) {
            ess_2N_[i] = ess_tdof[i];
            ess_2N_[i + ess_tdof.Size()] = ess_tdof[i] + tdof_per_block;
        }
    }

    void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
        A_->Mult(x, y);  // y = A*x (2N×2N system with BCs)

        // Apply τ·GGt to real block: y[0..N-1] += τ * GGt * x[0..N-1]
        {
            mfem::Vector x_r(const_cast<double*>(x.GetData()), N_);
            mfem::Vector y_r(y.GetData(), N_);
            GGt_->Mult(x_r, tmp_block_);
            y_r.Add(tau_, tmp_block_);
        }

        // Apply τ·GGt to imag block: y[N..2N-1] += τ * GGt * x[N..2N-1]
        {
            mfem::Vector x_i(const_cast<double*>(x.GetData()) + N_, N_);
            mfem::Vector y_i(y.GetData() + N_, N_);
            GGt_->Mult(x_i, tmp_block_);
            y_i.Add(tau_, tmp_block_);
        }

        // Restore BC rows: essential DOFs must satisfy y[i] = x[i]
        for (int i = 0; i < ess_2N_.Size(); i++) {
            y(ess_2N_[i]) = x(ess_2N_[i]);
        }
    }
};

// =========================================================================
// setup()
// =========================================================================
void ForwardSolver3D::setup(mfem::ParMesh& mesh,
                             const model::ConductivityModel& model,
                             const ForwardParams3D& params) {
    mesh_   = &mesh;
    model_  = &model;
    params_ = params;

    // Invalidate model-dependent caches (σ may have changed)
    GtσMG_.reset();          // G^T·(σM)·G depends on σ

    // HYPRE 3.x requires explicit HYPRE_Init() before any HYPRE operations.
    // MFEM's Mpi::Init() does NOT call this. Safe to call multiple times.
    HYPRE_Init();
#if MFEM_HYPRE_VERSION >= 21600
    HYPRE_SetPrintErrorMode(1);
#endif

    fec_ = std::make_unique<mfem::ND_FECollection>(params_.fe_order, 3);
    fespace_ = std::make_unique<mfem::ParFiniteElementSpace>(mesh_, fec_.get());

    int ndof = fespace_->GlobalTrueVSize();
    LOG0_INFO("ForwardSolver3D setup: order=" +
                   std::to_string(params_.fe_order) +
                   ", ndof=" + std::to_string(ndof) +
                   " (block system: " + std::to_string(2 * ndof) + ")");

    E1_real_ = std::make_unique<mfem::ParGridFunction>(fespace_.get());
    E1_imag_ = std::make_unique<mfem::ParGridFunction>(fespace_.get());
    E2_real_ = std::make_unique<mfem::ParGridFunction>(fespace_.get());
    E2_imag_ = std::make_unique<mfem::ParGridFunction>(fespace_.get());

    // Compute background conductivity as the median of earth elements.
    // This avoids depending on element ordering (element 0 could be
    // deep, air, or a refined sub-element).
    {
        int ne = mesh_->GetNE();
        std::vector<Real> earth_sigmas;
        earth_sigmas.reserve(ne);
        for (int e = 0; e < ne; ++e) {
            if (mesh_->GetAttribute(e) != 2) {  // skip air
                earth_sigmas.push_back(model_->sigma(e));
            }
        }
        if (!earth_sigmas.empty()) {
            std::sort(earth_sigmas.begin(), earth_sigmas.end());
            sigma_bg_ = earth_sigmas[earth_sigmas.size() / 2];
        } else {
            sigma_bg_ = 0.01;  // fallback: 100 Ωm
        }
        LOG0_INFO("Background sigma for 1D primary field: " +
                  std::to_string(sigma_bg_) + " S/m");
    }
    stations_found_ = false;
    system_ready_ = false;
    using_iterative_ = false;

    // Cache essential DOFs
    mfem::Array<int> ess_bdr(mesh_->bdr_attributes.Max());
    ess_bdr = 1;
    fespace_->GetEssentialTrueDofs(ess_bdr, ess_tdof_list_);

    // AMS auxiliary data is built on-demand (setup_ams_auxiliary_) if needed.
    // Currently BoomerAMG is used for the block system; AMS reserved for
    // future block-diagonal preconditioner with separate real/imag blocks.
}

// =========================================================================
// setup_ams_auxiliary_()
// =========================================================================
void ForwardSolver3D::setup_ams_auxiliary_() {
    if (ams_ready_) return;

    LOG0_INFO("Building AMS auxiliary data (H1 space + discrete gradient)...");

    // H1 nodal FE space (same order as edge space)
    h1_fec_ = std::make_unique<mfem::H1_FECollection>(params_.fe_order, 3);
    h1_fespace_ = std::make_unique<mfem::ParFiniteElementSpace>(
        mesh_, h1_fec_.get());

    // Discrete gradient matrix: maps H1 → Nedelec (grad operator)
    {
        mfem::ParDiscreteLinearOperator grad(h1_fespace_.get(), fespace_.get());
        grad.AddDomainInterpolator(new mfem::GradientInterpolator());
        grad.Assemble();
        grad.Finalize();
        grad_mat_.reset(grad.ParallelAssemble());
    }

    // Coordinate vectors for AMS (node positions)
    mfem::ParGridFunction x_gf(h1_fespace_.get());
    mfem::ParGridFunction y_gf(h1_fespace_.get());
    mfem::ParGridFunction z_gf(h1_fespace_.get());

    // Project coordinate functions onto H1 space
    auto x_func = [](const mfem::Vector& p) { return p(0); };
    auto y_func = [](const mfem::Vector& p) { return p(1); };
    auto z_func = [](const mfem::Vector& p) { return p(2); };

    mfem::FunctionCoefficient x_coeff(x_func);
    mfem::FunctionCoefficient y_coeff(y_func);
    mfem::FunctionCoefficient z_coeff(z_func);

    x_gf.ProjectCoefficient(x_coeff);
    y_gf.ProjectCoefficient(y_coeff);
    z_gf.ProjectCoefficient(z_coeff);

    x_coord_ = std::make_unique<mfem::HypreParVector>(h1_fespace_.get());
    y_coord_ = std::make_unique<mfem::HypreParVector>(h1_fespace_.get());
    z_coord_ = std::make_unique<mfem::HypreParVector>(h1_fespace_.get());

    x_gf.GetTrueDofs(*x_coord_);
    y_gf.GetTrueDofs(*y_coord_);
    z_gf.GetTrueDofs(*z_coord_);

    ams_ready_ = true;
    LOG0_INFO("AMS auxiliary data ready: H1 DOF=" +
              std::to_string(h1_fespace_->GlobalTrueVSize()));
}

// =========================================================================
// compute_primary_field_()
// =========================================================================
void ForwardSolver3D::compute_primary_field_(
    Real omega, int polarization,
    mfem::ParGridFunction& E0_real,
    mfem::ParGridFunction& E0_imag)
{
    Real alpha = std::sqrt(omega * constants::MU0 * sigma_bg_ / 2.0);
    int pol = polarization;
    Real a = alpha;

    auto E0_real_func = [pol, a](const mfem::Vector& x, mfem::Vector& E) {
        E.SetSize(3);
        E = 0.0;
        Real z = x(2);
        if (z >= 0.0) {
            E(pol) = 1.0;
        } else {
            Real depth = -z;
            E(pol) = std::exp(-a * depth) * std::cos(a * depth);
        }
    };

    auto E0_imag_func = [pol, a](const mfem::Vector& x, mfem::Vector& E) {
        E.SetSize(3);
        E = 0.0;
        Real z = x(2);
        if (z < 0.0) {
            Real depth = -z;
            E(pol) = -std::exp(-a * depth) * std::sin(a * depth);
        }
    };

    mfem::VectorFunctionCoefficient E0r_coeff(3, E0_real_func);
    mfem::VectorFunctionCoefficient E0i_coeff(3, E0_imag_func);

    E0_real.ProjectCoefficient(E0r_coeff);
    E0_imag.ProjectCoefficient(E0i_coeff);
}

// =========================================================================
// find_stations_()
// =========================================================================
void ForwardSolver3D::find_stations_(const data::MTData& data) {
    if (stations_found_) return;

    int ns = data.num_stations();
    station_pts_.SetSize(3, ns);

    // First attempt: use station elevation - 5m (just below terrain surface)
    for (int i = 0; i < ns; ++i) {
        const auto& s = data.station(i);
        station_pts_(0, i) = s.x;
        station_pts_(1, i) = s.y;
        station_pts_(2, i) = s.z - 5.0;
    }

    int found = mesh_->FindPoints(station_pts_, station_elem_ids_, station_ips_);

    // Each rank only has elem_ids >= 0 for stations in its LOCAL partition.
    // Use Allreduce to get global count.
    auto global_found = [&]() -> int {
        int local_count = 0;
        for (int i = 0; i < ns; ++i) {
            if (station_elem_ids_[i] >= 0) ++local_count;
        }
        int global_count = local_count;
#ifdef MAPLE3DMT_USE_MPI
        MPI_Allreduce(&local_count, &global_count, 1,
                       MPI_INT, MPI_SUM, mesh_->GetComm());
#endif
        return global_count;
    };

    int found_global = global_found();
    LOG0_INFO("Station search (1st pass): " + std::to_string(found_global) +
              "/" + std::to_string(ns) + " found in earth elements");

    // Check for stations in air (attr=2) on the owning rank
    for (int i = 0; i < ns; ++i) {
        if (station_elem_ids_[i] >= 0 &&
            mesh_->GetAttribute(station_elem_ids_[i]) == 2) {
            station_elem_ids_[i] = -1;  // mark for retry
        }
    }

    // Second attempt with z = -5.0 for unfound/air stations (flat surface fallback)
    found_global = global_found();
    if (found_global < ns) {
        mfem::DenseMatrix retry_pts(3, ns);
        for (int i = 0; i < ns; ++i) {
            retry_pts(0, i) = station_pts_(0, i);
            retry_pts(1, i) = station_pts_(1, i);
            retry_pts(2, i) = -5.0;  // 5m below flat surface
        }

        mfem::Array<int> retry_ids;
        mfem::Array<mfem::IntegrationPoint> retry_ips;
        mesh_->FindPoints(retry_pts, retry_ids, retry_ips);

        int relocated_local = 0;
        for (int i = 0; i < ns; ++i) {
            if (station_elem_ids_[i] >= 0) continue;  // already found in earth
            if (retry_ids[i] >= 0 && mesh_->GetAttribute(retry_ids[i]) == 1) {
                station_elem_ids_[i] = retry_ids[i];
                station_ips_[i] = retry_ips[i];
                station_pts_(2, i) = -5.0;
                ++relocated_local;
            }
        }

        int relocated_global = relocated_local;
#ifdef MAPLE3DMT_USE_MPI
        MPI_Allreduce(&relocated_local, &relocated_global, 1,
                       MPI_INT, MPI_SUM, mesh_->GetComm());
#endif

        if (relocated_global > 0) {
            LOG0_INFO("Relocated " + std::to_string(relocated_global) +
                      "/" + std::to_string(ns) +
                      " stations to z=-5.0 (5m below surface)");
        }
    }

    // Final global count
    found_global = global_found();
    if (found_global < ns) {
        LOG0_WARN("Only " + std::to_string(found_global) + "/" +
                  std::to_string(ns) + " stations found in mesh");
    } else {
        LOG0_INFO("All " + std::to_string(ns) + " stations located in mesh");
    }

    station_H_cache_.resize(ns);
    stations_found_ = true;
}

// =========================================================================
// assemble_and_factorize_()
// =========================================================================
void ForwardSolver3D::assemble_and_factorize_(Real omega) {
    current_omega_ = omega;

    // ── Memory limit: 85% of total physical RAM ──
    const double total_ram_gb = utils::total_memory_gb();
    const double mem_limit_gb = total_ram_gb * 0.85;
    const size_t mem_limit_bytes =
        static_cast<size_t>(mem_limit_gb * 1024.0 * 1024.0 * 1024.0);

    // ── Pre-factorization memory check + HYBRID decision ──
    bool use_direct = false;  // will be set based on backend + memory
    {
        double rss = utils::current_rss_gb();
        double avail = utils::available_memory_gb();

        // Estimate factorization cost for MUMPS LU on 3D unstructured hex mesh.
        // 3D FEM → wide bandwidth → fill-in depends heavily on ordering + BLR.
        // Previous estimate (40K bytes/DOF for BLR) was based on 1.26M-element mesh
        // without considering MPI distribution and BLR compression effectiveness.
        // BLR with tol~1e-3 typically achieves 5-10× compression on 3D FEM.
        // Updated estimates (total across all MPI ranks):
        //   BLR (tol~1e-3): ~10,000 bytes/DOF  (5-10× compression observed)
        //   No BLR:         ~90,000 bytes/DOF
        long long ndof = fespace_->GlobalTrueVSize();
        double bytes_per_dof = 10000.0;  // BLR: aggressive but MUMPS will fail+fallback if OOM
        if (params_.backend == SolverBackend::MUMPS) bytes_per_dof = 90000.0;
        double est_factor_gb = ndof * 2.0 * bytes_per_dof / 1e9;

        LOG0_DEBUG("Pre-assembly: RSS=" + utils::fmt_mem_gb(rss) +
                   " avail=" + utils::fmt_mem_gb(avail) +
                   " est_factor=" + utils::fmt_mem_gb(est_factor_gb) +
                   " limit=" + utils::fmt_mem_gb(mem_limit_gb));

        // Decide direct vs iterative vs complex
        if (params_.backend == SolverBackend::COMPLEX_BICGSTAB) {
            use_direct = false;
            using_complex_ = true;
            LOG0_INFO("COMPLEX_BICGSTAB: Complex N×N BiCGStab + ILU(0) selected");
        } else if (params_.backend == SolverBackend::ITERATIVE) {
            use_direct = false;
        } else if (params_.backend == SolverBackend::HYBRID) {
            // Use direct if estimated total fits within threshold
            double threshold = total_ram_gb * params_.direct_mem_fraction;
            use_direct = (rss + est_factor_gb < threshold);

            if (use_direct) {
                LOG0_DEBUG("HYBRID: direct solver selected (est " +
                          utils::fmt_mem_gb(rss + est_factor_gb) +
                          " < threshold " + utils::fmt_mem_gb(threshold) + ")");
            } else {
                LOG0_DEBUG("HYBRID: iterative solver selected (est " +
                          utils::fmt_mem_gb(rss + est_factor_gb) +
                          " >= threshold " + utils::fmt_mem_gb(threshold) + ")");
            }
        } else {
            // MUMPS, MUMPS_BLR — try direct, with OOC guard
            use_direct = true;
            if (rss + est_factor_gb > mem_limit_gb && !params_.ooc_enabled) {
                LOG0_WARN("MEMORY GUARD: est " + utils::fmt_mem_gb(rss + est_factor_gb) +
                          " > limit " + utils::fmt_mem_gb(mem_limit_gb) +
                          " — auto-enabling OOC");
                params_.ooc_enabled = true;
            }
        }
    }

    // ── Build sesquilinear form ──
    LOG0_INFO("[TIMING] Factorize: entering assemble_and_factorize_, ω=" +
              std::to_string(omega));
    auto t_factorize_start = std::chrono::high_resolution_clock::now();
    auto conv = mfem::ComplexOperator::HERMITIAN;
    sesq_form_ = std::make_unique<mfem::ParSesquilinearForm>(fespace_.get(), conv);

    inv_mu0_coeff_ = std::make_unique<mfem::ConstantCoefficient>(1.0 / constants::MU0);
    sesq_form_->AddDomainIntegrator(
        new mfem::CurlCurlIntegrator(*inv_mu0_coeff_), NULL);

    int ne = mesh_->GetNE();
    neg_omega_sigma_vec_.SetSize(ne);
    for (int i = 0; i < ne; ++i) {
        int attr = mesh_->GetAttribute(i);
        neg_omega_sigma_vec_(i) = (attr == 2)
            ? -omega * 1e-6
            : -omega * model_->sigma(i);
    }
    mass_coeff_ = std::make_unique<ElementCoefficient>(neg_omega_sigma_vec_);
    sesq_form_->AddDomainIntegrator(
        NULL, new mfem::VectorFEMassIntegrator(*mass_coeff_));

    LOG0_INFO("[TIMING] Factorize: assembling sesquilinear form...");
    sesq_form_->Assemble();
    mem_prof_.snap("after_assemble");
    LOG0_INFO("[TIMING] Factorize: sesquilinear form assembled");

    // ── Extract system matrix (tight scope to release intermediates) ──
    {
        static int factorize_call_count_ = 0;
        ++factorize_call_count_;
        int vsize = fespace_->GetVSize();
        int tvsize = fespace_->GetTrueVSize();

        LOG0_INFO("[DIAG] assemble_and_factorize_ call #" +
                  std::to_string(factorize_call_count_) +
                  "  vsize=" + std::to_string(vsize) +
                  "  true_vsize=" + std::to_string(tvsize) +
                  "  ess_tdof=" + std::to_string(ess_tdof_list_.Size()) +
                  "  NE=" + std::to_string(mesh_->GetNE()));
        LOG0_INFO("[DIAG] RSS=" + utils::fmt_mem_gb(utils::current_rss_gb()) +
                  " avail=" + utils::fmt_mem_gb(utils::available_memory_gb()));
        // Force flush all output before potentially hanging call
        std::cout << std::flush;
        std::cerr << std::flush;
        if (auto* f = fopen("/dev/null","r")) fclose(f); // force stdio flush

        // Note: MPI_Barrier removed here. The original hang was caused by
        // frequency parallelism: different groups called MFEM collective ops
        // at different times on the global communicator → deadlock.

        // Clear any HYPRE errors from previous frequency
        HYPRE_ClearAllErrors();

        // ── Manual system matrix extraction ──
        // Bypass ParSesquilinearForm::FormLinearSystem for system matrix.
        // Build real (curl-curl) and imaginary (mass) parts separately,
        // then combine into ComplexHypreParMatrix → monolithic 2N×2N system.
        // This avoids a suspected hang in complex FormLinearSystem.
        // sesq_form_ is kept alive for solve_polarization_ BC handling.
        LOG0_INFO("[DIAG] Building system matrix via manual ParallelAssemble...");
        std::cout << std::flush;

        mfem::HypreParMatrix* A_real_par = nullptr;
        mfem::HypreParMatrix* A_imag_par = nullptr;

        // Step 1: Real part = (1/μ₀) curl-curl
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            mfem::ParBilinearForm real_form(fespace_.get());
            real_form.AddDomainIntegrator(
                new mfem::CurlCurlIntegrator(*inv_mu0_coeff_));
            real_form.Assemble();
            real_form.Finalize();
            A_real_par = real_form.ParallelAssemble();
            elim_real_.reset(A_real_par->EliminateRowsCols(ess_tdof_list_));
            auto t1 = std::chrono::high_resolution_clock::now();
            LOG0_INFO("[DIAG] Step 1 (real curl-curl): " +
                      std::to_string(std::chrono::duration<double>(t1-t0).count()) +
                      " sec, rows=" + std::to_string(A_real_par->GetGlobalNumRows()));
            std::cout << std::flush;
        }

        // Step 2: Imaginary part = -ωσ mass
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            mfem::ParBilinearForm imag_form(fespace_.get());
            imag_form.AddDomainIntegrator(
                new mfem::VectorFEMassIntegrator(*mass_coeff_));
            imag_form.Assemble();
            imag_form.Finalize();
            A_imag_par = imag_form.ParallelAssemble();
            elim_imag_.reset(A_imag_par->EliminateRowsCols(ess_tdof_list_));
            auto t1 = std::chrono::high_resolution_clock::now();
            LOG0_INFO("[DIAG] Step 2 (imag mass): " +
                      std::to_string(std::chrono::duration<double>(t1-t0).count()) +
                      " sec, rows=" + std::to_string(A_imag_par->GetGlobalNumRows()));
            std::cout << std::flush;
        }

        // Step 3: Combine into ComplexHypreParMatrix → 2N×2N monolithic
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto conv = mfem::ComplexOperator::HERMITIAN;
            mfem::ComplexHypreParMatrix cA(A_real_par, A_imag_par,
                                           true, true, conv);
            system_matrix_.reset(cA.GetSystemMatrix());
            // cA destructor deletes A_real_par and A_imag_par (owns=true)
            auto t1 = std::chrono::high_resolution_clock::now();
            LOG0_INFO("[DIAG] Step 3 (ComplexHypreParMatrix): " +
                      std::to_string(std::chrono::duration<double>(t1-t0).count()) +
                      " sec");
            std::cout << std::flush;
        }

        LOG0_DEBUG("System matrix assembled: size=" +
                       std::to_string(system_matrix_->GetGlobalNumRows()));
    }
    {
        auto t_sysmat = std::chrono::high_resolution_clock::now();
        LOG0_INFO("[TIMING] System matrix assembly: " +
                  std::to_string(std::chrono::duration<double>(
                      t_sysmat - t_factorize_start).count()) + " sec");
    }
    // Note: ComplexHypreParMatrix cA owns and deletes A_real_par/A_imag_par.
    // system_matrix_ (from GetSystemMatrix) is an independent copy.

    // ── CCGD: Add τ · G · G^T to deflate gradient null space ──
    // (Dong & Egbert, 2019 GJI — simplified to bare topological G·G^T)
    //
    // The gradient null space of the curl-curl operator causes AMS to
    // stagnate at rel_res ~0.6 for point-source adjoint RHS.
    // Adding τ·G·G^T lifts null-space eigenvalues from 0 to O(τ),
    // breaking the stagnation.
    //
    // We use BARE G·G^T (topological, ±1 entries, eigenvalues O(1))
    // rather than mass-weighted G·M_σ·G^T because:
    //  - Simpler assembly (no H1 mass matrix needed)
    //  - No interaction with AMS internal G^T·A·G computation
    //  - τ is chosen adaptively to ensure τ·||GGt|| ≪ ||A||
    //
    // Auto τ: τ = target_ratio × maxdiag(A) / maxdiag(GGt)
    // Default target_ratio = 0.01 → CCGD is ~1% of system scale.
    // This is enough to lift null-space eigenvalues while barely
    // affecting forward solution accuracy.
    // CCGD disabled: adjoint now uses R-transform with forward solver,
    // so the gradient null-space augmentation (A + τ·GG^T) is no longer needed.
    // This saves ~80 MB/rank by not computing/caching G·G^T.
    if (params_.ccgd_enabled) {
        LOG0_INFO("CCGD: skipped — adjoint uses R-transform (forward solver reuse)");
        // Don't build ccgd_GGt_ or compute tau.
        // ccgd_op_ won't be created either (line below checks ccgd_GGt_).
    }

    // NOTE: A^T for adjoint solve is built lazily on first adjoint_solve() call.
    // This avoids expensive Transpose() when only forward solve is needed.

    // ── Memory check after assembly ──
    mem_prof_.snap("after_form_linear_system");
    {
        double rss = utils::current_rss_gb();
        double avail = utils::available_memory_gb();
        LOG0_DEBUG("Memory after assembly: RSS=" + utils::fmt_mem_gb(rss) +
                  " avail=" + utils::fmt_mem_gb(avail));

        // Guard: if already beyond limit after assembly alone, bail
        if (MemoryProfiler::check_limit(mem_limit_bytes, mesh_->GetComm())) {
            LOG0_WARN("MEMORY LIMIT EXCEEDED after assembly! RSS=" +
                      utils::fmt_mem_gb(rss) + " — cannot proceed");
            system_matrix_.reset();
            elim_real_.reset();
            elim_imag_.reset();
            sesq_form_.reset();
            inv_mu0_coeff_.reset();
            mass_coeff_.reset();
            neg_omega_sigma_vec_.Destroy();
            throw std::runtime_error(
                "Out of memory after assembly: RSS " + utils::fmt_mem_gb(rss) +
                " exceeds " + utils::fmt_mem_gb(mem_limit_gb));
        }

        // Last-resort OOC check (< 2 GB available)
        if (!params_.ooc_enabled && avail < 2.0) {
            LOG0_WARN("Critical: only " + utils::fmt_mem_gb(avail) +
                      " available — forcing OOC mode");
            params_.ooc_enabled = true;
        }
    }

    // ── Helper: build iterative solver (FGMRES + AMG on SPD preconditioner) ──
    //
    // Preconditioner: P = diag(S, S) where S = K + |ωσ|M   (2N×2N SPD)
    //
    // Assembled via ParSesquilinearForm(real=S, imag=0) → GetSystemMatrix.
    // This uses the SAME code path as system_matrix_, which is proven to
    // produce correct parallel structure on the buggy MFEM 4.9 hex ParMesh.
    //
    // ParBilinearForm::FormSystemMatrix → N×N → SEGFAULT (corrupted parallel data)
    // ParSesquilinearForm → GetSystemMatrix → 2N×2N → works!
    //
    // P⁻¹A has eigenvalues clustered near 1 ± iα (α < 1),
    // so FGMRES converges in ~20-100 iterations.
    //
    // CRITICAL: HYPRE_ClearAllErrors() before any HypreSolver construction.
    auto build_iterative_solver_ = [&]() {
        LOG0_INFO("[TIMING] Factorize: building iterative solver...");

        // ── Clear stale HYPRE error flags ──
        {
            HYPRE_Int herr = HYPRE_GetError();
            if (herr) {
                LOG0_WARN("Stale HYPRE error flag detected: " +
                          std::to_string(herr) + " — clearing before solver setup");
                HYPRE_ClearAllErrors();
            }
        }

        // ── 1. Build |ωσ| coefficient (shared by both 2N×2N and N×N paths) ──
        {
            int ne = mesh_->GetNE();
            abs_omega_sigma_vec_.SetSize(ne);
            for (int i = 0; i < ne; ++i) {
                int attr = mesh_->GetAttribute(i);
                double sigma = (attr == 2) ? 1e-6 : model_->sigma(i);
                abs_omega_sigma_vec_(i) = std::abs(current_omega_) * sigma;
            }
            abs_mass_coeff_ = std::make_unique<ElementCoefficient>(
                abs_omega_sigma_vec_);
        }

        HYPRE_ClearAllErrors();

        // ── 2. Build preconditioner ──
        std::string precond_name;
        mfem::Solver* prec_solver = nullptr;

        // NOTE: 2N×2N PRESB matrix (prec_matrix_) is only needed for DIAG and AMG
        // fallback paths. AMS path builds its own N×N SPD block (spd_nxn_) directly,
        // saving ~3 GB by avoiding the unnecessary 2N×2N assembly.

        // Helper lambda: build 2N×2N PRESB matrix (only for DIAG/AMG paths)
        auto build_prec_matrix_2N = [&]() {
            auto conv = mfem::ComplexOperator::HERMITIAN;
            mfem::ParSesquilinearForm prec_sesq(fespace_.get(), conv);
            prec_sesq.AddDomainIntegrator(
                new mfem::CurlCurlIntegrator(*inv_mu0_coeff_), NULL);
            prec_sesq.AddDomainIntegrator(
                new mfem::VectorFEMassIntegrator(*abs_mass_coeff_), NULL);
            prec_sesq.Assemble();
            int vsize = fespace_->GetVSize();
            mfem::Vector x_p(2 * vsize), b_p(2 * vsize);
            x_p = 0.0; b_p = 0.0;
            mfem::OperatorHandle A_p;
            mfem::Vector X_p, B_p;
            prec_sesq.FormLinearSystem(ess_tdof_list_, x_p, b_p, A_p, X_p, B_p);
            auto* cA_p = dynamic_cast<mfem::ComplexHypreParMatrix*>(A_p.Ptr());
            prec_matrix_.reset(cA_p->GetSystemMatrix());
            LOG0_INFO("PRESB 2N×2N SPD: size=" +
                      std::to_string(prec_matrix_->GetGlobalNumRows()));
        };

        if (params_.precond == PrecondType::DIAG) {
            build_prec_matrix_2N();
            auto* diag = new mfem::HypreDiagScale(*prec_matrix_);
            ams_prec_.reset(diag);
            prec_solver = diag;
            precond_name = "DiagScale(SPD-2N)";
        } else if (params_.precond == PrecondType::AMS) {
            // PRESB + AMS on N×N SPD block.
            // P = diag(S, S) where S = K+|ωσ|M.
            // AMS V-cycle(s) approximate S^{-1} for each block.

            // Ensure AMS auxiliary data (G, coords)
            setup_ams_auxiliary_();

            // Build N×N SPD block: S = K + |ωσ|M (edge space)
            {
                auto t_spd0 = std::chrono::high_resolution_clock::now();
                mfem::ParBilinearForm spd_form(fespace_.get());
                spd_form.AddDomainIntegrator(
                    new mfem::CurlCurlIntegrator(*inv_mu0_coeff_));
                spd_form.AddDomainIntegrator(
                    new mfem::VectorFEMassIntegrator(*abs_mass_coeff_));
                spd_form.Assemble();
                spd_form.Finalize();
                spd_nxn_.reset(spd_form.ParallelAssemble());
                auto* elim = spd_nxn_->EliminateRowsCols(ess_tdof_list_);
                delete elim;
                auto t_spd1 = std::chrono::high_resolution_clock::now();
                LOG0_INFO("[TIMING] S=K+|ωσ|M assembly: " +
                          std::to_string(std::chrono::duration<double>(
                              t_spd1 - t_spd0).count()) + " sec");
            }

            // ── Beta Poisson: G^T·S·G = G^T·K·G + |ω|·G^T·(σM)·G ──
            // Split into frequency-independent components for fast assembly:
            //   GtKG_:  G^T·K·G  — mesh-only, computed ONCE (cached forever)
            //   GtσMG_: G^T·(σM)·G — model σ only, recomputed per model update
            //   Per frequency: beta_poisson_ = GtKG_ + |ω| * GtσMG_ (ms)
            {
                auto t0 = std::chrono::high_resolution_clock::now();

                // Step 1: G^T·K·G (curl-curl part, mesh-only, compute once)
                if (!GtKG_ready_) {
                    // Build K (curl-curl only, no mass term)
                    mfem::ParBilinearForm K_form(fespace_.get());
                    K_form.AddDomainIntegrator(
                        new mfem::CurlCurlIntegrator(*inv_mu0_coeff_));
                    K_form.Assemble();
                    K_form.Finalize();
                    auto K_mat = std::unique_ptr<mfem::HypreParMatrix>(
                        K_form.ParallelAssemble());
                    auto* K_elim = K_mat->EliminateRowsCols(ess_tdof_list_);
                    delete K_elim;

                    auto Gt = std::unique_ptr<mfem::HypreParMatrix>(
                        grad_mat_->Transpose());
                    auto GtK = std::unique_ptr<mfem::HypreParMatrix>(
                        mfem::ParMult(Gt.get(), K_mat.get()));
                    GtKG_.reset(mfem::ParMult(GtK.get(), grad_mat_.get()));
                    GtKG_ready_ = true;

                    auto t_gtkg = std::chrono::high_resolution_clock::now();
                    double sec = std::chrono::duration<double>(t_gtkg - t0).count();
                    LOG0_INFO("G^T·K·G computed (one-time): " +
                              std::to_string(sec) + " sec, size=" +
                              std::to_string(GtKG_->GetGlobalNumRows()));
                }

                // Step 2: G^T·(σM)·G (mass part, depends on σ only)
                // Uses DiffusionIntegrator on H1 space: ∫σ∇u·∇v dx
                // Cached across frequencies (σ doesn't change between freqs).
                // Rebuilt only when model changes (new inversion iteration).
                if (!GtσMG_) {
                    int ne = mesh_->GetNE();
                    mfem::Vector sigma_vec(ne);
                    for (int i = 0; i < ne; ++i) {
                        int attr = mesh_->GetAttribute(i);
                        sigma_vec(i) = (attr == 2) ? 1e-6 : model_->sigma(i);
                    }
                    ElementCoefficient sigma_coeff(sigma_vec);

                    mfem::ParBilinearForm bp_form(h1_fespace_.get());
                    bp_form.AddDomainIntegrator(
                        new mfem::DiffusionIntegrator(sigma_coeff));
                    bp_form.Assemble();
                    bp_form.Finalize();
                    GtσMG_.reset(bp_form.ParallelAssemble());
                    LOG0_INFO("G^T·(σM)·G computed (cached per model), size=" +
                              std::to_string(GtσMG_->GetGlobalNumRows()));
                }

                // Step 3: Combine — beta_poisson_ = GtKG_ + |ω| * GtσMG_
                double omega_abs = std::abs(current_omega_);
                // HypreParMatrix Add: C = α*A + β*B
                beta_poisson_.reset(
                    mfem::Add(1.0, *GtKG_, omega_abs, *GtσMG_));

                auto t1 = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(t1 - t0).count();
                LOG0_INFO("Beta Poisson assembled in " +
                          std::to_string(sec) + " sec" +
                          (GtKG_ready_ ? " (GtKG cached)" : "") +
                          ", |ω|=" + std::to_string(omega_abs));
            }

            LOG0_INFO("PRESB N×N SPD (K+B): " +
                      std::to_string(spd_nxn_->GetGlobalNumRows()) +
                      ", inner PCG(AMS, tol=1e-2)");

            // BlockDiagAMS_PRESB: MFEM HypreAMS wrapper + inner PCG
            {
                auto t_ams0 = std::chrono::high_resolution_clock::now();
                auto* bdams = new BlockDiagAMS_PRESB(
                    *spd_nxn_, beta_poisson_.get(), fespace_.get(),
                    /*inner_tol=*/1e-2, /*inner_maxiter=*/20,
                    params_.ams_smooth_type, params_.ams_smooth_sweeps);
                block_prec_.reset(bdams);
                prec_solver = bdams;
                auto t_ams1 = std::chrono::high_resolution_clock::now();
                LOG0_INFO("[TIMING] BlockDiagAMS_PRESB construction: " +
                          std::to_string(std::chrono::duration<double>(
                              t_ams1 - t_ams0).count()) + " sec");
            }
            precond_name = "PRESB+PCG(AMS,GtKG+wGtsMG,1e-2)";
        } else {
            // Fallback: tuned AMG on 2N×2N SPD (stagnates for edge elements
            // but still usable as diagnostic baseline).
            build_prec_matrix_2N();
            auto* amg = new mfem::HypreBoomerAMG(*prec_matrix_);
            amg->SetPrintLevel(0);
            amg->SetStrongThresholdR(0.5);   // 0.5 for 3D
            amg->SetRelaxType(6);            // Symmetric GS
            amg->SetCycleNumSweeps(2, 3);    // 2 down, 3 up
            amg->SetCoarsening(10);          // HMIS
            amg->SetInterpolation(6);        // Extended+i
            ams_prec_.reset(amg);
            prec_solver = amg;
            precond_name = "AMG(SPD-2N,tuned)";
        }

        HYPRE_ClearAllErrors();
        LOG0_DEBUG("Preconditioner: " + precond_name);

        // ── 3. FGMRES on 2N×2N system ──
        auto* fgmres = new mfem::FGMRESSolver(mesh_->GetComm());
        // Forward solve: use original system_matrix_ (no CCGD).
        // CCGD is applied ONLY for the adjoint solve (point-source RHS),
        // where the gradient null space causes AMS to stagnate.
        // For forward (smooth RHS), AMS alone converges well.
        fgmres->SetOperator(*system_matrix_);

        // Build CCGDOperator for adjoint use (if enabled)
        if (ccgd_GGt_) {
            int tdof = fespace_->GetTrueVSize();
            ccgd_op_ = std::make_unique<CCGDOperator>(
                system_matrix_.get(), ccgd_GGt_.get(),
                ccgd_tau_actual_, ess_tdof_list_, tdof);
            LOG0_DEBUG("CCGD operator ready for adjoint (τ=" +
                      std::to_string(ccgd_tau_actual_) + ", adjoint-only)");
        }
        fgmres->SetPreconditioner(*prec_solver);
        fgmres->SetRelTol(params_.gmres_tol);
        fgmres->SetAbsTol(1e-15);
        fgmres->SetMaxIter(params_.gmres_maxiter);
        fgmres->SetKDim(params_.gmres_kdim);
        // Print level: -1 (auto) → suppress if progress callback is set
        int print_lvl = params_.gmres_print;
        if (print_lvl < 0) {
            print_lvl = freq_progress_cb_ ? 0 : 1;
        }
        fgmres->SetPrintLevel(my_rank(mesh_) == 0 ? print_lvl : 0);

        solver_.reset(fgmres);
        using_iterative_ = true;

        long long global_block_dof = fespace_->GlobalTrueVSize() * 2;
        double krylov_gb = (2.0 * params_.gmres_kdim + 1) * global_block_dof * 8.0 / 1e9;

        {
            double rss = utils::current_rss_gb();
            double avail = utils::available_memory_gb();
            LOG0_DEBUG("Iterative solver ready: FGMRES(k=" +
                      std::to_string(params_.gmres_kdim) +
                      ") + " + precond_name +
                      ", tol=" + std::to_string(params_.gmres_tol) +
                      ", maxiter=" + std::to_string(params_.gmres_maxiter) +
                      ", Krylov ~" + utils::fmt_mem_gb(krylov_gb) +
                      "  RSS=" + utils::fmt_mem_gb(rss) +
                      " avail=" + utils::fmt_mem_gb(avail));
        }
    };

    // ── Helper: rebuild system_matrix_ from separate real/imag forms ──
    auto rebuild_system_matrix_ = [&]() {
        // Manual rebuild matching the approach in assemble_and_factorize_
        mfem::ParBilinearForm real_form(fespace_.get());
        real_form.AddDomainIntegrator(
            new mfem::CurlCurlIntegrator(*inv_mu0_coeff_));
        real_form.Assemble();
        real_form.Finalize();
        auto* A_r = real_form.ParallelAssemble();
        elim_real_.reset(A_r->EliminateRowsCols(ess_tdof_list_));

        mfem::ParBilinearForm imag_form(fespace_.get());
        imag_form.AddDomainIntegrator(
            new mfem::VectorFEMassIntegrator(*mass_coeff_));
        imag_form.Assemble();
        imag_form.Finalize();
        auto* A_i = imag_form.ParallelAssemble();
        elim_imag_.reset(A_i->EliminateRowsCols(ess_tdof_list_));

        auto conv = mfem::ComplexOperator::HERMITIAN;
        mfem::ComplexHypreParMatrix cA(A_r, A_i, true, true, conv);
        system_matrix_.reset(cA.GetSystemMatrix());
    };

#ifdef MFEM_USE_MUMPS
    // ── Helper: attempt MUMPS direct factorization ──
    auto factorize_mumps_ext_ = [&](bool ooc_override) {
        MUMPSConfig cfg;
        cfg.blr_enabled   = (params_.backend == SolverBackend::MUMPS_BLR ||
                             params_.backend == SolverBackend::HYBRID);
        cfg.blr_tolerance = params_.blr_tolerance;
        cfg.ooc_enabled   = ooc_override;
        cfg.ooc_tmpdir    = params_.ooc_tmpdir;
        cfg.max_mem_mb    = params_.max_mem_mb;
        cfg.mem_relax_pct = params_.mem_relax_pct;
        cfg.unsymmetric   = true;
        auto ext = std::make_unique<MUMPSSolverExt>(mesh_->GetComm(), cfg);
        ext->SetOperator(*system_matrix_);
        solver_.reset(ext.release());
        system_matrix_.reset();
        using_iterative_ = false;
        LOG0_DEBUG("MUMPS-Ext factorization complete");
    };
#endif // MFEM_USE_MUMPS

    // ── Main solver selection ──
    if (use_direct) {
#ifdef MFEM_USE_MUMPS
        // Try direct factorization, with fallback chain:
        //   in-core → OOC → iterative
        // Note: factorize_mumps_ext_ uses unique_ptr, so on exception the
        // MUMPSSolverExt is properly destroyed (MUMPS context freed).
        // system_matrix_ remains valid after a failed attempt because
        // system_matrix_.reset() only runs AFTER successful factorization.
        try {
            factorize_mumps_ext_(params_.ooc_enabled);
        } catch (const std::exception& e) {
            LOG0_WARN("Direct factorization failed: " + std::string(e.what()));
            solver_.reset();  // clean up any partial solver state

            if (!params_.ooc_enabled) {
                // Retry 1: OOC (system_matrix_ still valid)
                LOG0_WARN("Retrying with OOC...");
                params_.ooc_enabled = true;

                try {
                    factorize_mumps_ext_(true);
                } catch (const std::exception& e2) {
                    // Retry 2: fall back to iterative
                    LOG0_WARN("OOC also failed: " + std::string(e2.what()));
                    LOG0_WARN("Falling back to iterative solver");
                    solver_.reset();
                    // system_matrix_ is still valid — no need to rebuild
                    if (!system_matrix_) rebuild_system_matrix_();
                    build_iterative_solver_();
                }
            } else {
                // OOC was already on — fall back to iterative
                LOG0_WARN("Falling back to iterative solver");
                // system_matrix_ is still valid — no need to rebuild
                if (!system_matrix_) rebuild_system_matrix_();
                build_iterative_solver_();
            }
        }
#else
        LOG0_WARN("MUMPS not available — using iterative solver");
        build_iterative_solver_();
#endif
    } else if (using_complex_) {
        // ── Complex N×N BiCGStab + ILU(0) path (ModEM-equivalent) ──
        complex_solver_ = std::make_unique<ComplexSolverWrapper>();
        ComplexSolverWrapper::Params cp;
        cp.tol        = params_.gmres_tol;
        cp.maxiter    = params_.gmres_maxiter;
        cp.adj_tol    = params_.adjoint_tol;
        cp.adj_maxiter = params_.adjoint_maxiter;
        cp.print_lvl  = (params_.gmres_print < 0)
                        ? (freq_progress_cb_ ? 0 : 1)
                        : params_.gmres_print;
        cp.ccgd_enabled = params_.ccgd_enabled;
        cp.ccgd_tau     = params_.ccgd_tau;

        complex_solver_->build(*sesq_form_, grad_mat_.get(),
                               ess_tdof_list_, mesh_->GetComm(), cp);

        using_iterative_ = true;  // for compatibility flags
        LOG0_INFO("Complex N×N solver ready (BiCGStab + ILU(0))");
    } else {
        // Iterative path (ITERATIVE mode, or HYBRID chose iterative)
        build_iterative_solver_();
    }

    // ── Post-solve memory report ──
    mem_prof_.snap("after_factorize");
    {
        auto t_factorize_end = std::chrono::high_resolution_clock::now();
        LOG0_INFO("[TIMING] Total Factorize phase: " +
                  std::to_string(std::chrono::duration<double>(
                      t_factorize_end - t_factorize_start).count()) + " sec");
    }
    {
        double rss = utils::current_rss_gb();
        double avail = utils::available_memory_gb();
        std::string mode = using_complex_ ? " [COMPLEX]"
                         : (using_iterative_ ? " [ITERATIVE]" : " [DIRECT]");
        LOG0_DEBUG("Memory after solver setup: RSS=" + utils::fmt_mem_gb(rss) +
                  " avail=" + utils::fmt_mem_gb(avail) + mode);

        // Post-factorize guard (mainly for direct solver)
        if (!using_iterative_ &&
            MemoryProfiler::check_limit(mem_limit_bytes, mesh_->GetComm())) {
            LOG0_WARN("MEMORY LIMIT EXCEEDED after factorize! RSS=" +
                      utils::fmt_mem_gb(rss) + " — releasing and aborting");
            solver_.reset();
            precond_.reset();
            system_matrix_.reset();
            elim_real_.reset();
            elim_imag_.reset();
            sesq_form_.reset();
            inv_mu0_coeff_.reset();
            mass_coeff_.reset();
            neg_omega_sigma_vec_.Destroy();
            throw std::runtime_error(
                "Out of memory after factorization: RSS " +
                utils::fmt_mem_gb(rss) + " exceeds " +
                utils::fmt_mem_gb(mem_limit_gb));
        }
    }

    system_ready_ = true;
}

// =========================================================================
// solve_polarization_()
// =========================================================================
void ForwardSolver3D::solve_polarization_(
    const mfem::ParGridFunction& E0_real,
    const mfem::ParGridFunction& E0_imag,
    mfem::ParGridFunction& E_real_out,
    mfem::ParGridFunction& E_imag_out,
    const mfem::ParGridFunction* /*seed_real*/,
    const mfem::ParGridFunction* /*seed_imag*/)
{
    // ── Manual BC handling (bypasses ParSesquilinearForm::FormLinearSystem) ──
    // Uses elim_real_/elim_imag_ (stored by assemble_and_factorize_) to compute
    // the RHS from essential BC values, and system_matrix_ for the solve.
    // This avoids a suspected hang in ParSesquilinearForm::FormLinearSystem.

    if (!elim_real_ || !elim_imag_) {
        throw std::runtime_error(
            "solve_polarization_: elim_real_/elim_imag_ not set. "
            "Was assemble_and_factorize_() called?");
    }

    auto* P = fespace_->GetProlongationMatrix();
    int vsize  = fespace_->GetVSize();
    int tdof   = fespace_->GetTrueVSize();

    // ── 1. Project E0 (primary field) to true-DOF space ──
    mfem::Vector X(2 * tdof), B(2 * tdof);
    X = 0.0; B = 0.0;

    // Views into real/imaginary halves
    mfem::Vector X_r(X.GetData(), tdof);
    mfem::Vector X_i(X.GetData() + tdof, tdof);
    mfem::Vector B_r(B.GetData(), tdof);
    mfem::Vector B_i(B.GetData() + tdof, tdof);

    // E0 in L-vector space → true-DOF space via P^T (restriction)
    {
        mfem::Vector e0r(const_cast<double*>(E0_real.GetData()), vsize);
        mfem::Vector e0i(const_cast<double*>(E0_imag.GetData()), vsize);
        P->MultTranspose(e0r, X_r);
        P->MultTranspose(e0i, X_i);
    }

    // ── 2. Compute RHS using eliminated parts ──
    // For HERMITIAN convention, block system:
    //   [A_r  -A_i] [x_r]   [0]
    //   [A_i   A_r] [x_i] = [0]
    //
    // RHS = -(eliminated part of A) * X
    //   B_r = -(elim_r * X_r - elim_i * X_i)   [real block RHS]
    //   B_i = -(elim_i * X_r + elim_r * X_i)   [imag block RHS]
    {
        mfem::Vector temp(tdof);

        // B_r = -(elim_r * X_r - elim_i * X_i)
        elim_real_->Mult(X_r, B_r);     // B_r = elim_r * X_r
        elim_imag_->Mult(X_i, temp);    // temp = elim_i * X_i
        B_r -= temp;                     // B_r = elim_r * X_r - elim_i * X_i
        B_r *= -1.0;                    // B_r = -(elim_r * X_r - elim_i * X_i)

        // B_i = -(elim_i * X_r + elim_r * X_i)
        elim_imag_->Mult(X_r, B_i);     // B_i = elim_i * X_r
        elim_real_->Mult(X_i, temp);    // temp = elim_r * X_i
        B_i += temp;                     // B_i = elim_i * X_r + elim_r * X_i
        B_i *= -1.0;                    // B_i = -(elim_i * X_r + elim_r * X_i)
    }

    // ── 3. Set essential DOF values ──
    for (int i = 0; i < ess_tdof_list_.Size(); i++) {
        int k = ess_tdof_list_[i];
        B_r(k) = X_r(k);
        B_i(k) = X_i(k);
    }

    {
        double rss = utils::current_rss_gb();
        double avail = utils::available_memory_gb();
        LOG0_DEBUG("Solving: |B|=" + std::to_string(B.Norml2()) +
                  "  B.Size=" + std::to_string(B.Size()) +
                  "  RSS=" + utils::fmt_mem_gb(rss) +
                  " avail=" + utils::fmt_mem_gb(avail));
    }

    // Clear any stale HYPRE errors before the solve.
    {
        HYPRE_Int herr = HYPRE_GetError();
        if (herr) {
            LOG0_WARN("Clearing stale HYPRE error " + std::to_string(herr) +
                      " before solve");
            HYPRE_ClearAllErrors();
        }
    }

    // ── 4. Solve ──
    X = 0.0;  // zero initial guess
    if (using_complex_ && complex_solver_ && complex_solver_->ready()) {
        mfem::Vector X_real(tdof), X_imag(tdof);
        X_real = 0.0;
        X_imag = 0.0;

        mfem::Vector B_real_view(B.GetData(), tdof);
        mfem::Vector B_imag_view(B.GetData() + tdof, tdof);
        auto result = complex_solver_->solve(B_real_view, B_imag_view,
                                             X_real, X_imag);
        LOG0_INFO("Complex solve: " +
                 std::string(result.converged ? "converged" : "NOT converged") +
                 " iter=" + std::to_string(result.iterations) +
                 " rel_res=" + std::to_string(result.final_residual));

        for (int i = 0; i < tdof; ++i) {
            X(i)        = X_real(i);
            X(i + tdof) = X_imag(i);
        }
    } else {
        // Block-real 2N×2N path (FGMRES+AMS or MUMPS)
        solver_->Mult(B, X);
    }

    LOG0_DEBUG("Solve completed, |X|=" + std::to_string(X.Norml2()));

    // ── 5. Recover solution: x = P * X (true-DOF → L-DOF) ──
    {
        mfem::Vector X_sol_r(X.GetData(), tdof);
        mfem::Vector X_sol_i(X.GetData() + tdof, tdof);
        mfem::Vector out_r(E_real_out.GetData(), vsize);
        mfem::Vector out_i(E_imag_out.GetData(), vsize);
        P->Mult(X_sol_r, out_r);
        P->Mult(X_sol_i, out_i);
    }
}

// =========================================================================
// compute_single_frequency()
// =========================================================================
void ForwardSolver3D::compute_single_frequency(
    int freq_idx,
    const data::MTData& observed,
    data::MTData& predicted)
{
    Real freq = observed.frequencies()[freq_idx];
    Real omega = constants::TWOPI * freq;

    int nf = observed.num_frequencies();
    LOG0_DEBUG("Frequency " + std::to_string(freq_idx + 1) + "/" +
              std::to_string(nf) + ": " + std::to_string(freq) + " Hz");

    find_stations_(observed);

    // Assemble and factorize (once per frequency)
    LOG0_INFO("[PROGRESS] phase=Factorize freq=" + std::to_string(freq_idx + 1) +
              "/" + std::to_string(nf));
    assemble_and_factorize_(omega);

    // Solve polarization 1 (X-directed)
    // NOTE: Frequency seeding (using previous freq's solution as initial guess)
    // was tested and found COUNTERPRODUCTIVE. PRESB+PCG(AMS) converges in ~2
    // iterations from zero, but seeded start takes 10-15+ iterations because
    // the seed's residual structure misaligns with the preconditioner's spectrum.
    LOG0_INFO("[PROGRESS] phase=Solve_Pol1 freq=" + std::to_string(freq_idx + 1) +
              "/" + std::to_string(nf));
    {
        mfem::ParGridFunction E0_real(fespace_.get()), E0_imag(fespace_.get());
        compute_primary_field_(omega, 0, E0_real, E0_imag);
        solve_polarization_(E0_real, E0_imag, *E1_real_, *E1_imag_);
    }

    // Solve polarization 2 (Y-directed)
    LOG0_INFO("[PROGRESS] phase=Solve_Pol2 freq=" + std::to_string(freq_idx + 1) +
              "/" + std::to_string(nf));
    {
        mfem::ParGridFunction E0_real(fespace_.get()), E0_imag(fespace_.get());
        compute_primary_field_(omega, 1, E0_real, E0_imag);
        solve_polarization_(E0_real, E0_imag, *E2_real_, *E2_imag_);
    }

    // Extract impedance at stations
    extract_impedance_(freq_idx, observed, predicted);

    LOG0_INFO("[PROGRESS] phase=Forward freq=" + std::to_string(freq_idx + 1) +
              "/" + std::to_string(nf) + " done");
    // NOTE: factorization retained for adjoint use. Call release_factorization()
    // externally when done with this frequency.
}

// =========================================================================
// factorize_frequency() — factorize only, no forward solve
// =========================================================================
void ForwardSolver3D::factorize_frequency(Real freq_hz) {
    Real omega = constants::TWOPI * freq_hz;
    LOG0_DEBUG("factorize_frequency: " + std::to_string(freq_hz) + " Hz");
    assemble_and_factorize_(omega);
}

// =========================================================================
// set_background_fields() — restore cached E fields for J·v computation
// =========================================================================
void ForwardSolver3D::set_background_fields(
    const mfem::ParGridFunction& E1_r,
    const mfem::ParGridFunction& E1_i,
    const mfem::ParGridFunction& E2_r,
    const mfem::ParGridFunction& E2_i)
{
    *E1_real_ = E1_r;
    *E1_imag_ = E1_i;
    *E2_real_ = E2_r;
    *E2_imag_ = E2_i;
}

// =========================================================================
// extract_impedance_()
// =========================================================================
void ForwardSolver3D::extract_impedance_(
    int freq_idx,
    const data::MTData& observed,
    data::MTData& predicted)
{
    Real omega = current_omega_;
    int ns = observed.num_stations();

    // Each process computes Z for stations in its local partition.
    // Stations not in local partition have elem_id < 0 → Z = 0.
    // MPI_Allreduce sums across all processes (only one process owns each station).
    // Pack: 8 doubles per station (Zxx_re, Zxx_im, Zxy_re, Zxy_im,
    //                               Zyx_re, Zyx_im, Zyy_re, Zyy_im)
    std::vector<double> Z_local(ns * 8, 0.0);

    // Also pack H-cache data for adjoint: 10 doubles per station
    // (Hx1_re, Hx1_im, Hy1_re, Hy1_im, Hx2_re, Hx2_im, Hy2_re, Hy2_im,
    //  detH_re, detH_im)
    std::vector<double> H_local(ns * 10, 0.0);

    for (int s = 0; s < ns; ++s) {
        if (station_elem_ids_[s] < 0) continue;

        int elem = station_elem_ids_[s];
        const mfem::IntegrationPoint& ip = station_ips_[s];

        mfem::ElementTransformation* T = mesh_->GetElementTransformation(elem);
        T->SetIntPoint(&ip);

        mfem::Vector E1r(3), E1i(3), E2r(3), E2i(3);
        E1_real_->GetVectorValue(*T, ip, E1r);
        E1_imag_->GetVectorValue(*T, ip, E1i);
        E2_real_->GetVectorValue(*T, ip, E2r);
        E2_imag_->GetVectorValue(*T, ip, E2i);

        mfem::Vector curlE1r(3), curlE1i(3), curlE2r(3), curlE2i(3);
        E1_real_->GetCurl(*T, curlE1r);
        E1_imag_->GetCurl(*T, curlE1i);
        E2_real_->GetCurl(*T, curlE2r);
        E2_imag_->GetCurl(*T, curlE2i);

        // H = curl(E) / (iωμ₀) = (curl_i - i·curl_r) / (ωμ₀)
        Real inv_wmu = 1.0 / (omega * constants::MU0);

        Complex Hx1(curlE1i(0) * inv_wmu, -curlE1r(0) * inv_wmu);
        Complex Hy1(curlE1i(1) * inv_wmu, -curlE1r(1) * inv_wmu);
        Complex Hx2(curlE2i(0) * inv_wmu, -curlE2r(0) * inv_wmu);
        Complex Hy2(curlE2i(1) * inv_wmu, -curlE2r(1) * inv_wmu);

        Complex Ex1(E1r(0), E1i(0));
        Complex Ey1(E1r(1), E1i(1));
        Complex Ex2(E2r(0), E2i(0));
        Complex Ey2(E2r(1), E2i(1));

        Complex det_H = Hx1 * Hy2 - Hx2 * Hy1;

        if (std::abs(det_H) < 1e-30) {
            LOG0_WARN("Singular H matrix at station " +
                           observed.station(s).name);
            continue;
        }

        Complex inv_det = 1.0 / det_H;

        Complex Zxx = (Ex1 * Hy2 - Ex2 * Hy1) * inv_det;
        Complex Zxy = (Ex2 * Hx1 - Ex1 * Hx2) * inv_det;
        Complex Zyx = (Ey1 * Hy2 - Ey2 * Hy1) * inv_det;
        Complex Zyy = (Ey2 * Hx1 - Ey1 * Hx2) * inv_det;

        int off = s * 8;
        Z_local[off + 0] = Zxx.real();  Z_local[off + 1] = Zxx.imag();
        Z_local[off + 2] = Zxy.real();  Z_local[off + 3] = Zxy.imag();
        Z_local[off + 4] = Zyx.real();  Z_local[off + 5] = Zyx.imag();
        Z_local[off + 6] = Zyy.real();  Z_local[off + 7] = Zyy.imag();

        // Pack H-cache for adjoint
        int hoff = s * 10;
        H_local[hoff + 0] = Hx1.real(); H_local[hoff + 1] = Hx1.imag();
        H_local[hoff + 2] = Hy1.real(); H_local[hoff + 3] = Hy1.imag();
        H_local[hoff + 4] = Hx2.real(); H_local[hoff + 5] = Hx2.imag();
        H_local[hoff + 6] = Hy2.real(); H_local[hoff + 7] = Hy2.imag();
        H_local[hoff + 8] = det_H.real(); H_local[hoff + 9] = det_H.imag();
    }

    // Allreduce: sum across processes (only one process has non-zero for each station)
    std::vector<double> Z_global(ns * 8, 0.0);
    std::vector<double> H_global(ns * 10, 0.0);
#ifdef MAPLE3DMT_USE_MPI
    MPI_Allreduce(Z_local.data(), Z_global.data(), ns * 8,
                  MPI_DOUBLE, MPI_SUM, mesh_->GetComm());
    MPI_Allreduce(H_local.data(), H_global.data(), ns * 10,
                  MPI_DOUBLE, MPI_SUM, mesh_->GetComm());
#else
    Z_global = Z_local;
    H_global = H_local;
#endif

    // Unpack results on all processes
    for (int s = 0; s < ns; ++s) {
        int off = s * 8;
        Complex Zxx(Z_global[off + 0], Z_global[off + 1]);
        Complex Zxy(Z_global[off + 2], Z_global[off + 3]);
        Complex Zyx(Z_global[off + 4], Z_global[off + 5]);
        Complex Zyy(Z_global[off + 6], Z_global[off + 7]);

        data::MTResponse resp;
        resp.Zxx.value = Zxx;
        resp.Zxy.value = Zxy;
        resp.Zyx.value = Zyx;
        resp.Zyy.value = Zyy;
        predicted.set_predicted(s, freq_idx, resp);

        // Restore H-cache on all processes (needed for adjoint)
        int hoff = s * 10;
        station_H_cache_[s] = {
            Complex(H_global[hoff + 0], H_global[hoff + 1]),
            Complex(H_global[hoff + 2], H_global[hoff + 3]),
            Complex(H_global[hoff + 4], H_global[hoff + 5]),
            Complex(H_global[hoff + 6], H_global[hoff + 7]),
            Complex(H_global[hoff + 8], H_global[hoff + 9])
        };
    }
}

// =========================================================================
// compute_responses()
// =========================================================================
void ForwardSolver3D::compute_responses(
    const data::MTData& observed,
    data::MTData& predicted)
{
    int nf = observed.num_frequencies();
    LOG0_INFO("Computing responses for " + std::to_string(nf) +
                   " frequencies, " + std::to_string(observed.num_stations()) +
                   " stations");

    mem_prof_.snap("before_freq_loop");

    for (int f = 0; f < nf; ++f) {
        compute_single_frequency(f, observed, predicted);
        mem_prof_.snap("after_solve_f" + std::to_string(f));
        release_factorization();
        mem_prof_.snap("after_release_f" + std::to_string(f));
        // Progress callback AFTER solve+release so it's not buried by FGMRES output
        if (freq_progress_cb_) {
            freq_progress_cb_(f, nf, observed.frequencies()[f], "Forward");
        }
    }

    mem_prof_.snap("after_all_freq");
}

// =========================================================================
// adjoint_solve()
// =========================================================================
void ForwardSolver3D::adjoint_solve(
    const mfem::Vector& rhs,
    mfem::ParGridFunction& adj_real,
    mfem::ParGridFunction& adj_imag,
    const mfem::ParGridFunction* seed_real,
    const mfem::ParGridFunction* seed_imag)
{
    if (!system_ready_ || (!solver_ && !complex_solver_)) {
        throw std::runtime_error("adjoint_solve: no active factorization. "
                                "Call compute_single_frequency() first.");
    }

    // ── Parallel assembly: L-vector → T-vector ──
    // build_adjoint_rhs builds the RHS in L-vector (local DOF) space.
    // Convert to T-vector (true DOF) space using P^T (prolongation transpose).
    // P^T accumulates shared-DOF contributions across MPI ranks, ensuring
    // ALL stations' sources are included in the global adjoint RHS.
    int tdof = fespace_->GetTrueVSize();
    int lvsize = fespace_->GetVSize();
    mfem::Vector rhs_true(2 * tdof);

    auto* P = fespace_->GetProlongationMatrix();
    if (P && rhs.Size() == 2 * lvsize) {
        // L-vector input: apply P^T per block (real, imag)
        mfem::Vector rhs_L_r(const_cast<double*>(rhs.GetData()), lvsize);
        mfem::Vector rhs_L_i(const_cast<double*>(rhs.GetData()) + lvsize, lvsize);
        mfem::Vector rhs_T_r(rhs_true.GetData(), tdof);
        mfem::Vector rhs_T_i(rhs_true.GetData() + tdof, tdof);
        P->MultTranspose(rhs_L_r, rhs_T_r);
        P->MultTranspose(rhs_L_i, rhs_T_i);
    } else {
        // Already T-vector (legacy callers or serial mode)
        rhs_true = rhs;
    }

    // Note: The P^T conversion above correctly places each rank's station
    // sources at its own true DOFs. The distributed iterative solver (FGMRES)
    // combines contributions from all ranks through the parallel matrix-vector
    // products. MUMPS centralized mode does NOT combine distributed RHS, so
    // we skip the MUMPS adjoint path and use the iterative solver instead.

    // Solve A^T x = b via R-transform: A^T = R·A·R where R = diag(I,-I)
    // (negates second block). This avoids relying on MultTranspose support.
    mfem::Vector rhs_R(2 * tdof), sol_R(2 * tdof);

    // Check for NaN/Inf in RHS
    {
        Real local_norm2 = rhs_true.Norml2();
        local_norm2 *= local_norm2;
        Real global_norm2 = local_norm2;
#ifdef MAPLE3DMT_USE_MPI
        MPI_Allreduce(&local_norm2, &global_norm2, 1, MPI_DOUBLE, MPI_SUM,
                       mesh_->GetComm());
#endif
        Real rhs_norm = std::sqrt(global_norm2);
        if (!std::isfinite(rhs_norm) || rhs_norm < 1e-30) {
            LOG0_DEBUG("  adjoint_solve: skipping (global RHS norm=" +
                      std::to_string(rhs_norm) + ", NaN/zero)");
            adj_real = 0.0;
            adj_imag = 0.0;
            return;  // safe: all ranks agree via Allreduce
        }
    }

    // ── Complex N×N adjoint path (BiCGStab + ILU(0), A^H) ──
    if (using_complex_ && complex_solver_ && complex_solver_->ready()) {
        mfem::Vector rhs_real(rhs_true.GetData(), tdof);
        mfem::Vector rhs_imag(rhs_true.GetData() + tdof, tdof);
        mfem::Vector sol_real(tdof), sol_imag(tdof);

        // Seed from previous polarization
        if (seed_real && seed_imag) {
            seed_real->GetTrueDofs(sol_real);
            seed_imag->GetTrueDofs(sol_imag);
        } else {
            sol_real = 0.0;
            sol_imag = 0.0;
        }

        auto result = complex_solver_->solve_adjoint(
            rhs_real, rhs_imag, sol_real, sol_imag);

        LOG0_INFO("  adjoint(complex): " +
                 std::string(result.converged ? "converged" : "NOT converged") +
                 " iter=" + std::to_string(result.iterations) +
                 " rel_res=" + std::to_string(result.final_residual));

        adj_real.SetFromTrueDofs(sol_real);
        adj_imag.SetFromTrueDofs(sol_imag);
        return;
    }

    // ── Strategy: Use MUMPS direct transpose solve for adjoint ──
    // Forward uses iterative (FGMRES+AMS) which works great for smooth RHS.
    // MUMPS adjoint path: factorize A and solve A^T x = b via ICNTL(9)=2.
    // MUMPSSolverExt::MultTranspose uses MPI_Gatherv to correctly assemble
    // the distributed adjoint RHS from all ranks before solving on rank 0.
#ifdef MFEM_USE_MUMPS
    if (using_iterative_ && params_.adjoint_direct && system_matrix_) {
        try {
            // Create MUMPS solver on first adjoint call for this frequency
            if (!adjoint_mumps_) {
                LOG0_DEBUG("  adjoint_solve: factorizing MUMPS for adjoint (cached)");
                MUMPSConfig cfg;
                cfg.blr_enabled   = true;
                cfg.blr_tolerance = params_.blr_tolerance;
                cfg.ooc_enabled   = params_.ooc_enabled;
                cfg.ooc_tmpdir    = params_.ooc_tmpdir;
                cfg.max_mem_mb    = params_.max_mem_mb;
                cfg.mem_relax_pct = params_.mem_relax_pct;
                cfg.unsymmetric   = true;   // block-real system is unsymmetric!
                cfg.print_level   = 0;

                adjoint_mumps_ = std::make_unique<MUMPSSolverExt>(mesh_->GetComm(), cfg);
                adjoint_mumps_->SetOperator(*system_matrix_);  // analyze + factorize
            }

            // Transpose solve reusing cached factorization
            adjoint_mumps_->MultTranspose(rhs_true, sol_R);
            LOG0_DEBUG("  adjoint_solve: MUMPS transpose solve completed");

            // Extract solution — no R-transform needed for direct A^T solve
            mfem::Vector sol_real_true(sol_R.GetData(), tdof);
            adj_real.SetFromTrueDofs(sol_real_true);
            mfem::Vector sol_imag_true(sol_R.GetData() + tdof, tdof);
            adj_imag.SetFromTrueDofs(sol_imag_true);
            return;
        } catch (const std::exception& e) {
            LOG0_WARN("  adjoint_solve: MUMPS failed (" +
                      std::string(e.what()) + "), falling back to iterative");
            adjoint_mumps_.reset();
            // Fall through to iterative solver below
        }
    }
#endif

    // ── R-transform adjoint: solve A^T x = b via forward solver ──
    // For HERMITIAN block system A = [K, -ωσM; ωσM, K]:
    //   A^T = [K, ωσM; -ωσM, K] = R·A·R  where R = diag(I, -I)
    // Therefore:  A^T x = b  ⟺  A (Rx) = Rb
    // Strategy: transform RHS, solve with forward solver, transform solution.
    // This reuses the forward FGMRES+PRESB(AMS) which converges in 2-3 iter.

    // Step 1: R-transform RHS — flip sign of imaginary block
    for (int i = 0; i < tdof; ++i) {
        rhs_R(i)        =  rhs_true(i);          // real: unchanged
        rhs_R(i + tdof) = -rhs_true(i + tdof);   // imag: sign flip (R·b)
    }

    // Compute global RHS norm for diagnostic (use original b, not R·b)
    Real rhs_R_norm = 0.0;
    {
        Real local_n2 = rhs_true.Norml2();
        local_n2 *= local_n2;
        Real global_n2 = local_n2;
#ifdef MAPLE3DMT_USE_MPI
        MPI_Allreduce(&local_n2, &global_n2, 1, MPI_DOUBLE, MPI_SUM,
                       mesh_->GetComm());
#endif
        rhs_R_norm = std::sqrt(global_n2);
    }

    LOG0_INFO("  adjoint_solve: R-transform + forward solver, ||rhs||=" +
              std::to_string(rhs_R_norm));

    // Step 2: Solve A·y = R·b using forward solver (FGMRES + PRESB/AMS)
    sol_R = 0.0;
    if (solver_) {
        solver_->Mult(rhs_R, sol_R);

        // Log convergence info from forward solver
        if (auto* fgmres = dynamic_cast<mfem::FGMRESSolver*>(solver_.get())) {
            int    n_iter    = fgmres->GetNumIterations();
            bool   converged = fgmres->GetConverged();
            Real   final_res = fgmres->GetFinalNorm();
            Real   sol_norm  = sol_R.Norml2();
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "  adjoint_solve(R-transform): %s iter=%d "
                "final=%.3e ||rhs||=%.3e ||sol||=%.3e",
                converged ? "converged" : "NOT converged",
                n_iter, final_res, rhs_R_norm, sol_norm);
            LOG0_INFO(std::string(buf));
        }
    } else {
        LOG0_WARN("  adjoint_solve: no forward solver available!");
    }

    // Step 3: R-transform solution — flip sign of imaginary block
    // x = R·y where y is the forward solve result
    // sol_R real part stays, imag part gets sign-flipped
    for (int i = 0; i < tdof; ++i) {
        // sol_R(i) unchanged (real part)
        sol_R(i + tdof) = -sol_R(i + tdof);   // imag: sign flip (R·y)
    }

    // Map from true DOFs to local DOFs
    mfem::Vector sol_real_true(sol_R.GetData(), tdof);
    adj_real.SetFromTrueDofs(sol_real_true);

    mfem::Vector sol_imag_true(sol_R.GetData() + tdof, tdof);
    adj_imag.SetFromTrueDofs(sol_imag_true);
}

// =========================================================================
// solve_forward_rhs()
// =========================================================================
void ForwardSolver3D::solve_forward_rhs(
    const mfem::Vector& rhs,
    mfem::ParGridFunction& sol_real,
    mfem::ParGridFunction& sol_imag)
{
    if (!system_ready_ || (!solver_ && !complex_solver_)) {
        throw std::runtime_error("solve_forward_rhs: no active factorization.");
    }

    int tdof = fespace_->GetTrueVSize();
    mfem::Vector X(2 * tdof);
    X = 0.0;

    solver_->Mult(rhs, X);

    mfem::Vector X_real(X.GetData(), tdof);
    sol_real.SetFromTrueDofs(X_real);
    mfem::Vector X_imag(X.GetData() + tdof, tdof);
    sol_imag.SetFromTrueDofs(X_imag);
}

// =========================================================================
// release_factorization()
// =========================================================================
void ForwardSolver3D::release_factorization() {
    mem_prof_.snap("before_release");
    // Release adjoint solvers (must be before system_matrix_ and precond_ reset)
    adjoint_fgmres_.reset();     // references adjoint_jacobi_/adjoint_div_prec_ and system_matrix_
    adjoint_jacobi_.reset();     // Jacobi preconditioner (ModEM-style adjoint)
    adjoint_div_prec_.reset();   // DivCorr helper (references grad_mat_)
#ifdef MFEM_USE_MUMPS
    adjoint_mumps_.reset();
#endif
    // Release complex solver (if active)
    if (complex_solver_) {
        complex_solver_->release();
        complex_solver_.reset();
    }
    using_complex_ = false;
    // Release solver first (FGMRES references ccgd_op_ / preconditioner / matrix)
    solver_.reset();
    ccgd_op_.reset();          // CCGDOperator references system_matrix_ and ccgd_GGt_
    // Release preconditioner components (order matters!)
    block_prec_.reset();       // BlockDiagAMS_PRESB (must be before spd_nxn_ / beta_poisson_ / prec_matrix_)
    ams_prec_.reset();
    spd_nxn_.reset();          // N×N SPD block (K+B) for BlockDiagAMS_PRESB
    beta_poisson_.reset();     // H1 beta Poisson (kept alive for HYPRE AMS reference)
    prec_matrix_.reset();      // 2N×2N PRESB preconditioner matrix
    abs_mass_coeff_.reset();   // coefficient for preconditioner
    abs_omega_sigma_vec_.Destroy();
    precond_.reset();          // legacy preconditioner (if any)
    // CCGD no longer used (adjoint uses R-transform with forward solver).
    // Release ccgd_GGt_ to free ~80 MB/rank.
    ccgd_GGt_.reset();
    ccgd_tau_actual_ = 0.0;
    ccgd_op_T_.reset();            // A^T + τ·GGt wrapper (unused)
    system_matrix_T_.reset();      // explicit A^T (unused, R-transform instead)
    mem_prof_.snap("after_solver_reset");
    system_matrix_.reset();
    elim_real_.reset();            // eliminated part for manual BC handling
    elim_imag_.reset();
    mem_prof_.snap("after_sysmat_reset");
    sesq_form_.reset();
    // Release coefficients (must be freed AFTER sesq_form_ which references them)
    inv_mu0_coeff_.reset();
    mass_coeff_.reset();
    neg_omega_sigma_vec_.Destroy();
    mem_prof_.snap("after_full_release");
    system_ready_ = false;
    using_iterative_ = false;
}

// =========================================================================
// build_adjoint_rhs()
// =========================================================================
void ForwardSolver3D::build_adjoint_rhs(
    int freq_idx,
    const data::MTData& observed,
    const data::MTData& predicted,
    const RealVec& data_weights,
    mfem::Vector& adj_rhs_pol1,
    mfem::Vector& adj_rhs_pol2)
{
    int ns = observed.num_stations();
    std::vector<std::array<Complex,4>> weighted_residual(ns);

    for (int s = 0; s < ns; ++s) {
        const auto& obs = observed.observed(s, freq_idx);
        const auto& pred = predicted.predicted(s, freq_idx);

        Complex r_xx = obs.Zxx.value - pred.Zxx.value;
        Complex r_xy = obs.Zxy.value - pred.Zxy.value;
        Complex r_yx = obs.Zyx.value - pred.Zyx.value;
        Complex r_yy = obs.Zyy.value - pred.Zyy.value;

        // data_weights layout: ns * 8 values, stride 8 per station
        // [Re(Zxx), Im(Zxx), Re(Zxy), Im(Zxy), Re(Zyx), Im(Zyx), Re(Zyy), Im(Zyy)]
        // Re/Im weights are identical (1/error), so we take every other one.
        int base = s * 8;
        Real w_xx = (base < static_cast<int>(data_weights.size()))
                        ? data_weights[base] : 1.0;
        Real w_xy = (base+2 < static_cast<int>(data_weights.size()))
                        ? data_weights[base+2] : 1.0;
        Real w_yx = (base+4 < static_cast<int>(data_weights.size()))
                        ? data_weights[base+4] : 1.0;
        Real w_yy = (base+6 < static_cast<int>(data_weights.size()))
                        ? data_weights[base+6] : 1.0;

        weighted_residual[s] = {
            w_xx * w_xx * r_xx,
            w_xy * w_xy * r_xy,
            w_yx * w_yx * r_yx,
            w_yy * w_yy * r_yy
        };
    }

    build_adjoint_rhs_from_residual(freq_idx, weighted_residual,
                                    adj_rhs_pol1, adj_rhs_pol2);
}

// =========================================================================
// build_adjoint_rhs_from_residual()
// =========================================================================
void ForwardSolver3D::build_adjoint_rhs_from_residual(
    int freq_idx,
    const std::vector<std::array<Complex,4>>& weighted_residual,
    mfem::Vector& adj_rhs_pol1,
    mfem::Vector& adj_rhs_pol2)
{
    // Build in L-vector (local DOF) space, NOT T-vector (true DOF) space.
    // GetElementVDofs returns local DOF indices (0..GetVSize()-1).
    // The caller must apply P^T to convert to T-vector before the solve,
    // which accumulates shared-DOF contributions from all MPI ranks.
    int lvsize = fespace_->GetVSize();
    adj_rhs_pol1.SetSize(2 * lvsize);
    adj_rhs_pol2.SetSize(2 * lvsize);
    adj_rhs_pol1 = 0.0;
    adj_rhs_pol2 = 0.0;

    int ns = static_cast<int>(weighted_residual.size());

    int n_skipped = 0;
    for (int s = 0; s < ns; ++s) {
        if (station_elem_ids_[s] < 0) continue;

        const auto& cache = station_H_cache_[s];

        // Guard: skip station if det(H) is near-singular (|det| < threshold).
        // This happens when both polarizations produce nearly parallel H-fields,
        // making impedance undefined.  The adjoint source for this station is
        // effectively zero (no useful gradient information).
        Real det_abs = std::abs(cache.det_H);
        if (det_abs < 1e-30 || !std::isfinite(det_abs)) {
            ++n_skipped;
            continue;
        }
        Complex inv_det = 1.0 / cache.det_H;

        // Q^T maps impedance residuals to E-field adjoint sources.
        // For polarization 1 (E1): d(Zxy)/dE1_x = -Hx2/det, d(Zyx)/dE1_y = Hy2/det
        // For polarization 2 (E2): d(Zxy)/dE2_x = Hx1/det, d(Zyx)/dE2_y = -Hy1/det

        Complex wr_xx = weighted_residual[s][0];
        Complex wr_xy = weighted_residual[s][1];
        Complex wr_yx = weighted_residual[s][2];
        Complex wr_yy = weighted_residual[s][3];

        // Adjoint source for pol 1: conj(dZ/dE1)^T * wr
        Complex src1_x = std::conj(cache.Hy2 * inv_det) * wr_xx
                       + std::conj(-cache.Hx2 * inv_det) * wr_xy;
        Complex src1_y = std::conj(cache.Hy2 * inv_det) * wr_yx
                       + std::conj(-cache.Hx2 * inv_det) * wr_yy;

        Complex src2_x = std::conj(-cache.Hy1 * inv_det) * wr_xx
                       + std::conj(cache.Hx1 * inv_det) * wr_xy;
        Complex src2_y = std::conj(-cache.Hy1 * inv_det) * wr_yx
                       + std::conj(cache.Hx1 * inv_det) * wr_yy;

        // Point-source injection at station location (delta function).
        // For Nédélec elements: use CalcVShape to get the vector-valued
        // basis functions N_k(x) at the station point.  The adjoint source
        // contribution for DOF k is:  rhs_k += N_k(x_s) · s
        // where s = [src_x, src_y, 0] is the adjoint source vector.
        int elem = station_elem_ids_[s];
        const mfem::IntegrationPoint& ip = station_ips_[s];
        mfem::ElementTransformation* T = mesh_->GetElementTransformation(elem);
        T->SetIntPoint(&ip);

        mfem::Array<int> vdofs;
        fespace_->GetElementVDofs(elem, vdofs);
        int ndof = vdofs.Size();

        const mfem::FiniteElement* fe = fespace_->GetFE(elem);
        mfem::DenseMatrix vshape(ndof, 3);
        fe->CalcVShape(*T, vshape);

        for (int k = 0; k < ndof; ++k) {
            int dof = vdofs[k];
            int sign = 1;
            if (dof < 0) { dof = -1 - dof; sign = -1; }

            // N_k · s = N_k_x * src_x + N_k_y * src_y  (z-component = 0)
            Real Nk_x = vshape(k, 0);
            Real Nk_y = vshape(k, 1);

            Complex dot1 = Nk_x * src1_x + Nk_y * src1_y;
            Complex dot2 = Nk_x * src2_x + Nk_y * src2_y;

            Real s = static_cast<Real>(sign);

            // Pol 1 — L-vector layout: [real(0..lvsize-1); imag(0..lvsize-1)]
            adj_rhs_pol1(dof)          += s * dot1.real();
            adj_rhs_pol1(dof + lvsize) += s * dot1.imag();

            // Pol 2
            adj_rhs_pol2(dof)          += s * dot2.real();
            adj_rhs_pol2(dof + lvsize) += s * dot2.imag();
        }
    }

    if (n_skipped > 0) {
        LOG0_DEBUG("  adjoint_rhs: skipped " + std::to_string(n_skipped) +
                  "/" + std::to_string(ns) +
                  " stations (singular det_H at freq " +
                  std::to_string(freq_idx) + ")");
    }
}

// =========================================================================
// extract_delta_impedance()
// =========================================================================
void ForwardSolver3D::extract_delta_impedance(
    const mfem::ParGridFunction& dE_real,
    const mfem::ParGridFunction& dE_imag,
    int polarization,
    std::vector<std::array<Complex,4>>& delta_Z)
{
    int ns = static_cast<int>(station_H_cache_.size());

    // Local computation: each rank evaluates dE at stations in its partition
    // (station_elem_ids_[s] < 0 means station not in this rank's partition)
    std::vector<double> dZ_local(ns * 8, 0.0);  // Re/Im for 4 components

    for (int s = 0; s < ns; ++s) {
        if (station_elem_ids_[s] < 0) continue;

        int elem = station_elem_ids_[s];
        const mfem::IntegrationPoint& ip = station_ips_[s];

        mfem::ElementTransformation* T = mesh_->GetElementTransformation(elem);
        T->SetIntPoint(&ip);

        mfem::Vector dEr(3), dEi(3);
        dE_real.GetVectorValue(*T, ip, dEr);
        dE_imag.GetVectorValue(*T, ip, dEi);

        Complex dEx(dEr(0), dEi(0));
        Complex dEy(dEr(1), dEi(1));

        const auto& cache = station_H_cache_[s];

        // Guard: skip station with singular det_H
        Real det_abs = std::abs(cache.det_H);
        if (det_abs < 1e-30 || !std::isfinite(det_abs)) {
            int off = s * 8;
            for (int c = 0; c < 8; ++c) dZ_local[off + c] = 0.0;
            continue;
        }
        Complex inv_det = 1.0 / cache.det_H;

        std::array<Complex,4> local_dZ;
        if (polarization == 0) {
            local_dZ[0] = dEx * cache.Hy2 * inv_det;
            local_dZ[1] = -dEx * cache.Hx2 * inv_det;
            local_dZ[2] = dEy * cache.Hy2 * inv_det;
            local_dZ[3] = -dEy * cache.Hx2 * inv_det;
        } else {
            local_dZ[0] = -dEx * cache.Hy1 * inv_det;
            local_dZ[1] = dEx * cache.Hx1 * inv_det;
            local_dZ[2] = -dEy * cache.Hy1 * inv_det;
            local_dZ[3] = dEy * cache.Hx1 * inv_det;
        }

        int off = s * 8;
        for (int c = 0; c < 4; ++c) {
            dZ_local[off + 2*c]     = local_dZ[c].real();
            dZ_local[off + 2*c + 1] = local_dZ[c].imag();
        }
    }

    // Allreduce: each station is owned by exactly one rank; sum collects all.
    std::vector<double> dZ_global(ns * 8, 0.0);
#ifdef MAPLE3DMT_USE_MPI
    MPI_Allreduce(dZ_local.data(), dZ_global.data(), ns * 8,
                  MPI_DOUBLE, MPI_SUM, mesh_->GetComm());
#else
    dZ_global = dZ_local;
#endif

    // Unpack into complex delta_Z
    delta_Z.resize(ns);
    for (int s = 0; s < ns; ++s) {
        int off = s * 8;
        for (int c = 0; c < 4; ++c) {
            delta_Z[s][c] = Complex(dZ_global[off + 2*c],
                                    dZ_global[off + 2*c + 1]);
        }
    }
}

// =========================================================================
// build_perturbation_rhs()
// =========================================================================
void ForwardSolver3D::build_perturbation_rhs(
    int polarization,
    const RealVec& delta_sigma,
    mfem::Vector& pert_rhs)
{
    int tdof = fespace_->GetTrueVSize();
    int vsize = fespace_->GetVSize();
    pert_rhs.SetSize(2 * tdof);
    pert_rhs = 0.0;

    auto* E_real = (polarization == 0) ? E1_real_.get() : E2_real_.get();
    auto* E_imag = (polarization == 0) ? E1_imag_.get() : E2_imag_.get();

    Real omega = current_omega_;
    int ne = mesh_->GetNE();

    // Perturbation equation: A·δE = iωδσ M E
    //   = iωδσ(E_r + iE_i) = iωδσ E_r - ωδσ E_i
    //   Real part: -ωδσ E_imag
    //   Imag part: +ωδσ E_real
    //
    // For each element e with coeff c_e = ωδσ_e:
    //   rhs_r += -c_e * M_e * E_i_e  (= -ωδσ E_i)
    //   rhs_i +=  c_e * M_e * E_r_e  (= +ωδσ E_r)

    mfem::Vector rhs_local_r(vsize), rhs_local_i(vsize);
    rhs_local_r = 0.0;
    rhs_local_i = 0.0;

    // Element-level mass matrix computation (no global assembly needed)
    for (int e = 0; e < ne; ++e) {
        int attr = mesh_->GetAttribute(e);
        Real ds = (attr == 2) ? 0.0 : delta_sigma[e];
        if (std::abs(ds) < 1e-30) continue;  // skip zero elements

        Real coeff = omega * ds;  // = +ωδσ_e

        // Get element DOFs
        mfem::Array<int> vdofs;
        fespace_->GetElementVDofs(e, vdofs);
        int ndof = vdofs.Size();

        // Get element transformation
        mfem::ElementTransformation* trans = mesh_->GetElementTransformation(e);
        const mfem::FiniteElement* fe = fespace_->GetFE(e);

        // Integration rule
        int order = 2 * fe->GetOrder() + trans->OrderW();
        const mfem::IntegrationRule& ir =
            mfem::IntRules.Get(fe->GetGeomType(), order);

        // Extract local E-field vectors
        mfem::Vector E_r_local(ndof), E_i_local(ndof);
        E_real->GetSubVector(vdofs, E_r_local);
        E_imag->GetSubVector(vdofs, E_i_local);

        // Compute element mass action: M_e * E_local
        mfem::Vector M_Er(ndof), M_Ei(ndof);
        M_Er = 0.0;
        M_Ei = 0.0;

        mfem::DenseMatrix vshape;
        for (int q = 0; q < ir.GetNPoints(); ++q) {
            const mfem::IntegrationPoint& ip = ir.IntPoint(q);
            trans->SetIntPoint(&ip);

            vshape.SetSize(ndof, 3);
            fe->CalcVShape(*trans, vshape);

            Real w = ip.weight * trans->Weight();

            // N^T (N · E) = M_e · E  at this quadrature point
            // Compute N · E_r and N · E_i (3-component dot products)
            for (int d = 0; d < 3; ++d) {
                Real NdotEr = 0.0, NdotEi = 0.0;
                for (int j = 0; j < ndof; ++j) {
                    NdotEr += vshape(j, d) * E_r_local(j);
                    NdotEi += vshape(j, d) * E_i_local(j);
                }
                for (int j = 0; j < ndof; ++j) {
                    M_Er(j) += w * vshape(j, d) * NdotEr;
                    M_Ei(j) += w * vshape(j, d) * NdotEi;
                }
            }
        }

        // Accumulate: rhs_r += -coeff * M_Ei, rhs_i += coeff * M_Er
        for (int j = 0; j < ndof; ++j) {
            int dof = vdofs[j];
            if (dof >= 0) {
                rhs_local_r(dof) += -coeff * M_Ei(j);
                rhs_local_i(dof) +=  coeff * M_Er(j);
            } else {
                // Negative DOF index: flip sign (MFEM convention)
                rhs_local_r(-1 - dof) -= -coeff * M_Ei(j);
                rhs_local_i(-1 - dof) -=  coeff * M_Er(j);
            }
        }
    }

    // Map local DOF → true DOF
    mfem::Vector rhs_r_true(tdof), rhs_i_true(tdof);
    fespace_->GetRestrictionMatrix()->Mult(rhs_local_r, rhs_r_true);
    fespace_->GetRestrictionMatrix()->Mult(rhs_local_i, rhs_i_true);

    for (int i = 0; i < tdof; ++i) {
        pert_rhs(i)        = rhs_r_true(i);
        pert_rhs(i + tdof) = rhs_i_true(i);
    }
}

// =========================================================================
// compute_sensitivity()
// =========================================================================
void ForwardSolver3D::compute_sensitivity(
    const mfem::ParGridFunction& E_real,
    const mfem::ParGridFunction& E_imag,
    const mfem::ParGridFunction& lambda_real,
    const mfem::ParGridFunction& lambda_imag,
    RealVec& sensitivity)
{
    int ne = mesh_->GetNE();
    sensitivity.assign(ne, 0.0);

    Real omega = current_omega_;

    for (int e = 0; e < ne; ++e) {
        if (mesh_->GetAttribute(e) == 2) continue;  // skip air

        const mfem::FiniteElement* fe = fespace_->GetFE(e);
        mfem::ElementTransformation* T = mesh_->GetElementTransformation(e);

        const mfem::IntegrationRule& ir =
            mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + 2);

        Real g_e = 0.0;
        for (int q = 0; q < ir.GetNPoints(); ++q) {
            const mfem::IntegrationPoint& ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            Real w = ip.weight * T->Weight();

            mfem::Vector Er(3), Ei(3), Lr(3), Li(3);
            E_real.GetVectorValue(*T, ip, Er);
            E_imag.GetVectorValue(*T, ip, Ei);
            lambda_real.GetVectorValue(*T, ip, Lr);
            lambda_imag.GetVectorValue(*T, ip, Li);

            // Sensitivity: original formula Lr·Ei - Li·Er
            // This combined with grad -= g and d = -g gives descent direction
            // (verified empirically: RMS decreases with this combination).
            Real dot = 0.0;
            for (int d = 0; d < 3; ++d) {
                dot += Lr(d) * Ei(d) - Li(d) * Er(d);
            }

            Real sigma_e = model_->sigma(e);
            g_e -= omega * sigma_e * dot * w;
        }

        sensitivity[e] = g_e;
    }
}

} // namespace forward
} // namespace maple3dmt
