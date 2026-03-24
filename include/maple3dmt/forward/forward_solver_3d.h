// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file forward_solver_3d.h
/// @brief 3D MT forward solver using edge (Nédélec) finite elements.
///
/// Solves the vector Helmholtz equation for the electric field:
///   curl(μ⁻¹ curl E) - iωσ E = iωJ_s
///
/// Uses MUMPS (with BLR compression) or STRUMPACK for the sparse direct solve.
/// Each frequency is factorized independently to manage memory.

#include "maple3dmt/common.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/forward/mumps_solver_ext.h"
#include "maple3dmt/forward/complex_solver.h"
#include "maple3dmt/utils/memory_profiler.h"
#include <mfem.hpp>
#include <complex>
#include <memory>
#ifdef MAPLE3DMT_USE_MPI
#include <mpi.h>
#endif

namespace maple3dmt {
namespace forward {

/// Sparse direct solver backend selection.
enum class SolverBackend {
    MUMPS,          // MUMPS with optional BLR compression
    MUMPS_BLR,      // MUMPS with BLR explicitly enabled
    STRUMPACK_HSS,  // STRUMPACK with HSS compression
    STRUMPACK_BLR,  // STRUMPACK with BLR compression
    ITERATIVE,      // FGMRES + block-diagonal preconditioner (lowest memory)
    COMPLEX_BICGSTAB, // Complex N×N BiCGStab + ILU(0) — ModEM-equivalent (fastest)
    HYBRID,         // Auto: direct if memory allows, otherwise iterative
};

/// Preconditioner type for iterative solver.
enum class PrecondType {
    AMG,    // BoomerAMG on 2N×2N SPD (fallback; stagnates for edge elements)
    AMS,    // BlockDiagAMS_PRESB: MFEM HypreAMS on N×N SPD per block (optimal for edge elements)
    DIAG,   // Diagonal scaling (baseline, guaranteed to work)
};

/// Forward solver configuration.
struct ForwardParams3D {
    SolverBackend backend = SolverBackend::HYBRID;
    PrecondType   precond = PrecondType::AMS;  // AMS: optimal for edge elements (HYPRE 2.31)
    int    fe_order       = 1;        // Nédélec element order (1 or 2)
    Real   blr_tolerance  = 1e-2;    // BLR compression tolerance (1e-2: ~10× compression)
    bool   reuse_pattern  = true;     // reuse symbolic factorization across freqs
    int    gmres_maxiter  = 2000;     // for ITERATIVE/HYBRID iterative mode
    Real   gmres_tol      = 2e-3;    // 2e-3: field ~0.2% error, ≪ data error (~5%)
    int    gmres_kdim     = 10;      // Krylov dim: PRESB+PCG(AMS) → 2-3 iter, 10 sufficient (saves ~60 MB/rank)
    int    gmres_print    = -1;      // FGMRES print level: -1=auto (0 if progress bar, 1 otherwise), 0=silent, 1=per-iter

    // Adjoint solver strategy.
    // Point-source RHS is hard for iterative solvers (AMS), but MUMPS BLR+OOC
    // is unstable with repeated create/destroy. Default: iterative with relaxed tol.
    // Non-converged FGMRES solution is still a usable approximate gradient for NLCG.
    bool   adjoint_direct  = false;  // true=MUMPS direct (unstable), false=iterative

    // Iterative adjoint fallback settings (used when adjoint_direct=false or MUMPS unavailable)
    Real   adjoint_tol     = 0.1;    // 50× more relaxed than forward (0.002)
    int    adjoint_maxiter = 500;    // adjoint needs more iterations than forward
    int    adjoint_kdim    = 500;    // Krylov dim for adjoint (large = no restart, better convergence for point-source RHS)

    // ModEM-style adjoint solver: outer DivCorr loop + inner FGMRES with Jacobi.
    // Divergence correction is applied BETWEEN Krylov solver calls (not inside preconditioner).
    // This matches ModEM (Egbert & Kelbert, 2012) and works for point-source adjoint RHS.
    bool   adjoint_div_corr      = false;   // ModEM-style outer DivCorr loop (OFF: proven harmful with AMS+CCGD)
    bool   adjoint_use_jacobi    = true;    // true=Jacobi (fast), false=AMS (strong but slow)
    int    adjoint_inner_iter    = 50;     // Inner FGMRES iterations per outer step
    int    adjoint_outer_iter    = 20;     // Max outer DivCorr iterations
    Real   adjoint_div_tol       = 1e-6;    // Tolerance for scalar Laplacian solve
    int    adjoint_div_maxiter   = 200;     // Max iterations for Laplacian solve

    // CCGD: Curl-Curl + Grad-Div regularization (Dong & Egbert, 2019 GJI).
    // Adds scaled grad-div term τ·∇(∇·E) to the curl-curl equation, which
    // deflates the gradient null space of the curl-curl operator.  This
    // dramatically improves convergence of iterative solvers for BOTH
    // forward (smooth RHS) and adjoint (point-source RHS) problems.
    // The discrete form adds τ·G·G^T to the system matrix, where G is
    // the discrete gradient (H1 → ND edge space).
    bool   ccgd_enabled = true;       // Enable CCGD regularization
    Real   ccgd_tau     = 0.0;        // Scaling: 0=auto (τ = 1/(ωσ_mean)), >0=fixed

    // DoubledAMS preconditioner tuning
    int    ams_max_vcycles = 5;      // Max AMS V-cycles (high-freq regime, was 3)
    Real   ams_omega_mid   = 0.01;   // ω threshold for 2 V-cycles (f ~ 0.0016 Hz)
    Real   ams_omega_high  = 0.1;    // ω threshold for max V-cycles (f ~ 0.016 Hz)
    int    ams_smooth_type = 6;      // AMS smoother: 6=hybrid-symGS/SSOR (strongest GS variant)
    int    ams_smooth_sweeps = 3;    // Smoother sweeps per AMS level

    // Memory management
    bool   ooc_enabled    = false;    // MUMPS out-of-core (factorization on disk)
    std::string ooc_tmpdir = "/tmp";  // OOC temporary directory
    int    max_mem_mb      = 0;       // Max MUMPS memory per process (0=auto)
    int    mem_relax_pct   = 50;      // MUMPS workspace relaxation %
    bool   use_ext_mumps   = true;    // Always use MUMPSSolverExt for memory efficiency

    // Hybrid mode thresholds
    Real   direct_mem_fraction = 0.80; // Use direct if est. < this fraction of RAM (BLR safe)
};

/// 3D MT forward solver.
///
/// Key design: one frequency at a time to manage memory.
///   1. Assemble system matrix A(ω, σ)
///   2. Factorize A (direct) or build preconditioner (iterative)
///   3. Solve for each source polarization (Ex, Ey)
///   4. Extract impedance at receiver locations
///   5. Release factorization before next frequency
class ForwardSolver3D {
public:
    ForwardSolver3D() = default;

    /// Initialize with mesh and parameters.
    void setup(mfem::ParMesh& mesh,
               const model::ConductivityModel& model,
               const ForwardParams3D& params);

    /// Compute MT responses for all stations and frequencies.
    /// This is the main entry point for the forward problem.
    void compute_responses(const data::MTData& observed,
                           data::MTData& predicted);

    /// Compute responses for a single frequency (memory-friendly).
    /// After this call, the factorization is retained for adjoint use.
    /// Call release_factorization() when done with this frequency.
    void compute_single_frequency(int freq_idx,
                                  const data::MTData& observed,
                                  data::MTData& predicted);

    /// Factorize system for given frequency without solving for E fields.
    /// Used by GN-CG inner loop: background fields are cached separately,
    /// only factorization is needed for perturbation + adjoint solves.
    /// Call release_factorization() when done.
    /// @param freq_hz  Frequency in Hz (NOT index).
    void factorize_frequency(Real freq_hz);

    /// Set background E fields from external cache (for J·v without re-solving).
    /// Copies the provided fields into internal E1_, E2_ storage.
    void set_background_fields(const mfem::ParGridFunction& E1_r,
                               const mfem::ParGridFunction& E1_i,
                               const mfem::ParGridFunction& E2_r,
                               const mfem::ParGridFunction& E2_i);

    /// Adjoint solve: A^T λ = rhs, reuses current factorization.
    /// rhs and output are real-valued block vectors of size 2*TrueVSize
    /// (block format: [real; imag]).
    /// Must be called after compute_single_frequency() or factorize_frequency()
    /// while factorization is still in memory.
    /// If seed_real/seed_imag are provided, uses them as initial guess
    /// for iterative solver (Seed strategy: Pol1 adjoint → Pol2 adjoint).
    void adjoint_solve(const mfem::Vector& rhs,
                       mfem::ParGridFunction& adj_real,
                       mfem::ParGridFunction& adj_imag,
                       const mfem::ParGridFunction* seed_real = nullptr,
                       const mfem::ParGridFunction* seed_imag = nullptr);

    /// Solve A x = rhs using current factorization (for J·v).
    /// rhs is block vector [real; imag], output in grid functions.
    void solve_forward_rhs(const mfem::Vector& rhs,
                           mfem::ParGridFunction& sol_real,
                           mfem::ParGridFunction& sol_imag);

    /// Build adjoint RHS vectors from impedance residuals.
    /// Maps data residuals (observed - predicted, weighted) back to
    /// FE-space RHS for adjoint solve via Q^T operator.
    /// Returns one RHS vector per polarization.
    void build_adjoint_rhs(int freq_idx,
                           const data::MTData& observed,
                           const data::MTData& predicted,
                           const RealVec& data_weights,
                           mfem::Vector& adj_rhs_pol1,
                           mfem::Vector& adj_rhs_pol2);

    /// Build adjoint RHS from pre-computed weighted residuals (for CG J^T step).
    /// weighted_residual[s] = {W²·r_xx, W²·r_xy, W²·r_yx, W²·r_yy} per station.
    void build_adjoint_rhs_from_residual(
        int freq_idx,
        const std::vector<std::array<Complex,4>>& weighted_residual,
        mfem::Vector& adj_rhs_pol1,
        mfem::Vector& adj_rhs_pol2);

    /// Build perturbation RHS for J·v: assembles -ωδσ ∫ N_k · E dV.
    /// delta_sigma is per-element (size = n_elements).
    /// polarization: 0 = use E1 fields, 1 = use E2 fields.
    void build_perturbation_rhs(int polarization,
                                const RealVec& delta_sigma,
                                mfem::Vector& pert_rhs);

    /// Extract δZ at stations from perturbation field δE.
    /// Uses cached H_mat from extract_impedance_().
    /// delta_Z[s] = {δZxx, δZxy, δZyx, δZyy} per station.
    void extract_delta_impedance(
        const mfem::ParGridFunction& dE_real,
        const mfem::ParGridFunction& dE_imag,
        int polarization,
        std::vector<std::array<Complex,4>>& delta_Z);

    /// Compute element-wise sensitivity from forward and adjoint fields.
    /// g_e = Re(iω σ_e ∫_e λ·E dV) for one polarization.
    /// sensitivity vector has size n_elements.
    void compute_sensitivity(const mfem::ParGridFunction& E_real,
                             const mfem::ParGridFunction& E_imag,
                             const mfem::ParGridFunction& lambda_real,
                             const mfem::ParGridFunction& lambda_imag,
                             RealVec& sensitivity);

    /// Release the current factorization to free memory.
    void release_factorization();

    /// Get the FE space (for external use, e.g., sensitivity computation).
    mfem::ParFiniteElementSpace* fespace() { return fespace_.get(); }

    /// Access forward solution fields (current frequency).
    mfem::ParGridFunction* E1_real() { return E1_real_.get(); }
    mfem::ParGridFunction* E1_imag() { return E1_imag_.get(); }
    mfem::ParGridFunction* E2_real() { return E2_real_.get(); }
    mfem::ParGridFunction* E2_imag() { return E2_imag_.get(); }

    /// Frequency progress callback: (freq_idx, total_freq, freq_hz, phase_name).
    using FreqProgressCallback =
        std::function<void(int, int, Real, const std::string&)>;
    void set_freq_progress_callback(FreqProgressCallback cb) {
        freq_progress_cb_ = std::move(cb);
    }

    /// Current angular frequency.
    Real current_omega() const { return current_omega_; }

    /// Access system matrix (for diagnostics).
    mfem::HypreParMatrix* system_matrix() { return system_matrix_.get(); }

    /// Number of mesh elements.
    int num_elements() const { return mesh_ ? mesh_->GetNE() : 0; }

    /// Access memory profiler for reporting.
    MemoryProfiler& mem_profiler() { return mem_prof_; }

#ifdef MAPLE3DMT_USE_MPI
    MPI_Comm comm() const { return mesh_->GetComm(); }
#endif

private:
    mfem::ParMesh* mesh_ = nullptr;
    const model::ConductivityModel* model_ = nullptr;
    ForwardParams3D params_;
    FreqProgressCallback freq_progress_cb_;  // optional progress reporting

    // FE space: Nédélec (edge) elements
    std::unique_ptr<mfem::ND_FECollection> fec_;
    std::unique_ptr<mfem::ParFiniteElementSpace> fespace_;

    // System matrix (monolithic 2N×2N from ParSesquilinearForm)
    std::unique_ptr<mfem::HypreParMatrix> system_matrix_;

    // Eliminated parts from BC elimination (for manual RHS computation in solve_polarization_)
    // These contain the original values of essential rows/columns that were zeroed.
    // Used to compute B = -[elim_r, -elim_i; elim_i, elim_r] * X for BC-modified RHS.
    std::unique_ptr<mfem::HypreParMatrix> elim_real_;   // eliminated part of real (curl-curl)
    std::unique_ptr<mfem::HypreParMatrix> elim_imag_;   // eliminated part of imaginary (mass)

    // Solver (owned, type depends on backend)
    std::unique_ptr<mfem::Solver> solver_;
    std::unique_ptr<mfem::Solver> precond_;  // AMS preconditioner (for iterative mode)

    // Forward solution fields (current frequency, per polarization)
    std::unique_ptr<mfem::ParGridFunction> E1_real_, E1_imag_;
    std::unique_ptr<mfem::ParGridFunction> E2_real_, E2_imag_;

    // Current frequency info
    Real current_omega_ = 0.0;
    Real sigma_bg_ = 0.01;

    // Essential boundary DOFs (cached for adjoint reuse)
    mfem::Array<int> ess_tdof_list_;
    bool system_ready_ = false;  // true after factorization is set up

    // Stored sesquilinear form (kept alive for reuse in solve_polarization_)
    std::unique_ptr<mfem::ParSesquilinearForm> sesq_form_;
    mfem::Vector neg_omega_sigma_vec_;  // coefficient data (must outlive form)

    // Coefficient objects (must outlive sesq_form_; freed in release_factorization)
    std::unique_ptr<mfem::ConstantCoefficient> inv_mu0_coeff_;
    std::unique_ptr<mfem::Coefficient> mass_coeff_;  // ElementCoefficient

    // Station finder results (cached)
    mfem::DenseMatrix station_pts_;
    mfem::Array<int> station_elem_ids_;
    mfem::Array<mfem::IntegrationPoint> station_ips_;
    bool stations_found_ = false;

    // Cached H_mat inverse at stations (for adjoint RHS)
    struct StationCache {
        Complex Hx1, Hy1, Hx2, Hy2;  // H-field components
        Complex det_H;                 // det(H_mat)
    };
    std::vector<StationCache> station_H_cache_;

    // AMS preconditioner support (for ITERATIVE / HYBRID fallback)
    std::unique_ptr<mfem::H1_FECollection> h1_fec_;
    std::unique_ptr<mfem::ParFiniteElementSpace> h1_fespace_;
    std::unique_ptr<mfem::HypreParMatrix> grad_mat_;       // discrete gradient
    std::unique_ptr<mfem::HypreParVector> x_coord_, y_coord_, z_coord_;

    /// Build AMS auxiliary data (once, reused across frequencies).
    void setup_ams_auxiliary_();
    bool ams_ready_ = false;

    // Preconditioner components (iterative mode)
    // Strategy: PRESB preconditioner P = diag(K+B, K+B) where K = curl-curl,
    // B = ωσ mass.  Eigenvalues of P^{-1}A ∈ [1/2, 1] (Axelsson & Neytcheva).
    // BlockDiagAMS_PRESB: applies MFEM HypreAMS on N×N SPD block independently
    // for real/imag parts. Two AMS calls per FGMRES iter, each on N DOFs.
    std::unique_ptr<mfem::HypreParMatrix> prec_matrix_;    // 2N×2N SPD (PRESB, for DIAG/AMG fallback)
    std::unique_ptr<mfem::HypreParMatrix> spd_nxn_;        // N×N SPD block: K+|ωσ|M (for AMS)
    std::unique_ptr<mfem::HypreParMatrix> beta_poisson_;   // G^T·S·G = GtKG + ω·GtσMG (kept alive for HYPRE AMS)
    std::unique_ptr<mfem::HypreSolver>     ams_prec_;      // AMG preconditioner (fallback)
    std::unique_ptr<mfem::Solver>          block_prec_;     // BlockDiagAMS_PRESB or DiagScale
    mfem::Vector abs_omega_sigma_vec_;       // |ωσ| coefficient data
    std::unique_ptr<mfem::Coefficient> abs_mass_coeff_;  // positive mass coefficient

    // Cached beta Poisson components (split for fast per-frequency assembly):
    //   G^T·S·G = G^T·K·G + |ω| · G^T·(σM)·G
    //   GtKG: mesh-only, computed once (never changes)
    //   GtσMG: model-dependent, recomputed per inversion iteration
    //   Per frequency: beta_poisson_ = GtKG + |ω| * GtσMG (ms, matrix add)
    std::unique_ptr<mfem::HypreParMatrix> GtKG_;    // G^T·K·G (cached, mesh-only)
    std::unique_ptr<mfem::HypreParMatrix> GtσMG_;   // G^T·(σM)·G (cached, per model)
    bool GtKG_ready_ = false;                         // true after first computation

    // CCGD (Curl-Curl + Grad-Div) regularization support
    // NOTE: CCGD is applied ONLY in the FGMRES operator (CCGDOperator), NOT in the
    // preconditioner. Adding GGt to AMS preconditioner causes hang because AMS
    // internally computes G^T·A·G, which with GGt produces (G^T G)² — catastrophic.
    std::unique_ptr<mfem::HypreParMatrix> ccgd_GGt_;       // N×N: G·M_σ·G^T
    Real ccgd_tau_actual_ = 0.0;                             // actual τ used
    std::unique_ptr<mfem::Operator> ccgd_op_;               // CCGDOperator wrapper (A + τ·GGt)

    // Explicit transpose for adjoint solve (ModEM approach: solve A^T x = b directly)
    std::unique_ptr<mfem::HypreParMatrix> system_matrix_T_;  // A^T (2N×2N)
    std::unique_ptr<mfem::Operator> ccgd_op_T_;              // CCGDOperator wrapping A^T + τ·GGt

    /// Complex N×N solver (ModEM-equivalent: BiCGStab + ILU(0))
    std::unique_ptr<ComplexSolverWrapper> complex_solver_;
    bool using_complex_ = false;   // true when COMPLEX_BICGSTAB backend is active

    /// True if current solve uses iterative (not direct) solver.
    bool using_iterative_ = false;

    /// Cached MUMPS solver for adjoint direct solve (reused across pol1/pol2).
    /// Created on first adjoint_solve call per frequency, released in release_factorization().
    /// Avoids MUMPS BLR global-state crash from repeated create/destroy.
#ifdef MFEM_USE_MUMPS
    std::unique_ptr<MUMPSSolverExt> adjoint_mumps_;
#endif

    /// Divergence-corrected preconditioner for adjoint solve (ModEM approach).
    /// Wraps existing AMS + projects out gradient null-space after each application.
    /// Cached per frequency, released in release_factorization().
    std::unique_ptr<mfem::Solver> adjoint_div_prec_;    // DivCorrectedPreconditioner (for outer DivCorr)
    std::unique_ptr<mfem::Solver> adjoint_jacobi_;     // Jacobi preconditioner for adjoint
    std::unique_ptr<mfem::FGMRESSolver> adjoint_fgmres_;  // dedicated FGMRES for adjoint

    /// Assemble system and factorize/precondition for current omega.
    void assemble_and_factorize_(Real omega);

    /// Solve one polarization using stored factorization.
    /// If seed_real/seed_imag are provided (non-null) and using iterative solver,
    /// uses them as initial guess for interior DOFs (Seed strategy: Pol1→Pol2).
    void solve_polarization_(const mfem::ParGridFunction& E0_real,
                             const mfem::ParGridFunction& E0_imag,
                             mfem::ParGridFunction& E_real_out,
                             mfem::ParGridFunction& E_imag_out,
                             const mfem::ParGridFunction* seed_real = nullptr,
                             const mfem::ParGridFunction* seed_imag = nullptr);

    void compute_primary_field_(Real omega, int polarization,
                                mfem::ParGridFunction& E0_real,
                                mfem::ParGridFunction& E0_imag);

    void assemble_rhs_(Real omega, int polarization,
                       const mfem::ParGridFunction& E0_real,
                       const mfem::ParGridFunction& E0_imag,
                       mfem::Vector& rhs);

    void find_stations_(const data::MTData& data);

    void extract_impedance_(int freq_idx,
                            const data::MTData& observed,
                            data::MTData& predicted);

    // Memory profiler
    MemoryProfiler mem_prof_;
};

} // namespace forward
} // namespace maple3dmt
