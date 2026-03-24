// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file forward_solver_fv.h
/// @brief Octree FV-based 3D MT forward solver.
///
/// Staggered grid (Yee on octree) + BiCGStab+Jacobi + DivCorr.
/// Complex N×N system (NOT 2N×2N real block).

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/octree/staggered_grid.h"
#include "maple3dmt/octree/operators.h"
#include "maple3dmt/forward/bicgstab.h"
#include "maple3dmt/forward/iforward_solver.h"
#include "maple3dmt/data/mt_data.h"

#include <functional>

namespace maple3dmt {

namespace model { class ConductivityModel; }

namespace forward {

/// Forward solver parameters.
struct ForwardParamsFV {
    enum class Backend { BICGSTAB_JACOBI, MUMPS_COMPLEX, HYBRID };
    Backend backend = Backend::BICGSTAB_JACOBI;

    // Iterative solver
    Real bicgstab_tol     = 1e-7;
    int  bicgstab_maxiter = 2000;
    int  print_level      = 1;    // 0=silent, 1=summary, 2=per-iter

    // Scattered field formulation
    bool scattered_field  = true;   // E = E_pri + E_sec, solve for E_sec

    // Air Dirichlet BC threshold
    // Edges with z > air_z_threshold are fixed to E_primary (kills E_scattered).
    // Low values (1.0) = safe convergence but biased ρ_a.
    // High values (10000+) = better ρ_a but needs good initial guess.
    // Set to domain_z_max to disable (only domain boundary BCs).
    Real air_z_threshold  = 1.0;

    // Air BC update iterations (two-pass scheme):
    //   0 or 1 = single pass (standard)
    //   ≥2     = iteratively update air BCs with E_scattered from previous solve
    int  air_bc_iterations = 1;

    // Divergence correction (ModEM-style interleaved outer loop)
    bool div_correction       = true;
    int  div_corr_iters       = 10;   // max Poisson iterations per DivCorr call
    Real div_corr_tol         = 2e-3; // Poisson convergence: ||divJ||/||divJ0|| < tol
    int  divcorr_outer_max    = 20;   // max outer DivCorr iterations
    int  divcorr_iter_per_dc  = 500;  // BiCGStab iters between DivCorr cleanups
    //  ModEM uses IterPerDivCor=2000 for ~100K DOFs.
    //  500 is a good balance: gives BiCGStab enough room to converge
    //  before DivCorr cleanup. Too small (100) → stagnation; too large
    //  (5000) → DivCorr never gets a chance to help.

    // Direct solver (fallback)
    Real blr_tolerance    = 1e-2;
    int  hybrid_threshold = 100000;
};

/// Frequency progress callback.
using FVFreqProgressCB = std::function<void(int freq_idx, int n_freq,
                                             const std::string& phase)>;

/// FV-based 3D MT forward solver on octree mesh.
/// Implements IForwardSolver for use with Inversion3D.
class ForwardSolverFV : public IForwardSolver {
public:
    ForwardSolverFV() = default;

    /// Setup: build operators from mesh.
    void setup(octree::OctreeMesh& mesh,
               const ForwardParamsFV& params);

    /// Set conductivity model (cell-centered sigma values).
    void set_sigma(const RealVec& sigma);

    // ---- IForwardSolver interface ----
    void update_sigma(const model::ConductivityModel& model) override;

    void compute_responses(const data::MTData& observed,
                           data::MTData& predicted) override;

    void compute_single_frequency(int freq_idx,
                                  const data::MTData& observed,
                                  data::MTData& predicted) override;

    void factorize_frequency(Real freq_hz) override;

    void release_factorization() override;

    int num_elements() const override { return ops_.num_cells(); }

    Real current_omega() const override { return current_omega_; }

    void build_adjoint_rhs_from_residual(
        int freq_idx,
        const std::vector<std::array<Complex,4>>& weighted_residual,
        ComplexVec& adj_rhs_pol1,
        ComplexVec& adj_rhs_pol2) override;

    void adjoint_solve_complex(const ComplexVec& rhs,
                               ComplexVec& lambda) override;

    void compute_sensitivity_complex(
        const ComplexVec& E_bg,
        const ComplexVec& lambda,
        RealVec& sensitivity) override;

    void build_perturbation_rhs_complex(
        int polarization,
        const RealVec& delta_sigma,
        ComplexVec& pert_rhs) override;

    void solve_rhs_complex(const ComplexVec& rhs,
                           ComplexVec& solution) override;

    void extract_delta_impedance_complex(
        const ComplexVec& dE, int polarization,
        std::vector<std::array<Complex,4>>& delta_Z) override;

    const ComplexVec& background_E(int pol) const override {
        return (pol == 0) ? E1_ : E2_;
    }

    void set_background_fields_complex(const ComplexVec& E1,
                                       const ComplexVec& E2) override;

    void set_freq_progress_callback(FreqProgressCB cb) override {
        freq_cb_ = [cb](int fi, int nf, const std::string& ph) {
            if (cb) cb(fi, nf, 0.0, ph);
        };
    }

    // ---- Legacy interface (backwards compatible) ----

    /// Single frequency solve (both polarizations).
    void solve_frequency(Real freq_hz, int freq_idx,
                         const data::MTData& observed,
                         data::MTData& predicted);

    /// Adjoint solve: same system, different RHS (A^T = A).
    void adjoint_solve(const ComplexVec& rhs, ComplexVec& lambda);

    /// Forward solve with arbitrary RHS (for J·v in GN-CG).
    void solve_rhs(const ComplexVec& rhs, ComplexVec& solution);

    /// Build perturbation RHS: -iω δσ Me E_bg.
    void build_perturbation_rhs(int polarization,
                                const RealVec& delta_sigma,
                                ComplexVec& pert_rhs);

    /// Sensitivity: g_cell = Re(iω Σ_edges conj(λ_e) * E_e * vol_e)
    void compute_sensitivity(const ComplexVec& E_bg,
                             const ComplexVec& lambda,
                             RealVec& sensitivity);

    /// Release memory (between frequencies).
    void release();

    /// Access E-field solutions.
    const ComplexVec& E1() const { return E1_; }
    const ComplexVec& E2() const { return E2_; }
    void set_background_fields(const ComplexVec& E1, const ComplexVec& E2);

    /// Set frequency progress callback (legacy).
    void set_freq_progress_cb(FVFreqProgressCB cb) { freq_cb_ = std::move(cb); }

    /// Temporarily override BiCGStab tolerance (for JtJ inner solves).
    /// Pass 0 to restore original params_.bicgstab_tol.
    void set_solver_tolerance_override(Real tol) override { tol_override_ = tol; }

    int num_edges() const { return ops_.num_edges(); }
    int num_cells() const { return ops_.num_cells(); }

    /// Access mesh.
    octree::OctreeMesh* mesh() const { return mesh_; }

    /// Access staggered grid.
    const octree::StaggeredGrid& staggered() const { return mesh_->staggered(); }

private:
    octree::OctreeMesh*      mesh_ = nullptr;
    ForwardParamsFV          params_;
    octree::DiscreteOperators ops_;

    RealVec sigma_;        // cell-centered conductivity
    Real    current_omega_ = 0;
    Real    sigma_bg_ = -1;    // background σ for scattered field (-1 = auto from mesh params)

    // System matrix (assembled per frequency)
    SparseMatC A_;

    // Solution fields
    ComplexVec E1_, E2_;

    // Station location mapping
    struct StationCell {
        int cell_id;
        Real wx, wy, wz;  // interpolation weights
    };
    std::vector<StationCell> station_cells_;
    bool stations_mapped_ = false;

    // Cached H-field and Z at stations (for adjoint RHS and δZ extraction)
    struct StationHCache {
        Complex Hx1, Hy1, Hx2, Hy2;
        Complex det_H;
        // Z matrix (cached for adjoint H-contribution)
        Complex Zxx, Zxy, Zyx, Zyy;
        // Cell dimensions (for curl transpose in adjoint)
        Real cdx, cdy, cdz;
    };
    std::vector<StationHCache> station_H_cache_;

    FVFreqProgressCB freq_cb_;
    Real tol_override_ = 0;  // 0 = use params_.bicgstab_tol

    // Internal methods
    void assemble_system_(Real omega);
    void solve_system_(const ComplexVec& rhs, ComplexVec& x);
    void divergence_correction_(ComplexVec& E, const ComplexVec& rhs,
                                const std::vector<bool>& is_bc, Real omega,
                                std::function<void(const ComplexVec&, ComplexVec&)> inner_solve);
    void divcorr_poisson_(ComplexVec& E, const ComplexVec* phi0 = nullptr);
    void solve_with_divcorr_(const ComplexVec& rhs, ComplexVec& x,
                              const ComplexVec* phi0, const char* label);
    void compute_bc_flags_();
    void compute_primary_field_(Real omega, int pol, ComplexVec& E0);
    void build_rhs_scattered_(Real omega, int pol,
                               const ComplexVec& E0, ComplexVec& rhs);
    void find_stations_(const data::MTData& observed);
    void extract_impedance_(int freq_idx,
                            const data::MTData& observed,
                            data::MTData& predicted);

    // Cached BC flags and DivCorr Laplacian (recomputed per frequency/sigma)
    std::vector<bool> cached_is_bc_;
    std::vector<bool> cached_bc_nodes_;
    SparseMatR cached_L_;         // node Laplacian for DivCorr
    RealVec cached_L_diag_;       // diagonal of L (for Jacobi precond)
    bool divcorr_setup_done_ = false;
};

} // namespace forward
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
