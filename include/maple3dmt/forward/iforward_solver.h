// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file iforward_solver.h
/// @brief Abstract interface for 3D MT forward solvers.
///
/// Both MFEM-based FEM (ForwardSolver3D) and Octree FV (ForwardSolverFV)
/// implement this interface, allowing Inversion3D to work with either backend.

#include "maple3dmt/common.h"
#include "maple3dmt/data/mt_data.h"
#include <functional>
#include <string>

namespace maple3dmt {

namespace model { class ConductivityModel; }

namespace forward {

/// Abstract forward solver interface.
///
/// Provides the minimal API needed by the inversion loop:
///   - Forward solve (single freq or all)
///   - Adjoint solve
///   - Perturbation RHS + solve (for J·v)
///   - Sensitivity computation (for J^T·v)
///   - Impedance delta extraction (for J·v output mapping)
///   - Factorization lifecycle management
class IForwardSolver {
public:
    virtual ~IForwardSolver() = default;

    /// Compute MT responses for all frequencies.
    virtual void compute_responses(const data::MTData& observed,
                                   data::MTData& predicted) = 0;

    /// Compute responses for a single frequency.
    /// Factorization/system is retained for subsequent adjoint/perturbation solves.
    virtual void compute_single_frequency(int freq_idx,
                                          const data::MTData& observed,
                                          data::MTData& predicted) = 0;

    /// Factorize/assemble system for given frequency (Hz) without solving.
    /// Used by GN-CG inner loop with cached background fields.
    virtual void factorize_frequency(Real freq_hz) = 0;

    /// Release current factorization to free memory.
    virtual void release_factorization() = 0;

    /// Update conductivity from model (cell-centered σ values).
    virtual void update_sigma(const model::ConductivityModel& model) = 0;

    /// Number of model elements (cells).
    virtual int num_elements() const = 0;

    /// Current angular frequency.
    virtual Real current_omega() const = 0;

    // ---- Adjoint / sensitivity (FV-native interface using ComplexVec) ----

    /// Build adjoint RHS from data residuals for both polarizations.
    /// weighted_residual[s] = {W²·r_xx, W²·r_xy, W²·r_yx, W²·r_yy}.
    /// Returns two complex RHS vectors (one per polarization).
    virtual void build_adjoint_rhs_from_residual(
        int freq_idx,
        const std::vector<std::array<Complex,4>>& weighted_residual,
        ComplexVec& adj_rhs_pol1,
        ComplexVec& adj_rhs_pol2) = 0;

    /// Adjoint solve: A^T λ = rhs (complex).
    virtual void adjoint_solve_complex(const ComplexVec& rhs,
                                       ComplexVec& lambda) = 0;

    /// Compute element-wise sensitivity from E and λ fields.
    /// g[cell] = Re(iω Σ_{edges∈cell} conj(λ_e) · E_e · vol_e)
    virtual void compute_sensitivity_complex(
        const ComplexVec& E_bg,
        const ComplexVec& lambda,
        RealVec& sensitivity) = 0;

    /// Build perturbation RHS for J·v: -iω δσ Me E_bg.
    virtual void build_perturbation_rhs_complex(
        int polarization,
        const RealVec& delta_sigma,
        ComplexVec& pert_rhs) = 0;

    /// Solve A x = rhs (forward solve with arbitrary RHS, for J·v).
    virtual void solve_rhs_complex(const ComplexVec& rhs,
                                   ComplexVec& solution) = 0;

    /// Extract δZ at stations from perturbation field δE.
    /// delta_Z[s] = {δZxx, δZxy, δZyx, δZyy}.
    virtual void extract_delta_impedance_complex(
        const ComplexVec& dE, int polarization,
        std::vector<std::array<Complex,4>>& delta_Z) = 0;

    // ---- Background field caching ----

    /// Get background E-field for polarization (0=pol1, 1=pol2).
    virtual const ComplexVec& background_E(int pol) const = 0;

    /// Set background E-fields from external cache.
    virtual void set_background_fields_complex(const ComplexVec& E1,
                                               const ComplexVec& E2) = 0;

    /// Override solver tolerance (for relaxed inner solves in GN-CG).
    /// Pass 0 to restore default. Default implementation is no-op.
    virtual void set_solver_tolerance_override(Real /*tol*/) {}

    /// Frequency progress callback.
    using FreqProgressCB = std::function<void(int, int, Real, const std::string&)>;
    virtual void set_freq_progress_callback(FreqProgressCB cb) = 0;
};

} // namespace forward
} // namespace maple3dmt
