// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file implicit_jtj_fv.cpp
/// @brief Matrix-free (JᵀW²J + λWᵀW)·v for octree FV inversion.

#include "maple3dmt/inversion/implicit_jtj_fv.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/inversion/regularization_octree.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/utils/logger.h"
#include <cmath>

#define LOG_INFO(msg)  MAPLE3DMT_LOG_INFO(msg)
#define LOG_DEBUG(msg) MAPLE3DMT_LOG_DEBUG(msg)

namespace maple3dmt {
namespace inversion {

// =========================================================================
// Constructor
// =========================================================================
ImplicitJtJOperatorFV::ImplicitJtJOperatorFV(
    forward::IForwardSolver& fwd,
    const data::MTData& data,
    const model::ConductivityModel& model,
    const RegularizationOctree& reg,
    Real lambda)
    : fwd_(&fwd), data_(&data), model_(&model), reg_(&reg),
      lambda_(lambda), n_active_(reg.n_active()) {}

// =========================================================================
// Cache background E fields
// =========================================================================
void ImplicitJtJOperatorFV::cache_background_fields() {
    int nf = data_->num_frequencies();
    bg_cache_.resize(nf);

    for (int f = 0; f < nf; ++f) {
        Real freq_hz = data_->frequencies()[f];
        bg_cache_[f].freq_hz = freq_hz;

        // Forward solve to get background fields
        data::MTData pred_dummy = *data_;
        fwd_->compute_single_frequency(f, *data_, pred_dummy);

        bg_cache_[f].E1 = fwd_->background_E(0);
        bg_cache_[f].E2 = fwd_->background_E(1);

        fwd_->release_factorization();
    }

    LOG_INFO("JtJ-FV: cached background fields for " + std::to_string(nf) + " frequencies");
}

// =========================================================================
// Ensure work vectors allocated
// =========================================================================
void ImplicitJtJOperatorFV::ensure_work_vectors_() const {
    if (work_allocated_) return;

    int ne = fwd_->num_elements();
    int ns = data_->num_stations();

    work_delta_sigma_.resize(ne);
    work_sensitivity_.resize(ne);
    work_wtw_.resize(n_active_);
    work_dZ_.resize(ns);
    work_weighted_dZ_.resize(ns);
    work_allocated_ = true;
}

// =========================================================================
// Mult: y = (JᵀW²J + λWᵀW) x
// =========================================================================
void ImplicitJtJOperatorFV::Mult(const RealVec& x, RealVec& y) const {
    ensure_work_vectors_();
    ++num_matvecs_;

    int ne = fwd_->num_elements();
    int ns = data_->num_stations();
    int nf = data_->num_frequencies();
    const auto& a2g = reg_->active_to_global();

    // Initialize output
    y.assign(n_active_, 0.0);

    // Map active-space x to global delta_sigma
    // x is in log-sigma space → delta_sigma = σ · δ(ln σ) = σ · x
    work_delta_sigma_.assign(ne, 0.0);
    for (int j = 0; j < n_active_; ++j) {
        int g = a2g[j];
        work_delta_sigma_[g] = model_->params()[g] * x[j];
        // model_->params() stores σ values. In log-space inversion,
        // we need d(σ)/d(ln σ) = σ. But if params are ln(σ), then
        // delta_sigma = exp(ln σ) * x = σ * x.
        // For simplicity: pass x directly as δ(ln σ), and the
        // perturbation RHS handles the σ multiplication internally.
    }
    // Actually: the perturbation RHS is -iω δσ Me E_bg.
    // δσ = σ · δ(ln σ) = σ · x_j for active cell j.
    // So work_delta_sigma should be σ * x.
    // Let's recompute properly:
    for (int j = 0; j < n_active_; ++j) {
        int g = a2g[j];
        Real sigma_g = std::exp(model_->params()[g]);  // params stores ln(σ)
        work_delta_sigma_[g] = sigma_g * x[j];
    }

    // Use relaxed tolerance for inner solves (inexact Newton principle)
    fwd_->set_solver_tolerance_override(1e-4);

    // Loop over frequencies
    for (int f = 0; f < nf; ++f) {
        // 1. Set up system for this frequency
        fwd_->factorize_frequency(bg_cache_[f].freq_hz);
        fwd_->set_background_fields_complex(bg_cache_[f].E1, bg_cache_[f].E2);

        // Per-frequency JᵀW²J contribution
        RealVec freq_sensitivity(ne, 0.0);

        for (int pol = 0; pol < 2; ++pol) {
            // --- J·v: perturbation forward solve ---
            // Build perturbation RHS: -iω δσ Me E_bg
            fwd_->build_perturbation_rhs_complex(pol + 1, work_delta_sigma_,
                                                  work_pert_rhs_);

            // Solve for δE
            fwd_->solve_rhs_complex(work_pert_rhs_, work_dE_);

            // Extract δZ at stations
            fwd_->extract_delta_impedance_complex(work_dE_, pol + 1, work_dZ_);

            // --- Apply W² to δZ ---
            work_weighted_dZ_.resize(ns);
            for (int s = 0; s < ns; ++s) {
                const auto& obs = data_->observed(s, f);
                auto weight_comp = [](const data::Datum& d, Complex dz) -> Complex {
                    if (d.weight <= 0.0 || d.error <= 0.0) return Complex(0, 0);
                    Real w = 1.0 / d.error;
                    return w * w * dz;
                };
                work_weighted_dZ_[s][0] = weight_comp(obs.Zxx, work_dZ_[s][0]);
                work_weighted_dZ_[s][1] = weight_comp(obs.Zxy, work_dZ_[s][1]);
                work_weighted_dZ_[s][2] = weight_comp(obs.Zyx, work_dZ_[s][2]);
                work_weighted_dZ_[s][3] = weight_comp(obs.Zyy, work_dZ_[s][3]);
            }

            // --- J^T·(W²·Jv): adjoint solve ---
            // Build adjoint RHS from weighted δZ
            ComplexVec adj_rhs_p1, adj_rhs_p2;
            fwd_->build_adjoint_rhs_from_residual(f, work_weighted_dZ_,
                                                   adj_rhs_p1, adj_rhs_p2);

            // Adjoint solve
            ComplexVec lam(adj_rhs_p1.size(), Complex(0, 0));
            // Only need the adjoint for the current polarization's contribution
            if (pol == 0) {
                fwd_->adjoint_solve_complex(adj_rhs_p1, lam);
            } else {
                fwd_->adjoint_solve_complex(adj_rhs_p2, lam);
            }

            // Sensitivity: g_cell = Re(iω Σ conj(λ)·E·vol)
            RealVec sens(ne, 0.0);
            const ComplexVec& E_bg = (pol == 0) ? bg_cache_[f].E1 : bg_cache_[f].E2;
            fwd_->compute_sensitivity_complex(E_bg, lam, sens);

            // Accumulate
            for (int e = 0; e < ne; ++e)
                freq_sensitivity[e] += sens[e];
        }

        fwd_->release_factorization();

        // Map sensitivity to active space and accumulate
        for (int j = 0; j < n_active_; ++j) {
            y[j] += freq_sensitivity[a2g[j]];
        }
    }

    // Restore original tolerance
    fwd_->set_solver_tolerance_override(0);

    // Add regularization: λ · WᵀW · x
    reg_->apply_WtW(x, work_wtw_);
    for (int j = 0; j < n_active_; ++j) {
        y[j] += lambda_ * work_wtw_[j];
    }
}

} // namespace inversion
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
