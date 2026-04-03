// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file implicit_jtj_fv.h
/// @brief Matrix-free (J^T W^2 J + λ W^T W)·v operator for octree FV.
///
/// Each matvec requires:
///   For each frequency:
///     1. J·v: build perturbation RHS from δσ → forward solve → extract δZ
///     2. J^T·(W²·J·v): build adjoint RHS from weighted δZ → adjoint solve → sensitivity
///   Then add λ · WtW · v
///
/// Key advantage over FEM version:
///   A^T = A (complex symmetric) → adjoint = forward system → true CG possible.

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/forward/iforward_solver.h"
#include "maple3dmt/data/mt_data.h"

namespace maple3dmt {

namespace model { class ConductivityModel; }

namespace inversion {

// Forward declare RegularizationOctree
class RegularizationOctree;

/// Matrix-free (JᵀW²J + λWᵀW)·v on octree FV.
///
/// Used as the operator in CG inner solve of Gauss-Newton.
/// Since A^T = A for FV, JᵀJ is exactly symmetric → CG converges.
class ImplicitJtJOperatorFV {
public:
    ImplicitJtJOperatorFV(forward::IForwardSolver& fwd,
                           const data::MTData& data,
                           const model::ConductivityModel& model,
                           const RegularizationOctree& reg,
                           Real lambda);

    /// y = (JᵀW²J + λWᵀW) x
    /// x and y are in active parameter space (n_active).
    void Mult(const RealVec& x, RealVec& y) const;

    /// Cache background E fields for all frequencies.
    /// Must be called ONCE before the CG loop begins.
    void cache_background_fields();

    /// Check if background fields are cached.
    bool has_cached_fields() const { return !bg_cache_.empty(); }

    /// Number of matvecs performed (= CG iterations).
    int num_matvecs() const { return num_matvecs_; }

    /// Dimension of operator (active parameters).
    int size() const { return n_active_; }

private:
    forward::IForwardSolver* fwd_;
    const data::MTData* data_;
    const model::ConductivityModel* model_;
    const RegularizationOctree* reg_;
    Real lambda_;
    int n_active_;
    mutable int num_matvecs_ = 0;

    /// Per-frequency cached background E fields.
    struct FreqCache {
        ComplexVec E1, E2;
        Real freq_hz;
    };
    std::vector<FreqCache> bg_cache_;

    // Pre-allocated work vectors (avoid allocation in Mult)
    mutable RealVec work_delta_sigma_;
    mutable RealVec work_sensitivity_;
    mutable RealVec work_wtw_;
    mutable ComplexVec work_pert_rhs_;
    mutable ComplexVec work_dE_;
    mutable ComplexVec work_adj_rhs1_, work_adj_rhs2_;
    mutable ComplexVec work_lam1_, work_lam2_;
    mutable std::vector<std::array<Complex,4>> work_dZ_;
    mutable std::vector<std::array<Complex,4>> work_weighted_dZ_;
    mutable bool work_allocated_ = false;

    void ensure_work_vectors_() const;
};

} // namespace inversion
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
