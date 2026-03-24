// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file regularization_octree.h
/// @brief Octree-based model regularization for FV inversion.
///
/// Builds smoothness operator WᵀW from octree cell adjacency.
/// Each face neighbor pair contributes to the Laplacian-style operator.
/// Weight = face_area / distance² × direction weight (alpha_x/y/z).
///
/// Also provides CmCmᵀ gradient preconditioning (ModEM-style diffusion smoother).

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/model/conductivity_model.h"

namespace maple3dmt {
namespace inversion {

/// Regularization parameters for octree.
struct RegParamsOctree {
    Real alpha_s = 1.0;     // overall weight
    Real alpha_x = 1.0;     // x-direction smoothing
    Real alpha_y = 1.0;     // y-direction smoothing
    Real alpha_z = 1.0;     // z-direction smoothing
    Real alpha_r = 0.0;     // reference model damping
    int  n_smooth = 2;      // CmCmᵀ smoothing passes per Cm
};

/// Octree-based regularization operator.
class RegularizationOctree {
public:
    RegularizationOctree() = default;

    /// Build WᵀW from octree mesh adjacency.
    /// Earth cells are active (invertible), air cells are fixed.
    void setup(const octree::OctreeMesh& mesh,
               const RegParamsOctree& params);

    /// Set reference model for damping (active-space parameters).
    void set_reference_model(const RealVec& ref_log_sigma);

    /// Evaluate R(m) = α_s ||Wm (m - m_ref)||² (active-space).
    Real evaluate(const model::ConductivityModel& m) const;

    /// Compute ∇R(m) = 2 α_s WᵀW (m - m_ref) (global space, size = n_cells).
    void gradient(const model::ConductivityModel& m, RealVec& grad) const;

    /// Apply WᵀW to vector in active space.
    void apply_WtW(const RealVec& x, RealVec& result) const;

    /// Get diagonal of WᵀW (for preconditioning).
    void diagonal_WtW(RealVec& diag) const;

    /// Apply CmCmᵀ smoothing (ModEM-style diffusion) in active space.
    void apply_CmCmT(const RealVec& x, RealVec& result) const;

    /// Number of active (invertible) parameters.
    int n_active() const { return n_active_; }

    /// Map: global cell → active index (-1 if air/fixed).
    const std::vector<int>& global_to_active() const { return global_to_active_; }

    /// Map: active index → global cell.
    const std::vector<int>& active_to_global() const { return active_to_global_; }

    /// Current α_s.
    Real alpha_s() const { return params_.alpha_s; }

    /// Update α_s.
    void update_alpha(Real new_alpha) { params_.alpha_s = new_alpha; }

    /// Reference model parameters (active-space).
    const RealVec& ref_params() const { return ref_params_; }

private:
    RegParamsOctree params_;
    const octree::OctreeMesh* mesh_ = nullptr;
    int n_active_ = 0;

    // Active/global mappings
    std::vector<int> global_to_active_;
    std::vector<int> active_to_global_;

    // WᵀW stored as CSR in active space
    SparseMatR WtW_;

    // Reference model (active space)
    RealVec ref_params_;

    // Neighbor data for CmCmᵀ (stored per active cell)
    struct NeighborEntry {
        int active_idx;   // neighbor's active index
        Real weight;      // smoothing weight
    };
    std::vector<std::vector<NeighborEntry>> adj_list_;  // per active cell
    RealVec diag_weights_;  // sum of weights per active cell
};

} // namespace inversion
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
