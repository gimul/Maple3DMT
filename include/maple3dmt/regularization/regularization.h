// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file regularization.h
/// @brief Model regularisation operators for MT inversion.
///
/// Supports smooth (Laplacian) regularisation with optional
/// anisotropy between horizontal and vertical directions.
/// Terrain-aware: smoothing respects mesh geometry so that
/// regularisation follows the terrain surface, not flat layers.

#include "maple3dmt/common.h"
#include "maple3dmt/mesh/terrain_mesh.h"
#include "maple3dmt/model/conductivity_model.h"
#include <mfem.hpp>

namespace maple3dmt {
namespace regularization {

/// Regularisation type.
enum class RegType {
    SMOOTH_L2,        // standard L2 (Tikhonov) smoothness
    SMOOTH_L1,        // L1 smoothness (edge-preserving)
    MINIMUM_GRADIENT, // minimum gradient support
};

/// Parameters for regularisation.
struct RegParams {
    RegType type = RegType::SMOOTH_L2;

    Real alpha_s = 1.0;     // overall weight (trade-off parameter)
    Real alpha_x = 1.0;     // horizontal smoothing weight
    Real alpha_z = 1.0;     // vertical smoothing weight

    // Reference model
    bool use_reference_model = false;
    Real alpha_r = 0.0;     // reference model weight

    // Quasi-3D perturbation mode damping (mode k≥1)
    Real alpha_pert = 1.0;  // extra damping for y-perturbation modes

    // Cooling schedule for trade-off parameter
    Real cooling_factor = 1.0;   // multiply alpha_s each iteration

    // CmCm^T smoothing passes (ModEM-style gradient preconditioning).
    // Each Cm application performs n_smooth passes of diffusion averaging.
    // CmCm^T = 2 × n_smooth total passes. Higher values = more smoothing.
    int  n_smooth = 2;          // passes per Cm (ModEM default: 2)
};

/// Regularisation operator.
class Regularization {
public:
    Regularization() = default;

    /// Build the regularisation operator on the mesh (pure 2D).
    void setup(const mesh::TerrainMesh& tmesh,
               const RegParams& params);

    /// Build the regularisation operator on a 3D ParMesh.
    /// Active elements determined by attribute (attr==1 = earth, attr==2 = air).
    void setup_3d(mfem::ParMesh& pmesh, const RegParams& params);

    /// Build regularisation for quasi-3D model.
    /// Mode 0: spatial smoothing (same as setup()).
    /// Mode k≥1: spatial smoothing + extra damping toward zero.
    void setup_quasi3d(const mesh::TerrainMesh& tmesh,
                       const model::ConductivityModel& model,
                       const RegParams& params);

    /// Set reference model for damping toward a target.
    void set_reference_model(const model::ConductivityModel& ref);

    /// Evaluate regularisation functional: R(m).
    Real evaluate(const model::ConductivityModel& m) const;

    /// Compute gradient of R(m): ∂R/∂m.
    void gradient(const model::ConductivityModel& m,
                  RealVec& grad) const;

    /// Apply regularisation Hessian (W^T W) to a vector.
    void apply_WtW(const RealVec& x, RealVec& result) const;

    /// Get diagonal of W^T W (for preconditioning).
    void diagonal_WtW(RealVec& diag) const;

    /// Apply CmCm^T (model covariance) smoothing to a vector (active space).
    /// ModEM-style diffusion smoother: each Cm is n_smooth passes of
    /// neighbor-weighted averaging using WtW connectivity.
    /// CmCm^T = apply Cm twice = 2*n_smooth total passes.
    void apply_CmCmT(const mfem::Vector& x, mfem::Vector& result) const;

    /// Get the assembled regularisation matrix.
    const mfem::SparseMatrix& matrix() const { return *WtW_; }

    /// Update trade-off parameter (e.g., cooling schedule).
    void update_alpha(Real new_alpha);

    /// Access reference model parameters (active-space, empty if not set).
    const RealVec& ref_params() const { return ref_params_; }

    /// Number of active (invertible) parameters.
    int n_active() const { return n_active_; }

    /// Current regularization weight α_s (for Hessian scaling: 2·α_s·WtW).
    Real alpha_s() const { return params_.alpha_s; }

    /// Map: global element index → active parameter index (-1 if air/fixed).
    const std::vector<int>& global_to_active() const { return global_to_active_; }

    /// Map: active parameter index → global element index.
    const std::vector<int>& active_to_global() const { return active_to_global_; }

private:
    RegParams params_;
    std::unique_ptr<mfem::SparseMatrix> WtW_;
    RealVec ref_params_;   // reference model parameters

    std::vector<int> global_to_active_;  // elem → active idx (-1 if air)
    std::vector<int> active_to_global_;  // active idx → elem
    int n_active_ = 0;

    // Shared-face data for MPI ghost-parameter exchange in gradient/apply_WtW.
    // Each entry represents a shared face between a local active element and
    // a ghost element on another rank.  The diagonal contribution (w) is
    // already in WtW_; these store the off-diagonal terms that require
    // the neighbor rank's parameter value.
    std::vector<double> shared_face_weights_;       // smoothing weight w
    std::vector<int>    shared_face_local_active_;  // active index of local elem
    std::vector<int>    shared_face_nbr_global_elem_; // ParMesh elem index of ghost

    mfem::ParMesh* pmesh_ = nullptr;  // non-owning; set in setup_3d()

    /// Build terrain-aware gradient operator.
    void build_gradient_operator_(const mesh::TerrainMesh& tmesh);
};

} // namespace regularization
} // namespace maple3dmt
