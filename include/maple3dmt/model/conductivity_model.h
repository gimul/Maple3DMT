// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file conductivity_model.h
/// @brief Conductivity model representation for 2.5D MT inversion.
///
/// The model is defined on the mesh as element-wise log-conductivity values.
/// Log parameterisation ensures positivity and improves inversion convergence.
///
/// === y-direction extension (v0.2+, quasi-3D) ===
///
/// For pure 2D (v0.1): σ(x,z) — one parameter per mesh element.
///
/// For quasi-3D (v0.2+): σ(x,y,z) ≈ Σ_k σ_k(x,z) · φ_k(y)
///   where φ_k(y) are Fourier basis functions in the strike direction.
///   - k=0: the base 2D model (same as v0.1)
///   - k≥1: perturbation modes that introduce y-variation
///   - Total params = n_elements × n_y_modes
///   - This breaks TE/TM symmetry and enables full tensor forward.

#include "maple3dmt/common.h"
#ifdef MAPLE3DMT_USE_MFEM
#include "maple3dmt/mesh/terrain_mesh.h"
#include "maple3dmt/mesh/mesh_projection.h"
#endif

namespace maple3dmt {
namespace model {

/// Model parameterisation type.
enum class Parameterisation {
    LOG_CONDUCTIVITY,     // ln(sigma)
    LOG_RESISTIVITY       // ln(rho)
};

/// Conductivity model living on a TerrainMesh.
///
/// Supports two modes:
///   - Pure 2D:     params_ has n_elements values (one per cell)
///   - Quasi-3D:    params_ has n_elements × n_y_modes values
///                  Indexed as [mode * n_elements + element]
class ConductivityModel {
public:
    ConductivityModel() = default;

#ifdef MAPLE3DMT_USE_MFEM
    /// Initialise with a mesh and uniform conductivity (pure 2D).
    void init(const mesh::TerrainMesh& tmesh, Real sigma_background);
#endif

    /// Initialise for 3D mesh with uniform conductivity.
    /// Does not depend on TerrainMesh — uses element count directly.
    void init_3d(int n_elements, Real sigma_background);

#ifdef MAPLE3DMT_USE_MFEM
    /// Initialise with quasi-3D y-modes.
    /// n_y_modes = 1 is equivalent to pure 2D.
    /// Mode 0 = background, modes 1..N = perturbations (initialised to zero).
    void init_quasi3d(const mesh::TerrainMesh& tmesh,
                      Real sigma_background,
                      int n_y_modes);

    /// Initialise with dual-mesh support.
    /// Parameters live on the inversion mesh; sigma_fwd() projects to the
    /// forward mesh via the MeshProjection.
    void init_dual(const mesh::TerrainMesh& inv_mesh,
                   const mesh::TerrainMesh& fwd_mesh,
                   const mesh::MeshProjection& proj,
                   Real sigma_background);
#endif

    /// Number of free parameters (total, including all y-modes).
    int num_params() const { return static_cast<int>(params_.size()); }

    /// Number of mesh elements (2D cross-section).
    int num_elements() const { return n_elements_; }

    /// Number of y-direction modes (1 = pure 2D).
    int num_y_modes() const { return n_y_modes_; }

    /// Is this a quasi-3D model?
    bool is_quasi3d() const { return n_y_modes_ > 1; }

    /// Access raw parameter vector (log-sigma or log-rho).
    const RealVec& params() const { return params_; }
    RealVec&       params()       { return params_; }

    /// Get conductivity for element i, mode 0 (base 2D model, always in S/m).
    /// This operates on the model mesh (inversion mesh in dual-mesh mode).
    Real sigma(int i) const;

    /// Get conductivity for forward-mesh element i.
    /// In dual-mesh mode, projects from inversion mesh via MeshProjection.
    /// In single-mesh mode (no projection), equivalent to sigma(i).
    Real sigma_fwd(int fwd_elem) const;

    /// Number of forward-mesh elements.
    /// Returns num_elements() in single-mesh mode.
    int num_fwd_elements() const;

    /// Get conductivity for element i, y-mode k.
    /// k=0: base model. k≥1: Fourier perturbation coefficient.
    Real sigma(int i, int k) const;

    /// Get raw perturbation coefficient for element i, y-mode k (k≥1).
    /// Unlike sigma(i,k) which applies exp(), this returns the raw parameter
    /// value directly — appropriate for linear Fourier coefficients.
    Real perturbation(int i, int k) const;

    /// Get resistivity for element i (mode 0, always in Ohm·m).
    Real rho(int i) const;

    /// Set parameterisation.
    void set_parameterisation(Parameterisation p) { param_type_ = p; }

    /// Set active region mask (true = invertible, false = fixed).
    /// Size must be n_elements (applied to all y-modes equally) or
    /// n_elements * n_y_modes (per-mode control).
    void set_active_mask(const std::vector<bool>& mask);

    /// Save model to HDF5 (includes y-mode metadata).
    void save(const fs::path& path) const;

    /// Load model from HDF5.
    void load(const fs::path& path);

    /// Perturb model parameter for gradient computation.
    /// Invalidates the forward sigma cache in dual-mesh mode.
    void perturb(int idx, Real delta);

    /// Invalidate the forward sigma projection cache.
    /// Call this after directly modifying params().
    void invalidate_cache() { cache_dirty_ = true; }

    /// Check if dual-mesh mode is active.
#ifdef MAPLE3DMT_USE_MFEM
    bool has_projection() const { return proj_ != nullptr; }
#else
    bool has_projection() const { return false; }
#endif

private:
    RealVec params_;                             // model parameters
    Parameterisation param_type_ = Parameterisation::LOG_CONDUCTIVITY;
    std::vector<bool> active_mask_;              // inversion mask
    int n_elements_ = 0;
    int n_y_modes_  = 1;                         // 1 = pure 2D

    // Dual-mesh support
#ifdef MAPLE3DMT_USE_MFEM
    const mesh::MeshProjection* proj_ = nullptr; // non-owning
#endif
    int n_fwd_elements_ = 0;                     // forward mesh element count
    mutable RealVec fwd_sigma_cache_;            // cached projected log(sigma)
    mutable bool cache_dirty_ = true;

    /// Index into params_ for element i, y-mode k.
    int idx_(int elem, int mode) const {
        return mode * n_elements_ + elem;
    }
};

} // namespace model
} // namespace maple3dmt
