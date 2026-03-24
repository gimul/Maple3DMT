// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file conductivity_model.cpp
/// @brief Implementation of conductivity model.

#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/utils/logger.h"
#include <cmath>
#include <stdexcept>

namespace maple3dmt {
namespace model {

#ifdef MAPLE3DMT_USE_MFEM
void ConductivityModel::init(const mesh::TerrainMesh& tmesh, Real sigma_background) {
    n_elements_ = tmesh.num_elements();
    Real log_val;
    if (param_type_ == Parameterisation::LOG_CONDUCTIVITY) {
        log_val = std::log(sigma_background);
    } else {
        log_val = std::log(1.0 / sigma_background);
    }
    params_.assign(n_elements_, log_val);
    active_mask_.assign(n_elements_, true);

    MAPLE3DMT_LOG_INFO("Model initialised: " + std::to_string(n_elements_) +
                   " cells, background sigma = " +
                   std::to_string(sigma_background) + " S/m");
}
#endif // MAPLE3DMT_USE_MFEM

void ConductivityModel::init_3d(int n_elements, Real sigma_background) {
    n_elements_ = n_elements;
    n_y_modes_ = 1;
    Real log_val;
    if (param_type_ == Parameterisation::LOG_CONDUCTIVITY) {
        log_val = std::log(sigma_background);
    } else {
        log_val = std::log(1.0 / sigma_background);
    }
    params_.assign(n_elements_, log_val);
    active_mask_.assign(n_elements_, true);
#ifdef MAPLE3DMT_USE_MFEM
    proj_ = nullptr;
#endif
    n_fwd_elements_ = 0;
    cache_dirty_ = true;

    MAPLE3DMT_LOG_INFO("3D model initialised: " + std::to_string(n_elements_) +
                   " elements, background sigma = " +
                   std::to_string(sigma_background) + " S/m");
}

Real ConductivityModel::sigma(int i) const {
    if (param_type_ == Parameterisation::LOG_CONDUCTIVITY) {
        return std::exp(params_[i]);
    } else {
        return 1.0 / std::exp(params_[i]);
    }
}

Real ConductivityModel::perturbation(int i, int k) const {
    return params_[idx_(i, k)];
}

Real ConductivityModel::rho(int i) const {
    return 1.0 / sigma(i);
}

void ConductivityModel::set_active_mask(const std::vector<bool>& mask) {
    if (static_cast<int>(mask.size()) != n_elements_) {
        throw std::runtime_error("Active mask size mismatch");
    }
    active_mask_ = mask;
}

void ConductivityModel::perturb(int idx, Real delta) {
    if (idx < 0 || idx >= static_cast<int>(params_.size())) {
        throw std::out_of_range("Model parameter index out of range");
    }
    params_[idx] += delta;
    cache_dirty_ = true;
}

#ifdef MAPLE3DMT_USE_MFEM
void ConductivityModel::init_dual(const mesh::TerrainMesh& inv_mesh,
                                   const mesh::TerrainMesh& fwd_mesh,
                                   const mesh::MeshProjection& proj,
                                   Real sigma_background) {
    // Initialise parameters on the inversion mesh
    init(inv_mesh, sigma_background);

    // Store projection info
    proj_ = &proj;
    n_fwd_elements_ = fwd_mesh.num_elements();
    fwd_sigma_cache_.resize(n_fwd_elements_);
    cache_dirty_ = true;

    MAPLE3DMT_LOG_INFO("Dual-mesh model: " + std::to_string(n_elements_) +
                   " inv elements → " + std::to_string(n_fwd_elements_) +
                   " fwd elements (ratio " +
                   std::to_string(n_fwd_elements_ / std::max(n_elements_, 1)) + "x)");
}
#endif // MAPLE3DMT_USE_MFEM

Real ConductivityModel::sigma_fwd(int fwd_elem) const {
#ifdef MAPLE3DMT_USE_MFEM
    if (proj_) {
        // Dual-mesh mode: lazy projection
        if (cache_dirty_) {
            proj_->project_sigma(params_, fwd_sigma_cache_);
            cache_dirty_ = false;
        }
        if (param_type_ == Parameterisation::LOG_CONDUCTIVITY) {
            return std::exp(fwd_sigma_cache_[fwd_elem]);
        } else {
            return 1.0 / std::exp(fwd_sigma_cache_[fwd_elem]);
        }
    }
#endif
    // Single-mesh mode: fall back to sigma()
    return sigma(fwd_elem);
}

int ConductivityModel::num_fwd_elements() const {
#ifdef MAPLE3DMT_USE_MFEM
    return proj_ ? n_fwd_elements_ : n_elements_;
#else
    return n_elements_;
#endif
}

void ConductivityModel::save(const fs::path& path) const {
    // TODO: implement HDF5 save
    MAPLE3DMT_LOG_INFO("Saving model to " + path.string());
}

void ConductivityModel::load(const fs::path& path) {
    // TODO: implement HDF5 load
    MAPLE3DMT_LOG_INFO("Loading model from " + path.string());
}

#ifdef MAPLE3DMT_USE_MFEM
void ConductivityModel::init_quasi3d(const mesh::TerrainMesh& tmesh,
                                     Real sigma_background,
                                     int n_y_modes) {
    n_elements_ = tmesh.num_elements();
    n_y_modes_  = n_y_modes;

    Real log_val;
    if (param_type_ == Parameterisation::LOG_CONDUCTIVITY) {
        log_val = std::log(sigma_background);
    } else {
        log_val = std::log(1.0 / sigma_background);
    }

    // Mode 0 = background, modes 1..N = perturbations (zero in log-space)
    params_.assign(n_elements_ * n_y_modes_, 0.0);
    for (int i = 0; i < n_elements_; ++i) {
        params_[idx_(i, 0)] = log_val;
    }

    active_mask_.assign(n_elements_, true);

    MAPLE3DMT_LOG_INFO("Quasi-3D model initialised: " + std::to_string(n_elements_) +
                   " cells × " + std::to_string(n_y_modes_) + " y-modes");
}
#endif // MAPLE3DMT_USE_MFEM

Real ConductivityModel::sigma(int i, int k) const {
    if (param_type_ == Parameterisation::LOG_CONDUCTIVITY) {
        return std::exp(params_[idx_(i, k)]);
    } else {
        return 1.0 / std::exp(params_[idx_(i, k)]);
    }
}

} // namespace model
} // namespace maple3dmt
