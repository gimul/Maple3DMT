// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file regularization_octree.cpp
/// @brief Octree-based model regularization.

#include "maple3dmt/inversion/regularization_octree.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/octree/staggered_grid.h"
#include "maple3dmt/utils/logger.h"
#include <cmath>
#include <map>
#include <algorithm>

#define LOG_INFO(msg)  MAPLE3DMT_LOG_INFO(msg)

namespace maple3dmt {
namespace inversion {

// =========================================================================
// Setup: build WᵀW from octree face adjacency
// =========================================================================
void RegularizationOctree::setup(const octree::OctreeMesh& mesh,
                                   const RegParamsOctree& params) {
    mesh_ = &mesh;
    params_ = params;

    int nc = mesh.num_cells_local();

    // Build active/global mappings (Earth cells only)
    global_to_active_.assign(nc, -1);
    active_to_global_.clear();
    active_to_global_.reserve(nc);

    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) == octree::CellType::EARTH) {
            global_to_active_[c] = static_cast<int>(active_to_global_.size());
            active_to_global_.push_back(c);
        }
    }
    n_active_ = static_cast<int>(active_to_global_.size());

    LOG_INFO("RegularizationOctree: " + std::to_string(n_active_) +
             " active / " + std::to_string(nc) + " total cells");

    // Build adjacency list and WᵀW from face neighbors.
    // For each face between two Earth cells, add a smoothing weight:
    //   w = face_area / distance² × alpha_{dir}
    // WᵀW[i,j] = -w, WᵀW[i,i] += w

    adj_list_.resize(n_active_);
    diag_weights_.assign(n_active_, 0.0);

    // Use a map to accumulate CSR entries
    // WᵀW is symmetric, n_active × n_active
    std::vector<std::map<int, Real>> wtw_map(n_active_);

    // Precompute cell volumes for volume-normalized regularization.
    // ModEM-style: weight = alpha_dir × sqrt(V_i × V_j) / dist
    // This makes Φm independent of cell size (octree level), so that
    // alpha_s has a consistent meaning across different meshes.
    std::vector<Real> cell_vol(nc, 0.0);
    for (int c = 0; c < nc; ++c) {
        Real dx, dy, dz;
        mesh.cell_size_xyz(c, dx, dy, dz);
        cell_vol[c] = dx * dy * dz;
    }

    const auto& faces = mesh.staggered().faces();
    int nf = static_cast<int>(faces.size());

    for (int fi = 0; fi < nf; ++fi) {
        const auto& face = faces[fi];
        int cm = face.cell_minus;
        int cp = face.cell_plus;

        // Both cells must be valid and Earth
        if (cm < 0 || cp < 0) continue;
        if (cm >= nc || cp >= nc) continue;

        int am = global_to_active_[cm];
        int ap = global_to_active_[cp];
        if (am < 0 || ap < 0) continue;  // skip air cells

        // Direction-dependent weight
        Real alpha_dir;
        switch (face.normal) {
            case 0: alpha_dir = params_.alpha_x; break;
            case 1: alpha_dir = params_.alpha_y; break;
            case 2: alpha_dir = params_.alpha_z; break;
            default: alpha_dir = 1.0;
        }

        // Distance between cell centers
        Real cx1, cy1, cz1, cx2, cy2, cz2;
        mesh.cell_center(cm, cx1, cy1, cz1);
        mesh.cell_center(cp, cx2, cy2, cz2);
        Real dist = std::sqrt((cx1-cx2)*(cx1-cx2) + (cy1-cy2)*(cy1-cy2) +
                               (cz1-cz2)*(cz1-cz2));
        if (dist < 1e-10) continue;

        // Volume-normalized smoothing weight (ModEM-style):
        //
        // Standard FV discretization of ||∇m||²:
        //   w_raw = face_area / dist
        // This is mesh-size dependent: bigger cells → bigger w.
        //
        // To make Φm mesh-independent, we divide by the geometric mean
        // volume of the two cells, making w dimensionless:
        //   w = alpha_dir × face_area / dist / V_geomean^{1/3}
        //
        // Then for a uniform mesh with cell size h: face_area=h², dist=h,
        // V^{1/3}=h → w = h²/h/h = 1 (dimensionless, independent of h).
        // For non-uniform octree: small faces get small w, large faces
        // get large w, but scaled by their cell size → balanced.
        Real V_gm = std::pow(cell_vol[cm] * cell_vol[cp], Real(1.0/6.0)); // (V1*V2)^(1/6) = geom-mean V^(1/3)
        Real w = alpha_dir * face.area / (dist * V_gm);

        // WᵀW entries: diagonal += w, off-diagonal = -w
        wtw_map[am][am] += w;
        wtw_map[am][ap] -= w;
        wtw_map[ap][ap] += w;
        wtw_map[ap][am] -= w;

        // Adjacency list for CmCmᵀ
        adj_list_[am].push_back({ap, w});
        adj_list_[ap].push_back({am, w});
        diag_weights_[am] += w;
        diag_weights_[ap] += w;
    }

    // Convert map to CSR
    WtW_.nrows = n_active_;
    WtW_.ncols = n_active_;
    WtW_.rowptr.resize(n_active_ + 1);
    WtW_.colidx.clear();
    WtW_.values.clear();

    int nnz = 0;
    for (int i = 0; i < n_active_; ++i)
        nnz += static_cast<int>(wtw_map[i].size());
    WtW_.colidx.reserve(nnz);
    WtW_.values.reserve(nnz);

    WtW_.rowptr[0] = 0;
    for (int i = 0; i < n_active_; ++i) {
        for (const auto& [col, val] : wtw_map[i]) {
            WtW_.colidx.push_back(col);
            WtW_.values.push_back(val);
        }
        WtW_.rowptr[i + 1] = static_cast<int>(WtW_.colidx.size());
    }

    // Normalize WtW so that the average diagonal = 1.0.
    // This makes alpha_s = 1.0 a universal "unit strength" regardless
    // of domain size, mesh level, or number of cells.
    // Φm(dm=1) ≈ alpha_s × n_active (interpretable: per-cell penalty).
    Real diag_sum = 0.0;
    for (int i = 0; i < n_active_; ++i)
        diag_sum += diag_weights_[i];
    Real diag_avg = (n_active_ > 0) ? diag_sum / n_active_ : 1.0;
    if (diag_avg > 1e-30) {
        Real scale = 1.0 / diag_avg;
        for (auto& v : WtW_.values) v *= scale;
        for (auto& d : diag_weights_) d *= scale;
        for (auto& adj : adj_list_)
            for (auto& e : adj) e.weight *= scale;
        LOG_INFO("RegularizationOctree: WᵀW normalized (avg_diag=" +
                 std::to_string(diag_avg) + " → 1.0, scale=" +
                 std::to_string(scale) + ")");
    }

    LOG_INFO("RegularizationOctree: WᵀW nnz=" + std::to_string(WtW_.nnz()) +
             " (avg " + std::to_string(n_active_ > 0 ? WtW_.nnz() / n_active_ : 0) +
             " per row)");

    // Initialize empty reference model
    ref_params_.assign(n_active_, 0.0);
}

// =========================================================================
// Set reference model
// =========================================================================
void RegularizationOctree::set_reference_model(const RealVec& ref_log_sigma) {
    ref_params_ = ref_log_sigma;
}

// =========================================================================
// Evaluate: R(m) = α_s · (m - m_ref)ᵀ WᵀW (m - m_ref)
// =========================================================================
Real RegularizationOctree::evaluate(const model::ConductivityModel& m) const {
    // Map model params to active space
    RealVec dm(n_active_);
    for (int j = 0; j < n_active_; ++j) {
        dm[j] = m.params()[active_to_global_[j]] - ref_params_[j];
    }

    // Compute dm^T WtW dm
    RealVec Wdm;
    WtW_.matvec(dm, Wdm);

    Real result = 0.0;
    for (int j = 0; j < n_active_; ++j)
        result += dm[j] * Wdm[j];

    return params_.alpha_s * result;
}

// =========================================================================
// Gradient: ∇R = 2 α_s WᵀW (m - m_ref)   (global space)
// =========================================================================
void RegularizationOctree::gradient(const model::ConductivityModel& m,
                                      RealVec& grad) const {
    int nc = static_cast<int>(global_to_active_.size());
    grad.assign(nc, 0.0);

    // Active-space difference
    RealVec dm(n_active_);
    for (int j = 0; j < n_active_; ++j) {
        dm[j] = m.params()[active_to_global_[j]] - ref_params_[j];
    }

    // WtW * dm
    RealVec Wdm;
    WtW_.matvec(dm, Wdm);

    // Map back to global
    for (int j = 0; j < n_active_; ++j) {
        grad[active_to_global_[j]] = 2.0 * params_.alpha_s * Wdm[j];
    }
}

// =========================================================================
// Apply WᵀW (active space)
// =========================================================================
void RegularizationOctree::apply_WtW(const RealVec& x, RealVec& result) const {
    WtW_.matvec(x, result);
}

// =========================================================================
// Diagonal of WᵀW (active space)
// =========================================================================
void RegularizationOctree::diagonal_WtW(RealVec& diag) const {
    diag.assign(n_active_, 0.0);
    for (int i = 0; i < n_active_; ++i) {
        for (int k = WtW_.rowptr[i]; k < WtW_.rowptr[i + 1]; ++k) {
            if (WtW_.colidx[k] == i) {
                diag[i] = WtW_.values[k];
                break;
            }
        }
    }
}

// =========================================================================
// CmCmᵀ smoothing (ModEM-style diffusion)
// =========================================================================
void RegularizationOctree::apply_CmCmT(const RealVec& x, RealVec& result) const {
    // Diffusion smoother: each pass averages a cell with its neighbors.
    // n_passes = 2 * n_smooth (CmCmᵀ = Cm applied twice).
    //
    // For each pass:
    //   y[i] = (x[i] + Σ_j w_ij * x[j]) / (1 + Σ_j w_ij)
    //        = (x[i] + Σ_j w_ij * x[j]) / (1 + diag_weights[i])

    int n_passes = 2 * params_.n_smooth;
    result = x;

    RealVec temp(n_active_);
    for (int pass = 0; pass < n_passes; ++pass) {
        for (int i = 0; i < n_active_; ++i) {
            Real sum = result[i];
            Real w_total = 1.0;
            for (const auto& nbr : adj_list_[i]) {
                // Normalize weight to be relative
                Real w_norm = nbr.weight / (diag_weights_[i] > 0 ? diag_weights_[i] : 1.0);
                sum += w_norm * result[nbr.active_idx];
                w_total += w_norm;
            }
            temp[i] = sum / w_total;
        }
        result = temp;
    }
}

} // namespace inversion
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
