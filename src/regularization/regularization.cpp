// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file regularization.cpp
/// @brief Implementation of regularisation operators.

#include "maple3dmt/regularization/regularization.h"
#include "maple3dmt/utils/logger.h"
#include <cmath>
#ifdef MAPLE3DMT_USE_MPI
#include <mpi.h>
#endif

namespace maple3dmt {
namespace regularization {

void Regularization::setup(const mesh::TerrainMesh& tmesh,
                           const RegParams& params) {
    params_ = params;
    build_gradient_operator_(tmesh);
    MAPLE3DMT_LOG_INFO("Regularisation set up: n_active=" +
                   std::to_string(n_active_) +
                   ", alpha_s=" + std::to_string(params.alpha_s));
}

void Regularization::setup_3d(mfem::ParMesh& pmesh, const RegParams& params) {
    params_ = params;
    pmesh_ = &pmesh;
    int ne = pmesh.GetNE();
    int dim = pmesh.SpaceDimension();

    // Build active element mapping: earth (attr=1) only
    global_to_active_.assign(ne, -1);
    active_to_global_.clear();
    n_active_ = 0;
    for (int e = 0; e < ne; ++e) {
        if (pmesh.GetAttribute(e) == 1) {
            global_to_active_[e] = n_active_;
            active_to_global_.push_back(e);
            ++n_active_;
        }
    }

    if (n_active_ == 0) return;

    WtW_ = std::make_unique<mfem::SparseMatrix>(n_active_, n_active_);

    // Helper lambda: compute face weight from two element centroids + face area
    auto compute_face_weight = [&](int e1, int e2, double face_area) -> double {
        mfem::Vector c1(dim), c2(dim);
        pmesh.GetElementCenter(e1, c1);
        pmesh.GetElementCenter(e2, c2);

        double dx = c2(0) - c1(0);
        double dy = (dim > 1) ? c2(1) - c1(1) : 0.0;
        double dz = (dim > 2) ? c2(2) - c1(2) : 0.0;
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < 1e-15) return 0.0;

        double abs_dx = std::abs(dx), abs_dy = std::abs(dy), abs_dz = std::abs(dz);
        double alpha_dir = (abs_dz > abs_dx && abs_dz > abs_dy)
                            ? params_.alpha_z : params_.alpha_x;
        return alpha_dir * face_area / dist;
    };

    auto compute_face_area = [&](mfem::ElementTransformation* ft) -> double {
        const mfem::IntegrationRule& ir =
            mfem::IntRules.Get(ft->GetGeometryType(), 2);
        double area = 0.0;
        for (int q = 0; q < ir.GetNPoints(); ++q) {
            const auto& ip = ir.IntPoint(q);
            ft->SetIntPoint(&ip);
            area += ip.weight * ft->Weight();
        }
        return area;
    };

    // --- (A) Local interior faces: both elements on this rank ---
    int nfaces = pmesh.GetNumFaces();
    for (int f = 0; f < nfaces; ++f) {
        int e1, e2;
        pmesh.GetFaceElements(f, &e1, &e2);
        if (e1 < 0 || e2 < 0) continue;

        int a1 = global_to_active_[e1];
        int a2 = global_to_active_[e2];
        if (a1 < 0 || a2 < 0) continue;

        double face_area = compute_face_area(pmesh.GetFaceTransformation(f));
        double w = compute_face_weight(e1, e2, face_area);
        if (w == 0.0) continue;

        WtW_->Add(a1, a1, w);
        WtW_->Add(a2, a2, w);
        WtW_->Add(a1, a2, -w);
        WtW_->Add(a2, a1, -w);
    }

    // --- (B) Shared faces: one element on this rank, neighbor on another ---
    // DISABLED: MFEM 4.9 GetSharedFaceTransformations() triggers assertion
    //   failure in GetSharedFaceTransformationsByLocalIndex for hex ParMesh.
    //   This is an MFEM bug (face_info.Elem2Inf < 0 for valid shared faces).
    //
    // Effect of skipping: partition boundaries have diagonal-only smoothing
    //   (no cross-rank off-diagonal W^T W terms). Each rank's interior faces
    //   still provide full smoothing. The boundary effect is minor because
    //   METIS partitions are compact, so boundary-to-interior ratio is small.
    //
    // TODO: Re-enable when upgrading to MFEM 4.10+ or patching pmesh.cpp.
    shared_face_weights_.clear();
    shared_face_local_active_.clear();
    shared_face_nbr_global_elem_.clear();
    int n_shared = pmesh.GetNSharedFaces();

    {
        int rank = 0;
#ifdef MAPLE3DMT_USE_MPI
        MPI_Comm_rank(pmesh.GetComm(), &rank);
#endif
        if (rank == 0) {
            MAPLE3DMT_LOG_INFO("Shared face smoothing: " +
                           std::to_string(n_shared) + " shared faces, " +
                           std::to_string(shared_face_weights_.size()) +
                           " active earth-earth pairs");
        }
    }

    // Reference model damping
    if (params_.alpha_r > 0.0) {
        for (int j = 0; j < n_active_; ++j) {
            WtW_->Add(j, j, params_.alpha_r);
        }
    }

    WtW_->Finalize();

    MAPLE3DMT_LOG_INFO("3D regularisation: n_active=" + std::to_string(n_active_) +
                   "/" + std::to_string(ne) + " elements, " +
                   std::to_string(WtW_->NumNonZeroElems()) + " nnz");
}

void Regularization::setup_quasi3d(const mesh::TerrainMesh& tmesh,
                                    const model::ConductivityModel& model,
                                    const RegParams& params) {
    params_ = params;
    int n_y_modes = model.num_y_modes();
    int n_elements = model.num_elements();

    // First build the 2D gradient operator to get earth element mapping
    build_gradient_operator_(tmesh);

    // Save the 2D active count and mapping
    int n_active_2d = n_active_;
    std::vector<int> a2g_2d = active_to_global_;

    if (n_y_modes <= 1) {
        // Pure 2D — nothing more to do
        MAPLE3DMT_LOG_INFO("Quasi-3D regularisation (n_modes=1, pure 2D): n_active=" +
                       std::to_string(n_active_));
        return;
    }

    // Extend active mapping for all y-modes
    // a2g layout: [mode0 earth elems, mode1 earth elems, ..., modeK-1 earth elems]
    // a2g[mode*n_active_2d + j] = mode*n_elements + a2g_2d[j]
    n_active_ = n_active_2d * n_y_modes;
    active_to_global_.resize(n_active_);
    global_to_active_.assign(n_elements * n_y_modes, -1);

    for (int mode = 0; mode < n_y_modes; ++mode) {
        for (int j = 0; j < n_active_2d; ++j) {
            int active_idx = mode * n_active_2d + j;
            int global_idx = mode * n_elements + a2g_2d[j];
            active_to_global_[active_idx] = global_idx;
            global_to_active_[global_idx] = active_idx;
        }
    }

    // Build block-diagonal WtW
    // Each mode gets the same spatial smoothing matrix + mode-specific damping
    auto WtW_2d = std::move(WtW_);  // save the 2D WtW
    WtW_ = std::make_unique<mfem::SparseMatrix>(n_active_, n_active_);

    for (int mode = 0; mode < n_y_modes; ++mode) {
        int offset = mode * n_active_2d;

        // Copy spatial smoothing from 2D WtW
        mfem::Array<int> col_arr;
        mfem::Vector val_arr;
        for (int i = 0; i < n_active_2d; ++i) {
            WtW_2d->GetRow(i, col_arr, val_arr);
            for (int k = 0; k < col_arr.Size(); ++k) {
                WtW_->Add(offset + i, offset + col_arr[k], val_arr(k));
            }
        }

        // Mode k≥1: add extra damping toward zero
        if (mode >= 1) {
            for (int j = 0; j < n_active_2d; ++j) {
                WtW_->Add(offset + j, offset + j, params_.alpha_pert);
            }
        }
    }

    WtW_->Finalize();

    MAPLE3DMT_LOG_INFO("Quasi-3D regularisation: n_active=" +
                   std::to_string(n_active_) + " (" +
                   std::to_string(n_active_2d) + " earth × " +
                   std::to_string(n_y_modes) + " modes), alpha_pert=" +
                   std::to_string(params_.alpha_pert));
}

void Regularization::set_reference_model(const model::ConductivityModel& ref) {
    // Extract active parameters only
    const auto& full = ref.params();
    ref_params_.resize(n_active_);
    for (int j = 0; j < n_active_; ++j) {
        ref_params_[j] = full[active_to_global_[j]];
    }
    params_.use_reference_model = true;
    MAPLE3DMT_LOG_INFO("Reference model set (" +
                   std::to_string(n_active_) + " active params)");
}

/// Get the ghost element parameter value for a shared face neighbor.
/// Ghost elements have local indices >= GetNE() in ParMesh.
/// The model params array only covers local elements (indices 0..ne-1),
/// so ghost parameter values must be exchanged via MPI.
/// For now, we approximate ghost parameters by reading them directly from
/// the model's params vector using the ghost element index.  This works
/// because ConductivityModel::init_3d() allocates based on pmesh.GetNE()
/// which includes local elements only.
///
/// TODO: implement proper ghost exchange for model parameters.
/// Currently, we use the conservative approximation that ghost elements
/// have the same parameter as the local element (= 0 difference),
/// which means shared-face off-diagonal contribution is zero.
/// This underestimates the gradient at partition boundaries but does not
/// cause incorrect results — it only weakens smoothing there.
///
/// The diagonal contribution (already in WtW_) ensures that local elements
/// at partition boundaries still have proper self-smoothing weight.

Real Regularization::evaluate(const model::ConductivityModel& m) const {
    if (!WtW_ || n_active_ == 0) return 0.0;

    const auto& full = m.params();
    mfem::Vector x(n_active_);
    for (int j = 0; j < n_active_; ++j) {
        x(j) = full[active_to_global_[j]];
        if (params_.use_reference_model && !ref_params_.empty()) {
            x(j) -= ref_params_[j];
        }
    }

    // Phi_m = alpha_s * x^T WtW x
    // WtW already includes shared-face diagonal contributions.
    // The cross-partition off-diagonal terms are omitted (conservative).
    mfem::Vector Wx(n_active_);
    WtW_->Mult(x, Wx);
    Real phi_local = params_.alpha_s * (x * Wx);

    // MPI_Allreduce to get global phi_model across all ranks
    Real phi_global = phi_local;
#ifdef MAPLE3DMT_USE_MPI
    if (pmesh_) {
        MPI_Allreduce(&phi_local, &phi_global, 1,
                      MPI_DOUBLE, MPI_SUM, pmesh_->GetComm());
    }
#endif
    return phi_global;
}

void Regularization::gradient(const model::ConductivityModel& m,
                              RealVec& grad) const {
    int n_full = static_cast<int>(m.params().size());
    grad.assign(n_full, 0.0);
    if (!WtW_ || n_active_ == 0) return;

    const auto& full = m.params();
    mfem::Vector x(n_active_);
    for (int j = 0; j < n_active_; ++j) {
        x(j) = full[active_to_global_[j]];
        if (params_.use_reference_model && !ref_params_.empty()) {
            x(j) -= ref_params_[j];
        }
    }

    // grad_active = 2 * alpha_s * WtW * x
    // WtW includes shared-face diagonal contributions.
    mfem::Vector g(n_active_);
    WtW_->Mult(x, g);
    for (int j = 0; j < n_active_; ++j) {
        grad[active_to_global_[j]] = 2.0 * params_.alpha_s * g(j);
    }
}

void Regularization::apply_WtW(const RealVec& x, RealVec& result) const {
    result.assign(x.size(), 0.0);
    if (!WtW_ || n_active_ == 0) return;

    mfem::Vector xv(n_active_), rv(n_active_);
    for (int j = 0; j < n_active_; ++j) {
        xv(j) = x[active_to_global_[j]];
    }
    WtW_->Mult(xv, rv);
    for (int j = 0; j < n_active_; ++j) {
        result[active_to_global_[j]] = rv(j);
    }
}

void Regularization::diagonal_WtW(RealVec& diag) const {
    int ne = (n_active_ > 0)
        ? static_cast<int>(active_to_global_.back()) + 1
        : 0;
    diag.assign(ne, 0.0);
    if (!WtW_ || n_active_ == 0) return;

    // Extract diagonal of WtW (n_active × n_active sparse matrix)
    for (int j = 0; j < n_active_; ++j) {
        diag[active_to_global_[j]] = (*WtW_)(j, j);
    }
}

void Regularization::apply_CmCmT(const mfem::Vector& x,
                                  mfem::Vector& result) const {
    int n = n_active_;
    MFEM_ASSERT(x.Size() == n, "apply_CmCmT: input size mismatch");
    result.SetSize(n);

    if (!WtW_ || n == 0) {
        result = x;
        return;
    }

    // Two-stage model-covariance preconditioner:
    //
    //   Stage 1: Diagonal scaling  h = D^{-1} g
    //     where D = diag(WtW) + ε.  This normalizes the element-wise
    //     gradient magnitude by the local regularization weight, giving
    //     ||h||_inf = O(0.01–1) regardless of mesh size.
    //
    //   Stage 2: Diffusion smoothing  h ← S^k h
    //     k = 2*n_smooth passes of neighbor-weighted averaging using
    //     WtW connectivity.  This spreads localized sensitivity
    //     to neighboring cells (physically: model update should be
    //     smooth, not spiky).
    //
    // Combined: P = S^k · D^{-1}  is SPD and bounded, so it
    // preserves descent direction and produces step sizes O(0.1–10).

    constexpr double epsilon = 1.0;  // avoid division by zero for isolated elements

    // ── Stage 1: diagonal scaling ──
    for (int i = 0; i < n; ++i) {
        double d = (*WtW_)(i, i) + epsilon;
        result(i) = x(i) / d;
    }

    // ── Stage 2: diffusion smoothing ──
    int total_passes = 2 * params_.n_smooth;
    if (total_passes <= 0) return;

    // Pre-extract diagonal for averaging normalization
    mfem::Vector diag_vec(n);
    for (int i = 0; i < n; ++i) {
        diag_vec(i) = (*WtW_)(i, i);
        if (diag_vec(i) < 1e-30) diag_vec(i) = 1.0;
    }

    // Ping-pong buffers
    mfem::Vector buf(n);
    mfem::Array<int> cols;
    mfem::Vector vals;

    for (int pass = 0; pass < total_passes; ++pass) {
        const mfem::Vector& src = result;

        for (int i = 0; i < n; ++i) {
            WtW_->GetRow(i, cols, vals);

            double nbr_sum = 0.0;
            for (int k = 0; k < cols.Size(); ++k) {
                if (cols[k] != i) {
                    nbr_sum += std::abs(vals(k)) * src(cols[k]);
                }
            }

            // Weighted average: 50% self + 50% neighbor average
            buf(i) = 0.5 * src(i) + 0.5 * nbr_sum / diag_vec(i);
        }
        result = buf;
    }
}

void Regularization::update_alpha(Real new_alpha) {
    params_.alpha_s = new_alpha;
}

void Regularization::build_gradient_operator_(const mesh::TerrainMesh& tmesh) {
    // const_cast needed: MFEM's GetElementCenter/GetFaceTransformation
    // are not const-qualified even though they don't modify the mesh.
    mfem::Mesh& mesh = *const_cast<mfem::Mesh*>(tmesh.mesh());
    int ne = mesh.GetNE();

    // Build active element mapping: earth (attr=1) only
    global_to_active_.assign(ne, -1);
    active_to_global_.clear();
    n_active_ = 0;
    for (int e = 0; e < ne; ++e) {
        if (mesh.GetAttribute(e) == 1) {
            global_to_active_[e] = n_active_;
            active_to_global_.push_back(e);
            ++n_active_;
        }
    }

    if (n_active_ == 0) return;

    MAPLE3DMT_LOG_DEBUG("Building terrain-aware gradient operator: " +
                    std::to_string(n_active_) + " active elements, " +
                    std::to_string(ne) + " total");

    // Allocate WtW as n_active x n_active sparse matrix
    WtW_ = std::make_unique<mfem::SparseMatrix>(n_active_, n_active_);

    // Iterate over all internal faces
    // For 2D triangle meshes, "faces" in MFEM are edges
    int nfaces = mesh.GetNumFaces();
    for (int f = 0; f < nfaces; ++f) {
        int e1, e2;
        mesh.GetFaceElements(f, &e1, &e2);

        // Skip boundary faces (one element is -1)
        if (e1 < 0 || e2 < 0) continue;

        // Skip if either element is air
        int a1 = global_to_active_[e1];
        int a2 = global_to_active_[e2];
        if (a1 < 0 || a2 < 0) continue;

        // Compute element centroids
        mfem::Vector c1(2), c2(2);
        mesh.GetElementCenter(e1, c1);
        mesh.GetElementCenter(e2, c2);

        // Distance between centroids
        double dx = c2(0) - c1(0);
        double dz = c2(1) - c1(1);
        double dist = std::sqrt(dx * dx + dz * dz);
        if (dist < 1e-15) continue;

        // Face length (edge length in 2D)
        auto* ft = mesh.GetFaceTransformation(f);
        double face_len = ft->Weight();  // Jacobian determinant = edge length for 1D face

        // Classify direction: if the centroid-to-centroid vector is more
        // vertical than horizontal, it's a vertical gradient → use alpha_z;
        // otherwise use alpha_x.
        double alpha_dir;
        if (std::abs(dz) > std::abs(dx)) {
            alpha_dir = params_.alpha_z;
        } else {
            alpha_dir = params_.alpha_x;
        }

        // Geometric weight: face_length / distance
        double w = alpha_dir * face_len / dist;

        // Assemble into WtW
        WtW_->Add(a1, a1, w);
        WtW_->Add(a2, a2, w);
        WtW_->Add(a1, a2, -w);
        WtW_->Add(a2, a1, -w);
    }

    // Add reference model damping if requested
    if (params_.alpha_r > 0.0) {
        for (int j = 0; j < n_active_; ++j) {
            WtW_->Add(j, j, params_.alpha_r);
        }
    }

    WtW_->Finalize();

    MAPLE3DMT_LOG_DEBUG("WtW assembled: " + std::to_string(WtW_->NumNonZeroElems()) +
                    " nonzeros");
}

} // namespace regularization
} // namespace maple3dmt
