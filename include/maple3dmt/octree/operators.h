// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file operators.h
/// @brief Discrete differential operators (curl, grad, div) on octree staggered grid.

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

namespace maple3dmt {
namespace octree {

class StaggeredGrid;

/// Discrete differential operators on the Yee staggered grid.
///
/// Operator dimensions:
///   Ce:  num_faces  x num_edges   (discrete curl: edges → faces)
///   Cf:  num_edges  x num_faces   (discrete curl transpose, scaled)
///   G:   num_edges  x num_nodes   (discrete gradient)
///   D:   num_cells  x num_faces   (discrete divergence)
///
/// The system matrix A = Cf * Mf_inv * Ce + iω * Me(σ)
/// where Mf_inv and Me are diagonal mass matrices.
class DiscreteOperators {
public:
    DiscreteOperators() = default;

    /// Build all operators from the staggered grid.
    void build(const StaggeredGrid& grid);

    /// Number of DOFs.
    int num_edges() const { return num_edges_; }
    int num_faces() const { return num_faces_; }
    int num_cells() const { return num_cells_; }
    int num_nodes() const { return num_nodes_; }
    Real edge_dual_area(int e) const { return edge_dual_area_[e]; }

    // --- Discrete curl: C * E_edges → B_faces ---
    // C[f][e] = ±1 (incidence, orientation-aware)
    const SparseMatR& curl_e2f() const { return Ce_; }

    // --- Discrete curl transpose (topological): C^T ---
    const SparseMatR& curl_f2e() const { return Cf_; }

    // --- Discrete gradient: G * φ_nodes → E_edges ---
    const SparseMatR& gradient() const { return G_; }

    // --- Node-to-edge gradient for DivCorr ---
    const SparseMatR& gradient_node() const { return G_node_; }

    // --- Discrete divergence: D * B_faces → div_cells ---
    const SparseMatR& divergence() const { return D_; }

    // --- Mass matrices (diagonal) ---

    /// Build edge mass: Me[e] = σ_avg(e) * edge_volume(e)
    /// σ_avg = harmonic mean of σ in cells adjacent to edge e.
    void build_edge_mass(const RealVec& sigma, DiagMatR& Me) const;

    /// Build face mass: Mf[f] = (1/μ₀) * face_volume(f)
    void build_face_mass(DiagMatR& Mf) const;

    /// Build face mass inverse: Mf_inv[f] = μ₀ / face_volume(f)
    void build_face_mass_inv(DiagMatR& Mf_inv) const;

    // --- System matrix assembly ---

    /// Assemble A = C^T * Mf_inv * C + iω * Me(σ)
    /// Returns complex sparse CSR matrix.
    void assemble_system(Real omega, const RealVec& sigma,
                         SparseMatC& A) const;

    /// Compute sparsity pattern of C^T * Mf_inv * C (once, reuse for all ω).
    void compute_stiffness_pattern();

    // --- Divergence correction helpers ---

    /// Build node-based Laplacian for DivCorr: L = G^T * Me_σ * G (nodes × nodes).
    /// SPD matrix. φ=0 Dirichlet on domain boundary nodes.
    void build_div_laplacian(const RealVec& sigma,
                             SparseMatR& L) const;

    /// Compute div(σE) on nodes: divJ = G^T * Me_σ * E (consistent with G_node_).
    void compute_div_sigma_E(const ComplexVec& E, const RealVec& sigma,
                             ComplexVec& divJ) const;

    /// Apply node-to-edge gradient: grad = G_node_ * φ.
    /// Guarantees curl(grad) = 0 (discrete de Rham identity).
    void apply_cell_gradient(const ComplexVec& phi, ComplexVec& grad) const;

private:
    int num_edges_ = 0;
    int num_faces_ = 0;
    int num_cells_ = 0;

    SparseMatR Ce_;  // curl: faces x edges
    SparseMatR Cf_;  // curl transpose: edges x faces
    SparseMatR G_;   // gradient: edges x nodes
    SparseMatR D_;   // divergence: cells x faces

    // Cached stiffness sparsity pattern (for system assembly)
    std::vector<int> stiff_rowptr_, stiff_colidx_;
    bool stiffness_pattern_built_ = false;

    // Staggered grid reference data (copied for mass assembly)
    std::vector<Real> edge_lengths_;
    std::vector<Real> face_areas_;
    std::vector<Real> cell_volumes_;

    // Per-cell half-extents (hx, hy, hz) for correct Hodge star computation
    // Cell physical dimensions: (2*hx) × (2*hy) × (2*hz)
    std::vector<Real> cell_hx_, cell_hy_, cell_hz_;

    // Face dual edge length: L_dual[f] = distance between adjacent cell centers
    // in the face-normal direction. Used for Mf = μ₀ * A / L_dual.
    std::vector<Real> face_dual_length_;

    // Edge dual face area: A_dual[e] = area of dual face perpendicular to edge.
    // Used for Me = σ * A_dual / L. For integral DOFs with Ce = ±1 (incidence).
    std::vector<Real> edge_dual_area_;

    // Edge-to-cell adjacency for σ averaging
    std::vector<std::vector<int>> edge_adj_cells_;

    // Cell center coordinates (for gradient computation)
    std::vector<Real> cell_cx_, cell_cy_, cell_cz_;

    // Face connectivity: face_normal_[f], face_cell_minus_[f], face_cell_plus_[f]
    std::vector<int> face_normal_;
    std::vector<int> face_cell_minus_, face_cell_plus_;

    // Edge direction and position (for gradient: which cells an edge connects)
    std::vector<int> edge_direction_;
    std::vector<Real> edge_x_, edge_y_, edge_z_;

    // Node-to-edge gradient matrix: G_node_ (num_edges x num_nodes)
    // G_node_[e, v1] = -1/L, G_node_[e, v2] = +1/L
    // Satisfies curl(G_node * φ) = 0 (discrete de Rham identity).
    // Used for DivCorr: guarantees correction doesn't change curl(E).
    SparseMatR G_node_;
    int num_nodes_ = 0;
    std::vector<Real> node_x_, node_y_, node_z_;  // node coordinates
    std::vector<int> edge_node1_, edge_node2_;     // edge → two endpoint nodes

    // Hanging face tracking: coarse faces replaced by fine sub-faces
    std::vector<bool> dead_faces_;

    // Hanging edge tracking: coarse edges fully replaced by fine children
    std::vector<bool> dead_edges_;
    // Hanging edge constraint pairs: coarse → (child1, child2)
    struct HangingEdgePair {
        int coarse, child1, child2;
    };
    std::vector<HangingEdgePair> hanging_pairs_;
public:
    /// Returns true if face f is a dead coarse face at a hanging boundary
    bool is_dead_face(int f) const {
        return f >= 0 && f < static_cast<int>(dead_faces_.size()) && dead_faces_[f];
    }
    /// Returns true if edge e is a dead coarse edge at a hanging face
    bool is_dead_edge(int e) const {
        return e >= 0 && e < static_cast<int>(dead_edges_.size()) && dead_edges_[e];
    }
    /// Hanging edge constraint pairs
    const std::vector<HangingEdgePair>& hanging_pairs() const { return hanging_pairs_; }
private:
    void build_curl_(const StaggeredGrid& grid);
    void build_gradient_(const StaggeredGrid& grid);
    void build_divergence_(const StaggeredGrid& grid);
    void build_node_gradient_matrix_();
    void handle_hanging_faces_(const StaggeredGrid& grid);
};

} // namespace octree
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
