// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file staggered_grid.h
/// @brief Yee staggered grid on octree: edge/face/cell DOF indexing.

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include <array>

namespace maple3dmt {
namespace octree {

class OctreeMesh;

/// Edge DOF info (E-field lives here).
struct EdgeInfo {
    Real x, y, z;          // edge midpoint
    int  direction;        // 0=x, 1=y, 2=z
    Real length;           // edge length (m)
    // Adjacent cells (up to 4 for interior edge, fewer at boundaries)
    std::vector<int> adj_cells;
};

/// Face DOF info (H-field lives here).
struct FaceInfo {
    Real x, y, z;          // face center
    int  normal;           // normal direction: 0=x, 1=y, 2=z
    Real area;             // face area (m^2)
    int  cell_minus;       // cell on negative side of normal (-1 if boundary)
    int  cell_plus;        // cell on positive side of normal (-1 if boundary)
    std::vector<int> edges; // edge IDs forming this face (4 for regular, more for hanging)
};

/// Hanging face: coarse face maps to 4 fine sub-faces.
struct HangingFace {
    int coarse_face;
    std::array<int, 4> fine_faces;
};

/// Hanging edge: coarse edge maps to 2 fine sub-edges.
struct HangingEdge {
    int coarse_edge;
    std::array<int, 2> fine_edges;
};

/// Yee staggered grid built on an OctreeMesh.
///
/// DOF placement:
///   - E (electric field): edge midpoints
///   - H (magnetic field): face centers
///   - sigma (conductivity): cell centers
class StaggeredGrid {
public:
    StaggeredGrid() = default;

    /// Build edge/face indexing from mesh adjacency.
    void build(const OctreeMesh& mesh);

    // --- DOF counts ---
    int num_edges() const { return static_cast<int>(edges_.size()); }
    int num_faces() const { return static_cast<int>(faces_.size()); }
    int num_cells() const { return num_cells_; }

    // --- Access ---
    const EdgeInfo& edge(int i) const { return edges_[i]; }
    const FaceInfo& face(int i) const { return faces_[i]; }

    const std::vector<EdgeInfo>& edges() const { return edges_; }
    const std::vector<FaceInfo>& faces() const { return faces_; }

    // --- Hanging entities ---
    const std::vector<HangingFace>& hanging_faces() const { return hanging_faces_; }
    const std::vector<HangingEdge>& hanging_edges() const { return hanging_edges_; }

    // --- Cell-to-edge mapping (for sensitivity/mass assembly) ---
    /// For cell i, list of (edge_id, edge_volume_fraction) pairs.
    struct CellEdge {
        int edge_id;
        Real volume_fraction;  // fraction of edge volume attributed to this cell
    };
    const std::vector<std::vector<CellEdge>>& cell_edges() const { return cell_edges_; }

private:
    std::vector<EdgeInfo> edges_;
    std::vector<FaceInfo> faces_;
    int num_cells_ = 0;

    std::vector<HangingFace> hanging_faces_;
    std::vector<HangingEdge> hanging_edges_;

    std::vector<std::vector<CellEdge>> cell_edges_;

    void enumerate_edges_(const OctreeMesh& mesh);
    void enumerate_faces_(const OctreeMesh& mesh);
    void detect_hanging_(const OctreeMesh& mesh);
    void build_cell_edge_map_(const OctreeMesh& mesh);
};

} // namespace octree
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
