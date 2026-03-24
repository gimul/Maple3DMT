// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#include "maple3dmt/octree/operators.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/octree/staggered_grid.h"
#include "maple3dmt/utils/logger.h"

#define LOG_INFO(msg) MAPLE3DMT_LOG_INFO(msg)
#include <algorithm>
#include <map>
#include <cmath>

namespace maple3dmt {
namespace octree {

void DiscreteOperators::build(const StaggeredGrid& grid) {
    num_edges_ = grid.num_edges();
    num_faces_ = grid.num_faces();
    num_cells_ = grid.num_cells();

    LOG_INFO("DiscreteOperators::build — edges=" + std::to_string(num_edges_) +
             " faces=" + std::to_string(num_faces_) +
             " cells=" + std::to_string(num_cells_));

    // Cache geometric data
    edge_lengths_.resize(num_edges_);
    edge_direction_.resize(num_edges_);
    edge_x_.resize(num_edges_);
    edge_y_.resize(num_edges_);
    edge_z_.resize(num_edges_);
    for (int e = 0; e < num_edges_; ++e) {
        const auto& ei = grid.edge(e);
        edge_lengths_[e] = ei.length;
        edge_direction_[e] = ei.direction;
        edge_x_[e] = ei.x;
        edge_y_[e] = ei.y;
        edge_z_[e] = ei.z;
    }

    face_areas_.resize(num_faces_);
    face_normal_.resize(num_faces_);
    face_cell_minus_.resize(num_faces_);
    face_cell_plus_.resize(num_faces_);
    for (int f = 0; f < num_faces_; ++f) {
        const auto& fi = grid.face(f);
        face_areas_[f] = fi.area;
        face_normal_[f] = fi.normal;
        face_cell_minus_[f] = fi.cell_minus;
        face_cell_plus_[f] = fi.cell_plus;
    }

    // Edge adjacency
    edge_adj_cells_.resize(num_edges_);
    for (int e = 0; e < num_edges_; ++e)
        edge_adj_cells_[e] = grid.edge(e).adj_cells;

    // Cell center coordinates and half-extents.
    // Compute cell centers from edge adjacency (average of edge midpoints).
    cell_cx_.resize(num_cells_, 0.0);
    cell_cy_.resize(num_cells_, 0.0);
    cell_cz_.resize(num_cells_, 0.0);
    cell_hx_.resize(num_cells_, 0.0);
    cell_hy_.resize(num_cells_, 0.0);
    cell_hz_.resize(num_cells_, 0.0);

    std::vector<int> cell_edge_count(num_cells_, 0);
    for (int e = 0; e < num_edges_; ++e) {
        for (int c : edge_adj_cells_[e]) {
            if (c >= 0 && c < num_cells_) {
                cell_cx_[c] += edge_x_[e];
                cell_cy_[c] += edge_y_[e];
                cell_cz_[c] += edge_z_[e];
                cell_edge_count[c]++;
            }
        }
    }
    for (int c = 0; c < num_cells_; ++c) {
        if (cell_edge_count[c] > 0) {
            cell_cx_[c] /= cell_edge_count[c];
            cell_cy_[c] /= cell_edge_count[c];
            cell_cz_[c] /= cell_edge_count[c];
        }
    }

    // Cell half-extents: derived from adjacent edges.
    // For each cell, find edges in each direction and use length/2 as half-extent.
    for (int e = 0; e < num_edges_; ++e) {
        int d = edge_direction_[e];
        Real half = edge_lengths_[e] / 2.0;
        for (int c : edge_adj_cells_[e]) {
            if (c < 0 || c >= num_cells_) continue;
            if (d == 0) { if (half > cell_hx_[c]) cell_hx_[c] = half; }
            if (d == 1) { if (half > cell_hy_[c]) cell_hy_[c] = half; }
            if (d == 2) { if (half > cell_hz_[c]) cell_hz_[c] = half; }
        }
    }

    // Face dual edge lengths: L_dual[f] = distance between cell centers
    // in face-normal direction. For boundary faces, use half-extent of the one cell.
    face_dual_length_.resize(num_faces_);
    for (int f = 0; f < num_faces_; ++f) {
        int n = face_normal_[f];
        int cm = face_cell_minus_[f];
        int cp = face_cell_plus_[f];

        if (cm >= 0 && cp >= 0) {
            // Interior face: distance between cell centers in normal direction
            Real cn_minus = (n == 0) ? cell_cx_[cm] : (n == 1) ? cell_cy_[cm] : cell_cz_[cm];
            Real cn_plus  = (n == 0) ? cell_cx_[cp] : (n == 1) ? cell_cy_[cp] : cell_cz_[cp];
            face_dual_length_[f] = std::abs(cn_plus - cn_minus);
        } else {
            // Boundary face: half-extent of the available cell
            int c = (cm >= 0) ? cm : cp;
            if (c >= 0) {
                face_dual_length_[f] = (n == 0) ? cell_hx_[c] :
                                       (n == 1) ? cell_hy_[c] : cell_hz_[c];
            } else {
                face_dual_length_[f] = std::sqrt(face_areas_[f]);  // fallback
            }
        }
        // Safety floor
        if (face_dual_length_[f] < 1e-20) face_dual_length_[f] = std::sqrt(face_areas_[f]);
    }

    // Edge dual face areas: A_dual[e] = area of dual face perpendicular to edge.
    // Each adjacent cell contributes a rectangular patch hp1(c) * hp2(c) to the
    // dual face.  The total is Σ_c hp1(c)*hp2(c) — NOT 4*avg(hp1)*avg(hp2).
    // The old formula (avg*avg) is only exact when all adjacent cells are the
    // same size; at octree level transitions it over-estimates by ~18%.
    edge_dual_area_.resize(num_edges_);
    for (int e = 0; e < num_edges_; ++e) {
        int d = edge_direction_[e];
        const auto& adj = edge_adj_cells_[e];

        if (adj.empty()) {
            edge_dual_area_[e] = edge_lengths_[e] * edge_lengths_[e]; // fallback
            continue;
        }

        Real sum_area = 0;
        int count = 0;
        for (int c : adj) {
            if (c < 0 || c >= num_cells_) continue;
            Real hp1, hp2;
            if (d == 0)      { hp1 = cell_hy_[c]; hp2 = cell_hz_[c]; }
            else if (d == 1) { hp1 = cell_hx_[c]; hp2 = cell_hz_[c]; }
            else             { hp1 = cell_hx_[c]; hp2 = cell_hy_[c]; }
            sum_area += hp1 * hp2;
            ++count;
        }

        edge_dual_area_[e] = (count > 0) ? sum_area
                                          : edge_lengths_[e] * edge_lengths_[e];
    }

    LOG_INFO("  Cell half-extents computed: hx[0]=" + std::to_string(cell_hx_[0]) +
             " hy[0]=" + std::to_string(cell_hy_[0]) +
             " hz[0]=" + std::to_string(cell_hz_[0]));

    build_curl_(grid);
    build_divergence_(grid);
    handle_hanging_faces_(grid);
    build_node_gradient_matrix_();

    LOG_INFO("  Curl Ce: " + std::to_string(Ce_.nnz()) + " nnz");
    LOG_INFO("  Divergence D: " + std::to_string(D_.nnz()) + " nnz");
}

// =========================================================================
// Discrete curl: Ce (faces × edges)
// Ce[f][e] = ±1 if edge e is on the boundary of face f (orientation)
// =========================================================================
void DiscreteOperators::build_curl_(const StaggeredGrid& grid) {
    // Build using COO (coordinate) format, then convert to CSR
    struct COOEntry { int row; int col; Real val; };
    std::vector<COOEntry> entries;

    // Curl orientation via right-hand rule (Stokes' theorem).
    //
    // For face with normal n̂, the 4 bounding edges form a loop.
    // The circulation direction is determined by the right-hand rule:
    //   normal=x(0): circulation in +y → +z → -y → -z
    //   normal=y(1): circulation in +z → +x → -z → -x
    //   normal=z(2): circulation in +x → +y → -x → -y
    //
    // Sign = +1 if edge direction aligns with circulation, -1 otherwise.
    // We determine alignment from the edge's position relative to face center.

    for (int f = 0; f < num_faces_; ++f) {
        const auto& fi = grid.face(f);
        for (int eid : fi.edges) {
            if (eid < 0 || eid >= num_edges_) continue;
            const auto& ei = grid.edge(eid);

            int n = fi.normal;  // face normal direction
            int d = ei.direction;  // edge parallel direction

            // Determine sign using right-hand rule circulation.
            // For normal=n, the two tangential directions are t1, t2 where
            // curl = ∂E_t2/∂t1 - ∂E_t1/∂t2 (in continuous form).
            //
            // Discrete: depends on which "side" of the face the edge is on.
            Real sign = 0.0;

            if (n == 0) {
                // x-normal face (yz plane): edges in y(1) and z(2)
                if (d == 1) {
                    // y-edge: sign depends on z-position relative to face center
                    sign = (ei.z < fi.z) ? 1.0 : -1.0;
                } else if (d == 2) {
                    // z-edge: sign depends on y-position relative to face center
                    sign = (ei.y < fi.y) ? -1.0 : 1.0;
                }
            } else if (n == 1) {
                // y-normal face (xz plane): edges in z(2) and x(0)
                if (d == 2) {
                    // z-edge: sign depends on x-position relative to face center
                    sign = (ei.x < fi.x) ? 1.0 : -1.0;
                } else if (d == 0) {
                    // x-edge: sign depends on z-position relative to face center
                    sign = (ei.z < fi.z) ? -1.0 : 1.0;
                }
            } else {
                // z-normal face (xy plane): edges in x(0) and y(1)
                if (d == 0) {
                    // x-edge: sign depends on y-position relative to face center
                    sign = (ei.y < fi.y) ? 1.0 : -1.0;
                } else if (d == 1) {
                    // y-edge: sign depends on x-position relative to face center
                    sign = (ei.x < fi.x) ? -1.0 : 1.0;
                }
            }

            if (std::abs(sign) > 0.5) {
                // Geometric curl: Ce_L[f,e] = ±L_edge (length-weighted).
                // Combined with Me = σ*A_dual*L (tangential Hodge), this gives
                // correct stiffness coupling at octree level transitions.
                entries.push_back({f, eid, sign * edge_lengths_[eid]});
            }
        }
    }

    // Convert COO to CSR
    Ce_.nrows = num_faces_;
    Ce_.ncols = num_edges_;
    Ce_.rowptr.assign(num_faces_ + 1, 0);

    for (const auto& e : entries)
        Ce_.rowptr[e.row + 1]++;
    for (int i = 0; i < num_faces_; ++i)
        Ce_.rowptr[i + 1] += Ce_.rowptr[i];

    Ce_.colidx.resize(entries.size());
    Ce_.values.resize(entries.size());
    std::vector<int> pos(Ce_.rowptr.begin(), Ce_.rowptr.end());

    for (const auto& e : entries) {
        int p = pos[e.row]++;
        Ce_.colidx[p] = e.col;
        Ce_.values[p] = e.val;
    }

    // Build transpose Cf = Ce^T
    // Build transpose Cf = Ce^T
    Cf_.nrows = num_edges_;
    Cf_.ncols = num_faces_;
    Cf_.rowptr.assign(num_edges_ + 1, 0);

    for (const auto& e : entries)
        Cf_.rowptr[e.col + 1]++;
    for (int i = 0; i < num_edges_; ++i)
        Cf_.rowptr[i + 1] += Cf_.rowptr[i];

    Cf_.colidx.resize(entries.size());
    Cf_.values.resize(entries.size());
    std::vector<int> pos2(Cf_.rowptr.begin(), Cf_.rowptr.end());

    for (const auto& e : entries) {
        int p = pos2[e.col]++;
        Cf_.colidx[p] = e.row;
        Cf_.values[p] = e.val;  // Ce^T
    }
}

// =========================================================================
// Discrete divergence: D (cells × faces)
// D[c][f] = ±area / volume (flux contribution)
// =========================================================================
void DiscreteOperators::build_divergence_(const StaggeredGrid& grid) {
    struct COOEntry { int row; int col; Real val; };
    std::vector<COOEntry> entries;

    cell_volumes_.resize(num_cells_);

    // Topological divergence: D[c][f] = ±1 (no area scaling).
    // Metric scaling goes into mass matrices.
    for (int f = 0; f < num_faces_; ++f) {
        const auto& fi = grid.face(f);

        if (fi.cell_minus >= 0 && fi.cell_minus < num_cells_) {
            entries.push_back({fi.cell_minus, f, 1.0});   // outward from cell_minus
        }
        if (fi.cell_plus >= 0 && fi.cell_plus < num_cells_) {
            entries.push_back({fi.cell_plus, f, -1.0});   // inward to cell_plus
        }
    }

    // Convert COO to CSR
    D_.nrows = num_cells_;
    D_.ncols = num_faces_;
    D_.rowptr.assign(num_cells_ + 1, 0);

    for (const auto& e : entries)
        D_.rowptr[e.row + 1]++;
    for (int i = 0; i < num_cells_; ++i)
        D_.rowptr[i + 1] += D_.rowptr[i];

    D_.colidx.resize(entries.size());
    D_.values.resize(entries.size());
    std::vector<int> pos(D_.rowptr.begin(), D_.rowptr.end());

    for (const auto& e : entries) {
        int p = pos[e.row]++;
        D_.colidx[p] = e.col;
        D_.values[p] = e.val;
    }
}

// =========================================================================
// Discrete gradient: placeholder
// =========================================================================
void DiscreteOperators::build_gradient_(const StaggeredGrid& /*grid*/) {
    // TODO: Build node-based gradient for divergence correction
    // G: num_edges × num_nodes
    // G[e][v1] = -1/L, G[e][v2] = +1/L
}

// =========================================================================
// Hanging face interpolation
// =========================================================================
void DiscreteOperators::handle_hanging_faces_(const StaggeredGrid& grid) {
    const auto& hfaces = grid.hanging_faces();
    const auto& hedges = grid.hanging_edges();
    dead_edges_.assign(num_edges_, false);
    dead_faces_.assign(num_faces_, false);
    hanging_pairs_.clear();
    if (hfaces.empty()) return;

    // ---------------------------------------------------------------
    // SimPEG/discretize-style handling of hanging faces/edges:
    //
    // Two modifications to Ce:
    //   (a) Dead (coarse hanging) face rows → zeroed
    //       Fine sub-faces already provide coupling for both cells.
    //       Keeping dead face would double-count contributions.
    //
    //   (b) In ALIVE face rows, replace dead edge references with
    //       0.5*child1 + 0.5*child2 (interpolation weights).
    //       Alive faces on the coarse cell still reference the coarse edge,
    //       but that edge is split — use children with half-weights.
    //
    // After both modifications, dead edges have ZERO columns in Ce:
    //   → zero stiffness (Ce^T * Mf_inv * Ce has no dead-edge entries)
    //   → zero mass (build_edge_mass skips dead edges)
    //   → identity elimination in forward solver, post-solve interpolation
    //
    // De Rham identity (Ce_mod * G = 0) is preserved because:
    //   G[D,:] = 0.5*G[c1,:] + 0.5*G[c2,:] when L_child = L_parent/2.
    // ---------------------------------------------------------------

    // Build dead edge → children map
    std::map<int, std::pair<int,int>> dead_to_children;
    for (const auto& he : hedges) {
        dead_edges_[he.coarse_edge] = true;
        dead_to_children[he.coarse_edge] = {he.fine_edges[0], he.fine_edges[1]};
        hanging_pairs_.push_back({he.coarse_edge, he.fine_edges[0], he.fine_edges[1]});
    }

    // Mark dead faces (member variable, used also in build_face_mass_inv)
    for (const auto& hf : hfaces) {
        dead_faces_[hf.coarse_face] = true;
    }

    // Modify Ce:
    // We need to rebuild Ce because replacing 1 dead edge with 2 children
    // adds entries (CSR structure changes). Rebuild in COO then convert.
    struct COOEntry { int row; int col; Real val; };
    std::vector<COOEntry> new_entries;
    new_entries.reserve(Ce_.nnz());

    int n_replaced = 0;
    int n_dead_face_zeroed = 0;
    for (int f = 0; f < num_faces_; ++f) {
        // (a) Dead (coarse hanging) face rows → ZEROED.
        //     Fine sub-faces already provide coupling for both cells.
        //     Keeping dead face would double-count curl-curl contributions.
        if (dead_faces_[f]) {
            ++n_dead_face_zeroed;
            continue;  // Skip — zero row in Ce
        }

        // (b) In ALIVE face rows, replace dead edge refs with
        //     0.5*child1 + 0.5*child2 (interpolation weights).
        for (int k = Ce_.rowptr[f]; k < Ce_.rowptr[f+1]; ++k) {
            int e = Ce_.colidx[k];
            Real val = Ce_.values[k];
            if (std::abs(val) < 1e-30) continue;

            if (dead_edges_[e]) {
                auto [c1, c2] = dead_to_children[e];
                new_entries.push_back({f, c1, 0.5 * val});
                new_entries.push_back({f, c2, 0.5 * val});
                ++n_replaced;
            } else {
                new_entries.push_back({f, e, val});
            }
        }
    }

    // Merge entries with same (row, col) — children might already appear
    // in the same face from their own sub-face edges
    std::map<std::pair<int,int>, Real> merged;
    for (const auto& e : new_entries) {
        merged[{e.row, e.col}] += e.val;
    }

    // Convert merged COO to CSR
    Ce_.nrows = num_faces_;
    Ce_.ncols = num_edges_;
    Ce_.rowptr.assign(num_faces_ + 1, 0);

    for (const auto& [key, val] : merged) {
        if (std::abs(val) > 1e-30)
            Ce_.rowptr[key.first + 1]++;
    }
    for (int f = 0; f < num_faces_; ++f)
        Ce_.rowptr[f + 1] += Ce_.rowptr[f];

    int total_nnz = Ce_.rowptr[num_faces_];
    Ce_.colidx.resize(total_nnz);
    Ce_.values.resize(total_nnz);
    std::vector<int> pos(Ce_.rowptr.begin(), Ce_.rowptr.end());

    for (const auto& [key, val] : merged) {
        if (std::abs(val) > 1e-30) {
            int p = pos[key.first]++;
            Ce_.colidx[p] = key.second;
            Ce_.values[p] = val;
        }
    }

    // Update cached data (face/edge adjacency from staggered grid)
    for (int f = 0; f < num_faces_; ++f) {
        face_cell_minus_[f] = grid.face(f).cell_minus;
        face_cell_plus_[f] = grid.face(f).cell_plus;
    }
    for (int e = 0; e < num_edges_; ++e)
        edge_adj_cells_[e] = grid.edge(e).adj_cells;

    LOG_INFO("  Hanging: " + std::to_string(n_dead_face_zeroed) + " dead faces ZEROED, " +
             std::to_string(n_replaced) + " dead-edge refs replaced in alive faces, " +
             std::to_string(hedges.size()) + " dead edges");
}

// =========================================================================
// Edge mass matrix: Me[e] = σ_avg(e) * A_dual(e) / L(e)
//
// Hodge star ⋆₁ for integral DOFs (Ce = ±1 incidence):
//   ⋆₁[e] = A_dual_perp(e) / L(e)
//
// A_dual_perp = area of dual face perpendicular to edge e.
// For cubic cells: A_dual = L², so ⋆₁ = L. (old formula)
// For rectangular cells: A_dual = (2*h_perp1)*(2*h_perp2) ≠ L².
//
// The system: Ce^T * (Mf_inv/μ₀) * Ce + iω * σ * ⋆₁ = 0
// For 1D halfspace: gives k² = iωμσ (correct dispersion).
// =========================================================================
void DiscreteOperators::build_edge_mass(const RealVec& sigma, DiagMatR& Me) const {
    Me.diag.assign(num_edges_, 0.0);

    // First pass: compute mass for ALL edges (including dead)
    for (int e = 0; e < num_edges_; ++e) {
        const auto& adj = edge_adj_cells_[e];
        if (adj.empty()) continue;

        // Arithmetic (volume-weighted) mean of σ in adjacent cells.
        // Tangential E-field is continuous across interfaces, so the mass
        // (= ∫σ E·E dV over dual cell) uses arithmetic averaging.
        // IMPORTANT: harmonic mean is WRONG here — it gives σ ≈ σ_air at
        // earth-air interfaces, making surface edges effectively insulating.
        Real sum_sigma = 0.0;
        int count = 0;
        for (int c : adj) {
            if (c >= 0 && c < static_cast<int>(sigma.size())) {
                sum_sigma += sigma[c];
                ++count;
            }
        }

        Real sigma_avg = (count > 0) ? sum_sigma / count : 1e-8;

        // Edge mass: Me = σ * A_dual * L (tangential Hodge).
        // With geometric Ce (±L), this gives the correct system.
        Real L = edge_lengths_[e];
        Real me_hodge = edge_dual_area_[e] * L;

        Me.diag[e] = sigma_avg * me_hodge;
    }

    // Second pass: zero dead edge mass, no redistribution.
    // Children already account for the full volume through their own cells.
    for (const auto& hp : hanging_pairs_) {
        Me.diag[hp.coarse] = 0.0;  // dead edge itself has zero mass in reduced system
    }
}

// =========================================================================
// Face mass matrix (permeability mass): Mf[f] = μ₀ * A_f / L_dual
//   where A_f = face area, L_dual = dual edge length crossing the face
//   L_dual = distance between cell centers on either side of the face.
//
// For cubic cells: L_dual = h = sqrt(A_f). (old formula, only correct for cubes)
// For rectangular cells: L_dual = h_normal ≠ sqrt(A_f).
//
// This appears in the system: A = C^T * Mf_inv * C + iω * Me(σ)
// Mf_inv = L_dual / (μ₀ * A_f)
// =========================================================================
void DiscreteOperators::build_face_mass(DiagMatR& Mf) const {
    Mf.diag.resize(num_faces_);
    for (int f = 0; f < num_faces_; ++f) {
        Real L_dual = face_dual_length_[f];
        Mf.diag[f] = constants::MU0 * face_areas_[f] / L_dual;
    }
}

void DiscreteOperators::build_face_mass_inv(DiagMatR& Mf_inv) const {
    Mf_inv.diag.resize(num_faces_);
    for (int f = 0; f < num_faces_; ++f) {
        // Dead faces have zeroed Ce rows → zero Mf_inv for safety
        if (!dead_faces_.empty() && dead_faces_[f]) {
            Mf_inv.diag[f] = 0.0;
            continue;
        }
        Real L_dual = face_dual_length_[f];
        Real mf = constants::MU0 * face_areas_[f] / L_dual;
        Mf_inv.diag[f] = (mf > 1e-30) ? 1.0 / mf : 0.0;
    }
}

// =========================================================================
// System assembly: A = C^T * Mf_inv * C + iω * Me(σ)
// =========================================================================
void DiscreteOperators::assemble_system(Real omega, const RealVec& sigma,
                                         SparseMatC& A) const {
    // Build mass matrices
    DiagMatR Me, Mf_inv;
    build_edge_mass(sigma, Me);
    build_face_mass_inv(Mf_inv);

    // ---------------------------------------------------------------
    // Stiffness: S = Ce^T * diag(Mf_inv) * Ce  (edges × edges)
    //
    // Memory optimization: The sparsity pattern depends only on mesh
    // topology (Ce), not on sigma or omega. We cache the CSR pattern
    // (rowptr, colidx) after the first assembly and reuse it for
    // subsequent frequencies, avoiding the expensive std::map which
    // uses ~64 bytes/entry (3× the final CSR cost).
    //
    // For N_e=100K, nnz~2.7M: map=173MB vs CSR=56MB.
    // For N_e=600K, nnz~16M:  map=1024MB vs CSR=333MB.
    // ---------------------------------------------------------------

    // --- Fast path: reuse cached sparsity pattern ---
    if (stiffness_pattern_built_) {
        int nnz = static_cast<int>(stiff_colidx_.size());
        A.nrows = num_edges_;
        A.ncols = num_edges_;
        A.rowptr = stiff_rowptr_;
        A.colidx = stiff_colidx_;
        A.values.assign(nnz, Complex(0, 0));

        // Fill stiffness: Ce^T * diag(Mf_inv) * Ce
        for (int f = 0; f < num_faces_; ++f) {
            Real mf_inv = Mf_inv.diag[f];
            for (int k1 = Ce_.rowptr[f]; k1 < Ce_.rowptr[f + 1]; ++k1) {
                int i = Ce_.colidx[k1];
                Real cf_if = Ce_.values[k1];
                for (int k2 = Ce_.rowptr[f]; k2 < Ce_.rowptr[f + 1]; ++k2) {
                    int j = Ce_.colidx[k2];
                    Real ce_fj = Ce_.values[k2];
                    // Binary search for (i,j) in CSR row i
                    auto begin = A.colidx.begin() + A.rowptr[i];
                    auto end   = A.colidx.begin() + A.rowptr[i + 1];
                    auto it = std::lower_bound(begin, end, j);
                    if (it != end && *it == j) {
                        int idx = static_cast<int>(it - A.colidx.begin());
                        A.values[idx] += Complex(cf_if * mf_inv * ce_fj, 0.0);
                    }
                }
            }
        }

        // Add iω * Me to diagonal
        Complex iw(0.0, omega);
        for (int e = 0; e < num_edges_; ++e) {
            auto begin = A.colidx.begin() + A.rowptr[e];
            auto end   = A.colidx.begin() + A.rowptr[e + 1];
            auto it = std::lower_bound(begin, end, e);
            if (it != end && *it == e) {
                int idx = static_cast<int>(it - A.colidx.begin());
                A.values[idx] += iw * Complex(Me.diag[e], 0.0);
            }
        }

        LOG_INFO("  System assembled (cached pattern): " +
                 std::to_string(num_edges_) + "×" + std::to_string(num_edges_) +
                 ", nnz=" + std::to_string(nnz));
        return;
    }

    // --- Slow path: first call, build sparsity pattern via map ---
    struct COOEntry { int row; int col; Complex val; };
    std::map<std::pair<int,int>, Complex> S_map;

    // For each face f, contribute Ce^T[i,f] * Mf_inv[f] * Ce[f,j]
    // Ce stores geometric entries (±L_edge), modified by 0.5 for dead edges.
    // Both alive and dead faces contribute (P^T·A·P projection).
    for (int f = 0; f < num_faces_; ++f) {
        Real mf_inv = Mf_inv.diag[f];

        for (int k1 = Ce_.rowptr[f]; k1 < Ce_.rowptr[f + 1]; ++k1) {
            int j = Ce_.colidx[k1];
            Real ce_fj = Ce_.values[k1];

            for (int k2 = Ce_.rowptr[f]; k2 < Ce_.rowptr[f + 1]; ++k2) {
                int i = Ce_.colidx[k2];
                Real cf_if = Ce_.values[k2];

                S_map[{i, j}] += Complex(cf_if * mf_inv * ce_fj, 0.0);
            }
        }
    }

    // Add iω * Me to diagonal
    Complex iw(0.0, omega);
    for (int e = 0; e < num_edges_; ++e) {
        S_map[{e, e}] += iw * Complex(Me.diag[e], 0.0);
    }

    // Grad-Div stabilization (DISABLED — experimental, not yet working correctly).
    // See doc/PHYSICS_INTEGRITY_REVIEW.md for analysis.
    // The GG^T penalty affects both gradient AND curl modes in discrete form,
    // making τ tuning extremely difficult. Air Dirichlet BCs are used instead.
    if (false && G_node_.nrows > 0 && G_node_.nnz() > 0) {
        // Compute a stabilization weight per edge:
        //   τ[e] = ω · μ₀ · max(σ_adj) · V_dual[e]
        // where V_dual = A_dual * L (dual volume around edge).
        // This matches the iω·Me scaling in earth and provides
        // comparable penalty in air.
        //
        // For air edges with no adjacent earth: τ = ω·μ₀·σ_bg·V
        Real sigma_bg_est = 0;
        for (int e = 0; e < num_edges_; ++e)
            if (Me.diag[e] > 0) sigma_bg_est = std::max(sigma_bg_est,
                Me.diag[e] / (edge_dual_area_[e] * edge_lengths_[e] + 1e-30));
        if (sigma_bg_est < 1e-10) sigma_bg_est = 0.01;  // fallback

        // τ must suppress null-space modes but not overwhelm the physical solution.
        // The iω·Me mass term provides regularization of O(ω·σ·V).
        // In earth, max(Me) ~ σ_bg·V gives ω·σ_bg·V.
        // We want τ·V·(2/L²) ~ ω·σ_bg·V, so τ ~ ω·σ_bg·L²/2.
        // For h=1000m, ω=0.0628, σ=0.01: τ ~ 0.0628*0.01*1e6/2 ≈ 314.
        // This provides the SAME level of regularization as iω·Me in earth.
        Real h_avg = 0;
        int h_count = 0;
        for (int e = 0; e < num_edges_; ++e) {
            if (edge_lengths_[e] > 0) { h_avg += edge_lengths_[e]; ++h_count; }
        }
        h_avg = (h_count > 0) ? h_avg / h_count : 1000.0;
        Real tau = omega * sigma_bg_est * h_avg * h_avg * 0.5;

        // G^T · diag(τ·V_dual) · G  (mass-weighted grad-div)
        // V_dual[e] = edge_dual_area_[e] * edge_lengths_[e]
        for (int e = 0; e < num_edges_; ++e) {
            if (!dead_edges_.empty() && dead_edges_[e]) continue;
            Real V_dual = edge_dual_area_[e] * edge_lengths_[e];
            Real tau_e = tau * V_dual;

            for (int k1 = G_node_.rowptr[e]; k1 < G_node_.rowptr[e + 1]; ++k1) {
                int v1 = G_node_.colidx[k1];
                Real g1 = G_node_.values[k1];  // ±1/L

                for (int k2 = G_node_.rowptr[e]; k2 < G_node_.rowptr[e + 1]; ++k2) {
                    int v2 = G_node_.colidx[k2];
                    Real g2 = G_node_.values[k2];

                    // This would give node×node matrix. We need edge×edge.
                    // G^T·G is node×node. We need a penalty on edges.
                    // Actually, the correct form is: G · (G^T · Me_τ · G)^{-1} · G^T?
                    // No — for grad-div penalty, we add to the edge system:
                    //   τ · G^T_node · M_node · G_node (edges × edges via nodes)
                    // But G_node is edges×nodes, so G^T·M·G is nodes×nodes.
                    // We need edges×edges penalty.
                    (void)v1; (void)v2; (void)g1; (void)g2;
                }
            }
        }

        // CORRECT APPROACH: The grad-div penalty on the EDGE system is:
        //   P = G_cell^T · M_cell^{-1} · G_cell
        // where G_cell (cells × edges) = D · Mf · Ce / ... no, this is divergence.
        //
        // Actually, the standard approach for edge-element null-space stabilization:
        //   P[e1, e2] = Σ_nodes τ · G[e1,v] · G[e2,v]  (G^T · G, rows=edges)
        //
        // Wait — G_node_ is (edges × nodes), so G_node_^T is (nodes × edges),
        // and G_node_ · G_node_^T is (edges × edges). But that's GG^T, not G^T G.
        //
        // For penalty:  A += τ · G · G^T  (edges × edges)
        // where G is the edge-to-node gradient (edges×nodes).
        // GG^T[e1,e2] = Σ_v G[e1,v] * G[e2,v]
        //
        // This penalizes edges that share nodes (gradient coupling).
        // For edge e with nodes v1,v2: G[e,v1]=-1/L, G[e,v2]=+1/L
        // GG^T[e,e] = 1/L² + 1/L² = 2/L² (diagonal)
        // GG^T[e1,e2] = -1/(L1·L2) if they share a node (negative coupling)

        // Simple approach: add τ·V_dual · GG^T per edge-pair
        for (int e = 0; e < num_edges_; ++e) {
            if (!dead_edges_.empty() && dead_edges_[e]) continue;
            Real V_dual = edge_dual_area_[e] * edge_lengths_[e];
            Real tw = tau * V_dual;

            for (int k1 = G_node_.rowptr[e]; k1 < G_node_.rowptr[e+1]; ++k1) {
                int v = G_node_.colidx[k1];
                Real gev = G_node_.values[k1];
                // Find all other edges sharing node v (via G_node_^T)
                // This requires iterating over G_node_ columns = nodes → O(N) scan
                // Too expensive for inline assembly.
            }
        }

        // For efficiency, build the penalty via COO:
        // P = Σ_e τ_e * g_e * g_e^T  where g_e is the e-th row of G_node_
        // P[i,j] = Σ_{v: v∈G[i] ∩ G[j]} G[i,v] * G[j,v] * (τ_i + τ_j)/2
        //
        // Actually simpler: P = G_node_ · G_node_^T with τ weighting
        // Assemble column-wise: for each node v, accumulate all (e1,e2) pairs.
        //
        // Build node-to-edge transpose map
        std::vector<std::vector<std::pair<int,Real>>> node_edges(num_nodes_);
        for (int e = 0; e < num_edges_; ++e) {
            if (!dead_edges_.empty() && dead_edges_[e]) continue;
            for (int k = G_node_.rowptr[e]; k < G_node_.rowptr[e+1]; ++k) {
                int v = G_node_.colidx[k];
                Real gval = G_node_.values[k];
                node_edges[v].push_back({e, gval});
            }
        }

        int n_grad_div = 0;
        for (int v = 0; v < num_nodes_; ++v) {
            const auto& elist = node_edges[v];
            for (size_t a = 0; a < elist.size(); ++a) {
                int e1 = elist[a].first;
                Real g1 = elist[a].second;
                Real V1 = edge_dual_area_[e1] * edge_lengths_[e1];
                for (size_t b = 0; b < elist.size(); ++b) {
                    int e2 = elist[b].first;
                    Real g2 = elist[b].second;
                    // Symmetric weight: average of both edge volumes
                    Real tw = tau * 0.5 * (V1 + edge_dual_area_[e2] * edge_lengths_[e2]);
                    S_map[{e1, e2}] += Complex(tw * g1 * g2, 0.0);
                    ++n_grad_div;
                }
            }
        }

        LOG_INFO("  Grad-Div stabilization: tau=" + std::to_string(tau) +
                 " sigma_bg_est=" + std::to_string(sigma_bg_est) +
                 " entries=" + std::to_string(n_grad_div));
    }

    // Dead edges have zero Ce/Me → A[dead,dead] = 0 (handled in forward solver)

    // Convert to CSR
    A.nrows = num_edges_;
    A.ncols = num_edges_;
    A.rowptr.assign(num_edges_ + 1, 0);

    // Count entries per row
    for (const auto& [key, val] : S_map)
        A.rowptr[key.first + 1]++;
    for (int i = 0; i < num_edges_; ++i)
        A.rowptr[i + 1] += A.rowptr[i];

    int nnz = static_cast<int>(S_map.size());
    A.colidx.resize(nnz);
    A.values.resize(nnz);

    std::vector<int> pos(A.rowptr.begin(), A.rowptr.end());
    for (const auto& [key, val] : S_map) {
        int p = pos[key.first]++;
        A.colidx[p] = key.second;
        A.values[p] = val;
    }

    // Sort column indices within each row (needed for binary search in fast path)
    for (int i = 0; i < num_edges_; ++i) {
        int start = A.rowptr[i], end = A.rowptr[i + 1];
        // Insertion sort (rows are typically short, <30 entries)
        for (int j = start + 1; j < end; ++j) {
            int col = A.colidx[j];
            Complex val = A.values[j];
            int k = j - 1;
            while (k >= start && A.colidx[k] > col) {
                A.colidx[k + 1] = A.colidx[k];
                A.values[k + 1] = A.values[k];
                --k;
            }
            A.colidx[k + 1] = col;
            A.values[k + 1] = val;
        }
    }

    // Cache sparsity pattern for subsequent frequencies
    // (const_cast is safe here: pattern is pure topology, doesn't change)
    auto* self = const_cast<DiscreteOperators*>(this);
    self->stiff_rowptr_ = A.rowptr;
    self->stiff_colidx_ = A.colidx;
    self->stiffness_pattern_built_ = true;
    LOG_INFO("  Stiffness sparsity pattern cached: nnz=" + std::to_string(nnz) +
             " (map freed, saving ~" + std::to_string(nnz * 48 / 1024 / 1024) + " MB/freq)");
}

// =========================================================================
// Build NODE-to-edge gradient matrix G_node_ (num_edges × num_nodes).
//
// Nodes are cell corner vertices on the Yee grid.
// For edge e between nodes v1, v2: G[e, v1] = -1/L, G[e, v2] = +1/L
// where L = edge length and the sign convention is v1→v2 = positive direction.
//
// KEY PROPERTY: curl(G_node * φ) = Ce * G_node * φ = 0 (discrete de Rham)
// This guarantees DivCorr correction doesn't alter curl(E).
// =========================================================================
void DiscreteOperators::build_node_gradient_matrix_() {
    // Step 1: Enumerate unique nodes from edge endpoints.
    // Each edge has two endpoint nodes at midpoint ± L/2 * direction_vector.
    struct NodeKey {
        int64_t ix, iy, iz;
        bool operator<(const NodeKey& o) const {
            if (ix != o.ix) return ix < o.ix;
            if (iy != o.iy) return iy < o.iy;
            return iz < o.iz;
        }
    };

    // Find minimum edge length for quantization
    Real min_L = 1e30;
    for (int e = 0; e < num_edges_; ++e)
        if (edge_lengths_[e] > 1e-10) min_L = std::min(min_L, edge_lengths_[e]);
    Real tol = min_L * 0.01;  // 1% of smallest edge

    auto quantize = [&](Real v) -> int64_t {
        return static_cast<int64_t>(std::round(v / tol));
    };

    std::map<NodeKey, int> node_map;  // position → global node ID
    edge_node1_.resize(num_edges_);
    edge_node2_.resize(num_edges_);

    auto get_or_create_node = [&](Real x, Real y, Real z) -> int {
        NodeKey key{quantize(x), quantize(y), quantize(z)};
        auto it = node_map.find(key);
        if (it != node_map.end()) return it->second;
        int id = static_cast<int>(node_map.size());
        node_map[key] = id;
        return id;
    };

    // Collect all nodes from edge endpoints
    for (int e = 0; e < num_edges_; ++e) {
        int d = edge_direction_[e];
        Real half_L = edge_lengths_[e] * 0.5;

        Real x1 = edge_x_[e], y1 = edge_y_[e], z1 = edge_z_[e];
        Real x2 = x1, y2 = y1, z2 = z1;

        if (d == 0) { x1 -= half_L; x2 += half_L; }
        else if (d == 1) { y1 -= half_L; y2 += half_L; }
        else { z1 -= half_L; z2 += half_L; }

        edge_node1_[e] = get_or_create_node(x1, y1, z1);
        edge_node2_[e] = get_or_create_node(x2, y2, z2);
    }

    num_nodes_ = static_cast<int>(node_map.size());

    // Store node coordinates
    node_x_.resize(num_nodes_);
    node_y_.resize(num_nodes_);
    node_z_.resize(num_nodes_);
    for (const auto& [key, id] : node_map) {
        node_x_[id] = key.ix * tol;
        node_y_[id] = key.iy * tol;
        node_z_[id] = key.iz * tol;
    }

    // Step 2: Build G_node_ sparse matrix (edges × nodes)
    // G[e, v1] = -1/L, G[e, v2] = +1/L  (v1 is minus end, v2 is plus end)
    G_node_.nrows = num_edges_;
    G_node_.ncols = num_nodes_;
    G_node_.rowptr.resize(num_edges_ + 1);

    // Each edge has exactly 2 entries (or 1 if v1==v2, which shouldn't happen)
    for (int e = 0; e <= num_edges_; ++e) G_node_.rowptr[e] = 2 * e;

    int nnz = 2 * num_edges_;
    G_node_.colidx.resize(nnz);
    G_node_.values.resize(nnz);

    for (int e = 0; e < num_edges_; ++e) {
        int v1 = edge_node1_[e];
        int v2 = edge_node2_[e];
        Real inv_L = 1.0 / edge_lengths_[e];

        // Store sorted by column index for CSR convention
        int k = 2 * e;
        if (v1 < v2) {
            G_node_.colidx[k]     = v1;  G_node_.values[k]     = -inv_L;
            G_node_.colidx[k + 1] = v2;  G_node_.values[k + 1] = +inv_L;
        } else {
            G_node_.colidx[k]     = v2;  G_node_.values[k]     = +inv_L;
            G_node_.colidx[k + 1] = v1;  G_node_.values[k + 1] = -inv_L;
        }
    }

    LOG_INFO("  Node gradient G_node: " + std::to_string(num_edges_) + " x " +
             std::to_string(num_nodes_) + ", " + std::to_string(nnz) + " nnz" +
             " (min_L=" + std::to_string(min_L) + " tol=" + std::to_string(tol) + ")");
}

// =========================================================================
// DivCorr: Node-based Laplacian L = G^T * Me_σ * G  (SPD, nodes × nodes)
//   where Me_σ[e] = σ_avg(e) * L_edge (edge mass).
//   L has positive diagonal, negative off-diagonals.
//   Boundary: handled via φ=0 Dirichlet at boundary nodes in the solver.
// =========================================================================
void DiscreteOperators::build_div_laplacian(const RealVec& sigma,
                                             SparseMatR& L) const {
    // Build edge mass Me_σ = σ_avg * A_dual * L  (tangential Hodge).
    // Consistent with system assembly Me and compute_div_sigma_E.
    RealVec me_sigma(num_edges_, 0.0);
    for (int e = 0; e < num_edges_; ++e) {
        // Dead edges should not contribute to Laplacian
        if (!dead_edges_.empty() && dead_edges_[e]) continue;

        const auto& adj = edge_adj_cells_[e];
        Real sum_sigma = 0.0;
        int count = 0;
        for (int c : adj) {
            if (c >= 0 && c < static_cast<int>(sigma.size())) {
                sum_sigma += sigma[c];
                ++count;
            }
        }
        Real sigma_avg = (count > 0) ? sum_sigma / count : 1e-8;
        Real L_e = edge_lengths_[e];
        me_sigma[e] = sigma_avg * edge_dual_area_[e] * L_e;
    }

    // L = G^T * diag(me_sigma) * G  via edge-by-edge accumulation.
    // Each edge e contributes a 2×2 block at (v1,v2) × (v1,v2).
    std::map<std::pair<int,int>, Real> L_map;

    for (int e = 0; e < num_edges_; ++e) {
        Real w = me_sigma[e];
        if (w < 1e-30) continue;

        for (int k1 = G_node_.rowptr[e]; k1 < G_node_.rowptr[e + 1]; ++k1) {
            int n1 = G_node_.colidx[k1];
            Real g1 = G_node_.values[k1];

            for (int k2 = G_node_.rowptr[e]; k2 < G_node_.rowptr[e + 1]; ++k2) {
                int n2 = G_node_.colidx[k2];
                Real g2 = G_node_.values[k2];

                L_map[{n1, n2}] += g1 * w * g2;
            }
        }
    }

    // Convert to CSR
    L.nrows = num_nodes_;
    L.ncols = num_nodes_;
    L.rowptr.assign(num_nodes_ + 1, 0);

    for (const auto& [key, val] : L_map)
        L.rowptr[key.first + 1]++;
    for (int i = 0; i < num_nodes_; ++i)
        L.rowptr[i + 1] += L.rowptr[i];

    int nnz = static_cast<int>(L_map.size());
    L.colidx.resize(nnz);
    L.values.resize(nnz);
    std::vector<int> pos(L.rowptr.begin(), L.rowptr.end());

    for (const auto& [key, val] : L_map) {
        int p = pos[key.first]++;
        L.colidx[p] = key.second;
        L.values[p] = val;
    }

    LOG_INFO("  DivCorr Laplacian (G^T Me G): " + std::to_string(num_nodes_) +
             " nodes, " + std::to_string(nnz) + " nnz");
}

// =========================================================================
// DivCorr: Compute div(σE) on NODES: divJ = G^T * Me_σ * E.
//   Consistent with G_node_: guarantees L*φ = divJ → E -= G*φ → new divJ = 0.
// =========================================================================
void DiscreteOperators::compute_div_sigma_E(const ComplexVec& E,
                                             const RealVec& sigma,
                                             ComplexVec& divJ) const {
    divJ.assign(num_nodes_, Complex(0, 0));

    for (int e = 0; e < num_edges_; ++e) {
        // Dead edges should not contribute to div(σE)
        if (!dead_edges_.empty() && dead_edges_[e]) continue;

        // Compute Me_σ[e] * E[e]
        const auto& adj = edge_adj_cells_[e];
        Real sum_sigma = 0.0;
        int count = 0;
        for (int c : adj) {
            if (c >= 0 && c < static_cast<int>(sigma.size())) {
                sum_sigma += sigma[c];
                ++count;
            }
        }
        Real sigma_avg = (count > 0) ? sum_sigma / count : 1e-8;
        Real L_e = edge_lengths_[e];
        // Tangential Hodge: Me = σ * A_dual * L (arithmetic avg).
        // Must match build_div_laplacian which also uses σ * A_dual * L.
        Real me = sigma_avg * edge_dual_area_[e] * L_e;

        Complex weighted_E = me * E[e];

        // Accumulate G^T: divJ[v] += G[e,v] * weighted_E
        for (int k = G_node_.rowptr[e]; k < G_node_.rowptr[e + 1]; ++k) {
            divJ[G_node_.colidx[k]] += G_node_.values[k] * weighted_E;
        }
    }
}

// =========================================================================
// DivCorr: Apply node-to-edge gradient: grad = G_node_ * φ.
//   curl(G_node * φ) = 0 by construction (discrete de Rham identity).
// =========================================================================
void DiscreteOperators::apply_cell_gradient(const ComplexVec& phi,
                                             ComplexVec& grad) const {
    grad.assign(num_edges_, Complex(0, 0));

    for (int e = 0; e < num_edges_; ++e) {
        Complex val(0, 0);
        for (int k = G_node_.rowptr[e]; k < G_node_.rowptr[e + 1]; ++k) {
            val += G_node_.values[k] * phi[G_node_.colidx[k]];
        }
        grad[e] = val;
    }
}

} // namespace octree
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
