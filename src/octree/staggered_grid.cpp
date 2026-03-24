// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#include "maple3dmt/octree/staggered_grid.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/utils/logger.h"

#define LOG_INFO(msg) MAPLE3DMT_LOG_INFO(msg)
#include <map>
#include <tuple>
#include <cmath>

namespace maple3dmt {
namespace octree {

// Quantized coordinate key for deduplication.
// All edge/face midpoints are quantized to integer grid at 2*max_level resolution.
using CoordKey = std::tuple<int64_t, int64_t, int64_t>;
using EdgeKey  = std::pair<CoordKey, int>;  // (midpoint, direction)
using FaceKey  = std::pair<CoordKey, int>;  // (center, normal)

namespace {
    struct GridContext {
        Real dx, dy, dz;
        Real x_min, y_min, z_min;
        double quant;  // quantization factor

        CoordKey quantize(Real px, Real py, Real pz) const {
            int64_t ix = static_cast<int64_t>(std::round((px - x_min) / dx * quant));
            int64_t iy = static_cast<int64_t>(std::round((py - y_min) / dy * quant));
            int64_t iz = static_cast<int64_t>(std::round((pz - z_min) / dz * quant));
            return {ix, iy, iz};
        }
    };

    GridContext make_ctx(const OctreeMesh& mesh) {
        const auto& p = mesh.params();
        GridContext ctx;
        ctx.dx = p.domain_x_max - p.domain_x_min;
        ctx.dy = p.domain_y_max - p.domain_y_min;
        ctx.dz = p.domain_z_max - p.domain_z_min;
        ctx.x_min = p.domain_x_min;
        ctx.y_min = p.domain_y_min;
        ctx.z_min = p.domain_z_min;
        ctx.quant = (1 << (p.max_level + 2));  // extra bit for edge midpoints
        return ctx;
    }

    // Cell half-extents in physical coords
    struct CellHalf {
        Real cx, cy, cz;  // center
        Real hx, hy, hz;  // half-size per axis
    };

    CellHalf cell_half(const OctreeMesh& mesh, const GridContext& ctx, int c) {
        CellHalf ch;
        mesh.cell_center(c, ch.cx, ch.cy, ch.cz);
        int level = mesh.cell_level(c);
        Real h_unit = 1.0 / (1 << level);
        ch.hx = h_unit * ctx.dx / 2.0;
        ch.hy = h_unit * ctx.dy / 2.0;
        ch.hz = h_unit * ctx.dz / 2.0;
        return ch;
    }
}

void StaggeredGrid::build(const OctreeMesh& mesh) {
    num_cells_ = mesh.num_cells_local();

    LOG_INFO("StaggeredGrid::build — enumerating DOFs for " +
             std::to_string(num_cells_) + " cells");

    auto ctx = make_ctx(mesh);

    // ---- Phase 1: Enumerate all unique edges ----
    std::map<EdgeKey, int> edge_map;
    edges_.clear();

    for (int c = 0; c < num_cells_; ++c) {
        auto ch = cell_half(mesh, ctx, c);

        // 12 edges per hex cell:
        // 4 x-edges: midpoint = (cx, cy±hy, cz±hz), length = 2*hx
        Real ey[] = {ch.cy - ch.hy, ch.cy - ch.hy, ch.cy + ch.hy, ch.cy + ch.hy};
        Real ez[] = {ch.cz - ch.hz, ch.cz + ch.hz, ch.cz - ch.hz, ch.cz + ch.hz};
        for (int i = 0; i < 4; ++i) {
            EdgeKey key{ctx.quantize(ch.cx, ey[i], ez[i]), 0};
            if (edge_map.find(key) == edge_map.end()) {
                edge_map[key] = static_cast<int>(edges_.size());
                edges_.push_back({ch.cx, ey[i], ez[i], 0, 2.0 * ch.hx, {}});
            }
            edges_[edge_map[key]].adj_cells.push_back(c);
        }

        // 4 y-edges: midpoint = (cx±hx, cy, cz±hz), length = 2*hy
        Real ex[] = {ch.cx - ch.hx, ch.cx - ch.hx, ch.cx + ch.hx, ch.cx + ch.hx};
        Real ez2[] = {ch.cz - ch.hz, ch.cz + ch.hz, ch.cz - ch.hz, ch.cz + ch.hz};
        for (int i = 0; i < 4; ++i) {
            EdgeKey key{ctx.quantize(ex[i], ch.cy, ez2[i]), 1};
            if (edge_map.find(key) == edge_map.end()) {
                edge_map[key] = static_cast<int>(edges_.size());
                edges_.push_back({ex[i], ch.cy, ez2[i], 1, 2.0 * ch.hy, {}});
            }
            edges_[edge_map[key]].adj_cells.push_back(c);
        }

        // 4 z-edges: midpoint = (cx±hx, cy±hy, cz), length = 2*hz
        Real ex3[] = {ch.cx - ch.hx, ch.cx + ch.hx, ch.cx - ch.hx, ch.cx + ch.hx};
        Real ey3[] = {ch.cy - ch.hy, ch.cy - ch.hy, ch.cy + ch.hy, ch.cy + ch.hy};
        for (int i = 0; i < 4; ++i) {
            EdgeKey key{ctx.quantize(ex3[i], ey3[i], ch.cz), 2};
            if (edge_map.find(key) == edge_map.end()) {
                edge_map[key] = static_cast<int>(edges_.size());
                edges_.push_back({ex3[i], ey3[i], ch.cz, 2, 2.0 * ch.hz, {}});
            }
            edges_[edge_map[key]].adj_cells.push_back(c);
        }
    }
    LOG_INFO("  Edges: " + std::to_string(edges_.size()));

    // ---- Phase 2: Enumerate all unique faces, linking to edges ----
    std::map<FaceKey, int> face_map;
    faces_.clear();

    for (int c = 0; c < num_cells_; ++c) {
        auto ch = cell_half(mesh, ctx, c);

        struct EdgeRef { Real mx, my, mz; int dir; };
        struct FaceDef {
            Real fx, fy, fz;
            int normal;
            Real area;
            EdgeRef edges[4];
        };

        FaceDef fdefs[6];

        // +x face
        fdefs[0].fx = ch.cx+ch.hx; fdefs[0].fy = ch.cy; fdefs[0].fz = ch.cz;
        fdefs[0].normal = 0; fdefs[0].area = 4*ch.hy*ch.hz;
        fdefs[0].edges[0] = {ch.cx+ch.hx, ch.cy, ch.cz-ch.hz, 1};
        fdefs[0].edges[1] = {ch.cx+ch.hx, ch.cy, ch.cz+ch.hz, 1};
        fdefs[0].edges[2] = {ch.cx+ch.hx, ch.cy-ch.hy, ch.cz, 2};
        fdefs[0].edges[3] = {ch.cx+ch.hx, ch.cy+ch.hy, ch.cz, 2};

        // -x face
        fdefs[1].fx = ch.cx-ch.hx; fdefs[1].fy = ch.cy; fdefs[1].fz = ch.cz;
        fdefs[1].normal = 0; fdefs[1].area = 4*ch.hy*ch.hz;
        fdefs[1].edges[0] = {ch.cx-ch.hx, ch.cy, ch.cz-ch.hz, 1};
        fdefs[1].edges[1] = {ch.cx-ch.hx, ch.cy, ch.cz+ch.hz, 1};
        fdefs[1].edges[2] = {ch.cx-ch.hx, ch.cy-ch.hy, ch.cz, 2};
        fdefs[1].edges[3] = {ch.cx-ch.hx, ch.cy+ch.hy, ch.cz, 2};

        // +y face
        fdefs[2].fx = ch.cx; fdefs[2].fy = ch.cy+ch.hy; fdefs[2].fz = ch.cz;
        fdefs[2].normal = 1; fdefs[2].area = 4*ch.hx*ch.hz;
        fdefs[2].edges[0] = {ch.cx, ch.cy+ch.hy, ch.cz-ch.hz, 0};
        fdefs[2].edges[1] = {ch.cx, ch.cy+ch.hy, ch.cz+ch.hz, 0};
        fdefs[2].edges[2] = {ch.cx-ch.hx, ch.cy+ch.hy, ch.cz, 2};
        fdefs[2].edges[3] = {ch.cx+ch.hx, ch.cy+ch.hy, ch.cz, 2};

        // -y face
        fdefs[3].fx = ch.cx; fdefs[3].fy = ch.cy-ch.hy; fdefs[3].fz = ch.cz;
        fdefs[3].normal = 1; fdefs[3].area = 4*ch.hx*ch.hz;
        fdefs[3].edges[0] = {ch.cx, ch.cy-ch.hy, ch.cz-ch.hz, 0};
        fdefs[3].edges[1] = {ch.cx, ch.cy-ch.hy, ch.cz+ch.hz, 0};
        fdefs[3].edges[2] = {ch.cx-ch.hx, ch.cy-ch.hy, ch.cz, 2};
        fdefs[3].edges[3] = {ch.cx+ch.hx, ch.cy-ch.hy, ch.cz, 2};

        // +z face
        fdefs[4].fx = ch.cx; fdefs[4].fy = ch.cy; fdefs[4].fz = ch.cz+ch.hz;
        fdefs[4].normal = 2; fdefs[4].area = 4*ch.hx*ch.hy;
        fdefs[4].edges[0] = {ch.cx, ch.cy-ch.hy, ch.cz+ch.hz, 0};
        fdefs[4].edges[1] = {ch.cx, ch.cy+ch.hy, ch.cz+ch.hz, 0};
        fdefs[4].edges[2] = {ch.cx-ch.hx, ch.cy, ch.cz+ch.hz, 1};
        fdefs[4].edges[3] = {ch.cx+ch.hx, ch.cy, ch.cz+ch.hz, 1};

        // -z face
        fdefs[5].fx = ch.cx; fdefs[5].fy = ch.cy; fdefs[5].fz = ch.cz-ch.hz;
        fdefs[5].normal = 2; fdefs[5].area = 4*ch.hx*ch.hy;
        fdefs[5].edges[0] = {ch.cx, ch.cy-ch.hy, ch.cz-ch.hz, 0};
        fdefs[5].edges[1] = {ch.cx, ch.cy+ch.hy, ch.cz-ch.hz, 0};
        fdefs[5].edges[2] = {ch.cx-ch.hx, ch.cy, ch.cz-ch.hz, 1};
        fdefs[5].edges[3] = {ch.cx+ch.hx, ch.cy, ch.cz-ch.hz, 1};

        for (int f = 0; f < 6; ++f) {
            auto& fd = fdefs[f];
            FaceKey key{ctx.quantize(fd.fx, fd.fy, fd.fz), fd.normal};

            if (face_map.find(key) == face_map.end()) {
                int fid = static_cast<int>(faces_.size());
                face_map[key] = fid;

                // Find 4 edge IDs
                std::vector<int> eid_list;
                for (int e = 0; e < 4; ++e) {
                    EdgeKey ek{ctx.quantize(fd.edges[e].mx, fd.edges[e].my, fd.edges[e].mz),
                               fd.edges[e].dir};
                    auto it = edge_map.find(ek);
                    if (it != edge_map.end()) {
                        eid_list.push_back(it->second);
                    }
                    // Edge might not exist if at domain boundary — skip
                }

                faces_.push_back(FaceInfo{
                    fd.fx, fd.fy, fd.fz,
                    fd.normal, fd.area,
                    -1, -1,
                    std::move(eid_list)
                });
            }

            int fid = face_map[key];
            if (f == 0 || f == 2 || f == 4) {
                faces_[fid].cell_minus = c;  // +face: cell is on minus side
            } else {
                faces_[fid].cell_plus = c;   // -face: cell is on plus side
            }
        }
    }
    LOG_INFO("  Faces: " + std::to_string(faces_.size()));

    // ---- Phase 3: Hanging entities (TODO for adaptive mesh) ----
    detect_hanging_(mesh);

    // ---- Phase 4: Cell-edge map ----
    build_cell_edge_map_(mesh);

    LOG_INFO("  DOFs: edges=" + std::to_string(num_edges()) +
             " faces=" + std::to_string(num_faces()) +
             " cells=" + std::to_string(num_cells_));
}

void StaggeredGrid::enumerate_edges_(const OctreeMesh&) { /* done in build() */ }
void StaggeredGrid::enumerate_faces_(const OctreeMesh&) { /* done in build() */ }

void StaggeredGrid::detect_hanging_(const OctreeMesh& mesh) {
    hanging_faces_.clear();
    hanging_edges_.clear();

    // Uniform mesh: no hanging faces possible
    if (mesh.params().min_level >= mesh.params().max_level) return;

    auto ctx = make_ctx(mesh);
    const auto& par = mesh.params();

    // Boundary tolerance: small fraction of finest possible cell
    Real min_h = std::min({ctx.dx / (1 << par.max_level),
                           ctx.dy / (1 << par.max_level),
                           ctx.dz / (1 << par.max_level)});
    Real btol = min_h * 0.01;

    // Build face lookup: (quantized center, normal) → face_id
    std::map<FaceKey, int> face_lk;
    for (int f = 0; f < static_cast<int>(faces_.size()); ++f) {
        FaceKey key{ctx.quantize(faces_[f].x, faces_[f].y, faces_[f].z), faces_[f].normal};
        face_lk[key] = f;
    }

    // Build edge lookup: (quantized midpoint, direction) → edge_id
    std::map<EdgeKey, int> edge_lk;
    for (int e = 0; e < static_cast<int>(edges_.size()); ++e) {
        EdgeKey key{ctx.quantize(edges_[e].x, edges_[e].y, edges_[e].z), edges_[e].direction};
        edge_lk[key] = e;
    }

    std::vector<bool> face_done(faces_.size(), false);
    std::vector<bool> edge_is_hanging(edges_.size(), false);

    for (int f = 0; f < static_cast<int>(faces_.size()); ++f) {
        if (face_done[f]) continue;
        auto& fi = faces_[f];

        // Need exactly one connected cell
        if (fi.cell_minus >= 0 && fi.cell_plus >= 0) continue;
        if (fi.cell_minus < 0 && fi.cell_plus < 0) continue;

        int n = fi.normal;

        // Skip domain boundary faces
        bool on_bnd = false;
        switch (n) {
            case 0: on_bnd = (fi.x < par.domain_x_min + btol || fi.x > par.domain_x_max - btol); break;
            case 1: on_bnd = (fi.y < par.domain_y_min + btol || fi.y > par.domain_y_max - btol); break;
            case 2: on_bnd = (fi.z < par.domain_z_min + btol || fi.z > par.domain_z_max - btol); break;
        }
        if (on_bnd) continue;

        // Non-boundary face with one missing cell → hanging interface candidate
        int ccell = (fi.cell_minus >= 0) ? fi.cell_minus : fi.cell_plus;
        auto ch = cell_half(mesh, ctx, ccell);

        // Fine half-sizes = coarse half-sizes / 2
        Real fhx = ch.hx / 2, fhy = ch.hy / 2, fhz = ch.hz / 2;

        // Try to find 4 sub-faces (= this face is the coarse one)
        bool found_all = true;
        std::array<int, 4> subs;
        int idx = 0;
        for (int s1 = -1; s1 <= 1; s1 += 2) {
            for (int s2 = -1; s2 <= 1; s2 += 2) {
                Real sx = fi.x, sy = fi.y, sz = fi.z;
                switch (n) {
                    case 0: sy += s1 * fhy; sz += s2 * fhz; break;
                    case 1: sx += s1 * fhx; sz += s2 * fhz; break;
                    case 2: sx += s1 * fhx; sy += s2 * fhy; break;
                }
                FaceKey key{ctx.quantize(sx, sy, sz), n};
                auto it = face_lk.find(key);
                if (it == face_lk.end() || face_done[it->second]) {
                    found_all = false; break;
                }
                subs[idx++] = it->second;
            }
            if (!found_all) break;
        }

        if (!found_all) continue;  // fine face — will be matched from the coarse side

        // Verify: each sub-face should have exactly one missing cell
        for (int ff : subs) {
            auto& ffi = faces_[ff];
            if (ffi.cell_minus >= 0 && ffi.cell_plus >= 0) {
                found_all = false; break;
            }
        }
        if (!found_all) continue;

        // ---- Coarse hanging face confirmed ----
        hanging_faces_.push_back({f, subs});
        face_done[f] = true;
        for (int ff : subs) face_done[ff] = true;

        // Fix adjacency: fine faces get the coarse cell on their missing side
        for (int ff : subs) {
            auto& ffi = faces_[ff];
            if (ffi.cell_minus < 0) ffi.cell_minus = ccell;
            else if (ffi.cell_plus < 0) ffi.cell_plus = ccell;
        }

        // Mark coarse face dead (both cells = -1)
        fi.cell_minus = -1;
        fi.cell_plus = -1;

        // Add coarse cell to fine face edges' adj_cells
        for (int ff : subs) {
            for (int eid : faces_[ff].edges) {
                if (eid < 0) continue;
                auto& adj = edges_[eid].adj_cells;
                bool has = false;
                for (int c : adj) { if (c == ccell) { has = true; break; } }
                if (!has) adj.push_back(ccell);
            }
        }

        // Detect hanging edges: each coarse face edge maps to 2 fine edges
        for (int eid : fi.edges) {
            if (eid < 0 || edge_is_hanging[eid]) continue;
            auto& ei = edges_[eid];
            int d = ei.direction;
            Real qlen = ei.length / 4.0;  // offset from midpoint to sub-edge midpoints

            Real fe1x = ei.x, fe1y = ei.y, fe1z = ei.z;
            Real fe2x = ei.x, fe2y = ei.y, fe2z = ei.z;
            switch (d) {
                case 0: fe1x -= qlen; fe2x += qlen; break;
                case 1: fe1y -= qlen; fe2y += qlen; break;
                case 2: fe1z -= qlen; fe2z += qlen; break;
            }

            EdgeKey k1{ctx.quantize(fe1x, fe1y, fe1z), d};
            EdgeKey k2{ctx.quantize(fe2x, fe2y, fe2z), d};
            auto it1 = edge_lk.find(k1), it2 = edge_lk.find(k2);

            if (it1 != edge_lk.end() && it2 != edge_lk.end()) {
                hanging_edges_.push_back({eid, {it1->second, it2->second}});
                edge_is_hanging[eid] = true;
            }
        }
    }

    if (!hanging_faces_.empty()) {
        LOG_INFO("  Hanging faces: " + std::to_string(hanging_faces_.size()) +
                 " coarse-fine groups, " + std::to_string(hanging_edges_.size()) +
                 " hanging edge pairs");
    }
}

void StaggeredGrid::build_cell_edge_map_(const OctreeMesh& mesh) {
    cell_edges_.resize(mesh.num_cells_local());
    for (auto& ce : cell_edges_) ce.clear();

    for (int e = 0; e < num_edges(); ++e) {
        const auto& ei = edges_[e];
        int n_adj = static_cast<int>(ei.adj_cells.size());
        if (n_adj == 0) continue;

        Real frac = 1.0 / n_adj;
        for (int c : ei.adj_cells) {
            if (c >= 0 && c < mesh.num_cells_local()) {
                cell_edges_[c].push_back(CellEdge{e, frac});
            }
        }
    }
}

} // namespace octree
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
