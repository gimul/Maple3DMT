// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file hex_mesh_3d.cpp
/// @brief 3D hexahedral mesh generation for MT inversion.
///
/// Extends the 2.5D structured mesh approach (terrain_mesh.cpp) to 3D:
///   - Tensor product of non-uniform x, y, z node distributions
///   - Terrain-conforming via transfinite interpolation (TFI)
///   - Air / earth attribute classification
///   - Station-proximity h-refinement

#include "maple3dmt/mesh/hex_mesh_3d.h"
#include "maple3dmt/utils/logger.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <set>

#ifdef MAPLE3DMT_USE_GDAL
#include <gdal_priv.h>
#include <cpl_conv.h>
#endif

namespace maple3dmt {
namespace mesh {

// ALOSDem implementation moved to dem.cpp (shared by FEM & FV backends)

// =========================================================================
// HexMeshGenerator3D — private helpers
// =========================================================================

/// Compute z-fractions with geometric growth (reusable helper).
/// Returns fractions in [0, 1] with frac[0]=0, frac.back()=1.
/// h_start: initial layer thickness, h_cap: max layer thickness,
/// total: total distance, growth: geometric growth rate.
static RealVec compute_z_fractions(Real h_start, Real h_cap,
                                   Real total, Real growth) {
    RealVec fracs;
    fracs.push_back(0.0);

    Real accumulated = 0.0;
    Real h = h_start;
    Real h_prev = h;

    while (accumulated < total - 1e-6) {
        Real remaining = total - accumulated;
        h = std::min(h, remaining);  // don't overshoot

        // If the remaining sliver is less than 30% of the previous layer,
        // merge it into the previous layer to avoid degenerate thin elements.
        if (remaining < h_prev * 0.3 && fracs.size() >= 2) {
            break;
        }

        h_prev = h;
        accumulated += h;
        fracs.push_back(accumulated / total);
        h = std::min(h * growth, h_cap);
    }

    fracs.back() = 1.0;
    return fracs;
}

std::vector<Real> HexMeshGenerator3D::build_axis_nodes_(
    Real min, Real max, Real h_fine, Real growth,
    Real roi_min, Real roi_max) {

    // Cap: max element size = domain span / 4
    Real h_cap = (max - min) * 0.25;

    // --- Left transition zone (roi_min → min, growing left) ---
    RealVec left_nodes;
    {
        Real pos = roi_min;
        Real h = h_fine;
        while (pos - min > h * 0.5) {
            pos -= h;
            if (pos < min) pos = min;
            left_nodes.push_back(pos);
            h = std::min(h * growth, h_cap);
        }
        if (left_nodes.empty() || left_nodes.back() > min + 1e-6) {
            left_nodes.push_back(min);
        }
        std::reverse(left_nodes.begin(), left_nodes.end());
    }

    // --- Inner ROI zone (uniform h_fine spacing) ---
    RealVec inner_nodes;
    {
        int n_inner = std::max(1, static_cast<int>(
            std::ceil((roi_max - roi_min) / h_fine)));
        Real dx = (roi_max - roi_min) / n_inner;
        for (int i = 0; i <= n_inner; ++i) {
            inner_nodes.push_back(roi_min + i * dx);
        }
    }

    // --- Right transition zone (roi_max → max, growing right) ---
    RealVec right_nodes;
    {
        Real pos = roi_max;
        Real h = h_fine;
        while (max - pos > h * 0.5) {
            pos += h;
            if (pos > max) pos = max;
            right_nodes.push_back(pos);
            h = std::min(h * growth, h_cap);
        }
        if (right_nodes.empty() || right_nodes.back() < max - 1e-6) {
            right_nodes.push_back(max);
        }
    }

    // --- Concatenate and deduplicate ---
    RealVec nodes;
    nodes.reserve(left_nodes.size() + inner_nodes.size() + right_nodes.size());
    for (Real v : left_nodes)  nodes.push_back(v);
    for (Real v : inner_nodes) nodes.push_back(v);
    for (Real v : right_nodes) nodes.push_back(v);

    RealVec unique_nodes;
    unique_nodes.push_back(nodes[0]);
    for (size_t i = 1; i < nodes.size(); ++i) {
        if (nodes[i] - unique_nodes.back() > 1e-3) {
            unique_nodes.push_back(nodes[i]);
        }
    }

    unique_nodes.front() = min;
    unique_nodes.back()  = max;

    return unique_nodes;
}

std::vector<Real> HexMeshGenerator3D::build_z_nodes_(
    Real terrain_elev, const MeshParams3D& params) {

    Real z_earth_total = terrain_elev - params.z_min;  // positive distance
    Real z_air_total   = params.z_air;                 // air layer height

    RealVec earth_nodes;  // absolute z-coordinates below surface

    if (params.skin_depth_mesh && !params.mesh_frequencies.empty()) {
        // ── Multi-zone skin-depth-based z-node distribution ──
        //
        // Instead of a single geometric growth from surface to bottom,
        // divide the earth into zones based on frequency bands:
        //
        //   Zone 1: surface ~ δ(f_max)     — finest (high-freq sensitivity)
        //   Zone 2: δ(f_max) ~ δ(f_mid)    — medium
        //   Zone 3: δ(f_mid) ~ δ(f_min)    — coarse
        //   Zone 4: δ(f_min) ~ δ(f_design) — very coarse (BC padding)
        //   Zone 5: δ(f_design) ~ z_min     — geometric padding to boundary
        //
        // Key insight: f_min의 임피던스를 정확히 계산하려면
        // δ(f_min) 아래 영역의 장이 올바르게 연속되어야 한다.
        // f_design = f_min/safety (1 decade below) 깊이까지
        // 적절한 해상도를 유지해야 경계조건 오차가 전파되지 않는다.

        RealVec freqs = params.mesh_frequencies;
        std::sort(freqs.begin(), freqs.end(), std::greater<Real>());  // descending

        // Build frequency bands: take ~5 representative frequencies
        // (f_max, intermediate bands, f_min, f_design)
        RealVec band_freqs;
        band_freqs.push_back(freqs.front());  // f_max
        if (freqs.size() >= 4) {
            // Add log-spaced intermediate frequencies
            Real lf_max = std::log10(freqs.front());
            Real lf_min = std::log10(freqs.back());
            int n_bands = std::min(4, static_cast<int>(freqs.size()) - 1);
            for (int i = 1; i < n_bands; ++i) {
                Real lf = lf_max - i * (lf_max - lf_min) / n_bands;
                band_freqs.push_back(std::pow(10.0, lf));
            }
        }
        band_freqs.push_back(freqs.back());   // f_min
        // Design frequency (safety margin for accurate BC at f_min)
        Real f_design = freqs.back() / params.skin_depth_safety;
        band_freqs.push_back(f_design);

        // Compute skin depths at each band frequency
        RealVec band_depths;
        for (Real f : band_freqs) {
            band_depths.push_back(skin_depth(params.rho_halfspace, f));
        }

        // Build z-nodes zone by zone (surface downward)
        earth_nodes.push_back(terrain_elev);  // surface

        Real current_z = terrain_elev;
        Real current_h = params.h_surface_z;

        for (size_t zone = 0; zone < band_depths.size(); ++zone) {
            Real zone_bottom_depth = band_depths[zone];  // depth below surface
            Real zone_bottom_z = terrain_elev - zone_bottom_depth;

            // Clamp to domain bottom
            zone_bottom_z = std::max(zone_bottom_z, params.z_min);

            if (current_z <= zone_bottom_z + 1e-3) continue;

            // Target element size for this zone: δ/N_per_skin
            // Shallow zones need more elements per skin depth.
            int n_per_skin;
            if (zone == 0)                          n_per_skin = 4;  // finest
            else if (zone < band_depths.size() - 2) n_per_skin = 3;  // medium
            else if (zone < band_depths.size() - 1) n_per_skin = 2;  // coarse
            else                                    n_per_skin = 2;  // design zone

            Real target_h = band_depths[zone] / n_per_skin;
            // Don't make elements smaller than what we already have
            target_h = std::max(target_h, current_h);

            // Growth rate within zone: gentle (1.1-1.2)
            Real zone_growth = (zone == 0) ? 1.1 : 1.2;

            Real h = current_h;
            while (current_z - zone_bottom_z > h * 0.3) {
                h = std::min(h, current_z - zone_bottom_z);  // don't overshoot
                current_z -= h;
                earth_nodes.push_back(current_z);
                // Grow h toward target, but cap at target_h
                h = std::min(h * zone_growth, target_h);
            }
            current_h = h;
        }

        // Final padding to z_min (if not already there)
        if (current_z > params.z_min + current_h * 0.3) {
            Real h = current_h;
            Real h_cap = z_earth_total * 0.25;
            while (current_z - params.z_min > h * 0.3) {
                h = std::min(h, current_z - params.z_min);
                current_z -= h;
                earth_nodes.push_back(current_z);
                h = std::min(h * 1.3, h_cap);
            }
        }

        // Ensure bottom node is exactly z_min
        if (std::abs(earth_nodes.back() - params.z_min) > 1e-3) {
            earth_nodes.push_back(params.z_min);
        }

        // Reverse to ascending order (z_min → terrain_elev)
        std::reverse(earth_nodes.begin(), earth_nodes.end());

        int nz_skin = static_cast<int>(earth_nodes.size()) - 1;
        MAPLE3DMT_LOG_INFO("Skin-depth z-nodes: " + std::to_string(nz_skin) +
                         " layers (" + std::to_string(band_depths.size()) +
                         " frequency zones)");
        // Log zone boundaries
        for (size_t i = 0; i < band_freqs.size(); ++i) {
            MAPLE3DMT_LOG_INFO("  zone " + std::to_string(i) +
                             ": f=" + std::to_string(band_freqs[i]) +
                             " Hz, δ=" + std::to_string(band_depths[i]/1000) + " km");
        }
    } else {
        // ── Original single-growth-rate approach ──
        Real h_deep_cap = z_earth_total * 0.25;
        auto earth_frac = compute_z_fractions(
            params.h_surface_z, h_deep_cap, z_earth_total, params.growth_z);

        // Reverse: frac=0 at bottom, frac=1 at surface
        {
            int n = static_cast<int>(earth_frac.size());
            RealVec reversed(n);
            for (int i = 0; i < n; ++i) {
                reversed[i] = 1.0 - earth_frac[n - 1 - i];
            }
            earth_frac = std::move(reversed);
        }

        // Convert fractions to absolute coordinates
        for (auto f : earth_frac) {
            earth_nodes.push_back(params.z_min + f * z_earth_total);
        }
    }

    // ── Air layers (same for both modes) ──
    Real h_air_cap = z_air_total * 0.25;
    auto air_frac = compute_z_fractions(
        params.h_air_start, h_air_cap, z_air_total, params.growth_air);

    int nz_air = static_cast<int>(air_frac.size()) - 1;

    // Combine earth + air into final z-node array
    RealVec z_nodes;
    z_nodes.reserve(earth_nodes.size() + nz_air);

    for (auto z : earth_nodes) {
        z_nodes.push_back(z);
    }

    // Air nodes: terrain_elev to terrain_elev + z_air
    // Skip k=0 (already added as surface node from earth_nodes)
    for (int k = 1; k <= nz_air; ++k) {
        z_nodes.push_back(terrain_elev + air_frac[k] * z_air_total);
    }

    return z_nodes;
}

// =========================================================================
// HexMeshGenerator3D — main generate()
// =========================================================================

std::unique_ptr<mfem::Mesh> HexMeshGenerator3D::generate(
    const MeshParams3D& params,
    const std::vector<Station3D>& stations,
    const ALOSDem* dem) {

    MAPLE3DMT_LOG_INFO("Generating 3D hexahedral mesh...");

    // ------------------------------------------------------------------
    // Step 1: Compute ROI bounds from stations
    // ------------------------------------------------------------------
    Real roi_x_min, roi_x_max, roi_y_min, roi_y_max;
    if (!stations.empty()) {
        Real sx_min = stations[0].x, sx_max = stations[0].x;
        Real sy_min = stations[0].y, sy_max = stations[0].y;
        for (const auto& s : stations) {
            sx_min = std::min(sx_min, s.x);
            sx_max = std::max(sx_max, s.x);
            sy_min = std::min(sy_min, s.y);
            sy_max = std::max(sy_max, s.y);
        }
        roi_x_min = sx_min - params.roi_x_pad;
        roi_x_max = sx_max + params.roi_x_pad;
        roi_y_min = sy_min - params.roi_y_pad;
        roi_y_max = sy_max + params.roi_y_pad;
    } else {
        // Default: center 20% of domain
        Real xspan = params.x_max - params.x_min;
        Real yspan = params.y_max - params.y_min;
        roi_x_min = params.x_min + 0.4 * xspan;
        roi_x_max = params.x_min + 0.6 * xspan;
        roi_y_min = params.y_min + 0.4 * yspan;
        roi_y_max = params.y_min + 0.6 * yspan;
    }
    // Clamp to domain
    roi_x_min = std::max(roi_x_min, params.x_min);
    roi_x_max = std::min(roi_x_max, params.x_max);
    roi_y_min = std::max(roi_y_min, params.y_min);
    roi_y_max = std::min(roi_y_max, params.y_max);

    // ------------------------------------------------------------------
    // Step 2: Build 1D node arrays
    // ------------------------------------------------------------------
    auto x_nodes = build_axis_nodes_(params.x_min, params.x_max,
                                     params.h_surface_x, params.growth_x,
                                     roi_x_min, roi_x_max);
    auto y_nodes = build_axis_nodes_(params.y_min, params.y_max,
                                     params.h_surface_y, params.growth_y,
                                     roi_y_min, roi_y_max);

    // Z fractions (reference column at terrain_elev = 0)
    Real z_earth_total = -params.z_min;  // positive
    Real z_air_total   = params.z_air;

    Real h_deep_cap = z_earth_total * 0.25;
    auto earth_frac = compute_z_fractions(
        params.h_surface_z, h_deep_cap, z_earth_total, params.growth_z);
    {
        int n = static_cast<int>(earth_frac.size());
        RealVec reversed(n);
        for (int i = 0; i < n; ++i) {
            reversed[i] = 1.0 - earth_frac[n - 1 - i];
        }
        earth_frac = std::move(reversed);
    }

    Real h_air_cap = z_air_total * 0.25;
    auto air_frac = compute_z_fractions(
        params.h_air_start, h_air_cap, z_air_total, params.growth_air);

    int nz_earth = static_cast<int>(earth_frac.size()) - 1;
    int nz_air   = static_cast<int>(air_frac.size()) - 1;
    int nz       = nz_earth + nz_air;

    int nx = static_cast<int>(x_nodes.size()) - 1;
    int ny = static_cast<int>(y_nodes.size()) - 1;

    MAPLE3DMT_LOG_INFO("  Grid: " + std::to_string(nx) + " x " +
                     std::to_string(ny) + " x " + std::to_string(nz) +
                     " (" + std::to_string(nz_earth) + " earth + " +
                     std::to_string(nz_air) + " air)");

    // ------------------------------------------------------------------
    // Step 3: Build coordinate transform for DEM lookup
    // ------------------------------------------------------------------
    // Simple offset-based transform: (x, y) → (lon, lat)
    // Computed from station data if available
    Real lon0 = 0.0, lat0 = 0.0;
    Real m_per_deg_lon = 111320.0;  // approximate at equator
    Real m_per_deg_lat = 110540.0;
    bool has_geo_transform = false;

    if (!stations.empty()) {
        // Average station geographic coords as origin
        Real sum_lon = 0, sum_lat = 0;
        Real sum_x = 0, sum_y = 0;
        for (const auto& s : stations) {
            sum_lon += s.lon;
            sum_lat += s.lat;
            sum_x += s.x;
            sum_y += s.y;
        }
        int ns = static_cast<int>(stations.size());
        lon0 = sum_lon / ns;
        lat0 = sum_lat / ns;
        Real x0 = sum_x / ns;
        Real y0 = sum_y / ns;

        // Meters per degree at this latitude
        m_per_deg_lat = 110540.0;
        m_per_deg_lon = 111320.0 * std::cos(lat0 * constants::PI / 180.0);

        // Adjust origin so that (x0, y0) maps to (lon0, lat0)
        lon0 -= x0 / m_per_deg_lon;
        lat0 -= y0 / m_per_deg_lat;
        has_geo_transform = true;
    }

    // Lambda: local (x,y) → terrain elevation
    bool dem_warning_issued = false;
    auto terrain_at = [&](Real x, Real y) -> Real {
        if (!dem || !params.use_terrain) return 0.0;
        if (has_geo_transform) {
            Real qlon = lon0 + x / m_per_deg_lon;
            Real qlat = lat0 + y / m_per_deg_lat;
            // Warn once if query falls outside DEM coverage (clamping occurs)
            if (!dem_warning_issued && !dem->lon.empty() &&
                (qlon < dem->lon.front() || qlon > dem->lon.back() ||
                 qlat < dem->lat.front() || qlat > dem->lat.back())) {
                MAPLE3DMT_LOG_WARNING(
                    "Mesh domain extends beyond DEM coverage. "
                    "Terrain will be clamped at DEM boundary.");
                dem_warning_issued = true;
            }
            return dem->interpolate(qlon, qlat);
        }
        // Fallback: assume DEM coords match local coords
        return dem->interpolate(x, y);
    };

    // ------------------------------------------------------------------
    // Step 4: Build hex mesh with TFI terrain deformation
    // ------------------------------------------------------------------
    int nv_total = (nx + 1) * (ny + 1) * (nz + 1);
    int ne_total = nx * ny * nz;
    // Boundary quads: 6 faces
    int nb_total = 2 * (nx * ny) + 2 * (ny * nz) + 2 * (nx * nz);

    auto* raw_mesh = new mfem::Mesh(3, nv_total, ne_total, nb_total);

    // Vertex index helper: V(i,j,k)
    auto V = [&](int i, int j, int k) -> int {
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
    };

    // Add vertices with TFI terrain deformation
    for (int k = 0; k <= nz; ++k) {
        bool is_earth = (k <= nz_earth);
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                Real x = x_nodes[i];
                Real y = y_nodes[j];
                Real topo_z = terrain_at(x, y);

                Real z;
                if (is_earth) {
                    // k ∈ [0, nz_earth]: z_min to surface
                    Real frac = earth_frac[k];
                    z = params.z_min + frac * (topo_z - params.z_min);
                } else {
                    // k ∈ (nz_earth, nz]: surface to air top
                    int air_k = k - nz_earth;
                    Real frac = air_frac[air_k];
                    z = topo_z + frac * z_air_total;
                }

                double coords[3] = {x, y, z};
                raw_mesh->AddVertex(coords);
            }
        }
    }

    // Add hex elements
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int v[8] = {
                    V(i,   j,   k),
                    V(i+1, j,   k),
                    V(i+1, j+1, k),
                    V(i,   j+1, k),
                    V(i,   j,   k+1),
                    V(i+1, j,   k+1),
                    V(i+1, j+1, k+1),
                    V(i,   j+1, k+1)
                };

                // Attribute: earth=1 if below surface, air=2 if above
                int attr = (k < nz_earth) ? 1 : 2;
                raw_mesh->AddHex(v, attr);
            }
        }
    }

    // Add boundary quads
    // Bottom face (k=0): bdr_attr=1
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int bv[4] = {V(i,j,0), V(i+1,j,0), V(i+1,j+1,0), V(i,j+1,0)};
            raw_mesh->AddBdrQuad(bv, 1);
        }
    }
    // Top face (k=nz): bdr_attr=2
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int bv[4] = {V(i,j,nz), V(i+1,j,nz), V(i+1,j+1,nz), V(i,j+1,nz)};
            raw_mesh->AddBdrQuad(bv, 2);
        }
    }
    // Front face (j=0): bdr_attr=3
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            int bv[4] = {V(i,0,k), V(i+1,0,k), V(i+1,0,k+1), V(i,0,k+1)};
            raw_mesh->AddBdrQuad(bv, 3);
        }
    }
    // Back face (j=ny): bdr_attr=4
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            int bv[4] = {V(i,ny,k), V(i+1,ny,k), V(i+1,ny,k+1), V(i,ny,k+1)};
            raw_mesh->AddBdrQuad(bv, 4);
        }
    }
    // Left face (i=0): bdr_attr=5
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            int bv[4] = {V(0,j,k), V(0,j+1,k), V(0,j+1,k+1), V(0,j,k+1)};
            raw_mesh->AddBdrQuad(bv, 5);
        }
    }
    // Right face (i=nx): bdr_attr=6
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            int bv[4] = {V(nx,j,k), V(nx,j+1,k), V(nx,j+1,k+1), V(nx,j,k+1)};
            raw_mesh->AddBdrQuad(bv, 6);
        }
    }

    // FinalizeHexMesh(generate_edges, refine, fix_orientation)
    // refine=0: skip MarkHexMeshForRefinement() which reorders face
    // vertices and can corrupt ParMesh shared face identification.
    // (MarkHexMeshForRefinement is only needed for non-conforming AMR,
    // which is disabled due to MFEM 4.9 hex NCMesh limitation.)
    raw_mesh->FinalizeHexMesh(1, 0, true);

    auto mesh = std::unique_ptr<mfem::Mesh>(raw_mesh);

    MAPLE3DMT_LOG_INFO("  Initial mesh: " + std::to_string(mesh->GetNE()) +
                     " elements, " + std::to_string(mesh->GetNV()) +
                     " vertices");

    // ------------------------------------------------------------------
    // Step 5: Refine near stations (serial mesh)
    // NOTE: For PARALLEL builds, set refine_near_stations=0 here and call
    //       refine_near_stations_parallel() on the ParMesh instead.
    //       Doing GeneralRefinement on the serial mesh creates NCMesh
    //       that can fail in ParMesh (GetSharedFaceTransformationsByLocalIndex).
    // ------------------------------------------------------------------
    if (params.refine_near_stations > 0 && !stations.empty()) {
        Real refine_radius = 2.0 * std::max(params.h_surface_x,
                                             params.h_surface_y);
        for (int level = 0; level < params.refine_near_stations; ++level) {
            if (level > 0) refine_radius *= 0.5;
            mfem::Array<int> marked;
            for (int e = 0; e < mesh->GetNE(); ++e) {
                mfem::Array<int> verts;
                mesh->GetElementVertices(e, verts);

                Real cx = 0, cy = 0;
                for (int vi = 0; vi < verts.Size(); ++vi) {
                    const Real* vcoord = mesh->GetVertex(verts[vi]);
                    cx += vcoord[0];
                    cy += vcoord[1];
                }
                cx /= verts.Size();
                cy /= verts.Size();

                for (const auto& s : stations) {
                    Real dx = cx - s.x;
                    Real dy = cy - s.y;
                    Real dist_h = std::sqrt(dx * dx + dy * dy);
                    if (dist_h < refine_radius) {
                        marked.Append(e);
                        break;
                    }
                }
            }

            if (marked.Size() > 0) {
                mesh->GeneralRefinement(marked);
                MAPLE3DMT_LOG_INFO("  Refinement level " +
                                 std::to_string(level + 1) + ": refined " +
                                 std::to_string(marked.Size()) +
                                 " elements -> " +
                                 std::to_string(mesh->GetNE()) + " total");
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 6: Build region map
    // ------------------------------------------------------------------
    region_map_.resize(mesh->GetNE());
    for (int e = 0; e < mesh->GetNE(); ++e) {
        region_map_[e] = mesh->GetAttribute(e);
    }

    MAPLE3DMT_LOG_INFO("Mesh generation complete: " +
                     std::to_string(mesh->GetNE()) + " elements, " +
                     std::to_string(mesh->GetNV()) + " vertices");

    return mesh;
}

// =========================================================================
// apply_terrain_ — post-hoc terrain deformation
// =========================================================================

void HexMeshGenerator3D::apply_terrain_(
    mfem::Mesh& mesh, const ALOSDem& dem,
    const MeshParams3D& params) {

    // This method deforms an existing flat mesh to conform to DEM terrain.
    // It identifies vertex columns (same x,y) and rescales z-coordinates.
    // NOTE: generate() already applies TFI inline, so this is for cases
    // where the DEM is provided after initial mesh creation.

    int nv = mesh.GetNV();
    if (nv == 0) return;

    // Group vertices by (x, y) column
    // Tolerance for matching x, y coordinates
    const Real tol = 1e-3;

    // Collect all unique (x, y) pairs
    struct Column {
        Real x, y;
        std::vector<int> vert_indices;  // sorted by z ascending
    };
    std::vector<Column> columns;

    for (int v = 0; v < nv; ++v) {
        const Real* coords = mesh.GetVertex(v);
        Real vx = coords[0], vy = coords[1];

        // Find existing column
        bool found = false;
        for (auto& col : columns) {
            if (std::abs(col.x - vx) < tol && std::abs(col.y - vy) < tol) {
                col.vert_indices.push_back(v);
                found = true;
                break;
            }
        }
        if (!found) {
            columns.push_back({vx, vy, {v}});
        }
    }

    // Sort each column's vertices by z
    for (auto& col : columns) {
        std::sort(col.vert_indices.begin(), col.vert_indices.end(),
                  [&mesh](int a, int b) {
                      return mesh.GetVertex(a)[2] < mesh.GetVertex(b)[2];
                  });
    }

    // Deform each column
    Real z_min = params.z_min;
    Real z_air = params.z_air;

    for (auto& col : columns) {
        Real terrain_z = dem.interpolate(col.x, col.y);
        int ncol = static_cast<int>(col.vert_indices.size());

        // Find surface vertex (closest to z=0 in the original flat mesh)
        int surface_idx = 0;
        Real min_dist = std::abs(mesh.GetVertex(col.vert_indices[0])[2]);
        for (int k = 1; k < ncol; ++k) {
            Real dist = std::abs(mesh.GetVertex(col.vert_indices[k])[2]);
            if (dist < min_dist) {
                min_dist = dist;
                surface_idx = k;
            }
        }

        // Rescale earth vertices (below surface)
        Real flat_earth_range = 0.0 - z_min;  // original flat range
        Real new_earth_range = terrain_z - z_min;
        for (int k = 0; k <= surface_idx; ++k) {
            int vi = col.vert_indices[k];
            Real old_z = mesh.GetVertex(vi)[2];
            Real frac = (old_z - z_min) / flat_earth_range;
            mesh.GetVertex(vi)[2] = z_min + frac * new_earth_range;
        }

        // Rescale air vertices (above surface)
        Real flat_air_range = z_air;  // original flat range above 0
        for (int k = surface_idx + 1; k < ncol; ++k) {
            int vi = col.vert_indices[k];
            Real old_z = mesh.GetVertex(vi)[2];
            Real frac = old_z / flat_air_range;  // old_z was relative to 0
            mesh.GetVertex(vi)[2] = terrain_z + frac * z_air;
        }
    }

    // Update element attributes based on deformed positions
    for (int e = 0; e < mesh.GetNE(); ++e) {
        mfem::Array<int> verts;
        mesh.GetElementVertices(e, verts);

        Real cx = 0, cy = 0, cz = 0;
        for (int vi = 0; vi < verts.Size(); ++vi) {
            const Real* vc = mesh.GetVertex(verts[vi]);
            cx += vc[0];
            cy += vc[1];
            cz += vc[2];
        }
        cx /= verts.Size();
        cy /= verts.Size();
        cz /= verts.Size();

        Real terrain_z = dem.interpolate(cx, cy);
        mesh.SetAttribute(e, (cz <= terrain_z) ? 1 : 2);
    }
}

// =========================================================================
// generate_parallel
// =========================================================================

#ifdef MAPLE3DMT_USE_MPI
std::unique_ptr<mfem::ParMesh> HexMeshGenerator3D::generate_parallel(
    const MeshParams3D& params,
    const std::vector<Station3D>& stations,
    const ALOSDem* dem,
    MPI_Comm comm) {

    // All ranks generate identical serial mesh (deterministic, conforming).
    // Non-conforming h-refinement is disabled due to MFEM 4.9 hex NCMesh
    // limitation (GetSharedFaceTransformationsByLocalIndex assertion).
    MeshParams3D p = params;
    p.refine_near_stations = 0;
    auto serial_mesh = generate(p, stations, dem);

    // Create conforming ParMesh (no NCMesh — avoids MFEM 4.9 bug)
    auto par_mesh = std::make_unique<mfem::ParMesh>(comm, *serial_mesh);
    serial_mesh.reset();

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        MAPLE3DMT_LOG_INFO("ParMesh created: " +
                         std::to_string(par_mesh->GetNE()) +
                         " local elements on rank 0");
        if (params.refine_near_stations > 0) {
            MAPLE3DMT_LOG_WARNING("h-refinement disabled (MFEM 4.9 hex NCMesh "
                                "limitation). Use finer h_surface instead.");
        }
    }

    return par_mesh;
}

// =========================================================================
// refine_near_stations_parallel — parallel-safe h-refinement
// =========================================================================
void HexMeshGenerator3D::refine_near_stations_parallel(
    mfem::ParMesh& pmesh,
    const std::vector<Station3D>& stations,
    int levels, Real h_surface_x, Real h_surface_y) {

    Real refine_radius = 2.0 * std::max(h_surface_x, h_surface_y);

    for (int level = 0; level < levels; ++level) {
        if (level > 0) refine_radius *= 0.5;

        mfem::Array<int> marked;
        int ne_local = pmesh.GetNE();

        for (int e = 0; e < ne_local; ++e) {
            mfem::Array<int> verts;
            pmesh.GetElementVertices(e, verts);

            Real cx = 0, cy = 0;
            for (int vi = 0; vi < verts.Size(); ++vi) {
                const Real* vcoord = pmesh.GetVertex(verts[vi]);
                cx += vcoord[0];
                cy += vcoord[1];
            }
            cx /= verts.Size();
            cy /= verts.Size();

            for (const auto& s : stations) {
                Real dx = cx - s.x;
                Real dy = cy - s.y;
                Real dist_h = std::sqrt(dx * dx + dy * dy);
                if (dist_h < refine_radius) {
                    marked.Append(e);
                    break;
                }
            }
        }

        // ParMesh::GeneralRefinement handles non-conforming shared faces
        // correctly, unlike serial-mesh-then-partition approach.
        if (marked.Size() > 0) {
            pmesh.GeneralRefinement(marked);
        }

        // Log from rank 0
        int rank;
        MPI_Comm_rank(pmesh.GetComm(), &rank);
        int ne_global = 0, ne_local_new = pmesh.GetNE();
        MPI_Allreduce(&ne_local_new, &ne_global, 1, MPI_INT, MPI_SUM,
                       pmesh.GetComm());
        if (rank == 0) {
            MAPLE3DMT_LOG_INFO("  Parallel refinement level " +
                             std::to_string(level + 1) + ": " +
                             std::to_string(ne_global) + " total elements");
        }
    }
}

#else
std::unique_ptr<mfem::ParMesh> HexMeshGenerator3D::generate_parallel(
    const MeshParams3D&,
    const std::vector<Station3D>&,
    const ALOSDem*,
    MPI_Comm) {
    throw std::runtime_error(
        "generate_parallel requires MPI. Rebuild with -DMAPLE3DMT_USE_MPI=ON");
}

void HexMeshGenerator3D::refine_near_stations_parallel(
    mfem::ParMesh&, const std::vector<Station3D>&,
    int, Real, Real) {
    throw std::runtime_error(
        "refine_near_stations_parallel requires MPI.");
}
#endif

// =========================================================================
// export_vtk
// =========================================================================

void HexMeshGenerator3D::export_vtk(const mfem::Mesh& mesh,
                                     const fs::path& path) {
    std::string ext = path.extension().string();
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open VTK file: " + path.string());
    }

    // MFEM PrintVTK/PrintVTU are non-const
    auto& mesh_nc = const_cast<mfem::Mesh&>(mesh);
    if (ext == ".vtu") {
        mesh_nc.PrintVTU(ofs);
    } else {
        mesh_nc.PrintVTK(ofs);
    }

    MAPLE3DMT_LOG_INFO("Exported mesh to " + path.string());
}

void HexMeshGenerator3D::export_surface_vtk(const mfem::Mesh& mesh,
                                             const fs::path& path) {
    // Export only the boundary faces as a surface mesh (much smaller file).
    // This is for GUI preview — ~1-3 MB vs ~150 MB for the full volume.
    int ne = mesh.GetNE();
    int nv = mesh.GetNV();
    int nbe = mesh.GetNBE();

    // Collect boundary face vertices and material info
    // For a hex mesh, each boundary face is a quad (4 vertices)
    struct SurfFace {
        int v[4];
        int material;  // 1=earth, 2=air
    };
    std::vector<SurfFace> faces;
    faces.reserve(nbe);

    for (int i = 0; i < nbe; ++i) {
        const mfem::Element* be = mesh.GetBdrElement(i);
        mfem::Array<int> verts;
        be->GetVertices(verts);

        SurfFace f;
        int nv_face = std::min(verts.Size(), 4);
        for (int j = 0; j < nv_face; ++j) f.v[j] = verts[j];
        for (int j = nv_face; j < 4; ++j) f.v[j] = verts[nv_face - 1];

        // Get parent element material
        int elem1, elem2;
        mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(i), &elem1, &elem2);
        int parent = (elem1 >= 0) ? elem1 : elem2;
        f.material = (parent >= 0 && parent < (int)region_map_.size())
                     ? region_map_[parent] : 0;
        faces.push_back(f);
    }

    // Also add earth/air interface faces (internal boundaries)
    for (int f = 0; f < mesh.GetNumFaces(); ++f) {
        int e1, e2;
        mesh.GetFaceElements(f, &e1, &e2);
        if (e1 >= 0 && e2 >= 0 &&
            e1 < (int)region_map_.size() && e2 < (int)region_map_.size()) {
            if (region_map_[e1] != region_map_[e2]) {
                // Earth/air boundary — add as earth face
                mfem::Array<int> verts;
                mesh.GetFaceVertices(f, verts);
                SurfFace sf;
                int nv_f = std::min(verts.Size(), 4);
                for (int j = 0; j < nv_f; ++j) sf.v[j] = verts[j];
                for (int j = nv_f; j < 4; ++j) sf.v[j] = verts[nv_f - 1];
                sf.material = 1;  // mark as earth surface
                faces.push_back(sf);
            }
        }
    }

    // Collect unique vertices used
    std::unordered_map<int, int> vert_map;
    for (const auto& f : faces) {
        for (int j = 0; j < 4; ++j) {
            if (vert_map.find(f.v[j]) == vert_map.end()) {
                int idx = (int)vert_map.size();
                vert_map[f.v[j]] = idx;
            }
        }
    }

    // Write VTK
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open: " + path.string());
    }

    ofs << "# vtk DataFile Version 3.0\n"
        << "Maple3DMT surface mesh\n"
        << "ASCII\n"
        << "DATASET UNSTRUCTURED_GRID\n";

    // Points
    int n_used = (int)vert_map.size();
    std::vector<const Real*> pts(n_used);
    for (const auto& [orig, idx] : vert_map) {
        pts[idx] = mesh.GetVertex(orig);
    }

    ofs << "POINTS " << n_used << " double\n";
    for (int i = 0; i < n_used; ++i) {
        ofs << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << "\n";
    }

    // Cells (quads = VTK type 9)
    int nf = (int)faces.size();
    ofs << "\nCELLS " << nf << " " << nf * 5 << "\n";
    for (const auto& f : faces) {
        ofs << "4";
        for (int j = 0; j < 4; ++j) ofs << " " << vert_map[f.v[j]];
        ofs << "\n";
    }

    ofs << "\nCELL_TYPES " << nf << "\n";
    for (int i = 0; i < nf; ++i) ofs << "9\n";  // VTK_QUAD

    // Material data
    ofs << "\nCELL_DATA " << nf << "\n"
        << "SCALARS material int 1\n"
        << "LOOKUP_TABLE default\n";
    for (const auto& f : faces) ofs << f.material << "\n";

    ofs.close();
    MAPLE3DMT_LOG_INFO("Exported surface mesh to " + path.string() +
                     " (" + std::to_string(nf) + " faces, " +
                     std::to_string(n_used) + " vertices)");
}

// =========================================================================
// Utility functions
// =========================================================================

std::vector<Station3D> stations_from_mt_data(const data::MTData& mt_data) {
    int ns = mt_data.num_stations();
    if (ns == 0) return {};

    // Compute average geographic origin
    Real sum_lon = 0, sum_lat = 0;
    int n_geo = 0;
    for (int i = 0; i < ns; ++i) {
        const auto& s = mt_data.station(i);
        if (s.has_geo) {
            sum_lon += s.lon;
            sum_lat += s.lat;
            ++n_geo;
        }
    }

    Real lon0 = 0, lat0 = 0;
    Real m_per_deg_lon = 111320.0;
    Real m_per_deg_lat = 110540.0;

    if (n_geo > 0) {
        lon0 = sum_lon / n_geo;
        lat0 = sum_lat / n_geo;
        m_per_deg_lon = 111320.0 * std::cos(lat0 * constants::PI / 180.0);
    }

    std::vector<Station3D> result;
    result.reserve(ns);

    for (int i = 0; i < ns; ++i) {
        const auto& s = mt_data.station(i);
        Station3D s3d;
        s3d.name = s.name;
        s3d.lon = s.lon;
        s3d.lat = s.lat;
        s3d.elevation = s.z;

        if (s.has_geo && n_geo > 0) {
            s3d.x = (s.lon - lon0) * m_per_deg_lon;
            s3d.y = (s.lat - lat0) * m_per_deg_lat;
            s3d.z = s.z;
        } else {
            // Use existing local coordinates
            s3d.x = s.x;
            s3d.y = s.y;
            s3d.z = s.z;
        }

        result.push_back(std::move(s3d));
    }

    MAPLE3DMT_LOG_INFO("Converted " + std::to_string(ns) +
                     " stations to local coords (origin: " +
                     std::to_string(lon0) + ", " + std::to_string(lat0) + ")");
    return result;
}

MeshParams3D auto_mesh_params(const std::vector<Station3D>& stations) {
    MeshParams3D p;

    if (stations.empty()) return p;

    // Station extent
    Real sx_min = stations[0].x, sx_max = stations[0].x;
    Real sy_min = stations[0].y, sy_max = stations[0].y;
    for (const auto& s : stations) {
        sx_min = std::min(sx_min, s.x);
        sx_max = std::max(sx_max, s.x);
        sy_min = std::min(sy_min, s.y);
        sy_max = std::max(sy_max, s.y);
    }

    Real x_range = sx_max - sx_min;
    Real y_range = sy_max - sy_min;

    // Domain: 3× station extent, minimum ±50km
    Real x_pad = std::max(50000.0, x_range * 1.5);
    Real y_pad = std::max(50000.0, y_range * 1.5);
    Real x_center = 0.5 * (sx_min + sx_max);
    Real y_center = 0.5 * (sy_min + sy_max);

    p.x_min = x_center - x_pad;
    p.x_max = x_center + x_pad;
    p.y_min = y_center - y_pad;
    p.y_max = y_center + y_pad;

    // Surface resolution: half of minimum inter-station spacing
    if (stations.size() > 1) {
        Real min_dist = 1e30;
        for (size_t i = 0; i < stations.size(); ++i) {
            for (size_t j = i + 1; j < stations.size(); ++j) {
                Real dx = stations[i].x - stations[j].x;
                Real dy = stations[i].y - stations[j].y;
                Real d = std::sqrt(dx * dx + dy * dy);
                if (d > 1e-3) min_dist = std::min(min_dist, d);
            }
        }
        if (min_dist < 1e30) {
            Real h = std::max(200.0, std::min(2000.0, min_dist * 0.5));
            p.h_surface_x = h;
            p.h_surface_y = h;
        }
    }

    // ROI padding: 20% of station range, minimum 5km
    p.roi_x_pad = std::max(5000.0, x_range * 0.2);
    p.roi_y_pad = std::max(5000.0, y_range * 0.2);

    MAPLE3DMT_LOG_INFO("Auto mesh params: domain [" +
                     std::to_string(p.x_min/1000) + ", " +
                     std::to_string(p.x_max/1000) + "] x [" +
                     std::to_string(p.y_min/1000) + ", " +
                     std::to_string(p.y_max/1000) + "] km, " +
                     "h_surface=" + std::to_string(p.h_surface_x) + " m");
    return p;
}

// =========================================================================
// auto_mesh_params — skin-depth-aware overload
// =========================================================================
MeshParams3D auto_mesh_params(const std::vector<Station3D>& stations,
                              const RealVec& frequencies,
                              Real rho_half,
                              Real safety) {
    // Start from station-based defaults
    MeshParams3D p = auto_mesh_params(stations);

    if (frequencies.empty()) return p;

    // Sort frequencies (ascending)
    RealVec freqs = frequencies;
    std::sort(freqs.begin(), freqs.end());
    Real f_min = freqs.front();
    Real f_max = freqs.back();

    // Design frequency: one decade (or safety factor) below actual f_min.
    // To accurately solve f_min, the domain must accommodate field behavior
    // at f_design because: (1) boundary conditions need negligible fields,
    // (2) deep structure affects impedance via upward field continuation.
    Real f_design = f_min / safety;

    // Skin depths at key frequencies (δ = 503.3 × √(ρ/f) meters)
    Real delta_fmax    = skin_depth(rho_half, f_max);     // shallowest penetration
    Real delta_fmin    = skin_depth(rho_half, f_min);     // deepest data freq
    Real delta_design  = skin_depth(rho_half, f_design);  // design (safety margin)

    MAPLE3DMT_LOG_INFO("Skin depth mesh optimization:");
    MAPLE3DMT_LOG_INFO("  f_max=" + std::to_string(f_max) + " Hz → δ=" +
                     std::to_string(delta_fmax/1000) + " km");
    MAPLE3DMT_LOG_INFO("  f_min=" + std::to_string(f_min) + " Hz → δ=" +
                     std::to_string(delta_fmin/1000) + " km");
    MAPLE3DMT_LOG_INFO("  f_design=" + std::to_string(f_design) + " Hz → δ=" +
                     std::to_string(delta_design/1000) + " km (safety=" +
                     std::to_string(safety) + "×)");

    // ── Surface resolution: resolve highest-frequency skin depth ──
    // At least 4 elements within δ_fmax for accurate surface fields.
    Real h_z_skin = delta_fmax / 4.0;
    p.h_surface_z = std::max(10.0, std::min(p.h_surface_z, h_z_skin));

    // ── Domain depth: 3× design skin depth ──
    // Fields at depth 3δ are ~5% of surface → adequate BC.
    Real required_depth = 3.0 * delta_design;
    p.z_min = -std::max(required_depth, std::abs(p.z_min));

    // ── Air layer: 3× highest skin depth (but air σ≈0, so generous) ──
    p.z_air = std::max(p.z_air, 3.0 * delta_fmax);

    // ── ROI depth: 1.5× δ at actual f_min ──
    // The region where inversion has meaningful resolution.
    p.roi_depth = std::max(p.roi_depth, 1.5 * delta_fmin);

    // ── Enable skin-depth z-node optimization ──
    p.skin_depth_mesh = true;
    p.rho_halfspace = rho_half;
    p.skin_depth_safety = safety;
    p.mesh_frequencies = freqs;

    MAPLE3DMT_LOG_INFO("  h_surface_z=" + std::to_string(p.h_surface_z) + " m" +
                     "  z_min=" + std::to_string(p.z_min/1000) + " km" +
                     "  z_air=" + std::to_string(p.z_air/1000) + " km" +
                     "  roi_depth=" + std::to_string(p.roi_depth/1000) + " km");

    return p;
}

void export_stations_csv(const std::vector<Station3D>& stations,
                         const fs::path& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + path.string());
    }

    ofs << "name,x,y,z,lon,lat,elevation\n";
    for (const auto& s : stations) {
        ofs << s.name << ","
            << s.x << "," << s.y << "," << s.z << ","
            << s.lon << "," << s.lat << "," << s.elevation << "\n";
    }

    MAPLE3DMT_LOG_INFO("Exported " + std::to_string(stations.size()) +
                     " stations to " + path.string());
}

} // namespace mesh
} // namespace maple3dmt
