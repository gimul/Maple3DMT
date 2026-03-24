// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file hex_mesh_3d.h
/// @brief 3D hexahedral mesh generation for MT inversion.
///
/// Generates terrain-conforming hexahedral meshes with:
///   - ALOS DEM (30m resolution) terrain surface fitting
///   - Geometric vertical growth from surface to depth
///   - Padding zones with progressive coarsening
///   - Air layer above terrain
///   - Station-centered local refinement

#include "maple3dmt/common.h"
#include "maple3dmt/mesh/dem.h"
#include "maple3dmt/data/mt_data.h"
#include <mfem.hpp>
#include <memory>
#include <vector>
#include <functional>
#ifdef MAPLE3DMT_USE_MPI
#include <mpi.h>
#endif

namespace maple3dmt {
namespace mesh {

/// 3D mesh generation parameters.
struct MeshParams3D {
    // Domain extents (in meters, centered on survey)
    Real x_min = -50000, x_max = 50000;
    Real y_min = -50000, y_max = 50000;
    Real z_min = -100000;  // depth (negative = below surface)
    Real z_air = 50000;    // air layer height above max terrain

    // Surface resolution
    Real h_surface_x = 500;    // horizontal element size at surface (m)
    Real h_surface_y = 500;
    Real h_surface_z = 50;     // first layer thickness below surface

    // Air layer resolution (separate from ground — air needs fewer elements)
    Real h_air_start = 500;    // first air layer thickness (m), ≫ h_surface_z
                               // 500m is fine: σ_air≈0, no current, E varies slowly

    // Growth rates
    Real growth_x = 1.3;       // horizontal padding growth
    Real growth_y = 1.3;
    Real growth_z = 1.15;      // vertical growth rate
    Real growth_air = 2.0;     // air layer growth rate (fast coarsening: σ≈0)

    // ROI (Region of Interest)
    Real roi_x_pad = 5000;     // padding around stations (m)
    Real roi_y_pad = 5000;
    Real roi_depth = 30000;    // ROI depth below surface

    // Terrain
    bool use_terrain = true;
    fs::path dem_path;          // ALOS DEM file path

    // Refinement
    int  refine_near_stations = 1;  // levels of h-refinement near stations

    // Skin-depth-based mesh optimization.
    // When set, auto_mesh_params computes h_surface_z, growth_z, roi_depth,
    // z_min from electromagnetic skin depth at each frequency band.
    // f_design = f_min / skin_depth_safety → one decade below actual f_min,
    // ensuring boundary conditions and deep field behavior are accurate.
    bool   skin_depth_mesh    = false;  // enable skin-depth-based z-node optimization
    Real   skin_depth_safety  = 10.0;   // design freq = f_min / this (default: 1 decade)
    Real   rho_halfspace      = 100.0;  // initial half-space resistivity (Ωm)
    RealVec mesh_frequencies;           // frequencies (Hz) for skin depth computation
};

/// Station location in geographic coordinates.
struct Station3D {
    std::string name;
    Real lon, lat, elevation;   // geographic coords + elevation (m)
    Real x, y, z;               // projected local coords (m)
};

/// 3D hexahedral mesh generator.
///
/// Produces a terrain-conforming hex mesh suitable for edge-element FEM.
/// The mesh has these zones (bottom to top):
///   1. Deep zone: coarse elements, geometric growth downward
///   2. ROI zone: fine elements around stations
///   3. Surface zone: terrain-fitted, finest elements
///   4. Air zone: geometric growth upward
class HexMeshGenerator3D {
public:
    HexMeshGenerator3D() = default;

    /// Generate mesh from parameters and station locations.
    /// Returns MFEM Mesh (serial) that can be ParMesh'd later.
    std::unique_ptr<mfem::Mesh> generate(
        const MeshParams3D& params,
        const std::vector<Station3D>& stations,
        const ALOSDem* dem = nullptr);

    /// Generate parallel mesh directly.
    std::unique_ptr<mfem::ParMesh> generate_parallel(
        const MeshParams3D& params,
        const std::vector<Station3D>& stations,
        const ALOSDem* dem,
        MPI_Comm comm);

    /// Refine a ParMesh near station locations (parallel-safe).
    /// This must be called AFTER creating the ParMesh from a conforming
    /// serial mesh. Doing refinement on the ParMesh (instead of serial mesh)
    /// ensures MFEM properly handles non-conforming shared faces.
    static void refine_near_stations_parallel(
        mfem::ParMesh& pmesh,
        const std::vector<Station3D>& stations,
        int levels, Real h_surface_x, Real h_surface_y);

    /// Export mesh to VTK for visualization (full volume).
    void export_vtk(const mfem::Mesh& mesh, const fs::path& path);

    /// Export surface-only VTK for lightweight GUI preview.
    void export_surface_vtk(const mfem::Mesh& mesh, const fs::path& path);

    /// Get element-to-region mapping (air=0, ground=1, ...).
    const std::vector<int>& region_map() const { return region_map_; }

private:
    std::vector<int> region_map_;

    /// Build 1D node distribution along an axis with refinement zones.
    std::vector<Real> build_axis_nodes_(
        Real min, Real max, Real h_fine, Real growth,
        Real roi_min, Real roi_max);

    /// Build vertical nodes with terrain conforming.
    std::vector<Real> build_z_nodes_(
        Real terrain_elev, const MeshParams3D& params);

    /// Deform flat mesh to conform to terrain surface.
    void apply_terrain_(mfem::Mesh& mesh, const ALOSDem& dem,
                        const MeshParams3D& params);
};

// =========================================================================
// Utility functions
// =========================================================================

/// Convert MTData stations to Station3D with geographic → local projection.
/// Uses average (lon, lat) as origin with latitude-corrected metric scaling.
std::vector<Station3D> stations_from_mt_data(const data::MTData& mt_data);

/// Auto-compute mesh parameters from station distribution.
/// Domain sized to 3× station extent (min ±50km), surface resolution
/// based on minimum inter-station spacing.
MeshParams3D auto_mesh_params(const std::vector<Station3D>& stations);

/// Auto-compute mesh parameters with skin-depth optimization.
/// Overload that accepts frequencies and half-space resistivity.
/// Computes z-layer distribution based on electromagnetic skin depth
/// at each frequency band, with safety margin (design freq = f_min / safety).
///
/// @param stations     Station locations
/// @param frequencies  Frequencies in Hz (ascending or descending)
/// @param rho_half     Initial half-space resistivity in Ωm
/// @param safety       Design frequency factor: f_design = f_min / safety
MeshParams3D auto_mesh_params(const std::vector<Station3D>& stations,
                              const RealVec& frequencies,
                              Real rho_half = 100.0,
                              Real safety = 10.0);

/// Compute electromagnetic skin depth: δ = 503.3 × √(ρ/f) meters.
inline Real skin_depth(Real rho, Real freq) {
    return 503.292 * std::sqrt(rho / freq);
}

/// Export station locations to CSV (name, x, y, z, lon, lat).
void export_stations_csv(const std::vector<Station3D>& stations,
                         const fs::path& path);

} // namespace mesh
} // namespace maple3dmt
