// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file dem.h
/// @brief DEM (Digital Elevation Model) data holder and interpolation.
///
/// Standalone header — no MFEM dependency.
/// Used by both FEM (hex_mesh_3d) and FV (octree) backends.
///
/// Supports dual-DEM mode: high-res land DEM (primary) merged with
/// bathymetry DEM (secondary, e.g. GEBCO/ETOPO) at query time.

#include "maple3dmt/common.h"
#include <memory>
#include <vector>

namespace maple3dmt {
namespace mesh {

/// ALOS / Copernicus DEM data holder.
struct ALOSDem {
    std::vector<Real> lon;       // longitude grid [n_lon]
    std::vector<Real> lat;       // latitude grid [n_lat]
    std::vector<Real> elevation; // elevation [n_lat × n_lon], row-major
    int n_lon = 0, n_lat = 0;

    /// Load from GeoTIFF file.
    void load_geotiff(const fs::path& path);

    /// Load from ASCII grid (x y z format).
    void load_ascii(const fs::path& path);

    /// Attach a bathymetry DEM for ocean regions.
    /// When interpolating, if the primary DEM returns ≤ 0 (or is out of range),
    /// the bathymetry DEM is consulted.  This allows SRTM (land only) +
    /// GEBCO/ETOPO (ocean floor) to be combined transparently.
    void set_bathymetry(std::shared_ptr<ALOSDem> bathy);

    /// Interpolate elevation at (lon, lat) using bilinear interpolation.
    /// If a bathymetry DEM is attached, the merge rule is:
    ///   1. Primary in-range and elev > 0  → return primary (land)
    ///   2. Primary in-range and elev ≤ 0  → return bathymetry if < 0 (ocean floor)
    ///   3. Primary out-of-range           → return bathymetry (extended coverage)
    ///   4. Neither in-range               → 0.0 (sea level fallback)
    Real interpolate(Real lon, Real lat) const;

    /// Check if DEM covers the given bounding box.
    /// Logs warning if coverage is insufficient.
    void check_coverage(Real lon_min, Real lon_max,
                        Real lat_min, Real lat_max) const;

    /// Diagnose land-ocean transition quality after bathymetry merge.
    /// Samples profiles crossing the coastline and checks for discontinuities.
    /// Call after set_bathymetry() and before mesh generation.
    void diagnose_coastline(Real lon_min, Real lon_max,
                            Real lat_min, Real lat_max,
                            int n_profiles = 4, int n_samples = 200) const;

    /// Check if a point is within the DEM grid extent.
    bool in_range(Real qlon, Real qlat) const;

private:
    std::shared_ptr<ALOSDem> bathymetry_;

    /// Raw bilinear interpolation on this grid only (no bathymetry fallback).
    /// Returns 0.0 if out of range.
    Real interpolate_grid(Real qlon, Real qlat) const;
};

} // namespace mesh
} // namespace maple3dmt
