// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file dem.cpp
/// @brief DEM loading, bilinear interpolation, and land+bathymetry merge.
///
/// Standalone — no MFEM dependency. Shared by FEM and FV backends.

#include "maple3dmt/mesh/dem.h"
#include "maple3dmt/utils/logger.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <set>

#ifdef MAPLE3DMT_USE_GDAL
#include <gdal_priv.h>
#include <cpl_conv.h>
#endif

namespace maple3dmt {
namespace mesh {

void ALOSDem::load_geotiff(const fs::path& path) {
#ifdef MAPLE3DMT_USE_GDAL
    GDALAllRegister();
    auto* dataset = static_cast<GDALDataset*>(
        GDALOpen(path.string().c_str(), GA_ReadOnly));
    if (!dataset) {
        throw std::runtime_error("ALOSDem: cannot open GeoTIFF: " +
                                 path.string());
    }

    n_lon = dataset->GetRasterXSize();
    n_lat = dataset->GetRasterYSize();

    double gt[6];
    dataset->GetGeoTransform(gt);

    lon.resize(n_lon);
    for (int i = 0; i < n_lon; ++i) {
        lon[i] = gt[0] + (i + 0.5) * gt[1];
    }

    lat.resize(n_lat);
    for (int j = 0; j < n_lat; ++j) {
        lat[j] = gt[3] + (j + 0.5) * gt[5];
    }

    if (n_lat > 1 && lat[0] > lat[1]) {
        std::reverse(lat.begin(), lat.end());
    }

    elevation.resize(n_lat * n_lon);
    auto* band = dataset->GetRasterBand(1);

    std::vector<float> row_buf(n_lon);
    for (int j = 0; j < n_lat; ++j) {
        int src_row = (lat[0] < lat[n_lat - 1])
                          ? (n_lat - 1 - j)
                          : j;
        band->RasterIO(GF_Read, 0, src_row, n_lon, 1,
                        row_buf.data(), n_lon, 1, GDT_Float32,
                        0, 0);
        for (int i = 0; i < n_lon; ++i) {
            elevation[j * n_lon + i] = static_cast<Real>(row_buf[i]);
        }
    }

    GDALClose(dataset);

    MAPLE3DMT_LOG_INFO("ALOSDem: loaded GeoTIFF " + path.string() +
                     " (" + std::to_string(n_lon) + " x " +
                     std::to_string(n_lat) + ")");
#else
    (void)path;
    throw std::runtime_error(
        "ALOSDem::load_geotiff requires GDAL. "
        "Rebuild with -DMAPLE3DMT_USE_GDAL=ON or use load_ascii().");
#endif
}

void ALOSDem::load_ascii(const fs::path& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("ALOSDem: cannot open ASCII file: " +
                                 path.string());
    }

    struct Point {
        Real lon, lat, elev;
    };
    std::vector<Point> points;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        Point p;
        if (iss >> p.lon >> p.lat >> p.elev) {
            points.push_back(p);
        }
    }

    if (points.empty()) {
        throw std::runtime_error("ALOSDem: no data in ASCII file");
    }

    std::set<Real> lon_set, lat_set;
    for (const auto& p : points) {
        lon_set.insert(p.lon);
        lat_set.insert(p.lat);
    }

    lon.assign(lon_set.begin(), lon_set.end());
    lat.assign(lat_set.begin(), lat_set.end());
    n_lon = static_cast<int>(lon.size());
    n_lat = static_cast<int>(lat.size());

    if (n_lon * n_lat != static_cast<int>(points.size())) {
        throw std::runtime_error(
            "ALOSDem: irregular grid. Expected " +
            std::to_string(n_lon) + " x " + std::to_string(n_lat) +
            " = " + std::to_string(n_lon * n_lat) +
            " points, got " + std::to_string(points.size()));
    }

    elevation.resize(n_lat * n_lon, 0.0);
    for (const auto& p : points) {
        auto ix = static_cast<int>(
            std::lower_bound(lon.begin(), lon.end(), p.lon) - lon.begin());
        auto iy = static_cast<int>(
            std::lower_bound(lat.begin(), lat.end(), p.lat) - lat.begin());
        if (ix < n_lon && iy < n_lat) {
            elevation[iy * n_lon + ix] = p.elev;
        }
    }

    MAPLE3DMT_LOG_INFO("ALOSDem: loaded ASCII " + path.string() +
                     " (" + std::to_string(n_lon) + " x " +
                     std::to_string(n_lat) + ")");
}

// =========================================================================
// Bathymetry merge
// =========================================================================

void ALOSDem::set_bathymetry(std::shared_ptr<ALOSDem> bathy) {
    bathymetry_ = std::move(bathy);
    if (bathymetry_) {
        MAPLE3DMT_LOG_INFO("ALOSDem: bathymetry DEM attached ("
                         + std::to_string(bathymetry_->n_lon) + "x"
                         + std::to_string(bathymetry_->n_lat) + ")");
    }
}

bool ALOSDem::in_range(Real qlon, Real qlat) const {
    if (lon.empty() || lat.empty()) return false;
    return qlon >= lon.front() && qlon <= lon.back() &&
           qlat >= lat.front() && qlat <= lat.back();
}

Real ALOSDem::interpolate_grid(Real qlon, Real qlat) const {
    if (lon.empty() || lat.empty()) return 0.0;

    if (!in_range(qlon, qlat)) return 0.0;

    auto ix_it = std::lower_bound(lon.begin(), lon.end(), qlon);
    int ix = static_cast<int>(ix_it - lon.begin());
    if (ix == 0) ix = 1;
    if (ix >= n_lon) ix = n_lon - 1;

    auto iy_it = std::lower_bound(lat.begin(), lat.end(), qlat);
    int iy = static_cast<int>(iy_it - lat.begin());
    if (iy == 0) iy = 1;
    if (iy >= n_lat) iy = n_lat - 1;

    Real tx = (qlon - lon[ix - 1]) / (lon[ix] - lon[ix - 1]);
    Real ty = (qlat - lat[iy - 1]) / (lat[iy] - lat[iy - 1]);

    Real e00 = elevation[(iy - 1) * n_lon + (ix - 1)];
    Real e10 = elevation[(iy - 1) * n_lon + ix];
    Real e01 = elevation[iy * n_lon + (ix - 1)];
    Real e11 = elevation[iy * n_lon + ix];

    return (1 - tx) * (1 - ty) * e00 + tx * (1 - ty) * e10 +
           (1 - tx) * ty * e01 + tx * ty * e11;
}

Real ALOSDem::interpolate(Real qlon, Real qlat) const {
    // --- Case 1: No bathymetry attached — original behaviour ---
    if (!bathymetry_) {
        return in_range(qlon, qlat) ? interpolate_grid(qlon, qlat) : 0.0;
    }

    // --- Case 2: Dual-DEM merge ---
    bool primary_ok = in_range(qlon, qlat);

    if (primary_ok) {
        Real val = interpolate_grid(qlon, qlat);

        // Definite land (positive elevation) → trust primary (higher res)
        if (val > 0.0) return val;

        // Primary says ≤ 0 — could be real sea-level coast or SRTM ocean gap.
        // Consult bathymetry for the true picture.
        Real bval = bathymetry_->interpolate_grid(qlon, qlat);

        if (bval < 0.0) {
            // Bathymetry confirms ocean → return actual seafloor depth
            return bval;
        }
        // Both near zero → genuine sea-level land or tidal flat.
        // Prefer primary since it has higher resolution at coast.
        return val;
    }

    // Primary out of range → bathymetry extends coverage
    if (bathymetry_->in_range(qlon, qlat)) {
        return bathymetry_->interpolate_grid(qlon, qlat);
    }

    return 0.0;  // neither DEM covers this point
}

// =========================================================================
// Coverage check
// =========================================================================

void ALOSDem::check_coverage(Real lon_min, Real lon_max,
                             Real lat_min, Real lat_max) const {
    if (lon.empty() || lat.empty()) {
        MAPLE3DMT_LOG_WARNING("ALOSDem: no DEM data loaded — terrain will be flat (z=0)");
        return;
    }

    bool ok = true;
    std::string msg;

    if (lon_min < lon.front()) {
        msg += "  Lon min: need " + std::to_string(lon_min) +
               ", DEM starts at " + std::to_string(lon.front()) + "\n";
        ok = false;
    }
    if (lon_max > lon.back()) {
        msg += "  Lon max: need " + std::to_string(lon_max) +
               ", DEM ends at " + std::to_string(lon.back()) + "\n";
        ok = false;
    }
    if (lat_min < lat.front()) {
        msg += "  Lat min: need " + std::to_string(lat_min) +
               ", DEM starts at " + std::to_string(lat.front()) + "\n";
        ok = false;
    }
    if (lat_max > lat.back()) {
        msg += "  Lat max: need " + std::to_string(lat_max) +
               ", DEM ends at " + std::to_string(lat.back()) + "\n";
        ok = false;
    }

    if (!ok) {
        // If bathymetry can fill the gap, downgrade to info
        if (bathymetry_) {
            bool bathy_covers = true;
            if (lon_min < lon.front() && !bathymetry_->in_range(lon_min, (lat_min + lat_max) / 2))
                bathy_covers = false;
            if (lon_max > lon.back() && !bathymetry_->in_range(lon_max, (lat_min + lat_max) / 2))
                bathy_covers = false;
            if (lat_min < lat.front() && !bathymetry_->in_range((lon_min + lon_max) / 2, lat_min))
                bathy_covers = false;
            if (lat_max > lat.back() && !bathymetry_->in_range((lon_min + lon_max) / 2, lat_max))
                bathy_covers = false;

            if (bathy_covers) {
                MAPLE3DMT_LOG_INFO(
                    "ALOSDem: primary DEM partial coverage — "
                    "bathymetry DEM covers the gap.\n" + msg);
                return;
            }
        }

        MAPLE3DMT_LOG_WARNING(
            "ALOSDem: DEM does NOT fully cover the station area!\n" + msg +
            "  Cells outside DEM will use z=0 (sea level).\n"
            "  This may misplace air/earth boundary near edge stations.\n"
            "  → Download a wider DEM covering the full station extent.");
    } else {
        MAPLE3DMT_LOG_INFO(
            "ALOSDem: DEM coverage OK for station area ("
            + std::to_string(lon_min) + "~" + std::to_string(lon_max) + " lon, "
            + std::to_string(lat_min) + "~" + std::to_string(lat_max) + " lat)");
    }
}

// =========================================================================
// Coastline transition diagnostics
// =========================================================================

void ALOSDem::diagnose_coastline(Real lon_min, Real lon_max,
                                 Real lat_min, Real lat_max,
                                 int n_profiles, int n_samples) const {
    if (!bathymetry_) {
        MAPLE3DMT_LOG_INFO("ALOSDem::diagnose_coastline: no bathymetry — skipped");
        return;
    }

    int n_transitions = 0;
    Real max_jump = 0.0;
    Real sum_jump = 0.0;
    int n_coast_pts = 0;

    // Sample profiles in both lon and lat directions
    for (int p = 0; p < n_profiles; ++p) {
        // Longitude profile (varying lon at fixed lat)
        {
            Real fixed_lat = lat_min + (lat_max - lat_min) * (p + 0.5) / n_profiles;
            Real prev_elev = interpolate(lon_min, fixed_lat);

            for (int i = 1; i < n_samples; ++i) {
                Real cur_lon = lon_min + (lon_max - lon_min) * i / (n_samples - 1);
                Real cur_elev = interpolate(cur_lon, fixed_lat);

                // Detect sign change (land↔ocean transition)
                if ((prev_elev > 0 && cur_elev < 0) ||
                    (prev_elev < 0 && cur_elev > 0)) {
                    ++n_transitions;
                }

                // Check for abrupt jumps (potential merge artifact)
                Real step = (lon_max - lon_min) / (n_samples - 1);
                Real dlon = step;
                Real jump = std::abs(cur_elev - prev_elev);

                // Large jump relative to horizontal distance ≈ unrealistic cliff
                // Typical: 1 m elevation per 30 m horizontal is steep but real.
                // Artifact: >100 m jump per grid step
                Real horiz_m = dlon * 111000.0;  // rough deg→m
                Real gradient = (horiz_m > 0) ? jump / horiz_m : 0;

                if (jump > max_jump) max_jump = jump;

                // Near coast (either side within ±50m of sea level)
                if (std::abs(prev_elev) < 50 || std::abs(cur_elev) < 50) {
                    sum_jump += jump;
                    ++n_coast_pts;

                    if (gradient > 0.5) {  // > 50% slope near coast = suspicious
                        // Don't spam — just count
                    }
                }

                prev_elev = cur_elev;
            }
        }

        // Latitude profile (varying lat at fixed lon)
        {
            Real fixed_lon = lon_min + (lon_max - lon_min) * (p + 0.5) / n_profiles;
            Real prev_elev = interpolate(fixed_lon, lat_min);

            for (int i = 1; i < n_samples; ++i) {
                Real cur_lat = lat_min + (lat_max - lat_min) * i / (n_samples - 1);
                Real cur_elev = interpolate(fixed_lon, cur_lat);

                if ((prev_elev > 0 && cur_elev < 0) ||
                    (prev_elev < 0 && cur_elev > 0)) {
                    ++n_transitions;
                }

                Real jump = std::abs(cur_elev - prev_elev);
                if (jump > max_jump) max_jump = jump;

                if (std::abs(prev_elev) < 50 || std::abs(cur_elev) < 50) {
                    sum_jump += jump;
                    ++n_coast_pts;
                }

                prev_elev = cur_elev;
            }
        }
    }

    Real avg_coast_jump = (n_coast_pts > 0) ? sum_jump / n_coast_pts : 0.0;

    std::ostringstream oss;
    oss << "ALOSDem coastline diagnostics ("
        << n_profiles << " lon + " << n_profiles << " lat profiles):\n"
        << "  Land-ocean transitions detected: " << n_transitions << "\n"
        << "  Max elevation jump between samples: "
        << std::fixed << std::setprecision(1) << max_jump << " m\n"
        << "  Avg jump near coast (|z|<50m): "
        << std::fixed << std::setprecision(1) << avg_coast_jump << " m\n";

    if (max_jump > 500) {
        oss << "  WARNING: Large jump (>" << std::setprecision(0) << max_jump
            << " m) — possible DEM merge artifact at coastline.\n"
            << "  Check that primary and bathymetry DEMs have consistent\n"
            << "  coastline definitions. Consider using ETOPO alone.";
        MAPLE3DMT_LOG_WARNING(oss.str());
    } else if (n_transitions == 0 && n_coast_pts > 0) {
        oss << "  INFO: No land-ocean transitions found in sampled area.\n"
            << "  If this is a coastal site, the DEM merge may not be working.";
        MAPLE3DMT_LOG_WARNING(oss.str());
    } else {
        MAPLE3DMT_LOG_INFO(oss.str());
    }
}

} // namespace mesh
} // namespace maple3dmt
