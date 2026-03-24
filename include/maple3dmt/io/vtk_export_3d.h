// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file vtk_export_3d.h
/// @brief 3D inversion result export for visualization.
///
/// Exports to:
///   - VTK/VTU: ParaView 3D volume rendering
///   - Sliced VTK: horizontal/vertical cross-sections
///   - GeoJSON: station locations for GIS overlay
///   - NetCDF: for GMT or Python xarray

#include "maple3dmt/common.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"
#include <mfem.hpp>
#include <vector>

namespace maple3dmt {
namespace io {

/// 3D model export options.
struct ExportParams {
    // VTK export
    bool export_vtk          = true;
    bool export_vtu_parallel = true;   // parallel VTU (PVTU) for large meshes
    int  vtk_precision       = 8;      // ASCII decimal digits (or 0 for binary)

    // Cross-section slices
    bool export_slices       = true;
    std::vector<Real> slice_depths;     // horizontal slices at these depths (m)
    std::vector<Real> slice_x;          // YZ vertical slices at these x positions
    std::vector<Real> slice_y;          // XZ vertical slices at these y positions
    // Auto-generate depth slices: every N km from surface to roi_depth
    Real auto_slice_interval = 2000;    // meters (0 = disabled)

    // Along-profile slices (for comparison with 2.5D)
    struct ProfileSlice {
        std::string name;
        Real x0, y0, x1, y1;           // start/end coordinates (m)
        int  n_points = 200;           // interpolation points
    };
    std::vector<ProfileSlice> profile_slices;

    // Station data
    bool export_station_geojson = true; // station locations as GeoJSON
    bool export_station_csv     = true; // station coordinates CSV

    // NetCDF (for GMT / xarray)
    bool export_netcdf       = false;
    Real netcdf_grid_dx      = 500;     // regular grid spacing for interpolation
    Real netcdf_grid_dy      = 500;
    Real netcdf_grid_dz      = 100;
};

/// Export 3D model to VTK (Unstructured Grid).
/// Includes: log10(resistivity), conductivity, element attributes.
void export_model_vtk(const mfem::ParMesh& mesh,
                      const model::ConductivityModel& model,
                      const fs::path& path,
                      int iteration = -1);

/// Export parallel VTU (PVTU) for large meshes.
/// Each MPI rank writes its partition; rank 0 writes the PVTU master.
void export_model_pvtu(const mfem::ParMesh& mesh,
                       const model::ConductivityModel& model,
                       const fs::path& dir,
                       int iteration = -1);

/// Export horizontal depth slice as VTK StructuredGrid.
/// Interpolates model from hex mesh to regular grid at given depth.
void export_depth_slice(const mfem::ParMesh& mesh,
                        const model::ConductivityModel& model,
                        Real depth,
                        const fs::path& path,
                        Real dx = 500, Real dy = 500);

/// Export vertical cross-section along a profile line.
/// Interpolates model along (x0,y0)→(x1,y1) to depth z_max.
void export_profile_slice(const mfem::ParMesh& mesh,
                          const model::ConductivityModel& model,
                          Real x0, Real y0, Real x1, Real y1,
                          Real z_max, const fs::path& path,
                          int n_along = 200, int n_depth = 100);

/// Export station locations as GeoJSON (for QGIS / web map overlay).
void export_stations_geojson(const data::MTData& data,
                             const fs::path& path);

/// Export station coordinates as CSV.
void export_stations_csv(const data::MTData& data,
                         const fs::path& path);

/// Export all configured outputs in one call.
void export_all(const mfem::ParMesh& mesh,
                const model::ConductivityModel& model,
                const data::MTData& data,
                const ExportParams& params,
                const fs::path& output_dir,
                int iteration = -1);

} // namespace io
} // namespace maple3dmt
