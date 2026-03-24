// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file vtk_export_octree.h
/// @brief Octree FV model export for visualization.
///
/// Outputs:
///   - VTK UnstructuredGrid (.vtu): ParaView 3D volume, octree cells as hexahedra
///   - Horizontal depth slices (.vtu): regular grid at given depth
///   - Station locations (.csv): for overlay

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"
#include <string>
#include <vector>
#include <map>

namespace maple3dmt {
namespace io {

/// Export parameters for octree visualization.
struct OctreeExportParams {
    // 3D VTU export
    bool export_vtu = true;

    // Depth slice export
    bool export_depth_slices = true;
    std::vector<Real> slice_depths;       // depths (m, positive downward)
    Real auto_slice_interval = 2000.0;    // auto-generate every N meters (0=disabled)
    Real auto_slice_max_depth = 30000.0;  // max depth for auto slices
    Real slice_dx = 500.0;               // regular grid spacing for slices
    Real slice_dy = 500.0;

    // Station export
    bool export_stations_csv = true;
};

/// Export octree model as VTK UnstructuredGrid (.vtu).
/// Each octree cell is written as a hexahedral cell.
/// Cell data: log10_resistivity, conductivity, cell_type, cell_level.
void export_octree_vtu(const octree::OctreeMesh& mesh,
                       const model::ConductivityModel& model,
                       const std::string& path,
                       int iteration = -1,
                       const std::map<std::string, RealVec>& extra_scalars = {});

/// Export horizontal depth slice as VTK StructuredGrid.
/// Finds the octree cell containing each grid point and samples its sigma.
void export_octree_depth_slice(const octree::OctreeMesh& mesh,
                               const model::ConductivityModel& model,
                               Real depth,
                               const std::string& path,
                               Real dx = 500.0, Real dy = 500.0);

/// Export station locations as CSV (x, y, z, name).
void export_stations_csv(const data::MTData& data,
                         const std::string& path);

/// Export obs vs pred data fitting CSV.
/// Columns: station, freq_hz, period_s, component,
///          obs_re, obs_im, pred_re, pred_im, error,
///          app_res_obs, phase_obs, app_res_pred, phase_pred
void export_data_fit_csv(const data::MTData& data,
                         const std::string& path,
                         int iteration = -1);

/// Export all: VTU + depth slices + stations.
/// Convenience wrapper that applies ExportParams.
void export_octree_all(const octree::OctreeMesh& mesh,
                       const model::ConductivityModel& model,
                       const data::MTData& data,
                       const std::string& output_dir,
                       const OctreeExportParams& params,
                       int iteration = -1);

/// Read conductivity array from a model VTU file (ASCII format).
/// Returns per-cell sigma values in the same order as the mesh.
/// Used for resume: load the last saved model and continue inversion.
RealVec load_conductivity_from_vtu(const std::string& path);

/// Save lightweight inversion state for resume (JSON).
/// Written to <output_dir>/inversion_state.json every iteration.
void save_inversion_state(const std::string& path,
                          int iteration,
                          Real lambda,
                          Real rms,
                          const std::vector<std::pair<int, Real>>& rms_history);

/// Load inversion state from JSON.
/// Returns false if file doesn't exist or can't be parsed.
struct InversionState {
    int  last_iteration = 0;
    Real lambda = 10.0;
    Real rms = 0.0;
    std::vector<std::pair<int, Real>> rms_history;  // (iter, rms)
};
bool load_inversion_state(const std::string& path, InversionState& state);

} // namespace io
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
