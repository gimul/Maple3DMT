// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file config_reader.h
/// @brief YAML-based configuration reader for NewMT.

#include "maple3dmt/common.h"
#include "maple3dmt/mesh/terrain_mesh.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/forward/forward_solver.h"
#include "maple3dmt/regularization/regularization.h"
#include "maple3dmt/inversion/inversion.h"
#include "maple3dmt/source/em_source.h"
#include "maple3dmt/survey/survey_line.h"

namespace maple3dmt {
namespace io {

/// Complete run configuration parsed from YAML.
struct RunConfig {
    fs::path                         config_path;
    fs::path                         data_file;
    fs::path                         topo_file;
    fs::path                         output_dir;

    mesh::MeshParams                 mesh_params;
    model::Parameterisation          param_type;
    Real                             sigma_background;

    source::WavenumberParams         ky_params;
    source::Polarisation             polarisation = source::Polarisation::BOTH;

    forward::ForwardOptions          fwd_opts;
    forward::Quasi3DOptions          q3d_opts;     // quasi-3D extension
    regularization::RegParams        reg_params;
    inversion::InversionParams       inv_params;

    // Survey line
    survey::SurveyLineParams         survey_params;
    fs::path                         etopo_file;   // ETOPO topo profile file
};

/// Read a YAML configuration file and populate RunConfig.
RunConfig read_config(const fs::path& yaml_path);

/// Write current configuration to YAML (for reproducibility).
void write_config(const RunConfig& cfg, const fs::path& yaml_path);

} // namespace io
} // namespace maple3dmt
