// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file terrain_mesh.h
/// @brief Legacy 2.5D terrain mesh stub for backward compatibility.
///
/// This is a placeholder to allow 2.5D-origin code (conductivity_model,
/// regularization, config_reader) to compile. These modules will be
/// updated to use HexMesh3D as the project progresses.
///
/// For 3D mesh generation, use hex_mesh_3d.h instead.

#include "maple3dmt/common.h"
#include <mfem.hpp>
#include <memory>
#include <vector>

namespace maple3dmt {
namespace mesh {

/// 2.5D mesh parameters (legacy, will be replaced by MeshParams3D).
struct MeshParams {
    Real x_min = -50000, x_max = 50000;
    Real z_min = -100000, z_max = 50000;
    Real h_surface = 500;
    Real h_deep = 5000;
    Real h_air = 5000;
    Real h_boundary = 10000;
    Real h_min = 50;
    Real h_x_max = 0;
    Real growth_rate = 1.3;
    Real x_growth_rate = 1.3;
    Real depth_growth = 1.5;
    int  refine_near_sites = 1;
    Real site_refine_radius = 2000;
    Real x_inner_margin = 5000;
    Real middle_zone_width = 10000;
    Real roi_depth = 30000;
    Real roi_outside_growth = 2.0;
};

/// Topographic profile (2.5D legacy).
struct TopoProfile {
    RealVec x, z;
    Real elevation_at(Real xp) const;
};

/// 2.5D terrain mesh (legacy stub).
class TerrainMesh {
public:
    TerrainMesh() = default;
    mfem::Mesh*       mesh()       { return mesh_.get(); }
    const mfem::Mesh* mesh() const { return mesh_.get(); }
    int num_elements() const { return mesh_ ? mesh_->GetNE() : 0; }
    int num_vertices() const { return mesh_ ? mesh_->GetNV() : 0; }
private:
    std::unique_ptr<mfem::Mesh> mesh_;
};

} // namespace mesh
} // namespace maple3dmt
