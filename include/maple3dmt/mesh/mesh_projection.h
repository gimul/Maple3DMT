// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file mesh_projection.h
/// @brief Legacy 2.5D mesh projection stub for backward compatibility.

#include "maple3dmt/common.h"

namespace maple3dmt {
namespace mesh {

/// Dual-mesh projection (2.5D legacy stub).
/// Will be replaced by 3D element-to-element projection.
class MeshProjection {
public:
    MeshProjection() = default;

    /// Project model parameters from inversion mesh to forward mesh.
    void project_sigma(const RealVec& inv_params, RealVec& fwd_params) const {
        fwd_params = inv_params;  // stub: identity projection
    }
};

} // namespace mesh
} // namespace maple3dmt
