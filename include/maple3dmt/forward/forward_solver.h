// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file forward_solver.h
/// @brief Legacy 2.5D forward solver stub for backward compatibility.
///
/// For 3D forward modelling, use forward_solver_3d.h instead.

#include "maple3dmt/common.h"

namespace maple3dmt {
namespace forward {

/// 2.5D forward options (legacy stub).
struct ForwardOptions {
    int    num_threads = 1;
    Real   tolerance = 1e-10;
};

/// 2.5D quasi-3D options (legacy stub).
struct Quasi3DOptions {
    bool enabled = false;
    int  n_y_modes = 1;
};

} // namespace forward
} // namespace maple3dmt
