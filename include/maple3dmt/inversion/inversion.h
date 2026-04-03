// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file inversion.h
/// @brief Legacy 2.5D inversion stub for backward compatibility.
///
/// For 3D inversion, use inversion_3d.h instead.

#include "maple3dmt/common.h"

namespace maple3dmt {
namespace inversion {

/// 2.5D inversion parameters (legacy stub).
struct InversionParams {
    int    max_iterations = 20;
    Real   target_rms = 1.0;
    Real   lambda_init = 10.0;
    Real   lambda_min = 0.01;
};

} // namespace inversion
} // namespace maple3dmt
