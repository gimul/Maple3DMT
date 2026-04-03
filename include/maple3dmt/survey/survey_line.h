// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file survey_line.h
/// @brief Legacy 2.5D survey line stub for backward compatibility.

#include "maple3dmt/common.h"

namespace maple3dmt {
namespace survey {

/// 2.5D survey line parameters (legacy stub).
struct SurveyLineParams {
    Real azimuth = 0.0;
    Real length = 0.0;
};

} // namespace survey
} // namespace maple3dmt
