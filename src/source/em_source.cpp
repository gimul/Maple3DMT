// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file em_source.cpp
/// @brief Implementation of EM source utilities.

#include "maple3dmt/source/em_source.h"
#include <cmath>

namespace maple3dmt {
namespace source {

Real EMSource::omega() const {
    return constants::TWOPI * frequency;
}

Real EMSource::skin_depth(Real sigma) const {
    Real w = omega();
    if (w <= 0 || sigma <= 0) return 0.0;
    return std::sqrt(2.0 / (w * constants::MU0 * sigma));
}

RealVec get_frequencies(const RealVec& data_frequencies) {
    // For v0.1, just pass through.
    // Future: could add padding frequencies for interpolation.
    return data_frequencies;
}

RealVec generate_ky_points(const WavenumberParams& params) {
    RealVec ky_values(params.n_ky);

    if (params.log_spacing) {
        Real log_min = std::log10(params.ky_min);
        Real log_max = std::log10(params.ky_max);
        for (int j = 0; j < params.n_ky; ++j) {
            Real t = (params.n_ky > 1)
                     ? static_cast<Real>(j) / (params.n_ky - 1)
                     : 0.0;
            ky_values[j] = std::pow(10.0, log_min + t * (log_max - log_min));
        }
    } else {
        for (int j = 0; j < params.n_ky; ++j) {
            Real t = (params.n_ky > 1)
                     ? static_cast<Real>(j) / (params.n_ky - 1)
                     : 0.0;
            ky_values[j] = params.ky_min + t * (params.ky_max - params.ky_min);
        }
    }

    return ky_values;
}

} // namespace source
} // namespace maple3dmt
