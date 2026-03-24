// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file em_source.h
/// @brief EM source and polarisation definitions for MT forward modeling.
///
/// === v0.1 (2D TE/TM) ===
/// MT source is a uniform plane wave. For a 2D model, TE and TM modes
/// decouple completely. No ky integration is needed — each frequency
/// requires exactly two 2D solves (TE + TM).
///
/// === v0.2+ (quasi-3D) ===
/// When the model has y-direction variation, ky modes become coupled.
/// WavenumberParams controls the ky quadrature for the coupled system.
/// The source is still a plane wave (ky=0 in source space), but the
/// y-varying conductivity redistributes energy across ky modes.

#include "maple3dmt/common.h"

namespace maple3dmt {
namespace source {

/// Polarisation mode.
enum class Polarisation {
    TE,   // E-polarisation: primary unknown Ey  → Zxy, Tx
    TM,   // H-polarisation: primary unknown Hy  → Zyx
    BOTH  // both modes (standard for full 2D MT)
};

/// Parameters for quasi-3D ky integration (v0.2+).
/// Not used in pure 2D mode (v0.1).
struct WavenumberParams {
    int    n_ky        = 15;       // number of ky quadrature points
    Real   ky_min      = 1e-6;    // minimum ky [1/m]
    Real   ky_max      = 1e-1;    // maximum ky
    bool   log_spacing = true;    // logarithmic ky distribution
};

/// EM source specification for one forward solve.
struct EMSource {
    Real         frequency = 1.0;  // Hz
    Polarisation polarisation = Polarisation::BOTH;

    /// Angular frequency ω = 2πf.
    Real omega() const;

    /// Skin depth in a uniform halfspace: δ = sqrt(2/(ωμσ)).
    Real skin_depth(Real sigma) const;
};

/// Generate frequency list for forward computation.
/// Returns the same frequencies as in the data, used to drive
/// the TE/TM solve loop.
RealVec get_frequencies(const RealVec& data_frequencies);

/// Generate ky quadrature points (for quasi-3D mode only).
RealVec generate_ky_points(const WavenumberParams& params);

} // namespace source
} // namespace maple3dmt
