// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file hdf5_io.h
/// @brief HDF5 I/O utilities for models, data, and results.

#include "maple3dmt/common.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"

namespace maple3dmt {
namespace io {

/// Save model to HDF5 file.
void save_model_hdf5(const model::ConductivityModel& m,
                     const fs::path& path);

/// Load model from HDF5 file.
void load_model_hdf5(model::ConductivityModel& m,
                     const fs::path& path);

/// Save MT data (observed + predicted) to HDF5.
void save_data_hdf5(const data::MTData& d,
                    const fs::path& path);

/// Load MT data from HDF5.
void load_data_hdf5(data::MTData& d,
                    const fs::path& path);

/// Save inversion results (model + responses + misfit) to HDF5.
void save_results_hdf5(const model::ConductivityModel& m,
                       const data::MTData& d,
                       const fs::path& path);

} // namespace io
} // namespace maple3dmt
