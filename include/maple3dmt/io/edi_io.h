// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file edi_io.h
/// @brief EDI (Electrical Data Interchange) format reader/writer for MT data.
///
/// EDI is the standard text-based format for magnetotelluric data exchange.
/// Each file contains one station with multiple frequencies.
/// Impedance tensor components (Zxx, Zxy, Zyx, Zyy) and tipper (Tx, Ty)
/// are stored as separate sections with real, imaginary, and variance arrays.

#include "maple3dmt/common.h"
#include "maple3dmt/data/mt_data.h"

namespace maple3dmt {
namespace io {

/// Parsed EDI record: one station with its frequency-dependent responses.
struct EDIRecord {
    data::Station station;
    RealVec frequencies;                      // Hz
    std::vector<data::MTResponse> responses;  // [freq_idx]
};

/// Read a single EDI file.
/// @param edi_path Path to the .edi file.
/// @return Parsed station metadata and responses.
/// @throws std::runtime_error if file cannot be opened or parsed.
EDIRecord read_edi(const fs::path& edi_path);

/// Write a single EDI file for one station.
/// @param edi_path Output file path.
/// @param station Station metadata.
/// @param frequencies Frequency array (Hz).
/// @param responses Per-frequency responses (same length as frequencies).
void write_edi(const fs::path& edi_path,
               const data::Station& station,
               const RealVec& frequencies,
               const std::vector<data::MTResponse>& responses);

/// Load a full MTData from a directory of EDI files.
/// Reads all *.edi files, computes the union of frequencies, and builds
/// the MTData object. Missing station-frequency pairs get weight=0.
/// @param dir Directory containing .edi files.
/// @param out Output MTData object (cleared and rebuilt).
void load_edi_directory(const fs::path& dir, data::MTData& out);

/// Save MTData to a directory of EDI files (one per station).
/// Creates the directory if it doesn't exist.
/// @param dir Output directory.
/// @param data MT dataset to save.
void save_edi_directory(const fs::path& dir, const data::MTData& data);

} // namespace io
} // namespace maple3dmt
