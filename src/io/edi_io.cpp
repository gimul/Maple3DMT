// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file edi_io.cpp
/// @brief EDI format reader/writer implementation.
///
/// Supports common EDI variants including:
/// - DMS coordinates (LAT=+35:32:28.7)
/// - Decimal coordinates (LAT=35.541306)
/// - NFREQ in >FREQ header or >=MTSECT section
/// - Section format: >ZXXR ROT=ZROT //39
/// - Tipper: >TXR.EXP, >TXI.EXP, >TYR.EXP, >TYI.EXP

#include "maple3dmt/io/edi_io.h"
#include "maple3dmt/utils/logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <set>
#include <stdexcept>
#include <regex>

namespace maple3dmt {
namespace io {

// =========================================================================
// Helper: trim whitespace
// =========================================================================
static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// =========================================================================
// Helper: parse key=value from a header line
// =========================================================================
static std::string extract_value(const std::string& line, const std::string& key) {
    auto pos = line.find(key + "=");
    if (pos == std::string::npos) {
        pos = line.find(key + " =");
        if (pos == std::string::npos) return "";
    }
    auto eq = line.find('=', pos);
    if (eq == std::string::npos) return "";
    auto val_start = eq + 1;
    // Skip whitespace after =
    while (val_start < line.size() && (line[val_start] == ' ' || line[val_start] == '\t'))
        ++val_start;
    // Handle quoted strings
    if (val_start < line.size() && line[val_start] == '"') {
        auto end_quote = line.find('"', val_start + 1);
        if (end_quote != std::string::npos)
            return line.substr(val_start + 1, end_quote - val_start - 1);
    }
    // Unquoted: read to next whitespace or end
    auto val_end = line.find_first_of(" \t\r\n", val_start);
    if (val_end == std::string::npos) val_end = line.size();
    return line.substr(val_start, val_end - val_start);
}

// =========================================================================
// Helper: parse DMS coordinate string to decimal degrees
// Supports formats: +35:32:28.7, -128:48:40.7, 35.541306, +35.541306
// =========================================================================
static Real parse_coordinate(const std::string& s) {
    if (s.empty()) return 0.0;

    // Check if it contains ':' → DMS format
    if (s.find(':') != std::string::npos) {
        Real sign = 1.0;
        std::string str = s;
        if (str[0] == '+') { str = str.substr(1); }
        else if (str[0] == '-') { sign = -1.0; str = str.substr(1); }

        // Split by ':'
        std::replace(str.begin(), str.end(), ':', ' ');
        std::istringstream iss(str);
        Real deg = 0, min = 0, sec = 0;
        iss >> deg;
        if (iss) iss >> min;
        if (iss) iss >> sec;

        return sign * (deg + min / 60.0 + sec / 3600.0);
    }

    // Decimal format
    return std::stod(s);
}

// =========================================================================
// Helper: extract count from section header
// Formats: ">FREQ NFREQ=39", ">FREQ  //39", "NFREQ=39"
// =========================================================================
static int extract_count(const std::string& line) {
    // Try NFREQ=
    auto val = extract_value(line, "NFREQ");
    if (!val.empty()) {
        try { return std::stoi(val); } catch (...) {}
    }

    // Try //N format (common in Phoenix/Geotools EDI)
    auto pos = line.find("//");
    if (pos != std::string::npos) {
        std::string num_str = trim(line.substr(pos + 2));
        try { return std::stoi(num_str); } catch (...) {}
    }

    return -1;
}

// =========================================================================
// Helper: read floating-point values from lines until next section marker
// =========================================================================
static RealVec read_float_block(std::ifstream& ifs, int expected_count) {
    RealVec values;
    values.reserve(expected_count);
    std::string line;
    while (values.size() < static_cast<size_t>(expected_count) &&
           std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '>' || line[0] == ';') {
            break;
        }
        std::istringstream iss(line);
        double v;
        while (iss >> v) {
            values.push_back(v);
            if (values.size() >= static_cast<size_t>(expected_count)) break;
        }
    }
    return values;
}

// =========================================================================
// Helper: check if a section header line starts with a given tag
// Handles formats like: >ZXXR ROT=ZROT //39, >ZXXR, >ZXXR ROT=0.0
// =========================================================================
static bool section_starts_with(const std::string& line, const std::string& tag) {
    if (line.size() < tag.size()) return false;
    if (line.substr(0, tag.size()) != tag) return false;
    // After tag: end of string, space, tab, or non-alpha
    if (line.size() == tag.size()) return true;
    char next = line[tag.size()];
    return (next == ' ' || next == '\t' || next == '.' || next == '/' || !std::isalpha(next));
}

// =========================================================================
// read_edi
// =========================================================================

EDIRecord read_edi(const fs::path& edi_path) {
    std::ifstream ifs(edi_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open EDI file: " + edi_path.string());
    }

    EDIRecord record;
    int nfreq = 0;

    // Component arrays (indexed by freq)
    RealVec zxxr, zxxi, zxxv;
    RealVec zxyr, zxyi, zxyv;
    RealVec zyxr, zyxi, zyxv;
    RealVec zyyr, zyyi, zyyv;
    RealVec txr, txi, txv;
    RealVec tyr, tyi, tyv;

    // Track which components are present
    bool has_zxx = false, has_zxy = false, has_zyx = false, has_zyy = false;
    bool has_tx = false, has_ty = false;

    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty() || line[0] == ';') continue;  // comment or empty

        // === HEAD section ===
        if (line.substr(0, 5) == ">HEAD") {
            while (std::getline(ifs, line)) {
                line = trim(line);
                if (line.empty() || line[0] == ';') continue;
                if (line[0] == '>') break;

                auto dataid = extract_value(line, "DATAID");
                if (!dataid.empty()) record.station.name = dataid;

                auto lat = extract_value(line, "LAT");
                if (!lat.empty()) {
                    record.station.lat = parse_coordinate(lat);
                    record.station.has_geo = true;
                }

                auto lon = extract_value(line, "LONG");
                if (lon.empty()) lon = extract_value(line, "LON");
                if (!lon.empty()) {
                    record.station.lon = parse_coordinate(lon);
                    record.station.has_geo = true;
                }

                auto elev = extract_value(line, "ELEV");
                if (!elev.empty()) {
                    try { record.station.z = std::stod(elev); } catch (...) {}
                }
            }
            // Fall through to process current line
        }

        // === MTSECT section (may contain NFREQ) ===
        if (line.find(">=MTSECT") != std::string::npos ||
            line.find(">MTSECT") != std::string::npos) {
            while (std::getline(ifs, line)) {
                line = trim(line);
                if (line.empty() || line[0] == ';') continue;
                if (line[0] == '>') break;

                auto nf_val = extract_value(line, "NFREQ");
                if (!nf_val.empty()) {
                    try { nfreq = std::stoi(nf_val); } catch (...) {}
                }
            }
            // Fall through
        }

        // === FREQ section ===
        if (line.find(">FREQ") == 0) {
            int n = extract_count(line);
            if (n > 0) nfreq = n;

            if (nfreq <= 0) {
                throw std::runtime_error("EDI: cannot determine NFREQ in " + edi_path.string());
            }
            record.frequencies = read_float_block(ifs, nfreq);
            if (static_cast<int>(record.frequencies.size()) != nfreq) {
                throw std::runtime_error("EDI: expected " + std::to_string(nfreq) +
                                         " frequencies, got " +
                                         std::to_string(record.frequencies.size()));
            }
            continue;
        }

        if (nfreq <= 0) continue;  // haven't seen FREQ yet

        // === Impedance sections ===
        // Match: >ZXXR, >ZXXR ROT=ZROT //39, >ZXXR ROT=0.0, etc.
        if (section_starts_with(line, ">ZXXR")) { zxxr = read_float_block(ifs, nfreq); has_zxx = true; continue; }
        if (section_starts_with(line, ">ZXXI")) { zxxi = read_float_block(ifs, nfreq); continue; }
        if (line.find(">ZXX.VAR") == 0)         { zxxv = read_float_block(ifs, nfreq); continue; }

        if (section_starts_with(line, ">ZXYR")) { zxyr = read_float_block(ifs, nfreq); has_zxy = true; continue; }
        if (section_starts_with(line, ">ZXYI")) { zxyi = read_float_block(ifs, nfreq); continue; }
        if (line.find(">ZXY.VAR") == 0)         { zxyv = read_float_block(ifs, nfreq); continue; }

        if (section_starts_with(line, ">ZYXR")) { zyxr = read_float_block(ifs, nfreq); has_zyx = true; continue; }
        if (section_starts_with(line, ">ZYXI")) { zyxi = read_float_block(ifs, nfreq); continue; }
        if (line.find(">ZYX.VAR") == 0)         { zyxv = read_float_block(ifs, nfreq); continue; }

        if (section_starts_with(line, ">ZYYR")) { zyyr = read_float_block(ifs, nfreq); has_zyy = true; continue; }
        if (section_starts_with(line, ">ZYYI")) { zyyi = read_float_block(ifs, nfreq); continue; }
        if (line.find(">ZYY.VAR") == 0)         { zyyv = read_float_block(ifs, nfreq); continue; }

        // === Tipper sections ===
        // Formats: >TXR.EXP, >TXR, >TXYR.EXP
        if (section_starts_with(line, ">TXR") || line.find(">TXYR.EXP") == 0) {
            txr = read_float_block(ifs, nfreq); has_tx = true; continue;
        }
        if (section_starts_with(line, ">TXI") || line.find(">TXYI.EXP") == 0) {
            txi = read_float_block(ifs, nfreq); continue;
        }
        if (line.find(">TX.VAR") == 0 || line.find(">TXY.VAR") == 0 || line.find(">TXVAR") == 0) {
            txv = read_float_block(ifs, nfreq); continue;
        }

        if (section_starts_with(line, ">TYR") || line.find(">TYYR.EXP") == 0) {
            tyr = read_float_block(ifs, nfreq); has_ty = true; continue;
        }
        if (section_starts_with(line, ">TYI") || line.find(">TYYI.EXP") == 0) {
            tyi = read_float_block(ifs, nfreq); continue;
        }
        if (line.find(">TY.VAR") == 0 || line.find(">TYY.VAR") == 0 || line.find(">TYVAR") == 0) {
            tyv = read_float_block(ifs, nfreq); continue;
        }

        if (line.find(">END") == 0) break;
    }

    if (nfreq <= 0) {
        throw std::runtime_error("EDI: no >FREQ section found in " + edi_path.string());
    }

    // Build responses
    record.responses.resize(nfreq);

    // ===================================================================
    // Unit conversion: EDI impedance is in [mV/km/nT], convert to SI [Ohm].
    //   Z_SI [V/m / (A/m)] = Z_EDI [mV/km / nT] × μ₀ × 10³
    // where μ₀ = 4π × 10⁻⁷.  Factor = 4π × 10⁻⁴ ≈ 1.2566e-3.
    // Variance scales as factor², error (std dev) as factor.
    // Tipper is dimensionless → no conversion needed.
    // ===================================================================
    // EDI sign convention:
    // The YTL (Korean) EDI files use e^{+iωt} convention, where Im(Zxy)>0
    // for a normal halfspace. The forward solver (e^{-iωt} physics convention,
    // Helmholtz K-iωσM) also produces Im(Zxy)>0 for a normal halfspace.
    // Both conventions agree on the sign of the imaginary part, so
    // NO conjugation is needed — store Re and Im directly.
    // ===================================================================
    static const Real Z_CONV = 4.0 * M_PI * 1e-4;  // [mV/km/nT] → [Ohm]

    auto fill_z_datum = [&](data::Datum& d, const RealVec& re, const RealVec& im,
                            const RealVec& var, int idx, bool present) {
        if (!present) {
            d.weight = 0.0;
            return;
        }
        Real r = (idx < static_cast<int>(re.size())) ? re[idx] : 0.0;
        Real i = (idx < static_cast<int>(im.size())) ? im[idx] : 0.0;
        // No conjugation needed: EDI data already in e^{+iωt} convention
        d.value = Complex(r * Z_CONV, i * Z_CONV);

        if (idx < static_cast<int>(var.size()) && var[idx] > 0.0) {
            d.error = std::sqrt(var[idx]) * Z_CONV;  // variance→std dev, then scale
        } else {
            d.error = 1.0 * Z_CONV;  // default if no variance
        }
        d.weight = 1.0;
    };

    auto fill_tip_datum = [&](data::Datum& d, const RealVec& re, const RealVec& im,
                              const RealVec& var, int idx, bool present) {
        if (!present) {
            d.weight = 0.0;
            return;
        }
        Real r = (idx < static_cast<int>(re.size())) ? re[idx] : 0.0;
        Real i = (idx < static_cast<int>(im.size())) ? im[idx] : 0.0;
        // No conjugation needed: EDI data already in e^{+iωt} convention
        d.value = Complex(r, i);

        if (idx < static_cast<int>(var.size()) && var[idx] > 0.0) {
            d.error = std::sqrt(var[idx]);
        } else {
            d.error = 0.1;  // default tipper error
        }
        d.weight = 1.0;
    };

    for (int f = 0; f < nfreq; ++f) {
        auto& resp = record.responses[f];
        fill_z_datum(resp.Zxx, zxxr, zxxi, zxxv, f, has_zxx);
        fill_z_datum(resp.Zxy, zxyr, zxyi, zxyv, f, has_zxy);
        fill_z_datum(resp.Zyx, zyxr, zyxi, zyxv, f, has_zyx);
        fill_z_datum(resp.Zyy, zyyr, zyyi, zyyv, f, has_zyy);
        fill_tip_datum(resp.Tx, txr, txi, txv, f, has_tx);
        fill_tip_datum(resp.Ty, tyr, tyi, tyv, f, has_ty);
    }

    MAPLE3DMT_LOG_INFO("Read EDI: " + record.station.name +
                   " (lat=" + std::to_string(record.station.lat) +
                   ", lon=" + std::to_string(record.station.lon) +
                   "), " + std::to_string(nfreq) + " freqs");
    return record;
}

// =========================================================================
// write_edi
// =========================================================================

void write_edi(const fs::path& edi_path,
               const data::Station& station,
               const RealVec& frequencies,
               const std::vector<data::MTResponse>& responses) {
    std::ofstream ofs(edi_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open EDI file for writing: " + edi_path.string());
    }

    int nfreq = static_cast<int>(frequencies.size());

    ofs << std::scientific << std::setprecision(6);

    // HEAD section
    ofs << ">HEAD\n";
    ofs << "  DATAID=\"" << station.name << "\"\n";
    ofs << "  LAT=" << std::fixed << std::setprecision(6) << station.lat << "\n";
    ofs << "  LONG=" << station.lon << "\n";
    ofs << "  ELEV=" << station.z << "\n";
    ofs << "\n";

    // FREQ section
    ofs << std::scientific << std::setprecision(6);
    ofs << ">FREQ NFREQ=" << nfreq << "\n";
    for (int f = 0; f < nfreq; ++f) {
        ofs << "  " << frequencies[f];
        if ((f + 1) % 6 == 0 || f == nfreq - 1) ofs << "\n";
    }
    ofs << "\n";

    // Inverse unit conversion: SI [Ohm] → EDI [mV/km/nT]
    static const Real Z_CONV = 4.0 * M_PI * 1e-4;
    static const Real Z_INV  = 1.0 / Z_CONV;

    // Lambda to write an impedance section (with SI→EDI conversion)
    auto write_z_section = [&](const std::string& tag_r, const std::string& tag_i,
                               const std::string& tag_var,
                               auto get_datum) {
        bool any_active = false;
        for (int f = 0; f < nfreq; ++f) {
            if (get_datum(responses[f]).weight > 0) { any_active = true; break; }
        }
        if (!any_active) return;

        ofs << ">" << tag_r << " ROT=0.0\n";
        for (int f = 0; f < nfreq; ++f) {
            ofs << "  " << get_datum(responses[f]).value.real() * Z_INV;
            if ((f + 1) % 6 == 0 || f == nfreq - 1) ofs << "\n";
        }

        // Write imag directly (e^{+iωt} convention in both internal and EDI)
        ofs << ">" << tag_i << " ROT=0.0\n";
        for (int f = 0; f < nfreq; ++f) {
            ofs << "  " << get_datum(responses[f]).value.imag() * Z_INV;
            if ((f + 1) % 6 == 0 || f == nfreq - 1) ofs << "\n";
        }

        ofs << ">" << tag_var << "\n";
        for (int f = 0; f < nfreq; ++f) {
            Real e = get_datum(responses[f]).error * Z_INV;
            ofs << "  " << (e * e);  // std dev → variance in EDI units
            if ((f + 1) % 6 == 0 || f == nfreq - 1) ofs << "\n";
        }
        ofs << "\n";
    };

    // Tipper writer (no unit conversion)
    auto write_tip_section = [&](const std::string& tag_r, const std::string& tag_i,
                                 const std::string& tag_var,
                                 auto get_datum) {
        bool any_active = false;
        for (int f = 0; f < nfreq; ++f) {
            if (get_datum(responses[f]).weight > 0) { any_active = true; break; }
        }
        if (!any_active) return;

        ofs << ">" << tag_r << " ROT=0.0\n";
        for (int f = 0; f < nfreq; ++f) {
            ofs << "  " << get_datum(responses[f]).value.real();
            if ((f + 1) % 6 == 0 || f == nfreq - 1) ofs << "\n";
        }
        // Write imag directly (e^{+iωt} convention)
        ofs << ">" << tag_i << " ROT=0.0\n";
        for (int f = 0; f < nfreq; ++f) {
            ofs << "  " << get_datum(responses[f]).value.imag();
            if ((f + 1) % 6 == 0 || f == nfreq - 1) ofs << "\n";
        }
        ofs << ">" << tag_var << "\n";
        for (int f = 0; f < nfreq; ++f) {
            Real e = get_datum(responses[f]).error;
            ofs << "  " << (e * e);
            if ((f + 1) % 6 == 0 || f == nfreq - 1) ofs << "\n";
        }
        ofs << "\n";
    };

    write_z_section("ZXXR", "ZXXI", "ZXX.VAR",
                    [](const data::MTResponse& r) -> const data::Datum& { return r.Zxx; });
    write_z_section("ZXYR", "ZXYI", "ZXY.VAR",
                    [](const data::MTResponse& r) -> const data::Datum& { return r.Zxy; });
    write_z_section("ZYXR", "ZYXI", "ZYX.VAR",
                    [](const data::MTResponse& r) -> const data::Datum& { return r.Zyx; });
    write_z_section("ZYYR", "ZYYI", "ZYY.VAR",
                    [](const data::MTResponse& r) -> const data::Datum& { return r.Zyy; });

    write_tip_section("TXR", "TXI", "TX.VAR",
                      [](const data::MTResponse& r) -> const data::Datum& { return r.Tx; });
    write_tip_section("TYR", "TYI", "TY.VAR",
                      [](const data::MTResponse& r) -> const data::Datum& { return r.Ty; });

    ofs << ">END\n";
    ofs.close();

    MAPLE3DMT_LOG_INFO("Wrote EDI: " + station.name + " -> " + edi_path.string());
}

// =========================================================================
// load_edi_directory
// =========================================================================

void load_edi_directory(const fs::path& dir, data::MTData& out) {
    if (!fs::is_directory(dir)) {
        throw std::runtime_error("Not a directory: " + dir.string());
    }

    // Collect all .edi files (case-insensitive extension)
    std::vector<fs::path> edi_files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".edi") {
            edi_files.push_back(entry.path());
        }
    }
    std::sort(edi_files.begin(), edi_files.end());

    if (edi_files.empty()) {
        throw std::runtime_error("No .edi files found in " + dir.string());
    }

    // Read all EDI files (skip malformed ones gracefully)
    std::vector<EDIRecord> records;
    records.reserve(edi_files.size());
    for (const auto& f : edi_files) {
        try {
            records.push_back(read_edi(f));
        } catch (const std::exception& e) {
            MAPLE3DMT_LOG_WARNING("Skipping " + f.filename().string() + ": " + e.what());
        }
    }

    if (records.empty()) {
        throw std::runtime_error("No valid .edi files found in " + dir.string());
    }

    // Union of all frequencies (sorted, deduplicated with tolerance)
    std::set<Real> freq_set;
    for (const auto& rec : records) {
        for (Real f : rec.frequencies) {
            bool found = false;
            for (Real existing : freq_set) {
                if (std::abs(f - existing) / std::max(std::abs(f), 1e-30) < 1e-6) {
                    found = true;
                    break;
                }
            }
            if (!found) freq_set.insert(f);
        }
    }

    RealVec all_freqs(freq_set.begin(), freq_set.end());
    // Sort descending (convention: highest frequency first)
    std::sort(all_freqs.begin(), all_freqs.end(), std::greater<Real>());

    out = data::MTData();
    out.set_frequencies(all_freqs);

    // Add stations and map frequencies
    for (const auto& rec : records) {
        int sidx = out.add_station(rec.station);

        for (int fi = 0; fi < static_cast<int>(rec.frequencies.size()); ++fi) {
            Real f = rec.frequencies[fi];
            int global_fi = -1;
            for (int gi = 0; gi < static_cast<int>(all_freqs.size()); ++gi) {
                if (std::abs(f - all_freqs[gi]) / std::max(std::abs(f), 1e-30) < 1e-6) {
                    global_fi = gi;
                    break;
                }
            }
            if (global_fi >= 0) {
                out.set_observed(sidx, global_fi, rec.responses[fi]);
            }
        }

        // Set weight=0 for missing frequency-station pairs
        for (int gi = 0; gi < static_cast<int>(all_freqs.size()); ++gi) {
            bool found = false;
            for (Real f : rec.frequencies) {
                if (std::abs(f - all_freqs[gi]) / std::max(std::abs(f), 1e-30) < 1e-6) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                data::MTResponse empty;
                empty.Zxx.weight = 0.0; empty.Zxy.weight = 0.0;
                empty.Zyx.weight = 0.0; empty.Zyy.weight = 0.0;
                empty.Tx.weight = 0.0;  empty.Ty.weight = 0.0;
                out.set_observed(sidx, gi, empty);
            }
        }
    }

    MAPLE3DMT_LOG_INFO("Loaded " + std::to_string(records.size()) + " stations, " +
                   std::to_string(all_freqs.size()) + " frequencies from " + dir.string());
}

// =========================================================================
// save_edi_directory
// =========================================================================

void save_edi_directory(const fs::path& dir, const data::MTData& data) {
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }

    for (int s = 0; s < data.num_stations(); ++s) {
        const auto& station = data.station(s);
        std::string filename = station.name.empty()
            ? "station_" + std::to_string(s) + ".edi"
            : station.name + ".edi";

        std::vector<data::MTResponse> responses(data.num_frequencies());
        for (int f = 0; f < data.num_frequencies(); ++f) {
            responses[f] = data.observed(s, f);
        }

        write_edi(dir / filename, station, data.frequencies(), responses);
    }

    MAPLE3DMT_LOG_INFO("Saved " + std::to_string(data.num_stations()) +
                   " EDI files to " + dir.string());
}

} // namespace io
} // namespace maple3dmt
