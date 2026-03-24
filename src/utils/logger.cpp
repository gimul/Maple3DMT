// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file logger.cpp
/// @brief Logger implementation.

#include "maple3dmt/utils/logger.h"
#include <chrono>
#include <iomanip>

namespace maple3dmt {
namespace utils {

void Logger::set_file(const fs::path& path) {
    file_.open(path, std::ios::app);
}

void Logger::log_(LogLevel level, const std::string& msg) {
    if (level < level_) return;

#ifdef MAPLE3DMT_USE_MPI
    // Only rank 0 prints to console by default
    if (rank_ != 0 && level < LogLevel::WARNING) return;
#endif

    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time);

    std::ostringstream ss;
    ss << std::put_time(&tm, "%H:%M:%S") << " ";

    // Level prefix
    switch (level) {
        case LogLevel::DEBUG:   ss << "[DEBUG] "; break;
        case LogLevel::INFO:    ss << "[INFO]  "; break;
        case LogLevel::WARNING: ss << "[WARN]  "; break;
        case LogLevel::ERROR:   ss << "[ERROR] "; break;
        default: break;
    }

#ifdef MAPLE3DMT_USE_MPI
    ss << "[rank " << rank_ << "] ";
#endif

    ss << msg;

    std::string line = ss.str();
    std::cout << line << std::endl;

    if (file_.is_open()) {
        file_ << line << std::endl;
    }
}

} // namespace utils
} // namespace maple3dmt
