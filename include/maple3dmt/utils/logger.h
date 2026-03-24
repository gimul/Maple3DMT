// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file logger.h
/// @brief Simple logging utility for NewMT.

#include "maple3dmt/common.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace maple3dmt {
namespace utils {

enum class LogLevel {
    DEBUG   = 0,
    INFO    = 1,
    WARNING = 2,
    ERROR   = 3,
    NONE    = 4
};

/// Singleton logger.
class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) { level_ = level; }
    LogLevel level() const { return level_; }
    void set_file(const fs::path& path);

    void debug(const std::string& msg)   { log_(LogLevel::DEBUG, msg); }
    void info(const std::string& msg)    { log_(LogLevel::INFO, msg); }
    void warning(const std::string& msg) { log_(LogLevel::WARNING, msg); }
    void error(const std::string& msg)   { log_(LogLevel::ERROR, msg); }

#ifdef MAPLE3DMT_USE_MPI
    void set_rank(int rank) { rank_ = rank; }
#endif
    int rank() const { return rank_; }

private:
    Logger() = default;
    LogLevel level_ = LogLevel::INFO;
    std::ofstream file_;
    int rank_ = 0;

    void log_(LogLevel level, const std::string& msg);
};

/// RAII guard to temporarily change log level.
/// Restores the original level when destroyed.
class LogLevelGuard {
public:
    explicit LogLevelGuard(LogLevel temporary_level)
        : saved_(Logger::instance().level()) {
        Logger::instance().set_level(temporary_level);
    }
    ~LogLevelGuard() { Logger::instance().set_level(saved_); }
    LogLevelGuard(const LogLevelGuard&) = delete;
    LogLevelGuard& operator=(const LogLevelGuard&) = delete;
private:
    LogLevel saved_;
};

// Convenience macros
#define MAPLE3DMT_LOG_DEBUG(msg)   maple3dmt::utils::Logger::instance().debug(msg)
#define MAPLE3DMT_LOG_INFO(msg)    maple3dmt::utils::Logger::instance().info(msg)
#define MAPLE3DMT_LOG_WARNING(msg) maple3dmt::utils::Logger::instance().warning(msg)
#define MAPLE3DMT_LOG_ERROR(msg)   maple3dmt::utils::Logger::instance().error(msg)

} // namespace utils
} // namespace maple3dmt
