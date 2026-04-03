// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file memory.h
/// @brief Memory monitoring utilities for macOS/Linux.

#pragma once

#include <cstddef>
#include <string>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <sys/resource.h>
#include <fstream>
#endif

namespace maple3dmt {
namespace utils {

/// Current process RSS in bytes
inline size_t current_rss_bytes() {
#ifdef __APPLE__
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
#else
    std::ifstream f("/proc/self/statm");
    size_t pages = 0;
    f >> pages;  // first field = total program size
    f >> pages;  // second field = resident set size
    return pages * 4096;
#endif
}

/// Current process RSS in GB
inline double current_rss_gb() {
    return static_cast<double>(current_rss_bytes()) / (1024.0 * 1024.0 * 1024.0);
}

/// Total physical memory in bytes
inline size_t total_physical_memory() {
#ifdef __APPLE__
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    int64_t mem = 0;
    size_t len = sizeof(mem);
    sysctl(mib, 2, &mem, &len, nullptr, 0);
    return static_cast<size_t>(mem);
#else
    std::ifstream f("/proc/meminfo");
    std::string label;
    size_t kb = 0;
    f >> label >> kb;
    return kb * 1024;
#endif
}

/// Available physical memory in bytes (approximate)
inline size_t available_memory() {
#ifdef __APPLE__
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm),
                          &count) == KERN_SUCCESS) {
        size_t page_size = 4096;
        return (vm.free_count + vm.inactive_count) * page_size;
    }
    return total_physical_memory() / 2;  // fallback
#else
    std::ifstream f("/proc/meminfo");
    std::string label;
    size_t kb = 0;
    // Skip MemTotal, MemFree
    f >> label >> kb >> label;  // MemTotal
    f >> label >> kb >> label;  // MemFree
    f >> label >> kb;           // MemAvailable
    return kb * 1024;
#endif
}

/// Total physical memory in GB
inline double total_memory_gb() {
    return static_cast<double>(total_physical_memory()) / (1024.0 * 1024.0 * 1024.0);
}

/// Available memory in GB
inline double available_memory_gb() {
    return static_cast<double>(available_memory()) / (1024.0 * 1024.0 * 1024.0);
}

/// Format memory as human-readable string
inline std::string fmt_mem_gb(double gb) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f GB", gb);
    return buf;
}

} // namespace utils
} // namespace maple3dmt
