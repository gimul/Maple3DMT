// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/**
 * @file memory_profiler.h
 * @brief RSS snapshot profiler for memory leak detection.
 *
 * DC3D에서 실증: 8개 지점 스냅샷으로 +3MB/iter leak → 0.2MB/iter 개선.
 * macOS: mach_task_basic_info, Linux: /proc/self/status VmRSS.
 */

#include <string>
#include <vector>
#include <cstdio>
#include <algorithm>

#ifdef __APPLE__
#include <mach/mach.h>
#else
#include <fstream>
#include <cstring>
#endif

#include <mpi.h>

namespace maple3dmt {

class MemoryProfiler {
public:
    struct Snapshot {
        std::string stage;
        double rss_mb;
        double peak_mb;
        double delta_mb;
        int extra;      // e.g., num_stored_solutions
        double time_s;  // wall time since first snapshot
    };

    MemoryProfiler() : peak_rss_(0), t0_(-1.0) {}

    /// Take RSS snapshot at a named stage
    void snap(const std::string& stage, int extra = 0) {
        double now = MPI_Wtime();
        if (t0_ < 0) t0_ = now;

        size_t rss = get_rss_bytes();
        peak_rss_ = std::max(peak_rss_, rss);

        double rss_mb = rss / 1048576.0;
        double peak_mb = peak_rss_ / 1048576.0;
        double delta = snapshots_.empty() ? 0.0 : rss_mb - snapshots_.back().rss_mb;

        snapshots_.push_back({stage, rss_mb, peak_mb, delta, extra, now - t0_});
    }

    /// Print full report (rank 0 only)
    void report(MPI_Comm comm = MPI_COMM_WORLD) const {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank != 0 || snapshots_.empty()) return;

        std::printf("\n══════════════ Memory Profile ══════════════\n");
        std::printf("  %-30s %8s %8s %9s %8s\n",
                    "Stage", "RSS(MB)", "Peak", "Delta", "Time(s)");
        std::printf("  ──────────────────────────────────────────────────────\n");

        for (const auto& s : snapshots_) {
            char delta_buf[16];
            std::snprintf(delta_buf, sizeof(delta_buf), "%+.1f", s.delta_mb);
            std::printf("  %-30s %8.1f %8.1f %9s %8.1f",
                        s.stage.c_str(), s.rss_mb, s.peak_mb, delta_buf, s.time_s);
            if (s.extra > 0)
                std::printf("  [%d]", s.extra);
            std::printf("\n");
        }

        // Iteration leak detection
        detect_leaks();
        std::printf("════════════════════════════════════════════\n\n");
    }

    /// Check if RSS exceeds limit (bytes). Returns true if over limit.
    static bool check_limit(size_t limit_bytes, MPI_Comm comm = MPI_COMM_WORLD) {
        size_t rss = get_rss_bytes();
        size_t max_rss = 0;
        MPI_Allreduce(&rss, &max_rss, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
        return max_rss > limit_bytes;
    }

    /// Get current RSS in bytes
    static size_t get_rss_bytes() {
#ifdef __APPLE__
        struct mach_task_basic_info info;
        mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                      (task_info_t)&info, &count) == KERN_SUCCESS) {
            return info.resident_size;
        }
        return 0;
#else
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line)) {
            if (line.compare(0, 6, "VmRSS:") == 0) {
                size_t kb = 0;
                std::sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
                return kb * 1024;
            }
        }
        return 0;
#endif
    }

    /// Get current RSS in MB (convenience)
    static double get_rss_mb() { return get_rss_bytes() / 1048576.0; }

    /// Clear all snapshots
    void clear() { snapshots_.clear(); peak_rss_ = 0; t0_ = -1.0; }

    const std::vector<Snapshot>& snapshots() const { return snapshots_; }

private:
    std::vector<Snapshot> snapshots_;
    size_t peak_rss_;
    double t0_;

    /// Detect iteration-level leaks by finding "outer_iter" snapshots
    void detect_leaks() const {
        std::vector<const Snapshot*> iter_snaps;
        for (const auto& s : snapshots_) {
            if (s.stage.find("outer_iter") != std::string::npos ||
                s.stage.find("gn_iter") != std::string::npos) {
                iter_snaps.push_back(&s);
            }
        }
        if (iter_snaps.size() < 2) return;

        double avg_delta = 0;
        for (size_t i = 1; i < iter_snaps.size(); ++i) {
            avg_delta += iter_snaps[i]->rss_mb - iter_snaps[i-1]->rss_mb;
        }
        avg_delta /= (iter_snaps.size() - 1);

        if (avg_delta > 1.0) {
            std::printf("  ⚠ LEAK DETECTED: +%.1f MB/iteration average\n", avg_delta);
        } else if (avg_delta > 0.1) {
            std::printf("  ⚡ Minor leak: +%.1f MB/iteration\n", avg_delta);
        } else {
            std::printf("  ✓ No iteration leak detected (%.2f MB/iter)\n", avg_delta);
        }
    }
};

} // namespace maple3dmt
