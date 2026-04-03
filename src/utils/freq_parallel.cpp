// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file freq_parallel.cpp
/// @brief 2-level MPI parallelization: frequency groups × spatial decomposition.

#include "maple3dmt/utils/freq_parallel.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/utils/logger.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace maple3dmt {
namespace utils {

FreqParallelManager::~FreqParallelManager() {
#ifdef MAPLE3DMT_USE_MPI
    if (comms_created_) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
            if (spatial_comm_ != MPI_COMM_NULL) MPI_Comm_free(&spatial_comm_);
            if (freq_comm_ != MPI_COMM_NULL)    MPI_Comm_free(&freq_comm_);
        }
        comms_created_ = false;
    }
#endif
}

void FreqParallelManager::setup(
#ifdef MAPLE3DMT_USE_MPI
    MPI_Comm world_comm,
#endif
    int n_total_freqs,
    int spatial_procs)
{
    n_total_freqs_ = n_total_freqs;
    if (n_total_freqs <= 0) {
        throw std::invalid_argument("FreqParallelManager: n_total_freqs must be > 0");
    }

#ifdef MAPLE3DMT_USE_MPI
    world_comm_ = world_comm;
    MPI_Comm_rank(world_comm, &world_rank_);
    MPI_Comm_size(world_comm, &world_size_);

    // ── Determine spatial group size ──
    if (spatial_procs <= 0) {
        // Auto mode: maximize frequency parallelism
        if (world_size_ <= n_total_freqs) {
            // More frequencies than processes → 1 spatial proc per group
            spatial_size_ = 1;
        } else {
            // More processes than frequencies → divide evenly
            spatial_size_ = world_size_ / n_total_freqs;
            // Ensure at least 1 frequency group per process
            if (spatial_size_ * n_total_freqs > world_size_) {
                spatial_size_ = world_size_ / n_total_freqs;
            }
        }
    } else {
        spatial_size_ = std::min(spatial_procs, world_size_);
    }

    // Ensure spatial_size divides world_size
    // If not, reduce spatial_size to largest divisor of world_size that's <= requested
    while (spatial_size_ > 1 && (world_size_ % spatial_size_ != 0)) {
        spatial_size_--;
    }

    n_freq_groups_ = world_size_ / spatial_size_;

    // ── Split communicators ──
    // spatial_comm: processes in the same frequency group (contiguous blocks)
    // freq_comm: processes with the same spatial rank across groups
    my_freq_group_ = world_rank_ / spatial_size_;
    spatial_rank_  = world_rank_ % spatial_size_;

    MPI_Comm_split(world_comm, my_freq_group_, spatial_rank_, &spatial_comm_);
    MPI_Comm_split(world_comm, spatial_rank_, my_freq_group_, &freq_comm_);
    comms_created_ = true;

#else
    world_rank_ = 0;
    world_size_ = 1;
    spatial_size_ = 1;
    n_freq_groups_ = 1;
    my_freq_group_ = 0;
    spatial_rank_ = 0;
#endif

    // ── Distribute frequencies across groups ──
    distribute_frequencies_();
}

void FreqParallelManager::distribute_frequencies_() {
    my_freq_indices_.clear();

    // Round-robin distribution for load balance.
    // High frequencies are typically cheaper (smaller skin depth, faster convergence).
    // Low frequencies are expensive. Round-robin ensures balanced load.
    //
    // Alternative: block distribution (contiguous ranges) is simpler but leads
    // to load imbalance since low-freq group takes much longer.
    //
    // Example: 81 freqs, 8 groups
    //   Group 0: freq 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80  (11 freqs)
    //   Group 1: freq 1, 9, 17, 25, 33, 41, 49, 57, 65, 73       (10 freqs)
    //   ...
    //   Group 7: freq 7, 15, 23, 31, 39, 47, 55, 63, 71, 79       (10 freqs)
    for (int f = my_freq_group_; f < n_total_freqs_; f += n_freq_groups_) {
        my_freq_indices_.push_back(f);
    }
}

void FreqParallelManager::allreduce_gradient(
    const RealVec& local_grad, RealVec& global_grad) const
{
    int n = static_cast<int>(local_grad.size());
    global_grad.resize(n);

#ifdef MAPLE3DMT_USE_MPI
    if (n_freq_groups_ > 1 && spatial_rank_ == 0) {
        // Only spatial-rank-0 processes participate in frequency allreduce
        // (all spatial ranks have the same gradient for their element subset)
        MPI_Allreduce(local_grad.data(), global_grad.data(), n,
                      MPI_DOUBLE, MPI_SUM, freq_comm_);
    } else if (n_freq_groups_ <= 1) {
        global_grad = local_grad;
    }

    // Broadcast from spatial-rank-0 to other spatial ranks (if spatial_size > 1)
    if (spatial_size_ > 1) {
        MPI_Bcast(global_grad.data(), n, MPI_DOUBLE, 0, spatial_comm_);
    }
#else
    global_grad = local_grad;
#endif
}

Real FreqParallelManager::allreduce_scalar(Real local_value) const {
#ifdef MAPLE3DMT_USE_MPI
    if (n_freq_groups_ > 1) {
        Real global_value = 0.0;
        // Only spatial-rank-0 processes contribute (avoid double counting)
        Real contrib = (spatial_rank_ == 0) ? local_value : 0.0;
        MPI_Allreduce(&contrib, &global_value, 1, MPI_DOUBLE, MPI_SUM, world_comm_);
        return global_value;
    }
#endif
    return local_value;
}

void FreqParallelManager::allreduce_predicted(data::MTData& data) const {
#ifdef MAPLE3DMT_USE_MPI
    if (n_freq_groups_ <= 1) return;  // nothing to sync

    int ns = data.num_stations();
    int nf = data.num_frequencies();
    if (ns == 0 || nf == 0) return;

    // Pack predicted data into a flat double buffer.
    // Each MTResponse has 6 Datum, each Datum has Complex value (2 doubles).
    // We only need to sync the value field (error/weight are already set everywhere).
    const int COMPONENTS = 6;  // Zxx, Zxy, Zyx, Zyy, Tx, Ty
    const int DOUBLES_PER_RESP = COMPONENTS * 2;  // real + imag per component
    int total = ns * nf * DOUBLES_PER_RESP;

    std::vector<double> local_buf(total, 0.0);

    // Pack: only fill in frequencies this rank actually solved
    auto pack_datum = [](const data::Datum& d, double* buf) {
        buf[0] = d.value.real();
        buf[1] = d.value.imag();
    };

    for (int s = 0; s < ns; ++s) {
        for (int fi : my_freq_indices_) {
            const auto& p = data.predicted(s, fi);
            int offset = (s * nf + fi) * DOUBLES_PER_RESP;
            pack_datum(p.Zxx, &local_buf[offset + 0]);
            pack_datum(p.Zxy, &local_buf[offset + 2]);
            pack_datum(p.Zyx, &local_buf[offset + 4]);
            pack_datum(p.Zyy, &local_buf[offset + 6]);
            pack_datum(p.Tx,  &local_buf[offset + 8]);
            pack_datum(p.Ty,  &local_buf[offset + 10]);
        }
    }

    // Allreduce sum — each frequency is only written by one rank, so sum = original
    std::vector<double> global_buf(total, 0.0);
    MPI_Allreduce(local_buf.data(), global_buf.data(), total,
                  MPI_DOUBLE, MPI_SUM, world_comm_);

    // Unpack back into MTData
    auto unpack_datum = [](data::Datum& d, const double* buf) {
        d.value = Complex(buf[0], buf[1]);
    };

    for (int s = 0; s < ns; ++s) {
        for (int f = 0; f < nf; ++f) {
            int offset = (s * nf + f) * DOUBLES_PER_RESP;
            data::MTResponse resp = data.predicted(s, f);  // copy to modify
            unpack_datum(resp.Zxx, &global_buf[offset + 0]);
            unpack_datum(resp.Zxy, &global_buf[offset + 2]);
            unpack_datum(resp.Zyx, &global_buf[offset + 4]);
            unpack_datum(resp.Zyy, &global_buf[offset + 6]);
            unpack_datum(resp.Tx,  &global_buf[offset + 8]);
            unpack_datum(resp.Ty,  &global_buf[offset + 10]);
            data.set_predicted(s, f, resp);
        }
    }

    MAPLE3DMT_LOG_INFO("FreqParallelManager: predicted data synchronized across " +
                     std::to_string(n_freq_groups_) + " frequency groups");
#endif
}

void FreqParallelManager::print_summary() const {
    if (world_rank_ != 0) return;

    std::ostringstream ss;
    ss << "=== Frequency Parallel Configuration ===\n"
       << "  Total MPI processes:    " << world_size_ << "\n"
       << "  Frequency groups:       " << n_freq_groups_ << "\n"
       << "  Spatial procs/group:    " << spatial_size_ << "\n"
       << "  Total frequencies:      " << n_total_freqs_ << "\n"
       << "  Freqs per group:        ~" << (n_total_freqs_ + n_freq_groups_ - 1) / n_freq_groups_ << "\n"
       << "  Distribution:           round-robin (load balanced)\n";

    if (n_freq_groups_ > 1) {
        ss << "  Speedup (ideal):        " << n_freq_groups_ << "×\n";
        ss << "  Freq-parallel:          ACTIVE\n";
    } else {
        ss << "  Freq-parallel:          DISABLED (single group)\n";
    }
    ss << "========================================\n";

    MAPLE3DMT_LOG_INFO(ss.str());
}

} // namespace utils
} // namespace maple3dmt
