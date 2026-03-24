// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

#pragma once
/// @file freq_parallel.h
/// @brief 2-level MPI parallelization: frequency groups × spatial decomposition.
///
/// Architecture:
///   MPI_COMM_WORLD is split into frequency groups, each handling a subset
///   of frequencies. Within each group, the mesh is spatially decomposed
///   across processes (standard MFEM ParMesh behavior).
///
///   MPI_COMM_WORLD (e.g., 64 processes)
///   ├── freq_group_0: ranks 0-7   → freq 0-9    (spatial_comm: 8 procs)
///   ├── freq_group_1: ranks 8-15  → freq 10-19
///   ├── ...
///   └── freq_group_7: ranks 56-63 → freq 70-80
///
/// The spatial_comm is used by MFEM for ParMesh and HYPRE for AMS.
/// The freq_comm connects same-rank processes across frequency groups
/// for gradient reduction (MPI_Allreduce).
///
/// For inversion, each frequency group computes:
///   1. Forward solve for its frequencies
///   2. Adjoint solve and gradient contribution
///   3. MPI_Allreduce gradient across freq_comm
///
/// References:
///   - ModEM (Egbert & Kelbert, 2012): frequency-parallel MPI
///   - WSINV3DMT: data-space parallel inversion
///   - "3D MT Parallel Inversion Using Data-Space Method" (AGU, 2013)

#include "maple3dmt/common.h"
#include "maple3dmt/data/mt_data.h"
#include <vector>
#ifdef MAPLE3DMT_USE_MPI
#include <mpi.h>
#endif

namespace maple3dmt {
namespace utils {

/// Manages 2-level MPI communicator splitting for frequency-parallel inversion.
///
/// Usage:
///   FreqParallelManager fpm;
///   fpm.setup(MPI_COMM_WORLD, n_total_freqs, spatial_procs_per_group);
///   auto my_freqs = fpm.my_frequency_indices();
///   MPI_Comm spatial = fpm.spatial_comm();  // use for ParMesh
///   // After local gradient computation:
///   fpm.allreduce_gradient(local_grad, global_grad);
class FreqParallelManager {
public:
    FreqParallelManager() = default;
    ~FreqParallelManager();

    /// Initialize 2-level communicator structure.
    ///
    /// @param world_comm       MPI_COMM_WORLD (or any parent communicator)
    /// @param n_total_freqs    Total number of frequencies in the dataset
    /// @param spatial_procs    Number of processes per spatial group (0 = auto)
    ///
    /// Auto mode (spatial_procs=0):
    ///   - If world_size <= n_freqs: 1 spatial proc per group (pure freq-parallel)
    ///   - If world_size > n_freqs: world_size/n_freqs spatial procs per group
    ///   - Rounds up to ensure all processes are used
    ///
    /// Example: 64 procs, 81 freqs, spatial_procs=0 → 64 groups of 1, each ~1-2 freqs
    /// Example: 64 procs, 8 freqs, spatial_procs=0 → 8 groups of 8
    /// Example: 64 procs, 81 freqs, spatial_procs=8 → 8 groups of 8, each ~10 freqs
    void setup(
#ifdef MAPLE3DMT_USE_MPI
        MPI_Comm world_comm,
#endif
        int n_total_freqs,
        int spatial_procs = 0);

    /// Get frequency indices assigned to this process's group.
    /// Returns sorted vector of 0-based frequency indices.
    const std::vector<int>& my_frequency_indices() const { return my_freq_indices_; }

    /// Number of frequency groups.
    int num_freq_groups() const { return n_freq_groups_; }

    /// This process's frequency group index.
    int my_freq_group() const { return my_freq_group_; }

    /// Number of spatial processes per frequency group.
    int spatial_size() const { return spatial_size_; }

    /// This process's rank within the spatial group.
    int spatial_rank() const { return spatial_rank_; }

#ifdef MAPLE3DMT_USE_MPI
    /// Communicator for spatial decomposition (use for ParMesh).
    /// All processes in the same frequency group share this communicator.
    MPI_Comm spatial_comm() const { return spatial_comm_; }

    /// Communicator connecting same spatial-rank across frequency groups.
    /// Used for gradient allreduce in inversion.
    MPI_Comm freq_comm() const { return freq_comm_; }
#endif

    /// Allreduce gradient vector across frequency groups.
    /// Each process contributes its local gradient (from its frequency subset).
    /// After call, global_grad contains the sum across all frequency groups.
    ///
    /// @param local_grad   Gradient from this group's frequencies (element-wise)
    /// @param global_grad  Output: sum of gradients from all groups (same size)
    void allreduce_gradient(const RealVec& local_grad, RealVec& global_grad) const;

    /// Allreduce scalar (e.g., data misfit) across frequency groups.
    Real allreduce_scalar(Real local_value) const;

    /// Synchronize predicted MTData across frequency groups.
    /// Each rank only has predicted values for its assigned frequencies;
    /// this call fills in the missing frequencies via MPI_Allreduce (sum).
    /// Must be called before exporting data_fit CSV.
    void allreduce_predicted(data::MTData& data) const;

    /// Is frequency parallelism active? (false if single group or no MPI)
    bool is_freq_parallel() const { return n_freq_groups_ > 1; }

    /// Print summary of communicator structure (rank 0 only).
    void print_summary() const;

private:
    int n_total_freqs_ = 0;
    int n_freq_groups_ = 1;
    int my_freq_group_ = 0;
    int spatial_size_ = 1;
    int spatial_rank_ = 0;
    int world_rank_ = 0;
    int world_size_ = 1;
    std::vector<int> my_freq_indices_;

#ifdef MAPLE3DMT_USE_MPI
    MPI_Comm world_comm_ = MPI_COMM_NULL;
    MPI_Comm spatial_comm_ = MPI_COMM_NULL;
    MPI_Comm freq_comm_ = MPI_COMM_NULL;
    bool comms_created_ = false;
#endif

    /// Distribute frequencies across groups (round-robin for load balance).
    void distribute_frequencies_();
};

} // namespace utils
} // namespace maple3dmt
