// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#pragma once
/// @file octree_mesh.h
/// @brief p4est-based octree mesh for 3D MT FV solver.

#include "maple3dmt/common.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include <p8est.h>
#include <p8est_extended.h>
#include <p8est_ghost.h>
#include <p8est_mesh.h>
#include <p8est_vtk.h>
#include <functional>

namespace maple3dmt {

// Forward declaration
namespace data { struct Station; }

namespace octree {

// Forward declaration
class StaggeredGrid;

/// Octree mesh refinement parameters.
/// Region refinement: refine all cells overlapping a box to a given level.
struct RefineRegion {
    Real x_min, x_max, y_min, y_max, z_min, z_max;
    int level;       // target refinement level
    Real padding;    // extra padding around the box (m)
};

struct RefinementParams {
    Real domain_x_min, domain_x_max;  // m
    Real domain_y_min, domain_y_max;
    Real domain_z_min, domain_z_max;  // z_min < 0 (depth), z_max > 0 (air)

    int  min_level = 3;
    int  max_level = 8;
    Real h_min     = 50.0;    // smallest cell (m), at max_level
    Real h_max     = 10000.0; // largest cell (m), at min_level

    // Station-based refinement
    Real station_refine_radius = 5000.0;
    int  station_refine_level  = 7;

    // Region-based refinement (anomaly bodies, etc.)
    std::vector<RefineRegion> refine_regions;

    // Terrain refinement
    int  terrain_refine_level  = 6;

    // Skin-depth refinement
    int  skin_depth_levels     = 2;     // cells per skin depth
    Real sigma_bg              = 0.01;  // background conductivity for skin depth

    // Depth-of-interest limit (m, positive value = depth below surface).
    // Below this depth, skin-depth refinement is capped at min_level+1,
    // drastically reducing cell count.  0 = no limit (use full skin-depth).
    Real max_interest_depth    = 0.0;

    // Air layer thicknesses (m).  Above the surface, cells coarsen rapidly
    // because σ_air ≈ 0 → fields are potential-like.
    //   surface layer: same level as Earth surface (L1) — E/H continuity
    //   mid layer:     L1 - 1 (2× coarser)
    //   upper layer:   L1 - 2 (4× coarser)
    //   above upper:   falls to base octree level
    Real air_surface_thickness = 2000.0;   // 0~2 km: match Earth surface
    Real air_mid_thickness     = 8000.0;   // 2~10 km: one level coarser
    Real air_upper_thickness   = 20000.0;  // 10~30 km: two levels coarser

    // Transition padding (m).  Extends skin-depth refinement region beyond
    // the inversion-region boundary to produce a smoother cell-size gradient.
    // p8est 2:1 balance always applies; this adds ADDITIONAL transition.
    // 0 = rely on 2:1 balance only.  Typical: 0.5 × delta_max.
    Real transition_padding    = 0.0;

    // Station bounding box (set by driver, used for skin-depth refinement)
    Real station_x_min = 0, station_x_max = 0;
    Real station_y_min = 0, station_y_max = 0;

    // MPI mesh replication: if true, all ranks own the full mesh (no partitioning).
    // Used for frequency-parallel inversion where each rank solves a subset of
    // frequencies on the full mesh (ModEM strategy). Default: false (p8est partition).
    bool replicate_mesh        = false;
};

/// Cell attribute.
enum class CellType : int { EARTH = 0, AIR = 1, OCEAN = 2 };

/// Face neighbor info for adjacency.
struct FaceNeighbor {
    int  neighbor_id;  // local cell id (-1 if ghost)
    int  ghost_id;     // ghost layer id (-1 if local)
    Real face_area;    // m^2
    Real distance;     // center-to-center distance (m)
    int  direction;    // 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z
};

/// 3D Octree mesh manager (wraps p4est / p8est).
class OctreeMesh {
public:
    OctreeMesh() = default;
    ~OctreeMesh();

    // Non-copyable
    OctreeMesh(const OctreeMesh&) = delete;
    OctreeMesh& operator=(const OctreeMesh&) = delete;

    /// Build mesh: create uniform base, refine, balance, partition.
    void setup(const RefinementParams& params,
               const std::vector<std::array<Real,3>>& station_xyz,
               const RealVec& frequencies,
               MPI_Comm comm);

    /// Set terrain function and classify Earth/Air.
    void set_terrain(std::function<Real(Real x, Real y)> dem_func);

    // --- Mesh queries ---
    int num_cells_local() const { return num_cells_local_; }
    int num_cells_global() const { return num_cells_global_; }

    /// Cell center coordinates.
    void cell_center(int local_id, Real& x, Real& y, Real& z) const;
    /// Cell size (minimum dimension, for backward compatibility).
    Real cell_size(int local_id) const;
    /// Cell size in each axis (accounts for non-cubic domain).
    void cell_size_xyz(int local_id, Real& dx, Real& dy, Real& dz) const;
    /// Cell refinement level.
    int  cell_level(int local_id) const;
    /// Cell type (Earth/Air).
    CellType cell_type(int local_id) const { return cell_types_[local_id]; }

    // --- Adjacency ---
    const std::vector<std::vector<FaceNeighbor>>& face_neighbors() const {
        return face_neighbors_;
    }

    /// Release face_neighbors_ after StaggeredGrid + Regularization setup.
    /// Saves ~24 bytes/cell × 6 neighbors/cell = ~144 bytes/cell.
    void release_face_neighbors() {
        face_neighbors_.clear();
        face_neighbors_.shrink_to_fit();
    }

    // --- Staggered grid (built after mesh setup) ---
    const StaggeredGrid& staggered() const { return *staggered_; }
    StaggeredGrid& staggered() { return *staggered_; }
    void build_staggered_grid();

    // --- Ghost exchange ---
    void exchange_ghost_scalar(RealVec& cell_data) const;

    // --- VTK output ---
    void write_vtk(const std::string& filename,
                   const RealVec& cell_scalar = {},
                   const std::string& scalar_name = "sigma") const;

    /// Validate that all stations are in Earth cells, not Air.
    /// If any station is in Air, reclassify that cell (and neighbors below)
    /// as Earth to ensure valid forward modelling.
    /// Returns number of stations that were initially in Air.
    int validate_stations_earth(
        const std::vector<std::array<Real,3>>& station_xyz);

    // --- p4est access ---
    p8est_t*       forest()       { return forest_; }
    const p8est_t* forest() const { return forest_; }
    p8est_ghost_t* ghost()        { return ghost_; }
    MPI_Comm       comm()  const  { return comm_; }

    // --- Domain info ---
    const RefinementParams& params() const { return params_; }

private:
    p8est_connectivity_t* conn_   = nullptr;
    p8est_t*              forest_ = nullptr;
    p8est_ghost_t*        ghost_  = nullptr;
    p8est_mesh_t*         mesh_   = nullptr;
    MPI_Comm              comm_   = MPI_COMM_WORLD;
    RefinementParams      params_;

    int num_cells_local_  = 0;
    int num_cells_global_ = 0;

    // Per-cell data (local cells)
    std::vector<Real>     cell_centers_x_, cell_centers_y_, cell_centers_z_;
    std::vector<Real>     cell_sizes_;
    std::vector<int>      cell_levels_;
    std::vector<CellType> cell_types_;

    // Adjacency
    std::vector<std::vector<FaceNeighbor>> face_neighbors_;

    // Staggered grid
    std::unique_ptr<StaggeredGrid> staggered_;

    // Terrain function
    std::function<Real(Real, Real)> terrain_func_;

    // Internal refinement callbacks
    void build_connectivity_();
    void refine_uniform_(int level);
    void refine_around_stations_(const std::vector<std::array<Real,3>>& stations);
    void refine_regions_();
    void refine_skin_depth_(const RealVec& frequencies);
    void refine_terrain_();
    void balance_and_partition_();
    void extract_cell_data_();
    void build_adjacency_();
    void classify_earth_air_();
};

} // namespace octree
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
