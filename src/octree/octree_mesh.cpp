// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#include "maple3dmt/octree/octree_mesh.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/octree/staggered_grid.h"
#include "maple3dmt/utils/logger.h"

#define LOG_INFO(msg) MAPLE3DMT_LOG_INFO(msg)
#define LOG_WARNING(msg) MAPLE3DMT_LOG_WARNING(msg)
#include <numeric>

namespace maple3dmt {
namespace octree {

OctreeMesh::~OctreeMesh() {
    if (mesh_)   p8est_mesh_destroy(mesh_);
    if (ghost_)  p8est_ghost_destroy(ghost_);
    if (forest_) p8est_destroy(forest_);
    if (conn_)   p8est_connectivity_destroy(conn_);
}

void OctreeMesh::setup(const RefinementParams& params,
                        const std::vector<std::array<Real,3>>& station_xyz,
                        const RealVec& frequencies,
                        MPI_Comm comm) {
    // In replicated mode, each rank builds the full mesh independently.
    // This is used for frequency-parallel inversion (ModEM strategy).
    comm_   = params.replicate_mesh ? MPI_COMM_SELF : comm;
    params_ = params;

    LOG_INFO("OctreeMesh::setup — building octree mesh");

    // 1. Create single-brick connectivity spanning the full domain
    build_connectivity_();

    // 2. Create uniform forest at min_level
    forest_ = p8est_new_ext(comm_, conn_, 0, params_.min_level,
                             /*fill_uniform=*/1, /*data_size=*/0,
                             /*init_fn=*/nullptr, /*user_pointer=*/this);

    LOG_INFO("  Uniform base: " + std::to_string(forest_->global_num_quadrants) + " cells at level " +
             std::to_string(params_.min_level));

    // 3. Adaptive refinement
    refine_around_stations_(station_xyz);
    refine_regions_();
    refine_skin_depth_(frequencies);

    // 4. 2:1 balance (FACE connectivity) + partition
    balance_and_partition_();

    // 5. Extract cell data
    extract_cell_data_();

    // 6. Build face adjacency
    build_adjacency_();

    LOG_INFO("  Final mesh: " + std::to_string(num_cells_global_) + " cells global, " +
             std::to_string(num_cells_local_) + " local");
}

void OctreeMesh::set_terrain(std::function<Real(Real x, Real y)> dem_func) {
    terrain_func_ = std::move(dem_func);

    // Refine near terrain surface
    refine_terrain_();

    // Re-balance and repartition
    balance_and_partition_();
    extract_cell_data_();
    build_adjacency_();

    // Classify Earth/Air
    classify_earth_air_();
}

void OctreeMesh::build_staggered_grid() {
    staggered_ = std::make_unique<StaggeredGrid>();
    staggered_->build(*this);
}

// =========================================================================
// Internal: connectivity (single unit cube mapped to domain)
// =========================================================================
void OctreeMesh::build_connectivity_() {
    // p8est_connectivity_new_unitcube creates a single-tree [0,1]^3 mesh.
    // We store the domain extents in params_ and map coordinates during extraction.
    conn_ = p8est_connectivity_new_unitcube();
}

// =========================================================================
// Internal: uniform refinement
// =========================================================================
void OctreeMesh::refine_uniform_(int level) {
    // Already done by p8est_new_ext with fill_uniform=1
    (void)level;
}

// =========================================================================
// Internal: refine around stations
// =========================================================================

namespace {
    // Callback data for station refinement
    struct StationRefineCtx {
        const OctreeMesh* mesh;
        const std::vector<std::array<Real,3>>* stations;
    };

    // Compute minimum distance from a point to an axis-aligned box.
    // If the point is inside the box, returns 0.
    inline double box_point_dist2(double px, double py, double pz,
                                   double bx_min, double by_min, double bz_min,
                                   double bx_max, double by_max, double bz_max) {
        auto clamp_dist = [](double v, double lo, double hi) -> double {
            if (v < lo) return lo - v;
            if (v > hi) return v - hi;
            return 0.0;
        };
        double dx = clamp_dist(px, bx_min, bx_max);
        double dy = clamp_dist(py, by_min, by_max);
        double dz = clamp_dist(pz, bz_min, bz_max);
        return dx*dx + dy*dy + dz*dz;
    }

    int station_refine_fn(p8est_t* p8est, p4est_topidx_t /*which_tree*/,
                          p8est_quadrant_t* quadrant) {
        auto* ctx = static_cast<StationRefineCtx*>(p8est->user_pointer);
        const auto& params = ctx->mesh->params();

        if (quadrant->level >= static_cast<int8_t>(params.station_refine_level))
            return 0;

        // p4est coordinates: quad->x,y,z are integers in [0, P8EST_ROOT_LEN)
        // Cell size in integer coords: P8EST_QUADRANT_LEN(level)
        double rl = (double)P8EST_ROOT_LEN;
        p4est_qcoord_t qlen = P8EST_QUADRANT_LEN(quadrant->level);

        // Quadrant corners in [0,1]^3
        double q0x = (double)quadrant->x / rl;
        double q0y = (double)quadrant->y / rl;
        double q0z = (double)quadrant->z / rl;
        double q1x = (double)(quadrant->x + qlen) / rl;
        double q1y = (double)(quadrant->y + qlen) / rl;
        double q1z = (double)(quadrant->z + qlen) / rl;

        // Map to physical domain (box corners)
        double Lx = params.domain_x_max - params.domain_x_min;
        double Ly = params.domain_y_max - params.domain_y_min;
        double Lz = params.domain_z_max - params.domain_z_min;

        double bx_min = params.domain_x_min + q0x * Lx;
        double by_min = params.domain_y_min + q0y * Ly;
        double bz_min = params.domain_z_min + q0z * Lz;
        double bx_max = params.domain_x_min + q1x * Lx;
        double by_max = params.domain_y_min + q1y * Ly;
        double bz_max = params.domain_z_min + q1z * Lz;

        // Limit station refinement height in air.
        // σ_air ≈ 0 → fields are smooth → only 2 fine cells above surface.
        // Station z is typically ≈ 0 (or terrain elevation).
        double domain_diag = std::max({Lx, Ly, Lz});
        double h_finest = domain_diag / (1 << params.station_refine_level);
        double air_max_z = 2.0 * h_finest;  // 2 fine cells above station z

        // Check minimum distance from cell box to any station
        double r2_thresh = params.station_refine_radius * params.station_refine_radius;
        for (const auto& st : *ctx->stations) {
            // Skip if cell is entirely above the air height limit
            if (bz_min > st[2] + air_max_z)
                continue;

            double d2 = box_point_dist2(st[0], st[1], st[2],
                                         bx_min, by_min, bz_min,
                                         bx_max, by_max, bz_max);
            if (d2 < r2_thresh)
                return 1;  // refine
        }
        return 0;
    }
} // anonymous

void OctreeMesh::refine_around_stations_(const std::vector<std::array<Real,3>>& stations) {
    if (stations.empty()) return;

    StationRefineCtx ctx{this, &stations};
    void* saved = forest_->user_pointer;
    forest_->user_pointer = &ctx;

    // Iteratively refine (each pass refines by 1 level)
    for (int pass = 0; pass < params_.max_level - params_.min_level; ++pass) {
        p8est_refine(forest_, /*recursive=*/0, station_refine_fn, /*init_fn=*/nullptr);
    }

    forest_->user_pointer = saved;
    LOG_INFO("  After station refinement: " + std::to_string(forest_->global_num_quadrants) + " cells");
}

// =========================================================================
// Internal: refine regions (anomaly bodies)
// =========================================================================
namespace {
    struct RegionRefineCtx {
        const OctreeMesh* mesh;
        const RefineRegion* region;
    };

    int region_refine_fn(p8est_t* p8est, p4est_topidx_t /*which_tree*/,
                         p8est_quadrant_t* quadrant) {
        auto* ctx = static_cast<RegionRefineCtx*>(p8est->user_pointer);
        const auto& params = ctx->mesh->params();
        const auto& reg = *ctx->region;

        if (quadrant->level >= static_cast<int8_t>(reg.level))
            return 0;

        double rl = (double)P8EST_ROOT_LEN;
        p4est_qcoord_t qlen = P8EST_QUADRANT_LEN(quadrant->level);

        double Lx = params.domain_x_max - params.domain_x_min;
        double Ly = params.domain_y_max - params.domain_y_min;
        double Lz = params.domain_z_max - params.domain_z_min;

        double bx_min = params.domain_x_min + (double)quadrant->x / rl * Lx;
        double by_min = params.domain_y_min + (double)quadrant->y / rl * Ly;
        double bz_min = params.domain_z_min + (double)quadrant->z / rl * Lz;
        double bx_max = params.domain_x_min + (double)(quadrant->x + qlen) / rl * Lx;
        double by_max = params.domain_y_min + (double)(quadrant->y + qlen) / rl * Ly;
        double bz_max = params.domain_z_min + (double)(quadrant->z + qlen) / rl * Lz;

        // Check if cell box overlaps the padded region
        double pad = reg.padding;
        if (bx_max < reg.x_min - pad || bx_min > reg.x_max + pad) return 0;
        if (by_max < reg.y_min - pad || by_min > reg.y_max + pad) return 0;
        if (bz_max < reg.z_min - pad || bz_min > reg.z_max + pad) return 0;

        return 1;  // overlaps → refine
    }
}

void OctreeMesh::refine_regions_() {
    if (params_.refine_regions.empty()) return;

    for (size_t ri = 0; ri < params_.refine_regions.size(); ++ri) {
        RegionRefineCtx ctx{this, &params_.refine_regions[ri]};
        void* saved = forest_->user_pointer;
        forest_->user_pointer = &ctx;

        int max_passes = params_.refine_regions[ri].level - params_.min_level;
        for (int pass = 0; pass < max_passes; ++pass) {
            p8est_refine(forest_, /*recursive=*/0, region_refine_fn, /*init_fn=*/nullptr);
        }

        forest_->user_pointer = saved;
        LOG_INFO("  After region " + std::to_string(ri) + " refinement: " +
                 std::to_string(forest_->global_num_quadrants) + " cells");
    }
}

// =========================================================================
// Internal: refine for skin depth (ModEM-style depth-dependent refinement)
// =========================================================================

namespace {
    struct SkinDepthRefineCtx {
        const OctreeMesh* mesh;
        // Horizontal bounding box for the inversion region
        double region_x_min, region_x_max;
        double region_y_min, region_y_max;
        // Depth layers: (z_top, z_bottom, target_level)
        // z_top >= z_bottom (z is positive up: z=0 is surface, z<0 is depth)
        struct Layer { double z_top, z_bot; int level; };
        std::vector<Layer> layers;
    };

    int skin_depth_refine_fn(p8est_t* p8est, p4est_topidx_t /*which_tree*/,
                             p8est_quadrant_t* quadrant) {
        auto* ctx = static_cast<SkinDepthRefineCtx*>(p8est->user_pointer);
        const auto& params = ctx->mesh->params();

        // Map quadrant to physical coordinates
        double rl = (double)P8EST_ROOT_LEN;
        p4est_qcoord_t qlen = P8EST_QUADRANT_LEN(quadrant->level);

        double Lx = params.domain_x_max - params.domain_x_min;
        double Ly = params.domain_y_max - params.domain_y_min;
        double Lz = params.domain_z_max - params.domain_z_min;

        double bx_min = params.domain_x_min + (double)quadrant->x / rl * Lx;
        double by_min = params.domain_y_min + (double)quadrant->y / rl * Ly;
        double bz_min = params.domain_z_min + (double)quadrant->z / rl * Lz;
        double bx_max = params.domain_x_min + (double)(quadrant->x + qlen) / rl * Lx;
        double by_max = params.domain_y_min + (double)(quadrant->y + qlen) / rl * Ly;
        double bz_max = params.domain_z_min + (double)(quadrant->z + qlen) / rl * Lz;

        // Check horizontal overlap with inversion region
        if (bx_max < ctx->region_x_min || bx_min > ctx->region_x_max) return 0;
        if (by_max < ctx->region_y_min || by_min > ctx->region_y_max) return 0;

        // Check depth layers: find highest target level for this cell's depth
        int target_level = params.min_level;
        for (const auto& layer : ctx->layers) {
            // Cell z range [bz_min, bz_max] overlaps layer [z_bot, z_top]?
            if (bz_max > layer.z_bot && bz_min < layer.z_top) {
                target_level = std::max(target_level, layer.level);
            }
        }

        return (quadrant->level < static_cast<int8_t>(target_level)) ? 1 : 0;
    }
} // anonymous

void OctreeMesh::refine_skin_depth_(const RealVec& frequencies) {
    if (frequencies.empty()) return;

    // Skin depths at min and max frequencies
    Real f_min = *std::min_element(frequencies.begin(), frequencies.end());
    Real f_max = *std::max_element(frequencies.begin(), frequencies.end());
    Real delta_max = std::sqrt(2.0 / (constants::TWOPI * f_min *
                                       constants::MU0 * params_.sigma_bg));
    Real delta_min = std::sqrt(2.0 / (constants::TWOPI * f_max *
                                       constants::MU0 * params_.sigma_bg));

    LOG_INFO("  Skin depth: fmin=" + std::to_string(f_min) +
             " Hz → δ=" + std::to_string(delta_max/1000) + " km"
             ", fmax=" + std::to_string(f_max) +
             " Hz → δ=" + std::to_string(delta_min/1000) + " km");

    // Skin-depth refinement provides BACKGROUND resolution across the
    // inversion volume.  Station refinement (separate pass) adds the finest
    // cells near stations.  The two complement each other:
    //
    //   Station refine:  max_level within refine_radius of each station
    //   Skin-depth:      max_level-2  (surface to 0.5δ)   ← "medium"
    //                    max_level-3  (0.5δ to 1.5δ)      ← "coarse"
    //                    max_level-4  (1.5δ to 3δ)        ← "background"
    //
    // Why NOT max_level everywhere: with domain ±200 km and level 8,
    // a 35 km thick layer of 1.67 km cells produces ~300K cells in layer 1
    // alone — far too many.  At level 6 (6.67 km), the same layer has
    // only ~5K cells, and station refinement adds ~6K level-8 cells.
    //
    // Target: ~30K-80K total cells (comparable to ModEM structured grid).

    SkinDepthRefineCtx ctx;
    ctx.mesh = this;

    // Horizontal region: station array + 0.5×δ_max + transition_padding
    // (not full δ — station refinement already covers near-station area)
    Real pad_h = 0.5 * delta_max + params_.transition_padding;
    if (params_.station_x_max > params_.station_x_min) {
        ctx.region_x_min = params_.station_x_min - pad_h;
        ctx.region_x_max = params_.station_x_max + pad_h;
        ctx.region_y_min = params_.station_y_min - pad_h;
        ctx.region_y_max = params_.station_y_max + pad_h;
    } else {
        Real sta_half_x = (params_.domain_x_max - params_.domain_x_min) * 0.2;
        Real sta_half_y = (params_.domain_y_max - params_.domain_y_min) * 0.2;
        ctx.region_x_min = -sta_half_x - pad_h;
        ctx.region_x_max =  sta_half_x + pad_h;
        ctx.region_y_min = -sta_half_y - pad_h;
        ctx.region_y_max =  sta_half_y + pad_h;
    }
    ctx.region_x_min = std::max(ctx.region_x_min, params_.domain_x_min);
    ctx.region_x_max = std::min(ctx.region_x_max, params_.domain_x_max);
    ctx.region_y_min = std::max(ctx.region_y_min, params_.domain_y_min);
    ctx.region_y_max = std::min(ctx.region_y_max, params_.domain_y_max);

    // Depth layers: use max_level-1 for the near-surface layer where
    // high-frequency sensitivity is concentrated.  Station refinement
    // adds max_level right at stations.
    //
    // Estimated cell counts (±210 km domain, max_level=8):
    //   L7 (3.3 km): ~28×28×17 = 13K cells in 0→36 km layer
    //   L6 (6.7 km): ~14×14×9  = 1.8K cells in 36→107 km layer
    //   L5 (13.3 km): ~7×7×9   = 0.4K cells in 107→214 km layer
    //   Station L8 (1.7 km): ~5K cells (27 stations × ~200 cells/station)
    //   Total target: ~25-40K cells (comparable to ModEM)
    //
    // Background level strategy (skin-depth refinement fills BETWEEN stations):
    //   Station refinement already places max_level cells near each station.
    //   So the background ("fill") only needs max_level-2 to provide
    //   reasonable interpolation between stations. Using max_level-1 for
    //   the entire horizontal region at L9+ creates too many cells (~700K+).
    //
    //   For max_level <= 8: L1=max-1 (coarse enough, e.g. L7=3.3km for ±210km)
    //   For max_level >= 9: L1=max-2 (e.g. L7 for L9, station refine=L9)
    int level_offset = (params_.max_level >= 9) ? 2 : 1;
    int L1 = std::max(params_.min_level + 1, params_.max_level - level_offset); // near-surface
    int L2 = std::max(params_.min_level + 1, params_.max_level - level_offset - 1); // mid-depth
    int L3 = std::max(params_.min_level,     params_.max_level - level_offset - 2); // deep
    int L_air = std::max(params_.min_level,  params_.max_level - level_offset - 1); // air

    // Depth-of-interest cutoff (z positive up, depth is negative)
    // Below this z, mesh becomes very coarse (min_level+1).
    Real z_interest_bot = -3.0 * delta_max;  // default: full 3×skin depth
    if (params_.max_interest_depth > 0) {
        z_interest_bot = -params_.max_interest_depth;
        LOG_INFO("  Depth of interest: 0 to " +
                 std::to_string(params_.max_interest_depth/1000) + " km");
    }

    // Layer 1: surface to 0.5×δ (clipped to interest depth)
    Real z_l1_bot = std::max(-0.5 * delta_max, z_interest_bot);
    if (z_l1_bot < 0)
        ctx.layers.push_back({0.0, z_l1_bot, L1});

    // Layer 2: 0.5δ to 1.5δ (only within interest depth)
    Real z_l2_top = -0.5 * delta_max;
    Real z_l2_bot = std::max(-1.5 * delta_max, z_interest_bot);
    if (z_l2_top > z_interest_bot && z_l2_bot < z_l2_top)
        ctx.layers.push_back({z_l2_top, z_l2_bot, L2});

    // Layer 3: 1.5δ to 3δ (only within interest depth)
    Real z_l3_top = -1.5 * delta_max;
    Real z_l3_bot = std::max(-3.0 * delta_max, z_interest_bot);
    if (z_l3_top > z_interest_bot && z_l3_bot < z_l3_top && L3 > params_.min_level)
        ctx.layers.push_back({z_l3_top, z_l3_bot, L3});

    // Below interest depth: very coarse (min_level+1) for BC validity
    int L_deep = params_.min_level + 1;
    if (z_interest_bot > params_.domain_z_min && L_deep > params_.min_level)
        ctx.layers.push_back({z_interest_bot, params_.domain_z_min, L_deep});

    // Air layers — minimal refinement, aggressive coarsening.
    //
    // Physics: σ_air ≈ 0 → curl-curl equation has no iωσ term in air.
    // Fields vary very smoothly (potential-like), so coarse cells suffice.
    //
    // Requirements:
    //   1) First 1~2 cells above z=0 ≈ Earth surface cell size
    //      → E/H tangential continuity at air-earth interface
    //   2) Above that: rapid geometric coarsening (factor ~2× per layer)
    //   3) Total air height > max skin depth for BC validity
    //
    // ModEM uses 8~12 air layers: first ~100m, growth 1.5~2×.
    // Our octree equivalent: 2 thin graded layers + base cells above.
    //
    // Background air: one level coarser than Earth surface (L1-1).
    // Station refinement already places max_level cells in the air
    // directly above each station, providing E/H continuity where
    // it matters.  The background air only needs moderate resolution.
    int L_air_surface = std::max(params_.min_level + 1, L1 - 1);     // one coarser than Earth
    int L_air_mid     = std::max(params_.min_level, L_air_surface - 1);  // two coarser
    int L_air_upper   = std::max(params_.min_level, L_air_mid - 1);     // three coarser

    // Air layers use configurable thicknesses from RefinementParams.
    Real z_air_1 = params_.air_surface_thickness;                    // default 2 km
    Real z_air_2 = z_air_1 + params_.air_mid_thickness;             // default 10 km
    Real z_air_3 = z_air_2 + params_.air_upper_thickness;           // default 30 km

    // Thin layer right above surface: same as Earth (E/H continuity)
    ctx.layers.push_back({z_air_1, 0.0, L_air_surface});

    // Transition layer: one level coarser
    ctx.layers.push_back({z_air_2, z_air_1, L_air_mid});

    // Upper air: two levels coarser — only for BC distance
    ctx.layers.push_back({z_air_3, z_air_2, L_air_upper});

    // Above 10 km: no explicit refinement — falls to base octree level.
    // p4est 2:1 balancing ensures no abrupt size jumps.

    LOG_INFO("  Skin depth refinement layers:");
    for (const auto& l : ctx.layers) {
        LOG_INFO("    z=[" + std::to_string(l.z_bot/1000) + ", " +
                 std::to_string(l.z_top/1000) + "] km → level " +
                 std::to_string(l.level));
    }
    LOG_INFO("  Horizontal region: x=[" +
             std::to_string(ctx.region_x_min/1000) + ", " +
             std::to_string(ctx.region_x_max/1000) + "] y=[" +
             std::to_string(ctx.region_y_min/1000) + ", " +
             std::to_string(ctx.region_y_max/1000) + "] km");

    void* saved = forest_->user_pointer;
    forest_->user_pointer = &ctx;

    // Iteratively refine (each pass refines by 1 level)
    int max_passes = params_.max_level - params_.min_level;
    for (int pass = 0; pass < max_passes; ++pass) {
        p4est_gloidx_t before = forest_->global_num_quadrants;
        p8est_refine(forest_, /*recursive=*/0, skin_depth_refine_fn, /*init_fn=*/nullptr);
        if (forest_->global_num_quadrants == before) break;  // no change
    }

    forest_->user_pointer = saved;
    LOG_INFO("  After skin-depth refinement: " +
             std::to_string(forest_->global_num_quadrants) + " cells");
}

// =========================================================================
// Internal: refine near terrain
// =========================================================================
void OctreeMesh::refine_terrain_() {
    if (!terrain_func_) return;
    // TODO: Refine cells that intersect the terrain surface
    LOG_INFO("  Terrain refinement: TODO (using station refinement for now)");
}

// =========================================================================
// Internal: balance + partition
// =========================================================================
void OctreeMesh::balance_and_partition_() {
    p8est_balance(forest_, P8EST_CONNECT_FACE, /*init_fn=*/nullptr);
    // With replicate_mesh=true, comm_=MPI_COMM_SELF → partition is a no-op (1 rank).
    p8est_partition(forest_, /*allow_for_coarsening=*/0, /*weight_fn=*/nullptr);

    // Rebuild ghost + mesh
    if (mesh_)  { p8est_mesh_destroy(mesh_);  mesh_  = nullptr; }
    if (ghost_) { p8est_ghost_destroy(ghost_); ghost_ = nullptr; }

    ghost_ = p8est_ghost_new(forest_, P8EST_CONNECT_FACE);
    mesh_  = p8est_mesh_new(forest_, ghost_, P8EST_CONNECT_FACE);
}

// =========================================================================
// Internal: extract cell data from p8est
// =========================================================================
void OctreeMesh::extract_cell_data_() {
    num_cells_local_ = static_cast<int>(forest_->local_num_quadrants);
    num_cells_global_ = static_cast<int>(forest_->global_num_quadrants);

    cell_centers_x_.resize(num_cells_local_);
    cell_centers_y_.resize(num_cells_local_);
    cell_centers_z_.resize(num_cells_local_);
    cell_sizes_.resize(num_cells_local_);
    cell_levels_.resize(num_cells_local_);
    cell_types_.resize(num_cells_local_, CellType::EARTH);

    Real dx = params_.domain_x_max - params_.domain_x_min;
    Real dy = params_.domain_y_max - params_.domain_y_min;
    Real dz = params_.domain_z_max - params_.domain_z_min;

    int idx = 0;
    Real rl = (Real)P8EST_ROOT_LEN;
    for (p4est_topidx_t t = forest_->first_local_tree; t <= forest_->last_local_tree; ++t) {
        p8est_tree_t* tree = p8est_tree_array_index(forest_->trees, t);
        for (size_t q = 0; q < tree->quadrants.elem_count; ++q) {
            p8est_quadrant_t* quad = p8est_quadrant_array_index(&tree->quadrants, q);

            int level = quad->level;
            p4est_qcoord_t qlen = P8EST_QUADRANT_LEN(level);

            // Quadrant center in [0,1]^3
            Real cx_unit = (quad->x + qlen / 2.0) / rl;
            Real cy_unit = (quad->y + qlen / 2.0) / rl;
            Real cz_unit = (quad->z + qlen / 2.0) / rl;
            Real h_unit  = (Real)qlen / rl;

            // Map to physical domain
            cell_centers_x_[idx] = params_.domain_x_min + cx_unit * dx;
            cell_centers_y_[idx] = params_.domain_y_min + cy_unit * dy;
            cell_centers_z_[idx] = params_.domain_z_min + cz_unit * dz;

            // Cell size: minimum physical dimension of the cell
            cell_sizes_[idx] = h_unit * std::min({dx, dy, dz});
            cell_levels_[idx] = level;

            ++idx;
        }
    }
}

// =========================================================================
// Cell center access
// =========================================================================
void OctreeMesh::cell_center(int local_id, Real& x, Real& y, Real& z) const {
    x = cell_centers_x_[local_id];
    y = cell_centers_y_[local_id];
    z = cell_centers_z_[local_id];
}

Real OctreeMesh::cell_size(int local_id) const {
    return cell_sizes_[local_id];
}

void OctreeMesh::cell_size_xyz(int local_id, Real& dx, Real& dy, Real& dz) const {
    int level = cell_levels_[local_id];
    Real h_unit = 1.0 / (1 << level);
    Real dom_dx = params_.domain_x_max - params_.domain_x_min;
    Real dom_dy = params_.domain_y_max - params_.domain_y_min;
    Real dom_dz = params_.domain_z_max - params_.domain_z_min;
    dx = h_unit * dom_dx;
    dy = h_unit * dom_dy;
    dz = h_unit * dom_dz;
}

int OctreeMesh::cell_level(int local_id) const {
    return cell_levels_[local_id];
}

// =========================================================================
// Build adjacency from p8est_mesh
// =========================================================================
void OctreeMesh::build_adjacency_() {
    face_neighbors_.resize(num_cells_local_);
    for (auto& fn : face_neighbors_) fn.clear();

    Real dx = params_.domain_x_max - params_.domain_x_min;
    Real dy = params_.domain_y_max - params_.domain_y_min;
    Real dz = params_.domain_z_max - params_.domain_z_min;

    // p8est_mesh encodes face neighbors for each local quadrant.
    // mesh_->quad_to_face[6*q + f] gives the neighbor info.
    for (int q = 0; q < num_cells_local_; ++q) {
        for (int f = 0; f < 6; ++f) {
            p4est_locidx_t neighbor_qid = mesh_->quad_to_quad[6 * q + f];
            int neighbor_face = mesh_->quad_to_face[6 * q + f];

            if (neighbor_qid == q && neighbor_face == f) {
                // Boundary face — no neighbor
                continue;
            }

            FaceNeighbor fn;
            fn.direction = f;

            // Determine if neighbor is local or ghost
            if (neighbor_qid < num_cells_local_) {
                fn.neighbor_id = neighbor_qid;
                fn.ghost_id = -1;
            } else {
                fn.neighbor_id = -1;
                fn.ghost_id = neighbor_qid - num_cells_local_;
            }

            // Compute face area and center-to-center distance
            int level = cell_levels_[q];
            Real h_unit = 1.0 / (1 << level);
            Real hx = h_unit * dx, hy = h_unit * dy, hz = h_unit * dz;

            // Face area depends on face direction
            if (f == 0 || f == 1) {      // ±x face
                fn.face_area = hy * hz;
                fn.distance = hx;
            } else if (f == 2 || f == 3) { // ±y face
                fn.face_area = hx * hz;
                fn.distance = hy;
            } else {                       // ±z face
                fn.face_area = hx * hy;
                fn.distance = hz;
            }

            face_neighbors_[q].push_back(fn);
        }
    }
}

// =========================================================================
// Classify Earth/Air/Ocean based on terrain
// =========================================================================
void OctreeMesh::classify_earth_air_() {
    int n_ocean = 0;

    if (!terrain_func_) {
        // No terrain — everything below z=0 is Earth, above is Air
        for (int i = 0; i < num_cells_local_; ++i) {
            cell_types_[i] = (cell_centers_z_[i] < 0.0) ? CellType::EARTH : CellType::AIR;
        }
        return;
    }

    for (int i = 0; i < num_cells_local_; ++i) {
        Real terrain_z = terrain_func_(cell_centers_x_[i], cell_centers_y_[i]);
        // Use z-dimension half-extent (not min of all dims)
        Real cdx_i, cdy_i, cdz_i;
        cell_size_xyz(i, cdx_i, cdy_i, cdz_i);
        Real h = cdz_i;  // z-direction cell size

        if (cell_centers_z_[i] + h / 2.0 < terrain_z) {
            cell_types_[i] = CellType::EARTH;
        } else if (cell_centers_z_[i] - h / 2.0 > terrain_z) {
            cell_types_[i] = CellType::AIR;
        } else {
            // Partial cell: classify by majority
            Real earth_frac = (terrain_z - (cell_centers_z_[i] - h / 2.0)) / h;
            earth_frac = std::clamp(earth_frac, 0.0, 1.0);
            cell_types_[i] = (earth_frac > 0.5) ? CellType::EARTH : CellType::AIR;
        }

        // Ocean: terrain below sea level (terrain_z < 0) and cell is
        // below sea level but above terrain → seawater, not earth.
        // Cells below terrain (even under ocean) remain EARTH (seafloor rock).
        if (terrain_z < 0.0 && cell_types_[i] == CellType::AIR &&
            cell_centers_z_[i] < 0.0) {
            // Cell is between terrain (bathymetry) and sea level → OCEAN
            cell_types_[i] = CellType::OCEAN;
            ++n_ocean;
        }
    }

    // Count cell types and validate transition
    int n_earth = 0, n_air = 0;
    int n_ocean_adj_earth = 0;  // ocean cells adjacent to earth (seafloor boundary)
    for (int i = 0; i < num_cells_local_; ++i) {
        if (cell_types_[i] == CellType::EARTH) ++n_earth;
        else if (cell_types_[i] == CellType::AIR) ++n_air;
    }

    if (n_ocean > 0) {
        LOG_INFO("  Cell classification: EARTH=" + std::to_string(n_earth)
                 + " AIR=" + std::to_string(n_air)
                 + " OCEAN=" + std::to_string(n_ocean));
    }
}

// =========================================================================
// Validate stations are in Earth cells — fix if not
// =========================================================================
int OctreeMesh::validate_stations_earth(
        const std::vector<std::array<Real,3>>& station_xyz) {
    int n_fixed = 0;

    for (size_t s = 0; s < station_xyz.size(); ++s) {
        Real sx = station_xyz[s][0];
        Real sy = station_xyz[s][1];
        Real sz = station_xyz[s][2];  // station elevation

        // Find the cell containing this station (nearest center match)
        int best_cell = -1;
        Real best_dist2 = 1e30;
        for (int i = 0; i < num_cells_local_; ++i) {
            Real dx = cell_centers_x_[i] - sx;
            Real dy = cell_centers_y_[i] - sy;
            Real dz = cell_centers_z_[i] - sz;
            Real d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best_dist2) {
                best_dist2 = d2;
                best_cell = i;
            }
        }

        if (best_cell < 0) continue;

        if (cell_types_[best_cell] == CellType::AIR) {
            // Station is in an Air cell — fix it
            LOG_WARNING("Station " + std::to_string(s) +
                        " at (" + std::to_string(sx) + ", " +
                        std::to_string(sy) + ", " + std::to_string(sz) +
                        ") is in AIR cell " + std::to_string(best_cell) +
                        " (center z=" + std::to_string(cell_centers_z_[best_cell]) +
                        "). Reclassifying to EARTH.");
            cell_types_[best_cell] = CellType::EARTH;

            // Also reclassify cells directly below this station down to
            // a reasonable depth, to ensure forward field propagates.
            // Fix all Air cells in the same (x,y) column below the station.
            Real cdx_s, cdy_s, cdz_s;
            cell_size_xyz(best_cell, cdx_s, cdy_s, cdz_s);
            Real tol_xy = std::max(cdx_s, cdy_s) * 0.6;

            for (int i = 0; i < num_cells_local_; ++i) {
                if (cell_types_[i] != CellType::AIR) continue;
                Real dx = std::abs(cell_centers_x_[i] - sx);
                Real dy = std::abs(cell_centers_y_[i] - sy);
                if (dx < tol_xy && dy < tol_xy &&
                    cell_centers_z_[i] < cell_centers_z_[best_cell]) {
                    cell_types_[i] = CellType::EARTH;
                }
            }
            ++n_fixed;
        }
    }

    if (n_fixed > 0) {
        LOG_WARNING("validate_stations_earth: " + std::to_string(n_fixed) +
                    " / " + std::to_string(station_xyz.size()) +
                    " stations were in AIR → reclassified to EARTH.");
        LOG_WARNING("This usually means the DEM does not cover the full "
                    "station area. Check DEM coverage.");
    } else {
        LOG_INFO("validate_stations_earth: all " +
                 std::to_string(station_xyz.size()) +
                 " stations are in EARTH cells — OK.");
    }
    return n_fixed;
}

// =========================================================================
// Ghost exchange
// =========================================================================
void OctreeMesh::exchange_ghost_scalar(RealVec& /*cell_data*/) const {
    // TODO: Use p8est_ghost_exchange_data for scalar ghost values
}

// =========================================================================
// VTK output
// =========================================================================
void OctreeMesh::write_vtk(const std::string& filename,
                            const RealVec& cell_scalar,
                            const std::string& scalar_name) const {
    // Use p8est built-in VTK writer for basic output
    // TODO: Add custom cell data (sigma, cell_type)
    p8est_vtk_write_file(forest_, nullptr, filename.c_str());
    (void)cell_scalar;
    (void)scalar_name;
}

} // namespace octree
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
