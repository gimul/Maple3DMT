// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file test_octree_basic.cpp
/// @brief Basic octree mesh test: create, refine, staggered grid, operators.

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/octree/staggered_grid.h"
#include "maple3dmt/octree/operators.h"
#include "maple3dmt/utils/logger.h"

#include <iostream>
#include <cmath>
#include <map>
#include <mpi.h>

using namespace maple3dmt;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    utils::Logger::instance().set_rank(rank);

    // ---- 1. Create octree mesh ----
    octree::RefinementParams params;
    params.domain_x_min = -50000;  params.domain_x_max = 50000;
    params.domain_y_min = -50000;  params.domain_y_max = 50000;
    params.domain_z_min = -50000;  params.domain_z_max = 20000;  // depth to air
    params.min_level = 3;
    params.max_level = 6;
    params.station_refine_radius = 10000;
    params.station_refine_level = 6;
    params.sigma_bg = 0.01;

    // Single station at center
    std::vector<std::array<Real,3>> stations = {{0.0, 0.0, 0.0}};
    RealVec freqs = {0.01, 0.1, 1.0, 10.0};

    octree::OctreeMesh mesh;
    mesh.setup(params, stations, freqs, MPI_COMM_WORLD);

    std::cout << "\n=== Octree Mesh Summary ===" << std::endl;
    std::cout << "  Cells (global): " << mesh.num_cells_global() << std::endl;
    std::cout << "  Cells (local):  " << mesh.num_cells_local() << std::endl;

    // Check cell data
    Real z_min = 1e30, z_max = -1e30;
    Real h_min = 1e30, h_max = 0;
    for (int i = 0; i < mesh.num_cells_local(); ++i) {
        Real x, y, z;
        mesh.cell_center(i, x, y, z);
        z_min = std::min(z_min, z);
        z_max = std::max(z_max, z);
        h_min = std::min(h_min, mesh.cell_size(i));
        h_max = std::max(h_max, mesh.cell_size(i));
    }
    std::cout << "  z range: [" << z_min << ", " << z_max << "]" << std::endl;
    std::cout << "  h range: [" << h_min << ", " << h_max << "]" << std::endl;

    // Level distribution
    std::map<int,int> level_counts;
    for (int i = 0; i < mesh.num_cells_local(); ++i)
        level_counts[mesh.cell_level(i)]++;
    std::cout << "  Level distribution:" << std::endl;
    for (auto& [lev, cnt] : level_counts)
        std::cout << "    level " << lev << ": " << cnt << " cells" << std::endl;

    // ---- 2. Set terrain (flat at z=0) ----
    mesh.set_terrain([](Real /*x*/, Real /*y*/) { return 0.0; });

    int n_earth = 0, n_air = 0;
    for (int i = 0; i < mesh.num_cells_local(); ++i) {
        if (mesh.cell_type(i) == octree::CellType::EARTH) ++n_earth;
        else ++n_air;
    }
    std::cout << "  Earth cells: " << n_earth << ", Air cells: " << n_air << std::endl;

    // ---- 3. Build staggered grid ----
    mesh.build_staggered_grid();
    const auto& sg = mesh.staggered();

    std::cout << "\n=== Staggered Grid ===" << std::endl;
    std::cout << "  Edges (E-field DOFs): " << sg.num_edges() << std::endl;
    std::cout << "  Faces (H-field DOFs): " << sg.num_faces() << std::endl;
    std::cout << "  Cells (σ DOFs):       " << sg.num_cells() << std::endl;

    // ---- 4. Build discrete operators ----
    octree::DiscreteOperators ops;
    ops.build(sg);

    std::cout << "\n=== Discrete Operators ===" << std::endl;
    std::cout << "  Curl Ce:      " << ops.curl_e2f().nrows << " x " << ops.curl_e2f().ncols
              << ", nnz=" << ops.curl_e2f().nnz() << std::endl;
    std::cout << "  Curl Cf (Ce^T): " << ops.curl_f2e().nrows << " x " << ops.curl_f2e().ncols
              << ", nnz=" << ops.curl_f2e().nnz() << std::endl;
    std::cout << "  Divergence D: " << ops.divergence().nrows << " x " << ops.divergence().ncols
              << ", nnz=" << ops.divergence().nnz() << std::endl;

    // ---- 5. Verify div(curl(E)) = 0 ----
    // Create random E vector on edges
    int ne = ops.num_edges();
    int nf = ops.num_faces();
    int nc = ops.num_cells();

    RealVec E_test(ne);
    for (int i = 0; i < ne; ++i)
        E_test[i] = std::sin(i * 0.1) + std::cos(i * 0.03);

    // curl(E) → B on faces
    RealVec B(nf);
    ops.curl_e2f().matvec(E_test, B);

    // div(B) → scalar on cells
    RealVec divB(nc);
    ops.divergence().matvec(B, divB);

    Real divB_norm = 0;
    for (auto v : divB) divB_norm += v * v;
    divB_norm = std::sqrt(divB_norm);

    Real E_norm = 0;
    for (auto v : E_test) E_norm += v * v;
    E_norm = std::sqrt(E_norm);

    std::cout << "\n=== Identity Check: div(curl(E)) = 0 ===" << std::endl;
    std::cout << "  ||E|| = " << E_norm << std::endl;
    std::cout << "  ||div(curl(E))|| = " << divB_norm << std::endl;
    std::cout << "  Relative: " << divB_norm / E_norm << std::endl;
    if (divB_norm / E_norm < 1e-10)
        std::cout << "  ✓ PASS" << std::endl;
    else
        std::cout << "  ✗ FAIL" << std::endl;

    // ---- 6. Assemble system matrix ----
    RealVec sigma(nc, 0.01);  // uniform 0.01 S/m
    Real omega = constants::TWOPI * 1.0;  // 1 Hz

    SparseMatC A;
    ops.assemble_system(omega, sigma, A);

    std::cout << "\n=== System Matrix A ===" << std::endl;
    std::cout << "  Size: " << A.nrows << " x " << A.ncols << std::endl;
    std::cout << "  NNZ: " << A.nnz() << std::endl;
    std::cout << "  NNZ/row: " << (A.nrows > 0 ? (double)A.nnz() / A.nrows : 0) << std::endl;

    // ---- 7. VTK output ----
    mesh.write_vtk("octree_test");
    std::cout << "\n  VTK output: octree_test.vtu" << std::endl;

    std::cout << "\n=== Test Complete ===" << std::endl;

    MPI_Finalize();
    return 0;
}
