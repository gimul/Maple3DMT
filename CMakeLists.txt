# Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
# All rights reserved.

cmake_minimum_required(VERSION 3.20)
project(Maple3DMT
    VERSION 0.1.0
    DESCRIPTION "3D MT forward modelling and inversion with terrain-fitted hex mesh"
    LANGUAGES CXX C
)

# ============================================================================
# Build options
# ============================================================================
option(MAPLE3DMT_USE_MPI       "Enable MPI parallelism"                ON)
option(MAPLE3DMT_USE_MFEM     "Enable MFEM-based FEM solver"          ON)
option(MAPLE3DMT_USE_OCTREE   "Enable Octree FV solver (p4est)"       OFF)
option(MAPLE3DMT_USE_MUMPS     "Enable MUMPS direct solver"            OFF)
option(MAPLE3DMT_USE_STRUMPACK "Enable STRUMPACK solver (HSS/BLR)"    OFF)
option(MAPLE3DMT_USE_PETSC     "Enable PETSc"                         OFF)
option(MAPLE3DMT_BUILD_TESTS   "Build unit tests"                     ON)
option(MAPLE3DMT_BUILD_EXAMPLES "Build example drivers"               ON)

# ============================================================================
# C++ standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ============================================================================
# Dependencies
# ============================================================================

# --- MFEM (required when USE_MFEM=ON) ---
if(MAPLE3DMT_USE_MFEM)
    find_package(MFEM REQUIRED
        HINTS ${MFEM_DIR} $ENV{MFEM_DIR}
              /opt/homebrew/lib/cmake/mfem
              /opt/homebrew
              /usr/local
    )
    add_compile_definitions(MAPLE3DMT_USE_MFEM)
endif()

# --- p4est (required when USE_OCTREE=ON) ---
if(MAPLE3DMT_USE_OCTREE)
    set(P4EST_DIR "$ENV{HOME}/local/p4est" CACHE PATH "p4est install dir")
    find_library(P4EST_LIB p4est HINTS ${P4EST_DIR}/lib REQUIRED NO_DEFAULT_PATH)
    find_library(SC_LIB sc HINTS ${P4EST_DIR}/lib REQUIRED NO_DEFAULT_PATH)
    find_path(P4EST_INCLUDE_DIR p8est.h HINTS ${P4EST_DIR}/include REQUIRED)
    message(STATUS "p4est library: ${P4EST_LIB}")
    message(STATUS "libsc library: ${SC_LIB}")
    add_compile_definitions(MAPLE3DMT_USE_OCTREE)
endif()

# --- MPI ---
if(MAPLE3DMT_USE_MPI)
    find_package(MPI REQUIRED)
    add_compile_definitions(MAPLE3DMT_USE_MPI)
endif()

# --- MUMPS ---
if(MAPLE3DMT_USE_MUMPS)
    # MUMPS is typically found through MFEM's configuration
    add_compile_definitions(MAPLE3DMT_USE_MUMPS)
endif()

# --- STRUMPACK (optional) ---
if(MAPLE3DMT_USE_STRUMPACK)
    find_package(STRUMPACK REQUIRED
        HINTS ${STRUMPACK_DIR} $ENV{STRUMPACK_DIR}
    )
    add_compile_definitions(MAPLE3DMT_USE_STRUMPACK)
endif()

# --- yaml-cpp ---
find_package(yaml-cpp REQUIRED HINTS /usr/local /opt/homebrew)

# --- HDF5 ---
find_package(HDF5 REQUIRED COMPONENTS CXX)

# ============================================================================
# Maple3DMT core library
# ============================================================================
# Core sources shared by both FEM and FV backends
set(MAPLE3DMT_SOURCES_COMMON
    src/data/mt_data.cpp
    src/data/static_shift.cpp
    src/io/edi_io.cpp
    src/io/hdf5_io.cpp
    src/source/em_source.cpp
    src/model/conductivity_model.cpp
    src/mesh/dem.cpp
    src/utils/logger.cpp
    src/utils/freq_parallel.cpp
)

# MFEM FEM backend sources
set(MAPLE3DMT_SOURCES_MFEM
    src/io/vtk_export_3d.cpp
    src/mesh/hex_mesh_3d.cpp
    src/forward/forward_solver_3d.cpp
    src/forward/complex_solver.cpp
    src/forward/mumps_solver_ext.cpp
    src/regularization/regularization.cpp
    src/inversion/inversion_3d.cpp
)

# Octree FV backend sources
set(MAPLE3DMT_SOURCES_OCTREE
    src/octree/octree_mesh.cpp
    src/octree/staggered_grid.cpp
    src/octree/operators.cpp
    src/forward/bicgstab.cpp
    src/forward/cocg.cpp
    src/forward/cocr.cpp
    src/forward/qmr.cpp
    src/forward/forward_solver_fv.cpp
    src/inversion/regularization_octree.cpp
    src/inversion/implicit_jtj_fv.cpp
    src/inversion/inversion_fv.cpp
    src/io/vtk_export_octree.cpp
)

# Assemble source list based on enabled backends
set(MAPLE3DMT_SOURCES ${MAPLE3DMT_SOURCES_COMMON})
if(MAPLE3DMT_USE_MFEM)
    list(APPEND MAPLE3DMT_SOURCES ${MAPLE3DMT_SOURCES_MFEM})
endif()
if(MAPLE3DMT_USE_OCTREE)
    list(APPEND MAPLE3DMT_SOURCES ${MAPLE3DMT_SOURCES_OCTREE})
endif()

add_library(maple3dmt ${MAPLE3DMT_SOURCES})

target_include_directories(maple3dmt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${HDF5_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
)

target_link_libraries(maple3dmt
    PUBLIC
        ${HDF5_LIBRARIES}
        yaml-cpp::yaml-cpp
)

# --- MFEM + HYPRE (when USE_MFEM=ON) ---
if(MAPLE3DMT_USE_MFEM)
    target_include_directories(maple3dmt PUBLIC ${MFEM_INCLUDE_DIRS})

    set(HYPRE_DIR "/Users/gimul-studio/local/hypre-2.31" CACHE PATH "HYPRE 2.31 install dir")
    find_library(HYPRE_LIB HYPRE HINTS ${HYPRE_DIR}/lib REQUIRED NO_DEFAULT_PATH)
    target_include_directories(maple3dmt PUBLIC ${HYPRE_DIR}/include)
    message(STATUS "HYPRE library: ${HYPRE_LIB}")

    set(CMAKE_INSTALL_RPATH "${HYPRE_DIR}/lib")
    set(CMAKE_BUILD_RPATH "${HYPRE_DIR}/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_RPATH TRUE)

    target_link_libraries(maple3dmt PUBLIC ${MFEM_LIBRARIES} ${HYPRE_LIB})
endif()

# --- p4est + libsc (when USE_OCTREE=ON) ---
if(MAPLE3DMT_USE_OCTREE)
    target_include_directories(maple3dmt PUBLIC ${P4EST_INCLUDE_DIR})
    target_link_libraries(maple3dmt PUBLIC ${P4EST_LIB} ${SC_LIB} z)
endif()

if(MAPLE3DMT_USE_MPI)
    target_link_libraries(maple3dmt PUBLIC MPI::MPI_CXX)
endif()

if(MAPLE3DMT_USE_MUMPS)
    # MUMPS is Fortran-based — need gfortran runtime
    find_library(GFORTRAN_LIB gfortran
        HINTS /opt/homebrew/lib/gcc/current
              /opt/homebrew/Cellar/gcc/15.2.0_1/lib/gcc/current)
    if(GFORTRAN_LIB)
        target_link_libraries(maple3dmt PUBLIC ${GFORTRAN_LIB})
    endif()
endif()

if(MAPLE3DMT_USE_STRUMPACK)
    target_link_libraries(maple3dmt PUBLIC STRUMPACK::strumpack)
endif()

# ============================================================================
# Examples
# ============================================================================
if(MAPLE3DMT_BUILD_EXAMPLES)
    add_executable(mesh_preview examples/mesh_preview.cpp)
    target_link_libraries(mesh_preview maple3dmt)

    add_executable(forward_test examples/forward_test.cpp)
    target_link_libraries(forward_test maple3dmt)

    add_executable(adjoint_test examples/adjoint_test.cpp)
    target_link_libraries(adjoint_test maple3dmt)

    add_executable(inversion_test examples/inversion_test.cpp)
    target_link_libraries(inversion_test maple3dmt)

    add_executable(gradient_check examples/gradient_check.cpp)
    target_link_libraries(gradient_check maple3dmt)

    add_executable(mesh_test examples/mesh_test.cpp)
    target_link_libraries(mesh_test maple3dmt)

    add_executable(real_forward_test examples/real_forward_test.cpp)
    target_link_libraries(real_forward_test maple3dmt)

    add_executable(real_inversion examples/real_inversion.cpp)
    target_link_libraries(real_inversion maple3dmt)

    add_executable(mpi_test examples/mpi_test.cpp)
    target_link_libraries(mpi_test maple3dmt)

    if(MAPLE3DMT_USE_OCTREE)
        add_executable(test_octree_basic examples/test_octree_basic.cpp)
        target_link_libraries(test_octree_basic maple3dmt)

        add_executable(test_forward_fv examples/test_forward_fv.cpp)
        target_link_libraries(test_forward_fv maple3dmt)

        add_executable(gradient_check_fv examples/gradient_check_fv.cpp)
        target_link_libraries(gradient_check_fv maple3dmt)

        add_executable(commemi_3d1 examples/commemi_3d1.cpp)
        target_link_libraries(commemi_3d1 maple3dmt)

        add_executable(halfspace_1d_test examples/halfspace_1d_test.cpp)
        target_link_libraries(halfspace_1d_test maple3dmt)

        add_executable(test_halfspace examples/test_halfspace.cpp)
        target_link_libraries(test_halfspace maple3dmt)

        add_executable(test_1d_layer examples/test_1d_layer.cpp)
        target_link_libraries(test_1d_layer maple3dmt)

        add_executable(real_inversion_octree examples/real_inversion_octree.cpp)
        target_link_libraries(real_inversion_octree maple3dmt)

        add_executable(test_block_uniform examples/test_block_uniform.cpp)
        target_link_libraries(test_block_uniform maple3dmt)

        add_executable(inversion_test_fv examples/inversion_test_fv.cpp)
        target_link_libraries(inversion_test_fv maple3dmt)

        add_executable(commemi_3d1_inversion examples/commemi_3d1_inversion.cpp)
        target_link_libraries(commemi_3d1_inversion maple3dmt)

        add_executable(commemi_3d2_inversion examples/commemi_3d2_inversion.cpp)
        target_link_libraries(commemi_3d2_inversion maple3dmt)

        # COMMEMI benchmark examples (EDI-based workflow)
        add_executable(generate_commemi_3d1a_data examples/commemi_3d1a/generate_data.cpp)
        target_link_libraries(generate_commemi_3d1a_data maple3dmt)

        add_executable(generate_commemi_3d2_data examples/commemi_3d2/generate_data.cpp)
        target_link_libraries(generate_commemi_3d2_data maple3dmt)

        add_executable(mpi_scalability_test examples/mpi_scalability_test.cpp)
        target_link_libraries(mpi_scalability_test maple3dmt)
    endif()
endif()

# ============================================================================
# Print configuration summary
# ============================================================================
message(STATUS "=====================================================")
message(STATUS "  Maple3DMT v${PROJECT_VERSION}")
message(STATUS "=====================================================")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  MPI          : ${MAPLE3DMT_USE_MPI}")
message(STATUS "  MFEM (FEM)   : ${MAPLE3DMT_USE_MFEM}")
message(STATUS "  Octree (FV)  : ${MAPLE3DMT_USE_OCTREE}")
message(STATUS "  MUMPS (BLR)  : ${MAPLE3DMT_USE_MUMPS}")
message(STATUS "  STRUMPACK    : ${MAPLE3DMT_USE_STRUMPACK}")
message(STATUS "  Tests        : ${MAPLE3DMT_BUILD_TESTS}")
message(STATUS "  Examples     : ${MAPLE3DMT_BUILD_EXAMPLES}")
message(STATUS "=====================================================")
