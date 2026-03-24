// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file mesh_test.cpp
/// @brief Test 3D hex mesh generation from real EDI station data.
///
/// Loads EDI files, generates terrain-conforming mesh, exports to VTK.
/// Usage: mpirun -np 1 mesh_test --edi-dir <path> [options]

#include "maple3dmt/io/edi_io.h"
#include "maple3dmt/mesh/hex_mesh_3d.h"
#include "maple3dmt/io/vtk_export_3d.h"
#include "maple3dmt/utils/logger.h"
#include <mfem.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace maple3dmt;

int main(int argc, char* argv[]) {
    // MPI init
    mfem::Mpi::Init(argc, argv);
    int rank = mfem::Mpi::WorldRank();

    // Parse arguments
    fs::path edi_dir;
    fs::path output_dir = "mesh_test_output";
    fs::path dem_path;
    Real h_surface = 0;   // auto if 0
    Real roi_pad = 0;     // auto if 0
    int refine = 0;       // refinement levels near stations
    bool auto_params = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--edi-dir" && i + 1 < argc) {
            edi_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--dem" && i + 1 < argc) {
            dem_path = argv[++i];
        } else if (arg == "--h-surface" && i + 1 < argc) {
            h_surface = std::stod(argv[++i]);
            auto_params = false;
        } else if (arg == "--roi-pad" && i + 1 < argc) {
            roi_pad = std::stod(argv[++i]);
            auto_params = false;
        } else if (arg == "--refine" && i + 1 < argc) {
            refine = std::stoi(argv[++i]);
        }
    }

    if (edi_dir.empty()) {
        if (rank == 0) {
            std::cerr << "Usage: mesh_test --edi-dir <path> [--output dir] "
                      << "[--dem path] [--h-surface m] [--roi-pad m] [--refine n]\n";
        }
        return 1;
    }

    utils::Logger::instance().set_rank(rank);
    MAPLE3DMT_LOG_INFO("=== 3D Mesh Generation Test ===");

    // =========================================================================
    // 1. Load EDI files
    // =========================================================================
    MAPLE3DMT_LOG_INFO("Loading EDI files from: " + edi_dir.string());
    data::MTData mt_data;
    io::load_edi_directory(edi_dir, mt_data);

    int ns = mt_data.num_stations();
    int nf = mt_data.num_frequencies();
    MAPLE3DMT_LOG_INFO("Loaded " + std::to_string(ns) + " stations, " +
                   std::to_string(nf) + " frequencies");

    if (ns == 0) {
        MAPLE3DMT_LOG_ERROR("No stations loaded!");
        return 1;
    }

    // Print station info
    if (rank == 0) {
        std::cout << "\n=== Station Summary ===\n"
                  << std::setw(6) << "Name"
                  << std::setw(12) << "Lat"
                  << std::setw(12) << "Lon"
                  << std::setw(10) << "Elev(m)"
                  << "\n" << std::string(40, '-') << "\n";
        for (int i = 0; i < ns; ++i) {
            const auto& s = mt_data.station(i);
            std::cout << std::setw(6) << s.name
                      << std::fixed << std::setprecision(4)
                      << std::setw(12) << s.lat
                      << std::setw(12) << s.lon
                      << std::setprecision(1)
                      << std::setw(10) << s.z
                      << "\n";
        }

        // Frequency range
        std::cout << "\n=== Frequency Range ===\n"
                  << "  f_max = " << mt_data.frequencies()[0] << " Hz"
                  << " (T_min = " << 1.0 / mt_data.frequencies()[0] << " s)\n"
                  << "  f_min = " << mt_data.frequencies()[nf - 1] << " Hz"
                  << " (T_max = " << 1.0 / mt_data.frequencies()[nf - 1] << " s)\n";
    }

    // =========================================================================
    // 2. Convert to Station3D + auto mesh params
    // =========================================================================
    auto stations = mesh::stations_from_mt_data(mt_data);

    if (rank == 0) {
        // Print local coordinates
        std::cout << "\n=== Local Coordinates (m) ===\n"
                  << std::setw(6) << "Name"
                  << std::setw(12) << "X(E)"
                  << std::setw(12) << "Y(N)"
                  << std::setw(10) << "Z"
                  << "\n" << std::string(40, '-') << "\n";
        Real x_min = 1e30, x_max = -1e30, y_min = 1e30, y_max = -1e30;
        for (const auto& s : stations) {
            std::cout << std::setw(6) << s.name
                      << std::fixed << std::setprecision(0)
                      << std::setw(12) << s.x
                      << std::setw(12) << s.y
                      << std::setprecision(1)
                      << std::setw(10) << s.z
                      << "\n";
            x_min = std::min(x_min, s.x);
            x_max = std::max(x_max, s.x);
            y_min = std::min(y_min, s.y);
            y_max = std::max(y_max, s.y);
        }
        std::cout << "\n  Station extent: X=[" << x_min << ", " << x_max
                  << "] (" << (x_max - x_min) / 1000 << " km)\n"
                  << "                  Y=[" << y_min << ", " << y_max
                  << "] (" << (y_max - y_min) / 1000 << " km)\n";
    }

    // Auto or custom mesh params
    mesh::MeshParams3D params;
    if (auto_params) {
        params = mesh::auto_mesh_params(stations);
    }

    // Override with user-specified values
    if (h_surface > 0) {
        params.h_surface_x = h_surface;
        params.h_surface_y = h_surface;
    }
    if (roi_pad > 0) {
        params.roi_x_pad = roi_pad;
        params.roi_y_pad = roi_pad;
    }
    params.refine_near_stations = refine;

    // For this test: disable terrain (no DEM loaded) unless specified
    if (dem_path.empty()) {
        params.use_terrain = false;
    } else {
        params.dem_path = dem_path;
        params.use_terrain = true;
    }

    if (rank == 0) {
        std::cout << "\n=== Mesh Parameters ===\n"
                  << "  Domain X: [" << params.x_min << ", " << params.x_max << "] m\n"
                  << "  Domain Y: [" << params.y_min << ", " << params.y_max << "] m\n"
                  << "  Domain Z: " << params.z_min << " m (depth)\n"
                  << "  Air:      " << params.z_air << " m\n"
                  << "  h_surface_x: " << params.h_surface_x << " m\n"
                  << "  h_surface_y: " << params.h_surface_y << " m\n"
                  << "  h_surface_z: " << params.h_surface_z << " m\n"
                  << "  Growth: x=" << params.growth_x
                  << " y=" << params.growth_y
                  << " z=" << params.growth_z << "\n"
                  << "  ROI pad: x=" << params.roi_x_pad
                  << " y=" << params.roi_y_pad << " m\n"
                  << "  ROI depth: " << params.roi_depth << " m\n"
                  << "  Terrain: " << (params.use_terrain ? "ON" : "OFF") << "\n"
                  << "  Refinement: " << params.refine_near_stations << " levels\n";
    }

    // =========================================================================
    // 3. Generate mesh
    // =========================================================================
    MAPLE3DMT_LOG_INFO("Generating 3D hex mesh...");

    mesh::HexMeshGenerator3D generator;
    mesh::ALOSDem* dem_ptr = nullptr;
    mesh::ALOSDem dem;

    if (params.use_terrain && !dem_path.empty()) {
        MAPLE3DMT_LOG_INFO("Loading DEM: " + dem_path.string());
        // Try GeoTIFF first, then ASCII
        std::string ext = dem_path.extension().string();
        if (ext == ".tif" || ext == ".tiff") {
            dem.load_geotiff(dem_path);
        } else {
            dem.load_ascii(dem_path);
        }
        dem_ptr = &dem;
        MAPLE3DMT_LOG_INFO("DEM loaded: " + std::to_string(dem.n_lon) + "x" +
                       std::to_string(dem.n_lat));
    }

    auto serial_mesh = generator.generate(params, stations, dem_ptr);

    int ne = serial_mesh->GetNE();
    int nv = serial_mesh->GetNV();
    MAPLE3DMT_LOG_INFO("Mesh generated: " + std::to_string(ne) + " elements, " +
                   std::to_string(nv) + " vertices");

    // Count earth/air elements
    int n_earth = 0, n_air = 0;
    for (int e = 0; e < ne; ++e) {
        if (serial_mesh->GetAttribute(e) == 1) n_earth++;
        else n_air++;
    }

    if (rank == 0) {
        std::cout << "\n=== Mesh Statistics ===\n"
                  << "  Total elements: " << ne << "\n"
                  << "  Earth elements: " << n_earth << "\n"
                  << "  Air elements:   " << n_air << "\n"
                  << "  Vertices:       " << nv << "\n";

        // Bounding box
        mfem::Vector bb_min(3), bb_max(3);
        serial_mesh->GetBoundingBox(bb_min, bb_max);
        std::cout << "  Bounding box:\n"
                  << "    X: [" << bb_min(0)/1000 << ", " << bb_max(0)/1000 << "] km\n"
                  << "    Y: [" << bb_min(1)/1000 << ", " << bb_max(1)/1000 << "] km\n"
                  << "    Z: [" << bb_min(2)/1000 << ", " << bb_max(2)/1000 << "] km\n";
    }

    // =========================================================================
    // 4. Export
    // =========================================================================
    fs::create_directories(output_dir);

    // Mesh VTK
    fs::path mesh_vtk = output_dir / "mesh.vtk";
    generator.export_vtk(*serial_mesh, mesh_vtk);
    MAPLE3DMT_LOG_INFO("Exported mesh VTK: " + mesh_vtk.string());

    // Station CSV
    mesh::export_stations_csv(stations, output_dir / "stations.csv");
    MAPLE3DMT_LOG_INFO("Exported stations CSV");

    // Quick mesh quality check — element aspect ratio
    if (rank == 0) {
        double min_vol = 1e30, max_vol = -1e30;
        double total_vol = 0;
        for (int e = 0; e < ne; ++e) {
            double vol = serial_mesh->GetElementVolume(e);
            min_vol = std::min(min_vol, vol);
            max_vol = std::max(max_vol, vol);
            total_vol += vol;
        }
        std::cout << "\n=== Element Volume Statistics ===\n"
                  << "  Min volume: " << min_vol << " m³\n"
                  << "  Max volume: " << max_vol << " m³\n"
                  << "  Total volume: " << total_vol / 1e9 << " km³\n"
                  << "  Ratio (max/min): " << max_vol / min_vol << "\n";
    }

    if (rank == 0) {
        std::cout << "\n=== Mesh Test PASSED ===\n"
                  << "Output directory: " << output_dir.string() << "\n"
                  << "View with: python scripts/visualize_3d.py --result-dir "
                  << output_dir.string() << " --mode mesh\n";
    }

    return 0;
}
