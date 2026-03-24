// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file mesh_preview.cpp
/// @brief Load EDI stations → generate 3D hex mesh → export VTK for visualization.
///
/// Usage:
///   mesh_preview <edi_directory> [--dem <dem_file>] [--output <dir>]
///   mesh_preview --demo [--output <dir>]   (uses dummy stations for testing)

#include "maple3dmt/mesh/hex_mesh_3d.h"
#include "maple3dmt/io/edi_io.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/utils/logger.h"
#include <iostream>
#include <string>
#include <filesystem>

using namespace maple3dmt;

namespace {

/// Create demo stations for testing without EDI files.
std::vector<mesh::Station3D> create_demo_stations() {
    std::vector<mesh::Station3D> stations;

    // 4×3 grid of stations at ~1km spacing, centered at origin
    // Simulating a small MT survey in Korea (~35.5°N, 128.5°E)
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 4; ++i) {
            mesh::Station3D s;
            s.name = "ST" + std::to_string(j * 4 + i + 1);
            s.x = (i - 1.5) * 1000.0;  // -1500 to +1500 m
            s.y = (j - 1.0) * 1000.0;  // -1000 to +1000 m
            s.z = 0.0;
            s.lon = 128.5 + s.x / 91000.0;  // approximate
            s.lat = 35.5 + s.y / 110540.0;
            s.elevation = 0.0;
            stations.push_back(std::move(s));
        }
    }

    return stations;
}

void print_usage() {
    std::cout << "Usage:\n"
              << "  mesh_preview <edi_directory> [options]\n"
              << "  mesh_preview --demo [options]\n"
              << "\nOptions:\n"
              << "  --dem <file>    ALOS DEM file (GeoTIFF or ASCII)\n"
              << "  --output <dir>  Output directory (default: ./mesh_output)\n"
              << "  --demo          Use dummy stations for testing\n"
              << "  --no-refine     Skip station-proximity refinement\n"
              << std::endl;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    // ------------------------------------------------------------------
    // Parse command line
    // ------------------------------------------------------------------
    std::string edi_dir;
    std::string dem_file;
    std::string output_dir = "mesh_output";
    bool demo_mode = false;
    bool no_refine = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dem" && i + 1 < argc) {
            dem_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--demo") {
            demo_mode = true;
        } else if (arg == "--no-refine") {
            no_refine = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg[0] != '-') {
            edi_dir = arg;
        }
    }

    if (!demo_mode && edi_dir.empty()) {
        print_usage();
        return 1;
    }

    // ------------------------------------------------------------------
    // Load stations
    // ------------------------------------------------------------------
    std::vector<mesh::Station3D> stations;

    if (demo_mode) {
        MAPLE3DMT_LOG_INFO("=== Demo mode: using dummy stations ===");
        stations = create_demo_stations();
    } else {
        MAPLE3DMT_LOG_INFO("=== Loading EDI files from: " + edi_dir + " ===");
        data::MTData mt_data;
        io::load_edi_directory(edi_dir, mt_data);
        stations = mesh::stations_from_mt_data(mt_data);

        if (stations.empty()) {
            MAPLE3DMT_LOG_ERROR("No stations loaded from EDI directory");
            return 1;
        }
    }

    std::cout << "\n--- Stations (" << stations.size() << ") ---\n";
    for (const auto& s : stations) {
        std::cout << "  " << s.name
                  << "  x=" << s.x << "  y=" << s.y << "  z=" << s.z
                  << "  (lon=" << s.lon << ", lat=" << s.lat << ")\n";
    }

    // ------------------------------------------------------------------
    // Auto-compute mesh parameters
    // ------------------------------------------------------------------
    auto params = mesh::auto_mesh_params(stations);
    if (no_refine) {
        params.refine_near_stations = 0;
    }

    std::cout << "\n--- Mesh Parameters ---\n"
              << "  Domain X: [" << params.x_min/1000 << ", " << params.x_max/1000 << "] km\n"
              << "  Domain Y: [" << params.y_min/1000 << ", " << params.y_max/1000 << "] km\n"
              << "  Domain Z: [" << params.z_min/1000 << ", +" << params.z_air/1000 << "] km\n"
              << "  Surface h: " << params.h_surface_x << " m (x), "
              << params.h_surface_y << " m (y), " << params.h_surface_z << " m (z)\n"
              << "  ROI pad: " << params.roi_x_pad/1000 << " km (x), "
              << params.roi_y_pad/1000 << " km (y)\n"
              << "  Refinement: " << params.refine_near_stations << " levels\n";

    // ------------------------------------------------------------------
    // Load DEM (optional)
    // ------------------------------------------------------------------
    std::unique_ptr<mesh::ALOSDem> dem;
    if (!dem_file.empty()) {
        dem = std::make_unique<mesh::ALOSDem>();
        std::string ext = fs::path(dem_file).extension().string();
        if (ext == ".tif" || ext == ".tiff") {
            // Try ASCII fallback first (same dir, .txt extension)
            fs::path ascii_path = fs::path(dem_file).replace_extension(".txt");
            if (fs::exists(ascii_path)) {
                MAPLE3DMT_LOG_INFO("Using ASCII DEM: " + ascii_path.string() +
                                 " (GeoTIFF requires GDAL)");
                dem->load_ascii(ascii_path);
            } else {
                // Try load_geotiff (requires GDAL build)
                try {
                    dem->load_geotiff(dem_file);
                } catch (const std::exception& e) {
                    MAPLE3DMT_LOG_ERROR(std::string("DEM load failed: ") + e.what());
                    MAPLE3DMT_LOG_ERROR("Convert GeoTIFF to ASCII first, or rebuild with -DMAPLE3DMT_USE_GDAL=ON");
                    MAPLE3DMT_LOG_INFO("Proceeding without terrain...");
                    dem.reset();
                }
            }
        } else {
            dem->load_ascii(dem_file);
        }
        params.use_terrain = (dem != nullptr);
    } else {
        params.use_terrain = false;
    }

    // ------------------------------------------------------------------
    // Generate mesh
    // ------------------------------------------------------------------
    std::cout << "\n--- Generating 3D hex mesh ---\n";
    mesh::HexMeshGenerator3D generator;
    auto mesh = generator.generate(params, stations, dem.get());

    // Count earth/air elements
    const auto& rmap = generator.region_map();
    int n_earth = 0, n_air = 0;
    for (int r : rmap) {
        if (r == 1) ++n_earth;
        else if (r == 2) ++n_air;
    }

    std::cout << "\n--- Mesh Statistics ---\n"
              << "  Elements: " << mesh->GetNE()
              << " (earth=" << n_earth << ", air=" << n_air << ")\n"
              << "  Vertices: " << mesh->GetNV() << "\n"
              << "  Boundary: " << mesh->GetNBE() << " faces\n";

    // ------------------------------------------------------------------
    // Export
    // ------------------------------------------------------------------
    fs::create_directories(output_dir);

    // Full volume mesh (for solver)
    fs::path vtk_path = fs::path(output_dir) / "mesh.vtk";
    generator.export_vtk(*mesh, vtk_path);

    // Lightweight surface mesh (for GUI preview)
    fs::path surf_path = fs::path(output_dir) / "mesh_surface.vtk";
    generator.export_surface_vtk(*mesh, surf_path);

    fs::path csv_path = fs::path(output_dir) / "stations.csv";
    mesh::export_stations_csv(stations, csv_path);

    std::cout << "\n--- Output ---\n"
              << "  Mesh VTK: " << vtk_path << "\n"
              << "  Stations: " << csv_path << "\n";

    // ------------------------------------------------------------------
    // Print visualization hint
    // ------------------------------------------------------------------
    std::cout << "\n--- Visualization ---\n"
              << "  ParaView: paraview " << vtk_path << "\n"
              << "  PyVista:  python scripts/visualize_3d.py"
              << " --mode mesh"
              << " --vtk " << vtk_path
              << " --stations " << csv_path << "\n"
              << std::endl;

    return 0;
}
