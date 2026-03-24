// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// All rights reserved.

/// @file vtk_export_3d.cpp
/// @brief 3D inversion result export for visualization.

#include "maple3dmt/io/vtk_export_3d.h"
#include "maple3dmt/utils/logger.h"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <mfem.hpp>

namespace maple3dmt {
namespace io {

// =========================================================================
// export_model_vtk
// =========================================================================
void export_model_vtk(const mfem::ParMesh& mesh,
                      const model::ConductivityModel& model,
                      const fs::path& path,
                      int iteration) {
    auto& mesh_nc = const_cast<mfem::ParMesh&>(mesh);
    int ne = mesh_nc.GetNE();

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open VTK file: " + path.string());
    }

    // Write mesh geometry via MFEM (includes CELL_DATA + SCALARS material)
    mesh_nc.PrintVTK(ofs);

    // Append additional cell scalars (CELL_DATA header already written by PrintVTK)
    const Real ln10 = std::log(10.0);
    constexpr double NODATA = -9999.0;

    // log10(resistivity)
    ofs << "SCALARS log10_rho double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int e = 0; e < ne; ++e) {
        if (mesh_nc.GetAttribute(e) == 2) {
            ofs << NODATA << "\n";  // air sentinel
        } else {
            Real log10_rho = -model.params()[e] / ln10;
            ofs << std::setprecision(8) << log10_rho << "\n";
        }
    }

    // conductivity (S/m)
    ofs << "SCALARS conductivity double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int e = 0; e < ne; ++e) {
        if (mesh_nc.GetAttribute(e) == 2) {
            ofs << NODATA << "\n";
        } else {
            ofs << std::setprecision(8) << model.sigma(e) << "\n";
        }
    }

    // element attribute (integer copy for filtering convenience)
    ofs << "SCALARS attribute int 1\n"
        << "LOOKUP_TABLE default\n";
    for (int e = 0; e < ne; ++e) {
        ofs << mesh_nc.GetAttribute(e) << "\n";
    }

    // iteration metadata
    if (iteration >= 0) {
        ofs << "FIELD FieldData 1\n"
            << "iteration 1 1 int\n"
            << iteration << "\n";
    }

    ofs.close();
    MAPLE3DMT_LOG_INFO("Exported model VTK: " + path.string() +
                   " (" + std::to_string(ne) + " elements)");
}

// =========================================================================
// export_model_pvtu
// =========================================================================
void export_model_pvtu(const mfem::ParMesh& mesh,
                       const model::ConductivityModel& model,
                       const fs::path& dir,
                       int iteration) {
    auto& mesh_nc = const_cast<mfem::ParMesh&>(mesh);

    fs::create_directories(dir);

    // Use MFEM's ParaViewDataCollection
    std::string prefix = "model";
    if (iteration >= 0) {
        prefix += "_iter_" + std::to_string(iteration);
    }

    mfem::L2_FECollection l2_fec(0, 3);  // order 0 = cell constant
    mfem::ParFiniteElementSpace l2_fes(&mesh_nc, &l2_fec);

    // log10(rho)
    mfem::ParGridFunction log10_rho_gf(&l2_fes);
    const Real ln10 = std::log(10.0);
    int ne = mesh_nc.GetNE();
    for (int e = 0; e < ne; ++e) {
        if (mesh_nc.GetAttribute(e) == 2) {
            log10_rho_gf(e) = 0.0;  // VTU doesn't handle NaN well; filter in Python
        } else {
            log10_rho_gf(e) = -model.params()[e] / ln10;
        }
    }

    // conductivity
    mfem::ParGridFunction sigma_gf(&l2_fes);
    for (int e = 0; e < ne; ++e) {
        sigma_gf(e) = (mesh_nc.GetAttribute(e) == 2) ? 0.0 : model.sigma(e);
    }

    // attribute
    mfem::ParGridFunction attr_gf(&l2_fes);
    for (int e = 0; e < ne; ++e) {
        attr_gf(e) = static_cast<double>(mesh_nc.GetAttribute(e));
    }

    mfem::ParaViewDataCollection pv(prefix, &mesh_nc);
    pv.SetPrefixPath(dir.string());
    pv.SetHighOrderOutput(false);
    pv.SetLevelsOfDetail(1);
    pv.RegisterField("log10_rho", &log10_rho_gf);
    pv.RegisterField("conductivity", &sigma_gf);
    pv.RegisterField("attribute", &attr_gf);
    pv.SetCycle(iteration >= 0 ? iteration : 0);
    pv.SetTime(iteration >= 0 ? static_cast<double>(iteration) : 0.0);
    pv.Save();

    MAPLE3DMT_LOG_INFO("Exported model PVTU: " + dir.string() + "/" + prefix);
}

// =========================================================================
// export_depth_slice
// =========================================================================
void export_depth_slice(const mfem::ParMesh& mesh,
                        const model::ConductivityModel& model,
                        Real depth,
                        const fs::path& path,
                        Real dx, Real dy) {
    auto& mesh_nc = const_cast<mfem::ParMesh&>(mesh);

    // Get bounding box (earth elements only)
    mfem::Vector bb_min(3), bb_max(3);
    mesh_nc.GetBoundingBox(bb_min, bb_max);

    // Build regular grid at z = -depth
    Real z_slice = -depth;
    int nx = static_cast<int>((bb_max(0) - bb_min(0)) / dx) + 1;
    int ny = static_cast<int>((bb_max(1) - bb_min(1)) / dy) + 1;
    int npts = nx * ny;

    mfem::DenseMatrix pts(3, npts);
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int idx = iy * nx + ix;
            pts(0, idx) = bb_min(0) + ix * dx;
            pts(1, idx) = bb_min(1) + iy * dy;
            pts(2, idx) = z_slice;
        }
    }

    // Find points in mesh
    mfem::Array<int> elem_ids;
    mfem::Array<mfem::IntegrationPoint> ips;
    mesh_nc.FindPoints(pts, elem_ids, ips);

    // Extract model values
    const Real ln10 = std::log(10.0);
    std::vector<double> log10_rho(npts, std::numeric_limits<double>::quiet_NaN());

    for (int i = 0; i < npts; ++i) {
        if (elem_ids[i] >= 0 && mesh_nc.GetAttribute(elem_ids[i]) == 1) {
            log10_rho[i] = -model.params()[elem_ids[i]] / ln10;
        }
    }

    // Write VTK StructuredGrid
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open slice file: " + path.string());
    }

    ofs << "# vtk DataFile Version 3.0\n"
        << "Depth slice at " << depth << " m\n"
        << "ASCII\n"
        << "DATASET STRUCTURED_GRID\n"
        << "DIMENSIONS " << nx << " " << ny << " 1\n"
        << "POINTS " << npts << " double\n";

    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            int idx = iy * nx + ix;
            ofs << pts(0, idx) << " " << pts(1, idx) << " " << pts(2, idx) << "\n";
        }
    }

    ofs << "POINT_DATA " << npts << "\n"
        << "SCALARS log10_rho double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int i = 0; i < npts; ++i) {
        ofs << std::setprecision(8)
            << (std::isnan(log10_rho[i]) ? -9999.0 : log10_rho[i]) << "\n";
    }

    ofs.close();
    MAPLE3DMT_LOG_INFO("Exported depth slice at " + std::to_string(depth) +
                   " m: " + path.string() +
                   " (" + std::to_string(nx) + "x" + std::to_string(ny) + ")");
}

// =========================================================================
// export_profile_slice
// =========================================================================
void export_profile_slice(const mfem::ParMesh& mesh,
                          const model::ConductivityModel& model,
                          Real x0, Real y0, Real x1, Real y1,
                          Real z_max, const fs::path& path,
                          int n_along, int n_depth) {
    auto& mesh_nc = const_cast<mfem::ParMesh&>(mesh);

    Real profile_len = std::sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
    if (profile_len < 1e-6) {
        MAPLE3DMT_LOG_WARNING("Profile length too short, skipping export");
        return;
    }

    Real ux = (x1 - x0) / profile_len;
    Real uy = (y1 - y0) / profile_len;

    int npts = n_along * n_depth;
    mfem::DenseMatrix pts(3, npts);

    Real dd = profile_len / std::max(n_along - 1, 1);
    Real dz = z_max / std::max(n_depth - 1, 1);

    for (int iz = 0; iz < n_depth; ++iz) {
        for (int id = 0; id < n_along; ++id) {
            int idx = iz * n_along + id;
            Real dist = id * dd;
            pts(0, idx) = x0 + dist * ux;
            pts(1, idx) = y0 + dist * uy;
            pts(2, idx) = -iz * dz;  // z = -depth (z-up convention)
        }
    }

    // Find points in mesh
    mfem::Array<int> elem_ids;
    mfem::Array<mfem::IntegrationPoint> ips;
    mesh_nc.FindPoints(pts, elem_ids, ips);

    // Extract model values
    const Real ln10 = std::log(10.0);
    std::vector<double> log10_rho(npts, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> distance(npts), depth_arr(npts);

    for (int iz = 0; iz < n_depth; ++iz) {
        for (int id = 0; id < n_along; ++id) {
            int idx = iz * n_along + id;
            distance[idx] = id * dd;
            depth_arr[idx] = iz * dz;
            if (elem_ids[idx] >= 0 && mesh_nc.GetAttribute(elem_ids[idx]) == 1) {
                log10_rho[idx] = -model.params()[elem_ids[idx]] / ln10;
            }
        }
    }

    // Write VTK StructuredGrid
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open profile file: " + path.string());
    }

    ofs << "# vtk DataFile Version 3.0\n"
        << "Profile slice (" << x0 << "," << y0 << ")->("
        << x1 << "," << y1 << ") z_max=" << z_max << "\n"
        << "ASCII\n"
        << "DATASET STRUCTURED_GRID\n"
        << "DIMENSIONS " << n_along << " " << n_depth << " 1\n"
        << "POINTS " << npts << " double\n";

    for (int i = 0; i < npts; ++i) {
        ofs << pts(0, i) << " " << pts(1, i) << " " << pts(2, i) << "\n";
    }

    ofs << "POINT_DATA " << npts << "\n";

    // log10_rho
    ofs << "SCALARS log10_rho double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int i = 0; i < npts; ++i) {
        ofs << std::setprecision(8)
            << (std::isnan(log10_rho[i]) ? -9999.0 : log10_rho[i]) << "\n";
    }

    // distance along profile
    ofs << "SCALARS distance double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int i = 0; i < npts; ++i) {
        ofs << std::setprecision(8) << distance[i] << "\n";
    }

    // depth
    ofs << "SCALARS depth double 1\n"
        << "LOOKUP_TABLE default\n";
    for (int i = 0; i < npts; ++i) {
        ofs << std::setprecision(8) << depth_arr[i] << "\n";
    }

    ofs.close();
    MAPLE3DMT_LOG_INFO("Exported profile slice: " + path.string() +
                   " (" + std::to_string(n_along) + "x" +
                   std::to_string(n_depth) + ", L=" +
                   std::to_string(static_cast<int>(profile_len)) + " m)");
}

// =========================================================================
// export_stations_geojson
// =========================================================================
void export_stations_geojson(const data::MTData& data,
                             const fs::path& path) {
    int ns = data.num_stations();

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open GeoJSON file: " + path.string());
    }

    ofs << "{\n"
        << "  \"type\": \"FeatureCollection\",\n"
        << "  \"features\": [\n";

    for (int i = 0; i < ns; ++i) {
        const auto& s = data.station(i);
        if (i > 0) ofs << ",\n";
        ofs << "    {\n"
            << "      \"type\": \"Feature\",\n"
            << "      \"geometry\": {\n"
            << "        \"type\": \"Point\",\n"
            << "        \"coordinates\": [";
        if (s.has_geo) {
            ofs << std::setprecision(8) << s.lon << ", " << s.lat << ", " << s.z;
        } else {
            ofs << std::setprecision(8) << s.x << ", " << s.y << ", " << s.z;
        }
        ofs << "]\n"
            << "      },\n"
            << "      \"properties\": {\n"
            << "        \"name\": \"" << s.name << "\",\n"
            << "        \"x\": " << std::setprecision(4) << s.x << ",\n"
            << "        \"y\": " << s.y << ",\n"
            << "        \"z\": " << s.z;
        if (s.has_geo) {
            ofs << ",\n        \"lon\": " << std::setprecision(8) << s.lon
                << ",\n        \"lat\": " << s.lat;
        }
        ofs << "\n      }\n"
            << "    }";
    }

    ofs << "\n  ]\n}\n";
    ofs.close();

    MAPLE3DMT_LOG_INFO("Exported stations GeoJSON: " + path.string() +
                   " (" + std::to_string(ns) + " stations)");
}

// =========================================================================
// export_stations_csv
// =========================================================================
void export_stations_csv(const data::MTData& data,
                         const fs::path& path) {
    int ns = data.num_stations();

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + path.string());
    }

    ofs << "name,x,y,z,lon,lat,elevation\n";
    for (int i = 0; i < ns; ++i) {
        const auto& s = data.station(i);
        ofs << s.name << ","
            << std::setprecision(4) << s.x << ","
            << s.y << ","
            << s.z << ","
            << std::setprecision(8) << s.lon << ","
            << s.lat << ","
            << s.z << "\n";
    }

    ofs.close();
    MAPLE3DMT_LOG_INFO("Exported stations CSV: " + path.string() +
                   " (" + std::to_string(ns) + " stations)");
}

// =========================================================================
// export_all
// =========================================================================
void export_all(const mfem::ParMesh& mesh,
                const model::ConductivityModel& model,
                const data::MTData& data,
                const ExportParams& params,
                const fs::path& output_dir,
                int iteration) {
    fs::create_directories(output_dir);

    std::string suffix = "";
    if (iteration >= 0) {
        suffix = "_iter_" + std::to_string(iteration);
    }

    // Volume VTK
    if (params.export_vtk) {
        export_model_vtk(mesh, model, output_dir / ("model" + suffix + ".vtk"),
                         iteration);
    }

    // Parallel VTU
    if (params.export_vtu_parallel) {
        export_model_pvtu(mesh, model, output_dir / "pvtu", iteration);
    }

    // Slices
    if (params.export_slices) {
        fs::path slice_dir = output_dir / "slices";
        fs::create_directories(slice_dir);

        // Auto-generate depth slices
        std::vector<Real> depths = params.slice_depths;
        if (params.auto_slice_interval > 0 && depths.empty()) {
            mfem::Vector bb_min(3), bb_max(3);
            const_cast<mfem::ParMesh&>(mesh).GetBoundingBox(bb_min, bb_max);
            Real max_depth = -bb_min(2);  // z is negative for depth
            for (Real d = params.auto_slice_interval; d <= max_depth;
                 d += params.auto_slice_interval) {
                depths.push_back(d);
            }
        }

        for (Real d : depths) {
            std::string fname = "depth_slice_" + std::to_string(static_cast<int>(d))
                                + "m" + suffix + ".vtk";
            export_depth_slice(mesh, model, d, slice_dir / fname);
        }

        // X-slices (YZ plane)
        for (Real x : params.slice_x) {
            mfem::Vector bb_min(3), bb_max(3);
            const_cast<mfem::ParMesh&>(mesh).GetBoundingBox(bb_min, bb_max);
            Real z_max = -bb_min(2);
            std::string fname = "xslice_" + std::to_string(static_cast<int>(x))
                                + "m" + suffix + ".vtk";
            export_profile_slice(mesh, model,
                                 x, bb_min(1), x, bb_max(1),
                                 z_max, slice_dir / fname);
        }

        // Y-slices (XZ plane)
        for (Real y : params.slice_y) {
            mfem::Vector bb_min(3), bb_max(3);
            const_cast<mfem::ParMesh&>(mesh).GetBoundingBox(bb_min, bb_max);
            Real z_max = -bb_min(2);
            std::string fname = "yslice_" + std::to_string(static_cast<int>(y))
                                + "m" + suffix + ".vtk";
            export_profile_slice(mesh, model,
                                 bb_min(0), y, bb_max(0), y,
                                 z_max, slice_dir / fname);
        }

        // Profile slices
        for (const auto& prof : params.profile_slices) {
            mfem::Vector bb_min(3), bb_max(3);
            const_cast<mfem::ParMesh&>(mesh).GetBoundingBox(bb_min, bb_max);
            Real z_max = -bb_min(2);
            std::string fname = "profile_" + prof.name + suffix + ".vtk";
            export_profile_slice(mesh, model,
                                 prof.x0, prof.y0, prof.x1, prof.y1,
                                 z_max, slice_dir / fname,
                                 prof.n_points);
        }
    }

    // Station exports
    if (params.export_station_geojson) {
        export_stations_geojson(data, output_dir / "stations.geojson");
    }
    if (params.export_station_csv) {
        export_stations_csv(data, output_dir / "stations.csv");
    }

    MAPLE3DMT_LOG_INFO("Export complete: " + output_dir.string());
}

} // namespace io
} // namespace maple3dmt
