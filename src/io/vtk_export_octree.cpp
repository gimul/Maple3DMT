// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

#include "maple3dmt/io/vtk_export_octree.h"

#ifdef MAPLE3DMT_USE_OCTREE

#include "maple3dmt/utils/logger.h"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>

#define LOG_INFO(msg) MAPLE3DMT_LOG_INFO(msg)

namespace maple3dmt {
namespace io {

// =========================================================================
// Helper: ensure directory exists
// =========================================================================
static void ensure_dir(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) != 0) {
        mkdir(dir.c_str(), 0755);
    }
}

// =========================================================================
// Export octree as VTU (VTK XML UnstructuredGrid)
// =========================================================================
void export_octree_vtu(const octree::OctreeMesh& mesh,
                       const model::ConductivityModel& model,
                       const std::string& path,
                       int iteration,
                       const std::map<std::string, RealVec>& extra_scalars) {
    int nc = mesh.num_cells_local();

    LOG_INFO("VTK export: " + std::to_string(nc) + " cells → " + path);

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        LOG_INFO("ERROR: Cannot open " + path);
        return;
    }

    // Each hex cell has 8 vertices. For octree cells we create independent
    // vertices per cell (no sharing) for simplicity.
    int n_pts = nc * 8;

    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    ofs << "<UnstructuredGrid>\n";
    ofs << "<Piece NumberOfPoints=\"" << n_pts << "\" NumberOfCells=\"" << nc << "\">\n";

    // ---- Points ----
    ofs << "<Points>\n";
    ofs << "  <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    for (int c = 0; c < nc; ++c) {
        Real cx, cy, cz;
        mesh.cell_center(c, cx, cy, cz);
        Real h = mesh.cell_size(c) * 0.5;

        // 8 corners of hex cell (VTK ordering)
        // VTK_HEXAHEDRON: bottom face (0,1,2,3) CCW from below, top face (4,5,6,7)
        Real corners[8][3] = {
            {cx - h, cy - h, cz - h},  // 0: bottom-left-front
            {cx + h, cy - h, cz - h},  // 1: bottom-right-front
            {cx + h, cy + h, cz - h},  // 2: bottom-right-back
            {cx - h, cy + h, cz - h},  // 3: bottom-left-back
            {cx - h, cy - h, cz + h},  // 4: top-left-front
            {cx + h, cy - h, cz + h},  // 5: top-right-front
            {cx + h, cy + h, cz + h},  // 6: top-right-back
            {cx - h, cy + h, cz + h},  // 7: top-left-back
        };

        for (int v = 0; v < 8; ++v)
            ofs << "    " << std::scientific << std::setprecision(6)
                << corners[v][0] << " " << corners[v][1] << " " << corners[v][2] << "\n";
    }

    ofs << "  </DataArray>\n";
    ofs << "</Points>\n";

    // ---- Cells ----
    ofs << "<Cells>\n";

    // Connectivity
    ofs << "  <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int c = 0; c < nc; ++c) {
        int base = c * 8;
        ofs << "    ";
        for (int v = 0; v < 8; ++v)
            ofs << (base + v) << " ";
        ofs << "\n";
    }
    ofs << "  </DataArray>\n";

    // Offsets
    ofs << "  <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    ofs << "    ";
    for (int c = 0; c < nc; ++c)
        ofs << (c + 1) * 8 << " ";
    ofs << "\n";
    ofs << "  </DataArray>\n";

    // Types (VTK_HEXAHEDRON = 12)
    ofs << "  <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    ofs << "    ";
    for (int c = 0; c < nc; ++c)
        ofs << "12 ";
    ofs << "\n";
    ofs << "  </DataArray>\n";

    ofs << "</Cells>\n";

    // ---- Cell Data ----
    ofs << "<CellData>\n";

    // Log10(resistivity)
    ofs << "  <DataArray type=\"Float64\" Name=\"log10_resistivity\" format=\"ascii\">\n";
    ofs << "    ";
    for (int c = 0; c < nc; ++c) {
        Real sigma = model.sigma(c);
        Real rho = (sigma > 1e-30) ? 1.0 / sigma : 1e30;
        Real log10_rho = (mesh.cell_type(c) == octree::CellType::AIR) ? -9999.0
                         : std::log10(rho);
        ofs << std::scientific << std::setprecision(6) << log10_rho << " ";
    }
    ofs << "\n  </DataArray>\n";

    // Conductivity
    ofs << "  <DataArray type=\"Float64\" Name=\"conductivity\" format=\"ascii\">\n";
    ofs << "    ";
    for (int c = 0; c < nc; ++c)
        ofs << std::scientific << std::setprecision(6) << model.sigma(c) << " ";
    ofs << "\n  </DataArray>\n";

    // Cell type (0=Earth, 1=Air, 2=Ocean)
    ofs << "  <DataArray type=\"Int32\" Name=\"cell_type\" format=\"ascii\">\n";
    ofs << "    ";
    for (int c = 0; c < nc; ++c)
        ofs << static_cast<int>(mesh.cell_type(c)) << " ";
    ofs << "\n  </DataArray>\n";

    // Refinement level
    ofs << "  <DataArray type=\"Int32\" Name=\"level\" format=\"ascii\">\n";
    ofs << "    ";
    for (int c = 0; c < nc; ++c)
        ofs << mesh.cell_level(c) << " ";
    ofs << "\n  </DataArray>\n";

    // Cell size
    ofs << "  <DataArray type=\"Float64\" Name=\"cell_size\" format=\"ascii\">\n";
    ofs << "    ";
    for (int c = 0; c < nc; ++c)
        ofs << std::scientific << std::setprecision(3) << mesh.cell_size(c) << " ";
    ofs << "\n  </DataArray>\n";

    // Iteration number (FIELD data)
    if (iteration >= 0) {
        ofs << "  <DataArray type=\"Int32\" Name=\"iteration\" format=\"ascii\">\n";
        ofs << "    ";
        for (int c = 0; c < nc; ++c)
            ofs << iteration << " ";
        ofs << "\n  </DataArray>\n";
    }

    // Extra scalars (e.g., sensitivity, gradient)
    for (const auto& [name, data] : extra_scalars) {
        if (static_cast<int>(data.size()) != nc) continue;
        ofs << "  <DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";
        ofs << "    ";
        for (int c = 0; c < nc; ++c)
            ofs << std::scientific << std::setprecision(6) << data[c] << " ";
        ofs << "\n  </DataArray>\n";
    }

    ofs << "</CellData>\n";

    ofs << "</Piece>\n";
    ofs << "</UnstructuredGrid>\n";
    ofs << "</VTKFile>\n";

    ofs.close();
    LOG_INFO("  VTU written: " + path);
}

// =========================================================================
// Export depth slice
// =========================================================================
void export_octree_depth_slice(const octree::OctreeMesh& mesh,
                               const model::ConductivityModel& model,
                               Real depth,
                               const std::string& path,
                               Real dx, Real dy) {
    // Domain bounds from mesh params
    const auto& p = mesh.params();
    Real xmin = p.domain_x_min, xmax = p.domain_x_max;
    Real ymin = p.domain_y_min, ymax = p.domain_y_max;
    Real z_target = -depth;  // depth is positive downward, z is positive up

    int nx = std::max(2, static_cast<int>((xmax - xmin) / dx));
    int ny = std::max(2, static_cast<int>((ymax - ymin) / dy));

    LOG_INFO("Depth slice at " + std::to_string(depth) + "m: " +
             std::to_string(nx) + "x" + std::to_string(ny) + " → " + path);

    // Sample conductivity at each grid point by finding the containing cell
    int nc = mesh.num_cells_local();
    std::vector<Real> rho_grid(nx * ny, -9999.0);

    for (int iy = 0; iy < ny; ++iy) {
        Real y = ymin + (iy + 0.5) * dy;
        for (int ix = 0; ix < nx; ++ix) {
            Real x = xmin + (ix + 0.5) * dx;

            // Find containing cell (brute force — acceptable for visualization)
            for (int c = 0; c < nc; ++c) {
                Real cx, cy, cz;
                mesh.cell_center(c, cx, cy, cz);
                Real h = mesh.cell_size(c) * 0.5;

                if (x >= cx - h && x < cx + h &&
                    y >= cy - h && y < cy + h &&
                    z_target >= cz - h && z_target < cz + h) {
                    if (mesh.cell_type(c) == octree::CellType::EARTH) {
                        Real sigma = model.sigma(c);
                        rho_grid[iy * nx + ix] = (sigma > 1e-30) ? std::log10(1.0 / sigma) : 6.0;
                    }
                    break;
                }
            }
        }
    }

    // Write as VTK StructuredGrid
    std::ofstream ofs(path);
    if (!ofs.is_open()) return;

    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    ofs << "<StructuredGrid WholeExtent=\"0 " << nx << " 0 " << ny << " 0 0\">\n";
    ofs << "<Piece Extent=\"0 " << nx << " 0 " << ny << " 0 0\">\n";

    // Points (nx+1 × ny+1 × 1)
    ofs << "<Points>\n";
    ofs << "  <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int iy = 0; iy <= ny; ++iy) {
        Real y = ymin + iy * dy;
        for (int ix = 0; ix <= nx; ++ix) {
            Real x = xmin + ix * dx;
            ofs << "    " << x << " " << y << " " << z_target << "\n";
        }
    }
    ofs << "  </DataArray>\n";
    ofs << "</Points>\n";

    // Cell data
    ofs << "<CellData>\n";
    ofs << "  <DataArray type=\"Float64\" Name=\"log10_resistivity\" format=\"ascii\">\n";
    ofs << "    ";
    for (int i = 0; i < nx * ny; ++i)
        ofs << std::fixed << std::setprecision(4) << rho_grid[i] << " ";
    ofs << "\n  </DataArray>\n";

    ofs << "  <DataArray type=\"Float64\" Name=\"depth\" format=\"ascii\">\n";
    ofs << "    ";
    for (int i = 0; i < nx * ny; ++i)
        ofs << std::fixed << std::setprecision(1) << depth << " ";
    ofs << "\n  </DataArray>\n";

    ofs << "</CellData>\n";

    ofs << "</Piece>\n";
    ofs << "</StructuredGrid>\n";
    ofs << "</VTKFile>\n";

    ofs.close();
    LOG_INFO("  Depth slice written: " + path);
}

// =========================================================================
// Export stations as CSV
// =========================================================================
void export_stations_csv(const data::MTData& data,
                         const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return;

    ofs << "station_id,x,y,z,name\n";
    for (int s = 0; s < data.num_stations(); ++s) {
        const auto& st = data.station(s);
        ofs << s << ","
            << std::fixed << std::setprecision(2)
            << st.x << "," << st.y << "," << st.z << ","
            << st.name << "\n";
    }
    ofs.close();
    LOG_INFO("Stations CSV: " + std::to_string(data.num_stations()) + " stations → " + path);
}

// =========================================================================
// Export data fit CSV (obs vs pred)
// =========================================================================
void export_data_fit_csv(const data::MTData& data,
                         const std::string& path,
                         int iteration) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        LOG_INFO("ERROR: Cannot open " + path);
        return;
    }

    ofs << "station,station_name,freq_hz,period_s,component,"
        << "obs_re,obs_im,pred_re,pred_im,error,"
        << "app_res_obs,phase_obs_deg,app_res_pred,phase_pred_deg,misfit\n";

    int ns = data.num_stations();
    int nf = data.num_frequencies();
    int n_written = 0;

    for (int s = 0; s < ns; ++s) {
        const auto& st = data.station(s);
        for (int f = 0; f < nf; ++f) {
            Real freq = data.frequencies()[f];
            Real period = 1.0 / freq;

            const auto& obs = data.observed(s, f);
            const auto& pred = data.predicted(s, f);

            // Helper: write one Z-component row
            auto write_comp = [&](const char* comp,
                                  const data::Datum& o,
                                  const data::Datum& p) {
                if (o.weight <= 0.0) return;

                // Apparent resistivity: ρ_a = |Z|² / (ωμ₀)
                // Z is in SI [Ω] (EDI reader already converted mV/km/nT → Ω)
                Real omega = 2.0 * M_PI * freq;
                Real rho_obs  = std::norm(o.value) / (omega * 4.0e-7 * M_PI);
                Real rho_pred = std::norm(p.value) / (omega * 4.0e-7 * M_PI);
                Real phase_obs  = std::atan2(o.value.imag(), o.value.real()) * 180.0 / M_PI;
                Real phase_pred = std::atan2(p.value.imag(), p.value.real()) * 180.0 / M_PI;

                // Normalized misfit per component
                Real w = (o.error > 0) ? 1.0 / o.error : 0.0;
                Complex r = o.value - p.value;
                Real misfit = w * std::abs(r);

                ofs << s << "," << st.name << ","
                    << std::scientific << std::setprecision(6) << freq << ","
                    << std::fixed << std::setprecision(4) << period << ","
                    << comp << ","
                    << std::scientific << std::setprecision(6)
                    << o.value.real() << "," << o.value.imag() << ","
                    << p.value.real() << "," << p.value.imag() << ","
                    << o.error << ","
                    << std::fixed << std::setprecision(4)
                    << rho_obs << "," << phase_obs << ","
                    << rho_pred << "," << phase_pred << ","
                    << std::setprecision(4) << misfit << "\n";
                ++n_written;
            };

            write_comp("Zxx", obs.Zxx, pred.Zxx);
            write_comp("Zxy", obs.Zxy, pred.Zxy);
            write_comp("Zyx", obs.Zyx, pred.Zyx);
            write_comp("Zyy", obs.Zyy, pred.Zyy);
        }
    }

    ofs.close();
    LOG_INFO("Data fit CSV: " + std::to_string(n_written) + " rows → " + path);
}

// =========================================================================
// Export all
// =========================================================================
void export_octree_all(const octree::OctreeMesh& mesh,
                       const model::ConductivityModel& model,
                       const data::MTData& data,
                       const std::string& output_dir,
                       const OctreeExportParams& params,
                       int iteration) {
    ensure_dir(output_dir);

    std::string iter_suffix = (iteration >= 0)
        ? "_iter" + std::to_string(iteration) : "";

    // 3D VTU
    if (params.export_vtu) {
        export_octree_vtu(mesh, model,
                          output_dir + "/model" + iter_suffix + ".vtu",
                          iteration);
    }

    // Depth slices
    if (params.export_depth_slices) {
        std::vector<Real> depths = params.slice_depths;

        // Auto-generate if list is empty
        if (depths.empty() && params.auto_slice_interval > 0) {
            for (Real d = params.auto_slice_interval;
                 d <= params.auto_slice_max_depth;
                 d += params.auto_slice_interval)
                depths.push_back(d);
        }

        for (Real d : depths) {
            char buf[64];
            snprintf(buf, sizeof(buf), "/slice_z%06.0fm%s.vts",
                     d, iter_suffix.c_str());
            export_octree_depth_slice(mesh, model, d,
                                      output_dir + buf,
                                      params.slice_dx, params.slice_dy);
        }
    }

    // Stations
    if (params.export_stations_csv) {
        export_stations_csv(data, output_dir + "/stations.csv");
    }
}

// =========================================================================
// Resume support: load conductivity from VTU
// =========================================================================
RealVec load_conductivity_from_vtu(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open VTU file: " + path);
    }

    RealVec sigma;
    std::string line;
    bool in_conductivity = false;

    while (std::getline(ifs, line)) {
        // Look for: <DataArray ... Name="conductivity" ...>
        if (line.find("Name=\"conductivity\"") != std::string::npos) {
            in_conductivity = true;
            continue;
        }
        if (in_conductivity) {
            // Skip closing tag
            if (line.find("</DataArray>") != std::string::npos) {
                break;
            }
            // Parse space-separated doubles
            std::istringstream iss(line);
            double val;
            while (iss >> val) {
                sigma.push_back(val);
            }
        }
    }

    LOG_INFO("Loaded conductivity from VTU: " + std::to_string(sigma.size()) +
             " cells from " + path);
    return sigma;
}

// =========================================================================
// Resume support: save/load inversion state (JSON)
// =========================================================================
void save_inversion_state(const std::string& path,
                          int iteration,
                          Real lambda,
                          Real rms,
                          const std::vector<std::pair<int, Real>>& rms_history) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return;

    ofs << "{\n";
    ofs << "  \"last_iteration\": " << iteration << ",\n";
    ofs << "  \"lambda\": " << std::scientific << std::setprecision(8) << lambda << ",\n";
    ofs << "  \"rms\": " << std::fixed << std::setprecision(6) << rms << ",\n";
    ofs << "  \"rms_history\": [";
    for (size_t i = 0; i < rms_history.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << "[" << rms_history[i].first << ", "
            << std::fixed << std::setprecision(6) << rms_history[i].second << "]";
    }
    ofs << "]\n";
    ofs << "}\n";
}

bool load_inversion_state(const std::string& path, InversionState& state) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;

    // Simple JSON parser for our known format
    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());

    // Parse last_iteration
    auto parse_int = [&](const std::string& key) -> int {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = content.find(':', pos);
        return std::stoi(content.substr(pos + 1));
    };
    auto parse_double = [&](const std::string& key) -> double {
        auto pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0.0;
        pos = content.find(':', pos);
        return std::stod(content.substr(pos + 1));
    };

    state.last_iteration = parse_int("last_iteration");
    state.lambda = parse_double("lambda");
    state.rms = parse_double("rms");

    // Parse rms_history: [[iter, rms], ...]
    state.rms_history.clear();
    auto hist_pos = content.find("\"rms_history\"");
    if (hist_pos != std::string::npos) {
        auto arr_start = content.find('[', hist_pos + 14);
        auto arr_end = content.find(']', arr_start);
        // Find inner arrays
        size_t pos = arr_start + 1;
        while (pos < arr_end) {
            auto inner_start = content.find('[', pos);
            if (inner_start == std::string::npos || inner_start >= arr_end) break;
            auto inner_end = content.find(']', inner_start);
            auto comma = content.find(',', inner_start + 1);
            if (comma < inner_end) {
                int it = std::stoi(content.substr(inner_start + 1));
                double r = std::stod(content.substr(comma + 1));
                state.rms_history.push_back({it, r});
            }
            pos = inner_end + 1;
        }
    }

    LOG_INFO("Loaded inversion state: iter=" + std::to_string(state.last_iteration) +
             " lambda=" + std::to_string(state.lambda) +
             " rms=" + std::to_string(state.rms));
    return true;
}

} // namespace io
} // namespace maple3dmt

#endif // MAPLE3DMT_USE_OCTREE
