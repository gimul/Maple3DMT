// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file commemi_3d1.cpp
/// @brief COMMEMI 3D-1 benchmark: conductive block in a homogeneous halfspace.
///
/// Model: 10 Ωm block (σ=0.1) embedded in 100 Ωm halfspace (σ=0.01).
///   Block: |x| ≤ 5000m, |y| ≤ 5000m, z ∈ [-5000, 0] (surface to 5km depth).
///
/// Reference: Zhdanov et al. (1997), "Methods for modelling electromagnetic
///   fields: results from COMMEMI — the international project on the comparison
///   of modelling methods for electromagnetic induction."
///
/// Output: apparent resistivity and phase at surface stations vs frequency.
/// Validation: compare with published COMMEMI results.

#include "maple3dmt/octree/octree_mesh.h"
#include "maple3dmt/octree/staggered_grid.h"
#include "maple3dmt/octree/operators.h"
#include "maple3dmt/forward/forward_solver_fv.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/io/vtk_export_octree.h"
#include "maple3dmt/utils/logger.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <mpi.h>

using namespace maple3dmt;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    utils::Logger::instance().set_rank(rank);

    std::cout << "\n=== COMMEMI 3D-1: Conductive Block Benchmark ===" << std::endl;

    // ---------------------------------------------------------------
    // 1. Model parameters
    // ---------------------------------------------------------------
    const Real sigma_bg    = 0.01;   // 100 Ωm halfspace
    const Real sigma_block = 0.1;    // 10 Ωm block
    const Real block_xmin  = -5000,  block_xmax = 5000;
    const Real block_ymin  = -5000,  block_ymax = 5000;
    const Real block_zmin  = -5000,  block_zmax = 0;    // surface to 5km

    // Frequencies: 3 decades
    std::vector<Real> frequencies;
    frequencies.push_back(0.01);   // deep — mostly sees halfspace
    frequencies.push_back(0.1);    // intermediate — transition zone
    frequencies.push_back(1.0);    // shallow — block dominates

    // Stations: x-profile (y=0) and y-profile (x=0) with fine spacing near block
    std::vector<std::array<Real,3>> station_xyz;
    std::vector<std::string> station_names;
    std::vector<int> station_profile;  // 0=x-profile, 1=y-profile

    // X-profile: y=0, from -30km to +30km, 1km spacing near block, 2.5km elsewhere
    auto add_x_stations = [&]() {
        std::vector<Real> xpos;
        // Dense near block: -15km to +15km in 1km steps
        for (Real x = -15000; x <= 15001; x += 1000) xpos.push_back(x);
        // Sparse outside: ±17.5, ±20, ±25, ±30 km
        for (Real x : {-30000.0, -25000.0, -20000.0, -17500.0,
                        17500.0,  20000.0,  25000.0,  30000.0}) {
            // Only add if not already present
            bool exists = false;
            for (Real xx : xpos) { if (std::abs(xx - x) < 100) { exists = true; break; } }
            if (!exists) xpos.push_back(x);
        }
        for (Real x : xpos) {
            station_xyz.push_back({x, 0.0, 0.0});
            char name[32]; snprintf(name, sizeof(name), "X%+.0f", x/1000);
            station_names.push_back(name);
            station_profile.push_back(0);
        }
    };

    // Y-profile: x=0, same spacing
    auto add_y_stations = [&]() {
        std::vector<Real> ypos;
        for (Real y = -15000; y <= 15001; y += 1000) {
            if (std::abs(y) < 1.0) continue;  // skip origin (already in x-profile)
            ypos.push_back(y);
        }
        for (Real y : {-30000.0, -25000.0, -20000.0, -17500.0,
                        17500.0,  20000.0,  25000.0,  30000.0}) {
            bool exists = false;
            for (Real yy : ypos) { if (std::abs(yy - y) < 100) { exists = true; break; } }
            if (!exists) ypos.push_back(y);
        }
        for (Real y : ypos) {
            station_xyz.push_back({0.0, y, 0.0});
            char name[32]; snprintf(name, sizeof(name), "Y%+.0f", y/1000);
            station_names.push_back(name);
            station_profile.push_back(1);
        }
    };

    add_x_stations();
    add_y_stations();

    int ns = static_cast<int>(station_xyz.size());
    int nf = static_cast<int>(frequencies.size());

    std::cout << "  Block: [" << block_xmin << "," << block_xmax << "] x ["
              << block_ymin << "," << block_ymax << "] x ["
              << block_zmin << "," << block_zmax << "]" << std::endl;
    std::cout << "  sigma_bg=" << sigma_bg << " sigma_block=" << sigma_block << std::endl;
    std::cout << "  Stations: " << ns << " (x-profile + y-profile)" << std::endl;
    std::cout << "  Frequencies: " << nf << " (" << frequencies.front()
              << " - " << frequencies.back() << " Hz)" << std::endl;

    // ---------------------------------------------------------------
    // 2. Build octree mesh
    // ---------------------------------------------------------------
    octree::RefinementParams mesh_params;
    // Domain: z=0 must align with cell faces at ALL levels (L5-L8).
    // z_range = 256000m so cell_z = 256000/2^N is exact for N=5..8.
    // |z_min|/cell_z = integer at all levels → z=0 is always on a face.
    // At L8: cell_z=1000m, |z_min|/1000=200 ✓
    // At L5: cell_z=8000m, |z_min|/8000=25 ✓
    mesh_params.domain_x_min = -200000;  mesh_params.domain_x_max = 200000;
    mesh_params.domain_y_min = -200000;  mesh_params.domain_y_max = 200000;
    mesh_params.domain_z_min = -200000;  mesh_params.domain_z_max = 56000;

    mesh_params.min_level = 5;
    mesh_params.max_level = 8;
    mesh_params.station_refine_radius = 25000;
    mesh_params.station_refine_level = 7;
    mesh_params.sigma_bg = sigma_bg;

    // Refine around the block for accurate resolution
    // L8: cell = 1562m × 1562m × 1000m → block ≈ 6×6×5 = 180 cells
    octree::RefineRegion block_region;
    block_region.x_min = block_xmin;  block_region.x_max = block_xmax;
    block_region.y_min = block_ymin;  block_region.y_max = block_ymax;
    block_region.z_min = block_zmin;  block_region.z_max = block_zmax;
    block_region.level = 8;
    block_region.padding = 5000;
    mesh_params.refine_regions.push_back(block_region);

    octree::OctreeMesh mesh;
    mesh.setup(mesh_params, station_xyz, {frequencies.front()}, MPI_COMM_WORLD);
    mesh.set_terrain([](Real, Real) { return 0.0; });
    mesh.build_staggered_grid();

    int nc = mesh.staggered().num_cells();
    int ne = mesh.staggered().num_edges();
    std::cout << "\n  Mesh: " << nc << " cells, " << ne << " edge DOFs" << std::endl;

    // ---------------------------------------------------------------
    // 3. Setup conductivity model with block anomaly
    // ---------------------------------------------------------------
    model::ConductivityModel model;
    model.init_3d(nc, sigma_bg);

    auto& log_sigma = model.params();
    int n_block_cells = 0;
    for (int c = 0; c < nc; ++c) {
        if (mesh.cell_type(c) != octree::CellType::EARTH) {
            log_sigma[c] = std::log(1e-8);  // air
            continue;
        }

        Real cx, cy, cz;
        mesh.cell_center(c, cx, cy, cz);

        if (cx >= block_xmin && cx <= block_xmax &&
            cy >= block_ymin && cy <= block_ymax &&
            cz >= block_zmin && cz <= block_zmax) {
            log_sigma[c] = std::log(sigma_block);
            ++n_block_cells;
        }
    }
    model.invalidate_cache();
    std::cout << "  Block cells: " << n_block_cells << std::endl;

    // ---------------------------------------------------------------
    // 4. Setup forward solver
    // ---------------------------------------------------------------
    forward::ForwardParamsFV fwd_params;
    fwd_params.bicgstab_tol = 1e-8;
    fwd_params.bicgstab_maxiter = 10000;
    fwd_params.print_level = 1;
    fwd_params.div_correction = true;
    fwd_params.scattered_field = false;
    // Air Dirichlet: fix only edges above 10km.
    // Near-surface air (0-10km) remains free → E_scattered can develop.
    // The 1D primary field initial guess keeps air edges stable during BiCGStab.
    // Two-pass air iteration: Pass 1 uses E_primary as air BC, Pass 2 updates
    // air BCs with E_scattered from Pass 1 solution.
    fwd_params.air_z_threshold = 1.0;   // restore standard threshold
    fwd_params.air_bc_iterations = 1;   // single pass (standard)

    forward::ForwardSolverFV fwd;
    fwd.setup(mesh, fwd_params);
    fwd.update_sigma(model);

    // ---------------------------------------------------------------
    // 5. Setup MT data
    // ---------------------------------------------------------------
    data::MTData observed;
    for (int s = 0; s < ns; ++s) {
        data::Station st;
        st.name = station_names[s];
        st.x = station_xyz[s][0];
        st.y = station_xyz[s][1];
        st.z = station_xyz[s][2];
        observed.add_station(st);
    }
    observed.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));

    for (int f = 0; f < nf; ++f) {
        Real omega = constants::TWOPI * frequencies[f];
        Complex iwmu(0, omega * constants::MU0);
        Complex Z_1d = std::sqrt(iwmu / Complex(sigma_bg, 0));

        for (int s = 0; s < ns; ++s) {
            data::MTResponse resp;
            resp.Zxy.value = Z_1d;
            resp.Zxy.error = 0.05 * std::abs(Z_1d);
            resp.Zxy.weight = 1.0;
            resp.Zyx.value = -Z_1d;
            resp.Zyx.error = 0.05 * std::abs(Z_1d);
            resp.Zyx.weight = 1.0;
            observed.set_observed(s, f, resp);
        }
    }

    data::MTData predicted;
    for (int s = 0; s < ns; ++s) {
        data::Station st;
        st.name = station_names[s];
        st.x = station_xyz[s][0];
        st.y = station_xyz[s][1];
        st.z = station_xyz[s][2];
        predicted.add_station(st);
    }
    predicted.set_frequencies(RealVec(frequencies.begin(), frequencies.end()));

    // ---------------------------------------------------------------
    // 6. Forward solve
    // ---------------------------------------------------------------
    std::cout << "\n--- Forward solve ---" << std::endl;
    fwd.compute_responses(observed, predicted);

    // ---------------------------------------------------------------
    // 6b. E-field vertical profile diagnostic at center (x≈0, y≈0)
    // ---------------------------------------------------------------
    {
        std::cout << "\n--- E-field vertical profile at center ---" << std::endl;
        const auto& sg = mesh.staggered();
        const auto& E1 = fwd.E1();  // Pol1 (Ex polarization)

        // Find x-directed edges near center
        struct EzEntry { Real z; Complex E; Real x, y; };
        std::vector<EzEntry> profile;
        for (int e = 0; e < static_cast<int>(E1.size()); ++e) {
            const auto& ei = sg.edge(e);
            if (ei.direction != 0) continue;  // x-directed only
            if (std::abs(ei.x) > 1000 || std::abs(ei.y) > 1000) continue;
            profile.push_back({ei.z, E1[e], ei.x, ei.y});
        }
        // Sort by z
        std::sort(profile.begin(), profile.end(),
                  [](const EzEntry& a, const EzEntry& b) { return a.z > b.z; });

        // Also compute 1D analytical for comparison
        Real omega0 = constants::TWOPI * frequencies[0];
        Complex ik_bg = std::sqrt(Complex(0, omega0 * constants::MU0 * sigma_bg));

        std::cout << "  " << std::setw(10) << "z(m)"
                  << "  " << std::setw(20) << "|E_num|"
                  << "  " << std::setw(20) << "|E_1D_bg|"
                  << "  " << std::setw(12) << "ratio"
                  << "  " << std::setw(12) << "E_re"
                  << "  " << std::setw(12) << "E_im"
                  << std::endl;

        for (const auto& ep : profile) {
            Complex E_1d = (ep.z >= 0) ? (Complex(1,0) + ik_bg * ep.z) : std::exp(ik_bg * ep.z);
            Real ratio = (std::abs(E_1d) > 1e-20) ? std::abs(ep.E) / std::abs(E_1d) : 0;
            std::cout << "  " << std::fixed << std::setprecision(1)
                      << std::setw(10) << ep.z
                      << "  " << std::scientific << std::setprecision(6)
                      << std::setw(20) << std::abs(ep.E)
                      << "  " << std::setw(20) << std::abs(E_1d)
                      << "  " << std::fixed << std::setprecision(4)
                      << std::setw(12) << ratio
                      << "  " << std::scientific << std::setprecision(4)
                      << std::setw(12) << ep.E.real()
                      << "  " << std::setw(12) << ep.E.imag()
                      << std::endl;
        }
    }

    // ---------------------------------------------------------------
    // 7. Print results per profile per frequency
    // ---------------------------------------------------------------
    auto print_profile = [&](const char* title, int prof_id, const char* dist_label) {
        std::cout << "\n--- " << title << " ---\n" << std::endl;
        std::cout << "  freq(Hz)  " << dist_label
                  << "(km)  rho_xy(Om)  phi_xy(deg)  rho_yx(Om)  phi_yx(deg)"
                  << std::endl;
        std::cout << "  " << std::string(75, '-') << std::endl;

        for (int f = 0; f < nf; ++f) {
            Real freq = frequencies[f];
            Real omega = constants::TWOPI * freq;

            for (int s = 0; s < ns; ++s) {
                if (station_profile[s] != prof_id) continue;
                Real dist = (prof_id == 0) ? station_xyz[s][0] : station_xyz[s][1];

                const auto& p = predicted.predicted(s, f);
                Complex Zxy = p.Zxy.value, Zyx = p.Zyx.value;
                Real rho_xy = std::norm(Zxy) / (omega * constants::MU0);
                Real phi_xy = std::arg(Zxy) * 180.0 / constants::PI;
                Real rho_yx = std::norm(Zyx) / (omega * constants::MU0);
                Real phi_yx = std::arg(Zyx) * 180.0 / constants::PI;

                std::cout << "  " << std::scientific << std::setprecision(2)
                          << std::setw(9) << freq
                          << "  " << std::fixed << std::setprecision(1)
                          << std::setw(7) << dist/1000.0
                          << "  " << std::setprecision(2)
                          << std::setw(10) << rho_xy
                          << "  " << std::setw(10) << phi_xy
                          << "  " << std::setw(10) << rho_yx
                          << "  " << std::setw(10) << phi_yx
                          << std::endl;
            }
            if (f < nf - 1) std::cout << std::endl;
        }
    };

    // Origin is in x-profile — extract separately
    int s_origin = -1;
    for (int s = 0; s < ns; ++s) {
        if (std::abs(station_xyz[s][0]) < 1 && std::abs(station_xyz[s][1]) < 1) {
            s_origin = s;
            break;
        }
    }

    print_profile("X-profile (y=0)", 0, "x");
    print_profile("Y-profile (x=0)", 1, "y");

    // ---------------------------------------------------------------
    // 8. Summary: key stations for COMMEMI comparison
    // ---------------------------------------------------------------
    std::cout << "\n\n=== COMMEMI 3D-1 Summary (key stations) ===" << std::endl;
    std::cout << "\n  Published range from multiple codes (Zhdanov et al. 1997):" << std::endl;
    std::cout << "    Center (0,0)  1Hz:   rho ~ 10-15 Ωm" << std::endl;
    std::cout << "    Center (0,0)  0.1Hz: rho ~ 20-35 Ωm" << std::endl;
    std::cout << "    Center (0,0)  0.01Hz: rho ~ 55-85 Ωm" << std::endl;
    std::cout << "    x=10km        1Hz:   rho_xy ~ 90-110, rho_yx ~ 85-100 Ωm" << std::endl;
    std::cout << "    x=10km        0.1Hz: rho_xy ~ 120-150, rho_yx ~ 70-85 Ωm" << std::endl;

    // Key positions: origin, block edge, outside, far field
    struct KeyStation { const char* label; Real x; Real y; };
    std::vector<KeyStation> keys = {
        {"center (0,0)",  0,     0},
        {"x=+5km edge",   5000,  0},
        {"x=+10km",      10000,  0},
        {"x=+20km",      20000,  0},
        {"y=+5km edge",   0,  5000},
        {"y=+10km",       0, 10000},
        {"y=+20km",       0, 20000},
    };

    std::cout << "\n  " << std::setw(16) << "station"
              << "  " << std::setw(8) << "freq"
              << "  " << std::setw(10) << "rho_xy"
              << "  " << std::setw(10) << "phi_xy"
              << "  " << std::setw(10) << "rho_yx"
              << "  " << std::setw(10) << "phi_yx" << std::endl;
    std::cout << "  " << std::string(72, '-') << std::endl;

    for (int f = 0; f < nf; ++f) {
        Real freq = frequencies[f];
        Real omega = constants::TWOPI * freq;

        for (const auto& ks : keys) {
            // Find closest station
            int best_s = -1;
            Real best_dist = 1e30;
            for (int s = 0; s < ns; ++s) {
                Real d = std::hypot(station_xyz[s][0] - ks.x, station_xyz[s][1] - ks.y);
                if (d < best_dist) { best_dist = d; best_s = s; }
            }
            if (best_s < 0 || best_dist > 500) continue;

            const auto& p = predicted.predicted(best_s, f);
            Complex Zxy = p.Zxy.value, Zyx = p.Zyx.value;
            Real rho_xy = std::norm(Zxy) / (omega * constants::MU0);
            Real phi_xy = std::arg(Zxy) * 180.0 / constants::PI;
            Real rho_yx = std::norm(Zyx) / (omega * constants::MU0);
            Real phi_yx = std::arg(Zyx) * 180.0 / constants::PI;

            char fbuf[16];
            snprintf(fbuf, sizeof(fbuf), "%.2g Hz", freq);
            std::cout << "  " << std::setw(16) << ks.label
                      << "  " << std::setw(8) << fbuf
                      << "  " << std::fixed << std::setprecision(1)
                      << std::setw(10) << rho_xy
                      << "  " << std::setw(10) << phi_xy
                      << "  " << std::setw(10) << rho_yx
                      << "  " << std::setw(10) << phi_yx << std::endl;
        }
        if (f < nf - 1) std::cout << std::endl;
    }

    // ---------------------------------------------------------------
    // 9. Export CSV for plotting
    // ---------------------------------------------------------------
    {
        FILE* fp = fopen("commemi3d1_xprofile.csv", "w");
        if (fp) {
            fprintf(fp, "freq_hz,x_km,rho_xy,phi_xy,rho_yx,phi_yx\n");
            for (int f = 0; f < nf; ++f) {
                Real freq = frequencies[f];
                Real omega = constants::TWOPI * freq;
                for (int s = 0; s < ns; ++s) {
                    if (station_profile[s] != 0) continue;
                    const auto& p = predicted.predicted(s, f);
                    Complex Zxy = p.Zxy.value, Zyx = p.Zyx.value;
                    Real rho_xy = std::norm(Zxy) / (omega * constants::MU0);
                    Real phi_xy = std::arg(Zxy) * 180.0 / constants::PI;
                    Real rho_yx = std::norm(Zyx) / (omega * constants::MU0);
                    Real phi_yx = std::arg(Zyx) * 180.0 / constants::PI;
                    fprintf(fp, "%.4e,%.3f,%.4f,%.4f,%.4f,%.4f\n",
                            freq, station_xyz[s][0]/1000.0,
                            rho_xy, phi_xy, rho_yx, phi_yx);
                }
            }
            fclose(fp);
            std::cout << "\n  Exported: commemi3d1_xprofile.csv" << std::endl;
        }
    }
    {
        FILE* fp = fopen("commemi3d1_yprofile.csv", "w");
        if (fp) {
            fprintf(fp, "freq_hz,y_km,rho_xy,phi_xy,rho_yx,phi_yx\n");
            for (int f = 0; f < nf; ++f) {
                Real freq = frequencies[f];
                Real omega = constants::TWOPI * freq;
                // Include origin for y-profile
                for (int s = 0; s < ns; ++s) {
                    if (station_profile[s] != 1 && s != s_origin) continue;
                    Real y = station_xyz[s][1];
                    const auto& p = predicted.predicted(s, f);
                    Complex Zxy = p.Zxy.value, Zyx = p.Zyx.value;
                    Real rho_xy = std::norm(Zxy) / (omega * constants::MU0);
                    Real phi_xy = std::arg(Zxy) * 180.0 / constants::PI;
                    Real rho_yx = std::norm(Zyx) / (omega * constants::MU0);
                    Real phi_yx = std::arg(Zyx) * 180.0 / constants::PI;
                    fprintf(fp, "%.4e,%.3f,%.4f,%.4f,%.4f,%.4f\n",
                            freq, y/1000.0,
                            rho_xy, phi_xy, rho_yx, phi_yx);
                }
            }
            fclose(fp);
            std::cout << "  Exported: commemi3d1_yprofile.csv" << std::endl;
        }
    }

    std::cout << "\n=== COMMEMI 3D-1 Complete ===" << std::endl;

    MPI_Finalize();
    return 0;
}
