// Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
// Licensed under GPL-3.0. See LICENSE for details.

/// @file forward_test.cpp
/// @brief Validation driver: uniform halfspace forward → compare with analytic.
///
/// Usage:
///   mpirun -np <N> forward_test [--sigma <S/m>] [--freq <Hz>] [--output <dir>]
///
/// For a uniform halfspace with conductivity sigma:
///   Zxy = Zyx = sqrt(omega*mu0/sigma) * exp(i*pi/4)  [in V/m per A/m]
///   Zxx = Zyy = 0

#include "maple3dmt/forward/forward_solver_3d.h"
#include "maple3dmt/mesh/hex_mesh_3d.h"
#include "maple3dmt/model/conductivity_model.h"
#include "maple3dmt/data/mt_data.h"
#include "maple3dmt/utils/logger.h"
#include <iostream>
#include <cmath>
#include <mfem.hpp>

using namespace maple3dmt;

namespace {

/// Analytic impedance for uniform halfspace.
Complex analytic_impedance(Real omega, Real sigma) {
    // Z = sqrt(omega * mu0 / sigma) * exp(i*pi/4)
    //   = sqrt(omega * mu0 / sigma) * (1 + i) / sqrt(2)
    Real mag = std::sqrt(omega * constants::MU0 / sigma);
    return Complex(mag / std::sqrt(2.0), mag / std::sqrt(2.0));
}

void print_usage() {
    std::cout << "Usage:\n"
              << "  mpirun -np <N> forward_test [options]\n"
              << "\nOptions:\n"
              << "  --sigma <val>   Background conductivity (S/m, default: 0.01)\n"
              << "  --freq <val>    Test frequency (Hz, default: 1.0)\n"
              << "  --order <val>   FE order (default: 1)\n"
              << "  --output <dir>  Output directory (default: forward_output)\n"
              << std::endl;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    // Initialize MPI
    mfem::Mpi::Init(argc, argv);
    int rank = mfem::Mpi::WorldRank();

    // Parse arguments
    Real sigma_bg = 0.01;
    Real test_freq = 1.0;
    int fe_order = 1;
    std::string output_dir = "forward_output";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sigma" && i + 1 < argc) {
            sigma_bg = std::stod(argv[++i]);
        } else if (arg == "--freq" && i + 1 < argc) {
            test_freq = std::stod(argv[++i]);
        } else if (arg == "--order" && i + 1 < argc) {
            fe_order = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            if (rank == 0) print_usage();
            mfem::Mpi::Finalize();
            return 0;
        }
    }

    if (rank == 0) {
        std::cout << "\n=== ForwardSolver3D Validation ===\n"
                  << "  sigma_bg = " << sigma_bg << " S/m\n"
                  << "  freq     = " << test_freq << " Hz\n"
                  << "  order    = " << fe_order << "\n"
                  << std::endl;
    }

    // ------------------------------------------------------------------
    // Create a small test mesh with stations
    // ------------------------------------------------------------------
    std::vector<mesh::Station3D> stations;
    // 3 stations along x-axis at z=0
    for (int i = 0; i < 3; ++i) {
        mesh::Station3D s;
        s.name = "ST" + std::to_string(i + 1);
        s.x = (i - 1) * 1000.0;  // -1000, 0, +1000
        s.y = 0.0;
        s.z = 0.0;
        s.lon = 128.5;
        s.lat = 35.5;
        s.elevation = 0.0;
        stations.push_back(std::move(s));
    }

    auto params = mesh::auto_mesh_params(stations);
    params.use_terrain = false;
    params.refine_near_stations = 0;  // keep it small for testing

    if (rank == 0) {
        std::cout << "--- Mesh Parameters ---\n"
                  << "  Domain X: [" << params.x_min/1000 << ", " << params.x_max/1000 << "] km\n"
                  << "  Domain Y: [" << params.y_min/1000 << ", " << params.y_max/1000 << "] km\n"
                  << "  Domain Z: [" << params.z_min/1000 << ", +" << params.z_air/1000 << "] km\n"
                  << std::endl;
    }

    // Generate serial mesh, then distribute
    mesh::HexMeshGenerator3D generator;
    auto serial_mesh = generator.generate(params, stations, nullptr);

    if (rank == 0) {
        std::cout << "  Elements: " << serial_mesh->GetNE()
                  << "  Vertices: " << serial_mesh->GetNV() << "\n"
                  << std::endl;
    }

    // Create parallel mesh
    mfem::ParMesh pmesh(MPI_COMM_WORLD, *serial_mesh);
    serial_mesh.reset();  // free serial mesh memory

    // ------------------------------------------------------------------
    // Create uniform conductivity model
    // ------------------------------------------------------------------
    model::ConductivityModel model;
    model.init_3d(pmesh.GetNE(), sigma_bg);

    // ------------------------------------------------------------------
    // Set up forward solver
    // ------------------------------------------------------------------
    forward::ForwardParams3D fwd_params;
    fwd_params.fe_order = fe_order;
    fwd_params.backend = forward::SolverBackend::ITERATIVE;

    forward::ForwardSolver3D solver;
    solver.setup(pmesh, model, fwd_params);

    // ------------------------------------------------------------------
    // Create MTData with test frequency and stations
    // ------------------------------------------------------------------
    data::MTData observed, predicted;
    for (const auto& s : stations) {
        data::Station ds;
        ds.name = s.name;
        ds.x = s.x;
        ds.y = s.y;
        ds.z = s.z;
        ds.lon = s.lon;
        ds.lat = s.lat;
        ds.has_geo = true;
        observed.add_station(ds);
        predicted.add_station(ds);
    }
    observed.set_frequencies({test_freq});
    predicted.set_frequencies({test_freq});

    // ------------------------------------------------------------------
    // Run forward solver
    // ------------------------------------------------------------------
    solver.compute_single_frequency(0, observed, predicted);

    // ------------------------------------------------------------------
    // Compare with analytic solution
    // ------------------------------------------------------------------
    Real omega = constants::TWOPI * test_freq;
    Complex Z_analytic = analytic_impedance(omega, sigma_bg);

    if (rank == 0) {
        std::cout << "\n=== Results ===\n"
                  << "Analytic Zxy = (" << Z_analytic.real()
                  << ", " << Z_analytic.imag() << ")\n"
                  << "  |Z| = " << std::abs(Z_analytic)
                  << "  phase = " << std::arg(Z_analytic) * 180.0 / constants::PI << " deg\n"
                  << "  app.rho = " << std::norm(Z_analytic) / (omega * constants::MU0)
                  << " Ohm.m (should be " << 1.0/sigma_bg << ")\n\n";

        for (int s = 0; s < observed.num_stations(); ++s) {
            const auto& resp = predicted.predicted(s, 0);
            Complex Zxy = resp.Zxy.value;
            Complex Zyx = resp.Zyx.value;
            Complex Zxx = resp.Zxx.value;
            Complex Zyy = resp.Zyy.value;

            Real err_xy = std::abs(Zxy - Z_analytic) / std::abs(Z_analytic) * 100.0;
            Real err_yx = std::abs(Zyx + Z_analytic) / std::abs(Z_analytic) * 100.0;  // Zyx = -Z0

            std::cout << "Station " << stations[s].name << ":\n"
                      << "  Zxx = (" << Zxx.real() << ", " << Zxx.imag() << ")\n"
                      << "  Zxy = (" << Zxy.real() << ", " << Zxy.imag()
                      << ")  error=" << err_xy << "%\n"
                      << "  Zyx = (" << Zyx.real() << ", " << Zyx.imag()
                      << ")  error=" << err_yx << "%\n"
                      << "  Zyy = (" << Zyy.real() << ", " << Zyy.imag() << ")\n"
                      << std::endl;
        }
    }

    solver.release_factorization();
    mfem::Mpi::Finalize();
    return 0;
}
