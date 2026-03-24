# Maple3DMT

**Adaptive Octree Finite-Volume Code for Memory-Efficient 3-D Magnetotelluric Inversion**

Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.

Maple3DMT is a C++ code for 3-D magnetotelluric (MT) forward modelling and inversion based on an adaptive octree mesh with mimetic finite-volume discretization. It combines the proven Yee staggered-grid formulation of ModEM with a p4est-managed octree mesh, reducing degrees of freedom by 3.7--8x compared to equivalent structured grids.

## Features

- **Adaptive octree mesh** (p4est) with two-pass AMR: station-centric + skin-depth refinement
- **Mimetic finite-volume** (Yee staggered grid) discretization, formulation-equivalent to ModEM
- **Hanging face/edge treatment**: dead-face elimination preserving the discrete de Rham identity
- **SSOR-preconditioned BiCGStab** solver with divergence correction
- **NLCG, L-BFGS, GN-CG** inversion algorithms with Occam-style regularization
- **Frequency-parallel MPI** with near-ideal linear scaling
- **Terrain support** via ALOS DEM (refined staircase approximation)
- **EDI file I/O**: standard MT data exchange format
- **VTK/VTU export** for ParaView visualization

## Prerequisites

| Dependency | Required | Notes |
|------------|----------|-------|
| C++17 compiler | Yes | GCC >= 9, Clang >= 12, or Apple Clang |
| CMake >= 3.20 | Yes | |
| MPI | Yes | OpenMPI or MPICH |
| [p4est](https://www.p4est.org/) >= 2.8 | Yes | Octree mesh management |
| [yaml-cpp](https://github.com/jbeder/yaml-cpp) | Yes | Configuration file parsing |
| HDF5 | Optional | HDF5 data I/O |
| GDAL | Optional | GeoTIFF DEM loading |
| MUMPS | Optional | Legacy FEM direct solver |

### Installing Dependencies (macOS with Homebrew)

```bash
brew install open-mpi cmake yaml-cpp hdf5
# p4est: build from source
git clone https://github.com/cburstedde/p4est.git
cd p4est && git submodule init && git submodule update
./configure --enable-mpi --prefix=/usr/local
make -j8 && make install
```

### Installing Dependencies (Ubuntu/Debian)

```bash
sudo apt install cmake libopenmpi-dev libyaml-cpp-dev libhdf5-dev
# p4est: build from source (same as above)
```

## Build

```bash
git clone https://github.com/gimul/Maple3DMT.git
cd Maple3DMT

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DMAPLE3DMT_USE_MPI=ON
cmake --build build -j8
```

Build output (in `build/`):
- `real_inversion_octree` — Main inversion driver (reads EDI files)
- `generate_commemi_3d1a_data` — COMMEMI 3D-1A synthetic data generator
- `generate_commemi_3d2_data` — COMMEMI 3D-2 synthetic data generator
- Various test drivers (`test_forward_fv`, `halfspace_1d_test`, etc.)

## Quick Start: COMMEMI Benchmark

### COMMEMI 3D-1A (conductive block in halfspace)

```bash
cd examples/commemi_3d1a
bash run.sh
```

This will:
1. Generate 121 synthetic EDI files (forward solve + 5% Gaussian noise)
2. Run 30 NLCG iterations from a uniform 100 Ohm-m halfspace
3. Export per-iteration VTU models and convergence CSV to `output/`

### COMMEMI 3D-2 (two blocks in layered earth)

```bash
cd examples/commemi_3d2
bash run.sh
```

See each example's `README.md` for model details and expected results.

## Running Inversion with Your Own Data

### Basic Usage

```bash
mpiexec -np 12 ./build/real_inversion_octree \
    --edi-dir /path/to/edi/files \
    --output-dir output \
    --sigma 0.01 \
    --niter 50
```

### Full Command-Line Options

```
Usage:
  mpirun -np <N> real_inversion_octree --edi-dir <path> [options]

Required:
  --edi-dir <path>          Directory containing EDI files

I/O:
  --output-dir <path>       Output directory (default: output_octree)
  --dem <path>              DEM file (ASCII xyz or GeoTIFF) for terrain
  --bathymetry <path>       Bathymetry DEM for coastal surveys
  --resume                  Resume from last checkpoint

Mesh:
  --domain-size <km>        Domain half-width in km (default: auto)
  --min-level <n>           Minimum octree level (default: 3)
  --max-level <n>           Maximum octree level (default: 7)
  --station-refine <n>      Refinement level near stations (default: max-level)
  --max-depth <km>          Depth of interest limit in km (default: auto)

Model:
  --sigma <val>             Starting conductivity in S/m (default: 0.01 = 100 Ohm-m)
  --sigma-ocean <val>       Seawater conductivity in S/m (default: 3.3)

Inversion:
  --solver <type>           nlcg (default), lbfgs, or gn-cg
  --niter <n>               Max iterations (default: 50)
  --target-rms <val>        Target RMS misfit (default: 1.0)
  --lambda <val>            Initial regularization parameter (default: 10.0)
  --lambda-dec <val>        Lambda decrease factor (default: 0.6)
  --lambda-strategy <type>  plateau (default) or ratio
  --plateau-tol <val>       RMS improvement threshold (default: 0.02)
  --plateau-patience <n>    Slow iterations before lambda decrease (default: 2)
  --plateau-dec <val>       Lambda multiplier on plateau (default: 0.5)

Data:
  --error-floor <pct>       Impedance error floor in % (default: 5)
  --fmin <val>              Minimum frequency in Hz
  --fmax <val>              Maximum frequency in Hz

Regularization:
  --alpha-s <val>           Smallness weight (default: 1.0)
  --alpha-x <val>           X-smoothing (default: 1.0)
  --alpha-y <val>           Y-smoothing (default: 1.0)
  --alpha-z <val>           Z-smoothing (default: 0.5)

Solver:
  --bicgstab-tol <val>      BiCGStab tolerance (default: 1e-7)
  --bicgstab-maxiter <n>    BiCGStab max iterations (default: 5000)

Output:
  --vtk-interval <n>        Export VTU model every N iterations (default: 1)
```

### Output Files

```
output/
  convergence.csv       # iteration, RMS, lambda, objective, time
  model_iter0.vtu       # Starting model (ParaView)
  model_iter1.vtu       # Iteration 1
  ...
  model_iterN.vtu       # Final model
  data_fit_final.csv    # Observed vs predicted impedance
  stations.csv          # Station coordinates
```

## Directory Structure

```
Maple3DMT/
├── include/maple3dmt/     # Headers
│   ├── octree/          #   Octree mesh, staggered grid, operators
│   ├── forward/         #   Forward solver (BiCGStab, SSOR, QMR, COCR)
│   ├── inversion/       #   NLCG, L-BFGS, GN-CG, regularization
│   ├── data/            #   MT data structures
│   ├── io/              #   EDI, VTK, HDF5 I/O
│   ├── model/           #   Conductivity model
│   └── utils/           #   Logger, memory profiler, freq-parallel
├── src/                 # Implementation (mirrors include/)
├── examples/
│   ├── commemi_3d1a/    #   COMMEMI 3D-1A benchmark (EDI workflow)
│   ├── commemi_3d2/     #   COMMEMI 3D-2 benchmark (EDI workflow)
│   └── *.cpp            #   Test and validation drivers
└── CMakeLists.txt
```

## Benchmark Results

Validated on COMMEMI synthetic benchmarks (121 stations, 11x11 areal grid, 5 frequencies):

| Model | Cells | Edge DOFs | Starting Model | Iterations | Final RMS | Wall Time |
|-------|-------|-----------|----------------|------------|-----------|-----------|
| 3D-1A | ~83K | ~273K | 100 Ohm-m uniform | 30 (NLCG) | 1.49 | ~8 min |
| 3D-2  | ~86K | ~280K | 10 Ohm-m uniform  | 50 (NLCG) | 1.01 | ~11 min |

Measured on a desktop workstation with 12 MPI ranks (Apple M-series, 8 cores).

## Citation

If you use Maple3DMT in your research, please cite:

> Oh, S., Maple3DMT: Adaptive Octree Finite-Volume Code for Memory-Efficient 3-D Magnetotelluric Inversion with Frequency-Parallel Scaling. *Computers & Geosciences* (submitted), 2026.

## License

Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.

This software is released under a custom academic license:
- Redistribution and use for **academic research** are permitted.
- **Modifications** may not be distributed without the author's written consent.
- **Commercial use** requires a separate agreement.

See [LICENSE](LICENSE) for full terms.
