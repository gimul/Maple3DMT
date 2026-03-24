# COMMEMI 3D-2 Benchmark Example

Synthetic inversion benchmark: **two blocks (conductive + resistive) in 3-layer earth**.

## Model

- **Layer 1**: 10 Ohm-m, 0 to -3 km
- **Layer 2**: 100 Ohm-m, -3 to -10 km
- **Layer 3**: 0.1 Ohm-m, below -10 km
- **Block A** (conductive): 1 Ohm-m, x in [-5,5] km, y in [2,8] km, z in [-3,-5] km
- **Block B** (resistive): 100 Ohm-m, x in [-5,5] km, y in [-8,-2] km, z in [-3,-5] km
- **Stations**: 121 (11x11 areal grid, 3 km spacing, -15 to +15 km)
- **Frequencies**: 5 (0.01, 0.032, 0.1, 0.32, 1.0 Hz)
- **Noise**: 5% Gaussian on impedance
- **Starting model**: Uniform 10 Ohm-m halfspace (no layered a priori)

## Quick Start

```bash
# Build (from project root)
cmake -B build -DMAPLE3DMT_USE_MPI=ON
cmake --build build -j8

# Run example
cd examples/commemi_3d2
bash run.sh
```

## What `run.sh` Does

### Step 1: Generate synthetic data
```bash
mpiexec -np 5 generate_commemi_3d2_data --output-dir edi
```
- Builds octree mesh (~86K cells) with refinement around both blocks
- Forward solves the true 3-layer + 2-block model
- Adds 5% Gaussian noise
- Writes 121 EDI files to `edi/`
- Exports `true_model.vtu`

### Step 2: Run inversion
```bash
mpiexec -np 12 real_inversion_octree \
    --edi-dir edi \
    --output-dir output \
    --sigma 0.1 \           # Starting model: uniform 10 Ohm-m
    --niter 50 \
    --lambda 10.0 \
    ...
```

## Expected Results

- **RMS**: 3.69 -> ~1.01 in 50 iterations
- **Time**: ~11 minutes (12 MPI ranks, desktop workstation)
- **Recovery**: Both blocks at correct positions; 3-layer background recovered

## Output Files

```
output/
  convergence.csv          # RMS, lambda, objective per iteration
  model_iter0.vtu          # Starting model (uniform 10 Ohm-m)
  model_iter1.vtu          # Iteration 1
  ...
  model_iter50.vtu         # Final model
  data_fit_final.csv       # Observed vs predicted
  stations.csv             # Station coordinates
```

## Customization

Edit `run.sh` to change inversion parameters:
- `--sigma 0.1`: Starting conductivity (S/m). Use `0.1` for 10 Ohm-m halfspace.
- `--niter 50`: Maximum iterations
- `--lambda 10.0`: Initial regularization parameter
- `--solver nlcg|lbfgs|gn-cg`: Optimization algorithm
- `NP_INV=12`: Number of MPI ranks
