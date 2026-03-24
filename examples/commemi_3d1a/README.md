# COMMEMI 3D-1A Benchmark Example

Synthetic inversion benchmark: **10 Ohm-m conductive block in 100 Ohm-m halfspace**.

## Model

- **Background**: 100 Ohm-m homogeneous halfspace
- **Block**: 10 Ohm-m, |x| <= 5 km, |y| <= 5 km, depth 0-5 km
- **Stations**: 121 (11x11 areal grid, 3 km spacing, -15 to +15 km)
- **Frequencies**: 5 (0.01, 0.032, 0.1, 0.32, 1.0 Hz)
- **Noise**: 5% Gaussian on impedance

## Quick Start

```bash
# Build (from project root)
cmake -B build -DMAPLE3DMT_USE_MPI=ON
cmake --build build -j8

# Run example
cd examples/commemi_3d1a
bash run.sh
```

## What `run.sh` Does

### Step 1: Generate synthetic data
```bash
mpiexec -np 5 generate_commemi_3d1a_data --output-dir edi
```
- Builds octree mesh (~83K cells)
- Forward solves the true model at 121 stations x 5 frequencies
- Adds 5% Gaussian noise
- Writes 121 EDI files to `edi/`
- Exports `true_model.vtu` for ParaView visualization

### Step 2: Run inversion
```bash
mpiexec -np 12 real_inversion_octree \
    --edi-dir edi \
    --output-dir output \
    --sigma 0.01 \          # Starting model: 100 Ohm-m uniform
    --niter 30 \
    --lambda 10.0 \
    ...
```
- Reads EDI files from `edi/`
- Automatically builds octree mesh from station positions
- Runs 30 NLCG iterations
- Exports per-iteration VTU models, convergence CSV, data fit

## Expected Results

- **RMS**: 6.63 -> ~1.49 in 30 iterations
- **Time**: ~8 minutes (12 MPI ranks, desktop workstation)
- **Block recovery**: Core ~8 Ohm-m, background ~100 Ohm-m

## Output Files

```
output/
  convergence.csv          # RMS, lambda, objective per iteration
  model_iter0.vtu          # Starting model
  model_iter1.vtu          # Iteration 1
  ...
  model_iter30.vtu         # Final model
  data_fit_final.csv       # Observed vs predicted impedance
  stations.csv             # Station coordinates
```

## Customization

Edit `run.sh` to change inversion parameters:
- `--sigma 0.01`: Starting conductivity (S/m). Use `0.01` for 100 Ohm-m halfspace.
- `--niter 30`: Maximum iterations
- `--lambda 10.0`: Initial regularization parameter
- `--solver nlcg|lbfgs|gn-cg`: Optimization algorithm
- `NP_INV=12`: Number of MPI ranks

## Using Your Own EDI Files

Replace the `edi/` directory with your own EDI files and run Step 2 only:
```bash
mpiexec -np 12 ../../build/real_inversion_octree \
    --edi-dir /path/to/your/edi/files \
    --output-dir output \
    --sigma 0.01 \
    --niter 50
```
