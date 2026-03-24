#!/bin/bash
# Copyright (c) 2024-2026 Kangwon National University, Prof. Seokhoon Oh.
# All rights reserved.

# =============================================================================
# COMMEMI 3D-1A Benchmark: Generate Data + Run Inversion
# =============================================================================
#
# This script runs the complete COMMEMI 3D-1A benchmark in two steps:
#   1. Generate synthetic EDI files from the true model (forward solve + noise)
#   2. Run 3D MT inversion using the generic real_inversion_octree driver
#
# Prerequisites:
#   - Maple3DMT built (cmake --build build)
#   - MPI available (mpiexec / mpirun)
#
# Usage:
#   cd examples/commemi_3d1a
#   bash run.sh
#
# Adjust NP (number of MPI ranks) as needed.
# =============================================================================

set -e

# Configuration
NP=5                    # MPI ranks for data generation (= number of frequencies)
NP_INV=12               # MPI ranks for inversion
BUILD_DIR="../../build"  # Path to build directory
EDI_DIR="edi"
OUTPUT_DIR="output"

# Binaries
GEN_BIN="${BUILD_DIR}/generate_commemi_3d1a_data"
INV_BIN="${BUILD_DIR}/real_inversion_octree"

# -----------------------------------------------------------------
# Step 1: Generate synthetic EDI files (if not already done)
# -----------------------------------------------------------------
if [ ! -d "${EDI_DIR}" ] || [ -z "$(ls -A ${EDI_DIR}/*.edi 2>/dev/null)" ]; then
    echo ""
    echo "============================================"
    echo "  Step 1: Generating synthetic EDI files"
    echo "============================================"
    echo ""

    if [ ! -f "${GEN_BIN}" ]; then
        echo "ERROR: ${GEN_BIN} not found. Build the project first:"
        echo "  cmake --build build -j8"
        exit 1
    fi

    mpiexec -np ${NP} "${GEN_BIN}" --output-dir "${EDI_DIR}"
    echo ""
    echo "  EDI files generated: $(ls ${EDI_DIR}/*.edi 2>/dev/null | wc -l) stations"
else
    echo "  EDI directory exists ($(ls ${EDI_DIR}/*.edi 2>/dev/null | wc -l) files). Skipping generation."
fi

# -----------------------------------------------------------------
# Step 2: Run inversion
# -----------------------------------------------------------------
echo ""
echo "============================================"
echo "  Step 2: Running inversion (NLCG)"
echo "============================================"
echo ""

if [ ! -f "${INV_BIN}" ]; then
    echo "ERROR: ${INV_BIN} not found. Build the project first:"
    echo "  cmake --build build -j8"
    exit 1
fi

mpiexec -np ${NP_INV} "${INV_BIN}" \
    --edi-dir "${EDI_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --domain-size 200 \
    --min-level 5 \
    --max-level 7 \
    --station-refine 7 \
    --sigma 0.01 \
    --solver nlcg \
    --niter 30 \
    --target-rms 1.0 \
    --lambda 10.0 \
    --lambda-dec 0.7 \
    --lambda-strategy ratio \
    --error-floor 5 \
    --alpha-s 1e-4 \
    --alpha-x 1.0 \
    --alpha-y 1.0 \
    --alpha-z 0.5 \
    --bicgstab-tol 1e-7 \
    --vtk-interval 1

echo ""
echo "============================================"
echo "  Inversion complete!"
echo "  Results: ${OUTPUT_DIR}/"
echo "============================================"
