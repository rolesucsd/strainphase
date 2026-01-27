#!/bin/bash
#SBATCH --job-name=strainphase_bench
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err
# Focused test script for real strains benchmarking
# Tests 6 key parameter configurations instead of full grid

set -e

GENOMES_DIR="/Users/reneeoles/Desktop/strainphase/strains"
OUTPUT_DIR="/Users/reneeoles/Desktop/strainphase/results/test_real_strains_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "FOCUSED BENCHMARK TEST - REAL STRAINS"
echo "============================================================"
echo ""
echo "Input strains: $GENOMES_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Test configurations (6 total):"
echo "  1. baseline      - Default recommended parameters"
echo "  2. sensitive     - More sensitive clustering"
echo "  3. strict        - Stricter clustering"
echo "  4. large_windows - Larger windows (50kb)"
echo "  5. small_windows - Smaller windows (10kb)"
echo "  6. high_quality  - Stricter quality filters"
echo ""
echo "Simulation parameters:"
echo "  - Timepoints: 4"
echo "  - Coverage: 30x per timepoint"
echo "  - Mode: Real strains (detect SNVs from FASTA differences)"
echo ""

python benchmarks/run_full_benchmark.py \
    --genomes "$GENOMES_DIR" \
    --output "$OUTPUT_DIR" \
    --use-real-strains \
    --timepoints 4 \
    --coverage 30 \
    --mode sequential \
    --passes 1 \
    --max-configs 6 \
    --workers 8 \
    --checkpoint-interval 2

echo ""
echo "============================================================"
echo "TEST COMPLETE"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "View report: $OUTPUT_DIR/report/benchmark_report.html"
