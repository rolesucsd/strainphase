#!/bin/bash
#SBATCH --job-name=strainphase_bench
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err
# Focused test script for real strains benchmarking
# Tests 6 key parameter configurations instead of full grid
#
# Usage:
#   sbatch benchmarks/run_focused_test.sh [GENOMES_DIR] [OUTPUT_DIR]
#   Or set environment variables:
#   export STRAINPHASE_GENOMES_DIR=/path/to/strains
#   export STRAINPHASE_OUTPUT_DIR=/path/to/output
#   sbatch benchmarks/run_focused_test.sh

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments or use environment variables
GENOMES_DIR="${1:-${STRAINPHASE_GENOMES_DIR:-${PROJECT_ROOT}/strains}}"
OUTPUT_DIR="${2:-${STRAINPHASE_OUTPUT_DIR:-${PROJECT_ROOT}/results/test_real_strains_$(date +%Y%m%d_%H%M%S)}}"

# Ensure absolute paths
if [ -d "$GENOMES_DIR" ]; then
    GENOMES_DIR="$(cd "$GENOMES_DIR" && pwd)"
elif [ -d "$(dirname "$GENOMES_DIR")" ]; then
    GENOMES_DIR="$(cd "$(dirname "$GENOMES_DIR")" && pwd)/$(basename "$GENOMES_DIR")"
else
    echo "ERROR: Genomes directory does not exist: $GENOMES_DIR" >&2
    exit 1
fi

# Create output directory parent if needed
mkdir -p "$(dirname "$OUTPUT_DIR")"
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "============================================================"
echo "FOCUSED BENCHMARK TEST - REAL STRAINS"
echo "============================================================"
echo ""
echo "Project root: $PROJECT_ROOT"
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

python "$PROJECT_ROOT/benchmarks/run_full_benchmark.py" \
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
