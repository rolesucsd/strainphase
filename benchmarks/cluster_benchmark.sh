#!/bin/bash
#SBATCH --job-name=strainphase_bench
#SBATCH --array=1-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err

###############################################################################
# Strainphase Full Benchmark Suite
#
# Runs parameter sweep at 3 different complexity levels:
#   1: Simple     (2 strains)
#   2: Medium     (4 strains)
#   3: Complex    (8 strains)
#
# Usage:
#   # On SLURM cluster:
#   sbatch benchmarks/cluster_benchmark.sh
#
#   # Locally (run specific complexity level):
#   bash benchmarks/cluster_benchmark.sh 2   # Run medium complexity
#
#   # Locally (run all sequentially):
#   for i in 1 2 3; do bash benchmarks/cluster_benchmark.sh $i; done
###############################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$SCRIPT_DIR")}"
GENOME_SOURCE="${GENOME_SOURCE:-$PROJECT_ROOT/tmp}"  # Directory with source genomes
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/results/cluster_benchmark}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
TIMEPOINTS="${TIMEPOINTS:-4}"
COVERAGE="${COVERAGE:-50}"
SEED="${SEED:-42}"

# Parameter sweep configuration
# MODE: "grid" for full sweep (13,824 configs), "sequential" for coordinate descent (~27 configs)
# Using "grid" mode with best_params.json to test exactly 4 best parameter combinations
MODE="${MODE:-grid}"
# MAX_CONFIGS: Limit configs for grid mode (set to 4 for best params)
MAX_CONFIGS="${MAX_CONFIGS:-4}"
# Custom parameter file with best 4 combinations (only used in grid mode)
# Sequential mode uses the default REQUIRED_GRID for coordinate descent optimization
PARAMS_FILE="${PARAMS_FILE:-$SCRIPT_DIR/best_params.json}"
# PASSES: Number of optimization passes for sequential mode (not used in grid mode)
PASSES="${PASSES:-1}"
# CHECKPOINT_INTERVAL: Save checkpoint every N configs
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
# WORKERS: Number of parallel workers for window processing (use SLURM cpus if available)
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

# Complexity levels: number of strains
declare -A COMPLEXITY_STRAINS=(
    [1]=2    # Simple
    [2]=4    # Medium
    [3]=8    # Complex
)

declare -A COMPLEXITY_NAMES=(
    [1]="simple"
    [2]="medium"
    [3]="complex"
)

# Get complexity level (from SLURM array or command line)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    COMPLEXITY_LEVEL=$SLURM_ARRAY_TASK_ID
elif [[ -n "${1:-}" ]]; then
    COMPLEXITY_LEVEL=$1
else
    echo "Usage: $0 <complexity_level 1-3>"
    echo "  1: Simple  (2 strains)"
    echo "  2: Medium  (4 strains)"
    echo "  3: Complex (8 strains)"
    exit 1
fi

N_STRAINS=${COMPLEXITY_STRAINS[$COMPLEXITY_LEVEL]}
COMPLEXITY_NAME=${COMPLEXITY_NAMES[$COMPLEXITY_LEVEL]}
SNV_COUNTS=""

case "$COMPLEXITY_LEVEL" in
    1)
        SNV_COUNTS="10000"
        ;;
    2)
        SNV_COUNTS="2500,5000,10000"
        ;;
    3)
        SNV_COUNTS="500,1000,2000,5000,6000,7000,10000"
        ;;
    *)
        echo "Invalid complexity level: $COMPLEXITY_LEVEL"
        exit 1
        ;;
esac

echo "============================================================"
echo "STRAINPHASE BENCHMARK - ${COMPLEXITY_NAME^^} COMPLEXITY"
echo "============================================================"
echo "Complexity level: $COMPLEXITY_LEVEL"
echo "Number of strains: $N_STRAINS"
echo "SNV counts (strains[1:]): $SNV_COUNTS"
echo "Timepoints: $TIMEPOINTS"
echo "Coverage: ${COVERAGE}x"
echo "Genome source: $GENOME_SOURCE"
echo "Sweep mode: $MODE"
if [[ "$MODE" == "sequential" ]]; then
    echo "Optimization passes: $PASSES"
    echo "Parameter grid: Default REQUIRED_GRID (coordinate descent)"
elif [[ "$MODE" == "grid" ]]; then
    if [[ -n "$MAX_CONFIGS" ]]; then
        echo "Max configs: $MAX_CONFIGS"
    fi
    if [[ -n "$PARAMS_FILE" && -f "$PARAMS_FILE" ]]; then
        echo "Custom parameter file: $PARAMS_FILE"
    else
        echo "Parameter grid: Default REQUIRED_GRID"
    fi
fi
echo "Parallel workers: $WORKERS"
echo "============================================================"

# Create output directories
OUTPUT_DIR="${OUTPUT_BASE}/${COMPLEXITY_NAME}"
GENOME_DIR="${OUTPUT_DIR}/genomes"
mkdir -p "$OUTPUT_DIR" "$GENOME_DIR" "$LOG_DIR"

# Select genomes for this complexity level
echo ""
echo "Selecting base genome..."

# Get list of available genomes
AVAILABLE_GENOMES=($(find "$GENOME_SOURCE" -name "*.fa" -o -name "*.fasta" -o -name "*.fna" 2>/dev/null | head -100))

if [[ ${#AVAILABLE_GENOMES[@]} -lt 1 ]]; then
    echo "ERROR: No genome files found in $GENOME_SOURCE"
    exit 1
fi

# For strain simulation, we use one base genome and create N_STRAINS copies
# This ensures all strains share the same contig structure
BASE_GENOME="${AVAILABLE_GENOMES[0]}"
BASE_NAME=$(basename "$BASE_GENOME" | sed 's/\.\(fa\|fasta\|fna\)$//')

echo "Using base genome: $BASE_NAME"
echo "Simulation will generate $N_STRAINS strains from base genome..."

# Copy base genome (single reference input)
cp "$BASE_GENOME" "${GENOME_DIR}/${BASE_NAME}.fa"

echo "Created $(ls -1 "$GENOME_DIR"/*.fa 2>/dev/null | wc -l) genome file"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    # Try to activate strainphase environment
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate strainphase 2>/dev/null || true
fi

# Change to project root
cd "$PROJECT_ROOT"

# Run the benchmark
echo ""
echo "Starting benchmark..."
echo "Output: $OUTPUT_DIR"
echo ""

# Build python command with optional arguments
PYTHON_CMD="python benchmarks/run_full_benchmark.py \
    --genomes $GENOME_DIR \
    --output $OUTPUT_DIR \
    --timepoints $TIMEPOINTS \
    --coverage $COVERAGE \
    --resume \
    --seed $SEED \
    --mode $MODE \
    --snv-counts $SNV_COUNTS \
    --fixed-strains-per-genome $N_STRAINS"

# Add mode-specific options
if [[ "$MODE" == "sequential" ]]; then
    PYTHON_CMD="$PYTHON_CMD --passes $PASSES"
    # Sequential mode uses default REQUIRED_GRID (do NOT pass params_file)
    echo "Sequential mode: Using default parameter grid for coordinate descent optimization"
elif [[ "$MODE" == "grid" && -n "$MAX_CONFIGS" ]]; then
    PYTHON_CMD="$PYTHON_CMD --max-configs $MAX_CONFIGS"
    # Grid mode: use custom parameter file if provided
    if [[ -n "$PARAMS_FILE" && -f "$PARAMS_FILE" ]]; then
        PYTHON_CMD="$PYTHON_CMD --params $PARAMS_FILE"
        echo "Grid mode: Using custom parameter file: $PARAMS_FILE"
    else
        echo "Grid mode: Using default parameter grid"
    fi
fi

PYTHON_CMD="$PYTHON_CMD --checkpoint-interval $CHECKPOINT_INTERVAL"
PYTHON_CMD="$PYTHON_CMD --workers $WORKERS"

# Run the benchmark
eval "$PYTHON_CMD" 2>&1 | tee "${OUTPUT_DIR}/benchmark.log"

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo "BENCHMARK COMPLETE: $COMPLEXITY_NAME"
    echo "============================================================"
    echo "Results: $OUTPUT_DIR"
    echo "Report: ${OUTPUT_DIR}/report/benchmark_report.html"
else
    echo ""
    echo "============================================================"
    echo "BENCHMARK FAILED: $COMPLEXITY_NAME"
    echo "============================================================"
    exit 1
fi
