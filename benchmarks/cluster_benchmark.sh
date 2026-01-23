#!/bin/bash
#SBATCH --job-name=strainphase_bench
#SBATCH --array=1-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err

###############################################################################
# Strainphase Full Benchmark Suite
#
# Runs parameter sweep at 5 different complexity levels:
#   1: Simple     (2 strains)
#   2: Low        (4 strains)
#   3: Medium     (8 strains)
#   4: High       (16 strains)
#   5: Complex    (32 strains)
#
# Usage:
#   # On SLURM cluster:
#   sbatch benchmarks/cluster_benchmark.sh
#
#   # Locally (run specific complexity level):
#   bash benchmarks/cluster_benchmark.sh 3   # Run medium complexity
#
#   # Locally (run all sequentially):
#   for i in 1 2 3 4 5; do bash benchmarks/cluster_benchmark.sh $i; done
###############################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GENOME_SOURCE="${GENOME_SOURCE:-$PROJECT_ROOT/tmp}"  # Directory with source genomes
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/results/cluster_benchmark}"
TIMEPOINTS="${TIMEPOINTS:-4}"
COVERAGE="${COVERAGE:-30}"
SEED="${SEED:-42}"

# Complexity levels: number of strains
declare -A COMPLEXITY_STRAINS=(
    [1]=2    # Simple
    [2]=4    # Low
    [3]=8    # Medium
    [4]=16   # High
    [5]=32   # Complex
)

declare -A COMPLEXITY_NAMES=(
    [1]="simple"
    [2]="low"
    [3]="medium"
    [4]="high"
    [5]="complex"
)

# Get complexity level (from SLURM array or command line)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    COMPLEXITY_LEVEL=$SLURM_ARRAY_TASK_ID
elif [[ -n "${1:-}" ]]; then
    COMPLEXITY_LEVEL=$1
else
    echo "Usage: $0 <complexity_level 1-5>"
    echo "  1: Simple  (2 strains)"
    echo "  2: Low     (4 strains)"
    echo "  3: Medium  (8 strains)"
    echo "  4: High    (16 strains)"
    echo "  5: Complex (32 strains)"
    exit 1
fi

N_STRAINS=${COMPLEXITY_STRAINS[$COMPLEXITY_LEVEL]}
COMPLEXITY_NAME=${COMPLEXITY_NAMES[$COMPLEXITY_LEVEL]}

echo "============================================================"
echo "STRAINPHASE BENCHMARK - ${COMPLEXITY_NAME^^} COMPLEXITY"
echo "============================================================"
echo "Complexity level: $COMPLEXITY_LEVEL"
echo "Number of strains: $N_STRAINS"
echo "Timepoints: $TIMEPOINTS"
echo "Coverage: ${COVERAGE}x"
echo "Genome source: $GENOME_SOURCE"
echo "============================================================"

# Create output directories
OUTPUT_DIR="${OUTPUT_BASE}/${COMPLEXITY_NAME}"
GENOME_DIR="${OUTPUT_DIR}/genomes"
mkdir -p "$OUTPUT_DIR" "$GENOME_DIR" "${PROJECT_ROOT}/logs"

# Select genomes for this complexity level
echo ""
echo "Selecting $N_STRAINS genomes..."

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
echo "Creating $N_STRAINS strain variants..."

# Copy base genome as strain_1 (reference)
cp "$BASE_GENOME" "${GENOME_DIR}/${BASE_NAME}_strain_1.fa"

# Create additional strain copies (simulation will introduce SNVs)
for i in $(seq 2 $N_STRAINS); do
    cp "$BASE_GENOME" "${GENOME_DIR}/${BASE_NAME}_strain_${i}.fa"
done

echo "Created $(ls -1 "$GENOME_DIR"/*.fa 2>/dev/null | wc -l) strain files"

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

python benchmarks/run_full_benchmark.py \
    --genomes "$GENOME_DIR" \
    --output "$OUTPUT_DIR" \
    --timepoints "$TIMEPOINTS" \
    --coverage "$COVERAGE" \
    --seed "$SEED" \
    2>&1 | tee "${OUTPUT_DIR}/benchmark.log"

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
