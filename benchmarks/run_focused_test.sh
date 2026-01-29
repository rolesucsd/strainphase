#!/bin/bash
#SBATCH --job-name=strainphase_bench
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err
# Focused test script for real strains benchmarking
# Uses real isolate BAMs and VCFs to create mixed samples

set -e

# =============================================================================
# Setup Python path for validation module
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Add project root to PYTHONPATH so validation module can be found
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "Project root: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"

# =============================================================================
# Configuration
# =============================================================================

STRAINS_DIR="${PROJECT_ROOT}/strains"
OUTPUT_DIR="${PROJECT_ROOT}/results/test_real_strains_$(date +%Y%m%d_%H%M%S)"
MIXED_DATA_DIR="${OUTPUT_DIR}/mixed_samples"

# Reference genome
REFERENCE="${STRAINS_DIR}/F9DRQ4_1_sample_1_1_reference.fasta"

# Input BAM files (one per isolate)
BAMS=(
    "${STRAINS_DIR}/F9DRQ4_1_sample_1.sorted.bam"
    "${STRAINS_DIR}/F9DRQ4_2_sample_2.sorted.bam"
    "${STRAINS_DIR}/F9DRQ4_4_sample_4.sorted.bam"
    "${STRAINS_DIR}/F9DRQ4_5_sample_5.sorted.bam"
)

# Input VCF files (one per isolate, same order as BAMs)
VCFS=(
    "${STRAINS_DIR}/F9DRQ4_1_sample_1.filtered.vcf.gz"
    "${STRAINS_DIR}/F9DRQ4_2_sample_2.filtered.vcf.gz"
    "${STRAINS_DIR}/F9DRQ4_4_sample_4.filtered.vcf.gz"
    "${STRAINS_DIR}/F9DRQ4_5_sample_5.filtered.vcf.gz"
)

# Benchmark parameters
N_TIMEPOINTS=4
TARGET_COVERAGE=100
N_WORKERS=8

# =============================================================================
# Main Script
# =============================================================================

echo "============================================================"
echo "REAL STRAINS BENCHMARK - ISOLATE MIX MODE"
echo "============================================================"
echo ""
echo "Input strains directory: $STRAINS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of isolates: ${#BAMS[@]}"
echo ""
echo "Step 1: Prepare mixed samples from isolate BAMs"
echo "Step 2: Run strainphase parameter sweep (sequential)"
echo "Step 3: Generate benchmark report"
echo ""
echo "Parameters:"
echo "  - Timepoints: $N_TIMEPOINTS"
echo "  - Target coverage: ${TARGET_COVERAGE}x per timepoint"
echo "  - Workers: $N_WORKERS"
echo ""

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Step 1: Prepare mixed samples from isolate BAMs and VCFs
# =============================================================================

echo "============================================================"
echo "STEP 1: Preparing mixed samples from isolates"
echo "============================================================"
echo ""

python "${PROJECT_ROOT}/benchmarks/prepare_isolate_mix.py" \
    --bams "${BAMS[@]}" \
    --vcfs "${VCFS[@]}" \
    --reference "$REFERENCE" \
    --output "$MIXED_DATA_DIR" \
    --timepoints "$N_TIMEPOINTS" \
    --target-coverage "$TARGET_COVERAGE" \
    --abundance-profile sweep

echo ""
echo "Mixed samples created in: $MIXED_DATA_DIR"
echo ""

# =============================================================================
# Step 2: Run parameter sweep on mixed data (sequential)
# =============================================================================

echo "============================================================"
echo "STEP 2: Running parameter sweep"
echo "============================================================"
echo ""

# The mixed data directory now contains:
# - T1.bam, T2.bam, T3.bam, T4.bam (mixed samples)
# - variants.vcf.gz (combined variant sites)
# - reference.fasta (reference genome)
# - truth_*.tsv files (ground truth for validation)

python "${PROJECT_ROOT}/benchmarks/parameter_sweep.py" \
    --bam-paths "${MIXED_DATA_DIR}/T1.bam" "${MIXED_DATA_DIR}/T2.bam" "${MIXED_DATA_DIR}/T3.bam" "${MIXED_DATA_DIR}/T4.bam" \
    --vcf-paths "${MIXED_DATA_DIR}/variants.vcf.gz" "${MIXED_DATA_DIR}/variants.vcf.gz" "${MIXED_DATA_DIR}/variants.vcf.gz" "${MIXED_DATA_DIR}/variants.vcf.gz" \
    --reference "${MIXED_DATA_DIR}/reference.fasta" \
    --timepoints T1 T2 T3 T4 \
    --output "${OUTPUT_DIR}/sweep_results" \
    --truth "$MIXED_DATA_DIR" \
    --mode sequential \
    --passes 3 \
    --workers "$N_WORKERS" \
    --checkpoint-interval 2

echo ""

# =============================================================================
# Step 3: Generate report
# =============================================================================

echo "============================================================"
echo "STEP 3: Generating benchmark report"
echo "============================================================"
echo ""

python "${PROJECT_ROOT}/benchmarks/generate_report.py" \
    --results "${OUTPUT_DIR}/sweep_results" \
    --output "${OUTPUT_DIR}/report" \
    --validation "${OUTPUT_DIR}/sweep_results" \
    2>/dev/null || echo "Report generation skipped (optional)"

echo ""
echo "============================================================"
echo "BENCHMARK COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key outputs:"
echo "  - Mixed samples: ${MIXED_DATA_DIR}/"
echo "  - Sweep results: ${OUTPUT_DIR}/sweep_results/"
echo "  - Report: ${OUTPUT_DIR}/report/benchmark_report.html"
echo ""
echo "Ground truth files:"
echo "  - ${MIXED_DATA_DIR}/truth_strains.tsv"
echo "  - ${MIXED_DATA_DIR}/truth_abundances.tsv"
echo "  - ${MIXED_DATA_DIR}/truth_haplotypes.tsv"
echo ""
