#!/bin/bash
# Run default-parameter validation across multiple coverage levels.
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# =============================================================================
# Configuration
# =============================================================================

STRAINS_DIR="${PROJECT_ROOT}/strains"
OUTPUT_ROOT="${PROJECT_ROOT}/results"

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
N_WORKERS=8
COVERAGES=(5 10 20 50)

# =============================================================================
# Main
# =============================================================================

echo "Project root: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"

for COV in "${COVERAGES[@]}"; do
    OUTPUT_DIR="${OUTPUT_ROOT}/test_real_strains_${COV}x"
    MIXED_DATA_DIR="${OUTPUT_DIR}/mixed_samples"

    mkdir -p "$OUTPUT_DIR"

    echo "============================================================"
    echo "REAL STRAINS BENCHMARK - ISOLATE MIX MODE"
    echo "============================================================"
    echo ""
    echo "Input strains directory: $STRAINS_DIR"
    echo "Output directory: $OUTPUT_DIR"
    echo "Number of isolates: ${#BAMS[@]}"
    echo "Timepoints: $N_TIMEPOINTS"
    echo "Target coverage: ${COV}x per timepoint"
    echo "Workers: $N_WORKERS"
    echo ""

    echo "Preparing mixed samples (coverage ${COV}x)..."
    python "${PROJECT_ROOT}/benchmarks/prepare_isolate_mix.py" \
        --bams "${BAMS[@]}" \
        --vcfs "${VCFS[@]}" \
        --reference "$REFERENCE" \
        --output "$MIXED_DATA_DIR" \
        --timepoints "$N_TIMEPOINTS" \
        --target-coverage "$COV" \
        --abundance-profile sweep

    echo "Running parameter_sweep in default mode..."
    python "${PROJECT_ROOT}/benchmarks/parameter_sweep.py" \
        --bam-paths "${MIXED_DATA_DIR}/T1.bam" "${MIXED_DATA_DIR}/T2.bam" "${MIXED_DATA_DIR}/T3.bam" "${MIXED_DATA_DIR}/T4.bam" \
        --vcf-paths "${MIXED_DATA_DIR}/variants.vcf.gz" "${MIXED_DATA_DIR}/variants.vcf.gz" "${MIXED_DATA_DIR}/variants.vcf.gz" "${MIXED_DATA_DIR}/variants.vcf.gz" \
        --reference "${MIXED_DATA_DIR}/reference.fasta" \
        --timepoints T1 T2 T3 T4 \
        --output "${OUTPUT_DIR}/sweep_results" \
        --truth "$MIXED_DATA_DIR" \
        --mode default \
        --workers "$N_WORKERS"

    echo "Done coverage ${COV}x."
done
