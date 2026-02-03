#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Run Floria on all timepoints, convert output, and validate against truth.
#
# Usage:
#   bash benchmarks/run_floria_validation.sh \
#       --bam-dir results/mixed_samples_10x \
#       --vcf results/mixed_samples_10x/variants.vcf \
#       --ref results/mixed_samples_10x/reference.fasta \
#       --truth-dir results/truth \
#       --output-dir results/floria_validation \
#       --timepoints T1,T2,T3,T4
#
# Expects BAM files named {bam-dir}/{timepoint}.bam (e.g. T1.bam, T2.bam, ...)
# =============================================================================

# Defaults
BAM_DIR=""
VCF=""
REF=""
TRUTH_DIR=""
OUTPUT_DIR=""
TIMEPOINTS="T1,T2,T3,T4"
FLORIA_BIN="floria"
STRAINPHASE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Required:
  --bam-dir DIR       Directory containing per-timepoint BAMs ({timepoint}.bam)
  --vcf FILE          VCF file (shared across timepoints, or use --vcf-per-sample)
  --ref FILE          Reference FASTA
  --truth-dir DIR     Truth directory for validation
  --output-dir DIR    Output directory for all Floria + validation results

Optional:
  --timepoints LIST   Comma-separated timepoint names (default: T1,T2,T3,T4)
  --floria-bin PATH   Path to floria binary (default: floria)
  --help              Show this help
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bam-dir)    BAM_DIR="$2"; shift 2 ;;
        --vcf)        VCF="$2"; shift 2 ;;
        --ref)        REF="$2"; shift 2 ;;
        --truth-dir)  TRUTH_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --timepoints) TIMEPOINTS="$2"; shift 2 ;;
        --floria-bin) FLORIA_BIN="$2"; shift 2 ;;
        --help)       usage ;;
        *)            echo "Unknown option: $1"; usage ;;
    esac
done

# Validate required arguments
for var in BAM_DIR VCF REF TRUTH_DIR OUTPUT_DIR; do
    if [[ -z "${!var}" ]]; then
        echo "ERROR: --$(echo "$var" | tr '_' '-' | tr '[:upper:]' '[:lower:]') is required"
        usage
    fi
done

IFS=',' read -ra TP_ARRAY <<< "$TIMEPOINTS"

echo "============================================================"
echo "Floria validation pipeline"
echo "============================================================"
echo "  BAM dir:     $BAM_DIR"
echo "  VCF:         $VCF"
echo "  Reference:   $REF"
echo "  Truth dir:   $TRUTH_DIR"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Timepoints:  ${TP_ARRAY[*]}"
echo "  Floria bin:  $FLORIA_BIN"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# -------------------------------------------------------------------------
# Step 1: Run Floria for each timepoint
# -------------------------------------------------------------------------
echo "=== Step 1: Running Floria per timepoint ==="

for TP in "${TP_ARRAY[@]}"; do
    BAM="$BAM_DIR/${TP}.bam"
    FLORIA_OUT="$OUTPUT_DIR/floria_${TP}"

    if [[ ! -f "$BAM" ]]; then
        echo "WARNING: BAM not found for $TP: $BAM — skipping"
        continue
    fi

    if [[ -d "$FLORIA_OUT" && -f "$FLORIA_OUT/cmd.log" ]]; then
        echo "  $TP: Floria output already exists at $FLORIA_OUT — skipping"
    else
        echo "  $TP: Running floria..."
        mkdir -p "$FLORIA_OUT"
        "$FLORIA_BIN" \
            -b "$BAM" \
            -v "$VCF" \
            -r "$REF" \
            -o "$FLORIA_OUT" \
            2>&1 | tee "$FLORIA_OUT/floria_stdout.log"
        echo "  $TP: Floria complete"
    fi
done

echo ""

# -------------------------------------------------------------------------
# Step 2: Convert each Floria output to per-timepoint lineages.tsv
# -------------------------------------------------------------------------
echo "=== Step 2: Converting Floria output to lineages.tsv ==="

CONVERT_DIR="$OUTPUT_DIR/converted"
mkdir -p "$CONVERT_DIR"

for TP in "${TP_ARRAY[@]}"; do
    FLORIA_OUT="$OUTPUT_DIR/floria_${TP}"
    TP_CONVERT_DIR="$CONVERT_DIR/${TP}"

    if [[ ! -d "$FLORIA_OUT" ]]; then
        echo "  $TP: No Floria output — skipping"
        continue
    fi

    echo "  $TP: Converting..."
    python "$STRAINPHASE_ROOT/validation/convert_floria.py" \
        --floria-dir "$FLORIA_OUT" \
        --vcf "$VCF" \
        --sample "$TP" \
        --output-dir "$TP_CONVERT_DIR"
    echo "  $TP: Wrote $TP_CONVERT_DIR/lineages.tsv"
done

echo ""

# -------------------------------------------------------------------------
# Step 3: Combine per-timepoint lineages into one file
# -------------------------------------------------------------------------
echo "=== Step 3: Combining lineages across timepoints ==="

COMBINED="$OUTPUT_DIR/lineages.tsv"
HEADER_WRITTEN=false

for TP in "${TP_ARRAY[@]}"; do
    TP_FILE="$CONVERT_DIR/${TP}/lineages.tsv"
    if [[ ! -f "$TP_FILE" ]]; then
        echo "  $TP: No lineages.tsv — skipping"
        continue
    fi

    if [[ "$HEADER_WRITTEN" == false ]]; then
        # Write header from first file
        head -1 "$TP_FILE" > "$COMBINED"
        HEADER_WRITTEN=true
    fi

    # Append data rows (skip header), prefixing lineage_id and track_id
    # with timepoint to avoid ID collisions across timepoints
    tail -n +2 "$TP_FILE" | while IFS=$'\t' read -r lineage_id sample contig track_id rest; do
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "${TP}_${lineage_id}" "$sample" "$contig" "${TP}_${track_id}" "$rest"
    done >> "$COMBINED"
done

N_LINES=$(( $(wc -l < "$COMBINED") - 1 ))
echo "  Combined $N_LINES haplotypes into $COMBINED"
echo ""

# -------------------------------------------------------------------------
# Step 4: Run validation
# -------------------------------------------------------------------------
echo "=== Step 4: Running validation ==="

VALIDATION_DIR="$OUTPUT_DIR/validation"
mkdir -p "$VALIDATION_DIR"

python -m validation.validate_haplotypes \
    --detected "$COMBINED" \
    --truth "$TRUTH_DIR" \
    --output "$VALIDATION_DIR"

echo ""
echo "============================================================"
echo "Validation complete. Results in: $VALIDATION_DIR"
echo "============================================================"
echo ""
echo "Key output files:"
echo "  Combined lineages:   $COMBINED"
echo "  Validation metrics:  $VALIDATION_DIR/validation_metrics.json"
echo "  Detailed report:     $VALIDATION_DIR/detailed_report.txt"
echo ""

# Print summary from JSON
if command -v python3 &>/dev/null; then
    python3 -c "
import json, sys
with open('$VALIDATION_DIR/validation_metrics.json') as f:
    m = json.load(f)
print('SUMMARY:')
print(f'  Haplotype precision: {m[\"precision\"]:.3f}')
print(f'  Haplotype recall:    {m[\"recall\"]:.3f}')
print(f'  Haplotype F1:        {m[\"f1\"]:.3f}')
print(f'  SNV precision:       {m[\"snv_precision\"]:.3f}')
print(f'  SNV recall:          {m[\"snv_recall\"]:.3f}')
print(f'  Abundance Pearson r: {m[\"abundance_pearson_r\"]:.3f}')
print(f'  False positives:     {len(m.get(\"false_positives\", []))}')
print(f'  False negatives:     {len(m.get(\"false_negatives\", []))}')
print(f'  N true:              {m[\"n_true\"]}')
print(f'  N detected:          {m[\"n_detected\"]}')
print(f'  N matched:           {m[\"n_matched\"]}')
"
fi
