#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Run Strainy on all timepoints, convert output, and validate against truth.
#
# Usage:
#   bash benchmarks/run_strainy_validation.sh \
#       --bam-dir results/mixed_samples_10x \
#       --vcf results/mixed_samples_10x/variants.vcf.gz \
#       --ref results/mixed_samples_10x/reference.fasta \
#       --truth-dir results/truth \
#       --output-dir results/strainy_10x \
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
STRAINY_BIN="strainy"
STRAINY_MODE="hifi"
STRAINPHASE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Required:
  --bam-dir DIR       Directory containing per-timepoint BAMs ({timepoint}.bam)
  --vcf FILE          VCF file (shared across timepoints)
  --ref FILE          Reference FASTA
  --truth-dir DIR     Truth directory for validation
  --output-dir DIR    Output directory for all Strainy + validation results

Optional:
  --timepoints LIST   Comma-separated timepoint names (default: T1,T2,T3,T4)
  --strainy-bin PATH  Path to strainy.py binary (default: strainy.py)
  --strainy-mode MODE Strainy mode: hifi or nano (default: hifi)
  --help              Show this help
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bam-dir)      BAM_DIR="$2"; shift 2 ;;
        --vcf)          VCF="$2"; shift 2 ;;
        --ref)          REF="$2"; shift 2 ;;
        --truth-dir)    TRUTH_DIR="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --timepoints)   TIMEPOINTS="$2"; shift 2 ;;
        --strainy-bin)  STRAINY_BIN="$2"; shift 2 ;;
        --strainy-mode) STRAINY_MODE="$2"; shift 2 ;;
        --help)         usage ;;
        *)              echo "Unknown option: $1"; usage ;;
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
echo "Strainy validation pipeline"
echo "============================================================"
echo "  BAM dir:     $BAM_DIR"
echo "  VCF:         $VCF"
echo "  Reference:   $REF"
echo "  Truth dir:   $TRUTH_DIR"
echo "  Output dir:  $OUTPUT_DIR"
echo "  Timepoints:  ${TP_ARRAY[*]}"
echo "  Strainy bin: $STRAINY_BIN"
echo "  Strainy mode: $STRAINY_MODE"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# -------------------------------------------------------------------------
# Step 1: Convert BAM to FASTQ for each timepoint
# -------------------------------------------------------------------------
echo "=== Step 1: Converting BAM to FASTQ ==="

FASTQ_DIR="$OUTPUT_DIR/fastq"
mkdir -p "$FASTQ_DIR"

for TP in "${TP_ARRAY[@]}"; do
    BAM="$BAM_DIR/${TP}.bam"
    FASTQ="$FASTQ_DIR/${TP}.fastq"

    if [[ ! -f "$BAM" ]]; then
        echo "  WARNING: BAM not found for $TP: $BAM — skipping"
        continue
    fi

    if [[ -f "$FASTQ" ]]; then
        echo "  $TP: FASTQ already exists at $FASTQ — skipping"
    else
        echo "  $TP: Converting BAM to FASTQ..."
        samtools fastq "$BAM" > "$FASTQ" 2>> "$OUTPUT_DIR/samtools_fastq.log"
        echo "  $TP: Created $FASTQ"
    fi
done

echo ""

# -------------------------------------------------------------------------
# Step 2: Run Strainy for each timepoint
# -------------------------------------------------------------------------
echo "=== Step 2: Running Strainy per timepoint ==="

for TP in "${TP_ARRAY[@]}"; do
    BAM="$BAM_DIR/${TP}.bam"
    FASTQ="$FASTQ_DIR/${TP}.fastq"
    STRAINY_OUT="$OUTPUT_DIR/strainy_${TP}"

    if [[ ! -f "$BAM" ]]; then
        echo "  WARNING: BAM not found for $TP: $BAM — skipping"
        continue
    fi

    if [[ ! -f "$FASTQ" ]]; then
        echo "  WARNING: FASTQ not found for $TP: $FASTQ — skipping"
        continue
    fi

    if [[ -d "$STRAINY_OUT" && -f "$STRAINY_OUT/strainy_final.gfa" ]]; then
        echo "  $TP: Strainy output already exists at $STRAINY_OUT — skipping"
    else
        echo "  $TP: Running strainy..."
        mkdir -p "$STRAINY_OUT"
        "$STRAINY_BIN" \
            --fasta_ref "$REF" \
            --fastq "$FASTQ" \
            --mode "$STRAINY_MODE" \
            --bam "$BAM" \
            --snp "$VCF" \
            --unitig-split-length 0 \
            --stage phase \
            --min-unitig-coverage 1 \
            --output "$STRAINY_OUT" \
            2>&1 | tee "$STRAINY_OUT/strainy_stdout.log" || {
                echo "  WARNING: Strainy failed for $TP"
                continue
            }
        echo "  $TP: Strainy complete"
    fi
done

echo ""

# -------------------------------------------------------------------------
# Step 3: Convert each Strainy output to per-timepoint lineages.tsv
# -------------------------------------------------------------------------
echo "=== Step 3: Converting Strainy output to lineages.tsv ==="

CONVERT_DIR="$OUTPUT_DIR/converted"
mkdir -p "$CONVERT_DIR"

for TP in "${TP_ARRAY[@]}"; do
    STRAINY_OUT="$OUTPUT_DIR/strainy_${TP}"
    TP_CONVERT_DIR="$CONVERT_DIR/${TP}"

    if [[ ! -d "$STRAINY_OUT" ]]; then
        echo "  $TP: No Strainy output — skipping"
        continue
    fi

    echo "  $TP: Converting..."
    python "$STRAINPHASE_ROOT/validation/convert_strainy.py" \
        --strainy-dir "$STRAINY_OUT" \
        --vcf "$VCF" \
        --sample "$TP" \
        --output-dir "$TP_CONVERT_DIR" \
        2>> "$OUTPUT_DIR/convert.log" || {
            echo "  WARNING: Conversion failed for $TP"
            continue
        }
    echo "  $TP: Wrote $TP_CONVERT_DIR/lineages.tsv"
done

echo ""

# -------------------------------------------------------------------------
# Step 4: Combine per-timepoint lineages into one file
# -------------------------------------------------------------------------
echo "=== Step 4: Combining lineages across timepoints ==="

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

if [[ -f "$COMBINED" ]]; then
    N_LINES=$(( $(wc -l < "$COMBINED") - 1 ))
    echo "  Combined $N_LINES haplotypes into $COMBINED"
else
    echo "  WARNING: No lineages.tsv files were combined"
fi

echo ""

# -------------------------------------------------------------------------
# Step 5: Run validation
# -------------------------------------------------------------------------
echo "=== Step 5: Running validation ==="

VALIDATION_DIR="$OUTPUT_DIR/validation"
mkdir -p "$VALIDATION_DIR"

if [[ -f "$COMBINED" ]]; then
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
try:
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
except FileNotFoundError:
    print('ERROR: validation_metrics.json not found')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: Failed to parse validation metrics: {e}')
    sys.exit(1)
"
    fi
else
    echo "ERROR: No combined lineages.tsv found — validation skipped"
    exit 1
fi
