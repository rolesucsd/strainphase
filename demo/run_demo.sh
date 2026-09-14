#!/usr/bin/env bash
# Run StrainPhase on the bundled demo dataset and compare the result to ground truth.
# Usage:  bash demo/run_demo.sh [output_dir]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/output}"

strainphase longitudinal \
  --samples T1,T2,T3,T4,T5,T6 \
  --bams "$HERE/bam/{sample}.bam" \
  --vcfs "$HERE/variants/{sample}.vcf.gz" \
  --reference "$HERE/reference.fasta" \
  --output-dir "$OUT"

echo
python3 "$HERE/check_demo.py" "$OUT"
