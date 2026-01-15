#!/bin/bash
#SBATCH --job-name=haplo_long
#SBATCH --array=4-54
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=48:00:00
#SBATCH -o logs/haplo_long_%A_%a.out
#SBATCH -e logs/haplo_long_%A_%a.err

BASE="/ddn_scratch/roles/strain_analysis/Larry"
REF="${BASE}/results/references/combined_bins.fasta"
OUT_BASE="${BASE}/results/haplotypes/longitudinal"

# Get MAG for this array index
MAG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" mags.txt)

# Samples: comma-separated from samples.txt
SAMPLES=$(paste -sd, "${BASE}/samples.txt")

echo "[$(date)] Processing MAG ${MAG} across samples: ${SAMPLES}"

python "${BASE}/scripts/run_longitudinal.py" \
  --samples "${SAMPLES}" \
  --bams "${BASE}/results/mapping/{sample}.sorted.bam" \
  --vcfs "${BASE}/results/variants/clair3/{sample}/pileup.vcf.gz" \
  --reference "${REF}" \
  --output-dir "${OUT_BASE}/${MAG}" \
  --mags "${MAG}" \
  --window-size 3000 \
  --max-reads 300 \
  --min-anchor-weight 0.15 \
  --rescued-min-weight 0.02 \
  --log-level INFO
