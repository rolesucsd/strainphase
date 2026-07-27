#!/bin/bash
#SBATCH --job-name=haplo_wlr
#SBATCH --array=0-254
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=48:00:00
#SBATCH -o logs_wlr/haplo_wlr_%A_%a.out
#SBATCH -e logs_wlr/haplo_wlr_%A_%a.err
# =============================================================================
# strainphase longitudinal — branch feature/window-linking-rework
#
#   sbatch run_longitudinal_array.sh
#
# One array task per MAG. Produces FOUR tables per MAG (no lineage table — see
# below):
#
#   haplotypes.tsv              one row per haplotype per window per sample
#   windows_within_sample.tsv   merged WITHIN a sample, across windows
#   windows_across_samples.tsv  merged ACROSS samples, at one fixed window
#   window_comparisons.tsv      every attempted comparison + why it passed/failed
#
# The final lineage table is NOT produced. Composing the two linking axes into a
# lineage is still an open decision; these tables are the substrate for it. Add
# --build-lineages to also emit the legacy greedy clustering for comparison.
#
# SVs: --sv-sidecars co-phases structural variants as pseudo-SNVs. They ARE loaded,
# phased and reported, but are EXCLUDED from the identity distance - an invertible
# promoter at af~0.5 flips independently of strain background, so using it as an
# identity marker would split a lineage in two every time the inversion flips,
# destroying the very trajectory it is there to show. Drop the flag (and SV_DIR) if
# you have no sidecars; nothing else is affected.
# =============================================================================
set -euo pipefail

# --- env ----------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-strainphase-wlr}"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# --- paths --------------------------------------------------------------------
BASE="/ddn_scratch/roles/strain_analysis/Larry"
REF="${BASE}/results/references/combined_bins.fasta"
OUT_BASE="${BASE}/results/haplotypes/longitudinal_wlr"
VCF_DIR="${BASE}/results/summarized_vcf"
SV_DIR="${BASE}/results/sv/sidecars"        # strainphase.sv_encoding output, one TSV per sample
BAM_DIR="${BASE}/results/mapping"
MAG_LIST="${BASE}/results/references/mag_list.txt"   # one MAG name per line

mkdir -p logs_wlr "${OUT_BASE}"

# --- pick this task's MAG -----------------------------------------------------
MAG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MAG_LIST}")
if [ -z "${MAG}" ]; then
    echo "No MAG at index ${SLURM_ARRAY_TASK_ID} in ${MAG_LIST} — nothing to do."
    exit 0
fi
OUT_DIR="${OUT_BASE}/${MAG}"
mkdir -p "${OUT_DIR}"

# --- samples ------------------------------------------------------------------
# IMPORTANT ordering note: with --cross-sample-method clique (the default) the
# sample ORDER DOES NOT MATTER — identity is an equivalence class with no time
# axis, which is why it is immune to irregular timepoint spacing.
#
# If you switch to --cross-sample-method reciprocal, this list MUST be in true
# CHRONOLOGICAL order. A plain `ls` sort is NOT chronological here: sample IDs
# mix numeric accessions (000066952) with date-encoded names (LS_9_6_15), and
# sorting those lexicographically scrambles 2012-2016 into arbitrary order.
SAMPLES=$(ls "${BAM_DIR}"/*.sorted.bam | xargs -n1 basename | sed 's/\.sorted\.bam$//' | paste -sd, -)

echo "=== MAG ${MAG} (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "  out     : ${OUT_DIR}"
echo "  samples : $(tr ',' '\n' <<<"${SAMPLES}" | wc -l)"
echo "  started : $(date)"

# --- run ----------------------------------------------------------------------
strainphase longitudinal \
    --samples "${SAMPLES}" \
    --bams    "${BAM_DIR}/{sample}.sorted.bam" \
    --vcfs    "${VCF_DIR}/{sample}.vcf.gz" \
    --sv-sidecars "${SV_DIR}/{sample}.sv_sidecar.tsv" \
    --reference "${REF}" \
    --output-dir "${OUT_DIR}" \
    --mags "${MAG}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --window-size 20000 \
    --max-reads 500 \
    --min-reads-per-window 10 \
    --min-reads-for-rescue 5 \
    --min-read-window-overlap-bp 1000 \
    --min-read-read-overlap-bp 1000 \
    --min-entity-overlap-bp 1000 \
    --min-cosupported-span-frac 0.25 \
    --max-num-diff 1 \
    --lineage-merge-distance 0.01 \
    --min-shared-for-lineage 3 \
    --cross-sample-method clique \
    --min-depth-site 3 \
    --log-level INFO

echo "  finished: $(date)"
echo
echo "=== output ==="
for f in haplotypes.tsv windows_within_sample.tsv windows_across_samples.tsv window_comparisons.tsv; do
    if [ -s "${OUT_DIR}/${f}" ]; then
        printf '  %-28s %8d rows\n' "${f}" "$(($(wc -l < "${OUT_DIR}/${f}") - 1))"
    else
        printf '  %-28s %8s\n' "${f}" "EMPTY"
    fi
done

# Comparison-outcome breakdown. A high failed_no_evidence share means dropouts
# (too few shared markers / too little overlap); a high failed_mismatch share
# means genuine genotypic disagreement, i.e. candidate recombination breakpoints.
if [ -s "${OUT_DIR}/window_comparisons.tsv" ]; then
    echo
    echo "  comparison outcomes:"
    awk -F'\t' 'NR>1 {c[$7]++} END {for (k in c) printf "    %-20s %8d\n", k, c[k]}' \
        "${OUT_DIR}/window_comparisons.tsv"
fi
