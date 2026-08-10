#!/usr/bin/env bash
#SBATCH --job-name=spb-run
#
# Stage 2: one array task per (dataset, tool) pair.
#
# The index-to-unit mapping is stable for a given config, so a failed task can
# be resubmitted by index alone:
#     spbench plan -c CONFIG          # see the mapping
#     sbatch --array=17,31 ...        # rerun just those
#
# A tool that is not installed exits 0 with a `skipped` status file; a tool that
# crashes exits 0 with a `failed` status file and the error attached. Neither
# takes down the array, and both are visible in results/runs.tsv.
# SLURM copies the batch script to a spool directory, so its own path cannot be
# used to locate sibling files. submit.sh exports the benchmark directory for
# exactly this reason; the fallback keeps the script runnable by hand.
source "${SPBENCH_BENCHMARK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/_common.sh"

: "${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}"

python -m spbench.cli plan -c "$SPBENCH_CONFIG" \
    | awk -v i="$SLURM_ARRAY_TASK_ID" -F'\t' '$1 == i {print "unit      : " $2 " / " $3}'

python -m spbench.cli run-one \
    -c "$SPBENCH_CONFIG" \
    -w "$SPBENCH_WORKDIR" \
    --index "$SLURM_ARRAY_TASK_ID"

echo "---"
echo "finished  : $(date -Is)"
