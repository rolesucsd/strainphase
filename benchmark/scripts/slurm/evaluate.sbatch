#!/usr/bin/env bash
#SBATCH --job-name=spb-eval
#
# Stage 3: score every prediction on disk and write the report.
#
# Depends on the array with `afterany`, so a crashed tool does not prevent the
# rest of the run from being scored. Reads only the common-format prediction
# files, so it can be rerun at any time - for example with a different
# match_threshold - without re-running a single tool.
# SLURM copies the batch script to a spool directory, so its own path cannot be
# used to locate sibling files. submit.sh exports the benchmark directory for
# exactly this reason; the fallback keeps the script runnable by hand.
source "${SPBENCH_BENCHMARK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/_common.sh"

python -m spbench.cli evaluate -c "$SPBENCH_CONFIG" -w "$SPBENCH_WORKDIR"

echo "---"
echo "Report: $SPBENCH_WORKDIR/results/report.md"
echo
sed -n '1,120p' "$SPBENCH_WORKDIR/results/report.md" || true
echo "---"
echo "finished  : $(date -Is)"
