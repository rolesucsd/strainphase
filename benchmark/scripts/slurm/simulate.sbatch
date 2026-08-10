#!/usr/bin/env bash
#SBATCH --job-name=spb-sim
#
# Stage 1: generate every dataset the config names.
#
# Runs once, before the array. Simulation is deterministic and cached by config
# fingerprint, so resubmitting after a partial failure regenerates only what
# changed.
# SLURM copies the batch script to a spool directory, so its own path cannot be
# used to locate sibling files. submit.sh exports the benchmark directory for
# exactly this reason; the fallback keeps the script runnable by hand.
source "${SPBENCH_BENCHMARK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/scripts/slurm/_common.sh"

python -m spbench.cli simulate -c "$SPBENCH_CONFIG" -w "$SPBENCH_WORKDIR"

echo "---"
echo "finished  : $(date -Is)"
