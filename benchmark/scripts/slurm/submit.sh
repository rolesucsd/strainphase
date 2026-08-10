#!/usr/bin/env bash
#
# Submit a benchmark run to SLURM as three dependent stages.
#
#   1. simulate   one job, generates every dataset
#   2. run        an array, one task per (dataset, tool) pair
#   3. evaluate   one job, scores everything and writes the report
#
# The dependencies matter. Stage 1 must finish before the array starts: array
# tasks share dataset directories, and letting them simulate on demand would
# have several processes writing the same files. Stage 3 uses `afterany`, not
# `afterok` - if one tool crashes on one dataset, its rows are recorded as
# `failed` and the rest of the run is still scored, which is what you want at
# 03:00 with 200 tasks queued.
#
# Usage:
#   scripts/slurm/submit.sh -c configs/standard.yaml -w /scratch/$USER/spbench
#
# Site-specific settings (partition, account, module loads, conda activation)
# live in an env file - copy scripts/slurm/env.sh.example and point at it with
# --env-setup, or set SPBENCH_ENV_SETUP.

set -euo pipefail

CONFIG=""
WORKDIR=""
ENV_SETUP="${SPBENCH_ENV_SETUP:-}"
PARTITION="${SPBENCH_PARTITION:-}"
ACCOUNT="${SPBENCH_ACCOUNT:-}"
THREADS="${SPBENCH_THREADS:-8}"
RUN_TIME="${SPBENCH_RUN_TIME:-08:00:00}"
RUN_MEM="${SPBENCH_RUN_MEM:-32G}"
SIM_TIME="${SPBENCH_SIM_TIME:-04:00:00}"
SIM_MEM="${SPBENCH_SIM_MEM:-16G}"
EVAL_TIME="${SPBENCH_EVAL_TIME:-02:00:00}"
EVAL_MEM="${SPBENCH_EVAL_MEM:-16G}"
MAX_CONCURRENT="${SPBENCH_MAX_CONCURRENT:-20}"
DRY_RUN=0

usage() {
    sed -n '3,22p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Required:
  -c, --config FILE     Benchmark config YAML
  -w, --workdir DIR     Output directory (use scratch, not \$HOME)

Optional:
  --env-setup FILE      Shell snippet sourced by every job (conda activate, module load)
  --partition NAME      SLURM partition
  --account NAME        SLURM account
  --threads N           CPUs per array task            [$THREADS]
  --run-time HH:MM:SS   Time limit per array task      [$RUN_TIME]
  --run-mem SIZE        Memory per array task          [$RUN_MEM]
  --max-concurrent N    Array throttle                 [$MAX_CONCURRENT]
  --dry-run             Print the sbatch commands without submitting
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)      CONFIG="$2"; shift 2 ;;
        -w|--workdir)     WORKDIR="$2"; shift 2 ;;
        --env-setup)      ENV_SETUP="$2"; shift 2 ;;
        --partition)      PARTITION="$2"; shift 2 ;;
        --account)        ACCOUNT="$2"; shift 2 ;;
        --threads)        THREADS="$2"; shift 2 ;;
        --run-time)       RUN_TIME="$2"; shift 2 ;;
        --run-mem)        RUN_MEM="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

[[ -n "$CONFIG"  ]] || { echo "ERROR: --config is required" >&2; usage; }
[[ -n "$WORKDIR" ]] || { echo "ERROR: --workdir is required" >&2; usage; }
[[ -f "$CONFIG"  ]] || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }

BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
mkdir -p "$WORKDIR"
WORKDIR="$(cd "$WORKDIR" && pwd)"
LOGDIR="$WORKDIR/slurm-logs"
mkdir -p "$LOGDIR"

if [[ -n "$ENV_SETUP" ]]; then
    [[ -f "$ENV_SETUP" ]] || { echo "ERROR: env setup not found: $ENV_SETUP" >&2; exit 1; }
    ENV_SETUP="$(cd "$(dirname "$ENV_SETUP")" && pwd)/$(basename "$ENV_SETUP")"
fi

# Count work units before submitting anything. This also fails fast on a config
# typo, which is much nicer than discovering it in 200 array task logs.
N_UNITS="$(cd "$BENCHMARK_DIR" && python -m spbench.cli plan -c "$CONFIG" --count)"
if [[ "$N_UNITS" -lt 1 ]]; then
    echo "ERROR: config produced no work units" >&2
    exit 1
fi

COMMON_ARGS=()
[[ -n "$PARTITION" ]] && COMMON_ARGS+=(--partition="$PARTITION")
[[ -n "$ACCOUNT"   ]] && COMMON_ARGS+=(--account="$ACCOUNT")

export SPBENCH_CONFIG="$CONFIG"
export SPBENCH_WORKDIR="$WORKDIR"
export SPBENCH_BENCHMARK_DIR="$BENCHMARK_DIR"
export SPBENCH_ENV_SETUP="$ENV_SETUP"

echo "============================================================"
echo "  config      : $CONFIG"
echo "  workdir     : $WORKDIR"
echo "  work units  : $N_UNITS  (array 0-$((N_UNITS - 1))%$MAX_CONCURRENT)"
echo "  env setup   : ${ENV_SETUP:-<none: relying on the current PATH>}"
echo "============================================================"

submit() {
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DRY RUN: sbatch $*" >&2
        echo "000000"
        return
    fi
    sbatch --parsable "$@"
}

SIM_ID=$(submit "${COMMON_ARGS[@]}" \
    --job-name=spb-sim \
    --time="$SIM_TIME" --mem="$SIM_MEM" --cpus-per-task=2 \
    --output="$LOGDIR/simulate-%j.log" \
    --export=ALL \
    "$BENCHMARK_DIR/scripts/slurm/simulate.sbatch")
echo "simulate  -> job $SIM_ID"

RUN_ID=$(submit "${COMMON_ARGS[@]}" \
    --job-name=spb-run \
    --dependency=afterok:"$SIM_ID" \
    --array="0-$((N_UNITS - 1))%$MAX_CONCURRENT" \
    --time="$RUN_TIME" --mem="$RUN_MEM" --cpus-per-task="$THREADS" \
    --output="$LOGDIR/run-%A_%a.log" \
    --export=ALL,SPBENCH_THREADS="$THREADS" \
    "$BENCHMARK_DIR/scripts/slurm/run_unit.sbatch")
echo "run array -> job $RUN_ID"

EVAL_ID=$(submit "${COMMON_ARGS[@]}" \
    --job-name=spb-eval \
    --dependency=afterany:"$RUN_ID" \
    --time="$EVAL_TIME" --mem="$EVAL_MEM" --cpus-per-task=2 \
    --output="$LOGDIR/evaluate-%j.log" \
    --export=ALL \
    "$BENCHMARK_DIR/scripts/slurm/evaluate.sbatch")
echo "evaluate  -> job $EVAL_ID"

cat <<EOF

Submitted. Watch with:
  squeue -u \$USER
  tail -f $LOGDIR/evaluate-*.log

When the last job finishes:
  $WORKDIR/results/report.md

If some array tasks failed, their rows appear in results/runs.tsv with
status=failed and the error message. Rerun just those indices:
  sbatch --array=<comma-separated indices> ... scripts/slurm/run_unit.sbatch
then resubmit the evaluate stage alone (or run: spbench evaluate -c CONFIG -w WORKDIR).
EOF
