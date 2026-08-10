# Sourced by every sbatch script. Sets up the environment and sanity-checks the
# variables submit.sh exported.
set -euo pipefail

: "${SPBENCH_CONFIG:?submit.sh must export SPBENCH_CONFIG}"
: "${SPBENCH_WORKDIR:?submit.sh must export SPBENCH_WORKDIR}"
: "${SPBENCH_BENCHMARK_DIR:?submit.sh must export SPBENCH_BENCHMARK_DIR}"

if [[ -n "${SPBENCH_ENV_SETUP:-}" && -f "${SPBENCH_ENV_SETUP}" ]]; then
    # shellcheck disable=SC1090
    source "${SPBENCH_ENV_SETUP}"
fi

cd "$SPBENCH_BENCHMARK_DIR"

if ! python -c "import spbench" 2>/dev/null; then
    echo "ERROR: spbench is not importable in this job's environment." >&2
    echo "       Point --env-setup at a file that activates the right conda env;" >&2
    echo "       see scripts/slurm/env.sh.example." >&2
    exit 1
fi

echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+ task ${SLURM_ARRAY_TASK_ID}}"
echo "python    : $(command -v python)"
echo "config    : $SPBENCH_CONFIG"
echo "workdir   : $SPBENCH_WORKDIR"
echo "started   : $(date -Is)"
echo "---"
