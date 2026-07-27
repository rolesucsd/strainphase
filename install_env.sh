#!/bin/bash
# =============================================================================
# strainphase — create the mamba env for branch feature/window-linking-rework
# =============================================================================
#
#   bash install_env.sh                 # creates env "strainphase-wlr"
#   bash install_env.sh myenvname       # or pick your own name
#
# Run this ONCE on the cluster (login node is fine — it only builds an env).
# Re-run it whenever the branch changes; the pip install is editable, so if you
# install from a git CLONE you only need `git pull` afterwards, not a reinstall.
# =============================================================================
set -euo pipefail

ENV_NAME="${1:-strainphase-wlr}"
BRANCH="feature/window-linking-rework"
REPO_URL="https://github.com/rolesucsd/strainphase.git"

# Where to clone. Change this if you want the source somewhere else.
SRC_DIR="${STRAINPHASE_SRC:-$HOME/src/strainphase-wlr}"

echo "=== strainphase env setup ==="
echo "  env    : ${ENV_NAME}"
echo "  branch : ${BRANCH}"
echo "  source : ${SRC_DIR}"
echo

# --- 1. mamba env -------------------------------------------------------------
# python >=3.11 is required by pyproject. pysam/samtools come from bioconda.
if mamba env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[1/3] env '${ENV_NAME}' already exists — skipping creation"
else
    echo "[1/3] creating env '${ENV_NAME}'"
    mamba create -y -n "${ENV_NAME}" \
        -c conda-forge -c bioconda \
        "python=3.11" \
        "numpy>=1.20" \
        "scipy>=1.7" \
        "networkx>=2.6" \
        "pandas>=1.3" \
        "python-louvain>=0.16" \
        "pysam>=0.19" \
        "samtools" \
        "bcftools" \
        "pytest>=7.0"
fi

# --- 2. source ----------------------------------------------------------------
if [ -d "${SRC_DIR}/.git" ]; then
    echo "[2/3] updating existing clone at ${SRC_DIR}"
    git -C "${SRC_DIR}" fetch origin
    git -C "${SRC_DIR}" checkout "${BRANCH}"
    git -C "${SRC_DIR}" pull --ff-only origin "${BRANCH}"
else
    echo "[2/3] cloning ${BRANCH} into ${SRC_DIR}"
    mkdir -p "$(dirname "${SRC_DIR}")"
    git clone --branch "${BRANCH}" "${REPO_URL}" "${SRC_DIR}"
fi

# --- 3. install ---------------------------------------------------------------
# Editable install so `git pull` is enough to pick up future changes. NOTE: an
# editable install points at SRC_DIR forever — if you delete or move that
# directory the env breaks.
echo "[3/3] installing strainphase (editable) into ${ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
pip install --no-deps -e "${SRC_DIR}"

echo
echo "=== verifying ==="
python -c "import strainphase, strainphase.window_groups, strainphase.coherence; print('strainphase', strainphase.__version__)"
python -c "
from strainphase.core import HaplotyperConfig
c = HaplotyperConfig()
print(f'  phase floor        {c.min_reads_per_window} reads')
print(f'  rescue floor       {c.min_reads_for_rescue} reads')
print(f'  read cap/window    {c.max_reads_per_window}')
print(f'  read-window overlap{c.min_read_window_overlap_bp:>5} bp')
print(f'  max_num_diff       {c.max_num_diff}')
print(f'  cross-sample method{c.cross_sample_method:>8}')
"
echo
echo "running the test suite (expect 145 passed)…"
( cd "${SRC_DIR}" && python -m pytest tests/ -q 2>&1 | tail -3 )

echo
echo "=== done ==="
echo "Activate with:  conda activate ${ENV_NAME}"
echo "Source lives at: ${SRC_DIR}"
