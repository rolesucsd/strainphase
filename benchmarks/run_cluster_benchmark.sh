#!/bin/bash
###############################################################################
# Strainphase Cluster Benchmark Runner
#
# Submits all complexity levels as SLURM array jobs and optionally
# consolidates results when complete.
#
# Usage:
#   # Submit to SLURM:
#   ./benchmarks/run_cluster_benchmark.sh
#
#   # Run locally (sequential):
#   ./benchmarks/run_cluster_benchmark.sh --local
#
#   # Run locally in parallel (background jobs):
#   ./benchmarks/run_cluster_benchmark.sh --local --parallel
#
#   # Just consolidate existing results:
#   ./benchmarks/run_cluster_benchmark.sh --consolidate-only
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$SCRIPT_DIR")}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/results/cluster_benchmark}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

# Parse arguments
LOCAL_MODE=false
PARALLEL_MODE=false
CONSOLIDATE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --parallel)
            PARALLEL_MODE=true
            shift
            ;;
        --consolidate-only)
            CONSOLIDATE_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

consolidate_results() {
    echo ""
    echo "============================================================"
    echo "CONSOLIDATING RESULTS"
    echo "============================================================"

    SUMMARY_FILE="${OUTPUT_BASE}/consolidated_summary.json"

    python3 << EOF
import json
import os
from pathlib import Path

output_base = "${OUTPUT_BASE}"
    complexities = ["simple", "medium", "complex"]
    n_strains = [2, 4, 8]

consolidated = {
    "benchmark_suite": "strainphase_parameter_sweep",
    "complexity_levels": [],
    "best_params_overall": None,
    "best_f1_overall": 0.0
}

for complexity, n_strain in zip(complexities, n_strains):
    result_dir = Path(output_base) / complexity
    sweep_summary = result_dir / "sweep_results" / "sweep_summary.json"

    if not sweep_summary.exists():
        print(f"  {complexity}: NOT FOUND")
        continue

    with open(sweep_summary) as f:
        summary = json.load(f)

    # Extract key metrics
    scenarios = summary.get("scenarios", {})
    best = summary.get("best_params_by_scenario", {})

    level_summary = {
        "complexity": complexity,
        "n_strains": n_strain,
        "n_configs_tested": summary.get("n_configs_tested", 0),
        "scenarios": {}
    }

    for scenario_name, stats in scenarios.items():
        level_summary["scenarios"][scenario_name] = {
            "n_lineages_mean": stats.get("n_lineages", {}).get("mean", 0),
            "n_lineages_std": stats.get("n_lineages", {}).get("std", 0),
            "snv_f1": stats.get("snv_f1"),
            "haplotype_f1": stats.get("haplotype_f1"),
            "converged_fraction": stats.get("converged_fraction", 0),
            "runtime_mean": stats.get("runtime", {}).get("mean", 0)
        }

        # Track best overall
        if best.get(scenario_name, {}).get("snv_f1") or 0 > consolidated["best_f1_overall"]:
            consolidated["best_f1_overall"] = best[scenario_name].get("snv_f1") or 0
            consolidated["best_params_overall"] = {
                "complexity": complexity,
                "params": best[scenario_name].get("params")
            }

    consolidated["complexity_levels"].append(level_summary)
    print(f"  {complexity}: {summary.get('n_configs_tested', 0)} configs tested")

# Save consolidated summary
with open("${OUTPUT_BASE}/consolidated_summary.json", "w") as f:
    json.dump(consolidated, f, indent=2)

print(f"\nConsolidated summary: ${OUTPUT_BASE}/consolidated_summary.json")

# Print summary table
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"{'Complexity':<12} {'Strains':<8} {'Configs':<10} {'Lineages':<15} {'SNV F1':<10}")
print("-"*70)

for level in consolidated["complexity_levels"]:
    for scenario, stats in level.get("scenarios", {}).items():
        lineages = f"{stats['n_lineages_mean']:.1f} +/- {stats['n_lineages_std']:.1f}"
        snv_f1 = f"{stats['snv_f1']:.3f}" if stats['snv_f1'] else "N/A"
        print(f"{level['complexity']:<12} {level['n_strains']:<8} {level['n_configs_tested']:<10} {lineages:<15} {snv_f1:<10}")

print("="*70)

if consolidated["best_params_overall"]:
    print(f"\nBest overall SNV F1: {consolidated['best_f1_overall']:.3f}")
    print(f"At complexity: {consolidated['best_params_overall']['complexity']}")
EOF
}

if $CONSOLIDATE_ONLY; then
    consolidate_results
    exit 0
fi

echo "============================================================"
echo "STRAINPHASE CLUSTER BENCHMARK SUITE"
echo "============================================================"
echo "Output directory: $OUTPUT_BASE"
echo "Mode: $(if $LOCAL_MODE; then echo "Local"; else echo "SLURM"; fi)"
if $LOCAL_MODE && $PARALLEL_MODE; then
    echo "Parallel: Yes"
fi
echo "============================================================"

if $LOCAL_MODE; then
    # Run locally
    PIDS=()

    for level in 1 2 3; do
        echo ""
        echo "Starting complexity level $level..."

        if $PARALLEL_MODE; then
            # Run in background
            bash "${SCRIPT_DIR}/cluster_benchmark.sh" "$level" &
            PIDS+=($!)
            echo "  Started (PID: ${PIDS[-1]})"
        else
            # Run sequentially
            bash "${SCRIPT_DIR}/cluster_benchmark.sh" "$level"
        fi
    done

    if $PARALLEL_MODE; then
        echo ""
        echo "Waiting for all jobs to complete..."
        echo "PIDs: ${PIDS[*]}"

        FAILED=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                echo "Job $pid failed"
                ((FAILED++))
            fi
        done

        if [[ $FAILED -gt 0 ]]; then
            echo "WARNING: $FAILED jobs failed"
        fi
    fi

    # Consolidate results
    consolidate_results

else
    # Submit to SLURM
    echo ""
    echo "Submitting SLURM array job..."

    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/cluster_benchmark.sh")
    echo "Submitted job array: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f logs/benchmark_${JOB_ID}_*.out"
    echo ""
    echo "When complete, consolidate results with:"
    echo "  $0 --consolidate-only"
fi

echo ""
echo "============================================================"
echo "BENCHMARK SUBMISSION COMPLETE"
echo "============================================================"
