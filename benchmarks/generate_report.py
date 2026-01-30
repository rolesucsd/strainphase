#!/usr/bin/env python3
"""
Generate benchmark report with figures and HTML summary.

This module is used by run_full_benchmark.py to generate final reports.
It can also be run standalone to regenerate reports from existing results.

Takes results from parameter_sweep.py and validation outputs to generate:
- Validation figures (haplotype accuracy, abundance correlation, etc.)
- Benchmarking figures (parameter heatmaps, sensitivity plots, etc.)
- HTML report with embedded figures and recommendations

Usage (standalone):
    python benchmarks/generate_report.py \
        --results benchmarks/sweep_results/ \
        --output benchmarks/report/

Note: For full benchmarking pipeline, use run_full_benchmark.py instead.
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import logging as _mpl_logging
    _mpl_logging.getLogger("matplotlib.font_manager").setLevel(_mpl_logging.ERROR)
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Ensure local project imports (e.g., validation) work when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Plot styling
# =============================================================================

# Professional color palette (used across report figures)
COLOR_PALETTE = {
    "primary": "#2C3E50",      # Dark blue-gray
    "secondary": "#34495E",    # Medium blue-gray
    "accent": "#3498DB",       # Bright blue
    "success": "#27AE60",      # Green
    "warning": "#F39C12",      # Orange
    "error": "#E74C3C",        # Red
    "info": "#9B59B6",         # Purple
    "neutral": "#95A5A6",      # Gray
    "light": "#ECF0F1",        # Light gray
    "dark": "#1A1A1A",         # Near black
}

# Color palette for scatter plot points
POINT_COLORS = ["#569667", "#4264a8", "#e6a432", "#8e4aa1"]

COLOR_SEQUENCES = {
    "qualitative": [
        "#2C3E50",
        "#3498DB",
        "#27AE60",
        "#F39C12",
        "#9B59B6",
        "#E74C3C",
        "#1ABC9C",
        "#E67E22",
    ],
}


def set_plot_style():
    """Set professional, clean figure style for report plots with Arial font and no grid."""
    if not HAS_MATPLOTLIB:
        return

    # Use clean style without grid
    plt.style.use("default")

    # Font settings - Arial family
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12  # Title size 12, not bold
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 16

    # Remove grid completely
    plt.rcParams["axes.grid"] = False

    # Clean spines - only show bottom and left
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True

    # Spine styling
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.edgecolor"] = COLOR_PALETTE["primary"]

    # Figure and axes background
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"

    # Tick styling
    plt.rcParams["xtick.color"] = COLOR_PALETTE["primary"]
    plt.rcParams["ytick.color"] = COLOR_PALETTE["primary"]
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    # Line and marker styling
    plt.rcParams["lines.linewidth"] = 2.0
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["patch.linewidth"] = 1.2

    # Legend styling
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.95
    plt.rcParams["legend.edgecolor"] = COLOR_PALETTE["neutral"]
    plt.rcParams["legend.facecolor"] = "white"

    # Default figure size
    plt.rcParams["figure.figsize"] = (9, 5.5)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


# =============================================================================
# Data loading
# =============================================================================

def load_sweep_results(results_dir: str) -> Tuple[List[Dict], Dict, List[Dict]]:
    """
    Load parameter sweep results.

    Returns: (results_list, summary_dict, stable_params_list)
    """
    results_path = Path(results_dir)

    results = []
    results_file = results_path / "sweep_results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)

    summary = {}
    summary_file = results_path / "sweep_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

    stable = []
    stable_file = results_path / "stable_parameters.json"
    if stable_file.exists():
        with open(stable_file) as f:
            stable = json.load(f)

    # Attach benchmark-level parameters (e.g., coverage) if available.
    summary_file = results_path.parent / "benchmark_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                benchmark_summary = json.load(f)
            params = benchmark_summary.get("parameters", {})
            for r in results:
                if r.get("coverage") is None and params.get("coverage") is not None:
                    r["coverage"] = params.get("coverage")
                if r.get("n_timepoints") is None and params.get("n_timepoints") is not None:
                    r["n_timepoints"] = params.get("n_timepoints")
        except (OSError, json.JSONDecodeError):
            pass

    return results, summary, stable


def load_validation_metrics(validation_dir: str) -> Optional[Dict]:
    """Load validation metrics if available."""
    metrics_file = Path(validation_dir) / "validation_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    
    # Also try loading from config-specific validation directories
    # (for parameter sweep with per-config validation)
    search_roots = [
        Path(validation_dir).parent,
        Path(validation_dir),
        Path(validation_dir) / "sweep_results",
        Path(validation_dir).parent / "sweep_results",
    ]
    config_dirs = []
    for root in search_roots:
        config_dirs.extend(list(root.glob("configs/*/validation")))
    if config_dirs:
        # Load from best config (highest F1) or first available
        best_metrics = None
        best_f1 = -1
        for config_dir in config_dirs:
            config_metrics_file = config_dir / "validation_metrics.json"
            if config_metrics_file.exists():
                with open(config_metrics_file) as f:
                    metrics = json.load(f)
                    f1 = metrics.get("f1", 0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_metrics = metrics
        return best_metrics
    
    return None


def _short_name_from_params(params: Dict) -> Optional[str]:
    try:
        return (
            f"mm{float(params['max_mismatch_frac']):.3f}_"
            f"mq{int(params['min_mapq'])}_"
            f"bq{int(params['min_base_quality'])}_"
            f"snv{int(params['min_shared_snvs_for_edge'])}_"
            f"md{float(params['merge_distance_threshold']):.3f}_"
            f"aw{float(params['min_weight_for_anchor']):.2f}_"
            f"rw{float(params['rescued_min_weight']):.2f}_"
            f"ws{int(params['window_size'])}"
        )
    except Exception:
        return None


def _top_configs_by_f1(results: List[Dict], n: int) -> List[Dict]:
    ranked = [r for r in results if r.get("haplotype_f1") is not None]
    ranked.sort(key=lambda r: r.get("haplotype_f1", -1), reverse=True)
    return ranked[:n]


def _config_validation_dir(results_dir: str, result: Dict) -> Optional[Path]:
    params = result.get("params") or {}
    config_name = _short_name_from_params(params)
    if not config_name:
        return None
    config_dir = Path(results_dir) / "configs" / config_name / "validation"
    return config_dir if config_dir.exists() else None


def _config_lineages_path(results_dir: str, result: Dict) -> Optional[Path]:
    params = result.get("params") or {}
    config_name = _short_name_from_params(params)
    if not config_name:
        return None
    base_dir = Path(results_dir) / "configs" / config_name
    # Prefer validation/lineages.tsv if present, else fall back to config root.
    candidate = base_dir / "validation" / "lineages.tsv"
    if candidate.exists():
        return candidate
    candidate = base_dir / "lineages.tsv"
    if candidate.exists():
        return candidate
    return None


def _load_lineage_details(path: Path) -> List[Dict]:
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(row)
    return records


def _load_lineages(path: Path) -> List[Dict]:
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(row)
    return records


def _load_truth_snvs(truth_dir: str) -> Dict[str, set]:
    truth_vcf = Path(truth_dir) / "truth_snvs.vcf"
    if not truth_vcf.exists():
        truth_vcf = Path(truth_dir) / "truth_variants.vcf"
    if not truth_vcf.exists():
        return {}
    truth = defaultdict(set)
    with open(truth_vcf) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            contig = parts[0]
            pos = int(parts[1])
            truth[contig].add(pos)
    return dict(truth)


def _load_truth_tracks(truth_dir: str) -> Dict[str, Dict[str, tuple[int, int]]]:
    tracks_file = Path(truth_dir) / "truth_tracks.tsv"
    if not tracks_file.exists():
        return {}
    tracks = defaultdict(dict)
    with open(tracks_file) as f:
        header = f.readline().strip().split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            parts = line.strip().split("\t")
            strain_id = parts[idx["strain_id"]]
            contig = parts[idx["contig"]]
            start = int(parts[idx["start"]])
            end = int(parts[idx["end"]])
            tracks[strain_id][contig] = (start, end)
    return dict(tracks)


def generate_validation_patchwork(
    results_dir: str,
    results: List[Dict],
    truth_dir: Optional[str],
    output_dir: str
) -> Optional[str]:
    """Generate a patchwork validation figure for the best parameter set."""
    if not HAS_MATPLOTLIB:
        return None
    if not truth_dir:
        logger.warning("No truth_dir provided; skipping validation patchwork.")
        return None

    # Pick best config by haplotype_f1
    best = None
    best_f1 = -1.0
    for r in results:
        f1 = r.get("haplotype_f1")
        if f1 is None:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best = r
    if best is None:
        best = results[0]

    config_name = _short_name_from_params(best.get("params", {}))
    if not config_name:
        logger.warning("Could not derive config name; skipping validation patchwork.")
        return None

    config_dir = Path(results_dir) / "configs" / config_name
    lineages_path = config_dir / "lineages.tsv"
    if not lineages_path.exists():
        logger.warning(f"Missing lineages.tsv for {config_name}; skipping validation patchwork.")
        return None

    try:
        from validation.validate_haplotypes import (
            load_ground_truth,
            load_detected_haplotypes,
            match_haplotypes,
            compute_haplotype_distance,
            compute_validation_metrics,
        )
        from validation.validate_tracks import load_truth_tracks
    except Exception as e:
        logger.warning(f"Failed to import validation helpers: {e}")
        return None

    true_haps, all_snv_positions = load_ground_truth(truth_dir)
    detected_haps = load_detected_haplotypes(str(lineages_path))
    matches = match_haplotypes(true_haps, detected_haps, allow_one_to_many=True)
    validation_result = compute_validation_metrics(
        true_haps, detected_haps, all_snv_positions
    )

    # Build figure
    n_contigs = len({c for h in detected_haps for c in h.snv_alleles.keys()})
    fig_height = 10 + max(0, n_contigs - 2) * 1.2
    fig = plt.figure(figsize=(15, fig_height))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.3)

    # Panel A: Abundance correlation
    ax_abund = fig.add_subplot(gs[0, 0])
    true_abundances = []
    detected_abundances = []
    for true_hap, det_hap, _ in matches:
        common_tps = set(true_hap.abundances.keys()) & set(det_hap.abundances.keys())
        for tp in common_tps:
            true_abundances.append(true_hap.abundances[tp])
            detected_abundances.append(det_hap.abundances[tp])
    if true_abundances:
        ax_abund.scatter(
            true_abundances, detected_abundances, alpha=0.7, s=40,
            color=COLOR_PALETTE['accent'], edgecolors=COLOR_PALETTE['primary'], linewidths=0.6
        )
        max_val = max(true_abundances) if true_abundances else 1.0
        ax_abund.plot([0, max_val], [0, max_val],
                      color=COLOR_PALETTE['error'], linestyle='--', linewidth=1.5, alpha=0.8)
        if len(true_abundances) >= 2:
            r = np.corrcoef(true_abundances, detected_abundances)[0, 1]
            ax_abund.text(0.05, 0.95, f"r = {r:.3f}",
                          transform=ax_abund.transAxes, va='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_abund.set_title("Abundance Correlation", color=COLOR_PALETTE['primary'])
    ax_abund.set_xlabel("True Abundance", color=COLOR_PALETTE['primary'])
    ax_abund.set_ylabel("Detected Abundance", color=COLOR_PALETTE['primary'])

    # Panel B: Reference coverage
    ax_cov = fig.add_subplot(gs[0, 1])
    coverages = []
    for true_hap, det_hap, _ in matches:
        total_true_snvs = 0
        recovered_snvs = 0
        for contig in det_hap.snv_alleles.keys():
            true_snvs = true_hap.snv_positions.get(contig, {})
            det_snvs = det_hap.snv_alleles.get(contig, {})
            total_true_snvs += len(true_snvs)
            for pos, true_allele in true_snvs.items():
                if pos in det_snvs and det_snvs[pos] == true_allele:
                    recovered_snvs += 1
        if total_true_snvs > 0:
            coverages.append(recovered_snvs / total_true_snvs)
    if coverages:
        bins = np.linspace(0, 1.0, 21)
        ax_cov.hist(coverages, bins=bins, color=COLOR_PALETTE['accent'], alpha=0.7, edgecolor='none')
        mean_cov = np.mean(coverages)
        ax_cov.axvline(mean_cov, color=COLOR_PALETTE['error'], linestyle='--',
                       linewidth=1.8, label=f"Mean: {mean_cov:.3f}", alpha=0.8)
        ax_cov.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
    ax_cov.set_title("Reference Coverage", color=COLOR_PALETTE['primary'])
    ax_cov.set_xlabel("Fraction of SNVs Recovered", color=COLOR_PALETTE['primary'])
    ax_cov.set_ylabel("Number of Strains", color=COLOR_PALETTE['primary'])

    # Panel C: Detailed matching (2x2)
    sub_gs = gs[1, 0].subgridspec(2, 2, wspace=0.35, hspace=0.35)
    ax_dm1 = fig.add_subplot(sub_gs[0, 0])
    ax_dm2 = fig.add_subplot(sub_gs[0, 1])
    ax_dm3 = fig.add_subplot(sub_gs[1, 0])
    ax_dm4 = fig.add_subplot(sub_gs[1, 1])

    match_details = []
    for true_hap, det_hap, _ in matches:
        dist, n_matches, n_shared, _ = compute_haplotype_distance(true_hap, det_hap)
        common_tps = set(true_hap.abundances.keys()) & set(det_hap.abundances.keys())
        abund_errors = []
        for tp in common_tps:
            abund_errors.append(abs(true_hap.abundances[tp] - det_hap.abundances[tp]))
        match_details.append({
            "snv_match_fraction": n_matches / n_shared if n_shared > 0 else 0.0,
            "abundance_mae": np.mean(abund_errors) if abund_errors else None,
            "n_detected_snvs": sum(len(snvs) for snvs in det_hap.snv_alleles.values()),
            "true_id": true_hap.strain_id,
            "det_id": det_hap.lineage_id,
        })

    snv_fractions = [m["snv_match_fraction"] for m in match_details]
    ax_dm1.hist(snv_fractions, bins=20, color=COLOR_PALETTE['accent'], alpha=0.7, edgecolor='none')
    ax_dm1.set_title("SNV Match Fraction", color=COLOR_PALETTE['primary'])
    ax_dm1.set_xlabel("Match Fraction", color=COLOR_PALETTE['primary'])
    ax_dm1.set_ylabel("Pairs", color=COLOR_PALETTE['primary'])

    abund_errors = [m["abundance_mae"] for m in match_details if m["abundance_mae"] is not None]
    if abund_errors:
        ax_dm2.hist(abund_errors, bins=20, color=COLOR_PALETTE['muted'], alpha=0.7, edgecolor='none')
    ax_dm2.set_title("Abundance MAE", color=COLOR_PALETTE['primary'])
    ax_dm2.set_xlabel("MAE", color=COLOR_PALETTE['primary'])
    ax_dm2.set_ylabel("Pairs", color=COLOR_PALETTE['primary'])

    true_counts = []
    detected_counts = []
    for m in match_details:
        true_hap = next((h for h in true_haps if h.strain_id == m["true_id"]), None)
        det_hap = next((h for h in detected_haps if h.lineage_id == m["det_id"]), None)
        if true_hap and det_hap:
            true_count = 0
            for contig in det_hap.snv_alleles.keys():
                true_count += len(true_hap.snv_positions.get(contig, {}))
            true_counts.append(true_count)
            detected_counts.append(m["n_detected_snvs"])
    if true_counts:
        ax_dm3.scatter(true_counts, detected_counts, alpha=0.7, s=30,
                       color=COLOR_PALETTE['accent'], edgecolors=COLOR_PALETTE['primary'], linewidths=0.5)
        max_count = max(max(true_counts), max(detected_counts))
        ax_dm3.plot([0, max_count], [0, max_count], color=COLOR_PALETTE['error'], linestyle='--', linewidth=1.2)
    ax_dm3.set_title("SNV Counts", color=COLOR_PALETTE['primary'])
    ax_dm3.set_xlabel("True SNVs (detected contigs)", color=COLOR_PALETTE['primary'])
    ax_dm3.set_ylabel("Detected SNVs", color=COLOR_PALETTE['primary'])

    if validation_result.per_timepoint_metrics:
        tps = sorted(validation_result.per_timepoint_metrics.keys())
        recalls = [validation_result.per_timepoint_metrics[tp]['recall'] for tp in tps]
        precisions = [validation_result.per_timepoint_metrics[tp]['precision'] for tp in tps]
        x = np.arange(len(tps))
        width = 0.35
        ax_dm4.bar(x - width/2, recalls, width, label='Recall', alpha=0.7, color=COLOR_PALETTE['accent'])
        ax_dm4.bar(x + width/2, precisions, width, label='Precision', alpha=0.7, color=COLOR_PALETTE['muted'])
        ax_dm4.set_xticks(x)
        ax_dm4.set_xticklabels(tps, color=COLOR_PALETTE['primary'])
        ax_dm4.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
    ax_dm4.set_title("Precision/Recall by Timepoint", color=COLOR_PALETTE['primary'])
    ax_dm4.set_ylabel("Score", color=COLOR_PALETTE['primary'])
    ax_dm4.set_ylim(0, 1.1)

    # Panel D: Track regions (per contig)
    track_gs = gs[1, 1].subgridspec(
        max(1, int(np.ceil(n_contigs / 2))), min(2, max(1, n_contigs)), wspace=0.3, hspace=0.5
    )
    track_axes = track_gs.subplots()
    if hasattr(track_axes, "flatten"):
        track_axes = track_axes.flatten()
    else:
        track_axes = [track_axes]
    truth_tracks = load_truth_tracks(truth_dir)

    # Build detected tracks from SNV spans
    detected_tracks: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)
    for det_hap in detected_haps:
        for contig, det_snvs in det_hap.snv_alleles.items():
            if not det_snvs:
                continue
            detected_tracks[det_hap.lineage_id][contig] = (min(det_snvs.keys()), max(det_snvs.keys()))

    track_to_strain = {}
    for true_hap, det_hap, _ in matches:
        if det_hap.lineage_id:
            track_to_strain[det_hap.lineage_id] = true_hap.strain_id

    all_contigs = sorted({c for tracks in list(truth_tracks.values()) + list(detected_tracks.values()) for c in tracks.keys()})
    for idx, contig in enumerate(all_contigs):
        if idx >= len(track_axes):
            break
        ax = track_axes[idx]
        max_pos = 0
        for tracks_dict in list(truth_tracks.values()) + list(detected_tracks.values()):
            if contig in tracks_dict:
                _, end = tracks_dict[contig]
                max_pos = max(max_pos, end)
        if max_pos == 0:
            ax.axis("off")
            continue

        y_offset = 0
        for strain_id, contig_dict in truth_tracks.items():
            if contig in contig_dict:
                start, end = contig_dict[contig]
                ax.barh(y_offset, end - start, left=start, height=0.6,
                        color=COLOR_PALETTE['light'], alpha=0.4)
                y_offset += 1

        for track_id, contig_dict in detected_tracks.items():
            if contig in contig_dict:
                start, end = contig_dict[contig]
                matched_strain = track_to_strain.get(track_id)
                color = COLOR_PALETTE['neutral']
                if matched_strain and matched_strain in truth_tracks:
                    color = COLOR_PALETTE['accent']
                ax.barh(y_offset, end - start, left=start, height=0.4, color=color, alpha=0.7)
                y_offset += 1

        ax.set_title(contig, color=COLOR_PALETTE['primary'])
        ax.set_xlim(0, max_pos * 1.05)
        ax.set_yticks([])
        ax.set_xlabel("Position (bp)", color=COLOR_PALETTE['primary'])

    for idx in range(len(all_contigs), len(track_axes)):
        track_axes[idx].axis("off")

    fig.suptitle(f"Validation Patchwork (best config: {config_name})", fontsize=14, color=COLOR_PALETTE['primary'])
    out_path = os.path.join(output_dir, "validation_patchwork.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _make_patchwork_axes(n_panels: int, cols: int, rows: int, title: str) -> Tuple[Any, List[Any]]:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.8))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i in range(rows * cols):
        if i >= n_panels:
            axes_list[i].axis("off")
    fig.suptitle(title, fontsize=14, color=COLOR_PALETTE["primary"])
    return fig, axes_list


def _short_param_label(params: Dict) -> str:
    """Compact label for patchwork panels."""
    if not params:
        return "cfg"
    ws = params.get("window_size")
    mm = params.get("max_mismatch_frac")
    mq = params.get("min_mapq")
    bq = params.get("min_base_quality")
    parts = []
    if ws is not None:
        parts.append(f"ws{int(ws)}")
    if mm is not None:
        parts.append(f"mm{float(mm):.3f}")
    if mq is not None:
        parts.append(f"mq{int(mq)}")
    if bq is not None:
        parts.append(f"bq{int(bq)}")
    return " ".join(parts) if parts else "cfg"


def generate_abundance_correlation_patchwork(
    results_dir: str,
    results: List[Dict],
    output_dir: str,
    top_n: int = 8,
    cols: int = 4,
    rows: int = 2,
) -> Optional[str]:
    top_configs = _top_configs_by_f1(results, top_n)
    if not top_configs:
        return None
    fig, axes = _make_patchwork_axes(len(top_configs), cols, rows, "Abundance Correlation (Top F1)")
    for idx, res in enumerate(top_configs):
        ax = axes[idx]
        config_dir = _config_validation_dir(results_dir, res)
        if not config_dir:
            ax.text(0.5, 0.5, "missing config", ha="center", va="center")
            ax.axis("off")
            continue
        details_path = config_dir / "lineage_details.tsv"
        if not details_path.exists():
            ax.text(0.5, 0.5, "missing lineage_details", ha="center", va="center")
            ax.axis("off")
            continue
        rows_data = _load_lineage_details(details_path)
        xs = []
        ys = []
        for row in rows_data:
            if row.get("matched_strain") == "UNMATCHED":
                continue
            try:
                xs.append(float(row["true_abundance"]))
                ys.append(float(row["detected_abundance"]))
            except (TypeError, ValueError, KeyError):
                continue
        if xs and ys:
            ax.scatter(xs, ys, s=8, alpha=0.7, color=COLOR_PALETTE["accent"])
            lo = min(xs + ys)
            hi = max(xs + ys)
            ax.plot([lo, hi], [lo, hi], color=COLOR_PALETTE["neutral"], linewidth=0.8)
        ax.set_xlabel("True", fontsize=7)
        ax.set_ylabel("Detected", fontsize=7)
        ax.tick_params(labelsize=6)
        params = res.get("params") or {}
        name = _short_param_label(params)
        f1 = res.get("haplotype_f1")
        f1_str = f"{f1:.2f}" if f1 is not None else "n/a"
        ax.set_title(f"{name}\nF1={f1_str}", fontsize=7)
    out_path = os.path.join(output_dir, "abundance_correlation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_reference_coverage_patchwork(
    results_dir: str,
    results: List[Dict],
    truth_dir: Optional[str],
    output_dir: str,
    top_n: int = 8,
    cols: int = 4,
    rows: int = 2,
) -> Optional[str]:
    if not truth_dir:
        logger.warning("No truth_dir provided; skipping reference coverage patchwork.")
        return None
    truth_snvs = _load_truth_snvs(truth_dir)
    if not truth_snvs:
        logger.warning("No truth SNVs found; skipping reference coverage patchwork.")
        return None
    top_configs = _top_configs_by_f1(results, top_n)
    if not top_configs:
        return None
    fig, axes = _make_patchwork_axes(len(top_configs), cols, rows, "Reference Coverage (Top F1)")
    for idx, res in enumerate(top_configs):
        ax = axes[idx]
        config_dir = _config_validation_dir(results_dir, res)
        if not config_dir:
            ax.text(0.5, 0.5, "missing config", ha="center", va="center")
            ax.axis("off")
            continue
        lineages_path = _config_lineages_path(results_dir, res)
        if not lineages_path or not lineages_path.exists():
            ax.text(0.5, 0.5, "missing lineages", ha="center", va="center")
            ax.axis("off")
            continue
        detected_pos = defaultdict(set)
        for row in _load_lineages(lineages_path):
            contig = row.get("contig")
            snv_alleles = row.get("snv_alleles", "")
            if not contig or not snv_alleles:
                continue
            for entry in snv_alleles.split(","):
                try:
                    pos = int(entry.split(":")[0])
                except Exception:
                    continue
                detected_pos[contig].add(pos)
        contigs = sorted(truth_snvs.keys())
        coverages = []
        for contig in contigs:
            truth_set = truth_snvs.get(contig, set())
            if not truth_set:
                coverages.append(0.0)
                continue
            recovered = len(truth_set & detected_pos.get(contig, set()))
            coverages.append(recovered / len(truth_set))
        ax.bar(range(len(contigs)), coverages, color=COLOR_PALETTE["accent"], alpha=0.8)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([])
        ax.set_ylabel("Coverage", fontsize=7)
        params = res.get("params") or {}
        name = _short_param_label(params)
        f1 = res.get("haplotype_f1")
        f1_str = f"{f1:.2f}" if f1 is not None else "n/a"
        ax.set_title(f"{name}\nF1={f1_str}", fontsize=7)
    out_path = os.path.join(output_dir, "reference_coverage.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_track_regions_patchwork(
    results_dir: str,
    results: List[Dict],
    truth_dir: Optional[str],
    output_dir: str,
    top_n: int = 8,
    cols: int = 4,
    rows: int = 2,
) -> Optional[str]:
    if not truth_dir:
        logger.warning("No truth_dir provided; skipping track regions patchwork.")
        return None
    truth_tracks = _load_truth_tracks(truth_dir)
    if not truth_tracks:
        logger.warning("No truth tracks found; skipping track regions patchwork.")
        return None
    top_configs = _top_configs_by_f1(results, top_n)
    if not top_configs:
        return None
    fig, axes = _make_patchwork_axes(len(top_configs), cols, rows, "Track Regions (Top F1)")
    for idx, res in enumerate(top_configs):
        ax = axes[idx]
        config_dir = _config_validation_dir(results_dir, res)
        if not config_dir:
            ax.text(0.5, 0.5, "missing config", ha="center", va="center")
            ax.axis("off")
            continue
        lineages_path = _config_lineages_path(results_dir, res)
        if not lineages_path or not lineages_path.exists():
            ax.text(0.5, 0.5, "missing lineages", ha="center", va="center")
            ax.axis("off")
            continue
        detected_tracks = defaultdict(list)
        for row in _load_lineages(lineages_path):
            contig = row.get("contig")
            snv_alleles = row.get("snv_alleles", "")
            if not contig or not snv_alleles:
                continue
            positions = []
            for entry in snv_alleles.split(","):
                try:
                    positions.append(int(entry.split(":")[0]))
                except Exception:
                    continue
            if positions:
                detected_tracks[contig].append((min(positions), max(positions)))
        contigs = sorted({c for tracks in list(truth_tracks.values()) for c in tracks.keys()} | set(detected_tracks.keys()))
        if not contigs:
            ax.text(0.5, 0.5, "no contigs", ha="center", va="center")
            ax.axis("off")
            continue
        y_base = 0.0
        for contig in contigs:
            offsets = 0
            for strain_id, contig_dict in truth_tracks.items():
                if contig in contig_dict:
                    start, end = contig_dict[contig]
                    ax.hlines(y_base + 0.15 + offsets * 0.04, start, end, color="#888888", linewidth=1, alpha=0.7)
                    offsets += 1
            for span in detected_tracks.get(contig, []):
                ax.hlines(y_base - 0.15, span[0], span[1], color=COLOR_PALETTE["primary"], linewidth=1.2, alpha=0.8)
            ax.text(0, y_base + 0.35, contig, fontsize=6, color=COLOR_PALETTE["neutral"])
            y_base += 1.0
        ax.set_yticks([])
        ax.set_xlabel("Position", fontsize=7)
        ax.tick_params(axis="x", labelsize=6)
        params = res.get("params") or {}
        name = _short_param_label(params)
        f1 = res.get("haplotype_f1")
        f1_str = f"{f1:.2f}" if f1 is not None else "n/a"
        ax.set_title(f"{name}\nF1={f1_str}", fontsize=7)
    out_path = os.path.join(output_dir, "track_regions.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _safe_metric_list(results: List[Dict], key: str) -> List[float]:
    vals = []
    for r in results:
        v = r.get(key)
        vals.append(0.0 if v is None else v)
    return vals


def generate_haplotype_accuracy_summary(results: List[Dict], output_dir: str) -> str:
    """Bar summary (per parameter set) for haplotype precision/recall/F1."""
    config_names = [r.get("config_name") or _short_name_from_params(r.get("params", {})) or f"cfg_{i+1}"
                    for i, r in enumerate(results)]
    precision = _safe_metric_list(results, "haplotype_precision")
    recall = _safe_metric_list(results, "haplotype_recall")
    f1 = _safe_metric_list(results, "haplotype_f1")

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(results) * 0.25), 9), sharex=True)
    metrics = [("Precision", precision), ("Recall", recall), ("F1", f1)]
    colors = [COLOR_PALETTE['accent'], COLOR_PALETTE['muted'], COLOR_PALETTE['primary']]

    x = np.arange(len(results))
    for ax, (label, vals), color in zip(axes, metrics, colors):
        ax.bar(x, vals, color=color, alpha=0.8)
        ax.set_ylabel(label, color=COLOR_PALETTE['primary'])
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.15)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(config_names, rotation=90, fontsize=7, color=COLOR_PALETTE['primary'])
    axes[0].set_title("Haplotype Accuracy by Parameter Set", color=COLOR_PALETTE['primary'])

    plt.tight_layout()
    out_path = os.path.join(output_dir, "haplotype_accuracy_summary.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def generate_lineage_accuracy_summary(results: List[Dict], output_dir: str) -> str:
    """Bar summary (per parameter set) for lineage precision/recall/F1."""
    config_names = [r.get("config_name") or _short_name_from_params(r.get("params", {})) or f"cfg_{i+1}"
                    for i, r in enumerate(results)]
    precision = _safe_metric_list(results, "lineage_precision")
    recall = _safe_metric_list(results, "lineage_recall")
    f1 = _safe_metric_list(results, "lineage_f1")

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(results) * 0.25), 9), sharex=True)
    metrics = [("Precision", precision), ("Recall", recall), ("F1", f1)]
    colors = [COLOR_PALETTE['accent'], COLOR_PALETTE['muted'], COLOR_PALETTE['primary']]

    x = np.arange(len(results))
    for ax, (label, vals), color in zip(axes, metrics, colors):
        ax.bar(x, vals, color=color, alpha=0.8)
        ax.set_ylabel(label, color=COLOR_PALETTE['primary'])
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.15)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(config_names, rotation=90, fontsize=7, color=COLOR_PALETTE['primary'])
    axes[0].set_title("Lineage Accuracy by Parameter Set", color=COLOR_PALETTE['primary'])

    plt.tight_layout()
    out_path = os.path.join(output_dir, "lineage_accuracy_summary.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def generate_linking_errors_summary(results: List[Dict], output_dir: str) -> str:
    """Bar summary (per parameter set) for linking errors."""
    config_names = [r.get("config_name") or _short_name_from_params(r.get("params", {})) or f"cfg_{i+1}"
                    for i, r in enumerate(results)]
    false_link = _safe_metric_list(results, "false_link_rate")
    missed_link = _safe_metric_list(results, "missed_link_rate")

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(results) * 0.25), 7), sharex=True)
    metrics = [("False Link Rate", false_link), ("Missed Link Rate", missed_link)]
    colors = [COLOR_PALETTE['error'], COLOR_PALETTE['muted']]

    x = np.arange(len(results))
    for ax, (label, vals), color in zip(axes, metrics, colors):
        ax.bar(x, vals, color=color, alpha=0.8)
        ax.set_ylabel(label, color=COLOR_PALETTE['primary'])
        ax.set_ylim(0, max(vals) * 1.1 if vals else 1.0)
        ax.grid(axis='y', alpha=0.15)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(config_names, rotation=90, fontsize=7, color=COLOR_PALETTE['primary'])
    axes[0].set_title("Linking Errors by Parameter Set", color=COLOR_PALETTE['primary'])

    plt.tight_layout()
    out_path = os.path.join(output_dir, "linking_errors_summary.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def generate_error_breakdown_summary(results: List[Dict], output_dir: str) -> str:
    """Bar summary (per parameter set) for false negatives and false positives."""
    config_names = [r.get("config_name") or _short_name_from_params(r.get("params", {})) or f"cfg_{i+1}"
                    for i, r in enumerate(results)]
    fn = []
    fp = []
    for r in results:
        fn.append(r.get("false_negatives_count") or 0)
        fp.append(r.get("false_positives_count") or 0)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(results) * 0.25), 7), sharex=True)
    metrics = [("False Negatives", fn), ("False Positives", fp)]
    colors = [COLOR_PALETTE['muted'], COLOR_PALETTE['error']]

    x = np.arange(len(results))
    for ax, (label, vals), color in zip(axes, metrics, colors):
        ax.bar(x, vals, color=color, alpha=0.85)
        ax.set_ylabel(label, color=COLOR_PALETTE['primary'])
        ax.grid(axis='y', alpha=0.15)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(config_names, rotation=90, fontsize=7, color=COLOR_PALETTE['primary'])
    axes[0].set_title("Error Breakdown by Parameter Set", color=COLOR_PALETTE['primary'])

    plt.tight_layout()
    out_path = os.path.join(output_dir, "error_breakdown_summary.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


# =============================================================================
# Figure generation
# =============================================================================

# Professional color palette
COLOR_PALETTE = {
    'primary': '#1F2933',      # Charcoal
    'secondary': '#3E4C59',    # Slate
    'accent': '#4B7F9D',       # Muted blue
    'muted': '#7A8A9A',        # Muted blue-gray
    'success': '#5B8A72',      # Muted green
    'warning': '#7C8AA5',      # Cool steel
    'error': '#6B7280',        # Cool gray
    'info': '#6A8EAE',         # Dusty blue
    'neutral': '#9AA5B1',      # Soft gray
    'light': '#E4E7EB',        # Light gray
    'dark': '#111827',         # Near black
}

# Professional color sequences for multi-series plots
COLOR_SEQUENCES = {
    'qualitative': [
        '#4B7F9D', '#5B8A72', '#6A8EAE', '#7C8AA5',
        '#8AA1B1', '#65748B', '#3E4C59', '#9AA5B1'
    ],
    'sequential': ['#E4E7EB', '#CBD2D9', '#9AA5B1', '#7B8794', '#52606D', '#3E4C59'],
    'diverging': ['#6B7280', '#7C8AA5', '#9AA5B1', '#5B8A72', '#4B7F9D'],
}

def set_figure_style():
    """Set Nature-style figure defaults for publication-quality plots."""
    if not HAS_MATPLOTLIB:
        return

    plt.style.use('default')

    # Font settings (Nature-style sans)
    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.sans-serif'] = [
        'Helvetica',
        'Arial',
        'DejaVu Sans',
        'Liberation Sans',
        'sans-serif',
    ]
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14

    # No gridlines
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.grid.axis'] = 'both'

    # Spines
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.edgecolor'] = '#1F2933'
    plt.rcParams['axes.linewidth'] = 0.9

    # Figure and axes background
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.edgecolor'] = 'none'

    # Tick styling
    plt.rcParams['xtick.color'] = '#1F2933'
    plt.rcParams['ytick.color'] = '#1F2933'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['ytick.minor.width'] = 0.5
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4

    # Line and marker styling
    plt.rcParams['lines.linewidth'] = 1.8
    plt.rcParams['lines.markersize'] = 5
    plt.rcParams['patch.linewidth'] = 0.8
    plt.rcParams['patch.edgecolor'] = '#1F2933'

    # Legend styling
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.95
    plt.rcParams['legend.edgecolor'] = '#CBD2D9'
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.borderpad'] = 0.4
    plt.rcParams['legend.labelspacing'] = 0.4

    # Figure sizing
    plt.rcParams['figure.figsize'] = (7.0, 4.5)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.bbox'] = 'tight'

    # Color cycle (colorblind-friendly)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLOR_SEQUENCES['qualitative'])


def _select_metric(results: List[Dict]) -> str:
    """Pick the best available metric for benchmarking plots."""
    if not results:
        return "n_lineages"
    metric_candidates = ["haplotype_f1", "snv_f1", "n_lineages"]
    for candidate in metric_candidates:
        if any(r.get(candidate) is not None for r in results):
            return candidate
    return "n_lineages"


def generate_parameter_heatmap(
    results: List[Dict],
    output_dir: str,
    metric: Optional[str] = None
) -> str:
    """
    Generate heatmap of metric across parameter combinations.

    Returns path to saved figure.
    """
    if not HAS_MATPLOTLIB:
        return ""
    if not results:
        raise ValueError("Parameter heatmap requires sweep results.")

    if metric is None:
        metric = _select_metric(results)

    # Group results by two key parameters for heatmap
    param1 = 'max_mismatch_frac'
    param2 = 'min_shared_snvs_for_edge'

    # Build matrix
    param1_vals = sorted(set(r['params'].get(param1, 0) for r in results))
    param2_vals = sorted(set(r['params'].get(param2, 0) for r in results))

    matrix = np.zeros((len(param2_vals), len(param1_vals)))
    counts = np.zeros((len(param2_vals), len(param1_vals)))

    for r in results:
        p1 = r['params'].get(param1, 0)
        p2 = r['params'].get(param2, 0)
        val = r.get(metric, r.get('n_lineages', 0))

        if p1 in param1_vals and p2 in param2_vals:
            i = param2_vals.index(p2)
            j = param1_vals.index(p1)
            matrix[i, j] += val
            counts[i, j] += 1

    # Average where multiple results
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.where(counts > 0, matrix / counts, 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Use professional colormap
    im = ax.imshow(matrix, cmap='viridis', aspect='auto', interpolation='nearest')

    ax.set_xticks(range(len(param1_vals)))
    ax.set_xticklabels([f"{v:.3f}" for v in param1_vals], rotation=45, ha='right',
                       fontsize=10, color=COLOR_PALETTE['primary'])
    ax.set_yticks(range(len(param2_vals)))
    ax.set_yticklabels([str(v) for v in param2_vals],
                       fontsize=10, color=COLOR_PALETTE['primary'])

    ax.set_xlabel('Max Mismatch Fraction', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_ylabel('Min Shared SNVs', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_title(f'Parameter Heatmap: {metric}', fontweight='bold', color=COLOR_PALETTE['primary'])

    # Add colorbar with professional styling
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric, fontsize=12, fontweight='bold', color=COLOR_PALETTE['primary'])
    cbar.ax.tick_params(labelsize=10, colors=COLOR_PALETTE['primary'])

    # Add value annotations with improved readability
    vmax = np.max(matrix) if matrix.size else 0
    for i in range(len(param2_vals)):
        for j in range(len(param1_vals)):
            value = matrix[i, j]
            text_color = "white" if vmax > 0 and value > (vmax * 0.6) else COLOR_PALETTE['primary']
            ax.text(j, i, f"{value:.2f}",
                    ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'parameter_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filepath


def generate_parameter_sensitivity(
    results: List[Dict],
    output_dir: str
) -> str:
    """
    Generate line plots showing metric sensitivity to each parameter.

    Returns path to saved figure.
    """
    if not HAS_MATPLOTLIB:
        return ""
    if not results:
        raise ValueError("Parameter sensitivity plot requires sweep results.")

    params_to_plot = [
        'max_mismatch_frac',
        'min_mapq',
        'min_shared_snvs_for_edge',
        'merge_distance_threshold',
        'min_weight_for_anchor'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    metric = _select_metric(results)

    for idx, param in enumerate(params_to_plot):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Group by parameter value
        by_value = defaultdict(list)
        for r in results:
            val = r['params'].get(param)
            if val is not None:
                by_value[val].append(r.get(metric, r.get('n_lineages', 0)))

        if not by_value:
            continue

        x = sorted(by_value.keys())
        y_mean = [np.mean(by_value[v]) for v in x]
        y_std = [np.std(by_value[v]) for v in x]

        ax.errorbar(x, y_mean, yerr=y_std, marker='o', capsize=5, linewidth=2.5,
                   color=COLOR_PALETTE['accent'], markerfacecolor=COLOR_PALETTE['accent'],
                   markeredgecolor=COLOR_PALETTE['primary'], markeredgewidth=1.0,
                   ecolor=COLOR_PALETTE['neutral'], capthick=1.5)
        ax.set_xlabel(param.replace('_', ' ').title(), fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_title(f'Sensitivity: {param}', fontweight='bold', color=COLOR_PALETTE['primary'])

    # Remove unused subplot
    if len(params_to_plot) < len(axes):
        for idx in range(len(params_to_plot), len(axes)):
            fig.delaxes(axes[idx])

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'parameter_sensitivity.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filepath

def generate_coverage_performance(
    results: List[Dict],
    output_dir: str
 ) -> str:
    """
    Plot performance vs coverage when coverage metadata is available.
    """
    if not HAS_MATPLOTLIB:
        return ""
    if not results:
        raise ValueError("Performance vs coverage requires sweep results.")

    metric = _select_metric(results)
    coverage_map: Dict[int, List[float]] = defaultdict(list)

    for r in results:
        cov = r.get("coverage")
        if cov is None:
            cov = r.get("params", {}).get("coverage")
        score = r.get(metric)
        if cov is None or score is None:
            continue
        coverage_map[int(cov)].append(score)

    if not coverage_map:
        raise ValueError("Performance vs coverage requires coverage metadata in results.")

    coverages = sorted(coverage_map.keys())
    means = [float(np.mean(coverage_map[c])) for c in coverages]
    stds = [float(np.std(coverage_map[c])) for c in coverages]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(
        coverages,
        means,
        yerr=stds,
        fmt='o-',
        color=COLOR_PALETTE['accent'],
        ecolor=COLOR_PALETTE['neutral'],
        capsize=4,
        linewidth=1.6,
        markersize=5,
    )
    ax.set_xlabel("Coverage (x)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Performance vs coverage")
    ax.set_ylim(0, 1.05 if metric.endswith("f1") else max(means) * 1.2)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "coverage_performance.png")
    plt.savefig(filepath, dpi=600, bbox_inches="tight")
    plt.close()
    return filepath


def generate_metric_correlation(
    results: List[Dict],
    output_dir: str
) -> str:
    """
    Plot correlation matrix among available numeric metrics.
    """
    if not HAS_MATPLOTLIB:
        return ""
    if not results:
        raise ValueError("Metric correlation matrix requires sweep results.")

    metric_candidates = [
        "haplotype_f1",
        "snv_f1",
        "abundance_pearson_r",
        "n_lineages",
        "runtime_seconds",
    ]

    metric_values = {}
    for metric in metric_candidates:
        values = [r.get(metric) for r in results if r.get(metric) is not None]
        if len(values) >= 3:
            metric_values[metric] = values

    if len(metric_values) < 2:
        raise ValueError("Metric correlation matrix requires at least two metrics with data.")

    metrics = list(metric_values.keys())
    data = np.array([metric_values[m] for m in metrics])
    corr = np.corrcoef(data)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", " ") for m in metrics], rotation=35, ha="right")
    ax.set_yticklabels([m.replace("_", " ") for m in metrics])

    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr[i, j]) > 0.5 else COLOR_PALETTE['dark'])

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    ax.set_title("Metric correlation matrix")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "metric_correlation.png")
    plt.savefig(filepath, dpi=600, bbox_inches="tight")
    plt.close()
    return filepath


def generate_optimal_params(
    stable_params: List[Dict],
    output_dir: str
) -> str:
    """
    Generate visualization of optimal parameter ranges.

    Returns path to saved figure.
    """
    if not HAS_MATPLOTLIB:
        return ""
    if not stable_params:
        raise ValueError("Optimal parameter ranges require stable_parameters.json.")

    # Filter to only parameters that have numeric values
    numeric_params = []
    for param in stable_params[0].keys():
        # Check if this parameter has at least one numeric value
        has_numeric = False
        for p in stable_params:
            val = p.get(param)
            if val is not None:
                try:
                    float(val)  # Check if numeric
                    has_numeric = True
                    break
                except (ValueError, TypeError):
                    continue
        if has_numeric:
            numeric_params.append(param)

    if not numeric_params:
        raise ValueError("Optimal parameter ranges require numeric parameter values.")

    n_params = len(numeric_params)

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    for idx, param in enumerate(numeric_params):
        ax = axes[idx]
        # Extract values and filter to only numeric ones
        values = []
        for p in stable_params:
            val = p.get(param)
            if val is not None:
                try:
                    # Try to convert to float to ensure it's numeric
                    float_val = float(val)
                    values.append(float_val)
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue
        
        if not values:
            raise ValueError(f"Optimal parameter ranges missing numeric values for {param}.")

        ax.hist(values, bins=10, color=COLOR_PALETTE['success'], 
               edgecolor=COLOR_PALETTE['primary'], alpha=0.8, linewidth=1.2)
        ax.set_xlabel(param.replace('_', ' ').title(), fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_ylabel('Count', fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_title(f'Stable Range', fontweight='bold', color=COLOR_PALETTE['primary'])

        # Mark optimal (most common)
        optimal = max(set(values), key=values.count)
        ax.axvline(optimal, color=COLOR_PALETTE['error'], linestyle='--', linewidth=2.0,
                   label=f'Optimal: {optimal:.3f}')
        ax.legend()

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'optimal_params.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filepath


def generate_precision_recall_scatter(
    validation_metrics: Optional[Dict],
    output_dir: str
) -> str:
    """
    Scatter precision vs recall per contig with iso-F1 curves.
    TODO: I want something like this but I don't like the current layout.
    """
    if not HAS_MATPLOTLIB:
        return ""

    validation_metrics = _require_validation_metrics(
        validation_metrics, "Per-contig precision vs recall"
    )
    per_contig = _require_per_contig(validation_metrics, "Per-contig precision vs recall")

    points = []
    for metrics in per_contig.values():
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        if precision is None or recall is None:
            continue
        points.append((precision, recall))

    if not points:
        raise ValueError("Per-contig precision vs recall requires precision/recall values.")

    fig, ax = plt.subplots(figsize=(6, 5))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=35, color=COLOR_PALETTE['accent'], edgecolor=COLOR_PALETTE['dark'])

    for f1 in [0.5, 0.7, 0.9]:
        recall = np.linspace(0.01, 1.0, 200)
        precision = (f1 * recall) / (2 * recall - f1)
        precision = np.clip(precision, 0, 1)
        ax.plot(precision, recall, color=COLOR_PALETTE['neutral'], linestyle='--', linewidth=1.0)
        ax.text(0.98, f1 * 0.98, f"F1={f1}", ha="right", va="bottom", fontsize=8, color=COLOR_PALETTE['neutral'])

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Per-contig precision vs recall")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "precision_recall_scatter.png")
    plt.savefig(filepath, dpi=600, bbox_inches="tight")
    plt.close()
    return filepath

def generate_pareto_front(
    results: List[Dict],
    output_dir: str,
    metric: Optional[str] = None
) -> str:
    """Plot Pareto front (metric vs runtime)."""
    if not HAS_MATPLOTLIB:
        return ""
    if not results:
        raise ValueError("Performance vs runtime tradeoff requires sweep results.")

    metric = metric or _select_metric(results)
    points = []
    for r in results:
        score = r.get(metric)
        runtime = r.get("runtime_seconds")
        if score is None or runtime is None:
            continue
        params = r.get("params", {})
        window_size = params.get("window_size", r.get("window_size"))
        points.append((runtime, score, window_size))

    if not points:
        raise ValueError("Performance vs runtime tradeoff requires metric and runtime data.")

    points.sort(key=lambda x: x[0])
    pareto = []
    best_score = -1.0
    for runtime, score, window_size in points:
        if score > best_score:
            pareto.append((runtime, score))
            best_score = score

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    runtimes = [p[0] for p in points]
    scores = [p[1] for p in points]
    colors = [p[2] if p[2] is not None else 0 for p in points]
    sc = ax.scatter(runtimes, scores, c=colors, cmap="viridis", s=35, alpha=0.75)

    ax.plot([p[0] for p in pareto], [p[1] for p in pareto],
            color=COLOR_PALETTE['error'], linewidth=1.8, label="Pareto front")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Performance vs runtime tradeoff")
    ax.legend(frameon=False, loc="lower right")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Window size")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "pareto_front.png")
    plt.savefig(filepath, dpi=600, bbox_inches="tight")
    plt.close()
    return filepath


def _save_no_data_plot(output_dir: str, filename: str, title: str) -> str:
    """Create a placeholder plot when data is unavailable."""
    if not HAS_MATPLOTLIB:
        return ""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=11)
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=600, bbox_inches="tight")
    plt.close()
    return filepath


def _require_validation_metrics(validation_metrics: Optional[Dict], plot_name: str) -> Dict:
    if not validation_metrics:
        raise ValueError(
            f"{plot_name} requires validation metrics. "
            "Run validation and pass --validation to generate_report."
        )
    return validation_metrics


def _require_per_contig(validation_metrics: Dict, plot_name: str) -> Dict:
    per_contig = validation_metrics.get("per_contig_metrics") or {}
    if not per_contig:
        raise ValueError(f"{plot_name} requires per-contig metrics in validation output.")
    return per_contig


def _require_per_timepoint(validation_metrics: Dict, plot_name: str) -> Dict:
    per_tp = validation_metrics.get("per_timepoint_metrics") or {}
    if not per_tp:
        raise ValueError(f"{plot_name} requires per-timepoint metrics in validation output.")
    return per_tp


def _require_results_fields(results: List[Dict], fields: List[str], plot_name: str) -> None:
    missing = []
    for field in fields:
        if not any(r.get(field) is not None for r in results):
            missing.append(field)
    if missing:
        raise ValueError(f"{plot_name} requires results fields: {', '.join(missing)}.")


def generate_error_decomposition(
    validation_metrics: Optional[Dict],
    output_dir: str
) -> str:
    """Plot error decomposition: false merges, false splits, missed lineages."""
    if not HAS_MATPLOTLIB:
        return ""

    validation_metrics = _require_validation_metrics(validation_metrics, "Error decomposition")
    per_contig = _require_per_contig(validation_metrics, "Error decomposition")
    false_negatives = validation_metrics.get("false_negatives") or []

    # Prefer explicit linkage error rates when available.
    false_link_rate = validation_metrics.get("false_link_rate")
    missed_link_rate = validation_metrics.get("missed_link_rate")

    total_true = sum(m.get("n_true", 0) for m in per_contig.values()) if per_contig else 0

    if false_link_rate is None or missed_link_rate is None:
        # Fallback: infer merge/split counts from per-contig lineage counts.
        merge_count = 0
        split_count = 0
        if per_contig:
            for metrics in per_contig.values():
                n_true = metrics.get("n_true", 0)
                n_detected = metrics.get("n_detected", 0)
                if n_detected < n_true:
                    merge_count += (n_true - n_detected)
                elif n_detected > n_true:
                    split_count += (n_detected - n_true)
        else:
            merge_count = 0
            split_count = 0

        if total_true > 0:
            merge_rate = (merge_count / total_true) * 100
            split_rate = (split_count / total_true) * 100
        else:
            merge_rate = float(merge_count)
            split_rate = float(split_count)
    else:
        merge_rate = false_link_rate * 100
        split_rate = missed_link_rate * 100

    if total_true > 0:
        missed_lineage_rate = (len(false_negatives) / total_true) * 100
    else:
        missed_lineage_rate = float(len(false_negatives))

    if merge_rate == 0 and split_rate == 0 and missed_lineage_rate == 0:
        logger.warning("Error decomposition has no non-zero values; skipping plot.")
        return ""

    labels = ["False merges", "False splits", "Missed lineages"]
    values = [merge_rate, split_rate, missed_lineage_rate]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, values, color=[
        COLOR_PALETTE['error'],
        COLOR_PALETTE['warning'],
        COLOR_PALETTE['accent'],
    ], edgecolor=COLOR_PALETTE['dark'], linewidth=0.8, alpha=0.85)

    ylabel = "Rate (%)" if total_true else "Count"
    ax.set_ylabel(ylabel)
    ax.set_title("Error decomposition")
    ax.set_ylim(0, max(values) * 1.25 if values else 1.0)
    ax.bar_label(bars, fmt="%.2f" if total_true else "%d", padding=3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "error_decomposition.png")
    plt.savefig(filepath, dpi=600, bbox_inches="tight")
    plt.close()
    return filepath


# =============================================================================
# HTML report generation
# =============================================================================

def generate_html_report(
    results: List[Dict],
    summary: Dict,
    stable_params: List[Dict],
    validation_metrics: Optional[Dict],
    validation_figures: Dict[str, str],
    figures: Dict[str, str],
    output_dir: str
) -> str:
    """
    Generate HTML benchmark report.

    Returns path to saved HTML file.
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strainphase Benchmark Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: Arial, 'Helvetica Neue', Helvetica, sans-serif;
            line-height: 1.7;
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 20px;
            background: #FAFAFA;
            color: #2C3E50;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 30px;
            border: 1px solid #E0E0E0;
        }}
        h1 {{
            color: #2C3E50;
            border-bottom: 4px solid #3498DB;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-size: 32px;
            font-weight: bold;
            letter-spacing: -0.5px;
        }}
        h2 {{
            color: #34495E;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 24px;
            font-weight: bold;
            border-bottom: 2px solid #ECF0F1;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #34495E;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #2C3E50;
            color: white;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
            padding: 12px 15px;
        }}
        tr:hover {{
            background: #F8F9FA;
        }}
        .metric {{
            display: inline-block;
            background: #F8F9FA;
            padding: 15px 25px;
            border-radius: 4px;
            margin: 8px;
            border: 1px solid #E0E0E0;
            text-align: center;
            min-width: 120px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2C3E50;
            font-family: Arial, sans-serif;
        }}
        .metric-label {{
            font-size: 13px;
            color: #7F8C8D;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            margin-top: 5px;
        }}
        .figure-container {{
            text-align: center;
            margin: 40px 0;
            padding: 25px;
            background: #FAFAFA;
            border-radius: 4px;
            border: 1px solid #E0E0E0;
        }}
        .figure-container h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            color: #2C3E50;
            font-size: 16px;
            font-weight: bold;
        }}
        .figure-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border: 1px solid #E0E0E0;
        }}
        .recommendation {{
            background: #F8F9FA;
            border-left: 4px solid #3498DB;
            padding: 25px;
            margin: 25px 0;
            border-radius: 4px;
            border: 1px solid #E0E0E0;
        }}
        .recommendation strong {{
            color: #2C3E50;
            font-size: 17px;
            display: block;
            margin-bottom: 12px;
            font-weight: bold;
        }}
        .recommendation p {{
            margin: 12px 0;
            line-height: 1.8;
        }}
        .warning {{
            background: #FEF9E7;
            border-left: 4px solid #F39C12;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
            border: 1px solid #E0E0E0;
        }}
        code {{
            background: #F5F5F5;
            padding: 3px 7px;
            border-radius: 3px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 13px;
            color: #E74C3C;
        }}
        ul, ol {{
            margin-left: 30px;
            line-height: 1.9;
        }}
        li {{
            margin: 10px 0;
            line-height: 1.7;
        }}
        p {{
            margin: 18px 0;
            line-height: 1.8;
            color: #34495E;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Strainphase Benchmark Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="container">
        <h2>Summary Metrics</h2>
        <div>
            <div class="metric">
                <div class="metric-value">{summary.get('n_configs_tested', len(results))}</div>
                <div class="metric-label">Parameter Configs Tested</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary.get('n_scenarios', 1)}</div>
                <div class="metric-label">Scenarios Evaluated</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(stable_params)}</div>
                <div class="metric-label">Stable Parameter Sets</div>
            </div>
        </div>
    </div>
"""

    # Add scenario results table
    if summary.get('scenarios'):
        html += """
    <div class="container">
        <h2>Scenario Performance</h2>
        <table>
            <tr>
                <th>Scenario</th>
                <th>Mean Lineages</th>
                <th>Lineage Std</th>
                <th>Sweep Detection</th>
                <th>Convergence</th>
                <th>Mean Runtime</th>
                <th>Haplotype F1</th>
                <th>SNV F1</th>
            </tr>
"""
        for name, stats in summary['scenarios'].items():
            n_lin = stats.get('n_lineages', {})
            sweep = stats.get('sweep_detection', {})
            runtime = stats.get('runtime', {})
            hap_f1 = stats.get('haplotype_f1')
            snv_f1 = stats.get('snv_f1')
            hap_f1_str = f"{hap_f1:.2f}" if hap_f1 is not None else "n/a"
            snv_f1_str = f"{snv_f1:.2f}" if snv_f1 is not None else "n/a"

            html += f"""
            <tr>
                <td>{name}</td>
                <td>{n_lin.get('mean', 0):.2f}</td>
                <td>{n_lin.get('std', 0):.2f}</td>
                <td>{sweep.get('detection_rate', 0):.1%}</td>
                <td>{stats.get('converged_fraction', 0):.1%}</td>
                <td>{runtime.get('mean', 0):.2f}s</td>
                <td>{hap_f1_str}</td>
                <td>{snv_f1_str}</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""

    # Add validation summary
    if validation_metrics:
        html += """
    <div class="container">
        <h2>Validation Summary</h2>
        <div>
"""
        metric_rows = [
            ("Precision", validation_metrics.get("precision")),
            ("Recall", validation_metrics.get("recall")),
            ("F1", validation_metrics.get("f1")),
            ("Abundance Pearson r", validation_metrics.get("abundance_pearson_r")),
            ("Abundance MAE", validation_metrics.get("abundance_mae")),
            ("SNV Precision", validation_metrics.get("snv_precision")),
            ("SNV Recall", validation_metrics.get("snv_recall")),
            ("Phasing Accuracy", validation_metrics.get("phasing_accuracy")),
            ("Detection Threshold", validation_metrics.get("detection_threshold")),
        ]
        for label, value in metric_rows:
            if value is None:
                continue
            html += f"""
            <div class="metric">
                <div class="metric-value">{value:.3f}</div>
                <div class="metric-label">{label}</div>
            </div>
"""
        html += """
        </div>
"""
        
        # Add detailed diagnostics if available
        if validation_metrics.get("false_negatives") or validation_metrics.get("false_positives"):
            html += """
        <h3>Error Breakdown</h3>
        <div style="margin-top: 20px;">
"""
            if validation_metrics.get("false_negatives"):
                html += f"""
            <h4>False Negatives ({len(validation_metrics['false_negatives'])} missing haplotypes)</h4>
            <ul>
"""
                for fn in validation_metrics['false_negatives'][:10]:  # Show first 10
                    html += f"                <li>{fn}</li>\n"
                if len(validation_metrics['false_negatives']) > 10:
                    html += f"                <li>... and {len(validation_metrics['false_negatives']) - 10} more</li>\n"
                html += """
            </ul>
"""
            
            if validation_metrics.get("false_positives"):
                html += f"""
            <h4>False Positives ({len(validation_metrics['false_positives'])} spurious lineages)</h4>
            <ul>
"""
                for fp in validation_metrics['false_positives'][:10]:  # Show first 10
                    html += f"                <li>{fp}</li>\n"
                if len(validation_metrics['false_positives']) > 10:
                    html += f"                <li>... and {len(validation_metrics['false_positives']) - 10} more</li>\n"
                html += """
            </ul>
"""
            html += """
        </div>
"""
        
        # Add per-contig and per-timepoint breakdowns if available
        if validation_metrics.get("per_contig_metrics"):
            html += """
        <h3>Per-Contig Performance</h3>
        <table>
            <tr>
                <th>Contig</th>
                <th>True</th>
                <th>Detected</th>
                <th>Matched</th>
                <th>Precision</th>
                <th>Recall</th>
            </tr>
"""
            for contig, metrics in sorted(validation_metrics['per_contig_metrics'].items()):
                html += f"""
            <tr>
                <td>{contig}</td>
                <td>{metrics['n_true']}</td>
                <td>{metrics['n_detected']}</td>
                <td>{metrics['n_matched']}</td>
                <td>{metrics['precision']:.3f}</td>
                <td>{metrics['recall']:.3f}</td>
            </tr>
"""
            html += """
        </table>
"""
        
        if validation_metrics.get("per_timepoint_metrics"):
            html += """
        <h3>Per-Timepoint Performance</h3>
        <table>
            <tr>
                <th>Timepoint</th>
                <th>True</th>
                <th>Detected</th>
                <th>Matched</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Abundance r</th>
                <th>Abundance MAE</th>
            </tr>
"""
            for tp, metrics in sorted(validation_metrics['per_timepoint_metrics'].items()):
                abund_r = f"{metrics['abundance_pearson_r']:.3f}" if metrics.get('abundance_pearson_r') is not None else "n/a"
                abund_mae = f"{metrics['abundance_mae']:.4f}" if metrics.get('abundance_mae') is not None else "n/a"
                html += f"""
            <tr>
                <td>{tp}</td>
                <td>{metrics['n_true']}</td>
                <td>{metrics['n_detected']}</td>
                <td>{metrics['n_matched']}</td>
                <td>{metrics['precision']:.3f}</td>
                <td>{metrics['recall']:.3f}</td>
                <td>{abund_r}</td>
                <td>{abund_mae}</td>
            </tr>
"""
            html += """
        </table>
"""
        
        html += """
    </div>
"""

    # Add validation figures
    if validation_figures:
        html += """
    <div class="container">
        <h2>Validation Figures</h2>
"""
        for filename, title in validation_figures.items():
            rel_path = os.path.basename(filename)
            html += f"""
        <div class="figure-container">
            <h3>{title}</h3>
            <img src="{rel_path}" alt="{title}">
        </div>
"""
        html += """
    </div>
"""

    # Add figures
    html += """
    <div class="container">
        <h2>Benchmarking Figures</h2>
"""

    # Generate benchmarking figures
    figures['parameter_heatmap.png'] = generate_parameter_heatmap(results, output_dir)
    figures['parameter_sensitivity.png'] = generate_parameter_sensitivity(results, output_dir)
    figures['precision_recall_scatter.png'] = generate_precision_recall_scatter(
        validation_metrics, output_dir
    )
    figures['pareto_front.png'] = generate_pareto_front(results, output_dir)
    figures['optimal_params.png'] = generate_optimal_params(stable_params, output_dir)
    try:
        figures['coverage_performance.png'] = generate_coverage_performance(results, output_dir)
    except Exception as e:
        logger.warning(f"Skipping coverage performance plot: {e}")
    figures['metric_correlation.png'] = generate_metric_correlation(results, output_dir)
    error_path = generate_error_decomposition(validation_metrics, output_dir)
    if error_path:
        figures['error_decomposition.png'] = error_path
    
    figure_titles = {
        'parameter_heatmap.png': 'Parameter Heatmap',
        'parameter_sensitivity.png': 'Parameter Sensitivity Analysis',
        'precision_recall_scatter.png': 'Per-contig Precision vs Recall',
        'pareto_front.png': 'Performance vs Runtime (Pareto Front)',
        'abundance_timepoint.png': 'Abundance Correlation Over Time',
        'optimal_params.png': 'Optimal Parameter Ranges',
        'coverage_performance.png': 'Performance vs Coverage',
        'metric_correlation.png': 'Metric Correlation Matrix',
        'error_decomposition.png': 'Error Decomposition',
        'haplotype_accuracy_summary.png': 'Haplotype Accuracy by Parameter Set',
        'lineage_accuracy_summary.png': 'Lineage Accuracy by Parameter Set',
        'linking_errors_summary.png': 'Linking Errors by Parameter Set',
        'error_breakdown_summary.png': 'Error Breakdown by Parameter Set',
        'abundance_correlation.png': 'Abundance Correlation (Top F1 Patchwork)',
        'reference_coverage.png': 'Reference Coverage (Top F1 Patchwork)',
        'track_regions.png': 'Track Regions (Top F1 Patchwork)',
    }

    for filename, title in figure_titles.items():
        if filename in figures and figures[filename]:
            # Use relative path for HTML
            rel_path = os.path.basename(figures[filename])
            html += f"""
        <div class="figure-container">
            <h3>{title}</h3>
            <img src="{rel_path}" alt="{title}">
        </div>
"""

    html += """
    </div>
"""

    # Best parameter sets per scenario
    best_params = summary.get("best_params_by_scenario", {})
    if best_params:
        html += """
    <div class="container">
        <h2>Best Parameters by Scenario</h2>
        <table>
            <tr>
                <th>Scenario</th>
                <th>Params</th>
                <th>Haplotype F1</th>
                <th>SNV F1</th>
                <th>Abundance r</th>
            </tr>
"""
        for scenario_name, info in best_params.items():
            params = info.get("params", {})
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            hap_f1 = info.get("haplotype_f1")
            snv_f1 = info.get("snv_f1")
            abundance_r = info.get("abundance_pearson_r")
            html += f"""
            <tr>
                <td>{scenario_name}</td>
                <td><code>{param_str}</code></td>
                <td>{hap_f1 if hap_f1 is not None else "n/a"}</td>
                <td>{snv_f1 if snv_f1 is not None else "n/a"}</td>
                <td>{abundance_r if abundance_r is not None else "n/a"}</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""

    # Failure mode analysis with detailed diagnostics
    if summary.get('scenarios'):
        issues = []
        detailed_issues = []
        
        for name, stats in summary['scenarios'].items():
            if stats.get('sweep_detection', {}).get('detection_rate', 1) < 0.5:
                issues.append(f"{name}: low sweep detection rate")
            if stats.get('converged_fraction', 1) < 0.9:
                issues.append(f"{name}: low convergence")
            if stats.get('n_lineages', {}).get('std', 0) > 1.0:
                issues.append(f"{name}: high lineage variance")
            if stats.get('haplotype_f1') is not None and stats.get('haplotype_f1') < 0.8:
                issues.append(f"{name}: haplotype F1 below 0.8")
                detailed_issues.append({
                    'scenario': name,
                    'haplotype_f1': stats.get('haplotype_f1'),
                    'haplotype_precision': stats.get('haplotype_precision'),
                    'haplotype_recall': stats.get('haplotype_recall'),
                })
        
        html += """
    <div class="container">
        <h2>Failure Mode Analysis</h2>
"""
        if issues:
            html += "        <ul>\n"
            for issue in issues:
                html += f"            <li>{issue}</li>\n"
            html += "        </ul>\n"
        else:
            html += "        <p>No major failure modes detected based on summary thresholds.</p>\n"
        
        # Add detailed breakdown for scenarios with low F1
        if detailed_issues:
            html += """
        <h3>Detailed Error Analysis</h3>
        <table>
            <tr>
                <th>Scenario</th>
                <th>Haplotype F1</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Diagnosis</th>
            </tr>
"""
            for issue in detailed_issues:
                f1 = issue['haplotype_f1']
                precision = issue.get('haplotype_precision')
                recall = issue.get('haplotype_recall')
                
                diagnosis = []
                if precision is not None and recall is not None:
                    if precision < 0.7:
                        diagnosis.append("High false positive rate (many spurious lineages)")
                    if recall < 0.7:
                        diagnosis.append("High false negative rate (missing true haplotypes)")
                    if precision < recall:
                        diagnosis.append("Over-detection (too many lineages)")
                    elif recall < precision:
                        diagnosis.append("Under-detection (too few lineages)")
                
                diagnosis_str = "; ".join(diagnosis) if diagnosis else "Check detailed validation reports"
                
                html += f"""
            <tr>
                <td>{issue['scenario']}</td>
                <td>{f1:.3f}</td>
                <td>{precision:.3f if precision is not None else 'n/a'}</td>
                <td>{recall:.3f if recall is not None else 'n/a'}</td>
                <td>{diagnosis_str}</td>
            </tr>
"""
            html += """
        </table>
        <p><em>For detailed per-config error breakdowns, check the validation reports in configs/*/validation/</em></p>
"""
        
        html += """
    </div>
"""

    # Add recommendations
    if stable_params:
        html += """
    <div class="container">
        <h2>Recommendations</h2>
        <div class="recommendation">
            <strong>Optimal Parameters Found</strong>
            <p>Based on the sweep analysis, the following parameter ranges produce stable results:</p>
            <ul>
"""
        # Aggregate stable param ranges
        param_ranges = defaultdict(list)
        for p in stable_params:
            for k, v in p.items():
                param_ranges[k].append(v)

        for param, values in param_ranges.items():
            values = [v for v in values if v is not None]
            if not values:
                continue
            min_v = min(values)
            max_v = max(values)
            if min_v == max_v:
                html += f"                <li><code>{param}</code>: {min_v}</li>\n"
            else:
                html += f"                <li><code>{param}</code>: {min_v} - {max_v}</li>\n"

        html += """
            </ul>
        </div>
    </div>
"""
    else:
        html += """
    <div class="container">
        <h2>Recommendations</h2>
        <div class="warning">
            <strong>No Stable Parameters Identified</strong>
            <p>Consider expanding the parameter grid or adjusting stability criteria.</p>
        </div>
    </div>
"""

    html += """
</body>
</html>
"""

    filepath = os.path.join(output_dir, 'benchmark_report.html')
    with open(filepath, 'w') as f:
        f.write(html)

    return filepath


# =============================================================================
# Main pipeline
# =============================================================================

def generate_report(
    results_dir: str,
    output_dir: str,
    validation_dir: Optional[str] = None,
    truth_dir: Optional[str] = None
) -> str:
    """
    Generate complete benchmark report.

    Args:
        results_dir: Directory containing sweep_results.json, etc.
        output_dir: Output directory for figures and HTML
        validation_dir: Optional validation results directory

    Returns:
        Path to generated HTML report
    """
    set_plot_style()
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed, figures will be skipped")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading results from {results_dir}")
    results, summary, stable_params = load_sweep_results(results_dir)

    if not results:
        logger.error("No results found in results directory")
        return ""

    logger.info(f"Loaded {len(results)} sweep results")

    # Set figure style
    set_figure_style()

    # Generate figures
    figures = {}
    validation_metrics = None
    validation_figures = {}

    # Fallback: redefine "stable" as configs with F1 >= 0.8
    if not stable_params:
        stable_params = []
        for r in results:
            f1 = r.get("haplotype_f1")
            if f1 is not None and f1 >= 0.8:
                params = r.get("params") or {}
                if params:
                    stable_params.append(params)
        if stable_params:
            logger.info(f"Using {len(stable_params)} configs with haplotype_f1 >= 0.8 as stable parameters")

    if validation_dir:
        validation_metrics = load_validation_metrics(validation_dir)
        validation_files = {
            "detection_sensitivity.png": "Detection Sensitivity",
            "confusion_matrix.png": "Haplotype Confusion Matrix",
            "abundance_trajectories.png": "Abundance Trajectories",
            "track_fragmentation.png": "Track Fragmentation",
            "detection_roc.png": "Detection Performance (ROC-like)",
            "scalability_analysis.png": "Scalability Analysis",
        }
        for filename, title in validation_files.items():
            src_path = os.path.join(validation_dir, filename)
            if os.path.exists(src_path):
                dest_path = os.path.join(output_dir, filename)
                shutil.copy2(src_path, dest_path)
                validation_figures[dest_path] = title

        patchwork_path = generate_validation_patchwork(results_dir, results, truth_dir, output_dir)
        if patchwork_path:
            validation_figures[patchwork_path] = "Validation Patchwork Summary"

        try:
            figures["haplotype_accuracy_summary.png"] = generate_haplotype_accuracy_summary(results, output_dir)
            figures["lineage_accuracy_summary.png"] = generate_lineage_accuracy_summary(results, output_dir)
            figures["linking_errors_summary.png"] = generate_linking_errors_summary(results, output_dir)
            figures["error_breakdown_summary.png"] = generate_error_breakdown_summary(results, output_dir)
        except Exception as e:
            logger.warning(f"Failed to generate summary bar charts: {e}")

        try:
            path = generate_abundance_correlation_patchwork(results_dir, results, output_dir)
            if path:
                figures["abundance_correlation.png"] = path
            path = generate_reference_coverage_patchwork(results_dir, results, truth_dir, output_dir)
            if path:
                figures["reference_coverage.png"] = path
            path = generate_track_regions_patchwork(results_dir, results, truth_dir, output_dir)
            if path:
                figures["track_regions.png"] = path
        except Exception as e:
            logger.warning(f"Failed to generate patchwork summaries: {e}")

    logger.info("Generating parameter heatmap...")
    figures['parameter_heatmap.png'] = generate_parameter_heatmap(results, output_dir)

    logger.info("Generating parameter sensitivity plot...")
    figures['parameter_sensitivity.png'] = generate_parameter_sensitivity(results, output_dir)

    logger.info("Generating precision-recall scatter...")
    figures['precision_recall_scatter.png'] = generate_precision_recall_scatter(
        validation_metrics, output_dir
    )

    logger.info("Generating Pareto front plot...")
    figures['pareto_front.png'] = generate_pareto_front(results, output_dir)

    logger.info("Generating optimal params visualization...")
    if stable_params:
        figures['optimal_params.png'] = generate_optimal_params(stable_params, output_dir)
    else:
        logger.warning("Skipping optimal params visualization (no stable configs with F1 >= 0.8)")
    
    # Additional publication-quality plots
    logger.info("Generating performance vs coverage plot...")
    try:
        figures['coverage_performance.png'] = generate_coverage_performance(results, output_dir)
    except Exception as e:
        logger.warning(f"Skipping coverage performance plot: {e}")
    
    logger.info("Generating metric correlation matrix...")
    figures['metric_correlation.png'] = generate_metric_correlation(results, output_dir)

    logger.info("Generating error decomposition plot...")
    error_path = generate_error_decomposition(validation_metrics, output_dir)
    if error_path:
        figures['error_decomposition.png'] = error_path

    # Generate HTML report
    logger.info("Generating HTML report...")
    report_path = generate_html_report(
        results, summary, stable_params, validation_metrics, validation_figures, figures, output_dir
    )

    logger.info(f"Report generated: {report_path}")
    return report_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark report from parameter sweep results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--results", required=True,
                        help="Directory containing sweep results (sweep_results.json)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for figures and HTML report")
    parser.add_argument("--validation",
                        help="Optional validation results directory")
    parser.add_argument("--truth",
                        help="Truth directory for generating validation patchwork")

    args = parser.parse_args()

    report_path = generate_report(
        results_dir=args.results,
        output_dir=args.output,
        validation_dir=args.validation,
        truth_dir=args.truth
    )

    if report_path:
        print(f"\nReport generated: {report_path}")
    else:
        print("\nFailed to generate report")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
