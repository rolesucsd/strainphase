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
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


# =============================================================================
# Figure generation
# =============================================================================

# Professional color palette
COLOR_PALETTE = {
    'primary': '#1F2933',      # Charcoal
    'secondary': '#3E4C59',    # Slate
    'accent': '#4B7F9D',       # Muted blue
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


def generate_runtime_scaling(
    results: List[Dict],
    output_dir: str
) -> str:
    """
    Generate runtime scaling plot.

    Returns path to saved figure.
    """
    if not HAS_MATPLOTLIB or not results:
        return ""

    runtimes = [r.get('runtime_seconds', 0) for r in results]
    n_lineages = [r.get('n_lineages', 0) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of runtimes
    ax1.hist(runtimes, bins=20, color=COLOR_PALETTE['accent'], 
            edgecolor=COLOR_PALETTE['primary'], linewidth=1.2, alpha=0.8)
    ax1.set_xlabel('Runtime (seconds)', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax1.set_ylabel('Count', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax1.set_title('Runtime Distribution', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax1.axvline(np.mean(runtimes), color=COLOR_PALETTE['error'], linestyle='--', linewidth=2.0,
                label=f'Mean: {np.mean(runtimes):.2f}s', alpha=0.8)
    ax1.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])

    # Runtime vs lineages
    ax2.scatter(n_lineages, runtimes, alpha=0.7, s=50, 
               color=COLOR_PALETTE['accent'], edgecolors=COLOR_PALETTE['primary'], 
               linewidths=0.8)
    ax2.set_xlabel('Number of Lineages', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax2.set_ylabel('Runtime (seconds)', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax2.set_title('Runtime vs Complexity', fontweight='bold', color=COLOR_PALETTE['primary'])

    # Add trend line (only if there's variance in the data)
    if len(n_lineages) > 2 and len(set(n_lineages)) > 1:
        try:
            z = np.polyfit(n_lineages, runtimes, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(n_lineages), max(n_lineages), 100)
            ax2.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Trend')
            handles, labels = ax2.get_legend_handles_labels()
            if handles:
                ax2.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
        except (np.linalg.LinAlgError, ValueError):
            pass  # Skip trend line if fitting fails

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'runtime_scaling.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filepath


def generate_complexity_comparison(
    results: List[Dict],
    summary: Dict,
    output_dir: str
) -> str:
    """
    Generate enhanced grouped bar chart comparing metrics across complexity levels.
    Publication-quality visualization with multiple metrics.

    Returns path to saved figure.
    """
    if not HAS_MATPLOTLIB:
        return ""
    if not results:
        raise ValueError("Complexity comparison requires sweep results.")

    scenarios = summary.get('scenarios', {})
    if not scenarios:
        raise ValueError("Complexity comparison requires scenario stats in sweep_summary.json.")

    scenario_names = list(scenarios.keys())
    
    # Enhanced metrics for publication
    metrics = ['haplotype_f1', 'snv_f1', 'abundance_pearson_r', 'track_fragmentation_mean']
    metric_labels = ['Haplotype F1', 'SNV F1', 'Abundance r', 'Track Fragmentation']
    
    # Extract metrics from results (more accurate than summary stats)
    metric_by_scenario = {name: {m: [] for m in metrics} for name in scenario_names}
    for r in results:
        scenario = r.get('community', r.get('scenario_name', 'default'))
        if scenario in scenario_names:
            metrics_dict = r.get('metrics', {})
            for metric in metrics:
                val = metrics_dict.get(metric)
                if val is not None:
                    metric_by_scenario[scenario][metric].append(val)
    
    # Compute means
    scenario_means = {}
    for scenario in scenario_names:
        scenario_means[scenario] = {}
        for metric in metrics:
            vals = metric_by_scenario[scenario][metric]
            scenario_means[scenario][metric] = np.mean(vals) if vals else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = COLOR_SEQUENCES['qualitative']

    x = np.arange(len(scenario_names))
    width = 0.6

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = [scenario_means[name].get(metric, 0) if scenario_means[name].get(metric) is not None else 0 
                 for name in scenario_names]
        
        bars = ax.bar(x, values, width, label=label, 
                     color=colors[idx % len(colors)], 
                     edgecolor=COLOR_PALETTE['primary'],
                     linewidth=1.2, alpha=0.85)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                       fontweight='bold', color=COLOR_PALETTE['primary'])
        
        ax.set_ylabel(label, fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_title(f'{label} by Complexity', fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=45, ha='right', color=COLOR_PALETTE['primary'])
        if metric == 'track_fragmentation_mean':
            max_val = max(values) if values else 0
            ax.set_ylim(0, max_val * 1.2 if max_val > 0 else 1.0)
            ax.axhline(y=1.0, color=COLOR_PALETTE['error'], linestyle='--', 
                      linewidth=2.0, alpha=0.7, label='Ideal (1.0)')
        else:
            ax.set_ylim(0, 1.1)
            ax.axhline(y=0.9, color=COLOR_PALETTE['warning'], linestyle='--',
                      linewidth=2.0, alpha=0.7, label='Target (0.9)')
        if idx == 0:
            ax.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'], fontsize=8)

    plt.suptitle('Performance Metrics Across Complexity Levels', 
                fontsize=16, fontweight='bold', color=COLOR_PALETTE['primary'], y=0.995)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'complexity_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filepath


def generate_ablation_summary(
    results: List[Dict],
    output_dir: str
) -> str:
    """
    Generate ablation summary showing delta-metrics for key ablations.
    
    Returns path to saved figure.
    """
    if not HAS_MATPLOTLIB:
        return ""
    if not results:
        raise ValueError("Ablation summary requires sweep results.")
    
    # Group by ablation mode
    by_ablation = defaultdict(list)
    for r in results:
        ablation = r.get('ablation', 'full')
        by_ablation[ablation].append(r)
    
    if len(by_ablation) < 2:
        raise ValueError("Ablation summary requires multiple ablation modes in results.")
    
    # Compare each ablation to 'full'
    full_results = by_ablation.get('full', [])
    if not full_results:
        raise ValueError("Ablation summary requires a 'full' ablation baseline in results.")
    
    ablation_modes = [m for m in by_ablation.keys() if m != 'full']
    metrics = ['haplotype_f1', 'snv_f1', 'track_fragmentation_mean', 'lineage_f1']
    metric_labels = ['Haplotype F1', 'SNV F1', 'Track Fragmentation', 'Lineage F1']
    
    # Compute baseline (full mode)
    baseline = {}
    for metric in metrics:
        values = [r.get('metrics', {}).get(metric) for r in full_results 
                 if r.get('metrics', {}).get(metric) is not None]
        baseline[metric] = np.mean(values) if values else 0.0
    
    # Compute deltas for each ablation
    deltas = {mode: {} for mode in ablation_modes}
    for mode in ablation_modes:
        mode_results = by_ablation[mode]
        for metric in metrics:
            values = [r.get('metrics', {}).get(metric) for r in mode_results
                     if r.get('metrics', {}).get(metric) is not None]
            if values:
                mean_val = np.mean(values)
                deltas[mode][metric] = mean_val - baseline[metric]
            else:
                deltas[mode][metric] = 0.0
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(ablation_modes)
    
    colors = COLOR_SEQUENCES['qualitative'][:len(ablation_modes)]
    for idx, mode in enumerate(ablation_modes):
        delta_values = [deltas[mode].get(m, 0) for m in metrics]
        ax.bar(x + idx * width, delta_values, width, label=mode.replace('_', ' '), 
              color=colors[idx], edgecolor=COLOR_PALETTE['primary'], linewidth=1.2, alpha=0.85)
    
    ax.set_xlabel('Metric', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_ylabel('Δ (Ablation - Full)', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_title('Ablation Summary: Performance Delta vs Full Mode',
                fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_xticks(x + width * (len(ablation_modes) - 1) / 2)
    ax.set_xticklabels(metric_labels, rotation=45, ha='right',
                      color=COLOR_PALETTE['primary'], fontsize=10)
    ax.axhline(y=0, color=COLOR_PALETTE['primary'], linestyle='-', linewidth=1.5)
    ax.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
    # No grid - clean professional look
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'ablation_summary.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def generate_seed_sensitivity(
    results: List[Dict],
    output_dir: str
) -> str:
    """
    Generate seed sensitivity plot showing metric variability across replicate seeds.
    
    Returns path to saved figure.
    """
    if not HAS_MATPLOTLIB or not results:
        return ""
    
    # Group by parameter config and seed
    by_config = defaultdict(lambda: defaultdict(list))
    for r in results:
        config_key = str(r.get('params', {}))
        seed = r.get('seed')
        if seed is not None:
            by_config[config_key][seed].append(r)
    
    # Find configs with multiple seeds
    multi_seed_configs = {k: v for k, v in by_config.items() if len(v) > 1}
    if not multi_seed_configs:
        raise ValueError("Seed sensitivity requires multiple seeds per config in results.")
    
    # Select metric
    metric = _select_metric(results)
    
    # Plot variability
    fig, ax = plt.subplots(figsize=(12, 6))
    
    config_names = []
    means = []
    stds = []
    
    for config_key, seeds_dict in list(multi_seed_configs.items())[:10]:  # Limit to 10 configs
        all_values = []
        for seed_results in seeds_dict.values():
            for r in seed_results:
                val = r.get('metrics', {}).get(metric, r.get(metric, r.get('n_lineages', 0)))
                if val is not None:
                    all_values.append(val)
        
        if len(all_values) > 1:
            config_names.append(config_key[:30] + '...' if len(config_key) > 30 else config_key)
            means.append(np.mean(all_values))
            stds.append(np.std(all_values))
    
    if not means:
        raise ValueError("Seed sensitivity requires metric values for multiple seeds.")
    
    x = np.arange(len(config_names))
    ax.errorbar(x, means, yerr=stds, marker='o', capsize=5, linestyle='None', 
               markersize=9, capthick=2, color=COLOR_PALETTE['accent'],
               markerfacecolor=COLOR_PALETTE['accent'], markeredgecolor=COLOR_PALETTE['primary'],
               markeredgewidth=1.0, ecolor=COLOR_PALETTE['neutral'], linewidth=2.0)
    ax.set_xlabel('Parameter Configuration', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', 
                 fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_title('Seed Sensitivity: Metric Variability Across Replicates',
                fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=8,
                       color=COLOR_PALETTE['primary'])
    # No grid - clean professional look
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'seed_sensitivity.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def generate_vcf_robustness(
    results: List[Dict],
    output_dir: str
) -> str:
    """
    Generate VCF robustness plot showing performance under perturbed VCF conditions.
    
    Returns path to saved figure.
    """
    if not HAS_MATPLOTLIB:
        return ""
    if not results:
        raise ValueError("VCF robustness requires sweep results with VCF condition metadata.")
    
    metric = _select_metric(results)

    # Expect results to include VCF condition metadata.
    condition_key = None
    for key in ("vcf_condition", "vcf_realism", "vcf_tag"):
        if any(r.get(key) is not None for r in results):
            condition_key = key
            break
    if condition_key is None:
        raise ValueError("VCF robustness requires VCF condition metadata in results.")

    condition_map: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        condition = r.get(condition_key)
        score = r.get(metric)
        if condition is None or score is None:
            continue
        condition_map[str(condition)].append(score)

    if not condition_map:
        raise ValueError("VCF robustness has no condition scores to plot.")

    conditions = list(condition_map.keys())
    values = [float(np.mean(condition_map[c])) for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(conditions, values, color=[COLOR_PALETTE['success'], COLOR_PALETTE['warning'], 
          COLOR_PALETTE['error'], COLOR_PALETTE['info']], 
          edgecolor=COLOR_PALETTE['primary'], linewidth=1.2, alpha=0.85)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', 
                 fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_title('VCF Robustness: Performance Under Perturbed VCF Conditions',
                fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, color=COLOR_PALETTE['primary'])
    
    for i, (cond, val) in enumerate(zip(conditions, values)):
        ax.text(i, val + 0.02, f'{val:.2f}', ha='center', fontsize=11, 
               fontweight='bold', color=COLOR_PALETTE['primary'])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'vcf_robustness.png')
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


def generate_per_contig_f1_distribution(
    validation_metrics: Optional[Dict],
    output_dir: str
) -> str:
    """Plot per-contig F1 distribution."""
    if not HAS_MATPLOTLIB:
        return ""

    validation_metrics = _require_validation_metrics(
        validation_metrics, "Per-contig F1 distribution"
    )
    per_contig = _require_per_contig(validation_metrics, "Per-contig F1 distribution")

    f1_values = []
    for metrics in per_contig.values():
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        if precision is None or recall is None:
            continue
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom else 0.0
        f1_values.append(f1)

    if not f1_values:
        raise ValueError("Per-contig F1 distribution requires precision/recall values.")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        f1_values,
        vert=True,
        widths=0.4,
        patch_artist=True,
        boxprops=dict(facecolor=COLOR_PALETTE['accent'], alpha=0.35),
        medianprops=dict(color=COLOR_PALETTE['dark'], linewidth=1.6),
    )
    jitter = np.random.normal(1, 0.04, size=len(f1_values))
    ax.scatter(jitter, f1_values, s=20, color=COLOR_PALETTE['primary'], alpha=0.6)
    ax.set_xlim(0.6, 1.4)
    ax.set_ylabel("F1 score")
    ax.set_xticks([])
    ax.set_title("Per-contig F1 distribution")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "per_contig_f1.png")
    plt.savefig(filepath, dpi=600, bbox_inches="tight")
    plt.close()
    return filepath


def generate_precision_recall_scatter(
    validation_metrics: Optional[Dict],
    output_dir: str
) -> str:
    """Scatter precision vs recall per contig with iso-F1 curves."""
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


def generate_lineage_count_error(
    validation_metrics: Optional[Dict],
    output_dir: str
) -> str:
    """Scatter of detected vs true lineage counts per contig."""
    if not HAS_MATPLOTLIB:
        return ""

    validation_metrics = _require_validation_metrics(validation_metrics, "Lineage count agreement")
    per_contig = _require_per_contig(validation_metrics, "Lineage count agreement")

    xs = []
    ys = []
    for metrics in per_contig.values():
        n_true = metrics.get("n_true")
        n_detected = metrics.get("n_detected")
        if n_true is None or n_detected is None:
            continue
        xs.append(n_true)
        ys.append(n_detected)

    if not xs:
        raise ValueError("Lineage count agreement requires n_true and n_detected.")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys, s=40, color=COLOR_PALETTE['success'], edgecolor=COLOR_PALETTE['dark'])
    min_v = min(xs + ys)
    max_v = max(xs + ys)
    ax.plot([min_v, max_v], [min_v, max_v], color=COLOR_PALETTE['neutral'], linestyle='--', linewidth=1.2)
    ax.set_xlabel("True lineages")
    ax.set_ylabel("Detected lineages")
    ax.set_title("Lineage count agreement by contig")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "lineage_count_error.png")
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


def generate_timepoint_abundance(
    validation_metrics: Optional[Dict],
    output_dir: str
) -> str:
    """Plot abundance correlation per timepoint."""
    if not HAS_MATPLOTLIB:
        return ""

    validation_metrics = _require_validation_metrics(
        validation_metrics, "Abundance correlation over time"
    )
    per_tp = _require_per_timepoint(validation_metrics, "Abundance correlation over time")

    pairs = []
    for tp, metrics in per_tp.items():
        val = metrics.get("abundance_pearson_r")
        if val is not None:
            pairs.append((tp, val))

    if not pairs:
        raise ValueError("Abundance correlation over time requires abundance_pearson_r values.")

    def _parse_tp(value: str) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    numeric_tps = [_parse_tp(tp) for tp, _ in pairs]
    if all(tp is not None for tp in numeric_tps):
        pairs = sorted(((numeric_tps[i], pairs[i][1]) for i in range(len(pairs))), key=lambda x: x[0])
        timepoints = [tp for tp, _ in pairs]
    else:
        pairs = sorted(pairs, key=lambda x: str(x[0]))
        timepoints = [tp for tp, _ in pairs]

    values = [val for _, val in pairs]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(timepoints, values, marker='o', color=COLOR_PALETTE['accent'])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Abundance Pearson r")
    ax.set_title("Abundance correlation over time")
    plt.tight_layout()
    filepath = os.path.join(output_dir, "abundance_timepoint.png")
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
        raise ValueError("Error decomposition has no non-zero values to plot.")

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
    figures['runtime_scaling.png'] = generate_runtime_scaling(results, output_dir)
    figures['per_contig_f1.png'] = generate_per_contig_f1_distribution(
        validation_metrics, output_dir
    )
    figures['precision_recall_scatter.png'] = generate_precision_recall_scatter(
        validation_metrics, output_dir
    )
    figures['lineage_count_error.png'] = generate_lineage_count_error(
        validation_metrics, output_dir
    )
    figures['pareto_front.png'] = generate_pareto_front(results, output_dir)
    figures['abundance_timepoint.png'] = generate_timepoint_abundance(
        validation_metrics, output_dir
    )
    figures['complexity_comparison.png'] = generate_complexity_comparison(results, summary, output_dir)
    figures['optimal_params.png'] = generate_optimal_params(stable_params, output_dir)
    figures['ablation_summary.png'] = generate_ablation_summary(results, output_dir)
    figures['seed_sensitivity.png'] = generate_seed_sensitivity(results, output_dir)
    figures['vcf_robustness.png'] = generate_vcf_robustness(results, output_dir)
    figures['coverage_performance.png'] = generate_coverage_performance(results, output_dir)
    figures['metric_correlation.png'] = generate_metric_correlation(results, output_dir)
    figures['error_decomposition.png'] = generate_error_decomposition(
        validation_metrics, output_dir
    )
    
    figure_titles = {
        'parameter_heatmap.png': 'Parameter Heatmap',
        'parameter_sensitivity.png': 'Parameter Sensitivity Analysis',
        'runtime_scaling.png': 'Runtime Scaling',
        'per_contig_f1.png': 'Per-contig F1 Distribution',
        'precision_recall_scatter.png': 'Per-contig Precision vs Recall',
        'lineage_count_error.png': 'Lineage Count Agreement',
        'pareto_front.png': 'Performance vs Runtime (Pareto Front)',
        'abundance_timepoint.png': 'Abundance Correlation Over Time',
        'complexity_comparison.png': 'Performance by Complexity',
        'optimal_params.png': 'Optimal Parameter Ranges',
        'ablation_summary.png': 'Ablation Summary',
        'seed_sensitivity.png': 'Seed Sensitivity',
        'vcf_robustness.png': 'VCF Robustness',
        'coverage_performance.png': 'Performance vs Coverage',
        'metric_correlation.png': 'Metric Correlation Matrix',
        'error_decomposition.png': 'Error Decomposition',
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
    validation_dir: Optional[str] = None
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

    if validation_dir:
        validation_metrics = load_validation_metrics(validation_dir)
        validation_files = {
            "haplotype_accuracy.png": "Haplotype Detection Accuracy",
            "abundance_correlation.png": "Abundance Correlation",
            "detection_sensitivity.png": "Detection Sensitivity",
            "confusion_matrix.png": "Haplotype Confusion Matrix",
            "detailed_matching.png": "Detailed Matching Analysis",
            "abundance_trajectories.png": "Abundance Trajectories",
            "track_fragmentation.png": "Track Fragmentation",
            "linking_errors.png": "Linking Errors",
            "lineage_accuracy.png": "Lineage Accuracy",
            "track_regions.png": "Track Regions on Contigs",
            "per_abundance_performance.png": "Performance by Abundance Range",
            "divergence_performance.png": "Performance vs Strain Divergence",
            "detection_roc.png": "Detection Performance (ROC-like)",
            "reference_coverage.png": "Reference Coverage Distribution",
            "error_breakdown.png": "Error Type Breakdown",
            "scalability_analysis.png": "Scalability Analysis",
        }
        for filename, title in validation_files.items():
            src_path = os.path.join(validation_dir, filename)
            if os.path.exists(src_path):
                dest_path = os.path.join(output_dir, filename)
                shutil.copy2(src_path, dest_path)
                validation_figures[dest_path] = title

    logger.info("Generating parameter heatmap...")
    figures['parameter_heatmap.png'] = generate_parameter_heatmap(results, output_dir)

    logger.info("Generating parameter sensitivity plot...")
    figures['parameter_sensitivity.png'] = generate_parameter_sensitivity(results, output_dir)

    logger.info("Generating runtime scaling plot...")
    figures['runtime_scaling.png'] = generate_runtime_scaling(results, output_dir)

    logger.info("Generating per-contig F1 distribution...")
    figures['per_contig_f1.png'] = generate_per_contig_f1_distribution(
        validation_metrics, output_dir
    )

    logger.info("Generating precision-recall scatter...")
    figures['precision_recall_scatter.png'] = generate_precision_recall_scatter(
        validation_metrics, output_dir
    )

    logger.info("Generating lineage count agreement plot...")
    figures['lineage_count_error.png'] = generate_lineage_count_error(
        validation_metrics, output_dir
    )

    logger.info("Generating Pareto front plot...")
    figures['pareto_front.png'] = generate_pareto_front(results, output_dir)

    logger.info("Generating timepoint abundance plot...")
    figures['abundance_timepoint.png'] = generate_timepoint_abundance(
        validation_metrics, output_dir
    )

    logger.info("Generating complexity comparison...")
    figures['complexity_comparison.png'] = generate_complexity_comparison(
        results, summary, output_dir
    )

    logger.info("Generating optimal params visualization...")
    figures['optimal_params.png'] = generate_optimal_params(stable_params, output_dir)
    
    logger.info("Generating ablation summary...")
    figures['ablation_summary.png'] = generate_ablation_summary(results, output_dir)
    
    logger.info("Generating seed sensitivity plot...")
    figures['seed_sensitivity.png'] = generate_seed_sensitivity(results, output_dir)
    
    logger.info("Generating VCF robustness plot...")
    figures['vcf_robustness.png'] = generate_vcf_robustness(results, output_dir)
    
    # Additional publication-quality plots
    logger.info("Generating performance vs coverage plot...")
    figures['coverage_performance.png'] = generate_coverage_performance(results, output_dir)
    
    logger.info("Generating metric correlation matrix...")
    figures['metric_correlation.png'] = generate_metric_correlation(results, output_dir)

    logger.info("Generating error decomposition plot...")
    figures['error_decomposition.png'] = generate_error_decomposition(
        validation_metrics, output_dir
    )

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

    args = parser.parse_args()

    report_path = generate_report(
        results_dir=args.results,
        output_dir=args.output,
        validation_dir=args.validation
    )

    if report_path:
        print(f"\nReport generated: {report_path}")
    else:
        print("\nFailed to generate report")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
