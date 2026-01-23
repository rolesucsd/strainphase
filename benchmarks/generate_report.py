#!/usr/bin/env python3
"""
Generate benchmark report with figures and HTML summary.

Takes results from parameter_sweep.py and validation outputs to generate:
- Validation figures (haplotype accuracy, abundance correlation, etc.)
- Benchmarking figures (parameter heatmaps, sensitivity plots, etc.)
- HTML report with embedded figures and recommendations

Usage:
    python benchmarks/generate_report.py \
        --results benchmarks/sweep_results/ \
        --output benchmarks/report/
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

    return results, summary, stable


def load_validation_metrics(validation_dir: str) -> Optional[Dict]:
    """Load validation metrics if available."""
    metrics_file = Path(validation_dir) / "validation_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None


# =============================================================================
# Figure generation
# =============================================================================

def set_figure_style():
    """Set consistent figure style."""
    if not HAS_MATPLOTLIB:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.25
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


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
    if not HAS_MATPLOTLIB or not results:
        return ""

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
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')

    ax.set_xticks(range(len(param1_vals)))
    ax.set_xticklabels([f"{v:.3f}" for v in param1_vals], rotation=45, ha='right')
    ax.set_yticks(range(len(param2_vals)))
    ax.set_yticklabels([str(v) for v in param2_vals])

    ax.set_xlabel('Max Mismatch Fraction')
    ax.set_ylabel('Min Shared SNVs')
    ax.set_title(f'Parameter Heatmap: {metric}')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)

    # Add value annotations
    vmax = np.max(matrix) if matrix.size else 0
    for i in range(len(param2_vals)):
        for j in range(len(param1_vals)):
            value = matrix[i, j]
            text_color = "white" if vmax > 0 and value > (vmax * 0.6) else "black"
            ax.text(j, i, f"{value:.1f}",
                    ha="center", va="center", color=text_color, fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'parameter_heatmap.png')
    plt.savefig(filepath, dpi=150)
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
    if not HAS_MATPLOTLIB or not results:
        return ""

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

        ax.errorbar(x, y_mean, yerr=y_std, marker='o', capsize=5, linewidth=2)
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Sensitivity: {param}')

    # Remove unused subplot
    if len(params_to_plot) < len(axes):
        for idx in range(len(params_to_plot), len(axes)):
            fig.delaxes(axes[idx])

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'parameter_sensitivity.png')
    plt.savefig(filepath, dpi=150)
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
    ax1.hist(runtimes, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Runtime (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title('Runtime Distribution')
    ax1.axvline(np.mean(runtimes), color='red', linestyle='--',
                label=f'Mean: {np.mean(runtimes):.2f}s')
    ax1.legend()

    # Runtime vs lineages
    ax2.scatter(n_lineages, runtimes, alpha=0.5, s=30)
    ax2.set_xlabel('Number of Lineages')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('Runtime vs Complexity')

    # Add trend line (only if there's variance in the data)
    if len(n_lineages) > 2 and len(set(n_lineages)) > 1:
        try:
            z = np.polyfit(n_lineages, runtimes, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(n_lineages), max(n_lineages), 100)
            ax2.plot(x_line, p(x_line), 'r--', alpha=0.8, label='Trend')
            ax2.legend()
        except (np.linalg.LinAlgError, ValueError):
            pass  # Skip trend line if fitting fails

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'runtime_scaling.png')
    plt.savefig(filepath, dpi=150)
    plt.close()

    return filepath


def generate_complexity_comparison(
    results: List[Dict],
    summary: Dict,
    output_dir: str
) -> str:
    """
    Generate grouped bar chart comparing metrics across complexity levels.

    Returns path to saved figure.
    """
    if not HAS_MATPLOTLIB:
        return ""

    scenarios = summary.get('scenarios', {})
    if not scenarios:
        return ""

    scenario_names = list(scenarios.keys())
    metrics = ['n_lineages', 'sweep_detection', 'converged_fraction']
    metric_labels = ['Mean Lineages', 'Sweep Detection Rate', 'Convergence Rate']

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#4c78a8', '#f58518', '#54a24b']

    x = np.arange(len(scenario_names))
    width = 0.25

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = []
        for name in scenario_names:
            stats = scenarios[name]
            if metric == 'n_lineages':
                values.append(stats.get('n_lineages', {}).get('mean', 0))
            elif metric == 'sweep_detection':
                values.append(stats.get('sweep_detection', {}).get('detection_rate', 0))
            else:
                values.append(stats.get(metric, 0))

        ax.bar(x + i * width, values, width, label=label, color=colors[i % len(colors)])

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Value')
    ax.set_title('Performance Across Complexity Levels')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenario_names, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'complexity_comparison.png')
    plt.savefig(filepath, dpi=150)
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
    if not HAS_MATPLOTLIB or not stable_params:
        return ""

    params = list(stable_params[0].keys())
    n_params = len(params)

    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    for idx, param in enumerate(params):
        ax = axes[idx]
        values = [p.get(param, 0) for p in stable_params]

        ax.hist(values, bins=10, color='green', edgecolor='black', alpha=0.7)
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Count')
        ax.set_title(f'Stable Range')

        # Mark optimal (most common)
        if values:
            optimal = max(set(values), key=values.count)
            ax.axvline(optimal, color='red', linestyle='--',
                      label=f'Optimal: {optimal}')
            ax.legend()

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'optimal_params.png')
    plt.savefig(filepath, dpi=150)
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
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            background: #ecf0f1;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
        }}
        .figure-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .figure-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .recommendation {{
            background: #e8f6f3;
            border-left: 4px solid #1abc9c;
            padding: 15px;
            margin: 20px 0;
        }}
        .warning {{
            background: #fef9e7;
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 20px 0;
        }}
        code {{
            background: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
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

    figure_titles = {
        'parameter_heatmap.png': 'Parameter Heatmap',
        'parameter_sensitivity.png': 'Parameter Sensitivity Analysis',
        'runtime_scaling.png': 'Runtime Scaling',
        'complexity_comparison.png': 'Performance by Complexity',
        'optimal_params.png': 'Optimal Parameter Ranges',
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

    # Failure mode analysis
    if summary.get('scenarios'):
        issues = []
        for name, stats in summary['scenarios'].items():
            if stats.get('sweep_detection', {}).get('detection_rate', 1) < 0.5:
                issues.append(f"{name}: low sweep detection rate")
            if stats.get('converged_fraction', 1) < 0.9:
                issues.append(f"{name}: low convergence")
            if stats.get('n_lineages', {}).get('std', 0) > 1.0:
                issues.append(f"{name}: high lineage variance")
            if stats.get('haplotype_f1') is not None and stats.get('haplotype_f1') < 0.8:
                issues.append(f"{name}: haplotype F1 below 0.8")
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

    if summary.get('scenarios'):
        logger.info("Generating complexity comparison...")
        figures['complexity_comparison.png'] = generate_complexity_comparison(
            results, summary, output_dir
        )

    if stable_params:
        logger.info("Generating optimal params visualization...")
        figures['optimal_params.png'] = generate_optimal_params(stable_params, output_dir)

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
