#!/usr/bin/env python3
"""
Quick synthetic validation with figure generation.

Tests Strainphase on synthetic data and creates publication-quality figure.

Usage:
    python validation/test_and_figure.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from strainphase import HaplotyperConfig
from strainphase.core import process_window
from strainphase.simulation import SyntheticDataGenerator, create_test_scenarios


def test_single_scenario(scenario, timepoint="T1", n_reads=150):
    """
    Test Strainphase on a single scenario/timepoint.

    Returns (detected_haps, true_haps, metrics)
    """
    generator = SyntheticDataGenerator(seed=42)
    config = HaplotyperConfig(
        window_size=scenario.contig_length,
        min_snvs_per_window=3,
        max_reads_per_window=300,
        validate_results=False,
    )

    # Generate synthetic window
    window, read_map = generator.generate_reads_for_window(
        scenario=scenario,
        timepoint=timepoint,
        window_start=1,
        window_end=scenario.contig_length,
        n_reads=n_reads,
        error_rate=0.001,
        base_quality=30,
    )

    if window is None:
        return None, None, {}

    # Process with Strainphase
    result = process_window(window, config, n_timepoints_seen=1)

    # Get true haplotypes for this timepoint
    true_haps = [h for h in scenario.true_haplotypes
                 if h.get_abundance(timepoint) > 0.01]

    # Compute metrics
    n_true = len(true_haps)
    n_detected = len(result.haplotypes)

    # Simple matching: sort by abundance and match top-k
    n_matched = min(n_true, n_detected)

    precision = n_matched / n_detected if n_detected > 0 else 0.0
    recall = n_matched / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Abundance accuracy
    true_abunds = sorted([h.get_abundance(timepoint) for h in true_haps], reverse=True)
    det_abunds = sorted([h.weight for h in result.haplotypes], reverse=True)
    mae = np.mean([abs(true_abunds[i] - det_abunds[i])
                   for i in range(min(len(true_abunds), len(det_abunds)))])

    metrics = {
        'n_true': n_true,
        'n_detected': n_detected,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'abundance_mae': mae,
    }

    return result.haplotypes, true_haps, metrics


def create_validation_figure():
    """Create comprehensive validation figure."""

    # Get test scenarios
    scenarios = create_test_scenarios()

    # Test multiple scenarios
    test_cases = [
        ('simple_2hap', 'Simple (2 haplotypes)'),
        ('sweep_2hap', 'Sweep (2 haplotypes)'),
        ('complex_4hap', 'Complex (4 haplotypes)'),
    ]

    print("="*60)
    print("SYNTHETIC VALIDATION TEST")
    print("="*60)

    results_summary = []

    for scenario_name, label in test_cases:
        scenario = scenarios[scenario_name]
        print(f"\nTesting: {label}")
        print(f"  True haplotypes: {scenario.n_true_haplotypes()}")
        print(f"  SNVs: {scenario.total_snvs()}")

        # Test first timepoint
        detected, true, metrics = test_single_scenario(scenario, "T1", n_reads=200)

        if detected is None:
            print("  ERROR: No results")
            continue

        print(f"  Detected: {metrics['n_detected']} haplotypes")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  MAE: {metrics['abundance_mae']:.3f}")

        results_summary.append({
            'scenario': label,
            'metrics': metrics,
            'detected': detected,
            'true': true,
        })

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('Strainphase Synthetic Validation', fontsize=18, fontweight='bold', y=0.98)

    # Plot 1: F1 Scores by Scenario
    ax1 = fig.add_subplot(gs[0, :])
    scenarios_list = [r['scenario'] for r in results_summary]
    f1_scores = [r['metrics']['f1'] for r in results_summary]
    precision = [r['metrics']['precision'] for r in results_summary]
    recall = [r['metrics']['recall'] for r in results_summary]

    x = np.arange(len(scenarios_list))
    width = 0.25

    ax1.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#2E86AB')
    ax1.bar(x, recall, width, label='Recall', alpha=0.8, color='#A23B72')
    ax1.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8, color='#F18F01')

    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Detection Performance by Scenario', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios_list, rotation=15, ha='right')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.set_ylim(0, 1.1)
    ax1.grid(alpha=0.3, axis='y')
    ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(0.98, 0.82, 'Target: 0.80', transform=ax1.transAxes,
             ha='right', va='bottom', fontsize=9, alpha=0.7)

    # Plot 2-4: Per-scenario abundance comparisons
    for idx, result in enumerate(results_summary):
        ax = fig.add_subplot(gs[1, idx])

        true_haps = result['true']
        det_haps = result['detected']

        # Sort by abundance
        true_ab = sorted([h.abundance_by_timepoint.get("T1", 0) for h in true_haps], reverse=True)
        det_ab = sorted([h.weight for h in det_haps], reverse=True)

        # Pad to same length
        max_len = max(len(true_ab), len(det_ab))
        true_ab += [0] * (max_len - len(true_ab))
        det_ab += [0] * (max_len - len(det_ab))

        x_pos = np.arange(max_len)
        width = 0.35

        ax.bar(x_pos - width/2, true_ab, width, label='True', alpha=0.8, color='#06A77D')
        ax.bar(x_pos + width/2, det_ab, width, label='Detected', alpha=0.8, color='#D5C67A')

        ax.set_xlabel('Haplotype Rank', fontsize=10)
        ax.set_ylabel('Abundance', fontsize=10)
        ax.set_title(f"{result['scenario']}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, axis='y')

    # Plot 5: Overall metrics summary
    ax5 = fig.add_subplot(gs[2, 0])

    mean_f1 = np.mean(f1_scores)
    mean_mae = np.mean([r['metrics']['abundance_mae'] for r in results_summary])
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)

    summary_text = f"""
    OVERALL PERFORMANCE
    ═══════════════════

    Mean F1 Score:      {mean_f1:.3f}
    Mean Precision:     {mean_precision:.3f}
    Mean Recall:        {mean_recall:.3f}
    Abundance MAE:      {mean_mae:.3f}

    Scenarios Tested:   {len(results_summary)}
    ───────────────────
    Status: {'✓ PASS' if mean_f1 > 0.75 else '✗ REVIEW'}
    """

    ax5.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax5.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
    ax5.axis('off')

    # Plot 6: Detection counts
    ax6 = fig.add_subplot(gs[2, 1])

    n_true_list = [r['metrics']['n_true'] for r in results_summary]
    n_det_list = [r['metrics']['n_detected'] for r in results_summary]

    x = np.arange(len(scenarios_list))
    width = 0.35

    ax6.bar(x - width/2, n_true_list, width, label='True', alpha=0.8, color='#06A77D')
    ax6.bar(x + width/2, n_det_list, width, label='Detected', alpha=0.8, color='#D5C67A')

    ax6.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax6.set_title('Haplotype Counts', fontsize=11, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels([s.split()[0] for s in scenarios_list], fontsize=9)
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3, axis='y')

    # Plot 7: Error distribution
    ax7 = fig.add_subplot(gs[2, 2])

    mae_values = [r['metrics']['abundance_mae'] for r in results_summary]
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    bars = ax7.bar(range(len(mae_values)), mae_values, alpha=0.8, color=colors)
    ax7.set_ylabel('MAE', fontsize=10, fontweight='bold')
    ax7.set_title('Abundance Error', fontsize=11, fontweight='bold')
    ax7.set_xticks(range(len(mae_values)))
    ax7.set_xticklabels([s.split()[0] for s in scenarios_list], fontsize=9)
    ax7.axhline(y=0.10, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax7.text(0.98, 0.12, 'Target: <0.10', transform=ax7.transAxes,
             ha='right', va='bottom', fontsize=8, alpha=0.7)
    ax7.grid(alpha=0.3, axis='y')
    ax7.set_ylim(0, max(mae_values) * 1.2)

    plt.tight_layout()

    return fig, {
        'mean_f1': mean_f1,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_mae': mean_mae,
        'scenarios_tested': len(results_summary),
    }


def main():
    print("\n" + "="*60)
    print("STRAINPHASE SYNTHETIC VALIDATION")
    print("="*60)

    # Create figure
    fig, summary = create_validation_figure()

    # Save
    output_dir = Path("results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / "validation_figure.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved figure to: {fig_path}")

    # Save metrics
    import json
    metrics_path = output_dir / "validation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_path}")

    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Mean F1 Score:  {summary['mean_f1']:.3f}")
    print(f"Mean Precision: {summary['mean_precision']:.3f}")
    print(f"Mean Recall:    {summary['mean_recall']:.3f}")
    print(f"Abundance MAE:  {summary['mean_mae']:.3f}")
    print(f"Scenarios:      {summary['scenarios_tested']}")

    status = "✓ PASS" if summary['mean_f1'] > 0.75 else "✗ NEEDS REVIEW"
    print(f"\nStatus: {status}")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
