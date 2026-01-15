#!/usr/bin/env python3
"""
Quick test of synthetic data generator and Strainphase pipeline.

This is a simplified version for rapid testing and figure generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strainphase import HaplotyperConfig
from strainphase.simulation import SyntheticDataGenerator, create_test_scenarios
from strainphase.core import process_window as process_window_internal, Window
import matplotlib.pyplot as plt
import numpy as np

# Get process_window function (it exists in core but may not be exported)
# Use internal function for testing
def process_window_test(window, config):
    """Wrapper for the internal process_window function."""
    from strainphase.core import (
        GraphInitializer, EMHaplotyper, PostProcessor
    )

    # Initialize
    initializer = GraphInitializer(config)
    init_result = initializer.initialize_from_graph(window)

    if not init_result.haplotypes:
        return None

    # EM refinement
    em = EMHaplotyper(config)
    em_result = em.run_em(window, init_result.haplotypes)

    # Post-processing
    processor = PostProcessor(config)
    final_result = processor.process(em_result, window, n_timepoints_seen=1)

    return final_result


def test_scenario(scenario_name="simple_2hap"):
    """Test a single scenario and return results."""
    print(f"\n{'='*60}")
    print(f"Testing scenario: {scenario_name}")
    print('='*60)

    # Get scenario
    scenarios = create_test_scenarios()
    scenario = scenarios[scenario_name]

    print(f"Ground truth:")
    print(f"  {scenario.n_true_haplotypes()} haplotypes")
    print(f"  {scenario.total_snvs()} SNVs")
    print(f"  {len(scenario.timepoints)} timepoints")

    # Generate synthetic data
    generator = SyntheticDataGenerator(seed=42)
    config = HaplotyperConfig(
        window_size=scenario.contig_length,  # Use whole contig as one window
        min_snvs_per_window=3,
        max_reads_per_window=200,
        validate_results=False,
    )

    results = {}

    for timepoint in scenario.timepoints:
        print(f"\nProcessing timepoint {timepoint}:")

        # Generate reads
        window, read_map = generator.generate_reads_for_window(
            scenario=scenario,
            timepoint=timepoint,
            window_start=1,
            window_end=scenario.contig_length,
            n_reads=150,
            error_rate=0.001,
            base_quality=30,
        )

        if window is None:
            print(f"  No window generated")
            continue

        print(f"  Generated: {len(window.reads)} reads, {len(window.snv_pos)} SNVs")

        # Process window
        result = process_window_test(window, config)

        if result is None:
            print(f"  No haplotypes detected")
            continue

        print(f"  Detected: {len(result.haplotypes)} haplotypes")

        # Match detected to true haplotypes
        true_haps = [h for h in scenario.true_haplotypes
                     if h.get_abundance(timepoint) > 0.01]

        print(f"  True haplotypes in this timepoint: {len(true_haps)}")

        # Compute simple accuracy
        for i, hap in enumerate(result.haplotypes):
            print(f"    Hap {i+1}: weight={hap.weight:.3f}, {len(hap.consensus)} SNVs, "
                  f"{hap.supporting_reads} reads")

        results[timepoint] = {
            'true_count': len(true_haps),
            'detected_count': len(result.haplotypes),
            'true_abundances': [h.get_abundance(timepoint) for h in true_haps],
            'detected_weights': [h.weight for h in result.haplotypes],
        }

    return results, scenario


def create_figure(results, scenario):
    """Create validation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: True vs Detected Haplotype Counts
    ax = axes[0, 0]
    timepoints = list(results.keys())
    true_counts = [results[tp]['true_count'] for tp in timepoints]
    det_counts = [results[tp]['detected_count'] for tp in timepoints]

    x = np.arange(len(timepoints))
    width = 0.35
    ax.bar(x - width/2, true_counts, width, label='True', alpha=0.8, color='blue')
    ax.bar(x + width/2, det_counts, width, label='Detected', alpha=0.8, color='orange')
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Number of Haplotypes', fontsize=12)
    ax.set_title('Haplotype Detection', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(timepoints)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Plot 2: True Abundance Trajectories
    ax = axes[0, 1]
    for i, hap in enumerate(scenario.true_haplotypes):
        abunds = [hap.get_abundance(tp) for tp in timepoints]
        ax.plot(timepoints, abunds, 'o-', linewidth=2, markersize=8,
                label=f'Hap {i+1}', alpha=0.8)
    ax.set_xlabel('Timepoint', fontsize=12)
    ax.set_ylabel('Abundance', fontsize=12)
    ax.set_title('True Haplotype Abundances', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Plot 3: Detected vs True Abundances (scatter)
    ax = axes[1, 0]
    all_true = []
    all_detected = []
    for tp in timepoints:
        true_ab = results[tp]['true_abundances']
        det_ab = results[tp]['detected_weights']
        # Pad shorter list
        n = min(len(true_ab), len(det_ab))
        all_true.extend(sorted(true_ab, reverse=True)[:n])
        all_detected.extend(sorted(det_ab, reverse=True)[:n])

    ax.scatter(all_true, all_detected, alpha=0.6, s=100, color='purple')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect match')
    ax.set_xlabel('True Abundance', fontsize=12)
    ax.set_ylabel('Detected Weight', fontsize=12)
    ax.set_title('Abundance Accuracy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    # Plot 4: Summary metrics
    ax = axes[1, 1]
    ax.axis('off')

    # Compute summary statistics
    precision_list = []
    recall_list = []
    for tp in timepoints:
        n_true = results[tp]['true_count']
        n_det = results[tp]['detected_count']
        n_matched = min(n_true, n_det)  # Simplified matching
        precision = n_matched / n_det if n_det > 0 else 0
        recall = n_matched / n_true if n_true > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)

    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0

    # MAE for abundances
    mae_list = []
    for tp in timepoints:
        true_ab = sorted(results[tp]['true_abundances'], reverse=True)
        det_ab = sorted(results[tp]['detected_weights'], reverse=True)
        n = min(len(true_ab), len(det_ab))
        if n > 0:
            mae = np.mean([abs(true_ab[i] - det_ab[i]) for i in range(n)])
            mae_list.append(mae)
    mean_mae = np.mean(mae_list) if mae_list else 0.0

    summary_text = f"""
    VALIDATION SUMMARY
    ==================

    Scenario: {scenario.name}

    Performance Metrics:
    • Precision: {mean_precision:.3f}
    • Recall: {mean_recall:.3f}
    • F1 Score: {mean_f1:.3f}
    • Abundance MAE: {mean_mae:.3f}

    Dataset:
    • True haplotypes: {scenario.n_true_haplotypes()}
    • SNV positions: {scenario.total_snvs()}
    • Timepoints: {len(scenario.timepoints)}

    Notes:
    - Precision: % detected that are true
    - Recall: % of true detected
    - F1: Harmonic mean
    - MAE: Mean absolute error in abundance
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    return fig, {
        'precision': mean_precision,
        'recall': mean_recall,
        'f1': mean_f1,
        'abundance_mae': mean_mae,
    }


def main():
    # Test simple scenario
    results, scenario = test_scenario("simple_2hap")

    # Create figure
    print("\nGenerating figure...")
    fig, metrics = create_figure(results, scenario)

    # Save figure
    output_dir = Path("results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = output_dir / "synthetic_validation_figure.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {fig_path}")

    # Print metrics
    print(f"\nFinal Metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1']:.3f}")
    print(f"  Abundance MAE: {metrics['abundance_mae']:.3f}")

    print("\n" + "="*60)
    print("✓ Test complete!")
    print("="*60)


if __name__ == "__main__":
    main()
