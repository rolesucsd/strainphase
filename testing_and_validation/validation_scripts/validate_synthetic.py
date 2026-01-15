#!/usr/bin/env python3
"""
Validation script for Strainphase using synthetic data.

Generates synthetic scenarios, runs Strainphase, and compares results
to ground truth to compute accuracy metrics.

Usage:
    python scripts/validate_synthetic.py --output results/validation/
    python scripts/validate_synthetic.py --quick  # Fast validation
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strainphase import HaplotyperConfig, process_window, link_windows
from strainphase.simulation import (
    SyntheticDataGenerator,
    SimulationScenario,
    create_test_scenarios,
    TrueHaplotype
)
from strainphase.core import WindowResult, Haplotype

import numpy as np


class ValidationMetrics:
    """Compute and store validation metrics."""

    def __init__(self):
        self.metrics = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'n_haplotypes_true': [],
            'n_haplotypes_detected': [],
            'abundance_mae': [],  # Mean absolute error in abundances
            'consensus_accuracy': [],
        }

    def compute_metrics(
        self,
        scenario: SimulationScenario,
        results: List[WindowResult],
        timepoint: str
    ) -> Dict:
        """
        Compute precision, recall, F1 for haplotype detection.

        Matching criteria:
        - Haplotypes match if consensus distance < 0.05 (5% mismatches)
        - Abundance within 0.15 (15%) of true value
        """
        # Get detected haplotypes (aggregate across windows)
        detected_haps = self._aggregate_detected_haplotypes(results)

        # Get true haplotypes for this timepoint
        true_haps = [
            h for h in scenario.true_haplotypes
            if h.get_abundance(timepoint) > 0.01  # >1% abundance
        ]

        # Match detected to true haplotypes
        matches = self._match_haplotypes(
            true_haps, detected_haps, scenario.snv_positions
        )

        # Compute metrics
        n_true = len(true_haps)
        n_detected = len(detected_haps)
        n_matched = len(matches)

        precision = n_matched / n_detected if n_detected > 0 else 0.0
        recall = n_matched / n_true if n_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Compute abundance MAE for matched haplotypes
        abundance_errors = []
        for true_hap, det_hap in matches:
            true_abund = true_hap.get_abundance(timepoint)
            det_abund = det_hap.weight
            abundance_errors.append(abs(true_abund - det_abund))

        abundance_mae = np.mean(abundance_errors) if abundance_errors else 0.0

        # Compute consensus accuracy for matched haplotypes
        consensus_accuracies = []
        for true_hap, det_hap in matches:
            shared_pos = set(true_hap.consensus.keys()) & set(det_hap.consensus.keys())
            if shared_pos:
                matches_count = sum(
                    1 for pos in shared_pos
                    if true_hap.consensus[pos] == det_hap.consensus[pos]
                )
                accuracy = matches_count / len(shared_pos)
                consensus_accuracies.append(accuracy)

        consensus_acc = np.mean(consensus_accuracies) if consensus_accuracies else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_true': n_true,
            'n_detected': n_detected,
            'n_matched': n_matched,
            'abundance_mae': abundance_mae,
            'consensus_accuracy': consensus_acc,
        }

    def _aggregate_detected_haplotypes(
        self, results: List[WindowResult]
    ) -> List[Haplotype]:
        """
        Aggregate haplotypes across windows by track_id.
        Returns one haplotype per track with merged consensus.
        """
        from collections import defaultdict

        tracks = defaultdict(list)
        for wr in results:
            for hap in wr.haplotypes:
                tid = hap.track_id or f"unlinked_{id(hap)}"
                tracks[tid].append(hap)

        # Merge haplotypes in each track
        merged = []
        for track_id, haps in tracks.items():
            # Merge consensus with weighted voting
            position_votes = defaultdict(lambda: defaultdict(float))
            total_weight = 0.0

            for hap in haps:
                total_weight += hap.weight
                for pos, base in hap.consensus.items():
                    position_votes[pos][base] += hap.weight

            consensus = {
                pos: max(votes.keys(), key=lambda b: votes[b])
                for pos, votes in position_votes.items()
            }

            # Create merged haplotype
            merged_hap = Haplotype(
                consensus=consensus,
                weight=total_weight / len(haps),
                supporting_reads=sum(h.supporting_reads for h in haps),
                confidence=np.mean([h.confidence for h in haps]),
                track_id=track_id
            )
            merged.append(merged_hap)

        return merged

    def _match_haplotypes(
        self,
        true_haps: List[TrueHaplotype],
        detected_haps: List[Haplotype],
        all_positions: List[int]
    ) -> List[Tuple[TrueHaplotype, Haplotype]]:
        """
        Match detected haplotypes to true haplotypes using Hungarian algorithm.

        Returns list of (true_hap, detected_hap) pairs.
        """
        if not true_haps or not detected_haps:
            return []

        # Compute distance matrix
        n_true = len(true_haps)
        n_det = len(detected_haps)

        dist_matrix = np.ones((n_true, n_det))

        for i, true_hap in enumerate(true_haps):
            for j, det_hap in enumerate(detected_haps):
                # Compute consensus distance
                shared_pos = set(true_hap.consensus.keys()) & set(det_hap.consensus.keys())
                if not shared_pos:
                    dist_matrix[i, j] = 1.0
                    continue

                mismatches = sum(
                    1 for pos in shared_pos
                    if true_hap.consensus[pos] != det_hap.consensus[pos]
                )
                dist = mismatches / len(shared_pos)
                dist_matrix[i, j] = dist

        # Simple greedy matching (good enough for validation)
        matches = []
        used_detected = set()

        for i in range(n_true):
            best_j = None
            best_dist = 0.05  # Max distance threshold

            for j in range(n_det):
                if j in used_detected:
                    continue
                if dist_matrix[i, j] < best_dist:
                    best_dist = dist_matrix[i, j]
                    best_j = j

            if best_j is not None:
                matches.append((true_haps[i], detected_haps[best_j]))
                used_detected.add(best_j)

        return matches

    def add_result(self, metrics: Dict):
        """Add metrics from one scenario."""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'mean_precision': np.mean(self.metrics['precision']),
            'mean_recall': np.mean(self.metrics['recall']),
            'mean_f1': np.mean(self.metrics['f1_score']),
            'mean_abundance_mae': np.mean(self.metrics['abundance_mae']),
            'mean_consensus_accuracy': np.mean(self.metrics['consensus_accuracy']),
            'all_metrics': self.metrics,
        }


def validate_scenario(
    scenario: SimulationScenario,
    config: HaplotyperConfig,
    output_dir: Path
) -> Dict:
    """
    Run validation on a single scenario.

    Returns summary metrics.
    """
    logging.info(f"Validating scenario: {scenario.name}")

    generator = SyntheticDataGenerator(seed=config.random_seed or 42)
    validator = ValidationMetrics()

    # Process each timepoint
    for timepoint in scenario.timepoints:
        logging.info(f"  Processing timepoint {timepoint}")

        # Generate window for this timepoint
        window = generator.generate_window(
            scenario=scenario,
            timepoint=timepoint,
            window_start=1,
            window_end=scenario.contig_length,
            n_reads=200,
            coverage=50,
            read_length=10000,
            error_rate=0.001,
        )

        # Process window
        result = process_window(window, config)

        # Link windows (in this case just one, but uses same code path)
        results = link_windows([result], config)

        # Compute metrics
        metrics = validator.compute_metrics(scenario, results, timepoint)
        validator.add_result(metrics)

        logging.info(f"    Precision: {metrics['precision']:.3f}")
        logging.info(f"    Recall: {metrics['recall']:.3f}")
        logging.info(f"    F1: {metrics['f1_score']:.3f}")

    # Get summary
    summary = validator.summary()

    # Save results
    output_file = output_dir / f"{scenario.name}_validation.json"
    with open(output_file, 'w') as f:
        json.dump({
            'scenario': scenario.name,
            'n_haplotypes': scenario.n_true_haplotypes(),
            'n_timepoints': len(scenario.timepoints),
            'summary': {
                'mean_precision': summary['mean_precision'],
                'mean_recall': summary['mean_recall'],
                'mean_f1': summary['mean_f1'],
                'mean_abundance_mae': summary['mean_abundance_mae'],
                'mean_consensus_accuracy': summary['mean_consensus_accuracy'],
            },
            'per_timepoint': {
                tp: {
                    'precision': summary['all_metrics']['precision'][i],
                    'recall': summary['all_metrics']['recall'][i],
                    'f1_score': summary['all_metrics']['f1_score'][i],
                }
                for i, tp in enumerate(scenario.timepoints)
            }
        }, f, indent=2)

    logging.info(f"Saved results to {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate Strainphase on synthetic data"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/validation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (fewer scenarios)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create scenarios
    if args.quick:
        scenarios = [
            create_test_scenarios()['simple_2hap'],
        ]
    else:
        scenario_dict = create_test_scenarios()
        scenarios = list(scenario_dict.values())

    # Configuration
    config = HaplotyperConfig(
        window_size=10000,
        max_reads_per_window=200,
        random_seed=args.seed,
        validate_results=True,
    )

    # Run validation
    all_results = []
    for scenario in scenarios:
        summary = validate_scenario(scenario, config, output_dir)
        all_results.append({
            'scenario': scenario.name,
            'summary': summary
        })

    # Overall summary
    overall = {
        'n_scenarios': len(scenarios),
        'mean_precision': np.mean([r['summary']['mean_precision'] for r in all_results]),
        'mean_recall': np.mean([r['summary']['mean_recall'] for r in all_results]),
        'mean_f1': np.mean([r['summary']['mean_f1'] for r in all_results]),
        'per_scenario': all_results
    }

    # Save overall summary
    summary_file = output_dir / "validation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(overall, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Scenarios tested: {overall['n_scenarios']}")
    print(f"Mean Precision:   {overall['mean_precision']:.3f}")
    print(f"Mean Recall:      {overall['mean_recall']:.3f}")
    print(f"Mean F1 Score:    {overall['mean_f1']:.3f}")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
