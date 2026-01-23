#!/usr/bin/env python3
"""
Parameter sweep framework for haplotyper pipeline.

Tests pipeline stability across a grid of parameters:
- mismatch thresholds: 0.5%, 1%, 2%, 4%
- MAPQ: 10, 20, 30
- shared SNVs: 2, 3, 4
- merge distance: 0.5%, 1%, 2%
- anchor weight: 10%, 20%, 30%

Evaluates:
- Number of lineages inferred
- Major lineage frequency trajectories
- Sweep detection stability

Requires file-based input (BAM/VCF). Use run_full_benchmark.py to generate
simulated data from real genomes and run the complete benchmark pipeline.

Usage:
    python benchmarks/parameter_sweep.py \
        --bam data/simulated/T1.bam \
        --vcf data/simulated/T1.vcf.gz \
        --truth data/simulated/ \
        --output benchmarks/sweep_results/
"""

import argparse
import itertools
import json
import logging
import time
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict
from pathlib import Path

import numpy as np

from strainphase.core import (
    HaplotyperConfig,
    link_windows,
    WindowResult,
)

# Import pysam for BAM/VCF processing
try:
    import pysam
    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ParameterSet:
    """A single parameter configuration to test."""
    max_mismatch_frac: float
    min_mapq: int
    min_base_quality: int
    min_shared_snvs_for_edge: int
    merge_distance_threshold: float
    min_weight_for_anchor: float
    rescued_min_weight: float

    def to_config(self, base_config: Optional[HaplotyperConfig] = None) -> HaplotyperConfig:
        """Convert to HaplotyperConfig."""
        if base_config is None:
            base_config = HaplotyperConfig()

        return HaplotyperConfig(
            # Parameters we're varying
            max_mismatch_frac=self.max_mismatch_frac,
            min_mapq=self.min_mapq,
            min_base_quality=self.min_base_quality,
            min_shared_snvs_for_edge=self.min_shared_snvs_for_edge,
            merge_distance_threshold=self.merge_distance_threshold,
            min_weight_for_anchor=self.min_weight_for_anchor,
            rescued_min_weight=self.rescued_min_weight,

            # Related parameters (keep consistent)
            max_link_distance=self.max_mismatch_frac,
            min_shared_snvs_for_link=self.min_shared_snvs_for_edge,
            min_shared_for_merge=self.min_shared_snvs_for_edge,
            min_shared_for_rescue=self.min_shared_snvs_for_edge,
            rescue_match_distance=self.merge_distance_threshold,
            lineage_merge_distance=self.max_mismatch_frac,
            min_shared_for_lineage=self.min_shared_snvs_for_edge,

            # Fixed parameters
            window_size=base_config.window_size,
            min_snvs_per_window=base_config.min_snvs_per_window,
            min_reads_per_window=base_config.min_reads_per_window,
            em_max_iter=base_config.em_max_iter,
            validate_results=False,  # Faster for sweep
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def short_name(self) -> str:
        return (f"mm{self.max_mismatch_frac:.3f}_"
                f"mq{self.min_mapq}_"
                f"bq{self.min_base_quality}_"
                f"snv{self.min_shared_snvs_for_edge}_"
                f"md{self.merge_distance_threshold:.3f}_"
                f"aw{self.min_weight_for_anchor:.2f}_"
                f"rw{self.rescued_min_weight:.2f}")


@dataclass
class SweepResult:
    """Results from a single parameter configuration run."""
    params: ParameterSet
    scenario_name: str

    # Lineage statistics
    n_lineages: int
    n_tracks_per_timepoint: Dict[str, int]

    # Abundance trajectories (lineage_id -> {timepoint -> abundance})
    lineage_trajectories: Dict[str, Dict[str, float]]

    # Sweep detection
    sweep_detected: bool
    sweep_winner: Optional[str]
    sweep_loser: Optional[str]

    # Runtime
    runtime_seconds: float

    # Quality metrics
    converged: bool
    mean_confidence: float

    # Accuracy metrics (when ground truth available)
    haplotype_precision: Optional[float] = None
    haplotype_recall: Optional[float] = None
    haplotype_f1: Optional[float] = None
    abundance_pearson_r: Optional[float] = None
    abundance_mae: Optional[float] = None
    snv_precision: Optional[float] = None
    snv_recall: Optional[float] = None
    snv_f1: Optional[float] = None
    memory_peak_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'params': self.params.to_dict(),
            'scenario_name': self.scenario_name,
            'n_lineages': self.n_lineages,
            'n_tracks_per_timepoint': self.n_tracks_per_timepoint,
            'lineage_trajectories': self.lineage_trajectories,
            'sweep_detected': self.sweep_detected,
            'sweep_winner': self.sweep_winner,
            'sweep_loser': self.sweep_loser,
            'runtime_seconds': self.runtime_seconds,
            'converged': self.converged,
            'mean_confidence': self.mean_confidence,
            'haplotype_precision': self.haplotype_precision,
            'haplotype_recall': self.haplotype_recall,
            'haplotype_f1': self.haplotype_f1,
            'abundance_pearson_r': self.abundance_pearson_r,
            'abundance_mae': self.abundance_mae,
            'snv_precision': self.snv_precision,
            'snv_recall': self.snv_recall,
            'snv_f1': self.snv_f1,
            'memory_peak_mb': self.memory_peak_mb,
        }


# =============================================================================
# Ground truth loading
# =============================================================================

def load_ground_truth_snvs(truth_dir: str) -> Dict[str, Dict[int, Dict[str, str]]]:
    """
    Load ground truth SNVs from simulation output.

    Returns: {contig: {pos: {strain_id: allele}}}
    """
    truth_path = Path(truth_dir)

    vcf_file = truth_path / "truth_snvs.vcf"
    if not vcf_file.exists():
        vcf_file = truth_path / "truth_variants.vcf"
    snvs: Dict[str, Dict[int, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))

    if not vcf_file.exists():
        return {}

    with open(vcf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            contig = parts[0]
            pos = int(parts[1]) - 1  # Convert to 0-indexed
            info = parts[7]

            # Parse STRAINS info field
            if 'STRAINS=' in info:
                strains_info = info.split('STRAINS=')[1].split(';')[0]
                for allele_info in strains_info.split(';'):
                    if ':' in allele_info:
                        allele, strain_list = allele_info.split(':')
                        for strain_id in strain_list.split(','):
                            snvs[contig][pos][strain_id] = allele

    return dict(snvs)


def load_ground_truth_abundances(truth_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Load ground truth abundances from simulation output.

    Returns: {strain_id: {timepoint: abundance}}
    """
    abundances: Dict[str, Dict[str, float]] = defaultdict(dict)

    abundances_file = Path(truth_dir) / "truth_abundances.tsv"
    if not abundances_file.exists():
        return {}

    with open(abundances_file) as f:
        header = f.readline().strip().split('\t')
        timepoints = header[1:]  # First column is strain_id
        for line in f:
            parts = line.strip().split('\t')
            strain_id = parts[0]
            for i, tp in enumerate(timepoints):
                abundances[strain_id][tp] = float(parts[i + 1])

    return dict(abundances)


# =============================================================================
# BAM/VCF processing
# =============================================================================

def get_contigs_from_bam(bam_path: str) -> Dict[str, int]:
    """Get contig names and lengths from BAM header."""
    if not HAS_PYSAM:
        raise ImportError("pysam required for BAM processing")

    contigs = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for sq in bam.header.get("SQ", []):
            contigs[sq["SN"]] = sq["LN"]
    return contigs


def process_contig_with_params(
    bam_path: str,
    vcf_path: str,
    contig_id: str,
    contig_length: int,
    params: ParameterSet,
    sample_id: str = "sample"
) -> List[WindowResult]:
    """
    Process a single contig with given parameter set.

    Uses strainphase.core.process_contig internally.
    """
    from strainphase.core import process_contig

    config = params.to_config()
    config.window_size = 10000  # Fixed window size for sweep

    try:
        results = process_contig(
            bam_path=bam_path,
            vcf_path=vcf_path,
            contig_id=contig_id,
            contig_length=contig_length,
            config=config,
            sample_id=sample_id
        )
        return results or []
    except Exception as e:
        logger.warning(f"Error processing {contig_id}: {e}")
        return []


# =============================================================================
# Parameter sweep class
# =============================================================================

class ParameterSweep:
    """
    Run pipeline across parameter grid and analyze stability.

    Requires file-based input (BAM/VCF files).
    """

    # Default parameter grid
    DEFAULT_GRID = {
        'max_mismatch_frac': [0.005, 0.01, 0.02, 0.04],
        'min_mapq': [10, 20, 30],
        'min_base_quality': [20, 30],
        'min_shared_snvs_for_edge': [2, 3, 4, 5],
        'merge_distance_threshold': [0.005, 0.01, 0.02],
        'min_weight_for_anchor': [0.05, 0.10, 0.15, 0.20],
        'rescued_min_weight': [0.01, 0.02, 0.05],
    }

    def __init__(
        self,
        grid: Optional[Dict[str, List]] = None,
        seed: int = 42,
    ):
        self.grid = grid or self.DEFAULT_GRID
        self.seed = seed
        self.results: List[SweepResult] = []

        # Data attributes
        self.bam_path: Optional[str] = None
        self.vcf_path: Optional[str] = None
        self.truth_dir: Optional[str] = None
        self.truth_snvs: Optional[Dict] = None
        self.truth_abundances: Optional[Dict] = None

    def generate_parameter_sets(self) -> List[ParameterSet]:
        """Generate all parameter combinations."""
        keys = list(self.grid.keys())
        values = [self.grid[k] for k in keys]

        param_sets = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            param_sets.append(ParameterSet(**param_dict))

        return param_sets

    def run_sweep(
        self,
        bam_path: str,
        vcf_path: str,
        truth_dir: Optional[str] = None,
        max_configs: Optional[int] = None,
        max_contigs: Optional[int] = None,
        verbose: bool = True
    ) -> List[SweepResult]:
        """
        Run parameter sweep on BAM/VCF data.

        Args:
            bam_path: Path to BAM file
            vcf_path: Path to VCF file
            truth_dir: Optional path to ground truth directory (from simulation)
            max_configs: Limit number of parameter configs to test
            max_contigs: Limit number of contigs to process
            verbose: Print progress
        """
        if not HAS_PYSAM:
            raise ImportError("pysam required for BAM/VCF processing")

        self.bam_path = bam_path
        self.vcf_path = vcf_path
        self.truth_dir = truth_dir

        # Load ground truth if provided
        if truth_dir:
            logger.info(f"Loading ground truth from {truth_dir}")
            self.truth_snvs = load_ground_truth_snvs(truth_dir)
            self.truth_abundances = load_ground_truth_abundances(truth_dir)

        # Get contigs from BAM
        contigs = get_contigs_from_bam(bam_path)
        if max_contigs:
            contig_list = list(contigs.items())[:max_contigs]
            contigs = dict(contig_list)

        logger.info(f"Processing {len(contigs)} contigs")

        param_sets = self.generate_parameter_sets()
        if max_configs:
            param_sets = param_sets[:max_configs]

        total_runs = len(param_sets)
        if verbose:
            print(f"Running {len(param_sets)} parameter configs on {len(contigs)} contigs")

        self.results = []

        for run_idx, params in enumerate(param_sets, 1):
            if verbose and run_idx % 5 == 0:
                print(f"  Progress: {run_idx}/{total_runs}")

            start_time = time.time()

            try:
                # Process all contigs with this parameter set
                all_window_results: List[WindowResult] = []

                for contig_id, contig_length in contigs.items():
                    results = process_contig_with_params(
                        bam_path, vcf_path, contig_id, contig_length,
                        params, sample_id="sample"
                    )
                    all_window_results.extend(results)

                runtime = time.time() - start_time

                # Extract metrics from results
                track_ids = {h.track_id for wr in all_window_results for h in wr.haplotypes if h.track_id}
                n_lineages = len(track_ids)

                # Compute accuracy if truth available
                accuracy_metrics = {}
                if self.truth_snvs:
                    accuracy_metrics = self._compute_accuracy_vs_truth(
                        all_window_results, self.truth_snvs
                    )

                # Build trajectories for single timepoint
                trajectories = {}
                for wr in all_window_results:
                    for hap in wr.haplotypes:
                        tid = hap.track_id or f"unlinked_{wr.window.start}"
                        if tid not in trajectories:
                            trajectories[tid] = {}
                        trajectories[tid]["T1"] = hap.weight

                result = SweepResult(
                    params=params,
                    scenario_name="file_based",
                    n_lineages=n_lineages,
                    n_tracks_per_timepoint={"T1": n_lineages},
                    lineage_trajectories=trajectories,
                    sweep_detected=False,
                    sweep_winner=None,
                    sweep_loser=None,
                    runtime_seconds=runtime,
                    converged=all(wr.converged for wr in all_window_results) if all_window_results else False,
                    mean_confidence=np.mean([h.confidence for wr in all_window_results for h in wr.haplotypes]) if all_window_results else 0.0,
                    snv_precision=accuracy_metrics.get("snv_precision"),
                    snv_recall=accuracy_metrics.get("snv_recall"),
                    snv_f1=accuracy_metrics.get("snv_f1"),
                )

                self.results.append(result)

            except Exception as e:
                if verbose:
                    print(f"    Error with {params.short_name()}: {e}")
                logger.exception(f"Error with params {params.short_name()}")

        return self.results

    def _compute_accuracy_vs_truth(
        self,
        window_results: List[WindowResult],
        truth_snvs: Dict[str, Dict[int, Dict[str, str]]]
    ) -> Dict[str, float]:
        """
        Compute accuracy metrics against ground truth SNVs.
        """
        total_correct = 0
        total_detected = 0
        total_true = 0

        for wr in window_results:
            contig = wr.window.contig if hasattr(wr.window, 'contig') else "unknown"
            true_snvs_for_contig = truth_snvs.get(contig, {})

            for hap in wr.haplotypes:
                for pos, allele in hap.consensus.items():
                    total_detected += 1
                    if pos in true_snvs_for_contig:
                        # Check if any strain has this allele
                        for strain_allele in true_snvs_for_contig[pos].values():
                            if strain_allele == allele:
                                total_correct += 1
                                break

        for contig_snvs in truth_snvs.values():
            total_true += len(contig_snvs)

        precision = total_correct / total_detected if total_detected > 0 else 0.0
        recall = total_correct / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "snv_precision": precision,
            "snv_recall": recall,
            "snv_f1": f1,
        }

    def summarize_results(self) -> Dict[str, Any]:
        """
        Summarize sweep results to assess stability.
        """
        if not self.results:
            return {'error': 'No results to summarize'}

        # Group by scenario
        by_scenario: Dict[str, List[SweepResult]] = defaultdict(list)
        for r in self.results:
            by_scenario[r.scenario_name].append(r)

        summary = {
            'n_configs_tested': len(self.generate_parameter_sets()),
            'n_scenarios': len(by_scenario),
            'scenarios': {}
        }

        best_by_scenario = {}

        for scenario_name, results in by_scenario.items():
            n_lineages = [r.n_lineages for r in results]
            sweep_detected = [r.sweep_detected for r in results]
            runtimes = [r.runtime_seconds for r in results]
            confidences = [r.mean_confidence for r in results]
            hap_f1 = [r.haplotype_f1 for r in results if r.haplotype_f1 is not None]
            snv_f1 = [r.snv_f1 for r in results if r.snv_f1 is not None]
            abundance_r = [r.abundance_pearson_r for r in results if r.abundance_pearson_r is not None]
            abundance_mae = [r.abundance_mae for r in results if r.abundance_mae is not None]

            summary['scenarios'][scenario_name] = {
                'n_configs': len(results),
                'n_lineages': {
                    'min': min(n_lineages) if n_lineages else 0,
                    'max': max(n_lineages) if n_lineages else 0,
                    'mean': float(np.mean(n_lineages)) if n_lineages else 0,
                    'std': float(np.std(n_lineages)) if n_lineages else 0,
                    'mode': max(set(n_lineages), key=n_lineages.count) if n_lineages else 0,
                },
                'sweep_detection': {
                    'detected_count': sum(sweep_detected),
                    'detection_rate': float(np.mean(sweep_detected)) if sweep_detected else 0,
                },
                'runtime': {
                    'mean': float(np.mean(runtimes)) if runtimes else 0,
                    'std': float(np.std(runtimes)) if runtimes else 0,
                },
                'confidence': {
                    'mean': float(np.mean(confidences)) if confidences else 0,
                    'min': float(min(confidences)) if confidences else 0,
                },
                'converged_fraction': float(np.mean([r.converged for r in results])) if results else 0,
                'haplotype_f1': float(np.mean(hap_f1)) if hap_f1 else None,
                'snv_f1': float(np.mean(snv_f1)) if snv_f1 else None,
                'abundance_pearson_r': float(np.mean(abundance_r)) if abundance_r else None,
                'abundance_mae': float(np.mean(abundance_mae)) if abundance_mae else None,
            }

            # Pick best parameter set for this scenario
            def score_result(res: SweepResult) -> float:
                score = 0.0
                if res.haplotype_f1 is not None:
                    score += res.haplotype_f1 * 2.0
                if res.snv_f1 is not None:
                    score += res.snv_f1
                if res.abundance_pearson_r is not None:
                    score += res.abundance_pearson_r * 0.5
                score += 0.2 if res.converged else 0.0
                score += res.mean_confidence * 0.1
                score += 0.5 if res.sweep_detected else 0.0
                return score

            best = max(results, key=score_result)
            best_by_scenario[scenario_name] = {
                "params": best.params.to_dict(),
                "score": score_result(best),
                "haplotype_f1": best.haplotype_f1,
                "snv_f1": best.snv_f1,
                "abundance_pearson_r": best.abundance_pearson_r,
                "abundance_mae": best.abundance_mae,
            }

        summary["best_params_by_scenario"] = best_by_scenario

        return summary

    def identify_stable_parameters(self) -> List[ParameterSet]:
        """
        Identify parameter configurations that give stable results.

        "Stable" means:
        - Consistent n_lineages across scenarios
        - Good convergence
        """
        if not self.results:
            return []

        # Group results by parameter config
        by_config: Dict[str, List[SweepResult]] = defaultdict(list)
        for r in self.results:
            key = r.params.short_name()
            by_config[key].append(r)

        stable_params = []

        for config_name, results in by_config.items():
            # Check stability criteria
            n_lineages = [r.n_lineages for r in results]
            lineage_std = np.std(n_lineages) if n_lineages else float('inf')

            # Check convergence
            convergence_rate = np.mean([r.converged for r in results]) if results else 0

            # Stable if low variance and good convergence
            if lineage_std < 2.0 and convergence_rate >= 0.8:
                stable_params.append(results[0].params)

        return stable_params


def run_parameter_sweep(
    bam_path: str,
    vcf_path: str,
    output_dir: str = "benchmarks/results",
    truth_dir: Optional[str] = None,
    params_file: Optional[str] = None,
    max_configs: Optional[int] = None,
    max_contigs: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete parameter sweep and save results.

    Args:
        bam_path: BAM file path
        vcf_path: VCF file path
        output_dir: Output directory for results
        truth_dir: Ground truth directory (optional, for accuracy metrics)
        params_file: Custom parameter grid JSON file (optional)
        max_configs: Limit number of configs to test
        max_contigs: Limit number of contigs
        verbose: Print progress
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load custom parameter grid if provided
    grid = None
    if params_file:
        with open(params_file) as f:
            grid = json.load(f)

    sweep = ParameterSweep(seed=42, grid=grid)

    print(f"Starting parameter sweep on: {bam_path}")
    results = sweep.run_sweep(
        bam_path=bam_path,
        vcf_path=vcf_path,
        truth_dir=truth_dir,
        max_configs=max_configs,
        max_contigs=max_contigs,
        verbose=verbose
    )

    # Save raw results
    results_data = [r.to_dict() for r in results]
    with open(os.path.join(output_dir, 'sweep_results.json'), 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    # Save metrics in documented schema
    metrics_payload = []
    for r in results:
        metrics_payload.append({
            "params": r.params.to_dict(),
            "community": r.scenario_name,
            "metrics": {
                "haplotype_precision": r.haplotype_precision,
                "haplotype_recall": r.haplotype_recall,
                "haplotype_f1": r.haplotype_f1,
                "abundance_pearson_r": r.abundance_pearson_r,
                "abundance_mae": r.abundance_mae,
                "snv_precision": r.snv_precision,
                "snv_recall": r.snv_recall,
                "runtime_seconds": r.runtime_seconds,
                "memory_peak_mb": r.memory_peak_mb,
            }
        })

    with open(os.path.join(output_dir, 'benchmark_metrics.json'), 'w') as f:
        json.dump(metrics_payload, f, indent=2, default=str)

    # Generate summary
    summary = sweep.summarize_results()
    with open(os.path.join(output_dir, 'sweep_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Identify stable parameters
    stable = sweep.identify_stable_parameters()
    stable_data = [p.to_dict() for p in stable]
    with open(os.path.join(output_dir, 'stable_parameters.json'), 'w') as f:
        json.dump(stable_data, f, indent=2)

    if verbose:
        print(f"\n=== SWEEP SUMMARY ===")
        print(f"Total configs tested: {len(results)}")
        print(f"Stable parameter sets found: {len(stable)}")

        for scenario_name, stats in summary.get('scenarios', {}).items():
            print(f"\n{scenario_name}:")
            n_lin = stats.get('n_lineages', {})
            print(f"  Lineages: {n_lin.get('mean', 0):.1f} ± {n_lin.get('std', 0):.1f} "
                  f"(range {n_lin.get('min', 0)}-{n_lin.get('max', 0)})")
            print(f"  Convergence: {stats.get('converged_fraction', 0):.1%}")
            if stats.get('snv_f1') is not None:
                print(f"  SNV F1: {stats.get('snv_f1'):.3f}")

    print(f"\nResults saved to: {output_dir}")
    return summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parameter sweep for strainphase haplotyper (file-based mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required inputs
    parser.add_argument("--bam", dest="bam_path", required=True,
                        help="Input BAM file")
    parser.add_argument("--vcf", dest="vcf_path", required=True,
                        help="Input VCF file")

    # Optional inputs
    parser.add_argument("--truth", dest="truth_dir",
                        help="Ground truth directory from simulation (optional)")
    parser.add_argument("--params", dest="params_file",
                        help="Custom parameter grid JSON file (optional)")

    # Output
    parser.add_argument("--output", "-o", default="benchmarks/results",
                        help="Output directory for results")

    # Limits
    parser.add_argument("--max-configs", type=int,
                        help="Limit number of parameter configs to test")
    parser.add_argument("--max-contigs", type=int,
                        help="Limit number of contigs to process")

    # Verbosity
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Reduce output verbosity")

    args = parser.parse_args()

    run_parameter_sweep(
        bam_path=args.bam_path,
        vcf_path=args.vcf_path,
        output_dir=args.output,
        truth_dir=args.truth_dir,
        params_file=args.params_file,
        max_configs=args.max_configs,
        max_contigs=args.max_contigs,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
