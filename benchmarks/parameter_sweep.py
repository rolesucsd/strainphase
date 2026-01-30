#!/usr/bin/env python3
"""
Parameter sweep framework for haplotyper pipeline.

This module is used by run_full_benchmark.py to test parameter configurations.
It can also be run standalone for custom parameter sweeps.

Tests pipeline stability across a grid of parameters (see REQUIRED_GRID for full list).
Automatically runs validation for each configuration when truth_dir is provided.

Evaluates:
- Number of lineages inferred
- Major lineage frequency trajectories
- Sweep detection stability
- Validation metrics (precision, recall, F1, track/linking, lineage metrics)

Usage (standalone):
    python benchmarks/parameter_sweep.py \
        --bam data/simulated/T1.bam \
        --vcf data/simulated/T1.vcf.gz \
        --truth data/simulated/ \
        --output benchmarks/sweep_results/

Note: For full benchmarking pipeline, use run_full_benchmark.py instead.
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
    window_size: int
    # New parameters (optional, with defaults)
    max_reads_per_window: Optional[int] = None
    junk_divergence_rate: Optional[float] = None
    max_link_distance: Optional[float] = None
    min_shared_snvs_for_link: Optional[int] = None
    rescue_match_distance: Optional[float] = None
    lineage_merge_distance: Optional[float] = None

    def to_config(self, base_config: Optional[HaplotyperConfig] = None, n_workers: int = 1) -> HaplotyperConfig:
        """Convert to HaplotyperConfig."""
        if base_config is None:
            base_config = HaplotyperConfig()

        # Get values for new parameters (with defaults if not in ParameterSet)
        max_reads_per_window = self.max_reads_per_window if self.max_reads_per_window is not None else base_config.max_reads_per_window
        junk_divergence_rate = self.junk_divergence_rate if self.junk_divergence_rate is not None else base_config.junk_divergence_rate
        max_link_distance = self.max_link_distance if self.max_link_distance is not None else self.max_mismatch_frac
        min_shared_snvs_for_link = self.min_shared_snvs_for_link if self.min_shared_snvs_for_link is not None else self.min_shared_snvs_for_edge
        rescue_match_distance = self.rescue_match_distance if self.rescue_match_distance is not None else base_config.rescue_match_distance
        lineage_merge_distance = self.lineage_merge_distance if self.lineage_merge_distance is not None else base_config.lineage_merge_distance

        return HaplotyperConfig(
            # Parameters we're varying
            max_mismatch_frac=self.max_mismatch_frac,
            min_mapq=self.min_mapq,
            min_base_quality=self.min_base_quality,
            min_shared_snvs_for_edge=self.min_shared_snvs_for_edge,
            merge_distance_threshold=self.merge_distance_threshold,
            min_weight_for_anchor=self.min_weight_for_anchor,
            rescued_min_weight=self.rescued_min_weight,
            window_size=self.window_size,
            max_reads_per_window=max_reads_per_window,
            junk_divergence_rate=junk_divergence_rate,

            # Related parameters (keep consistent)
            max_link_distance=max_link_distance,
            min_shared_snvs_for_link=min_shared_snvs_for_link,
            min_shared_for_merge=self.min_shared_snvs_for_edge,
            min_shared_for_rescue=self.min_shared_snvs_for_edge,
            rescue_match_distance=rescue_match_distance,
            lineage_merge_distance=lineage_merge_distance,
            min_shared_for_lineage=self.min_shared_snvs_for_edge,

            # Fixed parameters
            min_snvs_per_window=base_config.min_snvs_per_window,
            min_reads_per_window=base_config.min_reads_per_window,
            em_max_iter=base_config.em_max_iter,
            validate_results=False,  # Faster for sweep
            n_workers=n_workers,  # Parallel window processing
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
                f"rw{self.rescued_min_weight:.2f}_"
                f"ws{self.window_size}")


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
    false_negatives_count: Optional[int] = None
    false_positives_count: Optional[int] = None
    snv_true_total: Optional[int] = None
    snv_true_in_span: Optional[int] = None
    snv_detected_total: Optional[int] = None
    snv_correct_total: Optional[int] = None
    snv_span_coverage_frac: Optional[float] = None
    
    # Track/linking metrics (when ground truth available)
    track_fragmentation_mean: Optional[float] = None
    track_fragmentation_median: Optional[float] = None
    false_link_rate: Optional[float] = None
    missed_link_rate: Optional[float] = None
    track_consensus_error: Optional[float] = None
    
    # Lineage metrics (when ground truth available)
    lineage_precision: Optional[float] = None
    lineage_recall: Optional[float] = None
    lineage_f1: Optional[float] = None
    rescue_delta_recall_rare: Optional[float] = None
    abundance_trajectory_error: Optional[float] = None
    rescued_haplotypes: Optional[int] = None
    rescue_total_haplotypes: Optional[int] = None
    rescue_rate: Optional[float] = None
    
    # Performance metrics
    memory_peak_mb: Optional[float] = None
    
    # Metadata
    ablation: Optional[str] = None  # e.g., "full", "no_linking", "no_rescue", "no_junk", "no_1snp_guard"
    vcf_condition: Optional[str] = None  # e.g., "perfect", "missing_af_dp", "fp_sites", "fn_sites"
    config_full: Optional[Dict] = None  # Full HaplotyperConfig fields
    environment: Optional[Dict] = None  # Software versions, platform, CPU, threads
    seed: Optional[int] = None

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
            'false_negatives_count': self.false_negatives_count,
            'false_positives_count': self.false_positives_count,
            'snv_true_total': self.snv_true_total,
            'snv_true_in_span': self.snv_true_in_span,
            'snv_detected_total': self.snv_detected_total,
            'snv_correct_total': self.snv_correct_total,
            'snv_span_coverage_frac': self.snv_span_coverage_frac,
            # Track/linking metrics
            'track_fragmentation_mean': self.track_fragmentation_mean,
            'track_fragmentation_median': self.track_fragmentation_median,
            'false_link_rate': self.false_link_rate,
            'missed_link_rate': self.missed_link_rate,
            'track_consensus_error': self.track_consensus_error,
            # Lineage metrics
            'lineage_precision': self.lineage_precision,
            'lineage_recall': self.lineage_recall,
            'lineage_f1': self.lineage_f1,
            'rescue_delta_recall_rare': self.rescue_delta_recall_rare,
            'abundance_trajectory_error': self.abundance_trajectory_error,
            'rescued_haplotypes': self.rescued_haplotypes,
            'rescue_total_haplotypes': self.rescue_total_haplotypes,
            'rescue_rate': self.rescue_rate,
            # Performance and metadata
            'memory_peak_mb': self.memory_peak_mb,
            'ablation': self.ablation,
            'vcf_condition': self.vcf_condition,
            'config_full': self.config_full,
            'environment': self.environment,
            'seed': self.seed,
        }


# =============================================================================
# Progress tracking and checkpointing
# =============================================================================

@dataclass
class SweepProgress:
    """Tracks sweep progress for checkpointing and resume."""
    total_configs: int
    completed_configs: int
    current_config_idx: int
    start_time: float
    last_save_time: float
    completed_config_keys: List[str]  # short_name() of completed configs
    mode: str  # "grid" or "sequential"
    sequential_state: Optional[Dict[str, Any]] = None

    def eta_seconds(self) -> Optional[float]:
        """Estimate time remaining based on average config time."""
        if self.completed_configs == 0:
            return None
        elapsed = time.time() - self.start_time
        avg_per_config = elapsed / self.completed_configs
        remaining = self.total_configs - self.completed_configs
        return avg_per_config * remaining

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_configs': self.total_configs,
            'completed_configs': self.completed_configs,
            'current_config_idx': self.current_config_idx,
            'start_time': self.start_time,
            'last_save_time': self.last_save_time,
            'completed_config_keys': self.completed_config_keys,
            'mode': self.mode,
            'sequential_state': self.sequential_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SweepProgress":
        return cls(**data)


class ProgressLogger:
    """Handles detailed progress logging with timing and ETAs."""

    def __init__(self, total_configs: int, total_contigs: int, verbose: bool = True):
        self.total_configs = total_configs
        self.total_contigs = total_contigs
        self.verbose = verbose
        self.config_start_time: Optional[float] = None
        self.sweep_start_time = time.time()
        self.config_times: List[float] = []

    def log_config_start(self, config_idx: int, params: ParameterSet):
        """Log start of a new parameter configuration."""
        self.config_start_time = time.time()
        if self.verbose:
            eta = self._calculate_eta(config_idx)
            eta_str = f" | ETA: {self._format_time(eta)}" if eta else ""
            logger.info(f"Config {config_idx}/{self.total_configs} [{params.short_name()}]{eta_str}")

    def log_contig_progress(self, config_idx: int, contig_idx: int,
                            contig_id: str, n_windows: int):
        """Log per-contig progress within a config."""
        if self.verbose:
            logger.info(f"  Contig {contig_idx}/{self.total_contigs}: {contig_id} "
                       f"({n_windows} windows)")

    def log_config_complete(self, config_idx: int, result: SweepResult):
        """Log completion of a config with metrics."""
        elapsed = time.time() - self.config_start_time if self.config_start_time else 0
        self.config_times.append(elapsed)
        if self.verbose:
            snv_f1_str = f"{result.snv_f1:.3f}" if result.snv_f1 is not None else "n/a"
            logger.info(f"  Completed in {elapsed:.1f}s | "
                       f"lineages={result.n_lineages}, "
                       f"converged={result.converged}, "
                       f"snv_f1={snv_f1_str}")

    def _calculate_eta(self, completed: int) -> Optional[float]:
        if not self.config_times:
            return None
        avg_time = np.mean(self.config_times)
        remaining = self.total_configs - completed
        return avg_time * remaining

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class CheckpointManager:
    """Manages saving and loading checkpoints for sweep resume."""

    def __init__(self, output_dir: str, save_interval: int = 10):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.config_results_dir = self.checkpoint_dir / "config_results"
        self.save_interval = save_interval
        self._configs_since_save = 0

    def setup(self):
        """Create checkpoint directories."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config_results_dir.mkdir(parents=True, exist_ok=True)

    def save_config_result(self, result: SweepResult):
        """Save individual config result immediately."""
        filename = f"{result.params.short_name()}.json"
        filepath = self.config_results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    def save_progress(self, progress: SweepProgress, force: bool = False):
        """Save progress state, respecting save interval."""
        self._configs_since_save += 1
        if not force and self._configs_since_save < self.save_interval:
            return

        progress.last_save_time = time.time()
        filepath = self.checkpoint_dir / "progress.json"
        with open(filepath, 'w') as f:
            json.dump(progress.to_dict(), f, indent=2, default=str)
        self._configs_since_save = 0
        logger.debug(f"Checkpoint saved: {progress.completed_configs}/{progress.total_configs}")

    def load_progress(self) -> Optional[SweepProgress]:
        """Load existing progress if available."""
        filepath = self.checkpoint_dir / "progress.json"
        if not filepath.exists():
            return None
        with open(filepath) as f:
            data = json.load(f)
        return SweepProgress.from_dict(data)

    def load_completed_results(self) -> List[SweepResult]:
        """Load all completed config results."""
        results = []
        for filepath in sorted(self.config_results_dir.glob("*.json")):
            with open(filepath) as f:
                data = json.load(f)
            # Reconstruct SweepResult from dict
            params = ParameterSet(**data['params'])
            result = SweepResult(
                params=params,
                scenario_name=data['scenario_name'],
                n_lineages=data['n_lineages'],
                n_tracks_per_timepoint=data['n_tracks_per_timepoint'],
                lineage_trajectories=data['lineage_trajectories'],
                sweep_detected=data['sweep_detected'],
                sweep_winner=data['sweep_winner'],
                sweep_loser=data['sweep_loser'],
                runtime_seconds=data['runtime_seconds'],
                converged=data['converged'],
                mean_confidence=data['mean_confidence'],
                haplotype_precision=data.get('haplotype_precision'),
                haplotype_recall=data.get('haplotype_recall'),
                haplotype_f1=data.get('haplotype_f1'),
                abundance_pearson_r=data.get('abundance_pearson_r'),
                abundance_mae=data.get('abundance_mae'),
                snv_precision=data.get('snv_precision'),
                snv_recall=data.get('snv_recall'),
                snv_f1=data.get('snv_f1'),
                false_negatives_count=data.get('false_negatives_count'),
                false_positives_count=data.get('false_positives_count'),
                snv_true_total=data.get('snv_true_total'),
                snv_true_in_span=data.get('snv_true_in_span'),
                snv_detected_total=data.get('snv_detected_total'),
                snv_correct_total=data.get('snv_correct_total'),
                snv_span_coverage_frac=data.get('snv_span_coverage_frac'),
                # Track/linking metrics
                track_fragmentation_mean=data.get('track_fragmentation_mean'),
                track_fragmentation_median=data.get('track_fragmentation_median'),
                false_link_rate=data.get('false_link_rate'),
                missed_link_rate=data.get('missed_link_rate'),
                track_consensus_error=data.get('track_consensus_error'),
                # Lineage metrics
                lineage_precision=data.get('lineage_precision'),
                lineage_recall=data.get('lineage_recall'),
                lineage_f1=data.get('lineage_f1'),
                rescue_delta_recall_rare=data.get('rescue_delta_recall_rare'),
                abundance_trajectory_error=data.get('abundance_trajectory_error'),
                # Metadata
                ablation=data.get('ablation'),
                vcf_condition=data.get('vcf_condition'),
                config_full=data.get('config_full'),
                environment=data.get('environment'),
                seed=data.get('seed'),
                memory_peak_mb=data.get('memory_peak_mb'),
            )
            results.append(result)
        return results

    def is_config_completed(self, params: ParameterSet) -> bool:
        """Check if a config has already been completed."""
        filename = f"{params.short_name()}.json"
        return (self.config_results_dir / filename).exists()

    def get_cached_result(self, params: ParameterSet) -> Optional[SweepResult]:
        """Load a specific cached result if it exists."""
        filename = f"{params.short_name()}.json"
        filepath = self.config_results_dir / filename
        if not filepath.exists():
            return None
        with open(filepath) as f:
            data = json.load(f)
        params_obj = ParameterSet(**data['params'])
        return SweepResult(
            params=params_obj,
            scenario_name=data['scenario_name'],
            n_lineages=data['n_lineages'],
            n_tracks_per_timepoint=data['n_tracks_per_timepoint'],
            lineage_trajectories=data['lineage_trajectories'],
            sweep_detected=data['sweep_detected'],
            sweep_winner=data['sweep_winner'],
            sweep_loser=data['sweep_loser'],
            runtime_seconds=data['runtime_seconds'],
            converged=data['converged'],
            mean_confidence=data['mean_confidence'],
            haplotype_precision=data.get('haplotype_precision'),
            haplotype_recall=data.get('haplotype_recall'),
            haplotype_f1=data.get('haplotype_f1'),
            abundance_pearson_r=data.get('abundance_pearson_r'),
            abundance_mae=data.get('abundance_mae'),
            snv_precision=data.get('snv_precision'),
            snv_recall=data.get('snv_recall'),
            snv_f1=data.get('snv_f1'),
            false_negatives_count=data.get('false_negatives_count'),
            false_positives_count=data.get('false_positives_count'),
            snv_true_total=data.get('snv_true_total'),
            snv_true_in_span=data.get('snv_true_in_span'),
            snv_detected_total=data.get('snv_detected_total'),
            snv_correct_total=data.get('snv_correct_total'),
            snv_span_coverage_frac=data.get('snv_span_coverage_frac'),
            # Track/linking metrics
            track_fragmentation_mean=data.get('track_fragmentation_mean'),
            track_fragmentation_median=data.get('track_fragmentation_median'),
            false_link_rate=data.get('false_link_rate'),
            missed_link_rate=data.get('missed_link_rate'),
            track_consensus_error=data.get('track_consensus_error'),
            # Lineage metrics
            lineage_precision=data.get('lineage_precision'),
            lineage_recall=data.get('lineage_recall'),
            lineage_f1=data.get('lineage_f1'),
            rescue_delta_recall_rare=data.get('rescue_delta_recall_rare'),
            abundance_trajectory_error=data.get('abundance_trajectory_error'),
            # Metadata
            ablation=data.get('ablation'),
            vcf_condition=data.get('vcf_condition'),
            config_full=data.get('config_full'),
            environment=data.get('environment'),
            seed=data.get('seed'),
            memory_peak_mb=data.get('memory_peak_mb'),
        )


# =============================================================================
# BAM/VCF processing
# =============================================================================
# Note: Ground truth loading is handled by validation modules (validate_haplotypes.py)
# which loads truth data directly from truth_dir when needed.

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
    sample_id: str = "sample",
    n_workers: int = 1,
) -> List[WindowResult]:
    """
    Process a single contig with given parameter set.

    Uses strainphase.core.process_contig internally.
    """
    from strainphase.core import process_contig

    config = params.to_config(n_workers=n_workers)

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


def window_results_to_lineages_tsv(
    all_window_results: List[WindowResult],
    output_path: str,
    sample_id: str = "sample"
) -> str:
    """
    Convert WindowResults to lineages.tsv format for validation.
    
    Creates a simplified lineages.tsv that aggregates haplotypes by track_id
    across all windows and contigs.
    
    If no haplotypes are detected, creates an empty file with headers.
    """
    import csv
    
    # Aggregate haplotypes by track_id
    track_data = defaultdict(lambda: {
        'abundances': {},
        'snvs': defaultdict(dict),  # contig -> {pos -> allele}
        'contigs': set(),
    })
    
    for wr in all_window_results:
        contig_id = wr.window.contig  # Window uses 'contig', not 'contig_id'
        for hap in wr.haplotypes:
            track_id = hap.track_id or f"unlinked_{wr.window.start}"
            
            track_data[track_id]['contigs'].add(contig_id)
            track_data[track_id]['abundances'][sample_id] = hap.weight
            
            # Aggregate SNV alleles from consensus
            for pos, allele in hap.consensus.items():
                track_data[track_id]['snvs'][contig_id][pos] = allele
    
    # Write lineages.tsv (create even if empty)
    records = []
    for track_id, data in track_data.items():
        for contig_id in data['contigs']:
            snv_alleles_str = ','.join(
                f"{pos}:{allele}" 
                for pos, allele in sorted(data['snvs'][contig_id].items())
            )
            
            records.append({
                'lineage_id': track_id,
                'sample': sample_id,
                'contig': contig_id,
                'track_id': track_id,
                'abundance': data['abundances'].get(sample_id, 0.0),
                'snv_alleles': snv_alleles_str if snv_alleles_str else '.',
            })
    
    # Always create the file, even if empty (for validation)
    fieldnames = ['lineage_id', 'sample', 'contig', 'track_id', 'abundance', 'snv_alleles']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        if records:
            writer.writerows(records)
    
    return output_path


def window_results_to_prelink_lineages_tsv(
    all_window_results: List[WindowResult],
    output_path: str,
    sample_id: str = "sample"
) -> str:
    """
    Convert WindowResults to a pre-linking lineages.tsv format for validation.

    Each haplotype is kept distinct per window (no linking across windows).
    """
    import csv

    records = []
    for wr in all_window_results:
        contig_id = wr.window.contig
        for h_idx, hap in enumerate(wr.haplotypes):
            lineage_id = f"W{wr.window.start}_H{h_idx}"
            sample = wr.window.sample or sample_id
            snv_alleles_str = ','.join(
                f"{pos}:{allele}" for pos, allele in sorted(hap.consensus.items())
            )
            records.append({
                'lineage_id': lineage_id,
                'sample': sample,
                'contig': contig_id,
                'track_id': lineage_id,
                'abundance': hap.weight,
                'snv_alleles': snv_alleles_str if snv_alleles_str else '.',
            })

    fieldnames = ['lineage_id', 'sample', 'contig', 'track_id', 'abundance', 'snv_alleles']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        if records:
            writer.writerows(records)

    return output_path


def write_validation_comparison_tsv(
    output_path: str,
    prelink: 'ValidationResult',
    postlink: 'ValidationResult'
) -> None:
    import csv

    rows = [
        ("haplotype_precision", prelink.precision, postlink.precision),
        ("haplotype_recall", prelink.recall, postlink.recall),
        ("haplotype_f1", prelink.f1, postlink.f1),
        ("snv_precision", prelink.snv_precision, postlink.snv_precision),
        ("snv_recall", prelink.snv_recall, postlink.snv_recall),
        ("abundance_pearson_r", prelink.abundance_pearson_r, postlink.abundance_pearson_r),
        ("abundance_mae", prelink.abundance_mae, postlink.abundance_mae),
        ("detection_threshold", prelink.detection_threshold, postlink.detection_threshold),
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["metric", "prelink", "postlink"])
        for metric, pre, post in rows:
            writer.writerow([metric, pre, post])


# =============================================================================
# Parameter sweep class
# =============================================================================

class ParameterSweep:
    """
    Run pipeline across parameter grid and analyze stability.

    Requires file-based input (BAM/VCF files).
    """

    # Required parameter grid (agents.md)
    REQUIRED_GRID = {
        # Windowing / subsampling
        # Minimum window_size is 10000 to ensure sufficient shared SNVs for reliable linking.
        # With ~1-2 SNVs/kb, smaller windows have too few SNVs in the 50% overlap region.
        'window_size': [10000, 20000, 50000, 100000],
        'max_reads_per_window': [500],
        
        # Clustering parameters
        'max_mismatch_frac': [0.005, 0.02],
        'min_shared_snvs_for_edge': [4],
        
        # Quality filters
        'min_mapq': [10],
        'min_base_quality': [20],
        
        # Junk model sensitivity (publication expansion)
        'junk_divergence_rate': [0.05, 0.10, 0.20],
        
        # Merging thresholds
        'merge_distance_threshold': [0.005, 0.02],
        
        # Linking thresholds (publication expansion)
        'max_link_distance': [0.01],
        'min_shared_snvs_for_link': [3, 4],
        
        # Abundance thresholds
        'min_weight_for_anchor': [0.20],
        'rescued_min_weight': [0.02],
    }

    # Parameter order for sequential optimization (most impactful first)
    DEFAULT_PARAM_ORDER = [
        'window_size',           # Most fundamental, affects all downstream
        'max_mismatch_frac',     # Core clustering parameter
        'min_shared_snvs_for_edge',  # Graph construction sensitivity
        'merge_distance_threshold',  # Post-processing clustering
        'min_mapq',              # Read quality filter
        'min_base_quality',      # SNV call quality
        'min_weight_for_anchor', # Longitudinal linking
        'rescued_min_weight',    # Recovery threshold
    ]

    # Default/intermediate starting values for sequential optimization
    # These match the best parameters found in benchmarking
    DEFAULT_START_VALUES = {
        'window_size': 20000,
        'max_mismatch_frac': 0.02,
        'min_shared_snvs_for_edge': 3,
        'merge_distance_threshold': 0.02,
        'min_mapq': 20, 
        'min_base_quality': 20,
        'min_weight_for_anchor': 0.10,
        'rescued_min_weight': 0.02,
    }

    def __init__(
        self,
        grid: Optional[Dict[str, List]] = None,
        seed: int = 42,
    ):
        if grid is None:
            self.grid = {k: list(v) for k, v in self.REQUIRED_GRID.items()}
        else:
            self._validate_grid(grid)
            self.grid = grid
        self.seed = seed
        self.results: List[SweepResult] = []

        # Data attributes
        self.bam_path: Optional[str] = None
        self.vcf_path: Optional[str] = None
        self.truth_dir: Optional[str] = None

    def _validate_grid(self, grid: Dict[str, List]) -> None:
        """Enforce that the parameter grid matches agents.md.
        
        Allows subsets of required values (for custom parameter files),
        but requires all keys to be present and values to be valid.
        """
        required_keys = set(self.REQUIRED_GRID.keys())
        provided_keys = set(grid.keys())
        if required_keys != provided_keys:
            missing = sorted(required_keys - provided_keys)
            extra = sorted(provided_keys - required_keys)
            raise ValueError(f"Parameter grid mismatch: missing={missing}, extra={extra}")

        for key, required_values in self.REQUIRED_GRID.items():
            provided_values = grid.get(key, [])
            if not provided_values:
                raise ValueError(
                    f"Parameter grid mismatch for {key}: "
                    f"empty value list not allowed"
                )
            # Check that all provided values are valid (subset of required)
            invalid_values = set(provided_values) - set(required_values)
            if invalid_values:
                raise ValueError(
                    f"Parameter grid mismatch for {key}: "
                    f"invalid values {sorted(invalid_values)} not in required set {required_values}. "
                    f"Provided: {provided_values}, Required: {required_values}"
                )

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
        bam_paths: Dict[str, str],  # {timepoint -> bam_path}
        vcf_paths: Dict[str, str],  # {timepoint -> vcf_path}
        reference_path: Optional[str] = None,  # Required for longitudinal
        timepoints: Optional[List[str]] = None,  # List of timepoint IDs
        truth_dir: Optional[str] = None,
        max_configs: Optional[int] = None,
        max_contigs: Optional[int] = None,
        verbose: bool = True,
        resume: bool = False,
        checkpoint_interval: int = 10,
        output_dir: Optional[str] = None,
        n_workers: int = 1,
    ) -> List[SweepResult]:
        """
        Run parameter sweep on BAM/VCF data with checkpointing.
        
        Supports both single-timepoint and longitudinal (multi-timepoint) modes.

        Args:
            bam_paths: Dict mapping timepoint IDs to BAM file paths
            vcf_paths: Dict mapping timepoint IDs to VCF file paths
            reference_path: Reference FASTA path (required for longitudinal mode)
            timepoints: List of timepoint IDs (e.g., ["T1", "T2"])
            truth_dir: Optional path to ground truth directory (from simulation)
            max_configs: Limit number of parameter configs to test
            max_contigs: Limit number of contigs to process
            verbose: Print progress
            resume: If True, resume from last checkpoint
            checkpoint_interval: How often to save progress (in configs)
            output_dir: Output directory for checkpoints (required if resume=True)
            n_workers: Number of parallel workers for window processing
        """
        if not HAS_PYSAM:
            raise ImportError("pysam required for BAM/VCF processing")

        # Determine if longitudinal mode (multiple timepoints)
        use_longitudinal = (reference_path is not None and 
                           timepoints is not None and 
                           len(timepoints) > 1)
        
        # Store paths as instance variables for use in processing
        self.bam_paths = bam_paths
        self.vcf_paths = vcf_paths
        self.reference_path = reference_path
        self.timepoints = timepoints
        self.use_longitudinal = use_longitudinal
        
        # Store paths for backward compatibility and validation
        first_timepoint = timepoints[0] if timepoints else list(bam_paths.keys())[0]
        self.bam_path = bam_paths[first_timepoint]  # For backward compatibility
        self.vcf_path = vcf_paths[first_timepoint]   # For backward compatibility
        self.truth_dir = truth_dir

        # Setup checkpointing if output_dir provided
        checkpoint_mgr = None
        if output_dir:
            checkpoint_mgr = CheckpointManager(output_dir, checkpoint_interval)
            checkpoint_mgr.setup()

        # Store truth_dir for validation (validation loads its own ground truth)
        self.truth_dir = truth_dir
        if truth_dir:
            logger.info(f"Ground truth directory: {truth_dir}")

        # Get contigs from first BAM (or reference for longitudinal)
        if use_longitudinal and reference_path:
            from strainphase.longitudinal import parse_reference_contigs
            mags = parse_reference_contigs(reference_path, allowed_contigs=None)
            # Flatten MAG structure to get all contigs
            contigs = {}
            for mag_contigs in mags.values():
                contigs.update(mag_contigs)
        else:
            contigs = get_contigs_from_bam(bam_paths[first_timepoint])
        if max_contigs:
            contig_list = list(contigs.items())[:max_contigs]
            contigs = dict(contig_list)

        logger.info(f"Processing {len(contigs)} contigs")
        if use_longitudinal:
            logger.info(f"Using longitudinal mode with {len(timepoints)} timepoints")

        param_sets = self.generate_parameter_sets()
        if max_configs:
            param_sets = param_sets[:max_configs]

        # Resume handling
        completed_keys: set = set()
        if resume and checkpoint_mgr:
            existing_progress = checkpoint_mgr.load_progress()
            if existing_progress:
                self.results = checkpoint_mgr.load_completed_results()
                completed_keys = set(existing_progress.completed_config_keys)
                logger.info(f"Resuming from checkpoint: {len(self.results)} configs already completed")
            else:
                self.results = []
        else:
            self.results = []

        # Filter out already completed configs
        remaining_param_sets = [p for p in param_sets if p.short_name() not in completed_keys]
        total_configs = len(param_sets)
        already_completed = len(param_sets) - len(remaining_param_sets)

        # Setup progress tracking
        progress_logger = ProgressLogger(total_configs, len(contigs), verbose)

        if verbose:
            logger.info(f"Running {len(remaining_param_sets)} parameter configs on {len(contigs)} contigs")
            if already_completed > 0:
                logger.info(f"  ({already_completed} configs already completed)")

        # Initialize progress state
        progress = SweepProgress(
            total_configs=total_configs,
            completed_configs=already_completed,
            current_config_idx=already_completed,
            start_time=time.time(),
            last_save_time=time.time(),
            completed_config_keys=list(completed_keys),
            mode="grid"
        )

        for run_idx, params in enumerate(remaining_param_sets, already_completed + 1):
            progress.current_config_idx = run_idx
            progress_logger.log_config_start(run_idx, params)

            start_time = time.time()

            try:
                # Process all contigs with this parameter set
                all_window_results: List[WindowResult] = []
                
                if use_longitudinal:
                    # Longitudinal mode: use process_mag_longitudinal
                    from strainphase.longitudinal import process_mag_longitudinal
                    from strainphase.core import HaplotyperConfig
                    
                    config = params.to_config(n_workers=n_workers)
                    
                    # Process MAG with longitudinal integration
                    # mag_contigs should be {contig_id: length}, not nested
                    mag_results = process_mag_longitudinal(
                        mag_name="MAG_01",
                        mag_contigs=contigs,  # Pass contigs directly, not nested dict
                        samples=timepoints,
                        bam_paths=bam_paths,
                        vcf_paths=vcf_paths,
                        config=config,
                    )
                    
                    # Log per-timepoint results from process_mag_longitudinal
                    logger.info(f"    process_mag_longitudinal returned results for {len(mag_results)} timepoints")
                    for sample_id in timepoints:
                        sample_contigs = mag_results.get(sample_id, {})
                        total_windows = sum(len(contig_results) for contig_results in sample_contigs.values())
                        total_haplotypes = sum(
                            len(wr.haplotypes) 
                            for contig_results in sample_contigs.values() 
                            for wr in contig_results
                        )
                        logger.info(f"    {sample_id}: {len(sample_contigs)} contigs, {total_windows} windows, {total_haplotypes} haplotypes")
                    
                    # Check if any timepoints are missing
                    missing_samples = set(timepoints) - set(mag_results.keys())
                    if missing_samples:
                        logger.warning(f"    WARNING: process_mag_longitudinal did not return results for: {sorted(missing_samples)}")
                    
                    # Flatten results: {sample -> {contig -> [WindowResult]}} -> List[WindowResult]
                    for sample_results in mag_results.values():
                        for contig_results in sample_results.values():
                            all_window_results.extend(contig_results)
                    
                    # Per-contig logging
                    for contig_idx, (contig_id, contig_length) in enumerate(contigs.items(), 1):
                        n_windows = sum(len(mag_results.get(sample, {}).get(contig_id, [])) 
                                      for sample in timepoints)
                        progress_logger.log_contig_progress(run_idx, contig_idx, contig_id, n_windows)
                else:
                    # Single-timepoint mode: process each contig individually
                    for contig_idx, (contig_id, contig_length) in enumerate(contigs.items(), 1):
                        results = process_contig_with_params(
                            bam_paths[first_timepoint], vcf_paths[first_timepoint], 
                            contig_id, contig_length,
                            params, sample_id=first_timepoint, n_workers=n_workers
                        )
                        all_window_results.extend(results)

                        # Per-contig logging
                        progress_logger.log_contig_progress(run_idx, contig_idx, contig_id, len(results))

                runtime = time.time() - start_time

                # Extract metrics from results
                track_ids = {h.track_id for wr in all_window_results for h in wr.haplotypes if h.track_id}
                n_lineages = len(track_ids)

                # Compute accuracy if truth available
                accuracy_metrics = {}
                validation_result = None
                if self.truth_dir:
                    # Create temporary lineages.tsv for validation
                    # Use output_dir if provided, otherwise create temp directory
                    if output_dir:
                        config_output_dir = Path(output_dir) / "configs" / params.short_name()
                    else:
                        # Create temp directory if output_dir not provided
                        import tempfile
                        temp_base = Path(tempfile.gettempdir()) / "strainphase_benchmark" / params.short_name()
                        config_output_dir = temp_base
                    
                    config_output_dir.mkdir(parents=True, exist_ok=True)
                    lineages_path = str(config_output_dir / "lineages.tsv")
                    
                    # Convert WindowResults to lineages.tsv
                    try:
                        if use_longitudinal:
                            # For longitudinal mode, use build_lineage_table to create proper lineages.tsv
                            from strainphase.longitudinal import build_lineage_table
                            
                            # Reconstruct the structure expected by build_lineage_table
                            # {mag_name -> {sample -> {contig -> [WindowResult]}}}
                            structured_results = {"MAG_01": mag_results}
                            
                            # Build lineage table (creates records with lineage_id, sample, contig, etc.)
                            lineage_records = build_lineage_table(structured_results, config)
                            
                            # Log which timepoints have records BEFORE conversion
                            samples_in_records = set(rec.get('sample', '') for rec in lineage_records)
                            logger.info(f"    Lineage records include timepoints: {sorted(samples_in_records)}")
                            logger.info(f"    Total lineage records: {len(lineage_records)}")
                            
                            # Log per-timepoint record counts BEFORE conversion
                            from collections import Counter
                            sample_counts_raw = Counter(rec.get('sample', '') for rec in lineage_records)
                            logger.info(f"    Raw records per timepoint: {dict(sample_counts_raw)}")
                            
                            # Verify all expected timepoints are present
                            missing_timepoints = set(timepoints) - samples_in_records
                            if missing_timepoints:
                                logger.warning(f"    WARNING: Missing timepoints in lineage records: {sorted(missing_timepoints)}")
                            
                            # Convert records to format expected by validation
                            # build_lineage_table returns: mean_weight, consensus (pipe-separated)
                            # Validation expects: abundance, snv_alleles (comma-separated)
                            converted_records = []
                            for rec in lineage_records:
                                # Convert mean_weight -> abundance
                                abundance = rec.get('mean_weight', 0.0)
                                
                                # Convert consensus format: "pos1:base1|pos2:base2" -> "pos1:base1,pos2:base2"
                                consensus = rec.get('consensus', '')
                                snv_alleles = consensus.replace('|', ',') if consensus else ''
                                
                                converted_records.append({
                                    'lineage_id': rec.get('lineage_id', ''),
                                    'sample': rec.get('sample', ''),
                                    'contig': rec.get('contig', ''),
                                    'track_id': rec.get('track_id', ''),
                                    'abundance': abundance,
                                    'snv_alleles': snv_alleles,
                                })
                            
                            # Log per-timepoint counts AFTER conversion
                            sample_counts = Counter(rec['sample'] for rec in converted_records)
                            logger.info(f"    Converted records per timepoint: {dict(sample_counts)}")
                            
                            # Log per-contig breakdown
                            contig_counts = Counter(rec['contig'] for rec in converted_records)
                            logger.info(f"    Records per contig: {dict(contig_counts)}")
                            
                            # Write lineages.tsv from converted records
                            import csv
                            fieldnames = ['lineage_id', 'sample', 'contig', 'track_id', 'abundance', 'snv_alleles']
                            with open(lineages_path, 'w', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
                                writer.writeheader()
                                if converted_records:
                                    writer.writerows(converted_records)
                        else:
                            # Single-timepoint mode: use simple conversion
                            window_results_to_lineages_tsv(
                                all_window_results, lineages_path, sample_id=first_timepoint
                            )
                        
                        # Check if file was created
                        if not Path(lineages_path).exists():
                            logger.warning(f"Failed to create lineages.tsv at {lineages_path}")
                            raise FileNotFoundError(f"lineages.tsv not created")
                        
                        # Run detailed validation (handles empty files gracefully)
                        from validation.validate_haplotypes import run_validation
                        validation_output = str(config_output_dir / "validation")
                        prelink_validation_output = str(config_output_dir / "validation_prelink")

                        if verbose:
                            mode_str = "longitudinal" if use_longitudinal else "single-timepoint"
                            logger.info(f"    Running validation ({mode_str} mode)...")

                        # Pre-link validation (per-window haplotypes, no linking)
                        prelink_lineages_path = str(config_output_dir / "prelink_lineages.tsv")
                        prelink_sample_id = timepoints[0] if use_longitudinal else first_timepoint
                        window_results_to_prelink_lineages_tsv(
                            all_window_results, prelink_lineages_path, sample_id=prelink_sample_id
                        )
                        prelink_result = run_validation(
                            detected_file=prelink_lineages_path,
                            truth_dir=self.truth_dir,
                            output_dir=prelink_validation_output,
                            window_results=all_window_results
                        )

                        validation_result = run_validation(
                            detected_file=lineages_path,
                            truth_dir=self.truth_dir,
                            output_dir=validation_output,
                            window_results=all_window_results,  # For track validation
                            window_size=params.window_size  # For track validation
                        )

                        # Write rescue statistics if available from longitudinal integrator
                        rescued_haplotypes = None
                        rescue_total_haplotypes = None
                        rescue_rate = None
                        rescue_integrator = getattr(config, "_rescue_integrator", None) if use_longitudinal else None
                        if rescue_integrator:
                            try:
                                rescue_stats_path = str(Path(validation_output) / "rescue_statistics.tsv")
                                rescue_integrator.write_rescue_statistics(rescue_stats_path)
                                rescued_haplotypes = sum(1 for s in rescue_integrator.rescue_statistics if s.was_rescued)
                                rescue_total_haplotypes = len(rescue_integrator.rescue_statistics)
                                rescue_rate = (
                                    rescued_haplotypes / rescue_total_haplotypes if rescue_total_haplotypes else 0.0
                                )
                            except Exception as e:
                                logger.warning(f"Failed to write rescue statistics: {e}")

                        # Write prelink vs postlink comparison
                        comparison_path = str(Path(validation_output) / "validation_prelink_comparison.tsv")
                        try:
                            write_validation_comparison_tsv(comparison_path, prelink_result, validation_result)
                        except Exception as e:
                            logger.warning(f"Failed to write prelink comparison TSV: {e}")

                        # Extract metrics from validation (including track/linking and lineage metrics)
                        accuracy_metrics = {
                            "snv_precision": validation_result.snv_precision,
                            "snv_recall": validation_result.snv_recall,
                            "snv_f1": 2 * validation_result.snv_precision * validation_result.snv_recall /
                                     (validation_result.snv_precision + validation_result.snv_recall)
                                     if (validation_result.snv_precision + validation_result.snv_recall) > 0 else 0.0,
                            "false_negatives_count": len(validation_result.false_negatives or []),
                            "false_positives_count": len(validation_result.false_positives or []),
                            "snv_true_total": validation_result.snv_true_total,
                            "snv_true_in_span": validation_result.snv_true_in_span,
                            "snv_detected_total": validation_result.snv_detected_total,
                            "snv_correct_total": validation_result.snv_correct_total,
                            "snv_span_coverage_frac": validation_result.snv_span_coverage_frac,
                            "haplotype_precision": validation_result.precision,
                            "haplotype_recall": validation_result.recall,
                            "haplotype_f1": validation_result.f1,
                            "abundance_pearson_r": validation_result.abundance_pearson_r,
                            "abundance_mae": validation_result.abundance_mae,
                            # Track/linking metrics
                            "track_fragmentation_mean": validation_result.track_fragmentation_mean,
                            "track_fragmentation_median": validation_result.track_fragmentation_median,
                            "false_link_rate": validation_result.false_link_rate,
                            "missed_link_rate": validation_result.missed_link_rate,
                            "track_consensus_error": validation_result.track_consensus_error,
                            # Lineage metrics
                            "lineage_precision": validation_result.lineage_precision,
                            "lineage_recall": validation_result.lineage_recall,
                            "lineage_f1": validation_result.lineage_f1,
                            "rescue_delta_recall_rare": validation_result.rescue_delta_recall_rare,
                            "abundance_trajectory_error": validation_result.abundance_trajectory_error,
                            "rescued_haplotypes": rescued_haplotypes,
                            "rescue_total_haplotypes": rescue_total_haplotypes,
                            "rescue_rate": rescue_rate,
                        }
                        if verbose:
                            logger.info(f"    Validation complete: F1={accuracy_metrics['haplotype_f1']:.3f}, "
                                      f"Precision={accuracy_metrics['haplotype_precision']:.3f}, "
                                      f"Recall={accuracy_metrics['haplotype_recall']:.3f}")
                            if use_longitudinal:
                                logger.info(f"      Lineage F1={accuracy_metrics.get('lineage_f1', 0):.3f}, "
                                          f"Rescue ΔRecall={accuracy_metrics.get('rescue_delta_recall_rare', 0):.3f}")
                            logger.info(f"    Validation outputs saved to: {validation_output}")
                    except Exception as e:
                        logger.warning(f"Validation failed for {params.short_name()}: {e}")
                        logger.exception("Validation exception details:")
                        # No fallback - validation must succeed for metrics
                        accuracy_metrics = {}

                # Build trajectories for single timepoint
                trajectories = {}
                for wr in all_window_results:
                    for hap in wr.haplotypes:
                        tid = hap.track_id or f"unlinked_{wr.window.start}"
                        if tid not in trajectories:
                            trajectories[tid] = {}
                        trajectories[tid]["T1"] = hap.weight

                # Get full config and environment info
                config_obj = params.to_config()
                try:
                    from dataclasses import asdict as dataclass_asdict
                    config_full = dataclass_asdict(config_obj)
                except:
                    # Fallback if not a dataclass
                    config_full = {k: getattr(config_obj, k, None) for k in dir(config_obj) if not k.startswith('_')}
                
                import platform
                environment = {
                    'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    'platform': platform.system().lower(),
                    'cpu': platform.processor() or platform.machine(),
                    'threads': os.cpu_count() or 1,
                }
                
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
                    false_negatives_count=accuracy_metrics.get("false_negatives_count"),
                    false_positives_count=accuracy_metrics.get("false_positives_count"),
                    snv_true_total=accuracy_metrics.get("snv_true_total"),
                    snv_true_in_span=accuracy_metrics.get("snv_true_in_span"),
                    snv_detected_total=accuracy_metrics.get("snv_detected_total"),
                    snv_correct_total=accuracy_metrics.get("snv_correct_total"),
                    snv_span_coverage_frac=accuracy_metrics.get("snv_span_coverage_frac"),
                    haplotype_precision=accuracy_metrics.get("haplotype_precision"),
                    haplotype_recall=accuracy_metrics.get("haplotype_recall"),
                    haplotype_f1=accuracy_metrics.get("haplotype_f1"),
                    abundance_pearson_r=accuracy_metrics.get("abundance_pearson_r"),
                    abundance_mae=accuracy_metrics.get("abundance_mae"),
                    # Track/linking metrics
                    track_fragmentation_mean=accuracy_metrics.get("track_fragmentation_mean"),
                    track_fragmentation_median=accuracy_metrics.get("track_fragmentation_median"),
                    false_link_rate=accuracy_metrics.get("false_link_rate"),
                    missed_link_rate=accuracy_metrics.get("missed_link_rate"),
                    track_consensus_error=accuracy_metrics.get("track_consensus_error"),
                    # Lineage metrics
                    lineage_precision=accuracy_metrics.get("lineage_precision"),
                    lineage_recall=accuracy_metrics.get("lineage_recall"),
                    lineage_f1=accuracy_metrics.get("lineage_f1"),
                    rescue_delta_recall_rare=accuracy_metrics.get("rescue_delta_recall_rare"),
                    abundance_trajectory_error=accuracy_metrics.get("abundance_trajectory_error"),
                    rescued_haplotypes=accuracy_metrics.get("rescued_haplotypes"),
                    rescue_total_haplotypes=accuracy_metrics.get("rescue_total_haplotypes"),
                    rescue_rate=accuracy_metrics.get("rescue_rate"),
                    # Metadata
                    ablation=None,  # Can be set by caller for ablation studies
                    vcf_condition="perfect",
                    config_full=config_full,
                    environment=environment,
                    seed=self.seed,
                )

                # Log completion
                progress_logger.log_config_complete(run_idx, result)

                # Save result and checkpoint
                self.results.append(result)
                if checkpoint_mgr:
                    checkpoint_mgr.save_config_result(result)

                # Update progress
                progress.completed_configs = run_idx
                progress.completed_config_keys.append(params.short_name())
                if checkpoint_mgr:
                    checkpoint_mgr.save_progress(progress)

            except Exception as e:
                if verbose:
                    logger.error(f"    Error with {params.short_name()}: {e}")
                logger.exception(f"Error with params {params.short_name()}")
                # Still track this config as attempted so we skip on resume
                progress.completed_config_keys.append(params.short_name())
                if checkpoint_mgr:
                    checkpoint_mgr.save_progress(progress, force=True)

        # Final checkpoint save
        if checkpoint_mgr:
            checkpoint_mgr.save_progress(progress, force=True)

        return self.results

    def _score_result(self, res: SweepResult) -> float:
        """
        Score a result for optimization comparison.

        Higher is better. Optimize haplotype F1 only.
        """
        if res.haplotype_f1 is None:
            return float("-inf")
        return res.haplotype_f1

    def _run_single_config(
        self,
        params: ParameterSet,
        contigs: Dict[str, int],
        progress_logger: ProgressLogger,
        config_idx: int,
        n_workers: int = 1,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> SweepResult:
        """Run a single parameter configuration and return the result."""
        start_time = time.time()

        all_window_results: List[WindowResult] = []
        
        if hasattr(self, 'use_longitudinal') and self.use_longitudinal:
            # Longitudinal mode: use process_mag_longitudinal
            from strainphase.longitudinal import process_mag_longitudinal
            from strainphase.core import HaplotyperConfig
            
            config = params.to_config(n_workers=n_workers)

            # Process MAG with longitudinal integration
            # mag_contigs should be {contig_id: length}, not nested
            mag_results = process_mag_longitudinal(
                mag_name="MAG_01",
                mag_contigs=contigs,  # Pass contigs directly, not nested dict
                samples=self.timepoints,
                bam_paths=self.bam_paths,
                vcf_paths=self.vcf_paths,
                config=config,
            )
            
            # Log per-timepoint results from process_mag_longitudinal
            logger.info(f"    process_mag_longitudinal returned results for {len(mag_results)} timepoints")
            for sample_id in self.timepoints:
                sample_contigs = mag_results.get(sample_id, {})
                total_windows = sum(len(contig_results) for contig_results in sample_contigs.values())
                total_haplotypes = sum(
                    len(wr.haplotypes) 
                    for contig_results in sample_contigs.values() 
                    for wr in contig_results
                )
                logger.info(f"    {sample_id}: {len(sample_contigs)} contigs, {total_windows} windows, {total_haplotypes} haplotypes")
            
            # Check if any timepoints are missing
            missing_samples = set(self.timepoints) - set(mag_results.keys())
            if missing_samples:
                logger.warning(f"    WARNING: process_mag_longitudinal did not return results for: {sorted(missing_samples)}")
            
            # Flatten results: {sample -> {contig -> [WindowResult]}} -> List[WindowResult]
            for sample_results in mag_results.values():
                for contig_results in sample_results.values():
                    all_window_results.extend(contig_results)
            
            # Per-contig logging
            for contig_idx, (contig_id, contig_length) in enumerate(contigs.items(), 1):
                n_windows = sum(len(mag_results.get(sample, {}).get(contig_id, [])) 
                              for sample in self.timepoints)
                progress_logger.log_contig_progress(config_idx, contig_idx, contig_id, n_windows)
        else:
            # Single-timepoint mode: process each contig individually
            first_timepoint = self.timepoints[0] if hasattr(self, 'timepoints') and self.timepoints else list(self.bam_paths.keys())[0] if hasattr(self, 'bam_paths') else "sample"
            for contig_idx, (contig_id, contig_length) in enumerate(contigs.items(), 1):
                results = process_contig_with_params(
                    self.bam_path, self.vcf_path, contig_id, contig_length,
                    params, sample_id=first_timepoint, n_workers=n_workers
                )
                all_window_results.extend(results)
                progress_logger.log_contig_progress(config_idx, contig_idx, contig_id, len(results))

        runtime = time.time() - start_time

        # Extract metrics from results
        track_ids = {h.track_id for wr in all_window_results for h in wr.haplotypes if h.track_id}
        n_lineages = len(track_ids)

        # Compute accuracy if truth available
        accuracy_metrics = {}
        validation_result = None
        if self.truth_dir:
            # Create temporary lineages.tsv for validation
            # Use output_dir if provided, otherwise create temp directory
            if output_dir:
                config_output_dir = Path(output_dir) / "configs" / params.short_name()
            else:
                # Create temp directory if output_dir not provided
                import tempfile
                temp_base = Path(tempfile.gettempdir()) / "strainphase_benchmark" / params.short_name()
                config_output_dir = temp_base
            
            config_output_dir.mkdir(parents=True, exist_ok=True)
            lineages_path = str(config_output_dir / "lineages.tsv")
            
            # Convert WindowResults to lineages.tsv
            try:
                if hasattr(self, 'use_longitudinal') and self.use_longitudinal:
                    # For longitudinal mode, use build_lineage_table to create proper lineages.tsv
                    from strainphase.longitudinal import build_lineage_table
                    from strainphase.core import HaplotyperConfig
                    
                    config = params.to_config(n_workers=n_workers)
                    
                    # Reconstruct the structure expected by build_lineage_table
                    # {mag_name -> {sample -> {contig -> [WindowResult]}}}
                    # We need to reconstruct this from all_window_results
                    # Group by sample and contig
                    structured_results = {"MAG_01": {}}
                    for wr in all_window_results:
                        sample = wr.window.sample or self.timepoints[0]
                        contig = wr.window.contig
                        if sample not in structured_results["MAG_01"]:
                            structured_results["MAG_01"][sample] = {}
                        if contig not in structured_results["MAG_01"][sample]:
                            structured_results["MAG_01"][sample][contig] = []
                        structured_results["MAG_01"][sample][contig].append(wr)
                    
                    # Build lineage table (creates records with lineage_id, sample, contig, etc.)
                    lineage_records = build_lineage_table(structured_results, config)
                    
                    # Log which timepoints have records BEFORE conversion
                    samples_in_records = set(rec.get('sample', '') for rec in lineage_records)
                    logger.info(f"    Lineage records include timepoints: {sorted(samples_in_records)}")
                    logger.info(f"    Total lineage records: {len(lineage_records)}")
                    
                    # Log per-timepoint record counts BEFORE conversion
                    from collections import Counter
                    sample_counts_raw = Counter(rec.get('sample', '') for rec in lineage_records)
                    logger.info(f"    Raw records per timepoint: {dict(sample_counts_raw)}")
                    
                    # Verify all expected timepoints are present
                    missing_timepoints = set(self.timepoints) - samples_in_records
                    if missing_timepoints:
                        logger.warning(f"    WARNING: Missing timepoints in lineage records: {sorted(missing_timepoints)}")
                    
                    # Convert records to format expected by validation
                    # build_lineage_table returns: mean_weight, consensus (pipe-separated)
                    # Validation expects: abundance, snv_alleles (comma-separated)
                    converted_records = []
                    for rec in lineage_records:
                        # Convert mean_weight -> abundance
                        abundance = rec.get('mean_weight', 0.0)
                        
                        # Convert consensus format: "pos1:base1|pos2:base2" -> "pos1:base1,pos2:base2"
                        consensus = rec.get('consensus', '')
                        snv_alleles = consensus.replace('|', ',') if consensus else ''
                        
                        converted_records.append({
                            'lineage_id': rec.get('lineage_id', ''),
                            'sample': rec.get('sample', ''),
                            'contig': rec.get('contig', ''),
                            'track_id': rec.get('track_id', ''),
                            'abundance': abundance,
                            'snv_alleles': snv_alleles,
                        })
                    
                    # Log per-timepoint counts AFTER conversion
                    sample_counts = Counter(rec['sample'] for rec in converted_records)
                    logger.info(f"    Converted records per timepoint: {dict(sample_counts)}")
                    
                    # Log per-contig breakdown
                    contig_counts = Counter(rec['contig'] for rec in converted_records)
                    logger.info(f"    Records per contig: {dict(contig_counts)}")
                    
                    # Write lineages.tsv from converted records
                    import csv
                    fieldnames = ['lineage_id', 'sample', 'contig', 'track_id', 'abundance', 'snv_alleles']
                    with open(lineages_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
                        writer.writeheader()
                        if converted_records:
                            writer.writerows(converted_records)
                else:
                    # Single-timepoint mode: use simple conversion
                    first_timepoint = self.timepoints[0] if hasattr(self, 'timepoints') and self.timepoints else list(self.bam_paths.keys())[0] if hasattr(self, 'bam_paths') else "T1"
                    window_results_to_lineages_tsv(
                        all_window_results, lineages_path, sample_id=first_timepoint
                    )
                
                # Run detailed validation (with track/linking support)
                from validation.validate_haplotypes import run_validation
                validation_output = str(config_output_dir / "validation")
                prelink_validation_output = str(config_output_dir / "validation_prelink")

                if verbose:
                    mode_str = "longitudinal" if hasattr(self, 'use_longitudinal') and self.use_longitudinal else "single-timepoint"
                    logger.info(f"    Running validation ({mode_str} mode)...")

                # Pre-link validation (per-window haplotypes, no linking)
                prelink_lineages_path = str(config_output_dir / "prelink_lineages.tsv")
                if hasattr(self, 'use_longitudinal') and self.use_longitudinal and hasattr(self, 'timepoints') and self.timepoints:
                    prelink_sample_id = self.timepoints[0]
                else:
                    prelink_sample_id = first_timepoint
                window_results_to_prelink_lineages_tsv(
                    all_window_results, prelink_lineages_path, sample_id=prelink_sample_id
                )
                prelink_result = run_validation(
                    detected_file=prelink_lineages_path,
                    truth_dir=self.truth_dir,
                    output_dir=prelink_validation_output,
                    window_results=all_window_results
                )

                validation_result = run_validation(
                    detected_file=lineages_path,
                    truth_dir=self.truth_dir,
                    output_dir=validation_output,
                    window_results=all_window_results,  # For track validation
                    window_size=params.window_size  # For track validation
                )

                # Write rescue statistics if available from longitudinal integrator
                rescued_haplotypes = None
                rescue_total_haplotypes = None
                rescue_rate = None
                rescue_integrator = getattr(config, "_rescue_integrator", None)
                if not (hasattr(self, 'use_longitudinal') and self.use_longitudinal):
                    rescue_integrator = None
                if rescue_integrator:
                    try:
                        rescue_stats_path = str(Path(validation_output) / "rescue_statistics.tsv")
                        rescue_integrator.write_rescue_statistics(rescue_stats_path)
                        rescued_haplotypes = sum(1 for s in rescue_integrator.rescue_statistics if s.was_rescued)
                        rescue_total_haplotypes = len(rescue_integrator.rescue_statistics)
                        rescue_rate = (
                            rescued_haplotypes / rescue_total_haplotypes if rescue_total_haplotypes else 0.0
                        )
                    except Exception as e:
                        logger.warning(f"Failed to write rescue statistics: {e}")

                # Write prelink vs postlink comparison
                comparison_path = str(Path(validation_output) / "validation_prelink_comparison.tsv")
                try:
                    write_validation_comparison_tsv(comparison_path, prelink_result, validation_result)
                except Exception as e:
                    logger.warning(f"Failed to write prelink comparison TSV: {e}")

                # Extract metrics from validation (including track/linking and lineage metrics)
                accuracy_metrics = {
                    "snv_precision": validation_result.snv_precision,
                    "snv_recall": validation_result.snv_recall,
                    "snv_f1": 2 * validation_result.snv_precision * validation_result.snv_recall /
                             (validation_result.snv_precision + validation_result.snv_recall)
                             if (validation_result.snv_precision + validation_result.snv_recall) > 0 else 0.0,
                    "false_negatives_count": len(validation_result.false_negatives or []),
                    "false_positives_count": len(validation_result.false_positives or []),
                    "snv_true_total": validation_result.snv_true_total,
                    "snv_true_in_span": validation_result.snv_true_in_span,
                    "snv_detected_total": validation_result.snv_detected_total,
                    "snv_correct_total": validation_result.snv_correct_total,
                    "snv_span_coverage_frac": validation_result.snv_span_coverage_frac,
                    "haplotype_precision": validation_result.precision,
                    "haplotype_recall": validation_result.recall,
                    "haplotype_f1": validation_result.f1,
                    "abundance_pearson_r": validation_result.abundance_pearson_r,
                    "abundance_mae": validation_result.abundance_mae,
                    # Track/linking metrics
                    "track_fragmentation_mean": validation_result.track_fragmentation_mean,
                    "track_fragmentation_median": validation_result.track_fragmentation_median,
                    "false_link_rate": validation_result.false_link_rate,
                    "missed_link_rate": validation_result.missed_link_rate,
                    "track_consensus_error": validation_result.track_consensus_error,
                    # Lineage metrics
                    "lineage_precision": validation_result.lineage_precision,
                    "lineage_recall": validation_result.lineage_recall,
                    "lineage_f1": validation_result.lineage_f1,
                    "rescue_delta_recall_rare": validation_result.rescue_delta_recall_rare,
                    "abundance_trajectory_error": validation_result.abundance_trajectory_error,
                    "rescued_haplotypes": rescued_haplotypes,
                    "rescue_total_haplotypes": rescue_total_haplotypes,
                    "rescue_rate": rescue_rate,
                }
                if verbose:
                    logger.info(f"    Validation complete: F1={accuracy_metrics['haplotype_f1']:.3f}, "
                              f"Precision={accuracy_metrics['haplotype_precision']:.3f}, "
                              f"Recall={accuracy_metrics['haplotype_recall']:.3f}")
                    if hasattr(self, 'use_longitudinal') and self.use_longitudinal:
                        logger.info(f"      Lineage F1={accuracy_metrics.get('lineage_f1', 0):.3f}, "
                                  f"Rescue ΔRecall={accuracy_metrics.get('rescue_delta_recall_rare', 0):.3f}")
                    logger.info(f"    Validation outputs saved to: {validation_output}")
            except Exception as e:
                logger.warning(f"Validation failed for {params.short_name()}: {e}")
                if verbose:
                    logger.exception("Validation exception details:")
                # No fallback - validation must succeed for metrics
                accuracy_metrics = {}

        # Build trajectories for single timepoint
        trajectories = {}
        for wr in all_window_results:
            for hap in wr.haplotypes:
                tid = hap.track_id or f"unlinked_{wr.window.start}"
                if tid not in trajectories:
                    trajectories[tid] = {}
                trajectories[tid]["T1"] = hap.weight

        # Get full config and environment info
        config_obj = params.to_config()
        try:
            from dataclasses import asdict as dataclass_asdict
            config_full = dataclass_asdict(config_obj)
        except:
            # Fallback if not a dataclass
            config_full = {k: getattr(config_obj, k, None) for k in dir(config_obj) if not k.startswith('_')}
        
        import platform
        environment = {
            'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': platform.system().lower(),
            'cpu': platform.processor() or platform.machine(),
            'threads': os.cpu_count() or 1,
        }
        
        return SweepResult(
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
            false_negatives_count=accuracy_metrics.get("false_negatives_count"),
            false_positives_count=accuracy_metrics.get("false_positives_count"),
            snv_true_total=accuracy_metrics.get("snv_true_total"),
            snv_true_in_span=accuracy_metrics.get("snv_true_in_span"),
            snv_detected_total=accuracy_metrics.get("snv_detected_total"),
            snv_correct_total=accuracy_metrics.get("snv_correct_total"),
            snv_span_coverage_frac=accuracy_metrics.get("snv_span_coverage_frac"),
            haplotype_precision=accuracy_metrics.get("haplotype_precision"),
            haplotype_recall=accuracy_metrics.get("haplotype_recall"),
            haplotype_f1=accuracy_metrics.get("haplotype_f1"),
            abundance_pearson_r=accuracy_metrics.get("abundance_pearson_r"),
            abundance_mae=accuracy_metrics.get("abundance_mae"),
            # Track/linking metrics
            track_fragmentation_mean=accuracy_metrics.get("track_fragmentation_mean"),
            track_fragmentation_median=accuracy_metrics.get("track_fragmentation_median"),
            false_link_rate=accuracy_metrics.get("false_link_rate"),
            missed_link_rate=accuracy_metrics.get("missed_link_rate"),
            track_consensus_error=accuracy_metrics.get("track_consensus_error"),
            # Lineage metrics
            lineage_precision=accuracy_metrics.get("lineage_precision"),
            lineage_recall=accuracy_metrics.get("lineage_recall"),
            lineage_f1=accuracy_metrics.get("lineage_f1"),
            rescue_delta_recall_rare=accuracy_metrics.get("rescue_delta_recall_rare"),
            abundance_trajectory_error=accuracy_metrics.get("abundance_trajectory_error"),
            rescued_haplotypes=accuracy_metrics.get("rescued_haplotypes"),
            rescue_total_haplotypes=accuracy_metrics.get("rescue_total_haplotypes"),
            rescue_rate=accuracy_metrics.get("rescue_rate"),
            # Metadata
            ablation=None,  # Can be set by caller for ablation studies
            vcf_condition="perfect",
            config_full=config_full,
            environment=environment,
            seed=self.seed,
        )

    def run_sequential_sweep(
        self,
        bam_paths: Dict[str, str],  # {timepoint -> bam_path}
        vcf_paths: Dict[str, str],  # {timepoint -> vcf_path}
        reference_path: Optional[str] = None,  # Required for longitudinal
        timepoints: Optional[List[str]] = None,  # List of timepoint IDs
        output_dir: str = "benchmarks/results",
        truth_dir: Optional[str] = None,
        max_contigs: Optional[int] = None,
        verbose: bool = True,
        resume: bool = False,
        checkpoint_interval: int = 5,
        max_passes: int = 1,
        param_order: Optional[List[str]] = None,
        start_values: Optional[Dict[str, Any]] = None,
        n_workers: int = 1,
    ) -> tuple:
        """
        Run sequential/coordinate descent parameter optimization.

        Supports both single-timepoint and longitudinal (multi-timepoint) modes.
        
        Instead of testing all 13,824 combinations, tests parameters one at a time:
        1. Start with intermediate default values for all parameters
        2. Test all values for parameter 1, pick the best
        3. Fix that parameter, test all values for parameter 2, pick best
        4. Repeat until all parameters are optimized

        Args:
            bam_path: Path to BAM file
            vcf_path: Path to VCF file
            output_dir: Output directory for results and checkpoints
            truth_dir: Optional path to ground truth directory
            max_contigs: Limit number of contigs to process
            verbose: Print progress
            resume: If True, resume from last checkpoint
            checkpoint_interval: How often to save progress
            max_passes: Number of passes through all parameters (default: 1)
            param_order: Order to optimize parameters (default: most impactful first)
            start_values: Initial parameter values (default: intermediate values)
            n_workers: Number of parallel workers for window processing

        Returns:
            (all_results, best_params_dict)
        """
        if not HAS_PYSAM:
            raise ImportError("pysam required for BAM/VCF processing")

        if param_order is None:
            param_order = list(self.DEFAULT_PARAM_ORDER)
        if start_values is None:
            start_values = dict(self.DEFAULT_START_VALUES)

        # Determine if longitudinal mode (multiple timepoints)
        use_longitudinal = (reference_path is not None and
                           timepoints is not None and
                           len(timepoints) > 1)

        # Store paths as instance variables for use in processing
        self.bam_paths = bam_paths
        self.vcf_paths = vcf_paths
        self.reference_path = reference_path
        self.timepoints = timepoints
        self.use_longitudinal = use_longitudinal

        # Store paths for backward compatibility and validation
        first_timepoint = timepoints[0] if timepoints else list(bam_paths.keys())[0]
        self.bam_path = bam_paths[first_timepoint]  # For backward compatibility
        self.vcf_path = vcf_paths[first_timepoint]   # For backward compatibility
        self.truth_dir = truth_dir

        # Setup checkpointing
        checkpoint_mgr = CheckpointManager(output_dir, checkpoint_interval)
        checkpoint_mgr.setup()

        # Store truth_dir for validation (validation loads its own ground truth)
        self.truth_dir = truth_dir
        if truth_dir:
            logger.info(f"Ground truth directory: {truth_dir}")

        # Get contigs from first BAM (or reference for longitudinal)
        if use_longitudinal and reference_path:
            from strainphase.longitudinal import parse_reference_contigs
            mags = parse_reference_contigs(reference_path, allowed_contigs=None)
            # Flatten MAG structure to get all contigs
            contigs = {}
            for mag_contigs in mags.values():
                contigs.update(mag_contigs)
        else:
            contigs = get_contigs_from_bam(bam_paths[first_timepoint])
        if max_contigs:
            contig_list = list(contigs.items())[:max_contigs]
            contigs = dict(contig_list)

        logger.info(f"Processing {len(contigs)} contigs")

        # Calculate total configs: sum of all parameter value counts * passes
        total_configs = sum(len(self.grid[p]) for p in param_order) * max_passes

        # Initialize or resume state
        best_values = dict(start_values)
        best_score = float('-inf')
        current_pass = 1
        current_param_idx = 0
        optimization_history: List[Dict[str, Any]] = []

        if resume:
            existing_progress = checkpoint_mgr.load_progress()
            if existing_progress and existing_progress.sequential_state:
                seq_state = existing_progress.sequential_state
                best_values = seq_state.get('best_values', best_values)
                best_score = seq_state.get('best_score', best_score)
                current_pass = seq_state.get('current_pass', 1)
                current_param_idx = seq_state.get('current_param_idx', 0)
                optimization_history = seq_state.get('optimization_history', [])
                self.results = checkpoint_mgr.load_completed_results()
                logger.info(f"Resuming sequential optimization from pass {current_pass}, "
                           f"parameter {current_param_idx} ({param_order[current_param_idx] if current_param_idx < len(param_order) else 'done'})")
            else:
                self.results = []
        else:
            self.results = []

        # Setup progress tracking
        progress_logger = ProgressLogger(total_configs, len(contigs), verbose)
        config_count = len(self.results)

        if verbose:
            logger.info(f"Sequential optimization: {len(param_order)} parameters, {max_passes} pass(es)")
            logger.info(f"Total configs to test: {total_configs}")

        # Main optimization loop
        for pass_num in range(current_pass, max_passes + 1):
            if verbose:
                logger.info(f"=== Pass {pass_num}/{max_passes} ===")

            start_idx = current_param_idx if pass_num == current_pass else 0

            for param_idx in range(start_idx, len(param_order)):
                param_name = param_order[param_idx]
                param_values = self.grid[param_name]

                if verbose:
                    logger.info(f"Optimizing {param_name}: testing {len(param_values)} values")

                best_value_for_param = best_values[param_name]
                best_score_for_param = float('-inf')

                for value in param_values:
                    # Build parameter set with current best values + this test value
                    test_params_dict = dict(best_values)
                    test_params_dict[param_name] = value
                    params = ParameterSet(**test_params_dict)

                    # Check if already completed (for resume)
                    cached_result = checkpoint_mgr.get_cached_result(params)
                    if cached_result:
                        result = cached_result
                        score = self._score_result(result)
                        if verbose:
                            logger.info(f"    {param_name}={value}: score={score:.4f} (cached)")
                    else:
                        config_count += 1
                        progress_logger.log_config_start(config_count, params)

                        try:
                            result = self._run_single_config(
                                params, contigs, progress_logger, config_count, n_workers,
                                output_dir=output_dir, verbose=verbose
                            )
                            progress_logger.log_config_complete(config_count, result)

                            self.results.append(result)
                            checkpoint_mgr.save_config_result(result)

                            score = self._score_result(result)
                            if verbose:
                                logger.info(f"    {param_name}={value}: score={score:.4f}")

                        except Exception as e:
                            logger.exception(f"Error with params {params.short_name()}")
                            score = float('-inf')

                    # Track best for this parameter
                    if score > best_score_for_param:
                        best_score_for_param = score
                        best_value_for_param = value

                    optimization_history.append({
                        'pass': pass_num,
                        'param': param_name,
                        'value': value,
                        'score': score,
                    })

                # Update best value for this parameter
                best_values[param_name] = best_value_for_param
                if best_score_for_param > best_score:
                    best_score = best_score_for_param

                if verbose:
                    logger.info(f"  Best {param_name} = {best_value_for_param} (score: {best_score_for_param:.4f})")

                # Save progress
                progress = SweepProgress(
                    total_configs=total_configs,
                    completed_configs=config_count,
                    current_config_idx=config_count,
                    start_time=time.time(),
                    last_save_time=time.time(),
                    completed_config_keys=[r.params.short_name() for r in self.results],
                    mode="sequential",
                    sequential_state={
                        'best_values': best_values,
                        'best_score': best_score,
                        'current_pass': pass_num,
                        'current_param_idx': param_idx + 1,
                        'optimization_history': optimization_history,
                    }
                )
                checkpoint_mgr.save_progress(progress, force=True)

            # Reset param_idx for next pass
            current_param_idx = 0

        # Save best params
        best_params_file = Path(output_dir) / "best_params.json"
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_values': best_values,
                'best_score': best_score,
                'optimization_history': optimization_history,
                'passes': max_passes,
            }, f, indent=2)

        if verbose:
            logger.info(f"\n=== Sequential Optimization Complete ===")
            logger.info(f"Best parameters found:")
            for param, value in best_values.items():
                logger.info(f"  {param}: {value}")
            logger.info(f"Best score: {best_score:.4f}")

        return self.results, best_values

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


def write_parameter_grid_summary(
    results: List[SweepResult],
    stable_params: List[ParameterSet],
    output_dir: str
) -> str:
    """
    Write parameter_grid_summary.tsv - comprehensive parameter comparison.

    Long-format table with one row per parameter configuration, summarizing
    all accuracy, linking, longitudinal, speed, stability, and parameter metrics.
    """
    import csv

    output_path = os.path.join(output_dir, 'parameter_grid_summary.tsv')

    # Build set of stable config names for is_stable lookup
    stable_config_names = {p.short_name() for p in stable_params}

    records = []
    for result in results:
        params = result.params

        # Compute SNV F1 from precision/recall
        snv_f1 = 0.0
        if result.snv_precision is not None and result.snv_recall is not None:
            if (result.snv_precision + result.snv_recall) > 0:
                snv_f1 = 2 * result.snv_precision * result.snv_recall / (result.snv_precision + result.snv_recall)

        # Determine if this config is stable
        is_stable = params.short_name() in stable_config_names

        records.append({
            # Config identification
            'config_name': params.short_name(),

            # Accuracy metrics
            'haplotype_precision': f"{result.haplotype_precision:.6f}" if result.haplotype_precision is not None else "NA",
            'haplotype_recall': f"{result.haplotype_recall:.6f}" if result.haplotype_recall is not None else "NA",
            'haplotype_f1': f"{result.haplotype_f1:.6f}" if result.haplotype_f1 is not None else "NA",
            'snv_precision': f"{result.snv_precision:.6f}" if result.snv_precision is not None else "NA",
            'snv_recall': f"{result.snv_recall:.6f}" if result.snv_recall is not None else "NA",
            'snv_f1': f"{snv_f1:.6f}",
            'false_negatives_count': result.false_negatives_count if result.false_negatives_count is not None else "NA",
            'false_positives_count': result.false_positives_count if result.false_positives_count is not None else "NA",
            'snv_true_total': result.snv_true_total if result.snv_true_total is not None else "NA",
            'snv_true_in_span': result.snv_true_in_span if result.snv_true_in_span is not None else "NA",
            'snv_detected_total': result.snv_detected_total if result.snv_detected_total is not None else "NA",
            'snv_correct_total': result.snv_correct_total if result.snv_correct_total is not None else "NA",
            'snv_span_coverage_frac': f"{result.snv_span_coverage_frac:.6f}" if result.snv_span_coverage_frac is not None else "NA",
            'abundance_pearson_r': f"{result.abundance_pearson_r:.6f}" if result.abundance_pearson_r is not None else "NA",
            'abundance_mae': f"{result.abundance_mae:.6f}" if result.abundance_mae is not None else "NA",

            # Track/Linking metrics
            'track_fragmentation_mean': f"{result.track_fragmentation_mean:.6f}" if result.track_fragmentation_mean is not None else "NA",
            'track_fragmentation_median': f"{result.track_fragmentation_median:.6f}" if result.track_fragmentation_median is not None else "NA",
            'false_link_rate': f"{result.false_link_rate:.6f}" if result.false_link_rate is not None else "NA",
            'missed_link_rate': f"{result.missed_link_rate:.6f}" if result.missed_link_rate is not None else "NA",
            'track_consensus_error': f"{result.track_consensus_error:.6f}" if result.track_consensus_error is not None else "NA",

            # Longitudinal metrics
            'lineage_precision': f"{result.lineage_precision:.6f}" if result.lineage_precision is not None else "NA",
            'lineage_recall': f"{result.lineage_recall:.6f}" if result.lineage_recall is not None else "NA",
            'lineage_f1': f"{result.lineage_f1:.6f}" if result.lineage_f1 is not None else "NA",
            'rescue_delta_recall_rare': f"{result.rescue_delta_recall_rare:.6f}" if result.rescue_delta_recall_rare is not None else "NA",
            'abundance_trajectory_error': f"{result.abundance_trajectory_error:.6f}" if result.abundance_trajectory_error is not None else "NA",
            'rescued_haplotypes': result.rescued_haplotypes if result.rescued_haplotypes is not None else "NA",
            'rescue_total_haplotypes': result.rescue_total_haplotypes if result.rescue_total_haplotypes is not None else "NA",
            'rescue_rate': f"{result.rescue_rate:.6f}" if result.rescue_rate is not None else "NA",

            # Speed metrics
            'runtime_seconds': f"{result.runtime_seconds:.2f}",
            'memory_peak_mb': f"{result.memory_peak_mb:.2f}" if result.memory_peak_mb is not None else "NA",

            # Stability metrics
            'converged': result.converged,
            'mean_confidence': f"{result.mean_confidence:.6f}",
            'n_lineages': result.n_lineages,
            'is_stable': is_stable,

            # Parameters
            'window_size': params.window_size,
            'max_mismatch_frac': params.max_mismatch_frac,
            'min_shared_snvs_for_edge': params.min_shared_snvs_for_edge,
            'merge_distance_threshold': params.merge_distance_threshold,
            'min_mapq': params.min_mapq,
            'min_base_quality': params.min_base_quality,
            'min_weight_for_anchor': params.min_weight_for_anchor,
            'rescued_min_weight': params.rescued_min_weight,
        })

    # Write TSV
    if records:
        fieldnames = [
            # Config
            'config_name',
            # Accuracy
            'haplotype_precision', 'haplotype_recall', 'haplotype_f1',
            'snv_precision', 'snv_recall', 'snv_f1',
            'false_negatives_count', 'false_positives_count',
            'snv_true_total', 'snv_true_in_span', 'snv_detected_total',
            'snv_correct_total', 'snv_span_coverage_frac',
            'abundance_pearson_r', 'abundance_mae',
            # Track/Linking
            'track_fragmentation_mean', 'track_fragmentation_median',
            'false_link_rate', 'missed_link_rate', 'track_consensus_error',
            # Longitudinal
            'lineage_precision', 'lineage_recall', 'lineage_f1',
            'rescue_delta_recall_rare', 'abundance_trajectory_error',
            'rescued_haplotypes', 'rescue_total_haplotypes', 'rescue_rate',
            # Speed
            'runtime_seconds', 'memory_peak_mb',
            # Stability
            'converged', 'mean_confidence', 'n_lineages', 'is_stable',
            # Parameters
            'window_size', 'max_mismatch_frac', 'min_shared_snvs_for_edge',
            'merge_distance_threshold', 'min_mapq', 'min_base_quality',
            'min_weight_for_anchor', 'rescued_min_weight',
        ]
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(records)
    else:
        # Write empty file with headers
        with open(output_path, 'w') as f:
            f.write('\t'.join([
                'config_name',
                'haplotype_precision', 'haplotype_recall', 'haplotype_f1',
                'snv_precision', 'snv_recall', 'snv_f1',
                'false_negatives_count', 'false_positives_count',
                'snv_true_total', 'snv_true_in_span', 'snv_detected_total',
                'snv_correct_total', 'snv_span_coverage_frac',
                'abundance_pearson_r', 'abundance_mae',
                'track_fragmentation_mean', 'track_fragmentation_median',
                'false_link_rate', 'missed_link_rate', 'track_consensus_error',
                'lineage_precision', 'lineage_recall', 'lineage_f1',
                'rescue_delta_recall_rare', 'abundance_trajectory_error',
                'rescued_haplotypes', 'rescue_total_haplotypes', 'rescue_rate',
                'runtime_seconds', 'memory_peak_mb',
                'converged', 'mean_confidence', 'n_lineages', 'is_stable',
                'window_size', 'max_mismatch_frac', 'min_shared_snvs_for_edge',
                'merge_distance_threshold', 'min_mapq', 'min_base_quality',
                'min_weight_for_anchor', 'rescued_min_weight',
            ]) + '\n')

    logger.info(f"Wrote {len(records)} parameter grid summary records to {output_path}")
    return output_path


def run_parameter_sweep(
    bam_path: Optional[str] = None,
    vcf_path: Optional[str] = None,
    bam_paths: Optional[Dict[str, str]] = None,  # For longitudinal: {timepoint -> bam_path}
    vcf_paths: Optional[Dict[str, str]] = None,  # For longitudinal: {timepoint -> vcf_path}
    reference_path: Optional[str] = None,  # Required for longitudinal mode
    timepoints: Optional[List[str]] = None,  # List of timepoint IDs
    output_dir: str = "benchmarks/results",
    truth_dir: Optional[str] = None,
    params_file: Optional[str] = None,
    coverage: Optional[int] = None,
    max_configs: Optional[int] = None,
    max_contigs: Optional[int] = None,
    verbose: bool = True,
    mode: str = "grid",
    resume: bool = False,
    checkpoint_interval: int = 10,
    passes: int = 3,
    n_workers: int = 1,
) -> Dict[str, Any]:
    """
    Run complete parameter sweep and save results.

    Args:
        bam_path: Single BAM file path (for single-timepoint mode)
        vcf_path: Single VCF file path (for single-timepoint mode)
        bam_paths: Dict of {timepoint -> bam_path} (for longitudinal mode)
        vcf_paths: Dict of {timepoint -> vcf_path} (for longitudinal mode)
        reference_path: Reference FASTA path (required for longitudinal mode)
        timepoints: List of timepoint IDs (e.g., ["T1", "T2"])
        output_dir: Output directory for results
        truth_dir: Ground truth directory (optional, for accuracy metrics)
        params_file: Custom parameter grid JSON file (optional)
        coverage: Coverage metadata to attach to each result (optional)
        max_configs: Limit number of configs to test (grid mode only)
        max_contigs: Limit number of contigs
        verbose: Print progress
        mode: Optimization mode - "grid" for full sweep, "sequential" for coordinate descent
        resume: If True, resume from last checkpoint
        checkpoint_interval: How often to save progress (in configs)
        passes: Number of optimization passes (sequential mode only)
        n_workers: Number of parallel workers for window processing
    
    Note: Either provide (bam_path, vcf_path) for single-timepoint mode,
          or (bam_paths, vcf_paths, reference_path, timepoints) for longitudinal mode.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine mode: longitudinal (multiple timepoints) or single-timepoint
    use_longitudinal = (bam_paths is not None and vcf_paths is not None and 
                       reference_path is not None and timepoints is not None)
    
    if not use_longitudinal:
        # Single-timepoint mode: require bam_path and vcf_path
        if not bam_path or not vcf_path:
            raise ValueError("Either provide (bam_path, vcf_path) for single-timepoint mode, "
                           "or (bam_paths, vcf_paths, reference_path, timepoints) for longitudinal mode")
        # Convert to dict format for consistency
        bam_paths = {"T1": bam_path}
        vcf_paths = {"T1": vcf_path}
        timepoints = ["T1"]
        reference_path = None
        logger.info(f"Single-timepoint mode: using {bam_path}")
    else:
        logger.info(f"Longitudinal mode: {len(timepoints)} timepoints ({', '.join(timepoints)})")

    # Load custom parameter grid if provided
    grid = None
    if params_file:
        with open(params_file) as f:
            grid = json.load(f)

    sweep = ParameterSweep(seed=42, grid=grid)

    if mode == "sequential":
        logger.info(f"Starting sequential parameter optimization")
        results, best_params = sweep.run_sequential_sweep(
            bam_paths=bam_paths,
            vcf_paths=vcf_paths,
            reference_path=reference_path,
            timepoints=timepoints,
            output_dir=output_dir,
            truth_dir=truth_dir,
            max_contigs=max_contigs,
            verbose=verbose,
            resume=resume,
            checkpoint_interval=checkpoint_interval,
            max_passes=passes,
            n_workers=n_workers,
        )
    else:
        logger.info(f"Starting parameter sweep")
        results = sweep.run_sweep(
            bam_paths=bam_paths,
            vcf_paths=vcf_paths,
            reference_path=reference_path,
            timepoints=timepoints,
            truth_dir=truth_dir,
            max_configs=max_configs,
            max_contigs=max_contigs,
            verbose=verbose,
            resume=resume,
            checkpoint_interval=checkpoint_interval,
            output_dir=output_dir,
            n_workers=n_workers,
        )

    # Save raw results
    results_data = [r.to_dict() for r in results]
    if coverage is not None:
        for r in results_data:
            r.setdefault("coverage", coverage)
    with open(os.path.join(output_dir, 'sweep_results.json'), 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    # Save metrics in documented schema (agents.md Section 4.3)
    metrics_payload = []
    for idx, r in enumerate(results):
        metrics_payload.append({
            "params": r.params.to_dict(),
            "config_full": r.config_full or {},
            "community": r.scenario_name,
            "seed": r.seed,
            "replicate": idx + 1,  # Sequential index as replicate number
            "environment": r.environment or {},
            "ablation": r.ablation,  # e.g., "full", "no_linking", "no_rescue", etc.
            "coverage": coverage,
            "metrics": {
                "haplotype_precision": r.haplotype_precision,
                "haplotype_recall": r.haplotype_recall,
                "haplotype_f1": r.haplotype_f1,
                "abundance_pearson_r": r.abundance_pearson_r,
                "abundance_mae": r.abundance_mae,
                "snv_precision": r.snv_precision,
                "snv_recall": r.snv_recall,
                "snv_f1": r.snv_f1,
                # Track/linking metrics
                "track_fragmentation": r.track_fragmentation_mean,
                "false_link_rate": r.false_link_rate,
                "missed_link_rate": r.missed_link_rate,
                "track_consensus_error": r.track_consensus_error,
                # Lineage metrics
                "lineage_precision": r.lineage_precision,
                "lineage_recall": r.lineage_recall,
                "rescue_delta_recall_rare": r.rescue_delta_recall_rare,
                "abundance_trajectory_error": r.abundance_trajectory_error,
                # Performance
                "runtime_seconds": r.runtime_seconds,
                "memory_peak_mb": r.memory_peak_mb,
            }
        })

    with open(os.path.join(output_dir, 'benchmark_metrics.json'), 'w') as f:
        json.dump(metrics_payload, f, indent=2, default=str)

    # Generate summary
    summary = sweep.summarize_results()
    summary['mode'] = mode
    with open(os.path.join(output_dir, 'sweep_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Identify stable parameters
    stable = sweep.identify_stable_parameters()
    stable_data = [p.to_dict() for p in stable]
    with open(os.path.join(output_dir, 'stable_parameters.json'), 'w') as f:
        json.dump(stable_data, f, indent=2)

    # Write parameter grid summary TSV
    try:
        write_parameter_grid_summary(results, stable, output_dir)
    except Exception as e:
        logger.warning(f"Failed to write parameter_grid_summary.tsv: {e}")

    if verbose:
        logger.info(f"\n=== SWEEP SUMMARY ===")
        logger.info(f"Mode: {mode}")
        logger.info(f"Total configs tested: {len(results)}")
        logger.info(f"Stable parameter sets found: {len(stable)}")

        for scenario_name, stats in summary.get('scenarios', {}).items():
            logger.info(f"\n{scenario_name}:")
            n_lin = stats.get('n_lineages', {})
            logger.info(f"  Lineages: {n_lin.get('mean', 0):.1f} +/- {n_lin.get('std', 0):.1f} "
                  f"(range {n_lin.get('min', 0)}-{n_lin.get('max', 0)})")
            logger.info(f"  Convergence: {stats.get('converged_fraction', 0):.1%}")
            if stats.get('snv_f1') is not None:
                logger.info(f"  SNV F1: {stats.get('snv_f1'):.3f}")

    logger.info(f"\nResults saved to: {output_dir}")
    return summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parameter sweep for strainphase haplotyper (file-based mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required inputs - support both single and multi-timepoint modes
    parser.add_argument("--bam", dest="bam_path",
                        help="Input BAM file (single-timepoint mode)")
    parser.add_argument("--vcf", dest="vcf_path",
                        help="Input VCF file (single-timepoint mode)")

    # Multi-timepoint mode
    parser.add_argument("--bam-paths", nargs="+",
                        help="Input BAM files, one per timepoint (multi-timepoint mode)")
    parser.add_argument("--vcf-paths", nargs="+",
                        help="Input VCF files, one per timepoint (multi-timepoint mode)")
    parser.add_argument("--reference",
                        help="Reference FASTA file (required for multi-timepoint mode)")
    parser.add_argument("--timepoints", nargs="+",
                        help="Timepoint IDs (e.g., T1 T2 T3 T4)")

    # Optional inputs
    parser.add_argument("--truth", dest="truth_dir",
                        help="Ground truth directory from simulation (optional)")
    parser.add_argument("--params", dest="params_file",
                        help="Custom parameter grid JSON file (optional)")
    parser.add_argument("--coverage", type=int,
                        help="Coverage metadata to attach to each result (optional)")

    # Output
    parser.add_argument("--output", "-o", default="benchmarks/results",
                        help="Output directory for results")

    # Mode selection
    parser.add_argument("--mode", choices=["grid", "sequential"], default="sequential",
                        help="Optimization mode: 'sequential' for coordinate descent (~27 configs), "
                             "'grid' for full sweep (many configs)")

    # Limits
    parser.add_argument("--max-configs", type=int,
                        help="Limit number of parameter configs to test (grid mode only)")
    parser.add_argument("--max-contigs", type=int,
                        help="Limit number of contigs to process")

    # Checkpointing
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from last checkpoint if available (default: True)")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="Start fresh, ignoring any existing checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Save checkpoint every N configs")

    # Sequential mode options
    parser.add_argument("--passes", type=int, default=3,
                        help="Number of optimization passes (sequential mode only)")

    # Parallelization
    parser.add_argument("-j", "--workers", type=int, default=8,
                        help="Number of parallel workers for window processing")

    # Verbosity
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Reduce output verbosity")

    args = parser.parse_args()

    # Determine mode: multi-timepoint or single-timepoint
    if args.bam_paths and args.vcf_paths:
        # Multi-timepoint mode
        if not args.timepoints:
            # Generate default timepoint names
            args.timepoints = [f"T{i+1}" for i in range(len(args.bam_paths))]

        if len(args.bam_paths) != len(args.vcf_paths):
            parser.error("Number of BAM paths must match number of VCF paths")
        if len(args.bam_paths) != len(args.timepoints):
            parser.error("Number of BAM paths must match number of timepoints")

        bam_paths = {tp: bam for tp, bam in zip(args.timepoints, args.bam_paths)}
        vcf_paths = {tp: vcf for tp, vcf in zip(args.timepoints, args.vcf_paths)}

        run_parameter_sweep(
            bam_paths=bam_paths,
            vcf_paths=vcf_paths,
            reference_path=args.reference,
            timepoints=args.timepoints,
            output_dir=args.output,
            truth_dir=args.truth_dir,
            params_file=args.params_file,
            coverage=args.coverage,
            max_configs=args.max_configs,
            max_contigs=args.max_contigs,
            verbose=not args.quiet,
            mode=args.mode,
            resume=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            passes=args.passes,
            n_workers=args.workers,
        )
    elif args.bam_path and args.vcf_path:
        # Single-timepoint mode (backward compatible)
        run_parameter_sweep(
            bam_path=args.bam_path,
            vcf_path=args.vcf_path,
            output_dir=args.output,
            truth_dir=args.truth_dir,
            params_file=args.params_file,
            coverage=args.coverage,
            max_configs=args.max_configs,
            max_contigs=args.max_contigs,
            verbose=not args.quiet,
            mode=args.mode,
            resume=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            passes=args.passes,
            n_workers=args.workers,
        )
    else:
        parser.error("Must provide either --bam/--vcf (single-timepoint) or --bam-paths/--vcf-paths (multi-timepoint)")


if __name__ == "__main__":
    main()
