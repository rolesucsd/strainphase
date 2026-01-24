#!/usr/bin/env python3
"""
Parameter sweep framework for haplotyper pipeline.

Tests pipeline stability across a grid of parameters:
- mismatch thresholds: 0.5%, 1%, 2%, 4%
- MAPQ: 10, 20, 30
- base quality: 20, 30
- shared SNVs: 2, 3, 4, 5
- merge distance: 0.5%, 1%, 2%
- anchor weight: 5%, 10%, 15%, 20%
- rescued min weight: 1%, 2%, 5%
- window size: 3000, 5000, 7000, 10000

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
    window_size: int

    def to_config(self, base_config: Optional[HaplotyperConfig] = None, n_workers: int = 1) -> HaplotyperConfig:
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
            window_size=self.window_size,

            # Related parameters (keep consistent)
            max_link_distance=self.max_mismatch_frac,
            min_shared_snvs_for_link=self.min_shared_snvs_for_edge,
            min_shared_for_merge=self.min_shared_snvs_for_edge,
            min_shared_for_rescue=self.min_shared_snvs_for_edge,
            rescue_match_distance=self.merge_distance_threshold,
            lineage_merge_distance=self.max_mismatch_frac,
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
            memory_peak_mb=data.get('memory_peak_mb'),
        )


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
                for allele_info in strains_info.split('|'):
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
        'max_mismatch_frac': [0.005, 0.01, 0.02, 0.04],
        'min_mapq': [10, 20, 30],
        'min_base_quality': [20, 30],
        'min_shared_snvs_for_edge': [2, 3, 4, 5],
        'merge_distance_threshold': [0.005, 0.01, 0.02],
        'min_weight_for_anchor': [0.05, 0.10, 0.15, 0.20],
        'rescued_min_weight': [0.01, 0.02, 0.05],
        'window_size': [3000, 5000, 7000, 10000],
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
    DEFAULT_START_VALUES = {
        'window_size': 5000,
        'max_mismatch_frac': 0.01,
        'min_shared_snvs_for_edge': 3,
        'merge_distance_threshold': 0.01,
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
        self.truth_snvs: Optional[Dict] = None
        self.truth_abundances: Optional[Dict] = None

    def _validate_grid(self, grid: Dict[str, List]) -> None:
        """Enforce that the parameter grid matches agents.md."""
        required_keys = set(self.REQUIRED_GRID.keys())
        provided_keys = set(grid.keys())
        if required_keys != provided_keys:
            missing = sorted(required_keys - provided_keys)
            extra = sorted(provided_keys - required_keys)
            raise ValueError(f"Parameter grid mismatch: missing={missing}, extra={extra}")

        for key, required_values in self.REQUIRED_GRID.items():
            provided_values = grid.get(key, [])
            if sorted(required_values) != sorted(provided_values):
                raise ValueError(
                    f"Parameter grid mismatch for {key}: "
                    f"expected {required_values}, got {provided_values}"
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
        bam_path: str,
        vcf_path: str,
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

        Args:
            bam_path: Path to BAM file
            vcf_path: Path to VCF file
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

        self.bam_path = bam_path
        self.vcf_path = vcf_path
        self.truth_dir = truth_dir

        # Setup checkpointing if output_dir provided
        checkpoint_mgr = None
        if output_dir:
            checkpoint_mgr = CheckpointManager(output_dir, checkpoint_interval)
            checkpoint_mgr.setup()

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

                for contig_idx, (contig_id, contig_length) in enumerate(contigs.items(), 1):
                    results = process_contig_with_params(
                        bam_path, vcf_path, contig_id, contig_length,
                        params, sample_id="sample", n_workers=n_workers
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

    def _score_result(self, res: SweepResult) -> float:
        """
        Score a result for optimization comparison.

        Higher is better. Weights prioritize accuracy metrics.
        """
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

    def _run_single_config(
        self,
        params: ParameterSet,
        contigs: Dict[str, int],
        progress_logger: ProgressLogger,
        config_idx: int,
        n_workers: int = 1,
    ) -> SweepResult:
        """Run a single parameter configuration and return the result."""
        start_time = time.time()

        all_window_results: List[WindowResult] = []
        for contig_idx, (contig_id, contig_length) in enumerate(contigs.items(), 1):
            results = process_contig_with_params(
                self.bam_path, self.vcf_path, contig_id, contig_length,
                params, sample_id="sample", n_workers=n_workers
            )
            all_window_results.extend(results)
            progress_logger.log_contig_progress(config_idx, contig_idx, contig_id, len(results))

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
        )

    def run_sequential_sweep(
        self,
        bam_path: str,
        vcf_path: str,
        output_dir: str,
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

        self.bam_path = bam_path
        self.vcf_path = vcf_path
        self.truth_dir = truth_dir

        # Setup checkpointing
        checkpoint_mgr = CheckpointManager(output_dir, checkpoint_interval)
        checkpoint_mgr.setup()

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
                            result = self._run_single_config(params, contigs, progress_logger, config_count, n_workers)
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


def run_parameter_sweep(
    bam_path: str,
    vcf_path: str,
    output_dir: str = "benchmarks/results",
    truth_dir: Optional[str] = None,
    params_file: Optional[str] = None,
    max_configs: Optional[int] = None,
    max_contigs: Optional[int] = None,
    verbose: bool = True,
    mode: str = "grid",
    resume: bool = False,
    checkpoint_interval: int = 10,
    passes: int = 1,
    n_workers: int = 1,
) -> Dict[str, Any]:
    """
    Run complete parameter sweep and save results.

    Args:
        bam_path: BAM file path
        vcf_path: VCF file path
        output_dir: Output directory for results
        truth_dir: Ground truth directory (optional, for accuracy metrics)
        params_file: Custom parameter grid JSON file (optional)
        max_configs: Limit number of configs to test (grid mode only)
        max_contigs: Limit number of contigs
        verbose: Print progress
        mode: Optimization mode - "grid" for full sweep, "sequential" for coordinate descent
        resume: If True, resume from last checkpoint
        checkpoint_interval: How often to save progress (in configs)
        passes: Number of optimization passes (sequential mode only)
        n_workers: Number of parallel workers for window processing
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load custom parameter grid if provided
    grid = None
    if params_file:
        with open(params_file) as f:
            grid = json.load(f)

    sweep = ParameterSweep(seed=42, grid=grid)

    if mode == "sequential":
        logger.info(f"Starting sequential parameter optimization on: {bam_path}")
        results, best_params = sweep.run_sequential_sweep(
            bam_path=bam_path,
            vcf_path=vcf_path,
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
        logger.info(f"Starting parameter sweep on: {bam_path}")
        results = sweep.run_sweep(
            bam_path=bam_path,
            vcf_path=vcf_path,
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
    summary['mode'] = mode
    with open(os.path.join(output_dir, 'sweep_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Identify stable parameters
    stable = sweep.identify_stable_parameters()
    stable_data = [p.to_dict() for p in stable]
    with open(os.path.join(output_dir, 'stable_parameters.json'), 'w') as f:
        json.dump(stable_data, f, indent=2)

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

    # Mode selection
    parser.add_argument("--mode", choices=["grid", "sequential"], default="grid",
                        help="Optimization mode: 'grid' for full sweep (13,824 configs), "
                             "'sequential' for coordinate descent (~27 configs)")

    # Limits
    parser.add_argument("--max-configs", type=int,
                        help="Limit number of parameter configs to test (grid mode only)")
    parser.add_argument("--max-contigs", type=int,
                        help="Limit number of contigs to process")

    # Checkpointing
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint if available")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Save checkpoint every N configs")

    # Sequential mode options
    parser.add_argument("--passes", type=int, default=1,
                        help="Number of optimization passes (sequential mode only)")

    # Parallelization
    parser.add_argument("-j", "--workers", type=int, default=1,
                        help="Number of parallel workers for window processing (default: 1)")

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
        verbose=not args.quiet,
        mode=args.mode,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        passes=args.passes,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
