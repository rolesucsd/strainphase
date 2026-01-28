#!/usr/bin/env python3
"""
Longitudinal lineage validation metrics for strainphase.

This is a library module used by validate_haplotypes.py. It is not meant to be
run standalone - use validate_haplotypes.py or the benchmarking pipeline instead.

Validates:
- Lineage precision/recall (cluster correctness for inferred lineage_id vs truth)
- Rescue gain (Δrecall) for low-abundance strains
- Abundance trajectory error per lineage/strain
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LineageValidationResult:
    """Results of lineage validation."""
    lineage_precision: float
    lineage_recall: float
    lineage_f1: float
    rescue_delta_recall_rare: float  # Recall improvement for rare strains (<1%)
    abundance_trajectory_error: float  # Mean error across all lineages
    per_lineage_errors: Dict[str, float]  # lineage_id -> trajectory_error


def load_truth_lineages(truth_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Load truth lineage mapping from truth_lineages.tsv.
    
    Returns: {strain_id -> {contig -> lineage_id}}
    """
    truth_path = Path(truth_dir)
    lineages_file = truth_path / "truth_lineages.tsv"
    
    if not lineages_file.exists():
        logger.warning(f"truth_lineages.tsv not found at {lineages_file}")
        logger.warning(f"  Looking in: {truth_path.absolute()}")
        logger.warning(f"  Files in truth_dir: {list(truth_path.glob('*.tsv')) if truth_path.exists() else 'directory does not exist'}")
        return {}
    
    truth_lineages = defaultdict(dict)
    with open(lineages_file) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            row = dict(zip(header, parts))
            strain_id = row['strain_id']
            lineage_id = row['lineage_id']
            contig = row['contig']
            truth_lineages[strain_id][contig] = lineage_id
    
    logger.info(f"Loaded {len(truth_lineages)} truth lineage mappings")
    return dict(truth_lineages)


def compute_lineage_clustering_metrics(
    detected_lineages: Dict[str, Dict[str, str]],  # detected_lineage_id -> {contig -> strain_id}
    truth_lineages: Dict[str, Dict[str, str]]  # strain_id -> {contig -> lineage_id}
) -> Tuple[float, float, float]:
    """
    Compute lineage precision/recall/F1.
    
    A detected lineage matches a true lineage if:
    - They share the same strain assignments across contigs
    
    Returns: (precision, recall, f1)
    """
    # Build reverse mapping: true_lineage_id -> set of (strain_id, contig) pairs
    true_lineage_clusters: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for strain_id, contig_lineages in truth_lineages.items():
        for contig, lineage_id in contig_lineages.items():
            true_lineage_clusters[lineage_id].add((strain_id, contig))
    
    logger.debug(f"Built {len(true_lineage_clusters)} true lineage clusters")
    if true_lineage_clusters:
        sample_true = list(true_lineage_clusters.items())[0]
        logger.debug(f"Sample true lineage cluster: {sample_true[0]} -> {sample_true[1]}")
    
    # Build detected lineage clusters
    detected_lineage_clusters: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for detected_lineage_id, contig_strains in detected_lineages.items():
        for contig, strain_id in contig_strains.items():
            detected_lineage_clusters[detected_lineage_id].add((strain_id, contig))
    
    logger.debug(f"Built {len(detected_lineage_clusters)} detected lineage clusters")
    if detected_lineage_clusters:
        sample_det = list(detected_lineage_clusters.items())[0]
        logger.debug(f"Sample detected lineage cluster: {sample_det[0]} -> {sample_det[1]}")
    
    # Match detected to true lineages
    matched_detected = set()
    matched_true = set()
    
    for det_lineage_id, det_cluster in detected_lineage_clusters.items():
        for true_lineage_id, true_cluster in true_lineage_clusters.items():
            if det_cluster == true_cluster:
                matched_detected.add(det_lineage_id)
                matched_true.add(true_lineage_id)
                logger.debug(f"Matched detected lineage {det_lineage_id} to true lineage {true_lineage_id}")
                break
    
    n_detected = len(detected_lineage_clusters)
    n_true = len(true_lineage_clusters)
    n_matched = len(matched_detected)
    
    logger.debug(f"Lineage matching: {n_matched} matches out of {n_detected} detected and {n_true} true")
    
    precision = n_matched / n_detected if n_detected > 0 else 0.0
    recall = n_matched / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def compute_rescue_gain(
    true_abundances: Dict[str, Dict[str, float]],  # strain_id -> {timepoint -> abundance}
    detected_with_rescue: Dict[str, Dict[str, float]],  # strain_id -> {timepoint -> abundance}
    detected_without_rescue: Dict[str, Dict[str, float]],  # strain_id -> {timepoint -> abundance}
    rare_threshold: float = 0.01  # 1% abundance threshold
) -> float:
    """
    Compute recall improvement (Δrecall) for rare strains with rescue enabled.
    
    Returns: Δrecall = recall_with_rescue - recall_without_rescue for rare strains
    """
    # Identify rare strains (max abundance < threshold)
    rare_strains = []
    for strain_id, abundances in true_abundances.items():
        max_abund = max(abundances.values()) if abundances else 0.0
        if 0 < max_abund < rare_threshold:
            rare_strains.append(strain_id)
    
    if not rare_strains:
        return 0.0
    
    # Compute recall for rare strains with and without rescue
    detected_with_rescue_count = sum(1 for s in rare_strains if s in detected_with_rescue)
    detected_without_rescue_count = sum(1 for s in rare_strains if s in detected_without_rescue)
    
    recall_with_rescue = detected_with_rescue_count / len(rare_strains) if rare_strains else 0.0
    recall_without_rescue = detected_without_rescue_count / len(rare_strains) if rare_strains else 0.0
    
    delta_recall = recall_with_rescue - recall_without_rescue
    return delta_recall


def compute_abundance_trajectory_error(
    true_abundances: Dict[str, Dict[str, float]],  # strain_id -> {timepoint -> abundance}
    detected_abundances: Dict[str, Dict[str, float]],  # strain_id -> {timepoint -> abundance}
    timepoints: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Compute error between inferred and true abundance trajectories per lineage/strain.
    
    Returns: (mean_error, per_lineage_errors)
    """
    per_lineage_errors = {}
    
    for strain_id, true_traj in true_abundances.items():
        det_traj = detected_abundances.get(strain_id, {})
        
        # Compute trajectory error (mean absolute error across timepoints)
        errors = []
        for tp in timepoints:
            true_abund = true_traj.get(tp, 0.0)
            det_abund = det_traj.get(tp, 0.0)
            errors.append(abs(true_abund - det_abund))
        
        if errors:
            per_lineage_errors[strain_id] = np.mean(errors)
    
    mean_error = np.mean(list(per_lineage_errors.values())) if per_lineage_errors else 0.0
    return mean_error, per_lineage_errors


def validate_lineages(
    detected_lineages: Dict[str, Dict[str, str]],  # detected_lineage_id -> {contig -> strain_id}
    truth_dir: str,
    true_abundances: Dict[str, Dict[str, float]],  # strain_id -> {timepoint -> abundance}
    detected_abundances: Dict[str, Dict[str, float]],  # strain_id -> {timepoint -> abundance}
    detected_without_rescue: Optional[Dict[str, Dict[str, float]]] = None,
    timepoints: Optional[List[str]] = None
) -> LineageValidationResult:
    """
    Compute all lineage validation metrics.
    
    Args:
        detected_lineages: Detected lineage assignments (lineage_id -> {contig -> strain_id})
        truth_dir: Directory containing truth_lineages.tsv
        true_abundances: True abundances per strain/timepoint
        detected_abundances: Detected abundances per strain/timepoint (with rescue)
        detected_without_rescue: Optional detected abundances without rescue (for Δrecall)
        timepoints: List of timepoint names (defaults to all in true_abundances)
    
    Returns:
        LineageValidationResult with all metrics
    """
    # Load truth lineages
    truth_lineages = load_truth_lineages(truth_dir)
    
    if not truth_lineages:
        logger.warning("No truth lineages found - returning zero metrics")
        return LineageValidationResult(
            lineage_precision=0.0,
            lineage_recall=0.0,
            lineage_f1=0.0,
            rescue_delta_recall_rare=0.0,
            abundance_trajectory_error=0.0,
            per_lineage_errors={}
        )
    
    # Debug logging
    logger.info(f"Truth lineages loaded: {len(truth_lineages)} strains")
    logger.info(f"Detected lineages provided: {len(detected_lineages)} lineages")
    
    # Check for contig name mismatch (common issue)
    truth_contigs = set()
    for strain_id, contig_dict in truth_lineages.items():
        truth_contigs.update(contig_dict.keys())
    
    detected_contigs = set()
    for lineage_id, contig_dict in detected_lineages.items():
        detected_contigs.update(contig_dict.keys())
    
    logger.info(f"Truth contigs: {sorted(truth_contigs)}")
    logger.info(f"Detected contigs: {sorted(detected_contigs)}")
    
    # Check overlap
    overlap = truth_contigs & detected_contigs
    if not overlap and truth_contigs and detected_contigs:
        logger.warning("="*60)
        logger.warning("CONTIG NAME MISMATCH DETECTED!")
        logger.warning(f"  Truth contigs: {sorted(truth_contigs)}")
        logger.warning(f"  Detected contigs: {sorted(detected_contigs)}")
        logger.warning("  No overlap - lineage metrics will be ZERO!")
        logger.warning("  Fix: Ensure truth files use the same contig names as the reference.")
        logger.warning("="*60)
    elif overlap and len(overlap) < len(truth_contigs):
        logger.warning(f"Partial contig overlap: {len(overlap)}/{len(truth_contigs)} truth contigs matched")
    
    if detected_lineages:
        sample_lineage = list(detected_lineages.items())[0]
        logger.info(f"Sample detected lineage: {sample_lineage[0]} -> {sample_lineage[1]}")
    if truth_lineages:
        sample_strain = list(truth_lineages.items())[0]
        logger.info(f"Sample truth strain: {sample_strain[0]} -> {sample_strain[1]}")
    
    # Compute lineage clustering metrics
    lineage_precision, lineage_recall, lineage_f1 = compute_lineage_clustering_metrics(
        detected_lineages, truth_lineages
    )
    
    logger.info(f"Lineage clustering metrics: precision={lineage_precision:.3f}, recall={lineage_recall:.3f}, f1={lineage_f1:.3f}")
    
    # Compute rescue gain (if without-rescue data provided)
    if detected_without_rescue is not None:
        rescue_delta = compute_rescue_gain(
            true_abundances, detected_abundances, detected_without_rescue
        )
    else:
        rescue_delta = 0.0
    
    # Compute trajectory error
    if timepoints is None:
        timepoints = sorted(set(tp for traj in true_abundances.values() for tp in traj.keys()))
    
    trajectory_error, per_lineage_errors = compute_abundance_trajectory_error(
        true_abundances, detected_abundances, timepoints
    )
    
    return LineageValidationResult(
        lineage_precision=lineage_precision,
        lineage_recall=lineage_recall,
        lineage_f1=lineage_f1,
        rescue_delta_recall_rare=rescue_delta,
        abundance_trajectory_error=trajectory_error,
        per_lineage_errors=per_lineage_errors
    )
