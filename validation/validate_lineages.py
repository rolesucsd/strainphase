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
    Compute lineage precision/recall/F1 based on whether haplotypes from the same
    true strain are grouped into the same detected lineage.
    
    This uses a relaxed matching approach that handles partial contig coverage:
    - A detected lineage is CORRECT (for precision) if all its strain assignments
      on each contig belong to the same true strain (consistent clustering)
    - A true strain is RECOVERED (for recall) if at least one detected lineage
      correctly identifies it on at least one contig with data
    
    Returns: (precision, recall, f1)
    """
    if not detected_lineages or not truth_lineages:
        logger.info("Empty lineages - returning zeros")
        return 0.0, 0.0, 0.0
    
    # Get the set of contigs that have detected data
    detected_contigs = set()
    for contig_strains in detected_lineages.values():
        detected_contigs.update(contig_strains.keys())
    
    logger.debug(f"Detected contigs with data: {sorted(detected_contigs)}")
    
    # For each detected lineage, check if all its strain assignments are consistent
    # (i.e., all contigs point to the same true strain)
    correct_detected = set()
    detected_to_strain: Dict[str, str] = {}  # Maps detected lineage_id to its assigned strain_id
    
    for det_lineage_id, contig_strains in detected_lineages.items():
        # Get unique strain assignments for this detected lineage
        assigned_strains = set(contig_strains.values())
        
        if len(assigned_strains) == 1:
            # Consistent - all contigs assigned to same strain
            strain_id = list(assigned_strains)[0]
            correct_detected.add(det_lineage_id)
            detected_to_strain[det_lineage_id] = strain_id
            logger.debug(f"Detected lineage {det_lineage_id} is consistent -> {strain_id}")
        else:
            # Inconsistent - mixed strain assignments (this is an error)
            logger.debug(f"Detected lineage {det_lineage_id} is INCONSISTENT: {assigned_strains}")
    
    # For recall: check which true strains are correctly recovered
    # A true strain is recovered if at least one detected lineage correctly identifies it
    recovered_strains = set()
    
    for strain_id in truth_lineages.keys():
        # Check if any detected lineage correctly maps to this strain
        for det_lineage_id, mapped_strain in detected_to_strain.items():
            if mapped_strain == strain_id:
                recovered_strains.add(strain_id)
                logger.debug(f"True strain {strain_id} recovered by lineage {det_lineage_id}")
                break
    
    n_detected = len(detected_lineages)
    n_correct = len(correct_detected)
    n_true = len(truth_lineages)
    n_recovered = len(recovered_strains)
    
    logger.info(f"Lineage clustering: {n_correct}/{n_detected} correct detected, {n_recovered}/{n_true} true strains recovered")
    
    # Precision: fraction of detected lineages that are internally consistent AND map to a real strain
    # We need stricter precision - count only those that map to actual truth strains
    precision_correct = sum(1 for d in correct_detected if detected_to_strain.get(d) in truth_lineages)
    precision = precision_correct / n_detected if n_detected > 0 else 0.0
    
    # Recall: fraction of true strains that were recovered
    recall = n_recovered / n_true if n_true > 0 else 0.0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    logger.info(f"Lineage clustering metrics: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
    
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
