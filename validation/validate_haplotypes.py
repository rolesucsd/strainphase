#!/usr/bin/env python3
"""
Validate strainphase haplotype reconstruction against ground truth.

Compares detected haplotypes/lineages to known ground truth from simulation
and computes accuracy metrics (precision, recall, F1, abundance correlation).

Usage:
    python validation/validate_haplotypes.py \
        --detected results/lineages.tsv \
        --truth data/simulated/ \
        --output results/validation/
"""

import argparse
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
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
# Data structures
# =============================================================================

@dataclass
class TrueHaplotype:
    """Ground truth haplotype from simulation."""
    strain_id: str
    snv_positions: Dict[str, Dict[int, str]]  # contig -> {pos -> allele}
    abundances: Dict[str, float]  # timepoint -> abundance
    is_sweeping: bool = False


@dataclass
class DetectedHaplotype:
    """Detected haplotype from strainphase output."""
    lineage_id: str
    track_id: str
    snv_alleles: Dict[str, Dict[int, str]]  # contig -> {pos -> allele}
    abundances: Dict[str, float]  # timepoint -> abundance


@dataclass
class ValidationResult:
    """Results of validation comparison."""
    # Haplotype detection
    n_true: int
    n_detected: int
    n_matched: int
    precision: float
    recall: float
    f1: float

    # Abundance accuracy
    abundance_pearson_r: float
    abundance_mae: float

    # SNV accuracy
    snv_precision: float
    snv_recall: float
    phasing_accuracy: float

    # Detection threshold
    detection_threshold: float

    # Per-match details
    matches: List[Tuple[str, str, float]]  # (true_id, detected_id, distance)


# =============================================================================
# Plot styling
# =============================================================================

def set_plot_style():
    """Set consistent figure style for validation plots."""
    if not HAS_MATPLOTLIB:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (9, 5.5)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


# =============================================================================
# Load ground truth
# =============================================================================

def load_ground_truth(truth_dir: str) -> Tuple[List[TrueHaplotype], Dict[str, List[int]]]:
    """
    Load ground truth from simulation output.

    Returns: (list of TrueHaplotype, dict of contig -> snv_positions)
    """
    truth_path = Path(truth_dir)

    # Load strain info
    strains_file = truth_path / "truth_strains.tsv"
    strain_info = {}
    with open(strains_file) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            row = dict(zip(header, parts))
            strain_info[row['strain_id']] = {
                'is_sweeping': row.get('is_sweeping', 'False') == 'True',
                'snv_count': int(row.get('snv_count', 0))
            }

    # Load abundances
    abundances_file = truth_path / "truth_abundances.tsv"
    strain_abundances = defaultdict(dict)
    with open(abundances_file) as f:
        header = f.readline().strip().split('\t')
        timepoints = header[1:]  # First column is strain_id
        for line in f:
            parts = line.strip().split('\t')
            strain_id = parts[0]
            for i, tp in enumerate(timepoints):
                strain_abundances[strain_id][tp] = float(parts[i + 1])

    # Load SNV positions per strain from VCF
    vcf_file = truth_path / "truth_snvs.vcf"
    if not vcf_file.exists():
        vcf_file = truth_path / "truth_variants.vcf"
    strain_snvs = defaultdict(lambda: defaultdict(dict))  # strain -> contig -> pos -> allele

    if vcf_file.exists():
        with open(vcf_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                contig = parts[0]
                pos = int(parts[1]) - 1  # Convert to 0-indexed
                ref = parts[3]
                alts = parts[4].split(',')
                info = parts[7]

                # Parse STRAINS info field
                if 'STRAINS=' in info:
                    strains_info = info.split('STRAINS=')[1].split(';')[0]
                    for allele_info in strains_info.split(';'):
                        if ':' in allele_info:
                            allele, strain_list = allele_info.split(':')
                            for strain_id in strain_list.split(','):
                                strain_snvs[strain_id][contig][pos] = allele

    # Also load reference (first strain has no SNVs, uses reference alleles)
    # The reference strain's positions are all reference alleles

    # Build TrueHaplotype objects
    haplotypes = []
    for strain_id, info in strain_info.items():
        # Skip reference strain (has 0 SNVs)
        if info['snv_count'] == 0:
            continue

        hap = TrueHaplotype(
            strain_id=strain_id,
            snv_positions=dict(strain_snvs.get(strain_id, {})),
            abundances=dict(strain_abundances.get(strain_id, {})),
            is_sweeping=info['is_sweeping']
        )
        haplotypes.append(hap)

    # Load all SNV positions
    snv_file = truth_path / "truth_snv_positions.tsv"
    all_snv_positions = defaultdict(list)
    if snv_file.exists():
        with open(snv_file) as f:
            f.readline()  # Skip header
            for line in f:
                contig, pos = line.strip().split('\t')
                all_snv_positions[contig].append(int(pos))

    logger.info(f"Loaded {len(haplotypes)} true haplotypes with SNVs")
    return haplotypes, dict(all_snv_positions)


# =============================================================================
# Load detected haplotypes
# =============================================================================

def load_detected_haplotypes(lineages_file: str) -> List[DetectedHaplotype]:
    """
    Load detected haplotypes from strainphase output.

    Expects TSV with columns: lineage_id, sample, contig, track_id, abundance, snv_alleles, ...
    """
    detected = []
    lineage_data = defaultdict(lambda: {'abundances': {}, 'snvs': defaultdict(dict)})

    with open(lineages_file) as f:
        header = f.readline().strip().split('\t')

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue

            row = dict(zip(header, parts))

            lineage_id = row.get('lineage_id', row.get('track_id', ''))
            sample = row.get('sample', row.get('timepoint', ''))
            contig = row.get('contig', '')
            abundance = float(row.get('abundance', row.get('weight', 0)))

            # Store abundance
            lineage_data[lineage_id]['abundances'][sample] = abundance

            # Parse SNV alleles if present
            # Format might be: "pos1:A,pos2:G,pos3:T" or separate columns
            snv_col = row.get('snv_alleles', row.get('consensus', ''))
            if snv_col and snv_col != '.':
                for snv in snv_col.split(','):
                    if ':' in snv:
                        pos_str, allele = snv.split(':')
                        pos = int(pos_str)
                        lineage_data[lineage_id]['snvs'][contig][pos] = allele

    # Convert to DetectedHaplotype objects
    for lineage_id, data in lineage_data.items():
        hap = DetectedHaplotype(
            lineage_id=lineage_id,
            track_id=lineage_id,
            snv_alleles=dict(data['snvs']),
            abundances=data['abundances']
        )
        detected.append(hap)

    logger.info(f"Loaded {len(detected)} detected haplotypes")
    return detected


# =============================================================================
# Matching algorithm
# =============================================================================

def compute_haplotype_distance(
    true_hap: TrueHaplotype,
    detected_hap: DetectedHaplotype
) -> Tuple[float, int, int, float]:
    """
    Compute distance between true and detected haplotype based on SNV overlap.

    Returns: (distance, n_matches, n_shared, match_fraction)
    """
    n_shared = 0
    n_matches = 0

    for contig, true_snvs in true_hap.snv_positions.items():
        det_snvs = detected_hap.snv_alleles.get(contig, {})

        for pos, true_allele in true_snvs.items():
            if pos in det_snvs:
                n_shared += 1
                if det_snvs[pos] == true_allele:
                    n_matches += 1

    if n_shared == 0:
        return 1.0, 0, 0, 0.0

    match_fraction = n_matches / n_shared
    distance = 1.0 - match_fraction
    return distance, n_matches, n_shared, match_fraction


def _abundance_within_factor(
    true_hap: TrueHaplotype,
    detected_hap: DetectedHaplotype,
    factor: float = 2.0
) -> bool:
    """Check if detected abundance stays within a multiplicative factor."""
    common_tps = set(true_hap.abundances.keys()) & set(detected_hap.abundances.keys())
    if not common_tps:
        return False

    for tp in common_tps:
        true_val = true_hap.abundances.get(tp, 0.0)
        det_val = detected_hap.abundances.get(tp, 0.0)
        if true_val <= 0:
            continue
        ratio = det_val / true_val
        if ratio < 1.0 / factor or ratio > factor:
            return False
    return True


def match_haplotypes(
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    max_distance: float = 0.1,
    min_shared_snvs: int = 3,
    min_match_fraction: float = 0.9
) -> List[Tuple[TrueHaplotype, DetectedHaplotype, float]]:
    """
    Match detected haplotypes to true haplotypes using greedy algorithm.

    Returns list of (true_hap, detected_hap, distance) tuples.
    """
    if not true_haps or not detected_haps:
        return []

    # Compute distance matrix
    distances = []
    for true_hap in true_haps:
        for det_hap in detected_haps:
            dist, n_matches, n_shared, match_fraction = compute_haplotype_distance(
                true_hap, det_hap
            )
            if n_shared < min_shared_snvs:
                continue
            if match_fraction < min_match_fraction:
                continue
            if not _abundance_within_factor(true_hap, det_hap, factor=2.0):
                continue
            distances.append((dist, true_hap, det_hap, n_shared))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Greedy matching
    matches = []
    used_true = set()
    used_detected = set()

    for dist, true_hap, det_hap, n_shared in distances:
        if dist > max_distance:
            break
        if true_hap.strain_id in used_true:
            continue
        if det_hap.lineage_id in used_detected:
            continue

        matches.append((true_hap, det_hap, dist))
        used_true.add(true_hap.strain_id)
        used_detected.add(det_hap.lineage_id)

    return matches


# =============================================================================
# Compute metrics
# =============================================================================

def compute_validation_metrics(
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    all_snv_positions: Dict[str, List[int]]
) -> ValidationResult:
    """Compute all validation metrics."""

    # Match haplotypes
    matches = match_haplotypes(true_haps, detected_haps)

    n_true = len(true_haps)
    n_detected = len(detected_haps)
    n_matched = len(matches)

    # Precision / Recall / F1
    precision = n_matched / n_detected if n_detected > 0 else 0.0
    recall = n_matched / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Abundance correlation
    true_abundances = []
    detected_abundances = []

    for true_hap, det_hap, _ in matches:
        # Find common timepoints
        common_tps = set(true_hap.abundances.keys()) & set(det_hap.abundances.keys())
        for tp in common_tps:
            true_abundances.append(true_hap.abundances[tp])
            detected_abundances.append(det_hap.abundances[tp])

    if len(true_abundances) >= 2:
        abundance_pearson_r = np.corrcoef(true_abundances, detected_abundances)[0, 1]
        abundance_mae = np.mean(np.abs(np.array(true_abundances) - np.array(detected_abundances)))
    else:
        abundance_pearson_r = 0.0
        abundance_mae = 1.0

    # SNV accuracy (for matched haplotypes)
    total_true_snvs = 0
    total_detected_snvs = 0
    total_correct_snvs = 0

    for true_hap, det_hap, _ in matches:
        for contig, true_snvs in true_hap.snv_positions.items():
            total_true_snvs += len(true_snvs)
            det_snvs = det_hap.snv_alleles.get(contig, {})
            total_detected_snvs += len(det_snvs)

            for pos, true_allele in true_snvs.items():
                if pos in det_snvs and det_snvs[pos] == true_allele:
                    total_correct_snvs += 1

    snv_precision = total_correct_snvs / total_detected_snvs if total_detected_snvs > 0 else 0.0
    snv_recall = total_correct_snvs / total_true_snvs if total_true_snvs > 0 else 0.0
    phasing_accuracy = snv_recall

    detection_threshold, _ = compute_detection_sensitivity(true_haps, matches)

    match_details = [(m[0].strain_id, m[1].lineage_id, m[2]) for m in matches]

    return ValidationResult(
        n_true=n_true,
        n_detected=n_detected,
        n_matched=n_matched,
        precision=precision,
        recall=recall,
        f1=f1,
        abundance_pearson_r=abundance_pearson_r,
        abundance_mae=abundance_mae,
        snv_precision=snv_precision,
        snv_recall=snv_recall,
        phasing_accuracy=phasing_accuracy,
        detection_threshold=detection_threshold,
        matches=match_details
    )


def compute_detection_sensitivity(
    true_haps: List[TrueHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    n_bins: int = 8
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Compute detection sensitivity curve and threshold.

    Returns: (detection_threshold, curve_dict)
    """
    matched_ids = {true_hap.strain_id for true_hap, _, _ in matches}
    abundance_points = []
    for true_hap in true_haps:
        for tp, abund in true_hap.abundances.items():
            abundance_points.append((abund, true_hap.strain_id in matched_ids))

    if not abundance_points:
        return 0.0, {"bins": [], "recall": []}

    max_abund = max(a for a, _ in abundance_points) or 1.0
    bins = np.linspace(0, max_abund, n_bins + 1)
    recall_by_bin = []

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        in_bin = [det for abund, det in abundance_points if low <= abund <= high]
        if not in_bin:
            recall_by_bin.append(0.0)
            continue
        recall_by_bin.append(sum(in_bin) / len(in_bin))

    threshold = 0.0
    for i, recall in enumerate(recall_by_bin):
        if recall >= 0.5:
            threshold = float(bins[i])
            break

    curve = {"bins": bins.tolist(), "recall": recall_by_bin}
    return threshold, curve


# =============================================================================
# Figure generation
# =============================================================================

def generate_figures(
    result: ValidationResult,
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    output_dir: str
):
    """Generate validation figures."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed, skipping figures")
        return

    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()

    # Figure 1: Accuracy bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ['Precision', 'Recall', 'F1']
    values = [result.precision, result.recall, result.f1]
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    bars = ax.bar(metrics, values, color=colors, edgecolor='black')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Haplotype Detection Accuracy')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=12)

    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target (90%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'haplotype_accuracy.png'), dpi=150)
    plt.close()

    # Figure 2: Abundance correlation
    if matches:
        fig, ax = plt.subplots(figsize=(6, 6))

        true_abundances = []
        detected_abundances = []

        for true_hap, det_hap, _ in matches:
            common_tps = set(true_hap.abundances.keys()) & set(det_hap.abundances.keys())
            for tp in common_tps:
                true_abundances.append(true_hap.abundances[tp])
                detected_abundances.append(det_hap.abundances[tp])

        if true_abundances:
            ax.scatter(true_abundances, detected_abundances, alpha=0.6, s=50)
            ax.plot([0, max(true_abundances)], [0, max(true_abundances)],
                   'r--', label='Perfect correlation')
            ax.set_xlabel('True Abundance')
            ax.set_ylabel('Detected Abundance')
            ax.set_title(f'Abundance Correlation (r={result.abundance_pearson_r:.3f})')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'abundance_correlation.png'), dpi=150)
        plt.close()

    # Figure 3: Detection sensitivity
    threshold, curve = compute_detection_sensitivity(true_haps, matches)
    if curve["bins"]:
        bins = np.array(curve["bins"])
        bin_centers = (bins[:-1] + bins[1:]) / 2
        recalls = curve["recall"]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(bin_centers, recalls, marker='o', linewidth=2, color='#1f77b4')
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
        if threshold > 0:
            ax.axvline(threshold, color='#d62728', linestyle='--',
                       label=f'Threshold ~ {threshold:.3f}')
            ax.legend()
        ax.set_xlabel('True Abundance')
        ax.set_ylabel('Recall')
        ax.set_title('Detection Sensitivity')
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_sensitivity.png'), dpi=150)
        plt.close()

    # Figure 4: Confusion matrix (match fraction)
    if true_haps and detected_haps:
        max_items = 20
        true_subset = true_haps[:max_items]
        det_subset = detected_haps[:max_items]

        matrix = np.zeros((len(true_subset), len(det_subset)))
        for i, true_hap in enumerate(true_subset):
            for j, det_hap in enumerate(det_subset):
                _, _, _, match_fraction = compute_haplotype_distance(true_hap, det_hap)
                matrix[i, j] = match_fraction

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
        ax.set_title('Haplotype Match Confusion Matrix')
        ax.set_xlabel('Detected Haplotype')
        ax.set_ylabel('True Haplotype')
        ax.set_xticks(range(len(det_subset)))
        ax.set_xticklabels([d.lineage_id for d in det_subset], rotation=45, ha='right')
        ax.set_yticks(range(len(true_subset)))
        ax.set_yticklabels([t.strain_id for t in true_subset])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Match Fraction')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

    logger.info(f"Saved figures to {output_dir}")


# =============================================================================
# Main validation pipeline
# =============================================================================

def run_validation(
    detected_file: str,
    truth_dir: str,
    output_dir: str
) -> ValidationResult:
    """Run the full validation pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info("Loading ground truth...")
    true_haps, all_snv_positions = load_ground_truth(truth_dir)

    logger.info("Loading detected haplotypes...")
    detected_haps = load_detected_haplotypes(detected_file)

    # Compute metrics
    logger.info("Computing validation metrics...")
    result = compute_validation_metrics(true_haps, detected_haps, all_snv_positions)

    # Match for figures
    matches = match_haplotypes(true_haps, detected_haps)

    # Generate figures
    generate_figures(result, true_haps, detected_haps, matches, output_dir)

    # Save metrics
    metrics_file = os.path.join(output_dir, 'validation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'n_true_haplotypes': result.n_true,
            'n_detected_haplotypes': result.n_detected,
            'n_matched': result.n_matched,
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1,
            'abundance_pearson_r': result.abundance_pearson_r,
            'abundance_mae': result.abundance_mae,
            'snv_precision': result.snv_precision,
            'snv_recall': result.snv_recall,
            'phasing_accuracy': result.phasing_accuracy,
            'detection_threshold': result.detection_threshold,
            'matches': result.matches
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"True haplotypes:     {result.n_true}")
    print(f"Detected haplotypes: {result.n_detected}")
    print(f"Matched:             {result.n_matched}")
    print("-" * 60)
    print(f"Precision:           {result.precision:.3f}")
    print(f"Recall:              {result.recall:.3f}")
    print(f"F1 Score:            {result.f1:.3f}")
    print("-" * 60)
    print(f"Abundance Pearson r: {result.abundance_pearson_r:.3f}")
    print(f"Abundance MAE:       {result.abundance_mae:.3f}")
    print("-" * 60)
    print(f"SNV Precision:       {result.snv_precision:.3f}")
    print(f"SNV Recall:          {result.snv_recall:.3f}")
    print(f"Phasing Accuracy:    {result.phasing_accuracy:.3f}")
    print(f"Detection Threshold: {result.detection_threshold:.4f}")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate strainphase haplotypes against ground truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--detected", required=True,
                        help="Strainphase output file (lineages.tsv)")
    parser.add_argument("--truth", required=True,
                        help="Ground truth directory from simulation")
    parser.add_argument("--output", required=True,
                        help="Output directory for metrics and figures")

    args = parser.parse_args()

    run_validation(args.detected, args.truth, args.output)


if __name__ == "__main__":
    main()
