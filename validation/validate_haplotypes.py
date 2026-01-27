#!/usr/bin/env python3
"""
Validate strainphase haplotype reconstruction against ground truth.

This is the main validation module, used by:
1. benchmarks/parameter_sweep.py (automatic validation during benchmarking)
2. Standalone CLI (for manual validation runs)

Compares detected haplotypes/lineages to known ground truth from simulation
and computes accuracy metrics (precision, recall, F1, abundance correlation).

Usage (standalone):
    python validation/validate_haplotypes.py \
        --detected results/lineages.tsv \
        --truth data/simulated/ \
        --output results/validation/

Note: The benchmarking pipeline (run_full_benchmark.py) automatically runs
validation for each parameter configuration, so manual validation is typically
not needed unless testing specific results.
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
import warnings

try:
    import matplotlib.pyplot as plt
    # Suppress matplotlib warnings about categorical units
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
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

    # Track/linking validation (Strainphase-specific)
    track_fragmentation_mean: float = 0.0
    track_fragmentation_median: float = 0.0
    false_link_rate: float = 0.0
    missed_link_rate: float = 0.0
    track_consensus_error: float = 0.0

    # Longitudinal lineage validation (Strainphase-specific)
    lineage_precision: float = 0.0
    lineage_recall: float = 0.0
    lineage_f1: float = 0.0
    rescue_delta_recall_rare: float = 0.0
    abundance_trajectory_error: float = 0.0

    # Per-match details
    matches: List[Tuple[str, str, float]] = None  # (true_id, detected_id, distance)
    
    # Detailed diagnostics
    false_negatives: List[str] = None  # True haplotypes not detected
    false_positives: List[str] = None  # Detected lineages not matching truth
    match_details_full: List[Dict] = None  # Full match details with SNV counts, abundances, etc.
    per_contig_metrics: Dict[str, Dict] = None  # Metrics per contig
    per_timepoint_metrics: Dict[str, Dict] = None  # Metrics per timepoint
    
    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.matches is None:
            self.matches = []
        if self.false_negatives is None:
            self.false_negatives = []
        if self.false_positives is None:
            self.false_positives = []
        if self.match_details_full is None:
            self.match_details_full = []
        if self.per_contig_metrics is None:
            self.per_contig_metrics = {}
        if self.per_timepoint_metrics is None:
            self.per_timepoint_metrics = {}


# =============================================================================
# Plot styling
# =============================================================================

# Professional color palette (shared with generate_report.py)
COLOR_PALETTE = {
    'primary': '#2C3E50',      # Dark blue-gray
    'secondary': '#34495E',    # Medium blue-gray
    'accent': '#3498DB',       # Bright blue
    'success': '#27AE60',      # Green
    'warning': '#F39C12',      # Orange
    'error': '#E74C3C',        # Red
    'info': '#9B59B6',         # Purple
    'neutral': '#95A5A6',      # Gray
    'light': '#ECF0F1',        # Light gray
    'dark': '#1A1A1A',         # Near black
}

COLOR_SEQUENCES = {
    'qualitative': ['#2C3E50', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', '#E74C3C', '#1ABC9C', '#E67E22'],
}

def set_plot_style():
    """Set professional, clean figure style for validation plots with Arial font and no grid."""
    if not HAS_MATPLOTLIB:
        return

    # Use clean style without grid
    plt.style.use('default')
    
    # Font settings - Arial family
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Remove grid completely
    plt.rcParams['axes.grid'] = False
    
    # Clean spines - only show bottom and left
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    
    # Spine styling
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.edgecolor'] = COLOR_PALETTE['primary']
    
    # Figure and axes background
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    
    # Tick styling
    plt.rcParams['xtick.color'] = COLOR_PALETTE['primary']
    plt.rcParams['ytick.color'] = COLOR_PALETTE['primary']
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    
    # Line and marker styling
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['patch.linewidth'] = 1.2
    
    # Legend styling
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.95
    plt.rcParams['legend.edgecolor'] = COLOR_PALETTE['neutral']
    plt.rcParams['legend.facecolor'] = 'white'
    
    # Default figure size
    plt.rcParams['figure.figsize'] = (9, 5.5)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'


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
    all_strain_ids = set(strain_info.keys())  # All strain IDs (including reference)

    if vcf_file.exists():
        with open(vcf_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                contig = parts[0]
                pos = int(parts[1])  # VCF uses 1-indexed positions (matches consensus)
                ref = parts[3]
                alts = parts[4].split(',')
                info = parts[7]

                # Track which strains have alt alleles
                strains_with_alt = set()
                
                # Parse STRAINS info field
                if 'STRAINS=' in info:
                    strains_info = info.split('STRAINS=')[1].split(';')[0]
                    for allele_info in strains_info.split('|'):
                        if ':' in allele_info:
                            allele, strain_list = allele_info.split(':')
                            for strain_id in strain_list.split(','):
                                strain_snvs[strain_id][contig][pos] = allele
                                strains_with_alt.add(strain_id)
                
                # Strains not in STRAINS field have reference alleles
                # (This includes the reference strain and any strains not explicitly listed)
                for strain_id in all_strain_ids:
                    if strain_id not in strains_with_alt:
                        strain_snvs[strain_id][contig][pos] = ref

    # Build TrueHaplotype objects (including reference strain)
    haplotypes = []
    for strain_id, info in strain_info.items():
        # Include ALL strains, including reference (reference alleles are informative)
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

    return haplotypes, dict(all_snv_positions)


# =============================================================================
# Load detected haplotypes
# =============================================================================

def load_detected_haplotypes(lineages_file: str) -> List[DetectedHaplotype]:
    """
    Load detected haplotypes from strainphase output.

    Expects TSV with columns: lineage_id, sample, contig, track_id, abundance, snv_alleles, ...
    Returns empty list if file doesn't exist or is empty (no haplotypes detected).
    """
    detected = []
    lineage_data = defaultdict(lambda: {'abundances': {}, 'snvs': defaultdict(dict)})

    # Check if file exists
    if not Path(lineages_file).exists():
        logger.warning(f"Lineages file not found: {lineages_file} (no haplotypes detected)")
        return []

    with open(lineages_file) as f:
        header_line = f.readline().strip()
        if not header_line:
            logger.warning(f"Empty lineages file: {lineages_file}")
            return []
        
        header = header_line.split('\t')

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue

            row = dict(zip(header, parts))

            lineage_id = row.get('lineage_id', row.get('track_id', ''))
            if not lineage_id:
                continue
                
            sample = row.get('sample', row.get('timepoint', ''))
            contig = row.get('contig', '')
            # Try multiple field names for abundance (mean_weight from build_lineage_table, abundance/weight from other formats)
            abundance = float(row.get('abundance', row.get('mean_weight', row.get('weight', 0))))

            # Store abundance
            lineage_data[lineage_id]['abundances'][sample] = abundance

            # Parse SNV alleles if present
            # Format might be: "pos1:A,pos2:G,pos3:T" (comma-separated) or "pos1:A|pos2:G" (pipe-separated)
            snv_col = row.get('snv_alleles', row.get('consensus', ''))
            if snv_col and snv_col != '.':
                # Handle both comma and pipe separators
                snv_list = snv_col.replace('|', ',').split(',')
                for snv in snv_list:
                    snv = snv.strip()
                    if ':' in snv:
                        pos_str, allele = snv.split(':', 1)
                        try:
                            pos = int(pos_str)
                            lineage_data[lineage_id]['snvs'][contig][pos] = allele
                        except ValueError:
                            continue  # Skip invalid positions

    # Convert to DetectedHaplotype objects
    for lineage_id, data in lineage_data.items():
        hap = DetectedHaplotype(
            lineage_id=lineage_id,
            track_id=lineage_id,
            snv_alleles=dict(data['snvs']),
            abundances=data['abundances']
        )
        detected.append(hap)

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
    min_match_fraction: float = 0.9,
    allow_one_to_many: bool = True
) -> List[Tuple[TrueHaplotype, DetectedHaplotype, float]]:
    """
    Match detected haplotypes to true haplotypes.
    
    Since strainphase may split strains per-contig, one true strain can match
    multiple detected lineages (one per contig). Set allow_one_to_many=True
    to account for this.

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

    if allow_one_to_many:
        # Allow one true strain to match multiple detected lineages (per-contig splitting)
        # But ensure each detected lineage only matches one true strain
        matches = []
        used_detected = set()
        
        # Group detected haplotypes by contig to identify per-contig splits
        detected_by_contig = defaultdict(list)
        for det_hap in detected_haps:
            # Get the contig(s) this detected haplotype spans
            contigs = set(det_hap.snv_alleles.keys())
            if contigs:
                # Use first contig as primary (most detected haps span one contig)
                primary_contig = sorted(contigs)[0]
                detected_by_contig[primary_contig].append(det_hap)
        
        # Match each true strain to detected lineages
        # Allow multiple matches per true strain if they cover different SNV positions
        # (handles fragmentation where one strain is split into multiple detected lineages)
        for true_hap in true_haps:
            # Collect all valid matches for this true strain
            candidate_matches = []
            for dist, true_h, det_h, n_shared in distances:
                if true_h.strain_id != true_hap.strain_id:
                    continue
                if det_h.lineage_id in used_detected:
                    continue
                if dist > max_distance:
                    continue
                candidate_matches.append((dist, det_h, n_shared))
            
            # Sort by distance (best first)
            candidate_matches.sort(key=lambda x: x[0])
            
            # Match greedily: allow multiple matches if they don't overlap significantly
            # Two detected lineages overlap if they share many SNV positions
            matched_for_strain = []
            for dist, det_h, n_shared in candidate_matches:
                # Check if this detected lineage overlaps significantly with already-matched ones
                overlaps = False
                for _, existing_det_h, _ in matched_for_strain:
                    # Check SNV overlap between det_h and existing_det_h
                    overlap_count = 0
                    total_positions = 0
                    for contig in set(det_h.snv_alleles.keys()) | set(existing_det_h.snv_alleles.keys()):
                        det_positions = set(det_h.snv_alleles.get(contig, {}).keys())
                        existing_positions = set(existing_det_h.snv_alleles.get(contig, {}).keys())
                        overlap_count += len(det_positions & existing_positions)
                        total_positions += len(det_positions | existing_positions)
                    
                    # If >50% overlap, consider them duplicates (only match one)
                    if total_positions > 0 and overlap_count / total_positions > 0.5:
                        overlaps = True
                        break
                
                if not overlaps:
                    matched_for_strain.append((dist, det_h, n_shared))
                    matches.append((true_hap, det_h, dist))
                    used_detected.add(det_h.lineage_id)
        
        return matches
    else:
        # Original 1-to-1 matching
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
    """
    Compute all validation metrics with detailed diagnostics.
    
    Note: This accounts for per-contig splitting. If strainphase detects lineages
    per-contig (e.g., 2 strains × 3 contigs = 6 detected lineages), one true strain
    can match multiple detected lineages (one per contig). This is correct behavior
    and is accounted for in precision/recall calculations.
    """

    # Match haplotypes (allow one-to-many to account for per-contig splitting)
    matches = match_haplotypes(true_haps, detected_haps, allow_one_to_many=True)

    n_true = len(true_haps)
    n_detected = len(detected_haps)
    n_matches = len(matches)  # Number of match pairs (can be > n_true if per-contig splitting)

    # Identify matched strains and lineages
    matched_true_ids = {m[0].strain_id for m in matches}
    matched_detected_ids = {m[1].lineage_id for m in matches}
    
    # Precision: fraction of detected lineages that match a true strain
    # (With per-contig splitting, multiple detected lineages can match one true strain)
    precision = len(matched_detected_ids) / n_detected if n_detected > 0 else 0.0
    
    # Recall: fraction of true strains that were detected
    # (A true strain is "detected" if at least one detected lineage matches it)
    recall = len(matched_true_ids) / n_true if n_true > 0 else 0.0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    false_negatives = [h.strain_id for h in true_haps if h.strain_id not in matched_true_ids]
    false_positives = [h.lineage_id for h in detected_haps if h.lineage_id not in matched_detected_ids]

    # Build detailed match information
    match_details_full = []
    for true_hap, det_hap, distance in matches:
        # Compute SNV statistics
        n_shared_snvs = 0
        n_matching_snvs = 0
        n_true_snvs = sum(len(snvs) for snvs in true_hap.snv_positions.values())
        n_detected_snvs = sum(len(snvs) for snvs in det_hap.snv_alleles.values())
        
        for contig, true_snvs in true_hap.snv_positions.items():
            det_snvs = det_hap.snv_alleles.get(contig, {})
            for pos, true_allele in true_snvs.items():
                if pos in det_snvs:
                    n_shared_snvs += 1
                    if det_snvs[pos] == true_allele:
                        n_matching_snvs += 1
        
        # Compute abundance statistics
        common_tps = set(true_hap.abundances.keys()) & set(det_hap.abundances.keys())
        abundance_errors = []
        for tp in common_tps:
            true_abund = true_hap.abundances[tp]
            det_abund = det_hap.abundances[tp]
            abundance_errors.append(abs(true_abund - det_abund))
        
        match_details_full.append({
            'true_strain_id': true_hap.strain_id,
            'detected_lineage_id': det_hap.lineage_id,
            'distance': distance,
            'n_true_snvs': n_true_snvs,
            'n_detected_snvs': n_detected_snvs,
            'n_shared_snvs': n_shared_snvs,
            'n_matching_snvs': n_matching_snvs,
            'snv_match_fraction': n_matching_snvs / n_shared_snvs if n_shared_snvs > 0 else 0.0,
            'abundance_mae': np.mean(abundance_errors) if abundance_errors else None,
            'common_timepoints': list(common_tps),
            'true_abundances': {tp: true_hap.abundances[tp] for tp in common_tps},
            'detected_abundances': {tp: det_hap.abundances[tp] for tp in common_tps},
            'is_sweeping': true_hap.is_sweeping,
        })

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
    # Aggregate SNVs per true strain to handle fragmentation correctly:
    # If a strain is split into multiple tracks, we should count each true SNV
    # only once, and consider it "recovered" if ANY matching track has it correct.

    # Group matches by true strain
    matches_by_strain: Dict[str, List[DetectedHaplotype]] = defaultdict(list)
    for true_hap, det_hap, _ in matches:
        matches_by_strain[true_hap.strain_id].append(det_hap)

    total_true_snvs = 0
    total_detected_snvs = 0
    total_correct_snvs = 0

    for true_hap in true_haps:
        # Get all detected tracks matching this true strain
        matching_tracks = matches_by_strain.get(true_hap.strain_id, [])
        if not matching_tracks:
            # Count true SNVs even for unmatched strains (affects recall denominator)
            for contig, true_snvs in true_hap.snv_positions.items():
                total_true_snvs += len(true_snvs)
            continue

        # Aggregate detected SNVs across all matching tracks
        # detected_snvs_union[contig][pos] = allele (from any matching track)
        detected_snvs_union: Dict[str, Dict[int, str]] = defaultdict(dict)
        for det_hap in matching_tracks:
            for contig, det_snvs in det_hap.snv_alleles.items():
                for pos, allele in det_snvs.items():
                    # Keep first allele seen (they should all agree if tracks are correct)
                    if pos not in detected_snvs_union[contig]:
                        detected_snvs_union[contig][pos] = allele

        # Count true SNVs and check if they're recovered in union of detected tracks
        for contig, true_snvs in true_hap.snv_positions.items():
            total_true_snvs += len(true_snvs)
            det_snvs = detected_snvs_union.get(contig, {})

            for pos, true_allele in true_snvs.items():
                if pos in det_snvs and det_snvs[pos] == true_allele:
                    total_correct_snvs += 1

        # Count total detected SNVs (union across all matching tracks, avoid double-counting)
        for contig, det_snvs in detected_snvs_union.items():
            total_detected_snvs += len(det_snvs)

    snv_precision = total_correct_snvs / total_detected_snvs if total_detected_snvs > 0 else 0.0
    snv_recall = total_correct_snvs / total_true_snvs if total_true_snvs > 0 else 0.0
    phasing_accuracy = snv_recall

    detection_threshold, _ = compute_detection_sensitivity(true_haps, matches)

    match_details = [(m[0].strain_id, m[1].lineage_id, m[2]) for m in matches]

    # Per-contig metrics
    per_contig_metrics = {}
    for contig in all_snv_positions.keys():
        contig_true_haps = [h for h in true_haps if contig in h.snv_positions]
        contig_detected_haps = [h for h in detected_haps if contig in h.snv_alleles]
        contig_matches = [m for m in matches if contig in m[0].snv_positions and contig in m[1].snv_alleles]
        
        n_true_contig = len(contig_true_haps)
        n_detected_contig = len(contig_detected_haps)
        # Count unique matched true haplotypes (not match pairs, since allow_one_to_many=True)
        matched_true_ids_contig = {m[0].strain_id for m in contig_matches}
        matched_detected_ids_contig = {m[1].lineage_id for m in contig_matches}
        n_matched_true_contig = len(matched_true_ids_contig)
        n_matched_detected_contig = len(matched_detected_ids_contig)
        
        per_contig_metrics[contig] = {
            'n_true': n_true_contig,
            'n_detected': n_detected_contig,
            'n_matched': len(contig_matches),  # Total match pairs (for reference)
            'n_matched_true': n_matched_true_contig,  # Unique true haplotypes matched
            'n_matched_detected': n_matched_detected_contig,  # Unique detected lineages matched
            'precision': n_matched_detected_contig / n_detected_contig if n_detected_contig > 0 else 0.0,
            'recall': n_matched_true_contig / n_true_contig if n_true_contig > 0 else 0.0,
        }

    # Per-timepoint metrics
    all_timepoints = set()
    for h in true_haps:
        all_timepoints.update(h.abundances.keys())
    for h in detected_haps:
        all_timepoints.update(h.abundances.keys())
    
    per_timepoint_metrics = {}
    for tp in sorted(all_timepoints):
        tp_true_haps = [h for h in true_haps if tp in h.abundances and h.abundances[tp] > 0.01]
        tp_detected_haps = [h for h in detected_haps if tp in h.abundances and h.abundances[tp] > 0.01]
        tp_matches = [m for m in matches if tp in m[0].abundances and tp in m[1].abundances]
        
        n_true_tp = len(tp_true_haps)
        n_detected_tp = len(tp_detected_haps)
        # Count unique matched true haplotypes (not match pairs, since allow_one_to_many=True)
        matched_true_ids_tp = {m[0].strain_id for m in tp_matches}
        matched_detected_ids_tp = {m[1].lineage_id for m in tp_matches}
        n_matched_true_tp = len(matched_true_ids_tp)
        n_matched_detected_tp = len(matched_detected_ids_tp)
        
        # Abundance correlation for this timepoint
        tp_true_abunds = [m[0].abundances[tp] for m in tp_matches]
        tp_detected_abunds = [m[1].abundances[tp] for m in tp_matches]
        if len(tp_true_abunds) >= 2:
            tp_abund_r = np.corrcoef(tp_true_abunds, tp_detected_abunds)[0, 1]
            tp_abund_mae = np.mean(np.abs(np.array(tp_true_abunds) - np.array(tp_detected_abunds)))
        else:
            tp_abund_r = None
            tp_abund_mae = None
        
        per_timepoint_metrics[tp] = {
            'n_true': n_true_tp,
            'n_detected': n_detected_tp,
            'n_matched': len(tp_matches),  # Total match pairs (for reference)
            'n_matched_true': n_matched_true_tp,  # Unique true haplotypes matched
            'n_matched_detected': n_matched_detected_tp,  # Unique detected lineages matched
            'precision': n_matched_detected_tp / n_detected_tp if n_detected_tp > 0 else 0.0,
            'recall': n_matched_true_tp / n_true_tp if n_true_tp > 0 else 0.0,
            'abundance_pearson_r': tp_abund_r,
            'abundance_mae': tp_abund_mae,
        }

    return ValidationResult(
        n_true=n_true,
        n_detected=n_detected,
        n_matched=len(matched_true_ids),  # Number of unique true strains matched
        precision=precision,
        recall=recall,
        f1=f1,
        abundance_pearson_r=abundance_pearson_r,
        abundance_mae=abundance_mae,
        snv_precision=snv_precision,
        snv_recall=snv_recall,
        phasing_accuracy=phasing_accuracy,
        detection_threshold=detection_threshold,
        matches=match_details,
        false_negatives=false_negatives,
        false_positives=false_positives,
        match_details_full=match_details_full,
        per_contig_metrics=per_contig_metrics,
        per_timepoint_metrics=per_timepoint_metrics,
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
    output_dir: str,
    window_results: Optional[List] = None,  # Optional WindowResult list for track visualization
    truth_dir: Optional[str] = None  # Optional truth directory for loading truth tracks
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
    colors = [COLOR_PALETTE['success'], COLOR_PALETTE['accent'], COLOR_PALETTE['info']]

    bars = ax.bar(metrics, values, color=colors, edgecolor=COLOR_PALETTE['primary'], 
                  linewidth=1.5, alpha=0.85)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_title('Haplotype Detection Accuracy', fontweight='bold', color=COLOR_PALETTE['primary'])

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=12, fontweight='bold', 
                color=COLOR_PALETTE['primary'])

    ax.axhline(y=0.9, color=COLOR_PALETTE['warning'], linestyle='--', 
               linewidth=2.0, alpha=0.7, label='Target (90%)')
    ax.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'haplotype_accuracy.png'), dpi=300, bbox_inches='tight')
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
            ax.scatter(true_abundances, detected_abundances, alpha=0.7, s=60, 
                      color=COLOR_PALETTE['accent'], edgecolors=COLOR_PALETTE['primary'], 
                      linewidths=0.8)
            ax.plot([0, max(true_abundances)], [0, max(true_abundances)],
                   color=COLOR_PALETTE['error'], linestyle='--', linewidth=2.0, 
                   label='Perfect correlation', alpha=0.8)
            ax.set_xlabel('True Abundance', fontweight='bold', color=COLOR_PALETTE['primary'])
            ax.set_ylabel('Detected Abundance', fontweight='bold', color=COLOR_PALETTE['primary'])
            ax.set_title(f'Abundance Correlation (r={result.abundance_pearson_r:.3f})', 
                        fontweight='bold', color=COLOR_PALETTE['primary'])
            ax.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])

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
        ax.plot(bin_centers, recalls, marker='o', linewidth=2.5, markersize=8, 
               color=COLOR_PALETTE['accent'], markerfacecolor=COLOR_PALETTE['accent'],
               markeredgecolor=COLOR_PALETTE['primary'], markeredgewidth=1.0)
        ax.axhline(0.5, color=COLOR_PALETTE['neutral'], linestyle='--', linewidth=1.5, alpha=0.7)
        if threshold > 0:
            ax.axvline(threshold, color=COLOR_PALETTE['error'], linestyle='--', linewidth=2.0,
                       label=f'Threshold ~ {threshold:.3f}', alpha=0.8)
            ax.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
        ax.set_xlabel('True Abundance', fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_ylabel('Recall', fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_title('Detection Sensitivity', fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Figure 5: Per-haplotype matching details
    if result.match_details_full:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # SNV match fraction distribution
        snv_fractions = [m['snv_match_fraction'] for m in result.match_details_full]
        axes[0, 0].hist(snv_fractions, bins=20, edgecolor=COLOR_PALETTE['primary'], 
                       color=COLOR_PALETTE['accent'], alpha=0.8, linewidth=1.2)
        axes[0, 0].axvline(0.9, color=COLOR_PALETTE['error'], linestyle='--', 
                          linewidth=2.0, label='90% threshold', alpha=0.8)
        axes[0, 0].set_xlabel('SNV Match Fraction', fontweight='bold', color=COLOR_PALETTE['primary'])
        axes[0, 0].set_ylabel('Matched haplotype pairs', fontweight='bold', color=COLOR_PALETTE['primary'])
        axes[0, 0].set_title('SNV match fraction per matched haplotype pair', 
                            fontweight='bold', color=COLOR_PALETTE['primary'])
        axes[0, 0].legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
        
        # Abundance error distribution
        abund_errors = [m['abundance_mae'] for m in result.match_details_full if m['abundance_mae'] is not None]
        if abund_errors:
            axes[0, 1].hist(abund_errors, bins=20, edgecolor=COLOR_PALETTE['primary'], 
                          alpha=0.8, color=COLOR_PALETTE['warning'], linewidth=1.2)
            axes[0, 1].set_xlabel('Mean Absolute Abundance Error', 
                                fontweight='bold', color=COLOR_PALETTE['primary'])
            axes[0, 1].set_ylabel('Matched haplotype pairs', 
                                fontweight='bold', color=COLOR_PALETTE['primary'])
            axes[0, 1].set_title('Abundance MAE per matched haplotype pair',
                               fontweight='bold', color=COLOR_PALETTE['primary'])
        
        # SNV counts comparison
        true_counts = [m['n_true_snvs'] for m in result.match_details_full]
        detected_counts = [m['n_detected_snvs'] for m in result.match_details_full]
        axes[1, 0].scatter(true_counts, detected_counts, alpha=0.7, s=60,
                          color=COLOR_PALETTE['accent'], edgecolors=COLOR_PALETTE['primary'],
                          linewidths=0.8)
        max_count = max(max(true_counts, default=0), max(detected_counts, default=0))
        axes[1, 0].plot([0, max_count], [0, max_count], 
                       color=COLOR_PALETTE['error'], linestyle='--', linewidth=2.0,
                       label='Perfect match', alpha=0.8)
        axes[1, 0].set_xlabel('True SNV Count', fontweight='bold', color=COLOR_PALETTE['primary'])
        axes[1, 0].set_ylabel('Detected SNV Count', fontweight='bold', color=COLOR_PALETTE['primary'])
        axes[1, 0].set_title('SNV counts: true vs detected (matched pairs)', 
                           fontweight='bold', color=COLOR_PALETTE['primary'])
        axes[1, 0].legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
        
        # Per-timepoint recall
        if result.per_timepoint_metrics:
            tps = sorted(result.per_timepoint_metrics.keys())
            recalls = [result.per_timepoint_metrics[tp]['recall'] for tp in tps]
            precisions = [result.per_timepoint_metrics[tp]['precision'] for tp in tps]
            x = np.arange(len(tps))
            width = 0.35
            axes[1, 1].bar(x - width/2, recalls, width, label='Recall', 
                          alpha=0.85, color=COLOR_PALETTE['accent'], 
                          edgecolor=COLOR_PALETTE['primary'], linewidth=1.2)
            axes[1, 1].bar(x + width/2, precisions, width, label='Precision', 
                          alpha=0.85, color=COLOR_PALETTE['success'],
                          edgecolor=COLOR_PALETTE['primary'], linewidth=1.2)
            axes[1, 1].set_xlabel('Timepoint', fontweight='bold', color=COLOR_PALETTE['primary'])
            axes[1, 1].set_ylabel('Score', fontweight='bold', color=COLOR_PALETTE['primary'])
            axes[1, 1].set_title('Precision/Recall by Timepoint',
                               fontweight='bold', color=COLOR_PALETTE['primary'])
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(tps, color=COLOR_PALETTE['primary'])
            axes[1, 1].legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
            axes[1, 1].set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detailed_matching.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Figure 6: Abundance trajectories (single panel, color by timepoint, shape by true/detected)
    if matches:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Collect timepoints across matches
        all_tps = sorted({tp for m in matches for tp in m[0].abundances.keys()})
        tp_to_x = {tp: i for i, tp in enumerate(all_tps)}
        cmap = plt.get_cmap('viridis')
        colors = {tp: cmap(i / max(1, len(all_tps) - 1)) for i, tp in enumerate(all_tps)}

        # Plot points for each match
        for true_hap, det_hap, _ in matches:
            common_tps = sorted(set(true_hap.abundances.keys()) & set(det_hap.abundances.keys()))
            for tp in common_tps:
                x = tp_to_x[tp]
                ax.scatter(x, true_hap.abundances[tp], marker='o',
                           color=colors[tp], edgecolor=COLOR_PALETTE['primary'], s=40)
                ax.scatter(x, det_hap.abundances[tp], marker='s',
                           color=colors[tp], edgecolor=COLOR_PALETTE['primary'], s=40)

        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Abundance')
        ax.set_xticks(range(len(all_tps)))
        ax.set_xticklabels(all_tps)
        ax.set_ylim(0, 1.0)
        ax.set_title('Abundance trajectories (true vs detected)')

        # Combined legend: marker shape for true/detected, color for timepoint
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=COLOR_PALETTE['primary'], label='True', markersize=8),
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=COLOR_PALETTE['primary'], label='Detected', markersize=8),
        ]
        for tp in all_tps:
            handles.append(
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=colors[tp], label=f"{tp}", markersize=7)
            )
        ax.legend(handles=handles, frameon=True, loc="upper right", title="Legend")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'abundance_trajectories.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Figure 7: Track fragmentation (if track metrics available)
    if result.track_fragmentation_mean > 0 or result.per_contig_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Show fragmentation per contig if available
        if result.per_contig_metrics:
            contigs = sorted(result.per_contig_metrics.keys())
            # We'd need to extract fragmentation from per_contig_metrics if stored
            # For now, show overall mean/median
            ax.bar(['Mean', 'Median'], 
                  [result.track_fragmentation_mean, result.track_fragmentation_median],
                  color=[COLOR_PALETTE['accent'], COLOR_PALETTE['success']], 
                  edgecolor=COLOR_PALETTE['primary'], linewidth=1.5, alpha=0.85)
            ax.set_ylabel('Tracks per True Strain', fontweight='bold', color=COLOR_PALETTE['primary'])
            ax.set_title(f'Track Fragmentation (Mean: {result.track_fragmentation_mean:.2f}, Median: {result.track_fragmentation_median:.2f})',
                        fontweight='bold', color=COLOR_PALETTE['primary'])
            ax.axhline(y=1.0, color=COLOR_PALETTE['error'], linestyle='--', linewidth=2.0, 
                      label='Ideal (1 track per strain)', alpha=0.8)
            ax.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
        else:
            ax.text(0.5, 0.5, 'Track fragmentation data not available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'track_fragmentation.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Figure 8: Linking errors (if track metrics available)
    if result.false_link_rate > 0 or result.missed_link_rate > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = ['False Link Rate', 'Missed Link Rate']
        values = [result.false_link_rate, result.missed_link_rate]
        colors = [COLOR_PALETTE['error'], COLOR_PALETTE['warning']]
        
        bars = ax.bar(metrics, values, color=colors, edgecolor=COLOR_PALETTE['primary'], 
                     linewidth=1.5, alpha=0.85)
        ax.set_ylabel('Rate', fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_title('Window Linking Errors', fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_ylim(0, max(values) * 1.2 if values else 1.0)
        
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', fontsize=11, fontweight='bold',
                       color=COLOR_PALETTE['primary'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'linking_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Figure 9: Lineage accuracy (if lineage metrics available)
    if result.lineage_precision > 0 or result.lineage_recall > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = ['Lineage Precision', 'Lineage Recall', 'Lineage F1']
        values = [result.lineage_precision, result.lineage_recall, result.lineage_f1]
        colors = [COLOR_PALETTE['success'], COLOR_PALETTE['accent'], COLOR_PALETTE['info']]
        
        bars = ax.bar(metrics, values, color=colors, edgecolor=COLOR_PALETTE['primary'], 
                     linewidth=1.5, alpha=0.85)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score', fontweight='bold', color=COLOR_PALETTE['primary'])
        ax.set_title('Longitudinal Lineage Accuracy', fontweight='bold', color=COLOR_PALETTE['primary'])
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', fontsize=12, fontweight='bold', 
                   color=COLOR_PALETTE['primary'])
        
        if result.rescue_delta_recall_rare > 0:
            ax.text(0.5, -0.15, f'Rescue ΔRecall (rare): {result.rescue_delta_recall_rare:.3f}',
                   ha='center', transform=ax.transAxes, fontsize=11, style='italic',
                   color=COLOR_PALETTE['secondary'], fontweight='600')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lineage_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Figure 10: Track visualization on contigs
    if window_results and truth_dir:
        try:
            from validation.validate_tracks import load_truth_tracks
            
            # Extract detected tracks from window_results
            # Group by track_id and compute spans per contig
            detected_tracks: Dict[str, Dict[str, Tuple[int, int]]] = defaultdict(dict)  # track_id -> {contig -> (start, end)}
            track_to_strain: Dict[str, str] = {}  # track_id -> strain_id (from matches)
            
            # Build mapping from lineage_id to strain_id
            # Note: lineage_id in DetectedHaplotype equals track_id in window_results
            # (see window_results_to_lineages_tsv in parameter_sweep.py)
            for true_hap, det_hap, _ in matches:
                if det_hap.lineage_id:
                    # lineage_id equals track_id, so we can use it directly
                    track_to_strain[det_hap.lineage_id] = true_hap.strain_id
            
            # Extract track spans from window_results
            for wr in window_results:
                contig = wr.window.contig
                for hap in wr.haplotypes:
                    # Use track_id (lineage_id is not on Haplotype object, only in DetectedHaplotype)
                    track_id = hap.track_id
                    if not track_id:
                        # Fallback: create a unique ID for unlinked haplotypes
                        track_id = f"unlinked_{contig}_{wr.window.start}"
                    
                    if track_id:
                        if contig not in detected_tracks[track_id]:
                            detected_tracks[track_id][contig] = (wr.window.start, wr.window.end)
                        else:
                            # Extend span
                            old_start, old_end = detected_tracks[track_id][contig]
                            detected_tracks[track_id][contig] = (
                                min(old_start, wr.window.start),
                                max(old_end, wr.window.end)
                            )
            
            # Load truth tracks
            truth_tracks = load_truth_tracks(truth_dir)
            
            if truth_tracks and detected_tracks:
                # Group by contig
                all_contigs = set()
                for tracks_dict in [truth_tracks.values(), detected_tracks.values()]:
                    for contig_dict in tracks_dict:
                        all_contigs.update(contig_dict.keys())
                
                if all_contigs:
                    n_contigs = len(all_contigs)
                    n_cols = min(3, n_contigs)
                    n_rows = (n_contigs + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
                    if n_rows == 1:
                        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
                    axes = axes.flatten()
                    
                    for idx, contig in enumerate(sorted(all_contigs)):
                        ax = axes[idx]
                        
                        # Get contig length (max end position)
                        max_pos = 0
                        for tracks_dict in [truth_tracks.values(), detected_tracks.values()]:
                            for contig_dict in tracks_dict:
                                if contig in contig_dict:
                                    _, end = contig_dict[contig]
                                    max_pos = max(max_pos, end)
                        
                        if max_pos == 0:
                            ax.text(0.5, 0.5, f'No tracks on {contig}', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(contig)
                            continue
                        
                        # Plot truth tracks (gray background bars)
                        y_offset = 0
                        truth_y_positions = {}
                        truth_labels_added = set()
                        for strain_id, contig_dict in truth_tracks.items():
                            if contig in contig_dict:
                                start, end = contig_dict[contig]
                                truth_y_positions[strain_id] = y_offset
                                label = 'Truth track' if 'Truth track' not in truth_labels_added else ''
                                if label:
                                    truth_labels_added.add('Truth track')
                                ax.barh(y_offset, end - start, left=start, height=0.6,
                                       color=COLOR_PALETTE['light'], edgecolor=COLOR_PALETTE['neutral'], 
                                       alpha=0.4, linewidth=1.0, label=label)
                                # Add strain label
                                ax.text(start + (end - start) / 2, y_offset, strain_id[:12],
                                       ha='center', va='center', fontsize=9, rotation=0,
                                       weight='bold', color=COLOR_PALETTE['primary'])
                                y_offset += 1
                        
                        # Plot detected tracks (colored by match status)
                        detected_labels_added = set()
                        for track_id, contig_dict in detected_tracks.items():
                            if contig in contig_dict:
                                start, end = contig_dict[contig]
                                matched_strain = track_to_strain.get(track_id)
                                
                                # Determine color based on match status
                                color = COLOR_PALETTE['error']  # Default: red (false positive)
                                match_type = 'False positive'
                                
                                if matched_strain and matched_strain in truth_tracks:
                                    # Check if this is a perfect match (overlaps truth track)
                                    truth_start, truth_end = truth_tracks[matched_strain].get(contig, (0, 0))
                                    if truth_start > 0 or truth_end > 0:  # Valid truth track
                                        overlap_start = max(start, truth_start)
                                        overlap_end = min(end, truth_end)
                                        overlap_len = max(0, overlap_end - overlap_start)
                                        track_len = end - start
                                        truth_len = truth_end - truth_start
                                        
                                        if overlap_len > 0:
                                            # Compute overlap fraction
                                            overlap_frac = overlap_len / max(track_len, truth_len)
                                            if overlap_frac > 0.9:
                                                color = COLOR_PALETTE['success']  # Green: perfect match
                                                match_type = 'Perfect match'
                                            else:
                                                color = COLOR_PALETTE['warning']  # Orange: partial match
                                                match_type = 'Partial match'
                                        else:
                                            color = COLOR_PALETTE['error']  # Red: false positive (wrong position)
                                            match_type = 'False positive (wrong position)'
                                
                                # Add to legend only once per match type
                                label = ''
                                if match_type not in detected_labels_added:
                                    label = match_type
                                    detected_labels_added.add(match_type)
                                
                                # Plot detected track
                                ax.barh(y_offset, end - start, left=start, height=0.4,
                                       color=color, edgecolor=COLOR_PALETTE['primary'], 
                                       alpha=0.85, linewidth=1.2, label=label)
                                # Add track label
                                label_text = track_id[:12] if not track_id.startswith('unlinked') else 'unlinked'
                                text_color = 'white' if color == COLOR_PALETTE['success'] else COLOR_PALETTE['primary']
                                ax.text(start + (end - start) / 2, y_offset, label_text,
                                       ha='center', va='center', fontsize=8, rotation=0,
                                       color=text_color,
                                       weight='bold' if color == COLOR_PALETTE['success'] else '600')
                                y_offset += 1
                        
                        ax.set_xlabel('Position (bp)', fontweight='bold', color=COLOR_PALETTE['primary'])
                        ax.set_ylabel('Track', fontweight='bold', color=COLOR_PALETTE['primary'])
                        ax.set_title(f'{contig}\nTruth (gray) vs Detected (colored)',
                                   fontweight='bold', color=COLOR_PALETTE['primary'])
                        ax.set_xlim(0, max_pos * 1.05)
                        # No grid - clean professional look
                        
                        if idx == 0:
                            ax.legend(loc='upper right', fontsize=9, frameon=True, 
                                    framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
                    
                    # Hide unused subplots
                    for idx in range(len(all_contigs), len(axes)):
                        axes[idx].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'track_regions.png'), dpi=300, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            logger.warning(f"Track visualization failed: {e}")

    # Figure 11: Per-abundance-bin performance (publication standard)
    _generate_per_abundance_performance(true_haps, detected_haps, matches, output_dir)
    
    # Figure 12: Strain divergence vs performance
    _generate_divergence_performance(true_haps, detected_haps, matches, output_dir)
    
    # Figure 13: ROC-like detection curve
    _generate_detection_roc_curve(true_haps, detected_haps, matches, output_dir)
    
    # Figure 14: Reference coverage (completeness)
    _generate_reference_coverage(true_haps, detected_haps, matches, output_dir)
    
    # Figure 15: Error type breakdown
    _generate_error_breakdown(true_haps, detected_haps, matches, output_dir)
    
    # Figure 16: Performance vs strain count (scalability)
    _generate_scalability_analysis(true_haps, detected_haps, matches, output_dir)


def _generate_per_abundance_performance(
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    output_dir: str
):
    """Generate per-abundance-bin performance plot (publication standard)."""
    if not HAS_MATPLOTLIB or not true_haps:
        return
    
    # Define abundance bins (standard ranges from literature)
    bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.50), (0.50, 1.0)]
    bin_labels = ['0-1%', '1-5%', '5-10%', '10-50%', '50-100%']
    
    matched_strain_ids = {m[0].strain_id for m in matches}
    
    # Compute metrics per bin
    bin_precision = []
    bin_recall = []
    bin_f1 = []
    bin_counts = []
    
    for low, high in bins:
        # Get strains in this abundance range (check all timepoints)
        strains_in_bin = []
        for true_hap in true_haps:
            max_abund = max(true_hap.abundances.values()) if true_hap.abundances else 0.0
            if low <= max_abund < high:
                strains_in_bin.append(true_hap)
        
        if not strains_in_bin:
            bin_precision.append(0.0)
            bin_recall.append(0.0)
            bin_f1.append(0.0)
            bin_counts.append(0)
            continue
        
        # Count detected strains in this bin
        detected_in_bin = sum(1 for h in strains_in_bin if h.strain_id in matched_strain_ids)
        
        # For precision: count detected lineages that match strains in this bin
        detected_lineages_in_bin = set()
        for true_hap, det_hap, _ in matches:
            if true_hap.strain_id in {h.strain_id for h in strains_in_bin}:
                detected_lineages_in_bin.add(det_hap.lineage_id)
        
        # Count all detected lineages (for precision denominator)
        total_detected_in_bin = len([h for h in detected_haps if h.lineage_id in detected_lineages_in_bin])
        
        precision = detected_in_bin / total_detected_in_bin if total_detected_in_bin > 0 else 0.0
        recall = detected_in_bin / len(strains_in_bin) if len(strains_in_bin) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        bin_precision.append(precision)
        bin_recall.append(recall)
        bin_f1.append(f1)
        bin_counts.append(len(strains_in_bin))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bin_labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, bin_precision, width, label='Precision', 
                   color=COLOR_PALETTE['success'], edgecolor=COLOR_PALETTE['primary'],
                   linewidth=1.2, alpha=0.85)
    bars2 = ax.bar(x, bin_recall, width, label='Recall',
                   color=COLOR_PALETTE['accent'], edgecolor=COLOR_PALETTE['primary'],
                   linewidth=1.2, alpha=0.85)
    bars3 = ax.bar(x + width, bin_f1, width, label='F1',
                   color=COLOR_PALETTE['info'], edgecolor=COLOR_PALETTE['primary'],
                   linewidth=1.2, alpha=0.85)
    
    # Add count annotations
    for i, (bar, count) in enumerate(zip(bars2, bin_counts)):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, -0.05,
                   f'n={count}', ha='center', va='top', fontsize=9,
                   color=COLOR_PALETTE['secondary'], style='italic')
    
    ax.set_xlabel('Abundance Range', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_ylabel('Score', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_title('Performance by Abundance Range', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, color=COLOR_PALETTE['primary'])
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_abundance_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _generate_divergence_performance(
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    output_dir: str
):
    """Generate performance vs strain divergence (SNV density) plot."""
    if not HAS_MATPLOTLIB or not true_haps:
        return
    
    matched_strain_ids = {m[0].strain_id for m in matches}
    
    # Compute SNV density per strain (SNVs per 10kb)
    strain_divergences = []
    strain_detected = []
    
    for true_hap in true_haps:
        total_snvs = sum(len(snvs) for snvs in true_hap.snv_positions.values())
        total_length = sum(
            max(snvs.keys()) - min(snvs.keys()) if snvs else 10000
            for snvs in true_hap.snv_positions.values()
        ) if true_hap.snv_positions else 10000
        
        # SNVs per 10kb
        snv_density = (total_snvs / max(total_length, 1)) * 10000 if total_length > 0 else 0.0
        strain_divergences.append(snv_density)
        strain_detected.append(1 if true_hap.strain_id in matched_strain_ids else 0)
    
    if not strain_divergences:
        return
    
    # Bin by divergence
    max_div = max(strain_divergences) if strain_divergences else 50
    bins = np.linspace(0, max_div, 6)
    bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    
    bin_recall = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        in_bin = [j for j, div in enumerate(strain_divergences) if low <= div < high]
        if not in_bin:
            bin_recall.append(0.0)
            bin_counts.append(0)
            continue
        
        detected_count = sum(strain_detected[j] for j in in_bin)
        recall = detected_count / len(in_bin) if len(in_bin) > 0 else 0.0
        bin_recall.append(recall)
        bin_counts.append(len(in_bin))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bin_labels))
    
    bars = ax.bar(x, bin_recall, color=COLOR_PALETTE['accent'], 
                 edgecolor=COLOR_PALETTE['primary'], linewidth=1.5, alpha=0.85)
    
    # Add count annotations
    for bar, count in zip(bars, bin_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'n={count}', ha='center', fontsize=9,
                   color=COLOR_PALETTE['secondary'], style='italic')
    
    ax.set_xlabel('SNV Density (SNVs per 10kb)', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_ylabel('Recall', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_title('Detection Performance vs Strain Divergence', 
                fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', color=COLOR_PALETTE['primary'])
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'divergence_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _generate_detection_roc_curve(
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    output_dir: str
):
    """Generate ROC-like detection curve (detection rate vs false positive rate)."""
    if not HAS_MATPLOTLIB or not true_haps:
        return
    
    matched_strain_ids = {m[0].strain_id for m in matches}
    matched_lineage_ids = {m[1].lineage_id for m in matches}
    
    # Compute true positives, false positives, false negatives
    tp = len(matched_strain_ids)
    fp = len(detected_haps) - len(matched_lineage_ids)
    fn = len(true_haps) - len(matched_strain_ids)
    tn = 0  # Not applicable for this context
    
    # Detection rate (recall/TPR)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else fp / len(detected_haps) if detected_haps else 0.0
    
    # For a full ROC curve, we'd vary thresholds, but here we show the operating point
    # and add a diagonal reference line
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], '--', color=COLOR_PALETTE['neutral'], 
           linewidth=2.0, alpha=0.7, label='Random classifier')
    
    # Plot operating point
    ax.scatter([fpr], [tpr], s=200, color=COLOR_PALETTE['success'],
              edgecolors=COLOR_PALETTE['primary'], linewidths=2.5, zorder=5,
              label=f'Strainphase (TPR={tpr:.3f}, FPR={fpr:.3f})')
    
    ax.set_xlabel('False Positive Rate', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_ylabel('True Positive Rate (Recall)', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_title('Detection Performance (ROC-like)', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_roc.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _generate_reference_coverage(
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    output_dir: str
):
    """Generate reference coverage (completeness) plot."""
    if not HAS_MATPLOTLIB or not matches:
        return
    
    # Compute coverage per matched strain (fraction of true SNVs recovered)
    coverages = []
    strain_ids = []
    
    for true_hap, det_hap, _ in matches:
        total_true_snvs = sum(len(snvs) for snvs in true_hap.snv_positions.values())
        if total_true_snvs == 0:
            continue
        
        recovered_snvs = 0
        for contig, true_snvs in true_hap.snv_positions.items():
            det_snvs = det_hap.snv_alleles.get(contig, {})
            for pos, true_allele in true_snvs.items():
                if pos in det_snvs and det_snvs[pos] == true_allele:
                    recovered_snvs += 1
        
        coverage = recovered_snvs / total_true_snvs if total_true_snvs > 0 else 0.0
        coverages.append(coverage)
        strain_ids.append(true_hap.strain_id[:15])
    
    if not coverages:
        return
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1.0, 21)
    
    ax.hist(coverages, bins=bins, color=COLOR_PALETTE['accent'],
           edgecolor=COLOR_PALETTE['primary'], linewidth=1.2, alpha=0.85)
    
    # Add mean line
    mean_cov = np.mean(coverages)
    ax.axvline(mean_cov, color=COLOR_PALETTE['error'], linestyle='--',
              linewidth=2.5, label=f'Mean: {mean_cov:.3f}', alpha=0.8)
    
    ax.set_xlabel('Reference Coverage (Fraction of SNVs Recovered)', 
                 fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_ylabel('Number of Strains', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_title('Haplotype Reference Coverage Distribution',
                fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reference_coverage.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _generate_error_breakdown(
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    output_dir: str
):
    """Generate error type breakdown (false positives/negatives by category)."""
    if not HAS_MATPLOTLIB:
        return
    
    matched_strain_ids = {m[0].strain_id for m in matches}
    matched_lineage_ids = {m[1].lineage_id for m in matches}
    
    # False negatives: true strains not detected
    fn_strains = [h for h in true_haps if h.strain_id not in matched_strain_ids]
    
    # False positives: detected lineages not matching truth
    fp_lineages = [h for h in detected_haps if h.lineage_id not in matched_lineage_ids]
    
    # Categorize false negatives by abundance
    fn_low_abund = 0
    fn_med_abund = 0
    fn_high_abund = 0
    
    for h in fn_strains:
        if h.abundances:
            max_abund = max(h.abundances.values())
            if max_abund < 0.01:
                fn_low_abund += 1
            elif 0.01 <= max_abund < 0.10:
                fn_med_abund += 1
            else:
                fn_high_abund += 1
        else:
            fn_low_abund += 1  # No abundance data = treat as low
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # False negatives breakdown
    fn_categories = ['Low (<1%)', 'Medium (1-10%)', 'High (≥10%)']
    fn_counts = [fn_low_abund, fn_med_abund, fn_high_abund]
    colors_fn = [COLOR_PALETTE['error'], COLOR_PALETTE['warning'], COLOR_PALETTE['success']]
    
    bars1 = ax1.bar(fn_categories, fn_counts, color=colors_fn,
                   edgecolor=COLOR_PALETTE['primary'], linewidth=1.5, alpha=0.85)
    ax1.set_ylabel('Count', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax1.set_title('False Negatives by Abundance', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax1.set_xticklabels(fn_categories, color=COLOR_PALETTE['primary'])
    
    for bar, count in zip(bars1, fn_counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', fontsize=11, fontweight='bold',
                    color=COLOR_PALETTE['primary'])
    
    # False positives count
    fp_total = len(fp_lineages)
    ax2.bar(['False Positives'], [fp_total], color=COLOR_PALETTE['error'],
           edgecolor=COLOR_PALETTE['primary'], linewidth=1.5, alpha=0.85)
    ax2.set_ylabel('Count', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax2.set_title('False Positives', fontweight='bold', color=COLOR_PALETTE['primary'])
    
    if fp_total > 0:
        ax2.text(0, fp_total + 0.1, str(fp_total), ha='center', fontsize=11,
                fontweight='bold', color=COLOR_PALETTE['primary'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_breakdown.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _generate_scalability_analysis(
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    output_dir: str
):
    """Generate performance vs strain count (scalability analysis)."""
    if not HAS_MATPLOTLIB:
        return
    
    # This is a placeholder - in real benchmarking, we'd have multiple scenarios
    # with different strain counts. For now, we show the current scenario.
    n_strains = len(true_haps)
    n_detected = len(detected_haps)
    precision = len({m[1].lineage_id for m in matches}) / n_detected if n_detected > 0 else 0.0
    recall = len({m[0].strain_id for m in matches}) / n_strains if n_strains > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Plot single point (in full benchmark, this would show multiple scenarios)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter([n_strains], [f1], s=300, color=COLOR_PALETTE['success'],
              edgecolors=COLOR_PALETTE['primary'], linewidths=2.5, zorder=5,
              label=f'Current (n={n_strains}, F1={f1:.3f})')
    
    ax.set_xlabel('Number of True Strains', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_ylabel('F1 Score', fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_title('Scalability: Performance vs Strain Count',
                fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.set_xlim(0, max(n_strains * 1.2, 10))
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main validation pipeline
# =============================================================================

def run_validation(
    detected_file: str,
    truth_dir: str,
    output_dir: str,
    window_results: Optional[List] = None,  # Optional WindowResult list for track validation
    window_size: Optional[int] = None,  # Window size for track validation
    detected_without_rescue: Optional[Dict] = None  # Optional abundances without rescue for Δrecall
) -> ValidationResult:
    """Run the full validation pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    true_haps, all_snv_positions = load_ground_truth(truth_dir)
    detected_haps = load_detected_haplotypes(detected_file)

    # Compute basic metrics
    result = compute_validation_metrics(true_haps, detected_haps, all_snv_positions)

    # Match for figures and track validation
    matches = match_haplotypes(true_haps, detected_haps, allow_one_to_many=True)
    
    # Build strain matches for track validation: detected_track_id -> true_strain_id
    strain_matches = {}
    for true_hap, det_hap, _ in matches:
        # Use lineage_id as track_id (they should be equivalent)
        if det_hap.lineage_id:
            strain_matches[det_hap.lineage_id] = true_hap.strain_id
    
    # Build truth SNVs: strain_id -> {contig -> {pos -> allele}}
    truth_snvs = {}
    for true_hap in true_haps:
        truth_snvs[true_hap.strain_id] = true_hap.snv_positions

    # Track/linking validation (if window_results provided)
    if window_results and window_size:
        try:
            from validation.validate_tracks import validate_tracks
            track_result = validate_tracks(
                window_results, truth_dir, strain_matches, truth_snvs, window_size
            )
            result.track_fragmentation_mean = track_result.track_fragmentation_mean
            result.track_fragmentation_median = track_result.track_fragmentation_median
            result.false_link_rate = track_result.false_link_rate
            result.missed_link_rate = track_result.missed_link_rate
            result.track_consensus_error = track_result.track_consensus_error
        except Exception as e:
            logger.warning(f"Track validation failed: {e}")

    # Lineage validation
    try:
        from validation.validate_lineages import validate_lineages
        
        # Build detected lineages: lineage_id -> {contig -> strain_id}
        detected_lineages = {}
        for det_hap in detected_haps:
            if det_hap.lineage_id:
                if det_hap.lineage_id not in detected_lineages:
                    detected_lineages[det_hap.lineage_id] = {}
                # Map contigs from SNV alleles
                for contig in det_hap.snv_alleles.keys():
                    # Find matching true strain
                    matching_strain = None
                    for true_hap, det_match, _ in matches:
                        if det_match.lineage_id == det_hap.lineage_id and contig in true_hap.snv_positions:
                            matching_strain = true_hap.strain_id
                            break
                    if matching_strain:
                        detected_lineages[det_hap.lineage_id][contig] = matching_strain
        
        # Build abundance dictionaries
        true_abundances = {h.strain_id: h.abundances for h in true_haps}
        detected_abundances = {h.lineage_id: h.abundances for h in detected_haps if h.lineage_id}
        
        lineage_result = validate_lineages(
            detected_lineages, truth_dir, true_abundances, detected_abundances,
            detected_without_rescue=detected_without_rescue
        )
        result.lineage_precision = lineage_result.lineage_precision
        result.lineage_recall = lineage_result.lineage_recall
        result.lineage_f1 = lineage_result.lineage_f1
        result.rescue_delta_recall_rare = lineage_result.rescue_delta_recall_rare
        result.abundance_trajectory_error = lineage_result.abundance_trajectory_error
    except Exception as e:
        logger.warning(f"Lineage validation failed: {e}")

    # Generate figures
    generate_figures(
        result, true_haps, detected_haps, matches, output_dir,
        window_results=window_results,
        truth_dir=truth_dir
    )

    # Save metrics
    metrics_file = os.path.join(output_dir, 'validation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'n_true': result.n_true,
            'n_detected': result.n_detected,
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
            # Track/linking metrics
            'track_fragmentation_mean': result.track_fragmentation_mean,
            'track_fragmentation_median': result.track_fragmentation_median,
            'false_link_rate': result.false_link_rate,
            'missed_link_rate': result.missed_link_rate,
            'track_consensus_error': result.track_consensus_error,
            # Lineage metrics
            'lineage_precision': result.lineage_precision,
            'lineage_recall': result.lineage_recall,
            'lineage_f1': result.lineage_f1,
            'rescue_delta_recall_rare': result.rescue_delta_recall_rare,
            'abundance_trajectory_error': result.abundance_trajectory_error,
            # Detailed diagnostics
            'false_negatives': result.false_negatives,
            'false_positives': result.false_positives,
            'per_contig_metrics': result.per_contig_metrics,
            'per_timepoint_metrics': result.per_timepoint_metrics,
        }, f, indent=2)
    
    # Generate detailed text report
    report_file = os.path.join(output_dir, 'detailed_report.txt')
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary metrics
        f.write("SUMMARY METRICS\n")
        f.write("-" * 80 + "\n")
        n_contigs = len(result.per_contig_metrics) if result.per_contig_metrics else 1
        all_timepoints_list = sorted(result.per_timepoint_metrics.keys()) if result.per_timepoint_metrics else []
        n_timepoints = len(all_timepoints_list)
        total_contig_timepoint_pairs = n_contigs * n_timepoints if n_timepoints > 0 else n_contigs
        
        f.write(f"True strains (per genome):     {result.n_true}\n")
        f.write(f"Contigs evaluated:            {n_contigs}\n")
        f.write(f"Timepoints evaluated:         {n_timepoints}\n")
        f.write(f"Total contig-timepoint pairs: {total_contig_timepoint_pairs}\n")
        f.write(f"Detected lineages (total):    {result.n_detected}\n")
        f.write(f"Matched lineages:             {result.n_matched}\n")
        f.write(f"Matched true strains:         {len(matched_true_ids)}\n")
        f.write(f"Matched detected lineages:    {len(matched_detected_ids)}\n")
        f.write("\n(Note: Strainphase splits lineages per-contig, so one strain can produce multiple detected lineages per contig)\n")
        f.write(f"Precision:           {result.precision:.3f}\n")
        f.write(f"Recall:              {result.recall:.3f}\n")
        f.write(f"F1 Score:            {result.f1:.3f}\n")
        f.write(f"Abundance Pearson r: {result.abundance_pearson_r:.3f}\n")
        f.write(f"Abundance MAE:       {result.abundance_mae:.3f}\n")
        f.write(f"SNV Precision:       {result.snv_precision:.3f}\n")
        f.write(f"SNV Recall:          {result.snv_recall:.3f}\n")
        f.write(f"Detection Threshold: {result.detection_threshold:.4f}\n")
        f.write("\n")
        
        # False negatives
        f.write("FALSE NEGATIVES (True haplotypes not detected)\n")
        f.write("-" * 80 + "\n")
        if result.false_negatives:
            for fn_id in result.false_negatives:
                fn_hap = next((h for h in true_haps if h.strain_id == fn_id), None)
                if fn_hap:
                    max_abund = max(fn_hap.abundances.values()) if fn_hap.abundances else 0
                    n_snvs = sum(len(snvs) for snvs in fn_hap.snv_positions.values())
                    f.write(f"  {fn_id}: max_abundance={max_abund:.4f}, n_snvs={n_snvs}, "
                           f"sweeping={fn_hap.is_sweeping}\n")
        else:
            f.write("  None\n")
        f.write("\n")
        
        # False positives with diagnostic information
        f.write("FALSE POSITIVES (Detected lineages not matching truth)\n")
        f.write("-" * 80 + "\n")
        if result.false_positives:
            for fp_id in result.false_positives:
                fp_hap = next((h for h in detected_haps if h.lineage_id == fp_id), None)
                if fp_hap:
                    max_abund = max(fp_hap.abundances.values()) if fp_hap.abundances else 0
                    n_snvs = sum(len(snvs) for snvs in fp_hap.snv_alleles.values())
                    contigs = list(fp_hap.snv_alleles.keys())
                    f.write(f"\n  {fp_id}: max_abundance={max_abund:.4f}, n_snvs={n_snvs}, contigs={contigs}\n")
                    
                    # Show why it doesn't match each true strain
                    f.write(f"    Why it doesn't match:\n")
                    for true_hap in true_haps:
                        dist, n_matches, n_shared, match_fraction = compute_haplotype_distance(true_hap, fp_hap)
                        n_mismatches = n_shared - n_matches
                        common_tps = set(true_hap.abundances.keys()) & set(fp_hap.abundances.keys())
                        abundance_ok = _abundance_within_factor(true_hap, fp_hap, factor=2.0) if common_tps else False
                        
                        reasons = []
                        if n_shared < 3:
                            reasons.append(f"too few shared SNVs ({n_shared} < 3)")
                        if match_fraction < 0.9:
                            reasons.append(f"low match fraction ({match_fraction:.3f} < 0.9)")
                        if dist > 0.1:
                            reasons.append(f"distance too high ({dist:.3f} > 0.1)")
                        if not abundance_ok:
                            true_abund = true_hap.abundances.get(list(common_tps)[0] if common_tps else '', 0)
                            det_abund = fp_hap.abundances.get(list(common_tps)[0] if common_tps else '', 0)
                            reasons.append(f"abundance mismatch (true={true_abund:.3f}, det={det_abund:.3f}, not within 2x)")
                        
                        f.write(f"      vs {true_hap.strain_id}: ")
                        if reasons:
                            f.write("; ".join(reasons))
                            f.write(f"; distance={dist:.3f}, shared={n_shared}, matches={n_matches}, mismatches={n_mismatches}")
                        else:
                            f.write(f"distance={dist:.3f}, shared={n_shared}, matches={n_matches}, mismatches={n_mismatches}, "
                                  f"match_frac={match_fraction:.3f}, abund_ok={abundance_ok}")
                        f.write("\n")
        else:
            f.write("  None\n")
        f.write("\n")
        
        # Detailed matches
        f.write("DETAILED MATCH INFORMATION\n")
        f.write("-" * 80 + "\n")
        for match in result.match_details_full:
            f.write(f"\nTrue Strain: {match['true_strain_id']}\n")
            f.write(f"  → Detected Lineage: {match['detected_lineage_id']}\n")
            f.write(f"  Distance: {match['distance']:.4f}\n")
            f.write(f"  SNVs: true={match['n_true_snvs']}, detected={match['n_detected_snvs']}, "
                   f"shared={match['n_shared_snvs']}, matching={match['n_matching_snvs']}\n")
            f.write(f"  SNV Match Fraction: {match['snv_match_fraction']:.3f}\n")
            if match['abundance_mae'] is not None:
                f.write(f"  Abundance MAE: {match['abundance_mae']:.4f}\n")
            f.write(f"  Timepoints: {', '.join(match['common_timepoints'])}\n")
            if match['true_abundances']:
                f.write(f"  True abundances: {match['true_abundances']}\n")
                f.write(f"  Detected abundances: {match['detected_abundances']}\n")
            f.write(f"  Is sweeping: {match['is_sweeping']}\n")
        f.write("\n")
        
        # Per-contig breakdown
        f.write("PER-CONTIG METRICS\n")
        f.write("-" * 80 + "\n")
        for contig, metrics in sorted(result.per_contig_metrics.items()):
            f.write(f"{contig}:\n")
            f.write(f"  True: {metrics['n_true']}, Detected: {metrics['n_detected']}, "
                   f"Matched: {metrics['n_matched']}\n")
            f.write(f"  Precision: {metrics['precision']:.3f}, "
                   f"Recall: {metrics['recall']:.3f}\n")
        f.write("\n")
        
        # Per-timepoint breakdown
        f.write("PER-TIMEPOINT METRICS\n")
        f.write("-" * 80 + "\n")
        for tp, metrics in sorted(result.per_timepoint_metrics.items()):
            f.write(f"{tp}:\n")
            f.write(f"  True: {metrics['n_true']}, Detected: {metrics['n_detected']}, "
                   f"Matched: {metrics['n_matched']}\n")
            f.write(f"  Precision: {metrics['precision']:.3f}, "
                   f"Recall: {metrics['recall']:.3f}\n")
            if metrics['abundance_pearson_r'] is not None:
                f.write(f"  Abundance r: {metrics['abundance_pearson_r']:.3f}, "
                       f"MAE: {metrics['abundance_mae']:.4f}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    # Summary with explicit denominators
    n_contigs = len(result.per_contig_metrics) if result.per_contig_metrics else 1
    all_timepoints_list = sorted(result.per_timepoint_metrics.keys()) if result.per_timepoint_metrics else []
    n_timepoints = len(all_timepoints_list)
    total_contig_timepoint_pairs = n_contigs * n_timepoints if n_timepoints > 0 else n_contigs
    
    print(f"True strains (per genome):     {result.n_true}")
    print(f"Contigs evaluated:            {n_contigs}")
    print(f"Timepoints evaluated:         {n_timepoints}")
    print(f"Total contig-timepoint pairs: {total_contig_timepoint_pairs}")
    print(f"Detected lineages (total):    {result.n_detected}")
    print(f"Matched lineages:             {result.n_matched}")
    print(f"Matched true strains:         {len(matched_true_ids)}")
    print(f"Matched detected lineages:    {len(matched_detected_ids)}")
    print("-" * 60)
    
    # Unified breakdown: timepoint → contig
    if result.per_timepoint_metrics and result.per_contig_metrics:
        print("\nBREAKDOWN BY TIMEPOINT → CONTIG:")
        print("-" * 60)
        for tp in sorted(result.per_timepoint_metrics.keys()):
            tp_metrics = result.per_timepoint_metrics[tp]
            print(f"\n{tp}:")
            print(f"  Overall: {tp_metrics['n_true']} true, {tp_metrics['n_detected']} detected, "
                  f"{tp_metrics['n_matched_true']} matched true, {tp_metrics['n_matched_detected']} matched detected")
            print(f"  Precision: {tp_metrics['precision']:.3f}, Recall: {tp_metrics['recall']:.3f}")
            
            # Show per-contig breakdown for this timepoint
            print(f"  Per-contig:")
            for contig, contig_metrics in sorted(result.per_contig_metrics.items()):
                contig_short = contig.split('.')[-1] if '.' in contig else contig
                # Check if this contig has data for this timepoint
                # (we can't easily check per-contig-per-timepoint, so show all contigs)
                print(f"    {contig_short}: {contig_metrics['n_true']} true, "
                      f"{contig_metrics['n_detected']} detected, "
                      f"{contig_metrics['n_matched_true']} matched true")
    
    # Fallback: if no timepoint metrics, show contig breakdown
    elif result.per_contig_metrics:
        print("\nBREAKDOWN BY CONTIG:")
        print("-" * 60)
        for contig, metrics in sorted(result.per_contig_metrics.items()):
            contig_short = contig.split('.')[-1] if '.' in contig else contig
            print(f"{contig_short}:")
            print(f"  True haplotypes:   {metrics['n_true']}")
            print(f"  Detected lineages: {metrics['n_detected']}")
            print(f"  Matched true:      {metrics['n_matched_true']}")
            print(f"  Matched detected:  {metrics['n_matched_detected']}")
            print(f"  Precision:         {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
    
    print("\nOVERALL METRICS:")
    print("-" * 60)
    print(f"Precision:           {result.precision:.3f}")
    print(f"Recall:              {result.recall:.3f}")
    print(f"F1 Score:            {result.f1:.3f}")
    print("-" * 60)
    print(f"\nABUNDANCE METRICS:")
    print("-" * 60)
    print(f"Abundance Pearson r: {result.abundance_pearson_r:.3f}")
    print(f"Abundance MAE:       {result.abundance_mae:.3f}")
    print(f"\nSNV METRICS:")
    print("-" * 60)
    print(f"SNV Precision:       {result.snv_precision:.3f}")
    print(f"SNV Recall:          {result.snv_recall:.3f}")
    print(f"Phasing Accuracy:    {result.phasing_accuracy:.3f}")
    print(f"Detection Threshold: {result.detection_threshold:.4f}")
    print("=" * 60)
    
    # Print error breakdown with diagnostics
    if result.false_negatives or result.false_positives:
        print("\nERROR BREAKDOWN:")
        print("-" * 60)
        if result.false_negatives:
            print(f"False Negatives ({len(result.false_negatives)} missing):")
            for fn in result.false_negatives[:5]:
                print(f"  - {fn}")
            if len(result.false_negatives) > 5:
                print(f"  ... and {len(result.false_negatives) - 5} more")
        if result.false_positives:
            print(f"False Positives ({len(result.false_positives)} spurious):")
            for fp_id in result.false_positives[:5]:
                fp_hap = next((h for h in detected_haps if h.lineage_id == fp_id), None)
                if fp_hap:
                    max_abund = max(fp_hap.abundances.values()) if fp_hap.abundances else 0
                    n_snvs = sum(len(snvs) for snvs in fp_hap.snv_alleles.values())
                    contigs = list(fp_hap.snv_alleles.keys())
                    print(f"  - {fp_id}: abund={max_abund:.3f}, snvs={n_snvs}, contigs={len(contigs)}")
                    
                    # Show closest match with detailed SNV breakdown
                    best_dist = 1.0
                    best_strain = None
                    best_n_matches = 0
                    best_n_shared = 0
                    for true_hap in true_haps:
                        dist, n_matches, n_shared, match_fraction = compute_haplotype_distance(true_hap, fp_hap)
                        if dist < best_dist:
                            best_dist = dist
                            best_strain = true_hap.strain_id
                            best_n_matches = n_matches
                            best_n_shared = n_shared
                    if best_strain:
                        n_mismatches = best_n_shared - best_n_matches
                        print(f"    Closest to {best_strain}: distance={best_dist:.3f}, "
                              f"shared_snvs={best_n_shared}, matches={best_n_matches}, mismatches={n_mismatches}")
            if len(result.false_positives) > 5:
                print(f"  ... and {len(result.false_positives) - 5} more")
        print("-" * 60)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Detailed report: {report_file}")
    print(f"  - Figures: {output_dir}/*.png")

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """
    Standalone CLI for validation (also called automatically by parameter_sweep.py).
    
    This allows running validation independently:
        python validation/validate_haplotypes.py \
            --detected results/lineages.tsv \
            --truth data/simulated/ \
            --output results/validation/
    """
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
