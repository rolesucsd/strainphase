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
    # Silence missing font warnings (e.g., Arial not found)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
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

    # SNV count diagnostics
    snv_true_total: int = 0
    snv_true_in_span: int = 0
    snv_detected_total: int = 0
    snv_correct_total: int = 0
    snv_span_coverage_frac: float = 0.0

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

# Color palette for scatter plot points
POINT_COLORS = ['#569667', '#4264a8', '#e6a432', '#8e4aa1']

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
    plt.rcParams['axes.titlesize'] = 12  # Title size 12, not bold
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
            # Abundance threshold removed - match based on SNV similarity only
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

    # Abundance correlation with grouped truth
    # 
    # KEY INSIGHT: If multiple true strains have identical sequences within the
    # detected region, they are INDISTINGUISHABLE and should appear as a single
    # haplotype with combined abundance. The "effective truth" for comparison
    # should be the sum of indistinguishable strains' abundances.
    #
    # Example: Strains A (25%) and B (25%) are identical in a window →
    #          Expected detection: one haplotype at 50%
    #          Effective truth: 50%, not 25%
    
    true_abundances = []
    detected_abundances = []
    grouped_true_abundances = []  # For the corrected metric

    for true_hap, det_hap, _ in matches:
        # Get the SNV positions where this detected haplotype has calls
        detected_positions: Dict[str, set] = {}
        for contig, snvs in det_hap.snv_alleles.items():
            detected_positions[contig] = set(snvs.keys())
        
        # Find all true strains that are INDISTINGUISHABLE from true_hap
        # within the detected positions (identical alleles at all shared positions)
        indistinguishable_strains = [true_hap]  # Always includes self
        
        for other_hap in true_haps:
            if other_hap.strain_id == true_hap.strain_id:
                continue
            
            # Check if other_hap is identical to true_hap at detected positions
            is_identical = True
            has_overlap = False
            
            for contig, det_positions in detected_positions.items():
                true_snvs = true_hap.snv_positions.get(contig, {})
                other_snvs = other_hap.snv_positions.get(contig, {})
                
                for pos in det_positions:
                    true_allele = true_snvs.get(pos)
                    other_allele = other_snvs.get(pos)
                    
                    if true_allele is not None and other_allele is not None:
                        has_overlap = True
                        if true_allele != other_allele:
                            is_identical = False
                            break
                
                if not is_identical:
                    break
            
            # Only consider as indistinguishable if there was actual overlap
            # and all overlapping positions matched
            if is_identical and has_overlap:
                indistinguishable_strains.append(other_hap)
        
        # Find common timepoints
        common_tps = set(true_hap.abundances.keys()) & set(det_hap.abundances.keys())
        
        for tp in common_tps:
            # Original individual abundance (for backward compatibility)
            true_abundances.append(true_hap.abundances[tp])
            detected_abundances.append(det_hap.abundances[tp])
            
            # Grouped abundance: sum of all indistinguishable strains
            grouped_abundance = sum(
                h.abundances.get(tp, 0) for h in indistinguishable_strains
            )
            grouped_true_abundances.append(grouped_abundance)
    
    if len(true_abundances) >= 2:
        # Use GROUPED abundances for the primary metric (more accurate)
        abundance_pearson_r = np.corrcoef(grouped_true_abundances, detected_abundances)[0, 1]
        abundance_mae = np.mean(np.abs(np.array(grouped_true_abundances) - np.array(detected_abundances)))
        
        # Log the difference for transparency
        old_mae = np.mean(np.abs(np.array(true_abundances) - np.array(detected_abundances)))
        if abs(old_mae - abundance_mae) > 0.01:
            logger.debug(f"Abundance MAE improved from {old_mae:.3f} (individual) to {abundance_mae:.3f} (grouped)")
    else:
        abundance_pearson_r = 0.0
        abundance_mae = 1.0

    # SNV accuracy (for matched haplotypes)
    # Aggregate SNVs per true strain to handle fragmentation correctly:
    # If a strain is split into multiple tracks, we should count each true SNV
    # only once, and consider it "recovered" if ANY matching track has it correct.
    #
    # IMPORTANT: Only count true SNVs within the detected genomic span.
    # If there are gaps between windows (e.g., sparse SNV regions), we can't detect
    # SNVs there, so they shouldn't penalize recall. This gives "SNV recall within
    # the regions we actually processed."

    # Group matches by true strain
    matches_by_strain: Dict[str, List[DetectedHaplotype]] = defaultdict(list)
    for true_hap, det_hap, _ in matches:
        matches_by_strain[true_hap.strain_id].append(det_hap)

    total_true_snvs_in_span = 0
    total_true_snvs_global = 0
    total_detected_snvs = 0
    total_correct_snvs = 0

    for true_hap in true_haps:
        # Count global true SNVs (for reference)
        for contig, true_snvs in true_hap.snv_positions.items():
            total_true_snvs_global += len(true_snvs)
        
        # Get all detected tracks matching this true strain
        matching_tracks = matches_by_strain.get(true_hap.strain_id, [])
        if not matching_tracks:
            # For unmatched strains, we have no detected span, so these SNVs
            # don't contribute to the "within-span" recall calculation
            continue

        # Aggregate detected SNVs across all matching tracks and determine span per contig
        # detected_snvs_union[contig][pos] = allele (from any matching track)
        detected_snvs_union: Dict[str, Dict[int, str]] = defaultdict(dict)
        detected_span: Dict[str, Tuple[int, int]] = {}  # contig -> (min_pos, max_pos)
        
        for det_hap in matching_tracks:
            for contig, det_snvs in det_hap.snv_alleles.items():
                for pos, allele in det_snvs.items():
                    # Keep first allele seen (they should all agree if tracks are correct)
                    if pos not in detected_snvs_union[contig]:
                        detected_snvs_union[contig][pos] = allele
                
                # Update span for this contig
                if det_snvs:
                    min_pos = min(det_snvs.keys())
                    max_pos = max(det_snvs.keys())
                    if contig in detected_span:
                        curr_min, curr_max = detected_span[contig]
                        detected_span[contig] = (min(curr_min, min_pos), max(curr_max, max_pos))
                    else:
                        detected_span[contig] = (min_pos, max_pos)

        # Count true SNVs WITHIN detected span and check if they're recovered
        for contig, true_snvs in true_hap.snv_positions.items():
            det_snvs = detected_snvs_union.get(contig, {})
            span = detected_span.get(contig)
            
            if not span:
                # No detected SNVs on this contig for this strain - skip
                continue
            
            span_min, span_max = span
            
            for pos, true_allele in true_snvs.items():
                # Only count SNVs within the detected span
                if span_min <= pos <= span_max:
                    total_true_snvs_in_span += 1
                    if pos in det_snvs and det_snvs[pos] == true_allele:
                        total_correct_snvs += 1

        # Count total detected SNVs (union across all matching tracks, avoid double-counting)
        for contig, det_snvs in detected_snvs_union.items():
            total_detected_snvs += len(det_snvs)

    snv_precision = total_correct_snvs / total_detected_snvs if total_detected_snvs > 0 else 0.0
    snv_recall = total_correct_snvs / total_true_snvs_in_span if total_true_snvs_in_span > 0 else 0.0
    phasing_accuracy = snv_recall
    
    coverage_fraction = 0.0
    # Log the difference between global and within-span counts for transparency
    if total_true_snvs_global > 0:
        coverage_fraction = total_true_snvs_in_span / total_true_snvs_global
        logger.debug(f"SNV recall computed within detected span: {total_true_snvs_in_span}/{total_true_snvs_global} "
                    f"true SNVs ({coverage_fraction:.1%} of total) in detected regions")

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
        snv_true_total=total_true_snvs_global,
        snv_true_in_span=total_true_snvs_in_span,
        snv_detected_total=total_detected_snvs,
        snv_correct_total=total_correct_snvs,
        snv_span_coverage_frac=coverage_fraction if total_true_snvs_global > 0 else 0.0,
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

    # Figure 1: Accuracy bar plot (removed; now generated in report summary)

    # Figure 2: Abundance correlation (removed; now generated in report patchwork)

    # Figure 3: Detection sensitivity - REMOVED per user request

    # Figure 5: Per-haplotype matching details (removed; now generated in report patchwork)

    # Figure 6: Abundance trajectories - REMOVED per user request

    # Figure 7: Track fragmentation - REMOVED per user request

    # Figure 8: Linking errors (removed; now generated in report summary)

    # Figure 9: Lineage accuracy (removed; now generated in report summary)

    # Figure 10: Track visualization on contigs (removed; now generated in report patchwork)

    # Figure 11: Per-abundance-bin performance (removed by request)
    # Figure 12: Strain divergence vs performance (removed by request)
    
    # Figure 13: ROC-like detection curve - MOVED to generate once per full run
    # _generate_detection_roc_curve(true_haps, detected_haps, matches, output_dir)
    
    # Figure 14: Reference coverage (removed; now generated in report patchwork)
    
    # Figure 15: Error type breakdown (removed; now generated in report summary)
    
    # Figure 16: Performance vs strain count (scalability) - REMOVED per user request


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
        
        # Count detected strains in this bin (true positives)
        detected_in_bin = sum(1 for h in strains_in_bin if h.strain_id in matched_strain_ids)
        
        # For precision: count detected lineages that match strains in this bin (TP)
        # vs all detected lineages that match strains in this bin (TP + FP within bin)
        matched_lineage_ids_in_bin = set()
        for true_hap, det_hap, _ in matches:
            if true_hap.strain_id in {h.strain_id for h in strains_in_bin}:
                matched_lineage_ids_in_bin.add(det_hap.lineage_id)
        
        # Count detected lineages in this bin (those matching strains in bin)
        # This is TP for this bin
        tp_in_bin = len(matched_lineage_ids_in_bin)
        
        # Count all detected lineages that could be in this bin (check their abundance)
        # This includes both TP and FP
        all_detected_lineage_ids_in_bin = set()
        for det_hap in detected_haps:
            # Check if this detected haplotype has abundance in the bin range
            max_det_abund = max(det_hap.abundances.values()) if det_hap.abundances else 0.0
            if low <= max_det_abund < high:
                all_detected_lineage_ids_in_bin.add(det_hap.lineage_id)
        
        # Precision: TP / (TP + FP) = matched lineages in bin / all detected lineages in bin
        precision = tp_in_bin / len(all_detected_lineage_ids_in_bin) if len(all_detected_lineage_ids_in_bin) > 0 else 0.0
        # Recall: TP / (TP + FN) = detected strains / total strains in bin
        recall = detected_in_bin / len(strains_in_bin) if len(strains_in_bin) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        bin_precision.append(precision)
        bin_recall.append(recall)
        bin_f1.append(f1)
        bin_counts.append(len(strains_in_bin))
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bin_labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, bin_precision, width, label='Precision', 
                   color=POINT_COLORS[0], alpha=0.7)
    bars2 = ax.bar(x, bin_recall, width, label='Recall',
                   color=POINT_COLORS[1], alpha=0.7)
    bars3 = ax.bar(x + width, bin_f1, width, label='F1',
                   color=POINT_COLORS[2], alpha=0.7)
    
    # Add count annotations
    for i, (bar, count) in enumerate(zip(bars2, bin_counts)):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, -0.05,
                   f'n={count}', ha='center', va='top', fontsize=9,
                   color=COLOR_PALETTE['secondary'], style='italic')
    
    ax.set_xlabel('Abundance Range', color=COLOR_PALETTE['primary'])
    ax.set_ylabel('Score', color=COLOR_PALETTE['primary'])
    ax.set_title('Performance by Abundance Range', fontsize=12, color=COLOR_PALETTE['primary'])
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
        if total_snvs == 0:
            continue  # Skip strains with no SNVs
        
        # Compute total span across all contigs
        # Use max position - min position for each contig, sum them
        total_span = 0
        for contig, snvs in true_hap.snv_positions.items():
            if snvs:
                contig_span = max(snvs.keys()) - min(snvs.keys())
                # If span is 0 (single SNV), use a minimum span of 1000bp
                total_span += max(contig_span, 1000)
        
        # If no valid span, skip this strain
        if total_span == 0:
            continue
        
        # SNVs per 10kb
        snv_density = (total_snvs / total_span) * 10000
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
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bin_labels))
    
    bars = ax.bar(x, bin_recall, color=POINT_COLORS[1], alpha=0.7)
    
    # Add count annotations
    for bar, count in zip(bars, bin_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'n={count}', ha='center', fontsize=9,
                   color=COLOR_PALETTE['secondary'], style='italic')
    
    ax.set_xlabel('SNV Density (SNVs per 10kb)', color=COLOR_PALETTE['primary'])
    ax.set_ylabel('Recall', color=COLOR_PALETTE['primary'])
    ax.set_title('Detection Performance vs Strain Divergence', 
                fontsize=12, color=COLOR_PALETTE['primary'])
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
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], '--', color=COLOR_PALETTE['neutral'], 
           linewidth=2.0, alpha=0.7, label='Random classifier')
    
    # Plot operating point
    ax.scatter([fpr], [tpr], s=200, color=POINT_COLORS[2], alpha=0.7,
              edgecolors=COLOR_PALETTE['primary'], linewidths=2.5, zorder=5,
              label=f'Strainphase (TPR={tpr:.3f}, FPR={fpr:.3f})')
    
    ax.set_xlabel('False Positive Rate', color=COLOR_PALETTE['primary'])
    ax.set_ylabel('True Positive Rate (Recall)', color=COLOR_PALETTE['primary'])
    ax.set_title('Detection Performance (ROC-like)', fontsize=12, color=COLOR_PALETTE['primary'])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, framealpha=0.95, edgecolor=COLOR_PALETTE['neutral'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_roc.png'), dpi=300, bbox_inches='tight')
    plt.close()


# _generate_scalability_analysis - REMOVED per user request


# =============================================================================
# Detailed Output Files
# =============================================================================

def write_lineage_details(
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    output_dir: str
) -> str:
    """
    Write lineage_details.tsv - per-lineage raw data table.

    Long-format table with one row per lineage-contig-timepoint combination,
    exposing raw data behind computed metrics.
    """
    import csv

    output_path = os.path.join(output_dir, 'lineage_details.tsv')

    # Build match lookup: detected_lineage_id -> (true_hap, distance)
    match_lookup = {}
    for true_hap, det_hap, dist in matches:
        if det_hap.lineage_id not in match_lookup:
            match_lookup[det_hap.lineage_id] = (true_hap, dist)

    records = []

    for det_hap in detected_haps:
        lineage_id = det_hap.lineage_id
        matched_true_hap, dist = match_lookup.get(lineage_id, (None, None))
        matched_strain = matched_true_hap.strain_id if matched_true_hap else "UNMATCHED"

        # Process each contig
        for contig, det_snvs in det_hap.snv_alleles.items():
            if not det_snvs:
                continue

            # Compute SNV positions
            det_positions = sorted(det_snvs.keys())
            start_pos = min(det_positions) if det_positions else 0
            end_pos = max(det_positions) if det_positions else 0
            n_snvs_detected = len(det_positions)

            # Get true SNVs for this contig if matched
            n_snvs_true = 0
            n_shared_snvs = 0
            n_matching_snvs = 0
            n_different_snvs = 0
            snv_distance = 1.0

            if matched_true_hap and contig in matched_true_hap.snv_positions:
                true_snvs = matched_true_hap.snv_positions[contig]
                n_snvs_true = len(true_snvs)

                # Compute overlap statistics
                shared_positions = set(det_positions) & set(true_snvs.keys())
                n_shared_snvs = len(shared_positions)

                for pos in shared_positions:
                    if det_snvs[pos] == true_snvs[pos]:
                        n_matching_snvs += 1
                    else:
                        n_different_snvs += 1

                if n_shared_snvs > 0:
                    snv_distance = 1.0 - (n_matching_snvs / n_shared_snvs)

            # Process each timepoint
            for timepoint, det_abund in det_hap.abundances.items():
                true_abund = 0.0
                abundance_diff = det_abund

                if matched_true_hap and timepoint in matched_true_hap.abundances:
                    true_abund = matched_true_hap.abundances[timepoint]
                    abundance_diff = abs(det_abund - true_abund)

                records.append({
                    'lineage_id': lineage_id,
                    'matched_strain': matched_strain,
                    'timepoint': timepoint,
                    'contig': contig,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'n_snvs_detected': n_snvs_detected,
                    'n_snvs_true': n_snvs_true,
                    'n_shared_snvs': n_shared_snvs,
                    'n_matching_snvs': n_matching_snvs,
                    'n_different_snvs': n_different_snvs,
                    'snv_distance': f"{snv_distance:.6f}",
                    'detected_abundance': f"{det_abund:.6f}",
                    'true_abundance': f"{true_abund:.6f}",
                    'abundance_diff': f"{abundance_diff:.6f}",
                    'track_id': det_hap.track_id or lineage_id,
                })

    # Write TSV
    if records:
        fieldnames = [
            'lineage_id', 'matched_strain', 'timepoint', 'contig',
            'start_pos', 'end_pos', 'n_snvs_detected', 'n_snvs_true',
            'n_shared_snvs', 'n_matching_snvs', 'n_different_snvs',
            'snv_distance', 'detected_abundance', 'true_abundance',
            'abundance_diff', 'track_id'
        ]
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(records)
    else:
        # Write empty file with headers
        with open(output_path, 'w') as f:
            f.write('\t'.join([
                'lineage_id', 'matched_strain', 'timepoint', 'contig',
                'start_pos', 'end_pos', 'n_snvs_detected', 'n_snvs_true',
                'n_shared_snvs', 'n_matching_snvs', 'n_different_snvs',
                'snv_distance', 'detected_abundance', 'true_abundance',
                'abundance_diff', 'track_id'
            ]) + '\n')

    logger.info(f"Wrote {len(records)} lineage detail records to {output_path}")
    return output_path


def write_em_convergence(
    window_results: List,  # List of WindowResult
    output_dir: str
) -> str:
    """
    Write em_convergence.tsv - per-window EM performance metrics.

    Columns include convergence status, iterations, log-likelihood,
    and junk component statistics.
    """
    import csv

    output_path = os.path.join(output_dir, 'em_convergence.tsv')

    records = []

    for wr in window_results:
        window = wr.window
        n_reads = len(window.reads)
        n_haplotypes = len(wr.haplotypes)

        # Compute junk weight (last component in pi)
        junk_weight = 0.0
        if wr.pi is not None and len(wr.pi) > 0:
            junk_weight = float(wr.pi[-1])  # Last component is junk

        n_discarded_reads = int(junk_weight * n_reads)

        # Compute mean confidence from gamma
        mean_confidence = 0.0
        if wr.gamma is not None and wr.gamma.size > 0:
            # Mean of max assignment probabilities (excluding junk)
            if n_haplotypes > 0:
                hap_probs = wr.gamma[:, :n_haplotypes]
                if hap_probs.size > 0:
                    max_probs = np.max(hap_probs, axis=1)
                    # Only count reads assigned to haplotypes (not junk)
                    assigned_mask = np.argmax(wr.gamma, axis=1) < n_haplotypes
                    if np.any(assigned_mask):
                        mean_confidence = float(np.mean(max_probs[assigned_mask]))

        records.append({
            'sample': window.sample or 'unknown',
            'contig': window.contig,
            'window_start': window.start,
            'window_end': window.end,
            'n_reads': n_reads,
            'n_haplotypes': n_haplotypes,
            'converged': wr.converged,
            'iterations': wr.iterations,
            'log_likelihood': f"{wr.log_likelihood:.4f}" if not np.isinf(wr.log_likelihood) else "NA",
            'junk_weight': f"{junk_weight:.6f}",
            'n_discarded_reads': n_discarded_reads,
            'mean_confidence': f"{mean_confidence:.6f}",
        })

    # Write TSV
    if records:
        fieldnames = [
            'sample', 'contig', 'window_start', 'window_end',
            'n_reads', 'n_haplotypes', 'converged', 'iterations',
            'log_likelihood', 'junk_weight', 'n_discarded_reads', 'mean_confidence'
        ]
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(records)
    else:
        # Write empty file with headers
        with open(output_path, 'w') as f:
            f.write('\t'.join([
                'sample', 'contig', 'window_start', 'window_end',
                'n_reads', 'n_haplotypes', 'converged', 'iterations',
                'log_likelihood', 'junk_weight', 'n_discarded_reads', 'mean_confidence'
            ]) + '\n')

    logger.info(f"Wrote {len(records)} EM convergence records to {output_path}")
    return output_path


def write_linking_quality(
    detected_haps: List[DetectedHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    output_dir: str
) -> str:
    """
    Write linking_quality.tsv - cross-timepoint linking analysis.

    Analyzes how tracks/lineages link across timepoints and their
    consistency.
    """
    import csv

    output_path = os.path.join(output_dir, 'linking_quality.tsv')

    records = []

    # Group detected haplotypes by lineage_id
    lineage_groups: Dict[str, List[DetectedHaplotype]] = defaultdict(list)
    for det_hap in detected_haps:
        if det_hap.lineage_id:
            lineage_groups[det_hap.lineage_id].append(det_hap)

    # Build match lookup
    match_lookup = {}
    for true_hap, det_hap, dist in matches:
        if det_hap.lineage_id:
            match_lookup[det_hap.lineage_id] = true_hap.strain_id

    for lineage_id, group in lineage_groups.items():
        # Collect all timepoints and track_ids for this lineage
        timepoints = set()
        track_ids = set()
        abundances_by_tp = {}

        for det_hap in group:
            timepoints.update(det_hap.abundances.keys())
            track_ids.add(det_hap.track_id or det_hap.lineage_id)
            for tp, abund in det_hap.abundances.items():
                abundances_by_tp[tp] = abund

        n_timepoints = len(timepoints)
        n_tracks = len(track_ids)

        # Check linking consistency (all should map to same true strain)
        matched_strains = set()
        if lineage_id in match_lookup:
            matched_strains.add(match_lookup[lineage_id])
        linking_consistent = len(matched_strains) <= 1

        # Compute abundance trajectory
        sorted_tps = sorted(timepoints)
        abundance_trajectory = [abundances_by_tp.get(tp, 0.0) for tp in sorted_tps]

        # Compute trajectory smoothness (std dev of abundance changes)
        trajectory_smoothness = 0.0
        if len(abundance_trajectory) > 1:
            changes = [abs(abundance_trajectory[i+1] - abundance_trajectory[i])
                      for i in range(len(abundance_trajectory)-1)]
            trajectory_smoothness = float(np.std(changes)) if changes else 0.0

        # Track distance metrics (within lineage)
        min_track_distance = 0.0
        max_track_distance = 0.0
        # For now, we use 0 since we don't have track-level distance info here
        # This would require access to consensus sequences

        records.append({
            'lineage_id': lineage_id,
            'n_timepoints': n_timepoints,
            'n_tracks': n_tracks,
            'linking_consistent': linking_consistent,
            'min_track_distance': f"{min_track_distance:.6f}",
            'max_track_distance': f"{max_track_distance:.6f}",
            'track_ids': ','.join(sorted(track_ids)),
            'timepoints': ','.join(sorted_tps),
            'abundance_trajectory': ','.join(f"{a:.4f}" for a in abundance_trajectory),
            'trajectory_smoothness': f"{trajectory_smoothness:.6f}",
        })

    # Write TSV
    if records:
        fieldnames = [
            'lineage_id', 'n_timepoints', 'n_tracks', 'linking_consistent',
            'min_track_distance', 'max_track_distance', 'track_ids',
            'timepoints', 'abundance_trajectory', 'trajectory_smoothness'
        ]
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(records)
    else:
        # Write empty file with headers
        with open(output_path, 'w') as f:
            f.write('\t'.join([
                'lineage_id', 'n_timepoints', 'n_tracks', 'linking_consistent',
                'min_track_distance', 'max_track_distance', 'track_ids',
                'timepoints', 'abundance_trajectory', 'trajectory_smoothness'
            ]) + '\n')

    logger.info(f"Wrote {len(records)} linking quality records to {output_path}")
    return output_path


def write_rescue_statistics(
    window_results: List,  # List of WindowResult
    output_dir: str
) -> Optional[str]:
    """
    Write rescue_statistics.tsv from window results.

    Checks if rescue statistics are available via the config's _rescue_integrator
    attribute (set by process_mag_longitudinal).

    Returns the output path if written, None if no rescue stats available.
    """
    import csv

    if not window_results:
        return None

    # Try to get rescue integrator from the first window result's config
    # This is a bit indirect but avoids changing function signatures
    rescue_stats = []

    # Check if any window result has rescue statistics attached
    # The integrator is stored on config._rescue_integrator
    for wr in window_results:
        if hasattr(wr, 'window') and hasattr(wr.window, 'sample'):
            # Try to find integrator - it may be passed via config
            break

    # If no rescue statistics from integrator, create empty file
    output_path = os.path.join(output_dir, 'rescue_statistics.tsv')

    fieldnames = [
        'sample', 'contig', 'window_start', 'track_id',
        'was_rescued', 'original_weight', 'rescued_weight',
        'donor_timepoint', 'anchor_distance', 'n_shared_with_anchor'
    ]

    # Write empty file with headers (rescue stats will be written by parameter_sweep
    # which has access to the integrator)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for stat in rescue_stats:
            writer.writerow({
                'sample': stat.sample,
                'contig': stat.contig,
                'window_start': stat.window_start,
                'track_id': stat.track_id,
                'was_rescued': stat.was_rescued,
                'original_weight': f"{stat.original_weight:.6f}",
                'rescued_weight': f"{stat.rescued_weight:.6f}",
                'donor_timepoint': stat.donor_timepoint,
                'anchor_distance': f"{stat.anchor_distance:.6f}" if stat.anchor_distance >= 0 else "NA",
                'n_shared_with_anchor': stat.n_shared_with_anchor,
            })

    logger.info(f"Wrote {len(rescue_stats)} rescue statistics records to {output_path}")
    return output_path


def write_validation_summary(
    result: ValidationResult,
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    output_dir: str,
    window_results: Optional[List] = None
) -> str:
    """
    Write validation_summary.txt - enhanced human-readable summary.

    Consolidates data from all output files with explanations.
    """
    output_path = os.path.join(output_dir, 'validation_summary.txt')

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VALIDATION SUMMARY\n")
        f.write("Comprehensive analysis of strainphase haplotype reconstruction\n")
        f.write("=" * 80 + "\n\n")

        # Section 1: Overview
        f.write("1. OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"True strains in reference:  {result.n_true}\n")
        f.write(f"Detected lineages:          {result.n_detected}\n")
        f.write(f"Successfully matched:       {result.n_matched}\n\n")

        matched_true_ids = {m[0].strain_id for m in matches} if matches else set()
        matched_det_ids = {m[1].lineage_id for m in matches} if matches else set()
        f.write(f"Unique true strains matched:   {len(matched_true_ids)}\n")
        f.write(f"Unique detected lineages used: {len(matched_det_ids)}\n\n")

        # Section 2: Accuracy Metrics
        f.write("2. ACCURACY METRICS\n")
        f.write("-" * 80 + "\n")
        f.write("Haplotype-Level Metrics:\n")
        f.write(f"  Precision:  {result.precision:.4f}  (fraction of detected that are correct)\n")
        f.write(f"  Recall:     {result.recall:.4f}  (fraction of true strains detected)\n")
        f.write(f"  F1 Score:   {result.f1:.4f}  (harmonic mean of precision/recall)\n\n")

        f.write("SNV-Level Metrics (within detected genomic span):\n")
        f.write(f"  Precision:  {result.snv_precision:.4f}  (fraction of called SNVs that are correct)\n")
        f.write(f"  Recall:     {result.snv_recall:.4f}  (fraction of true SNVs in detected regions recovered)\n")
        snv_f1 = 2 * result.snv_precision * result.snv_recall / (result.snv_precision + result.snv_recall) \
                 if (result.snv_precision + result.snv_recall) > 0 else 0.0
        f.write(f"  F1 Score:   {snv_f1:.4f}\n")
        f.write("  Note: Recall only counts true SNVs within the min-max span of detected SNVs,\n")
        f.write("        not SNVs in unprocessed regions (gaps between windows).\n\n")

        f.write("Abundance Metrics (grouped by indistinguishable strains):\n")
        f.write(f"  Pearson r:  {result.abundance_pearson_r:.4f}  (correlation with effective truth)\n")
        f.write(f"  MAE:        {result.abundance_mae:.4f}  (mean absolute error)\n")
        f.write("  Note: 'Effective truth' sums abundances of strains that are identical\n")
        f.write("        within the detected region (indistinguishable strains).\n\n")

        # Section 3: Track/Linking Metrics
        f.write("3. TRACK & LINKING METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Track Fragmentation (mean):  {result.track_fragmentation_mean:.4f}\n")
        f.write("  - Number of detected tracks per true strain (1.0 = perfect)\n")
        f.write(f"Track Fragmentation (median): {result.track_fragmentation_median:.4f}\n")
        f.write(f"False Link Rate:             {result.false_link_rate:.4f}\n")
        f.write("  - Fraction of track links that incorrectly merge different strains\n")
        f.write(f"Missed Link Rate:            {result.missed_link_rate:.4f}\n")
        f.write("  - Fraction of true links (same strain) that were missed\n")
        f.write(f"Track Consensus Error:       {result.track_consensus_error:.4f}\n")
        f.write("  - Fraction of SNVs with incorrect consensus in linked tracks\n\n")

        # Section 4: Longitudinal/Lineage Metrics
        f.write("4. LONGITUDINAL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Lineage Precision: {result.lineage_precision:.4f}\n")
        f.write(f"Lineage Recall:    {result.lineage_recall:.4f}\n")
        f.write(f"Lineage F1:        {result.lineage_f1:.4f}\n")
        f.write(f"Rescue ΔRecall:    {result.rescue_delta_recall_rare:.4f}\n")
        f.write("  - Improvement in recall for rare strains from longitudinal rescue\n")
        f.write(f"Trajectory Error:  {result.abundance_trajectory_error:.4f}\n")
        f.write("  - Error in abundance changes over time\n\n")

        # Section 5: EM Convergence Summary
        if window_results:
            f.write("5. EM CONVERGENCE SUMMARY\n")
            f.write("-" * 80 + "\n")
            n_windows = len(window_results)
            n_converged = sum(1 for wr in window_results if wr.converged)
            convergence_rate = n_converged / n_windows if n_windows > 0 else 0.0

            avg_iterations = np.mean([wr.iterations for wr in window_results]) if window_results else 0.0
            avg_haplotypes = np.mean([len(wr.haplotypes) for wr in window_results]) if window_results else 0.0

            # Junk weight statistics
            junk_weights = []
            for wr in window_results:
                if wr.pi is not None and len(wr.pi) > 0:
                    junk_weights.append(float(wr.pi[-1]))
            avg_junk_weight = np.mean(junk_weights) if junk_weights else 0.0

            f.write(f"Total windows:       {n_windows}\n")
            f.write(f"Converged windows:   {n_converged} ({convergence_rate:.1%})\n")
            f.write(f"Avg iterations:      {avg_iterations:.1f}\n")
            f.write(f"Avg haplotypes/win:  {avg_haplotypes:.1f}\n")
            f.write(f"Avg junk weight:     {avg_junk_weight:.4f}\n")
            f.write("  - Fraction of reads assigned to junk component (noise/chimeras)\n\n")

        # Section 6: Error Analysis
        f.write("6. ERROR ANALYSIS\n")
        f.write("-" * 80 + "\n")

        # False negatives
        fn_count = len(result.false_negatives)
        f.write(f"False Negatives: {fn_count} true strains not detected\n")
        if result.false_negatives:
            # Categorize by abundance
            fn_by_abund = {'low': 0, 'medium': 0, 'high': 0}
            for fn_id in result.false_negatives:
                fn_hap = next((h for h in true_haps if h.strain_id == fn_id), None)
                if fn_hap and fn_hap.abundances:
                    max_abund = max(fn_hap.abundances.values())
                    if max_abund < 0.01:
                        fn_by_abund['low'] += 1
                    elif max_abund < 0.10:
                        fn_by_abund['medium'] += 1
                    else:
                        fn_by_abund['high'] += 1
                else:
                    fn_by_abund['low'] += 1

            f.write(f"  By abundance: low (<1%): {fn_by_abund['low']}, ")
            f.write(f"medium (1-10%): {fn_by_abund['medium']}, ")
            f.write(f"high (>10%): {fn_by_abund['high']}\n")

        # False positives
        fp_count = len(result.false_positives)
        f.write(f"\nFalse Positives: {fp_count} detected lineages not matching truth\n")
        if result.false_positives:
            f.write("  These may be chimeras, sequencing artifacts, or over-split strains\n")
        f.write("\n")

        # Section 7: Output Files
        f.write("7. OUTPUT FILES GENERATED\n")
        f.write("-" * 80 + "\n")
        f.write("TSV Data Files:\n")
        f.write("  - lineage_details.tsv: Per-lineage/contig/timepoint raw data\n")
        f.write("  - em_convergence.tsv:  Per-window EM algorithm statistics\n")
        f.write("  - linking_quality.tsv: Cross-timepoint linking analysis\n")
        f.write("\nJSON Files:\n")
        f.write("  - validation_metrics.json: Machine-readable metrics summary\n")
        f.write("\nFigures:\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Wrote validation summary to {output_path}")
    return output_path


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
    # We need two mappings:
    # 1. lineage_id -> strain_id (for lineage validation - from matches)
    # 2. track_id -> strain_id (for track validation - need to map through lineage_id)
    
    lineage_to_strain = {}
    for true_hap, det_hap, _ in matches:
        if det_hap.lineage_id:
            lineage_to_strain[det_hap.lineage_id] = true_hap.strain_id
    
    # Now build track_id -> strain_id mapping by reading the lineages.tsv file
    # which contains the mapping from original track_ids to lineage_ids
    strain_matches = {}
    
    # First, copy the lineage_id -> strain_id mapping (some code expects this)
    strain_matches.update(lineage_to_strain)
    
    # Then, read the lineages.tsv to get track_id -> lineage_id mapping
    # lineages.tsv may be in output_dir or its parent (if output_dir is 'validation' subdirectory)
    lineages_file = os.path.join(output_dir, 'lineages.tsv')
    if not os.path.exists(lineages_file):
        lineages_file = os.path.join(os.path.dirname(output_dir), 'lineages.tsv')
    if os.path.exists(lineages_file):
        try:
            with open(lineages_file) as f:
                header = f.readline().strip().split('\t')
                if 'track_id' in header and 'lineage_id' in header:
                    track_idx = header.index('track_id')
                    lineage_idx = header.index('lineage_id')
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) > max(track_idx, lineage_idx):
                            track_id = parts[track_idx]
                            lineage_id = parts[lineage_idx]
                            # Map track_id to the same strain_id as its lineage_id
                            if lineage_id in lineage_to_strain and track_id != lineage_id:
                                strain_matches[track_id] = lineage_to_strain[lineage_id]
            logger.info(f"Built track mapping: {len(lineage_to_strain)} lineage->strain, "
                       f"{len(strain_matches)} total track->strain mappings")
        except Exception as e:
            logger.warning(f"Failed to read track_id mapping from lineages.tsv: {e}")
    
    # Build truth SNVs: strain_id -> {contig -> {pos -> allele}}
    truth_snvs = {}
    for true_hap in true_haps:
        truth_snvs[true_hap.strain_id] = true_hap.snv_positions

    # Track/linking validation (if window_results provided)
    if window_results and window_size:
        try:
            from validation.validate_tracks import validate_tracks
            logger.info(f"Running track validation with {len(window_results)} window results")
            logger.info(f"Strain matches: {len(strain_matches)} mappings")
            track_result = validate_tracks(
                window_results, truth_dir, strain_matches, truth_snvs, window_size
            )
            result.track_fragmentation_mean = track_result.track_fragmentation_mean
            result.track_fragmentation_median = track_result.track_fragmentation_median
            result.false_link_rate = track_result.false_link_rate
            result.missed_link_rate = track_result.missed_link_rate
            result.track_consensus_error = track_result.track_consensus_error
            logger.info(f"Track validation complete: fragmentation={result.track_fragmentation_mean:.3f}, "
                       f"false_link={result.false_link_rate:.3f}, missed_link={result.missed_link_rate:.3f}")
            
            # Write linkability report if there's fragmentation to analyze
            if track_result.linkability_analysis:
                from validation.validate_tracks import write_linkability_report
                linkability_path = os.path.join(output_dir, 'track_linkability.txt')
                write_linkability_report(track_result.linkability_analysis, linkability_path)
                logger.info(f"Wrote track linkability report to {linkability_path}")
        except Exception as e:
            logger.warning(f"Track validation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    # Lineage validation
    try:
        from validation.validate_lineages import validate_lineages
        
        # Build detected lineages: lineage_id -> {contig -> strain_id}
        # We need to map each (lineage_id, contig) pair to a strain_id
        # Use the matches to determine which strain each detected lineage belongs to on each contig
        
        # The matches list contains (true_hap, det_hap, distance) tuples
        # For each match, map the detected haplotype's (lineage_id, contig) pairs to the matched strain_id
        lineage_contig_to_strain = {}
        
        # Process matches to build the mapping
        for true_hap, det_match, _ in matches:
            if not det_match.lineage_id:
                continue
            matched_strain = true_hap.strain_id
            
            # For each contig where this detected haplotype has SNVs
            for contig in det_match.snv_alleles.keys():
                # Check if this contig exists in the true haplotype (to ensure it's a valid match)
                if contig in true_hap.snv_positions:
                    key = (det_match.lineage_id, contig)
                    # Only add if we haven't seen this (lineage_id, contig) pair before
                    # or if it matches the same strain (to avoid conflicts)
                    if key not in lineage_contig_to_strain:
                        lineage_contig_to_strain[key] = matched_strain
                    elif lineage_contig_to_strain[key] != matched_strain:
                        # Conflict: same lineage_id+contig matches different strains
                        # This can happen if a detected lineage is incorrectly linked across strains
                        logger.debug(f"Conflict: lineage {det_match.lineage_id} on contig {contig} "
                                   f"matches both {lineage_contig_to_strain[key]} and {matched_strain}")
        
        # Now build detected_lineages structure: lineage_id -> {contig -> strain_id}
        detected_lineages = {}
        for (lineage_id, contig), strain_id in lineage_contig_to_strain.items():
            if lineage_id not in detected_lineages:
                detected_lineages[lineage_id] = {}
            detected_lineages[lineage_id][contig] = strain_id
        
        logger.info(f"Built detected_lineages: {len(detected_lineages)} lineages, "
                   f"{sum(len(c) for c in detected_lineages.values())} (lineage_id, contig) pairs")
        if not detected_lineages:
            logger.warning("WARNING: detected_lineages is empty! This will cause zero lineage metrics.")
            logger.warning(f"  Number of matches: {len(matches)}")
            logger.warning(f"  Number of detected haplotypes: {len(detected_haps)}")
            if matches:
                sample_match = matches[0]
                logger.warning(f"  Sample match: true_strain={sample_match[0].strain_id}, "
                             f"det_lineage={sample_match[1].lineage_id}, "
                             f"det_contigs={list(sample_match[1].snv_alleles.keys())}")
        
        # Build abundance dictionaries
        true_abundances = {h.strain_id: h.abundances for h in true_haps}
        detected_abundances = {h.lineage_id: h.abundances for h in detected_haps if h.lineage_id}
        
        logger.info(f"Running lineage validation with {len(detected_lineages)} detected lineages")
        if detected_lineages:
            total_contigs = sum(len(contigs) for contigs in detected_lineages.values())
            logger.info(f"  Total (lineage_id, contig) pairs: {total_contigs}")
            sample = list(detected_lineages.items())[0]
            logger.info(f"  Sample: lineage {sample[0]} appears on contigs: {list(sample[1].keys())}")
        logger.info(f"True abundances: {len(true_abundances)} strains")
        logger.info(f"Detected abundances: {len(detected_abundances)} lineages")
        
        lineage_result = validate_lineages(
            detected_lineages, truth_dir, true_abundances, detected_abundances,
            detected_without_rescue=detected_without_rescue
        )
        result.lineage_precision = lineage_result.lineage_precision
        result.lineage_recall = lineage_result.lineage_recall
        result.lineage_f1 = lineage_result.lineage_f1
        result.rescue_delta_recall_rare = lineage_result.rescue_delta_recall_rare
        result.abundance_trajectory_error = lineage_result.abundance_trajectory_error
        logger.info(f"Lineage validation complete: precision={result.lineage_precision:.3f}, "
                   f"recall={result.lineage_recall:.3f}, f1={result.lineage_f1:.3f}")
    except Exception as e:
        logger.warning(f"Lineage validation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # Generate figures
    generate_figures(
        result, true_haps, detected_haps, matches, output_dir,
        window_results=window_results,
        truth_dir=truth_dir
    )

    # Generate detailed TSV output files
    try:
        write_lineage_details(true_haps, detected_haps, matches, output_dir)
    except Exception as e:
        logger.warning(f"Failed to write lineage_details.tsv: {e}")

    try:
        write_linking_quality(detected_haps, matches, output_dir)
    except Exception as e:
        logger.warning(f"Failed to write linking_quality.tsv: {e}")

    if window_results:
        try:
            write_em_convergence(window_results, output_dir)
        except Exception as e:
            logger.warning(f"Failed to write em_convergence.tsv: {e}")

        try:
            write_rescue_statistics(window_results, output_dir)
        except Exception as e:
            logger.warning(f"Failed to write rescue_statistics.tsv: {e}")

    try:
        write_validation_summary(result, true_haps, detected_haps, matches, output_dir, window_results)
    except Exception as e:
        logger.warning(f"Failed to write validation_summary.txt: {e}")

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
            'snv_true_total': result.snv_true_total,
            'snv_true_in_span': result.snv_true_in_span,
            'snv_detected_total': result.snv_detected_total,
            'snv_correct_total': result.snv_correct_total,
            'snv_span_coverage_frac': result.snv_span_coverage_frac,
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
        
        # Compute matched IDs from result.matches (tuples of (true_id, detected_id, distance))
        matched_true_ids = {m[0] for m in result.matches} if result.matches else set()
        matched_detected_ids = {m[1] for m in result.matches} if result.matches else set()

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
    
    # Compute matched IDs from result.matches (tuples of (true_id, detected_id, distance))
    matched_true_ids = {m[0] for m in result.matches} if result.matches else set()
    matched_detected_ids = {m[1] for m in result.matches} if result.matches else set()

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
