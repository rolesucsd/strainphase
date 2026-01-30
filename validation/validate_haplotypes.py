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
from typing import Any, Dict, List, Optional, Set, Tuple
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
    matches: List[Tuple[str, str, float]] = None  # (true_ids_csv, detected_id, min_distance)
    
    # Detailed diagnostics
    false_negatives: List[str] = None  # Informative windows not detected
    false_positives: List[str] = None  # Detected lineages not matching truth
    window_recall: float = 0.0  # Pooled window-level recall
    window_informative_total: int = 0  # Total informative windows (pooled)
    window_detected_total: int = 0  # Detected windows (pooled)
    window_recall_by_timepoint: Dict[str, float] = None
    window_recall_by_contig: Dict[str, float] = None
    match_details_full: List[Dict] = None  # Full match details with SNV counts, grouped abundances, etc.
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
        if self.window_recall_by_timepoint is None:
            self.window_recall_by_timepoint = {}
        if self.window_recall_by_contig is None:
            self.window_recall_by_contig = {}
        if self.match_details_full is None:
            self.match_details_full = []
        if self.per_contig_metrics is None:
            self.per_contig_metrics = {}
        if self.per_timepoint_metrics is None:
            self.per_timepoint_metrics = {}


@dataclass
class StrainGroup:
    """Group of strains that are indistinguishable within a window."""
    strain_ids: List[str]
    consensus: Dict[int, str]  # pos -> allele (shared by all strains in group)
    total_abundance: float


@dataclass
class WindowMetrics:
    """Metrics computed for a single genomic window."""
    # Identification
    sample: str           # Timepoint
    contig: str
    window_start: int
    window_end: int

    # Haplotype-level (counts based on strain GROUPS, not individual strains)
    n_strain_groups: int  # Number of distinguishable strain groups in window
    n_detected: int
    n_matched: int
    precision: float      # matched_detected / detected
    recall: float         # matched_groups / n_strain_groups

    # SNV-level
    n_snv_positions: int
    n_informative_snvs: int
    n_snv_detected: int
    n_snv_correct: int
    snv_precision: float
    snv_recall: float

    # Abundance
    abundance_pairs: List[Tuple[float, float]]  # (true, detected)
    abundance_mae: Optional[float]

    # Match details
    matches: List[Tuple[List[str], int, float]]  # (group_strain_ids, hap_idx, distance)
    unmatched_true: List[str]  # strain_ids not matched
    unmatched_detected: List[int]  # hap indices not matched


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple windows."""
    n_windows: int
    n_windows_informative: int

    # Primary metrics (macro-averaged - equal weight per window since all windows same size)
    precision: float        # mean of per-window precision
    recall: float           # mean of per-window recall
    f1: float               # mean of per-window F1
    snv_precision: float
    snv_recall: float

    # Abundance (pooled)
    abundance_pearson_r: float
    abundance_mae: float
    n_abundance_pairs: int

    # Totals
    total_true: int
    total_detected: int
    total_matched: int
    total_snv_detected: int
    total_snv_correct: int


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
            abundance = float(row.get('abundance', ''))

            # Store abundance
            lineage_data[lineage_id]['abundances'][sample] = abundance

            # Parse SNV alleles
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
    # Only compare positions present in the detected haplotype; treat missing as unknown.
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

    # Distance is 1 - match fraction over shared SNVs.
    match_fraction = n_matches / n_shared
    distance = 1.0 - match_fraction
    return distance, n_matches, n_shared, match_fraction


def match_haplotypes(
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    max_distance: float = 0,
    min_shared_snvs: int = 1,
    min_match_fraction: float = 1
) -> List[Tuple[TrueHaplotype, DetectedHaplotype, float]]:
    """
    Match detected haplotypes to true haplotypes.

    Since strainphase may split strains per-contig, one true strain can match
    multiple detected lineages (one per contig).

    Returns list of (true_hap, detected_hap, distance) tuples.
    """
    if not true_haps or not detected_haps:
        return []

    # Compute distance matrix.
    distances = []
    for true_hap in true_haps:
        for det_hap in detected_haps:
            dist, n_shared, match_fraction = compute_haplotype_distance(
                true_hap, det_hap
            )
            if n_shared < min_shared_snvs:
                continue
            if match_fraction < min_match_fraction:
                continue
            distances.append((dist, true_hap, det_hap, n_shared))

    # Sort by distance.
    distances.sort(key=lambda x: x[0])

    def indistinguishable_at_detected_positions(
        det_hap: DetectedHaplotype, true_a: TrueHaplotype, true_b: TrueHaplotype
    ) -> bool:
        """
        Return True if true_a and true_b have identical alleles at all positions
        where det_hap has calls (requires at least one overlapping position).
        """
        has_overlap = False
        for contig, det_snvs in det_hap.snv_alleles.items():
            a_snvs = true_a.snv_positions.get(contig, {})
            b_snvs = true_b.snv_positions.get(contig, {})
            for pos in det_snvs.keys():
                a_allele = a_snvs.get(pos)
                b_allele = b_snvs.get(pos)
                if a_allele is None or b_allele is None:
                    continue
                has_overlap = True
                if a_allele != b_allele:
                    return False
        return has_overlap

    # Allow one true strain to match multiple detected lineages (per-contig splitting)
    # But ensure each detected lineage only matches one true strain, unless multiple
    # true strains are indistinguishable at the detected positions.
    matches = []
    used_detected = set()
    detected_to_true: Dict[str, List[TrueHaplotype]] = defaultdict(list)

    def record_match(true_h: TrueHaplotype, det_h: DetectedHaplotype, dist: float) -> None:
        matches.append((true_h, det_h, dist))
        used_detected.add(det_h.lineage_id)
        detected_to_true[det_h.lineage_id].append(true_h)

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
                # Allow multiple true strains to match the same detected lineage
                # only if they are indistinguishable at detected positions.
                existing = detected_to_true.get(det_h.lineage_id, [])
                if not all(
                    indistinguishable_at_detected_positions(det_h, true_hap, prev)
                    for prev in existing
                ):
                    continue
            if dist > max_distance:
                continue
            candidate_matches.append((dist, det_h, n_shared))

        # Sort by distance (best first)
        candidate_matches.sort(key=lambda x: x[0])

        # Match greedily: allow multiple matches per true strain.
        matched_for_strain = []
        for dist, det_h, n_shared in candidate_matches:
            # Exact matches (distance ~0) should still count as matched, even if redundant.
            if dist <= max_distance:
                matched_for_strain.append((dist, det_h, n_shared))
                record_match(true_hap, det_h, dist)
                continue
            matched_for_strain.append((dist, det_h, n_shared))
            record_match(true_hap, det_h, dist)

    return matches


# =============================================================================
# Compute metrics
# =============================================================================
def _grouped_true_abundances_for_match(
    *,
    true_hap: TrueHaplotype,
    det_hap: DetectedHaplotype,
    true_haps: Optional[List[TrueHaplotype]],
) -> Tuple[Dict[str, float], List[str]]:
    """
    Build grouped true abundances for a single true/detected match.

    For the detected haplotype, we identify which other true strains are
    indistinguishable from `true_hap` at the positions the detected haplotype
    actually covers. Those indistinguishable strains are grouped together and
    their abundances are summed at each shared timepoint.

    Returns:
        grouped_true_abundances: dict {timepoint -> summed true abundance}
        common_tps: list of timepoints present in both true and detected haplotypes
    """
    # Determine which positions the detected haplotype actually calls.
    detected_positions: Dict[str, set] = {}
    for contig, snvs in det_hap.snv_alleles.items():
        detected_positions[contig] = set(snvs.keys())

    # Start with the matched true haplotype and add any indistinguishable strains.
    indistinguishable_strains = [true_hap]
    if true_haps is not None:
        for other_hap in true_haps:
            if other_hap.strain_id == true_hap.strain_id:
                continue

            # Check equality only at detected positions where both strains have alleles.
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

            # Only group strains that overlap and are identical at all overlapping positions.
            if is_identical and has_overlap:
                indistinguishable_strains.append(other_hap)

    # Only compare timepoints that exist for both the true and detected haplotype.
    common_tps = list(set(true_hap.abundances.keys()) & set(det_hap.abundances.keys()))
    # Sum abundances across indistinguishable strains for each shared timepoint.
    grouped_true_abundances = {
        tp: sum(h.abundances.get(tp, 0.0) for h in indistinguishable_strains)
        for tp in common_tps
    }

    return grouped_true_abundances, common_tps


def _build_match_details_full(
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    true_haps: List[TrueHaplotype],
) -> List[Dict[str, Any]]:
    """
    Build per-match diagnostics for reporting/debugging.

    For each (true, detected) match, compute SNV overlap statistics and
    grouped abundance error using the same indistinguishable-strain logic
    as the global abundance metrics.
    """
    match_details_full = []
    for true_hap, det_hap, distance in matches:
        # Count SNVs for true and detected haplotypes.
        n_shared_snvs = 0
        n_matching_snvs = 0
        n_true_snvs = sum(len(snvs) for snvs in true_hap.snv_positions.values())
        n_detected_snvs = sum(len(snvs) for snvs in det_hap.snv_alleles.values())

        # Compare alleles only at positions called by the detected haplotype.
        for contig, true_snvs in true_hap.snv_positions.items():
            det_snvs = det_hap.snv_alleles.get(contig, {})
            for pos, true_allele in true_snvs.items():
                if pos in det_snvs:
                    n_shared_snvs += 1
                    if det_snvs[pos] == true_allele:
                        n_matching_snvs += 1

        # Group true abundances across indistinguishable strains and compute MAE.
        grouped_true_abundances, common_tps = _grouped_true_abundances_for_match(
            true_hap=true_hap,
            det_hap=det_hap,
            true_haps=true_haps,
        )
        abundance_errors = []
        for tp in common_tps:
            abundance_errors.append(abs(grouped_true_abundances[tp] - det_hap.abundances[tp]))

        # Assemble a full record for detailed reporting.
        match_details_full.append({
            'true_strain_id': true_hap.strain_id,
            'detected_lineage_id': det_hap.lineage_id,
            'distance': distance,
            'n_true_snvs': n_true_snvs,
            'n_detected_snvs': n_detected_snvs,
            'n_shared_snvs': n_shared_snvs,
            'n_matching_snvs': n_matching_snvs,
            'snv_match_fraction': n_matching_snvs / n_shared_snvs if n_shared_snvs > 0 else 0.0,
            'grouped_abundance_mae': np.mean(abundance_errors) if abundance_errors else None,
            'common_timepoints': list(common_tps),
            'grouped_true_abundances': {tp: grouped_true_abundances[tp] for tp in common_tps},
            'detected_abundances': {tp: det_hap.abundances[tp] for tp in common_tps},
            'is_sweeping': true_hap.is_sweeping,
        })

    return match_details_full


def _dedup_match_details(
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]]
) -> List[Tuple[str, str, float]]:
    """
    Collapse matches by detected lineage for concise summary reporting.

    Multiple true strains can match the same detected lineage; this function
    emits one row per detected lineage with a CSV list of true strain IDs and
    the minimum observed distance among those matches.
    """
    dedup_by_detected: Dict[str, Dict[str, Any]] = {}
    for true_hap, det_hap, dist in matches:
        entry = dedup_by_detected.get(det_hap.lineage_id)
        if entry is None:
            dedup_by_detected[det_hap.lineage_id] = {
                "true_ids": {true_hap.strain_id},
                "detected_id": det_hap.lineage_id,
                "min_distance": dist,
            }
        else:
            entry["true_ids"].add(true_hap.strain_id)
            if dist < entry["min_distance"]:
                entry["min_distance"] = dist

    match_details = []
    for entry in dedup_by_detected.values():
        true_ids_csv = ",".join(sorted(entry["true_ids"]))
        match_details.append((true_ids_csv, entry["detected_id"], entry["min_distance"]))
    return match_details


# =============================================================================
# Per-window metrics computation (new architecture)
# =============================================================================

def _group_indistinguishable_strains(
    present_strains: List[str],
    truth_alleles: Dict[str, Dict[str, Dict[int, str]]],
    strain_abundances: Dict[str, Dict[str, float]],
    contig: str,
    window_start: int,
    window_end: int,
    sample: str,
) -> List[StrainGroup]:
    """
    Group strains that have identical alleles at all window positions.

    Strains are indistinguishable if they have the same allele at every
    SNV position within the window. Returns a list of StrainGroup objects.
    """
    # Build window-local allele signatures for each present strain
    strain_signatures: Dict[str, Dict[int, str]] = {}
    for sid in present_strains:
        contig_snvs = truth_alleles.get(sid, {}).get(contig, {})
        window_alleles = {
            pos: allele
            for pos, allele in contig_snvs.items()
            if window_start <= pos < window_end
        }
        strain_signatures[sid] = window_alleles

    # Group strains by their signature (tuple of sorted (pos, allele) pairs)
    signature_to_strains: Dict[tuple, List[str]] = defaultdict(list)
    for sid, alleles in strain_signatures.items():
        sig = tuple(sorted(alleles.items()))
        signature_to_strains[sig].append(sid)

    # Build StrainGroup objects
    groups = []
    for sig, strain_ids in signature_to_strains.items():
        consensus = dict(sig)
        total_abundance = sum(
            strain_abundances.get(sid, {}).get(sample, 0.0)
            for sid in strain_ids
        )
        groups.append(StrainGroup(
            strain_ids=strain_ids,
            consensus=consensus,
            total_abundance=total_abundance
        ))

    return groups


def _match_window_haplotypes(
    detected_haps: List,  # List of haplotype objects with .consensus dict
    strain_groups: List[StrainGroup],
    window_positions: Set[int],
) -> Tuple[List[Tuple[List[str], int, float]], List[int], List[int]]:
    """
    Match detected haplotypes to strain groups within a window.

    Uses greedy assignment: sort by distance, each detected haplotype
    matches at most one strain group.

    Returns:
        matches: List of (group_strain_ids, hap_idx, distance)
        unmatched_group_indices: List of group indices not matched
        unmatched_hap_indices: List of hap indices not matched
    """
    if not detected_haps or not strain_groups:
        unmatched_groups = list(range(len(strain_groups)))
        unmatched_haps = list(range(len(detected_haps)))
        return [], unmatched_groups, unmatched_haps

    # Compute distances between each detected haplotype and strain group
    # Only compare at positions within the window
    distance_candidates = []
    for hap_idx, hap in enumerate(detected_haps):
        hap_alleles = hap.consensus  # Dict[int, str]
        for group_idx, group in enumerate(strain_groups):
            n_shared = 0
            n_matches = 0
            for pos in window_positions:
                hap_allele = hap_alleles.get(pos)
                group_allele = group.consensus.get(pos)
                if hap_allele is not None and group_allele is not None:
                    n_shared += 1
                    if hap_allele == group_allele:
                        n_matches += 1

            if n_shared > 0:
                distance = 1.0 - (n_matches / n_shared)
                distance_candidates.append((distance, hap_idx, group_idx, n_shared))

    # Sort by distance (best first)
    distance_candidates.sort(key=lambda x: (x[0], -x[3]))  # Lower distance, more shared SNVs

    # Greedy matching
    matched_groups = set()
    matched_haps = set()
    matches = []

    for distance, hap_idx, group_idx, n_shared in distance_candidates:
        if hap_idx in matched_haps or group_idx in matched_groups:
            continue
        # Only match if distance is reasonable (exact or near-exact match)
        if distance <= 0.1:  # Allow 10% mismatch for sequencing errors
            group = strain_groups[group_idx]
            matches.append((group.strain_ids, hap_idx, distance))
            matched_groups.add(group_idx)
            matched_haps.add(hap_idx)

    unmatched_group_indices = [i for i in range(len(strain_groups)) if i not in matched_groups]
    unmatched_hap_indices = [i for i in range(len(detected_haps)) if i not in matched_haps]

    return matches, unmatched_group_indices, unmatched_hap_indices


def compute_window_metrics(
    window_result,  # WindowResult object
    true_haps: List[TrueHaplotype],
) -> Optional[WindowMetrics]:
    """
    Compute all metrics for a single genomic window.

    Returns None if the window has no informative SNVs (no strains present
    or all strains identical within window).
    """
    sample = window_result.window.sample or ""
    contig = window_result.window.contig
    window_start = window_result.window.start
    window_end = window_result.window.end

    # Build lookups
    strain_abundances = {h.strain_id: h.abundances for h in true_haps}
    truth_alleles = {h.strain_id: h.snv_positions for h in true_haps}

    # Identify strains present at this timepoint (abundance > 0)
    present_strains = [
        h.strain_id for h in true_haps
        if h.abundances.get(sample, 0.0) > 0.0
    ]

    if not present_strains:
        return None

    # Find all SNV positions in window across present strains
    window_positions = set()
    for sid in present_strains:
        contig_snvs = truth_alleles.get(sid, {}).get(contig, {})
        for pos in contig_snvs.keys():
            if window_start <= pos < window_end:
                window_positions.add(pos)

    if not window_positions:
        return None

    # Group indistinguishable strains
    strain_groups = _group_indistinguishable_strains(
        present_strains, truth_alleles, strain_abundances,
        contig, window_start, window_end, sample
    )

    # Find informative positions (where strain groups differ)
    informative_positions = set()
    for pos in window_positions:
        alleles_at_pos = set()
        for group in strain_groups:
            allele = group.consensus.get(pos)
            if allele is not None:
                alleles_at_pos.add(allele)
        if len(alleles_at_pos) >= 2:
            informative_positions.add(pos)

    n_informative_snvs = len(informative_positions)

    # If no informative SNVs, all strains are identical within this window
    # Still compute metrics but with n_strain_groups based on non-empty groups
    if n_informative_snvs == 0:
        # All present strains are identical at these positions
        # Treat as single group
        pass

    # Get detected haplotypes from window result
    detected_haps = window_result.haplotypes
    n_detected = len(detected_haps)

    # Match detected haplotypes to strain groups
    matches, unmatched_group_indices, unmatched_hap_indices = _match_window_haplotypes(
        detected_haps, strain_groups, window_positions
    )
    n_matched = len(matches)

    # Get unmatched strain IDs
    unmatched_true = []
    for idx in unmatched_group_indices:
        unmatched_true.extend(strain_groups[idx].strain_ids)

    # Compute haplotype-level precision/recall
    # Precision: fraction of detected haplotypes that matched a group
    precision = n_matched / n_detected if n_detected > 0 else 0.0
    # Recall: fraction of strain groups that were matched
    n_strain_groups = len(strain_groups)
    recall = n_matched / n_strain_groups if n_strain_groups > 0 else 0.0

    # Compute SNV-level metrics
    # For each matched pair, count SNVs detected and correct
    n_snv_detected = 0
    n_snv_correct = 0

    for group_strain_ids, hap_idx, distance in matches:
        hap = detected_haps[hap_idx]
        hap_alleles = hap.consensus
        # Find the group
        matching_group = None
        for g in strain_groups:
            if set(g.strain_ids) == set(group_strain_ids):
                matching_group = g
                break
        if matching_group is None:
            continue

        for pos in window_positions:
            hap_allele = hap_alleles.get(pos)
            true_allele = matching_group.consensus.get(pos)
            if hap_allele is not None:
                n_snv_detected += 1
                if true_allele is not None and hap_allele == true_allele:
                    n_snv_correct += 1

    # Also count detected SNVs from unmatched haplotypes (false positives)
    for hap_idx in unmatched_hap_indices:
        hap = detected_haps[hap_idx]
        for pos in window_positions:
            if pos in hap.consensus:
                n_snv_detected += 1

    # SNV precision/recall
    snv_precision = n_snv_correct / n_snv_detected if n_snv_detected > 0 else 0.0
    # SNV recall: correct SNVs / total true SNV positions across matched groups
    # Expected = (window SNV positions) * (number of matched groups)
    expected_snv_count = len(window_positions) * n_matched if n_matched > 0 else len(window_positions)
    snv_recall = n_snv_correct / expected_snv_count if expected_snv_count > 0 else 0.0

    # Compute abundance pairs (true_group_abundance, detected_weight)
    abundance_pairs = []
    for group_strain_ids, hap_idx, distance in matches:
        hap = detected_haps[hap_idx]
        hap_weight = hap.weight if hasattr(hap, 'weight') else 0.0
        # Find group abundance
        for g in strain_groups:
            if set(g.strain_ids) == set(group_strain_ids):
                abundance_pairs.append((g.total_abundance, hap_weight))
                break

    abundance_mae = None
    if abundance_pairs:
        abundance_mae = np.mean([abs(t - d) for t, d in abundance_pairs])

    return WindowMetrics(
        sample=sample,
        contig=contig,
        window_start=window_start,
        window_end=window_end,
        n_strain_groups=n_strain_groups,
        n_detected=n_detected,
        n_matched=n_matched,
        precision=precision,
        recall=recall,
        n_snv_positions=len(window_positions),
        n_informative_snvs=n_informative_snvs,
        n_snv_detected=n_snv_detected,
        n_snv_correct=n_snv_correct,
        snv_precision=snv_precision,
        snv_recall=snv_recall,
        abundance_pairs=abundance_pairs,
        abundance_mae=abundance_mae,
        matches=matches,
        unmatched_true=unmatched_true,
        unmatched_detected=unmatched_hap_indices,
    )


def aggregate_window_metrics(
    window_metrics_list: List[WindowMetrics],
) -> AggregatedMetrics:
    """
    Aggregate metrics across windows using macro-averaging.

    All windows have equal weight (same size), so we take simple means
    of per-window metrics. Abundance pairs are pooled for correlation.
    """
    if not window_metrics_list:
        return AggregatedMetrics(
            n_windows=0,
            n_windows_informative=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            snv_precision=0.0,
            snv_recall=0.0,
            abundance_pearson_r=0.0,
            abundance_mae=0.0,
            n_abundance_pairs=0,
            total_true=0,
            total_detected=0,
            total_matched=0,
            total_snv_detected=0,
            total_snv_correct=0,
        )

    n_windows = len(window_metrics_list)
    n_windows_informative = sum(1 for w in window_metrics_list if w.n_informative_snvs > 0)

    # Macro-averaged metrics (mean across windows)
    precisions = [w.precision for w in window_metrics_list]
    recalls = [w.recall for w in window_metrics_list]
    f1s = [
        2 * w.precision * w.recall / (w.precision + w.recall)
        if (w.precision + w.recall) > 0 else 0.0
        for w in window_metrics_list
    ]
    snv_precisions = [w.snv_precision for w in window_metrics_list]
    snv_recalls = [w.snv_recall for w in window_metrics_list]

    precision = np.mean(precisions) if precisions else 0.0
    recall = np.mean(recalls) if recalls else 0.0
    f1 = np.mean(f1s) if f1s else 0.0
    snv_precision = np.mean(snv_precisions) if snv_precisions else 0.0
    snv_recall = np.mean(snv_recalls) if snv_recalls else 0.0

    # Pool abundance pairs for correlation
    all_abundance_pairs = []
    for w in window_metrics_list:
        all_abundance_pairs.extend(w.abundance_pairs)

    abundance_pearson_r = 0.0
    abundance_mae = 0.0
    if len(all_abundance_pairs) >= 2:
        true_abunds = [p[0] for p in all_abundance_pairs]
        det_abunds = [p[1] for p in all_abundance_pairs]
        abundance_pearson_r = np.corrcoef(true_abunds, det_abunds)[0, 1]
        if np.isnan(abundance_pearson_r):
            abundance_pearson_r = 0.0
        abundance_mae = np.mean([abs(t - d) for t, d in all_abundance_pairs])

    # Totals
    total_true = sum(w.n_strain_groups for w in window_metrics_list)
    total_detected = sum(w.n_detected for w in window_metrics_list)
    total_matched = sum(w.n_matched for w in window_metrics_list)
    total_snv_detected = sum(w.n_snv_detected for w in window_metrics_list)
    total_snv_correct = sum(w.n_snv_correct for w in window_metrics_list)

    return AggregatedMetrics(
        n_windows=n_windows,
        n_windows_informative=n_windows_informative,
        precision=precision,
        recall=recall,
        f1=f1,
        snv_precision=snv_precision,
        snv_recall=snv_recall,
        abundance_pearson_r=abundance_pearson_r,
        abundance_mae=abundance_mae,
        n_abundance_pairs=len(all_abundance_pairs),
        total_true=total_true,
        total_detected=total_detected,
        total_matched=total_matched,
        total_snv_detected=total_snv_detected,
        total_snv_correct=total_snv_correct,
    )


def aggregate_by_contig(
    window_metrics_list: List[WindowMetrics],
) -> Dict[str, AggregatedMetrics]:
    """Group windows by contig and aggregate each group."""
    by_contig: Dict[str, List[WindowMetrics]] = defaultdict(list)
    for w in window_metrics_list:
        by_contig[w.contig].append(w)

    return {
        contig: aggregate_window_metrics(windows)
        for contig, windows in by_contig.items()
    }


def aggregate_by_timepoint(
    window_metrics_list: List[WindowMetrics],
) -> Dict[str, AggregatedMetrics]:
    """Group windows by timepoint/sample and aggregate each group."""
    by_timepoint: Dict[str, List[WindowMetrics]] = defaultdict(list)
    for w in window_metrics_list:
        by_timepoint[w.sample].append(w)

    return {
        sample: aggregate_window_metrics(windows)
        for sample, windows in by_timepoint.items()
    }


def compute_all_window_metrics(
    window_results: List,
    true_haps: List[TrueHaplotype],
) -> Tuple[List[WindowMetrics], AggregatedMetrics, Dict[str, AggregatedMetrics], Dict[str, AggregatedMetrics]]:
    """
    Main orchestrator: compute per-window metrics then aggregate.

    Returns:
        all_metrics: List of WindowMetrics for each window
        overall: AggregatedMetrics across all windows
        by_contig: Dict mapping contig -> AggregatedMetrics
        by_timepoint: Dict mapping sample/timepoint -> AggregatedMetrics
    """
    all_metrics = []

    for wr in window_results:
        metrics = compute_window_metrics(wr, true_haps)
        if metrics is not None:
            all_metrics.append(metrics)

    overall = aggregate_window_metrics(all_metrics)
    by_contig = aggregate_by_contig(all_metrics)
    by_timepoint = aggregate_by_timepoint(all_metrics)

    return all_metrics, overall, by_contig, by_timepoint


def compute_window_recall_metrics(
    true_haps: List[TrueHaplotype],
    window_results: List,
) -> Tuple[float, int, int, Dict[str, float], Dict[str, float], List[Dict]]:
    """
    Compute window-level recall based on informative SNVs per window.

    A window is "informative" if it contains >=1 SNV position where at least two
    distinct allele groups exist among strains present at that timepoint.

    A window is "detected" if all informative SNVs in that window are covered by
    at least one haplotype consensus in that window.
    """
    # Build strain -> abundances and contig->pos->allele lookup from truth.
    strain_abundances = {h.strain_id: h.abundances for h in true_haps}
    truth_alleles = {h.strain_id: h.snv_positions for h in true_haps}

    per_timepoint_totals = defaultdict(lambda: {"informative": 0, "detected": 0})
    per_contig_totals = defaultdict(lambda: {"informative": 0, "detected": 0})
    missed_windows = []

    # Scan each window to compute informative and detected windows.
    for wr in window_results:
        sample = wr.window.sample or ""
        contig = wr.window.contig
        start = wr.window.start
        end = wr.window.end

        # Determine strains present at this timepoint (abundance > 0).
        present_strains = [
            sid for sid, abunds in strain_abundances.items() if abunds.get(sample, 0.0) > 0.0
        ]
        if not present_strains:
            continue

        # Compute informative SNV positions in this window.
        informative_positions = []
        for sid in present_strains:
            for pos, allele in truth_alleles.get(sid, {}).get(contig, {}).items():
                if start <= pos < end:
                    informative_positions.append(pos)
        if not informative_positions:
            continue

        informative_positions = sorted(set(informative_positions))
        truly_informative = []
        for pos in informative_positions:
            allele_groups = set()
            for sid in present_strains:
                allele = truth_alleles.get(sid, {}).get(contig, {}).get(pos)
                if allele is not None:
                    allele_groups.add(allele)
            if len(allele_groups) >= 2:
                truly_informative.append(pos)

        if not truly_informative:
            continue

        # Check detected coverage by haplotypes in this window
        detected_positions = set()
        for hap in wr.haplotypes:
            detected_positions.update(hap.consensus.keys())

        missing_positions = [pos for pos in truly_informative if pos not in detected_positions]
        is_detected = len(missing_positions) == 0

        per_timepoint_totals[sample]["informative"] += 1
        per_contig_totals[contig]["informative"] += 1
        if is_detected:
            per_timepoint_totals[sample]["detected"] += 1
            per_contig_totals[contig]["detected"] += 1
        else:
            missed_windows.append(
                {
                    "sample": sample,
                    "contig": contig,
                    "window_start": start,
                    "window_end": end,
                    "n_informative_snvs": len(truly_informative),
                    "n_missing_snvs": len(missing_positions),
                    "missing_positions": ",".join(str(p) for p in missing_positions),
                }
            )

    total_informative = sum(v["informative"] for v in per_timepoint_totals.values())
    total_detected = sum(v["detected"] for v in per_timepoint_totals.values())
    pooled_recall = total_detected / total_informative if total_informative > 0 else 0.0

    by_timepoint = {
        tp: (vals["detected"] / vals["informative"] if vals["informative"] > 0 else 0.0)
        for tp, vals in per_timepoint_totals.items()
    }
    by_contig = {
        contig: (vals["detected"] / vals["informative"] if vals["informative"] > 0 else 0.0)
        for contig, vals in per_contig_totals.items()
    }

    return pooled_recall, total_informative, total_detected, by_timepoint, by_contig, missed_windows



def compute_detection_sensitivity(
    true_haps: List[TrueHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    n_bins: int = 8
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Compute detection sensitivity curve and threshold.

    Returns: (detection_threshold, curve_dict)
    """
    # Build (abundance, matched?) points across all timepoints.
    matched_ids = {true_hap.strain_id for true_hap, _, _ in matches}
    abundance_points = []
    for true_hap in true_haps:
        for tp, abund in true_hap.abundances.items():
            abundance_points.append((abund, true_hap.strain_id in matched_ids))

    if not abundance_points:
        return 0.0, {"bins": [], "recall": []}

    # Bin abundances and compute recall per bin.
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

    # Define detection threshold as the first bin with >=50% recall.
    threshold = 0.0
    for i, recall in enumerate(recall_by_bin):
        if recall >= 0.5:
            threshold = float(bins[i])
            break

    curve = {"bins": bins.tolist(), "recall": recall_by_bin}
    return threshold, curve

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
    summary_path = os.path.join(output_dir, 'lineage_summary_by_track.tsv')

    # Build match lookup: detected_lineage_id -> (true_hap, distance)
    match_lookup = {}
    for true_hap, det_hap, dist in matches:
        if det_hap.lineage_id not in match_lookup:
            match_lookup[det_hap.lineage_id] = (true_hap, dist)

    records = []
    summary_by_track: Dict[str, Dict[str, Any]] = {}

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

                track_id = det_hap.track_id or lineage_id
                summary = summary_by_track.get(track_id)
                if summary is None:
                    summary = {
                        'track_id': track_id,
                        'matched_strain': matched_strain,
                        'lineage_ids': set(),
                        'contigs': set(),
                        'timepoints': set(),
                        'n_records': 0,
                        'n_snvs_detected_total': 0,
                        'n_snvs_true_total': 0,
                        'n_shared_snvs_total': 0,
                        'n_matching_snvs_total': 0,
                        'n_different_snvs_total': 0,
                        'snv_distance_sum': 0.0,
                        'abundance_diff_sum': 0.0,
                        'detected_abundance_sum': 0.0,
                        'true_abundance_sum': 0.0,
                    }
                    summary_by_track[track_id] = summary

                summary['lineage_ids'].add(lineage_id)
                summary['contigs'].add(contig)
                summary['timepoints'].add(timepoint)
                summary['n_records'] += 1
                summary['n_snvs_detected_total'] += n_snvs_detected
                summary['n_snvs_true_total'] += n_snvs_true
                summary['n_shared_snvs_total'] += n_shared_snvs
                summary['n_matching_snvs_total'] += n_matching_snvs
                summary['n_different_snvs_total'] += n_different_snvs
                summary['snv_distance_sum'] += snv_distance
                summary['abundance_diff_sum'] += abundance_diff
                summary['detected_abundance_sum'] += det_abund
                summary['true_abundance_sum'] += true_abund

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

    # Write per-track summary
    summary_records = []
    for track_id, summary in summary_by_track.items():
        n_records = summary['n_records']
        if n_records == 0:
            continue
        summary_records.append({
            'track_id': track_id,
            'matched_strain': summary['matched_strain'],
            'lineage_ids': ",".join(sorted(summary['lineage_ids'])),
            'n_lineages': len(summary['lineage_ids']),
            'n_contigs': len(summary['contigs']),
            'n_timepoints': len(summary['timepoints']),
            'n_records': n_records,
            'n_snvs_detected_total': summary['n_snvs_detected_total'],
            'n_snvs_true_total': summary['n_snvs_true_total'],
            'n_shared_snvs_total': summary['n_shared_snvs_total'],
            'n_matching_snvs_total': summary['n_matching_snvs_total'],
            'n_different_snvs_total': summary['n_different_snvs_total'],
            'snv_distance_mean': summary['snv_distance_sum'] / n_records,
            'abundance_diff_mean': summary['abundance_diff_sum'] / n_records,
            'detected_abundance_mean': summary['detected_abundance_sum'] / n_records,
            'true_abundance_mean': summary['true_abundance_sum'] / n_records,
        })

    summary_fieldnames = [
        'track_id', 'matched_strain', 'lineage_ids', 'n_lineages',
        'n_contigs', 'n_timepoints', 'n_records',
        'n_snvs_detected_total', 'n_snvs_true_total', 'n_shared_snvs_total',
        'n_matching_snvs_total', 'n_different_snvs_total',
        'snv_distance_mean', 'abundance_diff_mean',
        'detected_abundance_mean', 'true_abundance_mean',
    ]
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(summary_records)

    logger.info(f"Wrote {len(records)} lineage detail records to {output_path}")
    logger.info(f"Wrote {len(summary_records)} track summaries to {summary_path}")
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


def write_linking_diagnostics(window_results: List, output_dir: str):
    """
    Write linking_diagnostics.tsv with per-overlap linking decisions.
    """
    import csv

    if not window_results:
        return

    records = []
    for wr in window_results:
        debug_entries = getattr(wr, "linking_debug", None)
        if not debug_entries:
            continue
        records.extend(debug_entries)

    if not records:
        return

    output_path = os.path.join(output_dir, "linking_diagnostics.tsv")
    fieldnames = sorted({k for rec in records for k in rec.keys()})
    try:
        with open(output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
        logger.info(f"Wrote {len(records)} linking diagnostics records to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to write linking_diagnostics.tsv: {e}")


def write_false_positive_reads(
    result: ValidationResult,
    window_results: List,
    output_dir: str,
) -> Optional[str]:
    """
    Write false_positive_reads.tsv with per-read SNV alleles for windows containing FP lineages.

    Requires lineages.tsv to map lineage_id -> track_id. Emits all reads in any window
    that contains a track_id belonging to a false positive lineage.
    """
    import csv

    if not result.false_positives:
        return None

    # lineages.tsv may be in output_dir or its parent (if output_dir is 'validation' subdirectory)
    lineages_file = os.path.join(output_dir, "lineages.tsv")
    if not os.path.exists(lineages_file):
        lineages_file = os.path.join(os.path.dirname(output_dir), "lineages.tsv")
    if not os.path.exists(lineages_file):
        logger.warning("false_positive_reads.tsv not written: lineages.tsv not found.")
        return None

    lineage_to_tracks = defaultdict(set)
    try:
        with open(lineages_file) as f:
            header = f.readline().strip().split("\t")
            if "lineage_id" not in header or "track_id" not in header:
                logger.warning("false_positive_reads.tsv not written: lineages.tsv missing lineage_id/track_id.")
                return None
            lineage_idx = header.index("lineage_id")
            track_idx = header.index("track_id")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) <= max(lineage_idx, track_idx):
                    continue
                lineage_to_tracks[parts[lineage_idx]].add(parts[track_idx])
    except Exception as e:
        logger.warning(f"false_positive_reads.tsv not written: failed to read lineages.tsv ({e}).")
        return None

    output_path = os.path.join(output_dir, "false_positive_reads.tsv")
    fp_ids = set(result.false_positives)

    fieldnames = [
        "fp_lineage_id",
        "fp_track_id",
        "contig",
        "window_start",
        "window_end",
        "sample",
        "read_id",
        "assigned_track_id",
        "hap_id",
        "prob",
        "is_junk",
        "is_ambiguous",
        "snv_alleles",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for wr in window_results:
            # Map hap index -> track_id for this window.
            hap_track_ids = [hap.track_id for hap in wr.haplotypes]
            window_track_ids = {tid for tid in hap_track_ids if tid}

            # Build read assignment map by read_id.
            assignment_by_read = {a["read_id"]: a for a in (wr.assignments or [])}

            # Check which false-positive lineages intersect this window.
            for fp_lineage in fp_ids:
                fp_tracks = lineage_to_tracks.get(fp_lineage, set())
                fp_tracks_in_window = fp_tracks & window_track_ids
                if not fp_tracks_in_window:
                    continue

                for fp_track_id in sorted(fp_tracks_in_window):
                    for read in wr.window.reads:
                        assign = assignment_by_read.get(read.id, {})
                        hap_id = assign.get("hap_id")
                        assigned_track_id = None
                        if hap_id is not None and 0 <= hap_id < len(hap_track_ids):
                            assigned_track_id = hap_track_ids[hap_id]
                        sample = wr.window.sample or read.sample or ""
                        snv_alleles = ",".join(
                            f"{pos}:{base}" for pos, base in sorted(read.alleles.items())
                        )
                        writer.writerow(
                            {
                                "fp_lineage_id": fp_lineage,
                                "fp_track_id": fp_track_id,
                                "contig": wr.window.contig,
                                "window_start": wr.window.start,
                                "window_end": wr.window.end,
                                "sample": sample,
                                "read_id": read.id,
                                "assigned_track_id": assigned_track_id or "",
                                "hap_id": hap_id if hap_id is not None else "",
                                "prob": f"{assign.get('prob', 0.0):.6f}" if assign else "",
                                "is_junk": assign.get("is_junk", ""),
                                "is_ambiguous": assign.get("is_ambiguous", ""),
                                "snv_alleles": snv_alleles,
                            }
                        )

    logger.info(f"Wrote false positive reads report to {output_path}")
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
        'donor_timepoint', 'anchor_distance', 'n_shared_with_anchor', 'reason'
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
                'reason': getattr(stat, "reason", ""),
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

        # Section 2: Accuracy Metrics
        f.write("2. ACCURACY METRICS\n")
        f.write("-" * 80 + "\n")
        f.write("Haplotype-Level Metrics:\n")
        f.write(f"  Precision:  {result.precision:.4f}  (fraction of detected that are correct)\n")
        f.write(f"  Recall:     {result.recall:.4f}  (fraction of true variation covered by haplotypes)\n")
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

        # Section 4b: Window-level recall (informative windows)
        f.write("4b. WINDOW-LEVEL RECALL (INFORMATIVE WINDOWS)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Pooled Window Recall: {result.window_recall:.4f}\n")
        f.write(f"Total Informative Windows: {result.window_informative_total}\n")
        f.write(f"Detected Windows:         {result.window_detected_total}\n")
        if result.window_recall_by_timepoint:
            f.write("Per-Timepoint Window Recall:\n")
            for tp, val in sorted(result.window_recall_by_timepoint.items()):
                f.write(f"  {tp}: {val:.4f}\n")
        if result.window_recall_by_contig:
            f.write("Per-Contig Window Recall:\n")
            for contig, val in sorted(result.window_recall_by_contig.items()):
                f.write(f"  {contig}: {val:.4f}\n")
        f.write("\n")

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


def write_low_abundance_report(
    result: ValidationResult,
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    output_dir: str,
    window_results: Optional[List] = None,
    low_abundance_threshold: float = 0.01,
) -> str:
    """
    Write low_abundance.txt as a missed-SNV report.

    This reports true SNV positions that were not recovered within detected spans,
    or were never in any detected span due to missing haplotypes/coverage.
    """
    output_path = os.path.join(output_dir, "low_abundance.txt")

    matches = match_haplotypes(true_haps, detected_haps)
    matches_by_strain: Dict[str, List[DetectedHaplotype]] = defaultdict(list)
    for true_hap, det_hap, _ in matches:
        matches_by_strain[true_hap.strain_id].append(det_hap)

    missed_records = []
    for true_hap in true_haps:
        det_tracks = matches_by_strain.get(true_hap.strain_id, [])

        # Build detected SNV union and span per contig for this true haplotype.
        detected_snvs_union: Dict[str, Dict[int, str]] = defaultdict(dict)
        detected_span: Dict[str, Tuple[int, int]] = {}
        for det_hap in det_tracks:
            for contig, det_snvs in det_hap.snv_alleles.items():
                for pos, allele in det_snvs.items():
                    if pos not in detected_snvs_union[contig]:
                        detected_snvs_union[contig][pos] = allele
                if det_snvs:
                    min_pos = min(det_snvs.keys())
                    max_pos = max(det_snvs.keys())
                    if contig in detected_span:
                        curr_min, curr_max = detected_span[contig]
                        detected_span[contig] = (min(curr_min, min_pos), max(curr_max, max_pos))
                    else:
                        detected_span[contig] = (min_pos, max_pos)

        for contig, true_snvs in true_hap.snv_positions.items():
            det_snvs = detected_snvs_union.get(contig, {})
            span = detected_span.get(contig)
            for pos, true_allele in true_snvs.items():
                if not det_tracks:
                    reason = "no_detected_haplotype"
                    detected_allele = ""
                elif not span:
                    reason = "no_detected_span"
                    detected_allele = ""
                elif pos < span[0] or pos > span[1]:
                    reason = "outside_detected_span"
                    detected_allele = ""
                else:
                    detected_allele = det_snvs.get(pos, "")
                    if pos not in det_snvs:
                        reason = "missed_snv"
                    elif detected_allele != true_allele:
                        reason = "allele_mismatch"
                    else:
                        continue  # Recovered SNV

                missed_records.append(
                    {
                        "strain_id": true_hap.strain_id,
                        "contig": contig,
                        "pos": pos,
                        "true_allele": true_allele,
                        "detected_allele": detected_allele,
                        "reason": reason,
                    }
                )

    with open(output_path, "w") as f:
        f.write("MISSED SNVs REPORT\n")
        f.write("=" * 80 + "\n")
        f.write("Columns: strain_id, contig, pos, true_allele, detected_allele, reason\n\n")
        if not missed_records:
            f.write("No missed SNVs.\n")
        else:
            for r in missed_records:
                f.write(
                    f"{r['strain_id']}\t{r['contig']}\t{r['pos']}\t"
                    f"{r['true_allele']}\t{r['detected_allele']}\t{r['reason']}\n"
                )

    logger.info(f"Wrote missed SNVs report to {output_path}")
    return output_path


def write_missed_windows_report(
    missed_windows: List[Dict],
    output_dir: str,
) -> Optional[str]:
    """Write missed_windows.txt listing informative windows with missing SNVs."""
    output_path = os.path.join(output_dir, "missed_windows.txt")
    with open(output_path, "w") as f:
        f.write("MISSED INFORMATIVE WINDOWS\n")
        f.write("=" * 80 + "\n")
        f.write("Columns: sample, contig, window_start, window_end, n_informative_snvs, n_missing_snvs, missing_positions\n\n")
        if not missed_windows:
            f.write("None\n")
        else:
            for w in missed_windows:
                f.write(
                    f"{w['sample']}\t{w['contig']}\t{w['window_start']}\t{w['window_end']}\t"
                    f"{w['n_informative_snvs']}\t{w['n_missing_snvs']}\t{w['missing_positions']}\n"
                )
    logger.info(f"Wrote missed windows report to {output_path}")
    return output_path


# =============================================================================
# Main validation pipeline
# =============================================================================

def run_validation(
    detected_file: str,
    truth_dir: str,
    output_dir: str,
    window_results: Optional[List] = None,  # REQUIRED for window-level recall
    window_size: Optional[int] = None,  # Window size for track validation
    detected_without_rescue: Optional[Dict] = None  # Optional abundances without rescue for Δrecall
) -> ValidationResult:
    """Run the full validation pipeline."""

    os.makedirs(output_dir, exist_ok=True)
    true_haps, all_snv_positions = load_ground_truth(truth_dir)
    detected_haps = load_detected_haplotypes(detected_file)

    matches = match_haplotypes(true_haps, detected_haps)
    truth_snvs = {true_hap.strain_id: true_hap.snv_positions for true_hap in true_haps}

    result, missed_windows = _compute_haplotype_metrics(
        true_haps=true_haps,
        detected_haps=detected_haps,
        all_snv_positions=all_snv_positions,
        window_results=window_results,
        matches=matches,
    )

    _validate_tracks_and_lineages(
        result=result,
        matches=matches,
        truth_dir=truth_dir,
        truth_snvs=truth_snvs,
        detected_haps=detected_haps,
        true_haps=true_haps,
        window_results=window_results,
        window_size=window_size,
        detected_without_rescue=detected_without_rescue,
        output_dir=output_dir,
    )

    _write_validation_outputs(
        result=result,
        true_haps=true_haps,
        detected_haps=detected_haps,
        matches=matches,
        window_results=window_results,
        missed_windows=missed_windows,
        output_dir=output_dir,
    )

    _write_validation_reports(
        result=result,
        detected_haps=detected_haps,
        true_haps=true_haps,
        output_dir=output_dir,
    )

    return result


def _compute_haplotype_metrics(
    *,
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    all_snv_positions: Dict[str, List[int]],
    window_results: List,
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
) -> Tuple[ValidationResult, List[Dict[str, Any]]]:
    """
    Compute all haplotype validation metrics using per-window computation.

    Uses the new architecture where all metrics (precision, recall, abundance,
    SNV metrics) are computed per-window first, then aggregated.
    """
    n_true = len(true_haps)
    n_detected = len(detected_haps)

    # Compute per-window metrics and aggregate
    all_window_metrics, overall_agg, by_contig_agg, by_timepoint_agg = compute_all_window_metrics(
        window_results, true_haps
    )

    # Use aggregated metrics for primary validation metrics
    precision = overall_agg.precision
    recall = overall_agg.recall
    f1 = overall_agg.f1
    snv_precision = overall_agg.snv_precision
    snv_recall = overall_agg.snv_recall
    abundance_pearson_r = overall_agg.abundance_pearson_r
    abundance_mae = overall_agg.abundance_mae

    # Convert per-contig aggregated metrics to the expected dict format
    per_contig_metrics = {}
    for contig, agg in by_contig_agg.items():
        per_contig_metrics[contig] = {
            'n_true': agg.total_true,
            'n_detected': agg.total_detected,
            'n_matched': agg.total_matched,
            'n_matched_true': agg.total_matched,  # Same as n_matched in new model
            'n_matched_detected': agg.total_matched,
            'precision': agg.precision,
            'recall': agg.recall,
            'n_windows': agg.n_windows,
            'n_windows_informative': agg.n_windows_informative,
        }

    # Convert per-timepoint aggregated metrics to the expected dict format
    per_timepoint_metrics = {}
    for sample, agg in by_timepoint_agg.items():
        per_timepoint_metrics[sample] = {
            'n_true': agg.total_true,
            'n_detected': agg.total_detected,
            'n_matched': agg.total_matched,
            'n_matched_true': agg.total_matched,
            'n_matched_detected': agg.total_matched,
            'precision': agg.precision,
            'recall': agg.recall,
            'abundance_pearson_r': agg.abundance_pearson_r if agg.n_abundance_pairs >= 2 else None,
            'abundance_mae': agg.abundance_mae if agg.n_abundance_pairs >= 2 else None,
            'n_windows': agg.n_windows,
            'n_windows_informative': agg.n_windows_informative,
        }

    # Build window recall by timepoint/contig (for backward compatibility)
    window_recall_by_timepoint = {
        sample: agg.recall for sample, agg in by_timepoint_agg.items()
    }
    window_recall_by_contig = {
        contig: agg.recall for contig, agg in by_contig_agg.items()
    }

    # Build missed_windows list from window metrics with recall < 1
    missed_windows = []
    for wm in all_window_metrics:
        if wm.recall < 1.0 and wm.n_strain_groups > 0:
            missed_windows.append({
                'sample': wm.sample,
                'contig': wm.contig,
                'window_start': wm.window_start,
                'window_end': wm.window_end,
                'n_informative_snvs': wm.n_informative_snvs,
                'n_missing_snvs': len(wm.unmatched_true),
                'missing_positions': ','.join(wm.unmatched_true),
            })

    false_negatives = [
        f"{w['sample']}|{w['contig']}:{w['window_start']}-{w['window_end']}"
        for w in missed_windows
    ]

    # False positives: detected lineages not matching any truth (from global matches)
    matched_detected_ids = {m[1].lineage_id for m in matches}
    false_positives = [h.lineage_id for h in detected_haps if h.lineage_id not in matched_detected_ids]

    # Per-match diagnostics from global matching (still needed for detailed reports)
    match_details_full = _build_match_details_full(matches, true_haps)
    detection_threshold, _ = compute_detection_sensitivity(true_haps, matches)
    match_details = _dedup_match_details(matches)

    # Build result using window-aggregated metrics
    result = ValidationResult(
        n_true=n_true,
        n_detected=n_detected,
        n_matched=overall_agg.total_matched,
        precision=precision,
        recall=recall,
        f1=f1,
        abundance_pearson_r=abundance_pearson_r,
        abundance_mae=abundance_mae,
        snv_precision=snv_precision,
        snv_recall=snv_recall,
        phasing_accuracy=snv_recall,  # Use SNV recall as phasing accuracy
        snv_true_total=overall_agg.total_true,
        snv_true_in_span=overall_agg.total_snv_correct + (overall_agg.total_snv_detected - overall_agg.total_snv_correct),
        snv_detected_total=overall_agg.total_snv_detected,
        snv_correct_total=overall_agg.total_snv_correct,
        snv_span_coverage_frac=1.0 if overall_agg.n_windows > 0 else 0.0,  # All windows are in span
        detection_threshold=detection_threshold,
        matches=match_details,
        false_negatives=false_negatives,
        false_positives=false_positives,
        match_details_full=match_details_full,
        per_contig_metrics=per_contig_metrics,
        per_timepoint_metrics=per_timepoint_metrics,
        window_recall=recall,  # Same as overall recall in new model
        window_informative_total=overall_agg.n_windows_informative,
        window_detected_total=overall_agg.total_matched,
        window_recall_by_timepoint=window_recall_by_timepoint,
        window_recall_by_contig=window_recall_by_contig,
    )

    return result, missed_windows


def _validate_tracks_and_lineages(
    *,
    result: ValidationResult,
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    truth_dir: str,
    truth_snvs: Dict[str, Dict[str, Dict[int, str]]],
    detected_haps: List[DetectedHaplotype],
    true_haps: List[TrueHaplotype],
    window_results: List,
    window_size: int,
    detected_without_rescue: Optional[Dict],
    output_dir: str,
) -> None:
    lineage_to_strain = {det_hap.lineage_id: true_hap.strain_id
                         for true_hap, det_hap, _ in matches if det_hap.lineage_id}
    strain_matches = dict(lineage_to_strain)
    for det_hap in detected_haps:
        if det_hap.track_id and det_hap.lineage_id in lineage_to_strain:
            strain_matches[det_hap.track_id] = lineage_to_strain[det_hap.lineage_id]

    logger.info(f"Built track mapping: {len(lineage_to_strain)} lineage->strain, "
                f"{len(strain_matches)} total track->strain mappings")

    from validation.validate_tracks import validate_tracks, write_linkability_report
    logger.info(f"Running track validation with {len(window_results)} window results "
                f"({len(strain_matches)} strain mappings)")
    track_result = validate_tracks(
        window_results, truth_dir, strain_matches, truth_snvs, window_size
    )
    result.track_fragmentation_mean = track_result.track_fragmentation_mean
    result.track_fragmentation_median = track_result.track_fragmentation_median
    result.false_link_rate = track_result.false_link_rate
    result.missed_link_rate = track_result.missed_link_rate
    result.track_consensus_error = track_result.track_consensus_error
    logger.info(
        f"Track validation complete: fragmentation={result.track_fragmentation_mean:.3f}, "
        f"false_link={result.false_link_rate:.3f}, missed_link={result.missed_link_rate:.3f}"
    )

    if track_result.linkability_analysis:
        linkability_path = os.path.join(output_dir, 'track_linkability.txt')
        write_linkability_report(track_result.linkability_analysis, linkability_path)
        logger.info(f"Wrote track linkability report to {linkability_path}")

    from validation.validate_lineages import validate_lineages
    detected_lineages: Dict[str, Dict[str, str]] = {}
    for true_hap, det_match, _ in matches:
        if not det_match.lineage_id:
            continue
        for contig in det_match.snv_alleles.keys():
            if contig not in true_hap.snv_positions:
                continue
            detected_lineages.setdefault(det_match.lineage_id, {})[contig] = true_hap.strain_id

    n_pairs = sum(len(c) for c in detected_lineages.values())
    logger.info(f"Built detected_lineages: {len(detected_lineages)} lineages, {n_pairs} pairs")

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


def _write_validation_reports(
    *,
    result: ValidationResult,
    detected_haps: List[DetectedHaplotype],
    true_haps: List[TrueHaplotype],
    output_dir: str,
) -> None:
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
            'track_fragmentation_mean': result.track_fragmentation_mean,
            'track_fragmentation_median': result.track_fragmentation_median,
            'false_link_rate': result.false_link_rate,
            'missed_link_rate': result.missed_link_rate,
            'track_consensus_error': result.track_consensus_error,
            'lineage_precision': result.lineage_precision,
            'lineage_recall': result.lineage_recall,
            'lineage_f1': result.lineage_f1,
            'rescue_delta_recall_rare': result.rescue_delta_recall_rare,
            'abundance_trajectory_error': result.abundance_trajectory_error,
            'window_recall': result.window_recall,
            'window_informative_total': result.window_informative_total,
            'window_detected_total': result.window_detected_total,
            'window_recall_by_timepoint': result.window_recall_by_timepoint,
            'window_recall_by_contig': result.window_recall_by_contig,
            'false_negatives': result.false_negatives,
            'false_positives': result.false_positives,
            'per_contig_metrics': result.per_contig_metrics,
            'per_timepoint_metrics': result.per_timepoint_metrics,
        }, f, indent=2)

    report_file = os.path.join(output_dir, 'detailed_report.txt')
    with open(report_file, 'w') as f:
        lines = []
        def add(line: str = "") -> None:
            lines.append(line)

        add("=" * 80)
        add("DETAILED VALIDATION REPORT")
        add("=" * 80)
        add("")
        add("SUMMARY METRICS")
        add("-" * 80)
        n_contigs = len(result.per_contig_metrics) if result.per_contig_metrics else 1
        all_timepoints_list = sorted(result.per_timepoint_metrics.keys()) if result.per_timepoint_metrics else []
        n_timepoints = len(all_timepoints_list)
        total_contig_timepoint_pairs = n_contigs * n_timepoints if n_timepoints > 0 else n_contigs

        matched_true_ids = set()
        matched_detected_ids = set()
        if result.matches:
            for true_ids_csv, detected_id, _ in result.matches:
                matched_detected_ids.add(detected_id)
                for true_id in true_ids_csv.split(","):
                    if true_id:
                        matched_true_ids.add(true_id)

        add(f"True strains (per genome):     {result.n_true}")
        add(f"Contigs evaluated:            {n_contigs}")
        add(f"Timepoints evaluated:         {n_timepoints}")
        add(f"Total contig-timepoint pairs: {total_contig_timepoint_pairs}")
        add(f"Detected lineages (total):    {result.n_detected}")
        add(f"Matched lineages:             {result.n_matched}")
        add(f"Matched true strains:         {len(matched_true_ids)}")
        add(f"Matched detected lineages:    {len(matched_detected_ids)}")
        add("")
        add("(Note: Strainphase splits lineages per-contig, so one strain can produce multiple detected lineages per contig)")
        add(f"Precision:           {result.precision:.3f}")
        add(f"Recall:              {result.recall:.3f}  (window-level when available)")
        add(f"F1 Score:            {result.f1:.3f}")
        add(f"Window Recall:       {result.window_recall:.3f}")
        add(f"Abundance Pearson r: {result.abundance_pearson_r:.3f}")
        add(f"Abundance MAE:       {result.abundance_mae:.3f}")
        add(f"SNV Precision:       {result.snv_precision:.3f}")
        add(f"SNV Recall:          {result.snv_recall:.3f}")
        add(f"Detection Threshold: {result.detection_threshold:.4f}")
        add("")
        add("FALSE NEGATIVES (Informative windows not detected)")
        add("-" * 80)
        if result.false_negatives:
            for fn_id in result.false_negatives:
                add(f"  {fn_id}")
        else:
            add("  None")
        add("")
        add("FALSE POSITIVES (Detected lineages not matching truth)")
        add("-" * 80)
        if result.false_positives:
            for fp_id in result.false_positives:
                fp_hap = next((h for h in detected_haps if h.lineage_id == fp_id), None)
                if fp_hap:
                    max_abund = max(fp_hap.abundances.values()) if fp_hap.abundances else 0
                    n_snvs = sum(len(snvs) for snvs in fp_hap.snv_alleles.values())
                    contigs = list(fp_hap.snv_alleles.keys())
                    add("")
                    add(f"  {fp_id}: max_abundance={max_abund:.4f}, n_snvs={n_snvs}, contigs={contigs}")
                    add("    Why it doesn't match:")
                    for true_hap in true_haps:
                        dist, n_matches, n_shared, match_fraction = compute_haplotype_distance(true_hap, fp_hap)
                        n_mismatches = n_shared - n_matches
                        reasons = []
                        if n_shared < 3:
                            reasons.append(f"too few shared SNVs ({n_shared} < 3)")
                        if match_fraction < 0.9:
                            reasons.append(f"low match fraction ({match_fraction:.3f} < 0.9)")
                        if dist > 0.1:
                            reasons.append(f"distance too high ({dist:.3f} > 0.1)")

                        line = f"      vs {true_hap.strain_id}: "
                        if reasons:
                            line += "; ".join(reasons)
                            line += f"; distance={dist:.3f}, shared={n_shared}, matches={n_matches}, mismatches={n_mismatches}"
                        else:
                            line += (f"distance={dist:.3f}, shared={n_shared}, matches={n_matches}, "
                                     f"mismatches={n_mismatches}, match_frac={match_fraction:.3f}")
                        add(line)
        else:
            add("  None")
        add("")
        add("DETAILED MATCH INFORMATION")
        add("-" * 80)
        for match in result.match_details_full:
            add("")
            add(f"True Strain: {match['true_strain_id']}")
            add(f"  → Detected Lineage: {match['detected_lineage_id']}")
            add(f"  Distance: {match['distance']:.4f}")
            add(f"  SNVs: true={match['n_true_snvs']}, detected={match['n_detected_snvs']}, "
                f"shared={match['n_shared_snvs']}, matching={match['n_matching_snvs']}")
            add(f"  SNV Match Fraction: {match['snv_match_fraction']:.3f}")
            if match['grouped_abundance_mae'] is not None:
                add(f"  Grouped Abundance MAE: {match['grouped_abundance_mae']:.4f}")
            add(f"  Timepoints: {', '.join(match['common_timepoints'])}")
            if match['grouped_true_abundances']:
                add(f"  Grouped true abundances: {match['grouped_true_abundances']}")
                add(f"  Detected abundances: {match['detected_abundances']}")
            add(f"  Is sweeping: {match['is_sweeping']}")
        add("")
        add("PER-CONTIG METRICS")
        add("-" * 80)
        for contig, metrics in sorted(result.per_contig_metrics.items()):
            add(f"{contig}:")
            add(f"  True: {metrics['n_true']}, Detected: {metrics['n_detected']}, "
                f"Matched: {metrics['n_matched']}")
            add(f"  Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}")
        add("")
        add("PER-TIMEPOINT METRICS")
        add("-" * 80)
        for tp, metrics in sorted(result.per_timepoint_metrics.items()):
            add(f"{tp}:")
            add(f"  True: {metrics['n_true']}, Detected: {metrics['n_detected']}, "
                f"Matched: {metrics['n_matched']}")
            add(f"  Precision: {metrics['precision']:.3f}, "
                f"Recall: {metrics['recall']:.3f}")
            if metrics['abundance_pearson_r'] is not None:
                add(f"  Abundance r: {metrics['abundance_pearson_r']:.3f}, "
                    f"MAE: {metrics['abundance_mae']:.4f}")
        add("")
        add("=" * 80)
        f.write("\n".join(lines) + "\n")

    print("")
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    n_contigs = len(result.per_contig_metrics) if result.per_contig_metrics else 1
    all_timepoints_list = sorted(result.per_timepoint_metrics.keys()) if result.per_timepoint_metrics else []
    n_timepoints = len(all_timepoints_list)
    total_contig_timepoint_pairs = n_contigs * n_timepoints if n_timepoints > 0 else n_contigs

    matched_true_ids = set()
    matched_detected_ids = set()
    if result.matches:
        for true_ids_csv, detected_id, _ in result.matches:
            matched_detected_ids.add(detected_id)
            for true_id in true_ids_csv.split(","):
                if true_id:
                    matched_true_ids.add(true_id)

    print(f"True strains (per genome):     {result.n_true}")
    print(f"Contigs evaluated:            {n_contigs}")
    print(f"Timepoints evaluated:         {n_timepoints}")
    print(f"Total contig-timepoint pairs: {total_contig_timepoint_pairs}")
    print(f"Detected lineages (total):    {result.n_detected}")
    print(f"Matched lineages:             {result.n_matched}")
    print(f"Matched true strains:         {len(matched_true_ids)}")
    print(f"Matched detected lineages:    {len(matched_detected_ids)}")
    print("-" * 60)

    if result.per_timepoint_metrics and result.per_contig_metrics:
        print("")
        print("BREAKDOWN BY TIMEPOINT → CONTIG:")
        print("-" * 60)
        for tp in sorted(result.per_timepoint_metrics.keys()):
            tp_metrics = result.per_timepoint_metrics[tp]
            print("")
            print(f"{tp}:")
            print(f"  Overall: {tp_metrics['n_true']} true, {tp_metrics['n_detected']} detected, "
                  f"{tp_metrics['n_matched_true']} matched true, {tp_metrics['n_matched_detected']} matched detected")
            print(f"  Precision: {tp_metrics['precision']:.3f}, Recall: {tp_metrics['recall']:.3f}")
            print("  Per-contig:")
            for contig, contig_metrics in sorted(result.per_contig_metrics.items()):
                contig_short = contig.split('.')[-1] if '.' in contig else contig
                print(f"    {contig_short}: {contig_metrics['n_true']} true, "
                      f"{contig_metrics['n_detected']} detected, "
                      f"{contig_metrics['n_matched_true']} matched true")
    elif result.per_contig_metrics:
        print("")
        print("BREAKDOWN BY CONTIG:")
        print("-" * 60)
        for contig, metrics in sorted(result.per_contig_metrics.items()):
            contig_short = contig.split('.')[-1] if '.' in contig else contig
            print(f"{contig_short}:")
            print(f"  True haplotypes:   {metrics['n_true']}")
            print(f"  Detected lineages: {metrics['n_detected']}")
            print(f"  Matched true:      {metrics['n_matched_true']}")
            print(f"  Matched detected:  {metrics['n_matched_detected']}")
            print(f"  Precision:         {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")

    print("")
    print("OVERALL METRICS:")
    print("-" * 60)
    print(f"Precision:           {result.precision:.3f}")
    print(f"Recall:              {result.recall:.3f}")
    print(f"F1 Score:            {result.f1:.3f}")
    print("-" * 60)
    print("")
    print("ABUNDANCE METRICS:")
    print("-" * 60)
    print(f"Abundance Pearson r: {result.abundance_pearson_r:.3f}")
    print(f"Abundance MAE:       {result.abundance_mae:.3f}")
    print("")
    print("SNV METRICS:")
    print("-" * 60)
    print(f"SNV Precision:       {result.snv_precision:.3f}")
    print(f"SNV Recall:          {result.snv_recall:.3f}")
    print(f"Phasing Accuracy:    {result.phasing_accuracy:.3f}")
    print(f"Detection Threshold: {result.detection_threshold:.4f}")
    print("=" * 60)

    if result.false_negatives or result.false_positives:
        print("")
        print("ERROR BREAKDOWN:")
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
                    best_dist = 1.0
                    best_strain = None
                    best_n_matches = 0
                    best_n_shared = 0
                    for true_hap in true_haps:
                        dist, n_matches, n_shared, _ = compute_haplotype_distance(true_hap, fp_hap)
                        if dist < best_dist:
                            best_dist = dist
                            best_strain = true_hap.strain_id
                            best_n_matches = n_matches
                            best_n_shared = n_shared
                    if best_strain:
                        n_mismatches = best_n_shared - best_n_matches
                        print(f"    Closest to {best_strain}: distance={best_dist:.3f}, "
                              f"shared_snvs={best_n_shared}, matches={best_n_matches}, mismatches={n_mismatches}")


def _write_validation_outputs(
    *,
    result: ValidationResult,
    true_haps: List[TrueHaplotype],
    detected_haps: List[DetectedHaplotype],
    matches: List[Tuple[TrueHaplotype, DetectedHaplotype, float]],
    window_results: List,
    missed_windows: List[Dict[str, Any]],
    output_dir: str,
) -> None:
    """Write validation TSV/text outputs."""
    # Generate detailed TSV output files
    try:
        write_lineage_details(true_haps, detected_haps, matches, output_dir)
    except Exception as e:
        logger.warning(f"Failed to write lineage_details.tsv: {e}")

    try:
        write_linking_quality(detected_haps, matches, output_dir)
    except Exception as e:
        logger.warning(f"Failed to write linking_quality.tsv: {e}")

    try:
        write_linking_diagnostics(window_results, output_dir)
    except Exception as e:
        logger.warning(f"Failed to write linking_diagnostics.tsv: {e}")

    try:
        write_em_convergence(window_results, output_dir)
    except Exception as e:
        logger.warning(f"Failed to write em_convergence.tsv: {e}")

    # NOTE: rescue_statistics.tsv is written by parameter_sweep.py
    # via rescue_integrator.write_rescue_statistics() which has access
    # to the LongitudinalIntegrator's rescue statistics.
    # We skip writing here to avoid creating an empty file that would
    # overwrite or confuse the actual rescue statistics.

    if missed_windows:
        try:
            write_missed_windows_report(missed_windows, output_dir)
        except Exception as e:
            logger.warning(f"Failed to write missed_windows.txt: {e}")

    try:
        write_validation_summary(result, true_haps, detected_haps, matches, output_dir, window_results)
    except Exception as e:
        logger.warning(f"Failed to write validation_summary.txt: {e}")

    try:
        write_low_abundance_report(result, true_haps, detected_haps, output_dir, window_results)
    except Exception as e:
        logger.warning(f"Failed to write low_abundance.txt: {e}")

    try:
        write_false_positive_reads(result, window_results, output_dir)
    except Exception as e:
        logger.warning(f"Failed to write false_positive_reads.tsv: {e}")


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
