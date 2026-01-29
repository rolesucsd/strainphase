#!/usr/bin/env python3
"""
Track and linking validation metrics for strainphase.

This is a library module used by validate_haplotypes.py. It is not meant to be
run standalone - use validate_haplotypes.py or the benchmarking pipeline instead.

Validates:
- Track fragmentation (how many inferred tracks per true strain)
- False link rate (links joining haplotypes from different strains)
- Missed link rate (true adjacent-window links not recovered)
- Track consensus error (mismatch between inferred and true consensus)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from strainphase.core import WindowResult, Haplotype

logger = logging.getLogger(__name__)

# Import shared truth loading function to avoid duplication
from validation.validate_lineages import load_truth_lineages


@dataclass
class TrackLinkabilityInfo:
    """Information about whether two tracks could potentially be linked."""
    track_id_1: str
    track_id_2: str
    strain_id: str
    contig: str
    track1_span: Tuple[int, int]  # (start, end)
    track2_span: Tuple[int, int]  # (start, end)
    gap_bp: int  # Gap in base pairs between tracks (0 if overlapping)
    track1_snv_positions: List[int]
    track2_snv_positions: List[int]
    shared_snv_positions: List[int]  # SNVs in overlapping region
    is_linkable: bool  # True if tracks share SNV positions
    linkability_reason: str  # Explanation


@dataclass
class TrackValidationResult:
    """Results of track/linking validation."""
    track_fragmentation_mean: float
    track_fragmentation_median: float
    false_link_rate: float
    missed_link_rate: float
    track_consensus_error: float
    per_strain_fragmentation: Dict[str, Dict[str, int]]  # strain_id -> contig -> n_tracks
    false_links: List[Tuple[str, str, str]]  # (track_id1, track_id2, contig)
    missed_links: List[Tuple[str, str, str]]  # (strain_id, contig, window_pair)
    linkability_analysis: Optional[List[TrackLinkabilityInfo]] = None  # Track gap analysis


def load_truth_tracks(truth_dir: str) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    Load truth tracks from truth_tracks.tsv.
    
    Returns: {strain_id -> {contig -> (start, end)}}
    """
    truth_path = Path(truth_dir)
    tracks_file = truth_path / "truth_tracks.tsv"
    
    if not tracks_file.exists():
        logger.warning(f"truth_tracks.tsv not found at {tracks_file}")
        logger.warning(f"  Looking in: {truth_path.absolute()}")
        logger.warning(f"  Files in truth_dir: {list(truth_path.glob('*.tsv')) if truth_path.exists() else 'directory does not exist'}")
        return {}
    
    truth_tracks = defaultdict(dict)
    with open(tracks_file) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            row = dict(zip(header, parts))
            strain_id = row['strain_id']
            contig = row['contig']
            start = int(row['start'])
            end = int(row['end'])
            truth_tracks[strain_id][contig] = (start, end)
    
    logger.info(f"Loaded {len(truth_tracks)} truth tracks")
    return dict(truth_tracks)




def compute_track_fragmentation(
    window_results: List[WindowResult],
    truth_tracks: Dict[str, Dict[str, Tuple[int, int]]],
    strain_matches: Dict[str, str]  # detected_track_id -> true_strain_id
) -> Tuple[float, float, Dict[str, Dict[str, int]]]:
    """
    Compute track fragmentation: number of inferred tracks per true strain per contig.
    
    Returns: (mean_fragmentation, median_fragmentation, per_strain_fragmentation)
    """
    # Group detected tracks by true strain and contig
    strain_contig_tracks: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    
    # Debug: collect statistics
    total_haps = 0
    haps_with_track_id = 0
    haps_matched = 0
    sample_track_ids = set()
    
    for wr in window_results:
        contig = wr.window.contig
        for hap in wr.haplotypes:
            total_haps += 1
            if hap.track_id:
                haps_with_track_id += 1
                sample_track_ids.add(hap.track_id)
                if hap.track_id in strain_matches:
                    haps_matched += 1
                    true_strain_id = strain_matches[hap.track_id]
                    strain_contig_tracks[true_strain_id][contig].add(hap.track_id)
    
    logger.info(f"Track fragmentation debug: {total_haps} total haplotypes, "
                f"{haps_with_track_id} with track_id, {haps_matched} matched to strain")
    
    # Show sample track_ids if no matches
    if haps_with_track_id > 0 and haps_matched == 0:
        sample_hap_tracks = list(sample_track_ids)[:5]
        sample_match_keys = list(strain_matches.keys())[:5]
        logger.warning(f"No track_id matches found!")
        logger.warning(f"  Sample haplotype track_ids: {sample_hap_tracks}")
        logger.warning(f"  Sample strain_matches keys: {sample_match_keys}")
    
    # Count tracks per strain/contig
    fragmentation_counts = []
    per_strain_fragmentation = {}
    
    for strain_id, contig_tracks in strain_contig_tracks.items():
        per_strain_fragmentation[strain_id] = {}
        for contig, track_ids in contig_tracks.items():
            n_tracks = len(track_ids)
            fragmentation_counts.append(n_tracks)
            per_strain_fragmentation[strain_id][contig] = n_tracks
    
    if not fragmentation_counts:
        logger.info("No fragmentation data collected - returning zeros")
        return 0.0, 0.0, {}
    
    mean_frag = np.mean(fragmentation_counts)
    median_frag = np.median(fragmentation_counts)
    
    logger.info(f"Track fragmentation: {len(fragmentation_counts)} strain/contig pairs, "
                f"mean={mean_frag:.3f}, median={median_frag:.3f}")
    
    return mean_frag, median_frag, per_strain_fragmentation


def compute_linking_metrics(
    window_results: List[WindowResult],
    truth_tracks: Dict[str, Dict[str, Tuple[int, int]]],
    strain_matches: Dict[str, str],  # detected_track_id -> true_strain_id
    window_size: int
) -> Tuple[float, float, List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    Compute false link rate and missed link rate.
    
    Returns: (false_link_rate, missed_link_rate, false_links, missed_links)
    """
    # Sort windows by position
    sorted_results = sorted(window_results, key=lambda wr: wr.window.start)
    
    # Build detected links: (track_id1, track_id2) pairs in adjacent windows
    detected_links: Set[Tuple[str, str]] = set()
    track_to_windows: Dict[str, List[int]] = defaultdict(list)
    
    for i, wr in enumerate(sorted_results):
        for hap in wr.haplotypes:
            if hap.track_id:
                track_to_windows[hap.track_id].append(i)
    
    # Find links between adjacent windows (50% overlap)
    step_size = window_size // 2
    false_links = []
    total_detected_links = 0
    
    for i in range(len(sorted_results) - 1):
        curr_wr = sorted_results[i]
        next_wr = sorted_results[i + 1]
        
        # Check if windows overlap (50% overlap expected)
        if next_wr.window.start < curr_wr.window.end:
            # Windows overlap - check for links
            curr_tracks = {hap.track_id for hap in curr_wr.haplotypes if hap.track_id}
            next_tracks = {hap.track_id for hap in next_wr.haplotypes if hap.track_id}
            
            for track1 in curr_tracks:
                for track2 in next_tracks:
                    if track1 == track2:
                        total_detected_links += 1
                        detected_links.add((track1, track2))
                        
                        # Check if this is a false link (different true strains)
                        strain1 = strain_matches.get(track1)
                        strain2 = strain_matches.get(track2)
                        if strain1 and strain2 and strain1 != strain2:
                            false_links.append((track1, track2, curr_wr.window.contig))
    
    false_link_rate = len(false_links) / total_detected_links if total_detected_links > 0 else 0.0
    
    # Compute missed links: true adjacent-window links not recovered
    # For each true strain/contig, check if adjacent windows should be linked
    missed_links = []
    total_expected_links = 0
    
    for strain_id, contig_tracks in truth_tracks.items():
        for contig, (start, end) in contig_tracks.items():
            # Find windows that overlap this strain's track
            strain_windows = []
            for i, wr in enumerate(sorted_results):
                if wr.window.contig == contig:
                    # Check if window overlaps strain track
                    if wr.window.start < end and wr.window.end > start:
                        strain_windows.append(i)
            
            # Check for missed links between adjacent windows
            for j in range(len(strain_windows) - 1):
                w1_idx = strain_windows[j]
                w2_idx = strain_windows[j + 1]
                
                if w2_idx == w1_idx + 1:  # Adjacent windows
                    total_expected_links += 1
                    w1_tracks = {hap.track_id for hap in sorted_results[w1_idx].haplotypes 
                                if hap.track_id and strain_matches.get(hap.track_id) == strain_id}
                    w2_tracks = {hap.track_id for hap in sorted_results[w2_idx].haplotypes 
                                if hap.track_id and strain_matches.get(hap.track_id) == strain_id}
                    
                    # Check if there's a link between these windows
                    linked = bool(w1_tracks & w2_tracks)
                    if not linked and w1_tracks and w2_tracks:
                        missed_links.append((strain_id, contig, f"w{w1_idx}-w{w2_idx}"))
    
    missed_link_rate = len(missed_links) / total_expected_links if total_expected_links > 0 else 0.0
    
    return false_link_rate, missed_link_rate, false_links, missed_links


def compute_track_consensus_error(
    window_results: List[WindowResult],
    truth_snvs: Dict[str, Dict[str, Dict[int, str]]],  # strain_id -> contig -> pos -> allele
    strain_matches: Dict[str, str]  # detected_track_id -> true_strain_id
) -> float:
    """
    Compute mismatch fraction between inferred track consensus and truth at shared SNV sites.
    
    Returns: mean error rate across all tracks
    """
    # Build track consensus: track_id -> {contig -> {pos -> allele}}
    track_consensus: Dict[str, Dict[str, Dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    
    for wr in window_results:
        contig = wr.window.contig
        for hap in wr.haplotypes:
            if hap.track_id:
                for pos, allele in hap.consensus.items():
                    # Use most common allele across windows (simple consensus)
                    if pos not in track_consensus[hap.track_id][contig]:
                        track_consensus[hap.track_id][contig][pos] = allele
    
    # Compare to truth
    total_shared_positions = 0
    total_mismatches = 0
    
    for track_id, true_strain_id in strain_matches.items():
        if true_strain_id not in truth_snvs:
            continue
        
        track_snvs = track_consensus.get(track_id, {})
        true_snvs = truth_snvs[true_strain_id]
        
        for contig, true_contig_snvs in true_snvs.items():
            track_contig_snvs = track_snvs.get(contig, {})
            
            for pos, true_allele in true_contig_snvs.items():
                if pos in track_contig_snvs:
                    total_shared_positions += 1
                    if track_contig_snvs[pos] != true_allele:
                        total_mismatches += 1
    
    error_rate = total_mismatches / total_shared_positions if total_shared_positions > 0 else 0.0
    return error_rate


def analyze_track_linkability(
    window_results: List[WindowResult],
    strain_matches: Dict[str, str],  # detected_track_id -> true_strain_id
) -> List[TrackLinkabilityInfo]:
    """
    Analyze whether fragmented tracks from the same strain could potentially be linked.
    
    For each pair of tracks belonging to the same true strain on the same contig,
    determines if they:
    1. Overlap in genomic position (could potentially share SNVs)
    2. Have shared SNV positions (can be linked by consensus comparison)
    3. Have a gap that makes linking impossible
    
    Returns: List of TrackLinkabilityInfo for each track pair
    """
    # Group tracks by (strain_id, contig)
    # track_info: (strain_id, contig) -> [(track_id, min_pos, max_pos, snv_positions)]
    track_info: Dict[Tuple[str, str], List[Tuple[str, int, int, List[int]]]] = defaultdict(list)
    
    # Build track consensus positions per track
    track_snv_positions: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    
    for wr in window_results:
        contig = wr.window.contig
        for hap in wr.haplotypes:
            if hap.track_id and hap.track_id in strain_matches:
                for pos in hap.consensus.keys():
                    if pos not in track_snv_positions[hap.track_id][contig]:
                        track_snv_positions[hap.track_id][contig].append(pos)
    
    # Build track spans
    for track_id, contig_positions in track_snv_positions.items():
        strain_id = strain_matches.get(track_id)
        if not strain_id:
            continue
        
        for contig, positions in contig_positions.items():
            if positions:
                positions_sorted = sorted(positions)
                min_pos = positions_sorted[0]
                max_pos = positions_sorted[-1]
                track_info[(strain_id, contig)].append(
                    (track_id, min_pos, max_pos, positions_sorted)
                )
    
    # Analyze pairs
    linkability_results = []
    
    for (strain_id, contig), tracks in track_info.items():
        if len(tracks) < 2:
            continue  # No fragmentation to analyze
        
        # Sort tracks by start position
        tracks_sorted = sorted(tracks, key=lambda x: x[1])
        
        # Analyze each adjacent pair
        for i in range(len(tracks_sorted) - 1):
            track1_id, track1_min, track1_max, track1_positions = tracks_sorted[i]
            track2_id, track2_min, track2_max, track2_positions = tracks_sorted[i + 1]
            
            # Calculate gap
            if track2_min > track1_max:
                gap_bp = track2_min - track1_max
            else:
                gap_bp = 0  # Overlapping
            
            # Find shared SNV positions (positions that exist in both tracks)
            positions_set1 = set(track1_positions)
            positions_set2 = set(track2_positions)
            shared_positions = sorted(positions_set1 & positions_set2)
            
            # Determine linkability
            if shared_positions:
                is_linkable = True
                reason = f"Linkable: {len(shared_positions)} shared SNV positions"
            elif gap_bp == 0:
                # Overlapping spans but no shared SNVs
                is_linkable = False
                overlap_start = max(track1_min, track2_min)
                overlap_end = min(track1_max, track2_max)
                reason = f"Unlinkable: Spans overlap ({overlap_start}-{overlap_end}) but no shared SNVs"
            else:
                is_linkable = False
                reason = f"Unlinkable: {gap_bp:,}bp gap between tracks (no SNV overlap possible)"
            
            linkability_results.append(TrackLinkabilityInfo(
                track_id_1=track1_id,
                track_id_2=track2_id,
                strain_id=strain_id,
                contig=contig,
                track1_span=(track1_min, track1_max),
                track2_span=(track2_min, track2_max),
                gap_bp=gap_bp,
                track1_snv_positions=track1_positions,
                track2_snv_positions=track2_positions,
                shared_snv_positions=shared_positions,
                is_linkable=is_linkable,
                linkability_reason=reason
            ))
    
    return linkability_results


def write_linkability_report(
    linkability_analysis: List[TrackLinkabilityInfo],
    output_path: str
) -> None:
    """Write a human-readable linkability report."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRACK LINKABILITY ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("This report analyzes whether fragmented tracks from the same strain\n")
        f.write("could potentially be linked based on shared SNV positions.\n\n")
        
        if not linkability_analysis:
            f.write("No track pairs to analyze (no fragmentation detected).\n")
            return
        
        # Group by strain
        by_strain: Dict[str, List[TrackLinkabilityInfo]] = defaultdict(list)
        for info in linkability_analysis:
            by_strain[info.strain_id].append(info)
        
        # Summary statistics
        total_pairs = len(linkability_analysis)
        linkable_pairs = sum(1 for info in linkability_analysis if info.is_linkable)
        unlinkable_gap = sum(1 for info in linkability_analysis if not info.is_linkable and info.gap_bp > 0)
        unlinkable_no_shared = sum(1 for info in linkability_analysis if not info.is_linkable and info.gap_bp == 0)
        
        f.write("-" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total track pairs analyzed: {total_pairs}\n")
        f.write(f"  Linkable (shared SNVs):   {linkable_pairs} ({100*linkable_pairs/total_pairs:.1f}%)\n")
        f.write(f"  Unlinkable (gap):         {unlinkable_gap} ({100*unlinkable_gap/total_pairs:.1f}%)\n")
        f.write(f"  Unlinkable (no shared):   {unlinkable_no_shared} ({100*unlinkable_no_shared/total_pairs:.1f}%)\n\n")
        
        # Interpretation
        if linkable_pairs == total_pairs:
            f.write("INTERPRETATION: All track pairs have shared SNVs and COULD be linked.\n")
            f.write("Fragmentation is due to algorithm limitations, not physical gaps.\n\n")
        elif linkable_pairs == 0:
            f.write("INTERPRETATION: No track pairs share SNV positions.\n")
            f.write("Fragmentation is UNAVOIDABLE due to sparse SNV distribution.\n\n")
        else:
            f.write(f"INTERPRETATION: {linkable_pairs}/{total_pairs} pairs could be linked.\n")
            f.write("Some fragmentation is avoidable, some is due to SNV gaps.\n\n")
        
        # Detail per strain
        f.write("-" * 80 + "\n")
        f.write("DETAIL BY STRAIN\n")
        f.write("-" * 80 + "\n\n")
        
        for strain_id in sorted(by_strain.keys()):
            pairs = by_strain[strain_id]
            f.write(f"Strain: {strain_id}\n")
            f.write(f"  Track pairs: {len(pairs)}\n")
            
            for info in sorted(pairs, key=lambda x: x.track1_span[0]):
                f.write(f"\n  {info.track_id_1} <-> {info.track_id_2} ({info.contig})\n")
                f.write(f"    Track 1: {info.track1_span[0]:,}-{info.track1_span[1]:,} "
                       f"({len(info.track1_snv_positions)} SNVs)\n")
                f.write(f"    Track 2: {info.track2_span[0]:,}-{info.track2_span[1]:,} "
                       f"({len(info.track2_snv_positions)} SNVs)\n")
                if info.gap_bp > 0:
                    f.write(f"    Gap: {info.gap_bp:,} bp\n")
                else:
                    f.write(f"    Overlap: spans overlap\n")
                f.write(f"    Shared SNVs: {len(info.shared_snv_positions)}\n")
                f.write(f"    Status: {info.linkability_reason}\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF LINKABILITY REPORT\n")
        f.write("=" * 80 + "\n")


def validate_tracks(
    window_results: List[WindowResult],
    truth_dir: str,
    strain_matches: Dict[str, str],  # detected_track_id -> true_strain_id
    truth_snvs: Dict[str, Dict[str, Dict[int, str]]],  # strain_id -> contig -> pos -> allele
    window_size: int
) -> TrackValidationResult:
    """
    Compute all track/linking validation metrics.
    
    Args:
        window_results: Detected window results from strainphase
        truth_dir: Directory containing truth_tracks.tsv
        strain_matches: Mapping from detected track_id to true strain_id
        truth_snvs: True SNV alleles per strain/contig
        window_size: Window size used for processing
    
    Returns:
        TrackValidationResult with all metrics
    """
    # Load truth tracks
    truth_tracks = load_truth_tracks(truth_dir)
    
    if not truth_tracks:
        logger.warning("No truth tracks found - returning zero metrics")
        return TrackValidationResult(
            track_fragmentation_mean=0.0,
            track_fragmentation_median=0.0,
            false_link_rate=0.0,
            missed_link_rate=0.0,
            track_consensus_error=0.0,
            per_strain_fragmentation={},
            false_links=[],
            missed_links=[]
        )
    
    # Check for contig name mismatch
    truth_contigs = set()
    for strain_id, contig_dict in truth_tracks.items():
        truth_contigs.update(contig_dict.keys())
    
    detected_contigs = set()
    for wr in window_results:
        detected_contigs.add(wr.window.contig)
    
    overlap = truth_contigs & detected_contigs
    if not overlap and truth_contigs and detected_contigs:
        logger.warning("="*60)
        logger.warning("CONTIG NAME MISMATCH IN TRACK VALIDATION!")
        logger.warning(f"  Truth contigs: {sorted(truth_contigs)}")
        logger.warning(f"  Detected contigs: {sorted(detected_contigs)}")
        logger.warning("  No overlap - track metrics will be ZERO!")
        logger.warning("  Fix: Ensure truth files use the same contig names as the reference.")
        logger.warning("="*60)
    
    logger.info(f"Track validation: {len(truth_tracks)} truth strains, {len(strain_matches)} strain matches")
    logger.info(f"  Truth contigs: {sorted(truth_contigs)}")
    logger.info(f"  Detected contigs: {sorted(detected_contigs)}")
    
    # Compute fragmentation
    mean_frag, median_frag, per_strain_frag = compute_track_fragmentation(
        window_results, truth_tracks, strain_matches
    )
    
    # Compute linking metrics
    false_link_rate, missed_link_rate, false_links, missed_links = compute_linking_metrics(
        window_results, truth_tracks, strain_matches, window_size
    )
    
    # Compute consensus error
    consensus_error = compute_track_consensus_error(
        window_results, truth_snvs, strain_matches
    )
    
    # Analyze track linkability (why tracks are fragmented)
    linkability_analysis = analyze_track_linkability(window_results, strain_matches)
    
    # Log linkability summary
    if linkability_analysis:
        total_pairs = len(linkability_analysis)
        linkable = sum(1 for info in linkability_analysis if info.is_linkable)
        logger.info(f"Track linkability: {linkable}/{total_pairs} pairs could be linked")
    
    return TrackValidationResult(
        track_fragmentation_mean=mean_frag,
        track_fragmentation_median=median_frag,
        false_link_rate=false_link_rate,
        missed_link_rate=missed_link_rate,
        track_consensus_error=consensus_error,
        per_strain_fragmentation=per_strain_frag,
        false_links=false_links,
        missed_links=missed_links,
        linkability_analysis=linkability_analysis
    )
