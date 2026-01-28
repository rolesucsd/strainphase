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
    
    return TrackValidationResult(
        track_fragmentation_mean=mean_frag,
        track_fragmentation_median=median_frag,
        false_link_rate=false_link_rate,
        missed_link_rate=missed_link_rate,
        track_consensus_error=consensus_error,
        per_strain_fragmentation=per_strain_frag,
        false_links=false_links,
        missed_links=missed_links
    )
