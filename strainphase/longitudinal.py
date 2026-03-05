#!/usr/bin/env python3
"""
Longitudinal integration script for haplotyper v3.

This script:
1. Runs haplotyping per contig, per sample (using haplotyper.process_contig)
2. Performs cross-timepoint rescue of low-abundance haplotypes (LongitudinalIntegrator)
3. Builds a lineage table by clustering similar haplotypes across samples
4. Writes:
   - lineages.tsv        (one row per (MAG, contig, window, sample, lineage))
   - longitudinal_summary.tsv
   - <sample>.rescued.tsv (per-sample haplotypes after rescue)

Recommended usage for efficiency:
    * Run ONE MAG per job using --mags <MAG_NAME>
    * Use --contig-filter to restrict to high-coverage / high-breadth contigs
    * Keep validate_results=False (default) for production

Example:
    python run_longitudinal.py \
        --samples bc2001,bc2002,... \
        --bams /ddn_scratch/.../mapping/{sample}.sorted.bam \
        --vcfs /ddn_scratch/.../variants/clair3/{sample}/pileup.vcf.gz \
        --reference /ddn_scratch/.../references/combined_bins.fasta \
        --output-dir /ddn_scratch/.../haplotypes/longitudinal/BF_MAG_01 \
        --mags BF_MAG_01 \
        --contig-filter good_contigs.tsv \
        --window-size 3000 \
        --max-reads 300 \
        --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict

import pandas as pd
import pysam  # noqa: F401

from strainphase.core import (
    Haplotype,
    HaplotyperConfig,
    LongitudinalIntegrator,
    RescueStatistic,
    WindowResult,
    link_windows,
    process_contig,
    results_to_dataframe,
)

# -----------------------------------------------------------------------------#
# Reference parsing and contig filtering
# -----------------------------------------------------------------------------#


def load_allowed_contigs(path: str) -> set[str]:
    """
    Load an optional contig filter file.

    Expected formats:
      - Simple 1-column file: each line is a contig name
      - TSV with header containing a 'contig' column

    Returns a set of contig IDs to keep.
    """
    allowed: set[str] = set()
    with open(path) as f:
        first = f.readline().strip()
        if not first:
            return allowed

        cols = first.split("\t")
        if len(cols) == 1:
            # No header, first line is a contig name
            allowed.add(cols[0])
            for line in f:
                line = line.strip()
                if line:
                    allowed.add(line)
        else:
            # Assume header; require 'contig' column
            header = cols
            if "contig" not in header:
                raise ValueError(
                    f"--contig-filter file {path} has multiple columns but no 'contig' header"
                )
            idx = header.index("contig")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) <= idx:
                    continue
                allowed.add(parts[idx])

    logging.info(f"Loaded {len(allowed)} contigs from filter {path}")
    return allowed


def parse_reference_contigs(
    fasta_path: str, allowed_contigs: set[str] | None = None
) -> dict[str, dict[str, int]]:
    """
    Parse reference .fai to get contig info grouped by MAG.

    Headers are assumed to look like:
        MAGNAME_contig_1
        MAGNAME_contig_2
        ...

    If allowed_contigs is provided, only those contigs are kept.
    """
    fai_path = fasta_path + ".fai"
    mags: dict[str, dict[str, int]] = defaultdict(dict)

    with open(fai_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            contig_name = parts[0]
            length = int(parts[1])

            if allowed_contigs is not None and contig_name not in allowed_contigs:
                continue

            if "_contig_" in contig_name:
                mag_name = contig_name.rsplit("_contig_", 1)[0]
            else:
                mag_name = contig_name

            mags[mag_name][contig_name] = length

    return dict(mags)


# -----------------------------------------------------------------------------#
# Core longitudinal logic
# -----------------------------------------------------------------------------#


def process_mag_longitudinal(
    mag_name: str | None,
    mag_contigs: dict[str, int],
    samples: list[str],
    bam_paths: dict[str, str],
    vcf_paths: dict[str, str],
    config: HaplotyperConfig,
) -> tuple[dict[str, dict[str, list[WindowResult]]], "LongitudinalIntegrator | None"]:
    """
    Process a single MAG across all samples with longitudinal rescue.

    Haplotypes are linked across windows after processing and after rescue.

    Returns:
        Tuple of:
        - {sample_id: {contig_id: [WindowResult, ...]}}
        - LongitudinalIntegrator instance (or None if single timepoint)
    """
    mag_label = mag_name or "<unknown>"
    logging.info(
        f"Processing MAG {mag_label} across {len(samples)} samples " f"({len(mag_contigs)} contigs)"
    )

    # ------------------ First pass: per-sample EM haplotyping ------------------
    # (process_contig now includes window linking)
    all_results: dict[str, dict[str, list[WindowResult]]] = {}

    for sample_id in samples:
        logging.info(f"  Sample {sample_id}: initial contig processing")
        all_results[sample_id] = {}

        for contig_id, contig_length in mag_contigs.items():
            try:
                results = process_contig(
                    bam_path=bam_paths[sample_id],
                    vcf_path=vcf_paths[sample_id],
                    contig_id=contig_id,
                    contig_length=contig_length,
                    config=config,
                    sample_id=sample_id,
                )

                if results:
                    all_results[sample_id][contig_id] = results
                    n_haps = sum(len(wr.haplotypes) for wr in results)
                    # Count unique tracks
                    track_ids = {h.track_id for wr in results for h in wr.haplotypes if h.track_id}
                    logging.debug(
                        f"    {contig_id}: {len(results)} windows, {n_haps} haplotypes, "
                        f"{len(track_ids)} tracks"
                    )
            except Exception as e:
                logging.warning(f"    Error on contig {contig_id} in {sample_id}: {e}")
                continue

    # ------------------ Second pass: cross-timepoint rescue -------------------
    integrator = None
    if len(samples) >= 2:
        logging.info(f"  Performing longitudinal rescue across {len(samples)} samples")
        integrator = LongitudinalIntegrator(config)

        for contig_id in mag_contigs.keys():
            # Collect results for this contig across samples
            results_by_timepoint: dict[str, list[WindowResult]] = {}
            for sample_id in samples:
                sample_contigs = all_results.get(sample_id, {})
                if contig_id in sample_contigs:
                    results_by_timepoint[sample_id] = sample_contigs[contig_id]

            if len(results_by_timepoint) >= 2:
                # Diagnostic: log window counts and junk statistics before rescue
                n_windows_per_sample = {s: len(wrs) for s, wrs in results_by_timepoint.items()}
                total_haps = sum(len(wr.haplotypes) for wrs in results_by_timepoint.values() for wr in wrs)

                # Count junk reads across all windows
                total_reads = 0
                total_junk_reads = 0
                for wrs in results_by_timepoint.values():
                    for wr in wrs:
                        n_reads = wr.gamma.shape[0]
                        junk_idx = wr.gamma.shape[1] - 1
                        junk_reads = (wr.gamma[:, junk_idx] > 0.5).sum()
                        total_reads += n_reads
                        total_junk_reads += junk_reads

                junk_pct = 100 * total_junk_reads / total_reads if total_reads > 0 else 0
                logging.info(
                    f"    Contig {contig_id}: windows={n_windows_per_sample}, "
                    f"haplotypes={total_haps}, junk_reads={total_junk_reads}/{total_reads} ({junk_pct:.1f}%)"
                )

                # Apply rescue
                rescued = integrator.rescue_low_abundance(results_by_timepoint)

                # Update results and re-link windows (rescue may add new haplotypes)
                for sample_id, window_results in rescued.items():
                    window_results = link_windows(window_results, config)
                    all_results[sample_id][contig_id] = window_results

        # Log rescue statistics
        n_rescued = sum(1 for s in integrator.rescue_statistics if s.was_rescued)
        n_total = len(integrator.rescue_statistics)
        logging.info(f"  Rescue completed: {n_rescued}/{n_total} haplotypes rescued")

    # Log integrator status for debugging
    if integrator:
        logging.info(f"  Returning integrator with {len(integrator.rescue_statistics)} statistics records")
    else:
        logging.info(f"  No integrator (len(samples)={len(samples)})")

    return all_results, integrator


def build_lineage_table(
    all_results: dict[str, dict[str, dict[str, list[WindowResult]]]], config: HaplotyperConfig
) -> tuple[list[dict], list[dict]]:
    """
    Build lineage tracking table across samples using TRACKS.

    Clusters similar tracks (linked haplotypes) across samples by
    consensus similarity. Each track spans multiple windows within
    a sample, and similar tracks across samples get the same lineage_id.

    Clustering logic:
    1. Group tracks by contig
    2. For each pair of tracks:
       - Check span overlap (must be at same locus, within max_span_gap_for_lineage)
       - Compute consensus distance on shared SNV positions
       - Merge into same lineage if dist <= lineage_merge_distance and n_shared >= min_shared_for_lineage

    Key parameters (from config):
    - lineage_merge_distance: Max distance to merge (default 0.02 = 2%)
    - min_shared_for_lineage: Min shared SNVs to consider (default 3)
    - max_span_gap_for_lineage: Max bp gap between tracks to consider same locus (default 5000)

    Returns:
        List of dicts suitable for pd.DataFrame() or TSV writing.
    """
    records: list[dict] = []
    haplotype_records: list[dict] = []
    lineage_counter = 0

    # Process each MAG
    for mag_name, mag_results in all_results.items():
        # Collect all tracks by contig
        # Structure: {contig_id: [track_info, ...]}
        tracks_by_contig: dict[str, list[dict]] = defaultdict(list)

        for sample_id, contig_results in mag_results.items():
            for contig_id, window_results in contig_results.items():
                # Group haplotypes by track_id within this sample/contig
                track_haps: dict[str, list[tuple[WindowResult, Haplotype, int]]] = defaultdict(list)

                for wr in window_results:
                    for h_idx, hap in enumerate(wr.haplotypes):
                        tid = hap.track_id or f"unlinked_{wr.window.start}"
                        track_haps[tid].append((wr, hap, h_idx))

                # Build track info
                for track_id, members in track_haps.items():
                    window_span_start = min(wr.window.start for wr, _, _ in members)
                    window_span_end = max(wr.window.end for wr, _, _ in members)
                    n_windows = len(members)

                    # Merge consensus across all windows in track
                    position_votes: dict[int, dict[str, float]] = defaultdict(
                        lambda: defaultdict(float)
                    )
                    # Abundance is computed from the EM mixture proportions but
                    # explicitly conditioned on NON-junk reads. For each window we
                    # take pi_k / (1 - pi_junk), where pi_k is the haplotype's
                    # mixture weight and pi_junk is the junk component. This
                    # removes the influence of the junk bucket on abundance
                    # estimates while still using the soft-assignments from EM.
                    window_abundance_sum = 0.0
                    total_reads_sum = 0
                    reads_sum = 0  # reads matching consensus at max_mismatch_frac

                    for _wr, hap, hap_idx in members:
                        # total_reads: reads assigned to any haplotype (non-junk),
                        # derived from gamma so it reflects the state after rescue.
                        n_reads = getattr(_wr, "n_reads_examined", len(_wr.window.reads))
                        junk_col = _wr.gamma.shape[1] - 1
                        n_junk = int((_wr.gamma[:, junk_col] >= 0.5).sum())
                        total_reads_sum += n_reads - n_junk
                        # reads: gamma-based supporting reads for THIS haplotype,
                        # updated after rescue so rescued haplotypes are included.
                        reads_sum += hap.supporting_reads
                        # Per-window abundance for this haplotype, conditioned on
                        # non-junk reads in the window. If the window is junk-only
                        # (pi_junk ~= 1), treat its contribution as zero.
                        pi_vec = getattr(_wr, "pi", None)
                        window_abundance = 0.0
                        if pi_vec is not None and len(pi_vec) > hap_idx:
                            pi_junk = float(pi_vec[-1])
                            denom = 1.0 - pi_junk
                            if denom > 0:
                                pi_k = float(pi_vec[hap_idx])
                                window_abundance = max(0.0, min(1.0, pi_k / denom))
                        window_abundance_sum += window_abundance
                        for pos, base in hap.consensus.items():
                            position_votes[pos][base] += hap.weight

                    merged_consensus = {}
                    for pos, votes in position_votes.items():
                        merged_consensus[pos] = max(votes.keys(), key=lambda b: votes[b])

                    # Span: first SNV position to last SNV position (fallback to window span if no SNVs).
                    if merged_consensus:
                        span_start = min(merged_consensus.keys())
                        span_end = max(merged_consensus.keys())
                    else:
                        span_start = window_span_start
                        span_end = window_span_end

                    # Abundance = mean per-window conditional mixture weight
                    # (pi_k / (1 - pi_junk)) across windows, clamped to [0, 1].
                    abundance = (
                        max(0.0, min(1.0, window_abundance_sum / n_windows))
                        if n_windows > 0
                        else 0.0
                    )

                    tracks_by_contig[contig_id].append(
                        {
                            "sample": sample_id,
                            "track_id": track_id,
                            "span_start": span_start,
                            "span_end": span_end,
                            "n_windows": n_windows,
                            "consensus": merged_consensus,
                            "abundance": abundance,
                            "total_reads_sum": total_reads_sum,
                            "reads_sum": reads_sum,
                            "members": members,
                        }
                    )

        # Deduplicate tracks within the same sample that have identical consensus
        for contig_id in tracks_by_contig:
            tracks = tracks_by_contig[contig_id]
            if not tracks:
                continue

            # Group tracks by (sample_id, consensus_tuple)
            # Merge tracks with identical consensus within the same sample
            dedup_key_to_tracks: dict[tuple, list[int]] = defaultdict(list)
            for i, track in enumerate(tracks):
                sample_id = track["sample"]
                consensus = track["consensus"]
                consensus_key = tuple(sorted(consensus.items()))
                dedup_key_to_tracks[(sample_id, consensus_key)].append(i)

            # Build deduplicated track list
            deduped_tracks = []
            for (sample_id, _consensus_key), indices in dedup_key_to_tracks.items():
                if len(indices) == 1:
                    # No duplicates, keep as-is
                    deduped_tracks.append(tracks[indices[0]])
                else:
                    # Merge duplicate tracks
                    # Combine: track_ids (use first), spans (union), n_windows (sum),
                    # consensus (same), abundance (weighted average), total_reads (sum), reads (sum)
                    first = tracks[indices[0]]
                    merged_track_id = first["track_id"]  # Keep first track_id
                    merged_consensus = first["consensus"]  # Same consensus
                    if merged_consensus:
                        merged_span_start = min(merged_consensus.keys())
                        merged_span_end = max(merged_consensus.keys())
                    else:
                        merged_span_start = min(tracks[i]["span_start"] for i in indices)
                        merged_span_end = max(tracks[i]["span_end"] for i in indices)
                    merged_n_windows = sum(tracks[i]["n_windows"] for i in indices)
                    # Weighted average of abundance by n_windows, clamped to [0, 1].
                    total_weight_sum = sum(
                        tracks[i]["abundance"] * tracks[i]["n_windows"] for i in indices
                    )
                    merged_abundance = (
                        max(0.0, min(1.0, total_weight_sum / merged_n_windows))
                        if merged_n_windows > 0
                        else 0.0
                    )
                    merged_total_reads = sum(tracks[i]["total_reads_sum"] for i in indices)
                    merged_reads = sum(tracks[i]["reads_sum"] for i in indices)
                    merged_members = []
                    for i in indices:
                        merged_members.extend(tracks[i]["members"])

                    deduped_tracks.append({
                        "sample": sample_id,
                        "track_id": merged_track_id,
                        "span_start": merged_span_start,
                        "span_end": merged_span_end,
                        "n_windows": merged_n_windows,
                        "consensus": merged_consensus,
                        "abundance": merged_abundance,
                        "total_reads_sum": merged_total_reads,
                        "reads_sum": merged_reads,
                        "members": merged_members,
                    })

            tracks_by_contig[contig_id] = deduped_tracks

        # Cluster tracks across samples by consensus similarity
        # FIXED: Compare each track to ALL existing clusters (not just unassigned tracks)
        for contig_id, tracks in tracks_by_contig.items():
            if not tracks:
                continue

            # clusters[i] = list of track indices in cluster i
            # cluster_consensus[i] = merged consensus for cluster i
            # cluster_spans[i] = (min_start, max_end) for cluster i
            clusters: list[list[int]] = []
            cluster_consensus: list[dict[int, str]] = []
            cluster_spans: list[tuple[int, int]] = []

            for i in range(len(tracks)):
                span_start_i = tracks[i]["span_start"]
                span_end_i = tracks[i]["span_end"]
                consensus_i = tracks[i]["consensus"]
                positions_i = set(consensus_i.keys())

                # Try to find an existing cluster to join
                best_cluster = -1
                best_dist = float("inf")

                for c_idx in range(len(clusters)):
                    c_span_start, c_span_end = cluster_spans[c_idx]
                    c_consensus = cluster_consensus[c_idx]

                    # Check span overlap
                    span_gap = max(0, max(span_start_i, c_span_start) - min(span_end_i, c_span_end))
                    if span_gap > config.max_span_gap_for_lineage:
                        continue

                    # Compute distance on shared positions
                    shared_pos = positions_i & set(c_consensus.keys())
                    if len(shared_pos) < config.min_shared_for_lineage:
                        continue

                    mismatches = sum(
                        1 for p in shared_pos if consensus_i.get(p) != c_consensus.get(p)
                    )
                    dist = mismatches / len(shared_pos)

                    if dist <= config.lineage_merge_distance and dist < best_dist:
                        best_cluster = c_idx
                        best_dist = dist

                if best_cluster >= 0:
                    # Join existing cluster
                    clusters[best_cluster].append(i)
                    # Update cluster consensus (add new positions, keep existing)
                    for pos, base in consensus_i.items():
                        if pos not in cluster_consensus[best_cluster]:
                            cluster_consensus[best_cluster][pos] = base
                    # Update cluster span
                    old_start, old_end = cluster_spans[best_cluster]
                    cluster_spans[best_cluster] = (
                        min(old_start, span_start_i),
                        max(old_end, span_end_i),
                    )
                else:
                    # Create new cluster
                    clusters.append([i])
                    cluster_consensus.append(dict(consensus_i))
                    cluster_spans.append((span_start_i, span_end_i))

            # Emit lineage records
            for cluster in clusters:
                lineage_counter += 1
                lineage_id = f"L{lineage_counter:06d}"
                lineage_span_start = min(tracks[idx]["span_start"] for idx in cluster)
                lineage_span_end = max(tracks[idx]["span_end"] for idx in cluster)
                lineage_total_span = lineage_span_end - lineage_span_start

                # Merge any tracks from the same sample that ended up in this
                # cluster (can happen when two tracks have similar but non-identical
                # consensuses that both fall within lineage_merge_distance).
                # Result: exactly one row per (lineage, sample).
                sample_to_cluster_tracks: dict[str, list[dict]] = defaultdict(list)
                for idx in cluster:
                    sample_to_cluster_tracks[tracks[idx]["sample"]].append(tracks[idx])

                merged_cluster_tracks: list[dict] = []
                for s_id, s_tracks in sample_to_cluster_tracks.items():
                    if len(s_tracks) == 1:
                        merged_cluster_tracks.append(s_tracks[0])
                    else:
                        m_n_windows = sum(t["n_windows"] for t in s_tracks)
                        m_weight_sum = sum(t["abundance"] * t["n_windows"] for t in s_tracks)
                        m_abundance = max(0.0, min(1.0, m_weight_sum / m_n_windows if m_n_windows > 0 else 0.0))
                        m_total_reads = sum(t["total_reads_sum"] for t in s_tracks)
                        m_reads = sum(t["reads_sum"] for t in s_tracks)
                        m_members: list = []
                        for t in s_tracks:
                            m_members.extend(t["members"])
                        # Merge consensus by weighted vote
                        m_pos_votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
                        for t in s_tracks:
                            w = t["abundance"] * t["n_windows"]
                            for pos, base in t["consensus"].items():
                                m_pos_votes[pos][base] += w
                        m_consensus = {pos: max(votes, key=votes.get) for pos, votes in m_pos_votes.items()}
                        if m_consensus:
                            m_span_start = min(m_consensus.keys())
                            m_span_end = max(m_consensus.keys())
                        else:
                            m_span_start = min(t["span_start"] for t in s_tracks)
                            m_span_end = max(t["span_end"] for t in s_tracks)
                        merged_cluster_tracks.append({
                            "sample": s_id,
                            "track_id": s_tracks[0]["track_id"],
                            "span_start": m_span_start,
                            "span_end": m_span_end,
                            "n_windows": m_n_windows,
                            "consensus": m_consensus,
                            "abundance": m_abundance,
                            "total_reads_sum": m_total_reads,
                            "reads_sum": m_reads,
                            "members": m_members,
                        })

                n_timepoints = len(merged_cluster_tracks)

                for track in merged_cluster_tracks:
                    sample_id = track["sample"]
                    track_id = track["track_id"]
                    span_start = track["span_start"]
                    span_end = track["span_end"]
                    n_windows = track["n_windows"]
                    consensus = track["consensus"]
                    abundance = track["abundance"]
                    # Report totals (summed across windows in this track).
                    # total_reads: reads assigned to any haplotype (non-junk).
                    # reads: reads assigned to this specific haplotype/track.
                    total_reads_total = track["total_reads_sum"]
                    reads_total = track["reads_sum"]

                    # Skip rows with zero supporting reads — these are phantom
                    # rescued haplotypes where the model assigned no reads.
                    if reads_total == 0:
                        continue

                    consensus_str = "|".join(
                        f"{pos}:{base}" for pos, base in sorted(consensus.items())
                    )

                    records.append(
                        {
                            "lineage_id": lineage_id,
                            "mag": mag_name,
                            "contig": contig_id,
                            "sample": sample_id,
                            "track_id": track_id,
                            "span_start": span_start,
                            "span_end": span_end,
                            "span_bp": span_end - span_start,
                            "total_span": lineage_total_span,
                            "abundance": abundance,
                            "reads": reads_total,
                            "total_reads": total_reads_total,
                            "n_snvs": len(consensus),
                            "consensus": consensus_str,
                            "n_timepoints": n_timepoints,
                        }
                    )

                    for wr, hap, hap_idx in track["members"]:
                        # Skip haplotype rows with zero reads for the same reason.
                        hap_reads = hap.supporting_reads
                        if hap_reads == 0:
                            continue
                        hap_consensus_str = "|".join(
                            f"{pos}:{base}" for pos, base in sorted(hap.consensus.items())
                        )
                        hap_id = f"{track_id}_W{wr.window.start}_H{hap_idx}"
                        n_reads_w = getattr(wr, "n_reads_examined", len(wr.window.reads))
                        junk_col_w = wr.gamma.shape[1] - 1
                        n_junk_w = int((wr.gamma[:, junk_col_w] >= 0.5).sum())
                        hap_total_reads = n_reads_w - n_junk_w
                        if hap.consensus:
                            hap_span_start = min(hap.consensus.keys())
                            hap_span_end = max(hap.consensus.keys())
                            hap_span_bp = hap_span_end - hap_span_start
                        else:
                            hap_span_start = wr.window.start
                            hap_span_end = wr.window.end
                            hap_span_bp = hap_span_end - hap_span_start
                        # Per-window abundance for this haplotype, conditioned on
                        # non-junk reads (pi_k / (1 - pi_junk)).
                        pi_vec = getattr(wr, "pi", None)
                        hap_abundance = 0.0
                        if pi_vec is not None and len(pi_vec) > hap_idx:
                            pi_junk_w = float(pi_vec[-1])
                            denom_w = 1.0 - pi_junk_w
                            if denom_w > 0:
                                pi_k_w = float(pi_vec[hap_idx])
                                hap_abundance = max(0.0, min(1.0, pi_k_w / denom_w))
                        haplotype_records.append(
                            {
                                "lineage_id": lineage_id,
                                "haplotype_id": hap_id,
                                "mag": mag_name,
                                "contig": contig_id,
                                "sample": sample_id,
                                "track_id": track_id,
                                "span_start": hap_span_start,
                                "span_end": hap_span_end,
                                "span_bp": hap_span_bp,
                                "n_windows": 1,
                                "total_span": lineage_total_span,
                                "abundance": hap_abundance,
                                "reads": hap_reads,
                                "total_reads": hap_total_reads,
                                "n_snvs": len(hap.consensus),
                                "consensus": hap_consensus_str,
                                "n_timepoints": n_timepoints,
                            }
                        )

    return records, haplotype_records


def write_lineage_tables(
    lineage_records: list[dict],
    haplotype_records: list[dict],
    output_dir: str,
) -> tuple[str, str]:
    """Write lineages.tsv and haplotypes.tsv with consistent headers."""
    import csv

    os.makedirs(output_dir, exist_ok=True)

    lineage_path = os.path.join(output_dir, "lineages.tsv")
    haplotype_path = os.path.join(output_dir, "haplotypes.tsv")

    lineage_fieldnames = list(lineage_records[0].keys()) if lineage_records else [
        "lineage_id", "mag", "contig", "sample", "track_id",
        "span_start", "span_end", "span_bp",
        "total_span",
        "abundance", "reads", "total_reads",
        "n_snvs", "consensus", "n_timepoints",
    ]
    with open(lineage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=lineage_fieldnames, delimiter="	")
        writer.writeheader()
        if lineage_records:
            writer.writerows(lineage_records)

    hap_fieldnames = list(haplotype_records[0].keys()) if haplotype_records else [
        "lineage_id", "haplotype_id", "mag", "contig", "sample", "track_id",
        "span_start", "span_end", "span_bp", "n_windows",
        "total_span",
        "abundance", "reads", "total_reads",
        "n_snvs", "consensus", "n_timepoints",
    ]
    with open(haplotype_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hap_fieldnames, delimiter="	")
        writer.writeheader()
        if haplotype_records:
            writer.writerows(haplotype_records)

    return lineage_path, haplotype_path


def write_longitudinal_outputs(
    all_results: dict[str, dict[str, dict[str, list[WindowResult]]]],
    lineage_records: list[dict],
    haplotype_records: list[dict],
    output_dir: str,
) -> str:
    """
    Write longitudinal analysis outputs into output_dir.

    Files:
      - {output_dir}/lineages.tsv
      - {output_dir}/longitudinal_summary.tsv
      - {output_dir}/{sample}.rescued.tsv  (per-sample haplotypes after rescue)
        (If multiple MAGs are processed, each sample file will contain
         haplotypes from all those MAGs; MAG is a column in the TSV.)
    """
    os.makedirs(output_dir, exist_ok=True)    # 1. Lineage + haplotype tables
    lineage_path, haplotype_path = write_lineage_tables(
        lineage_records, haplotype_records, output_dir
    )
    logging.info(f"Wrote {len(lineage_records)} lineage records to {lineage_path}")
    logging.info(f"Wrote {len(haplotype_records)} haplotype records to {haplotype_path}")


    # 2. Per-sample haplotypes (post-rescue)
    #    Note: if multiple MAGs are processed, each sample file will contain
    #    rows for multiple MAGs (with 'mag' column distinguishing them).
    per_sample_records: dict[str, list[dict]] = defaultdict(list)

    for mag_name, mag_results in all_results.items():
        for sample_id, contig_results in mag_results.items():
            contig_records = []
            for contig_id, window_results in contig_results.items():
                contig_records.extend(results_to_dataframe({contig_id: window_results}))

            for rec in contig_records:
                rec["mag"] = mag_name
            per_sample_records[sample_id].extend(contig_records)

    for sample_id, records in per_sample_records.items():
        if not records:
            continue

        sample_path = os.path.join(output_dir, f"{sample_id}.rescued.tsv")

        df = pd.DataFrame(records)
        df.to_csv(sample_path, sep="\t", index=False)

    # 3. Summary statistics
    summary_path = os.path.join(output_dir, "longitudinal_summary.tsv")

    with open(summary_path, "w") as f:
        f.write(
            "mag\tn_lineages\tn_multi_timepoint_lineages\t"
            "mean_timepoints_per_lineage\tn_samples\n"
        )

        if lineage_records:
            df = pd.DataFrame(lineage_records)
            for mag_name in df["mag"].unique():
                mag_df = df[df["mag"] == mag_name]
                lineages = mag_df.groupby("lineage_id").first()

                n_lineages = len(lineages)
                n_multi = (lineages["n_timepoints"] >= 2).sum()
                mean_tp = lineages["n_timepoints"].mean()
                n_samples = mag_df["sample"].nunique()

                f.write(f"{mag_name}\t{n_lineages}\t{n_multi}\t{mean_tp:.2f}\t{n_samples}\n")

    logging.info(f"Wrote summary to {summary_path}")
    return lineage_path


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#


def main():
    parser = argparse.ArgumentParser(
        description="Longitudinal haplotype integration across samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--samples", required=True, help="Comma-separated list of sample names")
    parser.add_argument("--bams", required=True, help="BAM path template with {sample} placeholder")
    parser.add_argument("--vcfs", required=True, help="VCF path template with {sample} placeholder")
    parser.add_argument("--reference", required=True, help="Combined reference FASTA")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    # Optional / efficiency-related
    parser.add_argument(
        "--mags", help="Comma-separated list of MAGs to process (default: all MAGs in reference)"
    )
    parser.add_argument(
        "--contig-filter", help="TSV or text file listing contigs to include (see docstring)"
    )
    parser.add_argument("--window-size", type=int, default=6000, help="Window size for haplotyping")
    parser.add_argument(
        "--max-reads",
        type=int,
        default=1000,
        help="Max reads per window (subsampling for performance)",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--validate-results",
        action="store_true",
        help="Enable internal consistency checks on WindowResult (slower)",
    )

    # Longitudinal hyperparameters exposed for transparency
    parser.add_argument(
        "--min-anchor-weight",
        type=float,
        default=0.15,
        help="Min mixture weight for a haplotype to serve as an anchor",
    )
    parser.add_argument(
        "--rescued-min-weight",
        type=float,
        default=0.02,
        help="Min mixture weight to assign to rescued haplotypes",
    )

    # Lineage clustering parameters
    parser.add_argument(
        "--lineage-merge-distance",
        type=float,
        default=0.02,
        help="Max distance to merge tracks into same lineage (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--min-shared-for-lineage",
        type=int,
        default=3,
        help="Min shared SNVs to consider merging tracks into lineage",
    )
    parser.add_argument(
        "--max-span-gap",
        type=int,
        default=10000,
        help="Max bp gap between track spans to consider same locus",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse samples
    samples = [s.strip() for s in args.samples.split(",") if s.strip()]
    if not samples:
        logging.error("No valid samples provided in --samples")
        sys.exit(1)
    logging.info(f"Processing {len(samples)} samples: {samples}")

    # Build file paths
    bam_paths = {s: args.bams.replace("{sample}", s) for s in samples}
    vcf_paths = {s: args.vcfs.replace("{sample}", s) for s in samples}

    # Validate files exist
    for sample in samples:
        for path, name in [(bam_paths[sample], "BAM"), (vcf_paths[sample], "VCF")]:
            if not os.path.exists(path):
                logging.error(f"{name} not found for {sample}: {path}")
                sys.exit(1)

    # Load optional contig filter
    allowed_contigs: set[str] | None = None
    if args.contig_filter:
        allowed_contigs = load_allowed_contigs(args.contig_filter)

    # Build config for haplotyper
    config = HaplotyperConfig(
        window_size=args.window_size,
        max_reads_per_window=args.max_reads,
        random_seed=args.seed,
        validate_results=args.validate_results,
        min_weight_for_anchor=args.min_anchor_weight,
        rescued_min_weight=args.rescued_min_weight,
        # Lineage clustering
        lineage_merge_distance=args.lineage_merge_distance,
        min_shared_for_lineage=args.min_shared_for_lineage,
        max_span_gap_for_lineage=args.max_span_gap,
    )

    # Get MAG -> contigs map, optionally filtered by allowed_contigs
    all_mags = parse_reference_contigs(args.reference, allowed_contigs)

    if args.mags:
        mag_names = [m.strip() for m in args.mags.split(",") if m.strip()]
        mags_to_process = {m: all_mags[m] for m in mag_names if m in all_mags}
        missing = sorted(set(mag_names) - set(mags_to_process.keys()))
        if missing:
            logging.warning(f"Requested MAGs not found in reference: {missing}")
    else:
        mags_to_process = all_mags

    if not mags_to_process:
        logging.error("No MAGs to process after applying filters")
        sys.exit(1)

    logging.info(f"Processing {len(mags_to_process)} MAGs: {sorted(mags_to_process)}")

    # Process each MAG with longitudinal integration
    all_results: dict[str, dict[str, dict[str, list[WindowResult]]]] = {}
    all_integrators = []

    for mag_name, mag_contigs in mags_to_process.items():
        mag_results, integrator = process_mag_longitudinal(
            mag_name=mag_name,
            mag_contigs=mag_contigs,
            samples=samples,
            bam_paths=bam_paths,
            vcf_paths=vcf_paths,
            config=config,
        )
        all_results[mag_name] = mag_results
        if integrator:
            all_integrators.append(integrator)

    # Build lineage table
    logging.info("Building lineage table across processed MAGs")
    lineage_records, haplotype_records = build_lineage_table(all_results, config)

    # Write outputs
    write_longitudinal_outputs(all_results, lineage_records, haplotype_records, args.output_dir)

    # Summary in logs
    n_lineages = len({r["lineage_id"] for r in lineage_records}) if lineage_records else 0
    logging.info(f"DONE: {len(mags_to_process)} MAGs processed, {n_lineages} lineages identified")


if __name__ == "__main__":
    main()
