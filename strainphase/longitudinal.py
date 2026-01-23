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

import argparse
import logging
import os
import sys
from collections import defaultdict

try:
    import pysam  # noqa: F401

    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not installed, some features limited", file=sys.stderr)

from strainphase.core import (
    Haplotype,
    HaplotyperConfig,
    LongitudinalIntegrator,
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
) -> dict[str, dict[str, list[WindowResult]]]:
    """
    Process a single MAG across all samples with longitudinal rescue.

    Haplotypes are linked across windows after processing and after rescue.

    Returns:
        {sample_id: {contig_id: [WindowResult, ...]}}
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
                # Apply rescue
                rescued = integrator.rescue_low_abundance(results_by_timepoint)

                # Update results and re-link windows (rescue may add new haplotypes)
                for sample_id, window_results in rescued.items():
                    window_results = link_windows(window_results, config)
                    all_results[sample_id][contig_id] = window_results

    return all_results


def build_lineage_table(
    all_results: dict[str, dict[str, dict[str, list[WindowResult]]]], config: HaplotyperConfig
) -> list[dict]:
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
    lineage_counter = 0

    # Process each MAG
    for mag_name, mag_results in all_results.items():
        # Collect all tracks by contig
        # Structure: {contig_id: [(sample_id, track_id, span_start, span_end, merged_consensus, stats), ...]}
        tracks_by_contig: dict[str, list[tuple]] = defaultdict(list)

        for sample_id, contig_results in mag_results.items():
            for contig_id, window_results in contig_results.items():
                # Group haplotypes by track_id within this sample/contig
                track_haps: dict[str, list[tuple[WindowResult, Haplotype]]] = defaultdict(list)

                for wr in window_results:
                    for hap in wr.haplotypes:
                        tid = hap.track_id or f"unlinked_{wr.window.start}"
                        track_haps[tid].append((wr, hap))

                # Build track info
                for track_id, members in track_haps.items():
                    span_start = min(wr.window.start for wr, _ in members)
                    span_end = max(wr.window.end for wr, _ in members)
                    n_windows = len(members)

                    # Merge consensus across all windows in track
                    position_votes: dict[int, dict[str, float]] = defaultdict(
                        lambda: defaultdict(float)
                    )
                    total_weight = 0.0
                    total_reads = 0

                    for _wr, hap in members:
                        total_weight += hap.weight
                        total_reads += hap.supporting_reads
                        for pos, base in hap.consensus.items():
                            position_votes[pos][base] += hap.weight

                    merged_consensus = {}
                    for pos, votes in position_votes.items():
                        merged_consensus[pos] = max(votes.keys(), key=lambda b: votes[b])

                    tracks_by_contig[contig_id].append(
                        (
                            sample_id,
                            track_id,
                            span_start,
                            span_end,
                            n_windows,
                            merged_consensus,
                            total_weight / n_windows,  # mean_weight
                            total_reads,
                        )
                    )

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
                _, _, span_start_i, span_end_i, _, consensus_i, _, _ = tracks[i]
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
                n_timepoints = len({tracks[idx][0] for idx in cluster})

                for idx in cluster:
                    (
                        sample_id,
                        track_id,
                        span_start,
                        span_end,
                        n_windows,
                        consensus,
                        mean_weight,
                        total_reads,
                    ) = tracks[idx]

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
                            "n_windows": n_windows,
                            "mean_weight": mean_weight,
                            "total_supporting_reads": total_reads,
                            "n_snvs": len(consensus),
                            "consensus": consensus_str,
                            "n_timepoints": n_timepoints,
                        }
                    )

    return records


def write_longitudinal_outputs(
    all_results: dict[str, dict[str, dict[str, list[WindowResult]]]],
    lineage_records: list[dict],
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
    os.makedirs(output_dir, exist_ok=True)

    # 1. Lineage table
    lineage_path = os.path.join(output_dir, "lineages.tsv")

    if lineage_records and HAS_PANDAS:
        df = pd.DataFrame(lineage_records)
        df.to_csv(lineage_path, sep="\t", index=False)
    elif lineage_records:
        import csv

        with open(lineage_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=lineage_records[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(lineage_records)
    else:
        with open(lineage_path, "w") as f:
            f.write(
                "lineage_id\tmag\tcontig\tsample\ttrack_id\t"
                "span_start\tspan_end\tspan_bp\tn_windows\t"
                "mean_weight\ttotal_supporting_reads\t"
                "n_snvs\tconsensus\tn_timepoints\n"
            )

    logging.info(f"Wrote {len(lineage_records)} lineage records to {lineage_path}")

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

        if HAS_PANDAS:
            df = pd.DataFrame(records)
            df.to_csv(sample_path, sep="\t", index=False)
        else:
            import csv

            with open(sample_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys(), delimiter="\t")
                writer.writeheader()
                writer.writerows(records)

    # 3. Summary statistics
    summary_path = os.path.join(output_dir, "longitudinal_summary.tsv")

    with open(summary_path, "w") as f:
        f.write(
            "mag\tn_lineages\tn_multi_timepoint_lineages\t"
            "mean_timepoints_per_lineage\tn_samples\n"
        )

        if HAS_PANDAS and lineage_records:
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
    if not HAS_PYSAM:
        print("ERROR: pysam required (install with: pip install pysam)", file=sys.stderr)
        sys.exit(1)
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

    for mag_name, mag_contigs in mags_to_process.items():
        mag_results = process_mag_longitudinal(
            mag_name=mag_name,
            mag_contigs=mag_contigs,
            samples=samples,
            bam_paths=bam_paths,
            vcf_paths=vcf_paths,
            config=config,
        )
        all_results[mag_name] = mag_results

    # Build lineage table
    logging.info("Building lineage table across processed MAGs")
    lineage_records = build_lineage_table(all_results, config)

    # Write outputs
    write_longitudinal_outputs(all_results, lineage_records, args.output_dir)

    # Summary in logs
    n_lineages = len({r["lineage_id"] for r in lineage_records}) if lineage_records else 0
    logging.info(f"DONE: {len(mags_to_process)} MAGs processed, {n_lineages} lineages identified")


if __name__ == "__main__":
    main()
