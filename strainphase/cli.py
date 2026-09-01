#!/usr/bin/env python3
"""
Strainphase command-line interface.

Usage:
    strainphase run          # Process single contig
    strainphase longitudinal # Process MAG across timepoints
    strainphase test         # Run test suite
    strainphase version      # Show version
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from strainphase import __version__


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run haplotyper on a single contig."""
    from strainphase import HaplotyperConfig, process_contig, results_to_dataframe

    setup_logging(args.log_level)

    config = HaplotyperConfig(
        window_size=args.window_size,
        max_reads_per_window=args.max_reads,
        min_mapq=args.min_mapq,
        max_mismatch_frac=args.max_mismatch,
        min_depth_site=args.min_depth_site,
        random_seed=args.seed,
        validate_results=not args.no_validate,
    )

    logging.info(f"Processing contig {args.contig} ({args.length} bp)")

    results = process_contig(
        bam_path=args.bam,
        vcf_path=args.vcf,
        contig_id=args.contig,
        contig_length=args.length,
        config=config,
        sample_id=args.sample,
        vcf_sample_name=args.vcf_sample,
    )

    records = results_to_dataframe({args.contig: results})

    if records:
        import csv

        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(records)
        logging.info(f"Wrote {len(records)} haplotypes to {args.output}")
    else:
        logging.warning("No haplotypes found")

    return 0


def cmd_longitudinal(args: argparse.Namespace) -> int:
    """Run longitudinal analysis across multiple samples."""
    setup_logging(args.log_level)

    # Import here to avoid pysam requirement for other commands
    import importlib.util

    if importlib.util.find_spec("pysam") is None:
        logging.error(
            "pysam is required for longitudinal analysis. Install with: pip install pysam"
        )
        return 1

    from strainphase import HaplotyperConfig
    from strainphase.longitudinal import (
        build_window_tables,
        load_allowed_contigs,
        parse_reference_contigs,
        process_mag_longitudinal,
        write_window_tables,
    )

    # Parse samples
    samples = [s.strip() for s in args.samples.split(",")]
    logging.info(f"Processing {len(samples)} samples: {samples}")

    if len(set(samples)) != len(samples):
        logging.error(f"--samples contains duplicates: {args.samples}")
        return 1

    # Build path mappings
    bam_paths = {s: args.bams.format(sample=s) for s in samples}
    vcf_paths = {s: args.vcfs.format(sample=s) for s in samples}
    sv_sidecar_paths = None
    if getattr(args, "sv_sidecars", None):
        sv_sidecar_paths = {s: args.sv_sidecars.format(sample=s) for s in samples}

    # A --bams template without {sample} in it resolves to ONE alignment file for
    # every timepoint, and the existence check below then passes for all of them.
    # Nothing downstream can notice: every timepoint is phased from one sample's
    # reads and the run looks entirely normal. A shared --vcfs (a cohort/union VCF)
    # or a shared --sv-sidecars is legitimate by contrast — those are variant
    # catalogues, and the BAM is what makes a sample a sample.
    if len(set(bam_paths.values())) != len(samples):
        logging.error(
            f"--bams does not resolve to a distinct BAM per sample: {args.bams!r} -> "
            f"{sorted(set(bam_paths.values()))}. The template must contain {{sample}}."
        )
        return 1

    # Verify files exist
    for sample in samples:
        if not os.path.exists(bam_paths[sample]):
            logging.error(f"BAM not found: {bam_paths[sample]}")
            return 1
        if not os.path.exists(vcf_paths[sample]):
            logging.error(f"VCF not found: {vcf_paths[sample]}")
            return 1
        if sv_sidecar_paths and not os.path.exists(sv_sidecar_paths[sample]):
            logging.error(f"SV sidecar not found: {sv_sidecar_paths[sample]}")
            return 1

    # Load contig filter if provided
    allowed_contigs = None
    if args.contig_filter:
        allowed_contigs = load_allowed_contigs(args.contig_filter)

    # Parse reference to get MAGs and contigs
    mags = parse_reference_contigs(args.reference, allowed_contigs)

    # Filter to requested MAGs
    if args.mags:
        requested = set(args.mags.split(","))
        mags = {k: v for k, v in mags.items() if k in requested}

    if not mags:
        logging.error("No MAGs to process")
        return 1

    logging.info(f"Processing {len(mags)} MAGs")

    # Configure
    config = HaplotyperConfig(
        window_size=args.window_size,
        max_reads_per_window=args.max_reads,
        min_weight_for_anchor=args.min_anchor_weight,
        rescued_min_weight=args.rescued_min_weight,
        min_depth_site=args.min_depth_site,
        n_workers=max(1, getattr(args, "workers", 1)),
        validate_results=False,
        # Depth policy + identity gates (see HaplotyperConfig for the reasoning).
        min_reads_per_window=args.min_reads_per_window,
        min_reads_for_rescue=args.min_reads_for_rescue,
        min_read_window_overlap_bp=args.min_read_window_overlap_bp,
        min_read_read_overlap_bp=args.min_read_read_overlap_bp,
        min_entity_overlap_bp=args.min_entity_overlap_bp,
        min_cosupported_span_frac=args.min_cosupported_span_frac,
        min_shared_snvs_for_link=args.min_shared_snvs_for_link,
        identity_distance=args.identity_distance,
        min_shared_markers=args.min_shared_markers,
        track_merge_min_shared_markers=args.track_merge_min_shared_markers,
        link_window_reach=args.link_window_reach,
        link_min_shared_reads=args.link_min_shared_reads,
        cross_sample_method=args.cross_sample_method,
        random_seed=args.seed,
        min_shared_reads_for_link=args.min_shared_reads_for_link,
        lineage_max_bad_frac=args.lineage_max_bad_frac,
        transitive_abundance_check=not args.no_transitive_abundance_check,
        step1_veto_min_timepoints=args.step1_veto_min_timepoints,
        spill_results_to_disk=not args.no_spill,
        window_batch_factor=args.window_batch_factor,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each MAG
    all_results = {}
    all_integrators = []
    for mag_name, mag_contigs in mags.items():
        logging.info(f"Processing MAG {mag_name}")

        results, integrator = process_mag_longitudinal(
            mag_name=mag_name,
            mag_contigs=mag_contigs,
            samples=samples,
            bam_paths=bam_paths,
            vcf_paths=vcf_paths,
            config=config,
            sv_sidecar_paths=sv_sidecar_paths,
            # REQUIRED for read spilling. Without it _SpillStore falls back to
            # _NullSpill and every sample's reads stay resident for the whole MAG,
            # which is the OOM the spill work exists to prevent.
            output_dir=args.output_dir,
        )
        all_results[mag_name] = results
        if integrator:
            all_integrators.append(integrator)

    # ---- Window-level tables (the deliverables) ----
    # lineages.tsv comes back from here too, under config.build_lineages (default on):
    # composing the within-sample and across-sample linking axes was the open decision
    # these tables were built as the substrate for, and step 3 now makes it.
    (hap_rows, within_rows, across_rows, edge_rows, mismatch_rows,
     edge_counts, lineage_rows) = build_window_tables(
        args.output_dir, all_results, config, sample_order=samples
    )
    write_window_tables(
        hap_rows, within_rows, across_rows, edge_rows, args.output_dir, mismatch_rows,
        lineage_rows
    )

    n_within = len({(r["sample"], r["contig"], r["within_sample_id"]) for r in within_rows})
    n_mismatch = edge_counts.get("failed_mismatch", 0)
    n_noev = sum(1 for e in edge_rows if e["reason"] == "failed_no_evidence")
    logging.info(
        f"DONE: {len(hap_rows)} window-haplotypes | {n_within} within-sample entities | "
        f"{len(across_rows)} across-sample window groups "
        f"(method={config.cross_sample_method}) | comparisons: "
        f"{len(edge_rows) - n_mismatch - n_noev} linked, {n_noev} no-evidence, "
        f"{n_mismatch} mismatch"
    )


    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Run test suite."""
    setup_logging("INFO")

    print(f"\n{'=' * 60}")
    print(f"STRAINPHASE v{__version__} - TEST SUITE")
    print(f"{'=' * 60}\n")

    try:
        import subprocess

        cmd = ["python", "-m", "pytest", "tests/", "-v" if args.verbose else "-q"]
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode
    except Exception as e:
        logging.error(f"Could not run tests: {e}")
        logging.info("Try running: pip install strainphase[dev] && pytest tests/")
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    """Show version information."""
    print(f"strainphase {__version__}")
    return 0


def main(argv: list | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="strainphase",
        description="Hybrid graph-probabilistic haplotype reconstruction for PacBio HiFi metagenomic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single contig
    strainphase run --bam sample.bam --vcf variants.vcf --contig ctg1 --length 50000

    # Longitudinal analysis
    strainphase longitudinal --samples T1,T2,T3 \\
        --bams mapping/{sample}.bam --vcfs variants/{sample}.vcf.gz \\
        --reference ref.fasta --output-dir results/

    # Run tests
    strainphase test
        """,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =========== RUN subcommand ===========
    run_parser = subparsers.add_parser(
        "run",
        help="Process a single contig",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument("--bam", required=True, help="Input BAM file")
    run_parser.add_argument("--vcf", required=True, help="Input VCF file (Clair3)")
    run_parser.add_argument("--contig", required=True, help="Contig ID to process")
    run_parser.add_argument("--length", type=int, required=True, help="Contig length")
    run_parser.add_argument("--sample", help="Sample ID")
    run_parser.add_argument("--vcf-sample", help="Sample name in VCF")
    run_parser.add_argument("--output", "-o", default="haplotypes.tsv", help="Output file")
    run_parser.add_argument("--window-size", type=int, default=20000, help="Window size (bp)")
    run_parser.add_argument("--max-reads", type=int, default=300, help="Max reads per window")
    run_parser.add_argument("--min-mapq", type=int, default=20, help="Minimum MAPQ")
    run_parser.add_argument(
        "--max-mismatch", type=float, default=0.01, help="Max mismatch fraction"
    )
    run_parser.add_argument(
        "--min-depth-site",
        type=int,
        default=3,
        help="Min VCF DP to load a site. Set 1 for pre-called SNV lists (e.g. "
        "SNooPy) where the BAM re-genotyping is the real depth gate.",
    )
    run_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Seeded by default: it drives both read subsampling above "
        "--max-reads and Louvain read clustering, so an unseeded run is not reproducible.",
    )
    run_parser.add_argument("--no-validate", action="store_true", help="Skip result validation")
    run_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    run_parser.set_defaults(func=cmd_run)

    # =========== LONGITUDINAL subcommand ===========
    long_parser = subparsers.add_parser(
        "longitudinal",
        help="Process MAG across multiple timepoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    long_parser.add_argument("--samples", required=True, help="Comma-separated sample IDs")
    long_parser.add_argument("--bams", required=True, help="BAM path template with {sample}")
    long_parser.add_argument("--vcfs", required=True, help="VCF path template with {sample}")
    long_parser.add_argument(
        "--sv-sidecars",
        help="Optional SV sidecar TSV template with {sample} "
        "(from strainphase.sv_encoding) to co-phase structural variants",
    )
    long_parser.add_argument("--reference", required=True, help="Reference FASTA")
    long_parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    long_parser.add_argument("--mags", help="Comma-separated MAG names (default: all)")
    long_parser.add_argument("--contig-filter", help="File with allowed contig names")
    long_parser.add_argument("--window-size", type=int, default=20000, help="Window size (bp)")
    long_parser.add_argument("--max-reads", type=int, default=300, help="Max reads per window")
    long_parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for within-MAG window processing "
        "(default: 1). Set to --cpus-per-task so reserved cores are used.",
    )
    long_parser.add_argument(
        "--min-anchor-weight", type=float, default=0.15, help="Min weight for anchor"
    )
    long_parser.add_argument(
        "--rescued-min-weight", type=float, default=0.02, help="Min weight after rescue"
    )
    long_parser.add_argument(
        "--min-depth-site",
        type=int,
        default=3,
        help="Min VCF DP to load a site. Set 1 for pre-called SNV lists (e.g. "
        "SNooPy) where the BAM re-genotyping is the real depth gate.",
    )
    long_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    # --- depth policy ---
    long_parser.add_argument(
        "--min-reads-per-window", type=int, default=10,
        help="Reads needed to PHASE a window de novo. Below this no haplotype is "
             "invented and the trajectory carries a gap.",
    )
    long_parser.add_argument(
        "--min-reads-for-rescue", type=int, default=5,
        help="Reads needed for a window to be BUILT, so rescue can still populate it. "
             "Clamped to --min-reads-per-window if higher.",
    )
    # --- identity gates ---
    long_parser.add_argument(
        "--min-read-window-overlap-bp", type=int, default=1000,
        help="A read must cover this many bp of a window to be counted in it.",
    )
    long_parser.add_argument(
        "--min-read-read-overlap-bp", type=int, default=1000,
        help="Two reads must physically overlap by this much to be compared.",
    )
    long_parser.add_argument(
        "--min-entity-overlap-bp", type=int, default=1000,
        help="Min physical overlap between two entities; below it the verdict is an "
             "explicit non-merge rather than 'unknown'.",
    )
    # --- within-sample window linking (the HORIZONTAL axis) ---
    long_parser.add_argument(
        "--min-shared-snvs-for-link", type=int, default=3,
        help="Min shared SNV POSITIONS between two overlapping windows before their "
             "haplotypes are even compared.",
    )
    long_parser.add_argument(
        "--min-cosupported-span-frac", type=float, default=0.25,
        help="Min co-supported span between two haplotypes as a fraction of their "
             "shared region. 0.25 rejects ~16%% of adjacent-window pairs; 0.50 rejects ~30%%.",
    )
    # --- step 3 (build_lineages) linking + vetoes -------------------------------
    long_parser.add_argument(
        "--min-shared-reads-for-link", type=int, default=3,
        help="Reads that must sit in BOTH groups before --link-by-read-overlap joins "
             "them. Observed overlap on real joins was ~33 reads, so the default 3 is "
             "a floor against coincidence rather than a real gate.",
    )
    long_parser.add_argument(
        "--no-transitive-abundance-check", action="store_true",
        help="Skip the END-TO-END abundance re-test that cuts a finished lineage where "
             "it drifts. That test's power grows with chain length, so the longest "
             "lineages are the most likely to be cut; turn it off to measure how much "
             "contiguity it is costing.",
    )
    long_parser.add_argument(
        "--lineage-max-bad-frac", type=float, default=0.0,
        help="Fraction of testable samples whose per-sample read shares may disagree "
             "before the abundance veto refuses a lineage continuation. 0.0 is zero "
             "tolerance; 1.0 disables the veto.",
    )
    long_parser.add_argument(
        "--step1-veto-min-timepoints", type=int, default=2,
        help="Distinct timepoints that must flag a within-sample link mismatch before "
             "it vetoes a cross-window lineage continuation.",
    )
    long_parser.add_argument(
        "--identity-distance", type=float, default=0.02,
        help="Max mismatch RATE at which two consensuses are one entity. ONE knob for "
             "every consensus-vs-consensus comparison: the post-EM merge inside a "
             "window, step 1's link across adjacent windows, and steps 2/3 across "
             "samples and along the genome. Read-level thresholds (--max-mismatch-frac, "
             "--rescue-match-distance) are separate on purpose - a read carries "
             "sequencing error a consensus has already averaged out.",
    )
    long_parser.add_argument(
        "--link-window-reach", type=int, default=2,
        help="How many windows ahead step 1 may link. 1 is the pre-2026-08-31 "
             "overlap-only rule, which caps step 1 at a 10 kb reach on a 20 kb/10 kb "
             "tiling while reads reach 30 kb; above 1 also links NON-overlapping "
             "windows on shared reads.",
    )
    long_parser.add_argument(
        "--link-min-shared-reads", type=int, default=2,
        help="Reads two haplotypes must share to link a NON-overlapping window pair. "
             "Those windows call disjoint positions so consensus cannot gate them, "
             "making this and reciprocal best match the entire evidence bar.",
    )
    long_parser.add_argument(
        "--track-merge-min-shared-markers", type=int, default=1,
        help="Agreeing markers two step-1 tracks need before the cross-sample merge "
             "joins them. 1 is a permissive first pass; exact agreement still cannot "
             "fuse genotypes that disagree anywhere both called, so this trades only "
             "evidence volume. Raising it splits hard: on 000089747_1 contig_2, 1 gives "
             "118 entities and 3 gives 4,474.",
    )
    long_parser.add_argument(
        "--min-shared-markers", type=int, default=3,
        help="Shared markers required before that rate means anything. Same scope as "
             "--identity-distance.",
    )
    long_parser.add_argument(
        "--cross-sample-method", choices=["clique", "reciprocal"], default="clique",
        help="How haplotypes are grouped across samples at a fixed window. 'clique' = "
             "complete linkage, no time axis, immune to irregular timepoint spacing. "
             "'reciprocal' = unique-best + mutual between consecutive samples; requires "
             "--samples in true chronological order.",
    )
    # Keep these in step with the same flags on strainphase/longitudinal.py's parser -
    # they are two hand-maintained arg lists over one HaplotyperConfig, and a flag added
    # to only one of them is accepted by `python -m strainphase.longitudinal` but
    # rejected by `strainphase longitudinal`. That is exactly how --seed broke.
    long_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Seeded by default: it drives both read subsampling above "
             "--max-reads and Louvain read clustering, so an unseeded run is not "
             "reproducible.",
    )
    long_parser.add_argument(
        "--no-spill", action="store_true",
        help="Keep every sample's reads in memory until the whole MAG finishes (the old "
             "behaviour). By default reads are parked in <output-dir>/tmp/spill after "
             "each sample is phased and reloaded one sample at a time for rescue.",
    )
    long_parser.add_argument(
        "--window-batch-factor", type=int, default=4,
        help="Windows are dispatched to the worker pool in batches of workers * this. "
             "Lower it to cut peak memory on variant-dense contigs.",
    )
    long_parser.set_defaults(func=cmd_longitudinal)

    # =========== TEST subcommand ===========
    test_parser = subparsers.add_parser(
        "test",
        help="Run test suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    test_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    test_parser.set_defaults(func=cmd_test)

    # =========== VERSION subcommand ===========
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=cmd_version)

    # Parse and execute
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
