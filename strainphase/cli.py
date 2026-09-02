#!/usr/bin/env python3
"""
Strainphase command-line interface.

Usage:
    strainphase run          # Process single contig
    strainphase longitudinal # Process MAG across timepoints
    strainphase sv           # Sidecar tools (reconcile, verify)
    strainphase test         # Run test suite
    strainphase version      # Show version
"""

from __future__ import annotations

import argparse

from strainphase.core import DEFAULT_CONFIG as _D
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
    from strainphase import process_contig, results_to_dataframe
    from strainphase.core import config_from_args

    setup_logging(args.log_level)

    config = config_from_args(args)

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


def _report_mem_profile() -> None:
    """Log tracemalloc peak + top allocation sites. Call at the integration point."""
    import collections
    import tracemalloc
    snap = tracemalloc.take_snapshot()
    cur, peak = tracemalloc.get_traced_memory()
    logging.info(f"[mem-profile] tracemalloc peak={peak / 1e9:.2f} GB  resident-now={cur / 1e9:.2f} GB")
    by_file: collections.Counter = collections.Counter()
    for st in snap.statistics("lineno"):
        by_file[os.path.basename(st.traceback[0].filename)] += st.size
    logging.info("[mem-profile] top modules (resident at integration point):")
    for fn_, sz in by_file.most_common(8):
        logging.info(f"[mem-profile]   {sz / 1e9:6.3f} GB  {fn_}")
    logging.info("[mem-profile] top 15 allocation sites:")
    for st in snap.statistics("lineno")[:15]:
        fr = st.traceback[0]
        logging.info(f"[mem-profile]   {st.size / 1e9:6.3f} GB  {os.path.basename(fr.filename)}:{fr.lineno}")
    tracemalloc.stop()


def cmd_longitudinal(args: argparse.Namespace) -> int:
    """Run longitudinal analysis across multiple samples."""
    setup_logging(args.log_level)

    _mem_profile = getattr(args, "mem_profile", False)
    if _mem_profile:
        import tracemalloc
        tracemalloc.start(25)

    # Import here to avoid pysam requirement for other commands
    import importlib.util

    if importlib.util.find_spec("pysam") is None:
        logging.error(
            "pysam is required for longitudinal analysis. Install with: pip install pysam"
        )
        return 1

    from strainphase.core import config_from_args
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

    # A --bams template without {sample} resolves to ONE BAM for every timepoint, so
    # every timepoint is phased from one sample's reads and the run looks normal. A
    # shared --vcfs or --sv-sidecars is legitimate by contrast (variant catalogues);
    # the BAM is what makes a sample a sample.
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
    config = config_from_args(args)

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

    if _mem_profile:
        _report_mem_profile()

    # ---- Window-level tables (the deliverables) ----
    # lineages.tsv comes back from here too: the cross-sample track merge that
    # produces the lineages runs inside build_window_tables.
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
        f"{len(across_rows)} across-sample window groups | comparisons: "
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


# Tuning flags shared by every subcommand that phases reads. `run` calls
# process_contig, which calls link_windows, so these gates already act on `run`.
# Declared once so `run` and `longitudinal` cannot drift.
def _add_phasing_args(p) -> None:
    p.add_argument("--window-size", type=int, default=_D.window_size, help="Window size (bp)")
    p.add_argument("--max-reads", type=int, default=_D.max_reads_per_window,
                   help="Max reads per window")
    p.add_argument("--min-mapq", type=int, default=_D.min_mapq, help="Minimum MAPQ")
    p.add_argument(
        "--min-depth-site", type=int, default=_D.min_depth_site,
        help="Min VCF DP to load a site. Set 1 for pre-called SNV lists (e.g. SNooPy) "
             "where the BAM re-genotyping is the real depth gate.",
    )
    p.add_argument(
        "--af-range", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None,
        help="Only load VCF sites whose alt-allele frequency falls in [LOW, HIGH). "
             "Off by default, which loads every site that clears --min-depth-site.",
    )
    p.add_argument(
        "--seed", type=int, default=_D.random_seed,
        help="Random seed. Seeded by default: it drives both read subsampling above "
             "--max-reads and Louvain read clustering, so an unseeded run is not "
             "reproducible.",
    )
    # --- depth policy ---
    p.add_argument(
        "--min-reads-per-window", type=int, default=_D.min_reads_per_window,
        help="Reads needed to PHASE a window de novo.",
    )
    p.add_argument(
        "--min-reads-for-rescue", type=int, default=_D.min_reads_for_rescue,
        help="Reads needed for a window to be BUILT, so rescue can still populate it.",
    )
    # --- identity: ONE rate, and per-stage evidence ---
    p.add_argument(
        "--identity-distance", type=float, default=_D.identity_distance,
        help="Max mismatch RATE at which two things are one entity. ONE rate for every "
             "comparison - read to read, haplotype to haplotype, and rescue. How much "
             "evidence each needs first is a separate per-stage threshold.",
    )
    p.add_argument(
        "--min-shared-markers", type=int, default=_D.min_shared_markers,
        help="Shared markers required before that rate means anything, comparing two "
             "haplotype consensuses.",
    )
    p.add_argument(
        "--min-shared-snvs-for-link", type=int, default=_D.min_shared_snvs_for_link,
        help="Shared SNV POSITIONS two overlapping windows need before their haplotypes "
             "are compared at all. A cheap precheck for --min-shared-markers.",
    )
    p.add_argument(
        "--min-overlap-bp", type=int, default=_D.min_entity_overlap_bp,
        help="Physical overlap two things must share before they are compared at all - "
             "read against window, read against read, and entity against entity.",
    )
    p.add_argument(
        "--min-cosupported-span-frac", type=float, default=_D.min_cosupported_span_frac,
        help="Min co-supported span between two haplotypes as a fraction of their "
             "shared region.",
    )
    # --- abundance: one threshold for every abundance verdict ---
    p.add_argument(
        "--abundance-coherence-alpha", type=float, default=_D.abundance_coherence_alpha,
        help="Significance at which two read counts are declared incompatible "
             "abundances. Step 1's eliminator and the merge's gap-filling test share it.",
    )
    p.add_argument(
        "--min-reads-for-coherence", type=int, default=_D.min_reads_for_coherence,
        help="Reads a window needs before its abundance is tested at all.",
    )
    # --- step-1 linking ---
    p.add_argument(
        "--link-window-reach", type=int, default=_D.link_window_reach,
        help="How many windows ahead step 1 may link. 1 is the overlap-only rule; above "
             "1 also links NON-overlapping windows on shared reads.",
    )
    p.add_argument(
        "--link-min-shared-reads", type=int, default=_D.link_min_shared_reads,
        help="Reads two haplotypes must share to link a NON-overlapping window pair. "
             "Consensus cannot gate those, so this is the whole evidence bar.",
    )
    p.add_argument(
        "--link-shared-read-frac", type=float, default=_D.link_shared_read_frac,
        help="Coverage-invariant add-on to --link-min-shared-reads: a NON-overlapping link "
             "also needs shared reads >= this fraction of each haplotype's continuing "
             "reads. Scales the bar with depth so high coverage stops over-merging. "
             "0.0 = off (default); ~0.25 is a good starting value.",
    )


# Flags that only mean something with MANY samples: the cross-sample merge, the
# cross-timepoint anchor panel, and the multi-sample driver's memory behaviour.
def _add_longitudinal_args(p) -> None:
    p.add_argument(
        "--track-merge-min-shared-markers", type=int,
        default=_D.track_merge_min_shared_markers,
        help="Agreeing markers two step-1 tracks need before the cross-sample merge "
             "joins them. 1 is a permissive first pass.",
    )
    p.add_argument("--min-anchor-weight", type=float, default=_D.min_weight_for_anchor,
                   help="Min weight for anchor")
    p.add_argument("--rescued-min-weight", type=float, default=_D.rescued_min_weight,
                   help="Min weight after rescue")
    p.add_argument(
        "--no-spill", action="store_true",
        help="Keep every sample's reads in memory until the whole MAG finishes.",
    )
    p.add_argument(
        "--window-batch-factor", type=int, default=_D.window_batch_factor,
        help="Windows are dispatched to the worker pool in batches of workers * this. "
             "Lower it to cut peak memory on variant-dense contigs.",
    )
    p.add_argument(
        "--mem-profile", action="store_true",
        help="Trace peak memory with tracemalloc and log the top allocation sites at the "
             "integration point. Diagnostic only; adds tracing overhead.",
    )


def cmd_sv(args) -> int:
    from strainphase.sv_encoding import run_sv

    return run_sv(args.sv_args)


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
    run_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    _add_phasing_args(run_parser)
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
    long_parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for within-MAG window processing "
        "(default: 1). Set to --cpus-per-task so reserved cores are used.",
    )
    long_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    _add_phasing_args(long_parser)
    _add_longitudinal_args(long_parser)
    long_parser.set_defaults(func=cmd_longitudinal)

    # =========== SV subcommand ===========
    sv_parser = subparsers.add_parser(
        "sv", help="SV sidecar tools (reconcile, verify)",
    )
    sv_parser.add_argument(
        "sv_args", nargs=argparse.REMAINDER,
        help="{reconcile,verify} plus that tool's own arguments",
    )
    sv_parser.set_defaults(func=cmd_sv)

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
