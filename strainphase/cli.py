#!/usr/bin/env python3
"""
Strainphase command-line interface.

Usage:
    strainphase run          # Process single contig
    strainphase longitudinal # Process MAG across timepoints
    strainphase test         # Run test suite
    strainphase sweep        # Run parameter sensitivity analysis
    strainphase version      # Show version
"""

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
        build_lineage_table,
        load_allowed_contigs,
        parse_reference_contigs,
        process_mag_longitudinal,
    )

    # Parse samples
    samples = [s.strip() for s in args.samples.split(",")]
    logging.info(f"Processing {len(samples)} samples: {samples}")

    # Build path mappings
    bam_paths = {s: args.bams.format(sample=s) for s in samples}
    vcf_paths = {s: args.vcfs.format(sample=s) for s in samples}

    # Verify files exist
    for sample in samples:
        if not os.path.exists(bam_paths[sample]):
            logging.error(f"BAM not found: {bam_paths[sample]}")
            return 1
        if not os.path.exists(vcf_paths[sample]):
            logging.error(f"VCF not found: {vcf_paths[sample]}")
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
        validate_results=False,
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
        )
        all_results[mag_name] = results
        if integrator:
            all_integrators.append(integrator)

    # Build lineage table
    lineage_records = build_lineage_table(all_results, config)

    # Write outputs
    if lineage_records:
        import csv

        output_path = os.path.join(args.output_dir, "lineages.tsv")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=lineage_records[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(lineage_records)
        logging.info(f"Wrote {len(lineage_records)} lineage records to {output_path}")

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


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run parameter sensitivity analysis."""
    setup_logging("INFO")

    print(f"\n{'=' * 60}")
    print(f"STRAINPHASE v{__version__} - PARAMETER SWEEP")
    print(f"{'=' * 60}\n")

    try:
        import subprocess

        output_dir = args.output_dir or "strainphase_sweep_results"
        cmd = ["python", "benchmarks/parameter_sweep.py"]
        if args.max_configs:
            # Pass via environment or modify script to accept args
            pass
        print("Running parameter sweep from benchmarks/parameter_sweep.py...")
        print(f"Results will be saved to: {output_dir}\n")
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode
    except Exception as e:
        logging.error(f"Could not run sweep: {e}")
        logging.info("Try running: python benchmarks/parameter_sweep.py")
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

    # Parameter sweep
    strainphase sweep --quick
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
    run_parser.add_argument("--window-size", type=int, default=3000, help="Window size (bp)")
    run_parser.add_argument("--max-reads", type=int, default=300, help="Max reads per window")
    run_parser.add_argument("--min-mapq", type=int, default=20, help="Minimum MAPQ")
    run_parser.add_argument(
        "--max-mismatch", type=float, default=0.02, help="Max mismatch fraction"
    )
    run_parser.add_argument("--seed", type=int, help="Random seed")
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
    long_parser.add_argument("--reference", required=True, help="Reference FASTA")
    long_parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    long_parser.add_argument("--mags", help="Comma-separated MAG names (default: all)")
    long_parser.add_argument("--contig-filter", help="File with allowed contig names")
    long_parser.add_argument("--window-size", type=int, default=3000, help="Window size (bp)")
    long_parser.add_argument("--max-reads", type=int, default=300, help="Max reads per window")
    long_parser.add_argument(
        "--min-anchor-weight", type=float, default=0.15, help="Min weight for anchor"
    )
    long_parser.add_argument(
        "--rescued-min-weight", type=float, default=0.02, help="Min weight after rescue"
    )
    long_parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
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

    # =========== SWEEP subcommand ===========
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run parameter sensitivity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sweep_parser.add_argument("--quick", action="store_true", help="Use reduced parameter grid")
    sweep_parser.add_argument("--output-dir", "-o", help="Output directory")
    sweep_parser.add_argument("--max-configs", type=int, help="Limit number of configurations")
    sweep_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    sweep_parser.set_defaults(func=cmd_sweep)

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
