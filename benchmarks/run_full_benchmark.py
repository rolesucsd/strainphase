#!/usr/bin/env python3
"""
Unified benchmark pipeline for Strainphase.

Orchestrates the complete benchmark workflow:
1. Simulate HiFi reads from user-provided bacterial genomes
2. Convert SAM to BAM and index
3. Run parameter sweep across configurations
4. Generate HTML report with figures
5. Run performance profiling (optional)

Usage:
    # Full benchmark with 1 genome (test run)
    python benchmarks/run_full_benchmark.py \
        --genomes data/test_genomes/ \
        --output results/benchmark/ \
        --max-configs 5

    # Full benchmark with all options
    python benchmarks/run_full_benchmark.py \
        --genomes data/genomes/ \
        --output results/benchmark/ \
        --timepoints 4 \
        --coverage 30 \
        --include-performance
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Check that required Python packages are available."""
    try:
        import pysam
        return True
    except ImportError:
        logger.error("pysam is required but not installed")
        logger.error("Install with: pip install pysam")
        return False


def run_command(cmd: list, description: str, cwd: Optional[str] = None) -> bool:
    """Run a command and log output."""
    logger.info(f"{description}...")
    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Command failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False


def simulate_reads(
    genomes_dir: str,
    output_dir: str,
    n_timepoints: int = 4,
    coverage: int = 30,
    snv_density: int = 10,
    error_rate: float = 0.001,
    seed: int = 42,
    max_strains: Optional[int] = None,
) -> bool:
    """
    Run read simulation from bacterial genomes.

    Uses validation/simulate_reads.py to generate synthetic data.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Simulating reads from genomes")
    logger.info("=" * 60)

    # Build command
    script_path = Path(__file__).parent.parent / "validation" / "simulate_reads.py"

    cmd = [
        sys.executable, str(script_path),
        "--genomes", genomes_dir,
        "--output", output_dir,
        "--timepoints", str(n_timepoints),
        "--coverage", str(coverage),
        "--snv-density", str(snv_density),
        "--error-rate", str(error_rate),
        "--seed", str(seed),
    ]

    if max_strains:
        cmd.extend(["--max-strains", str(max_strains)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Simulation failed: {result.stderr}")
        return False

    logger.info(result.stdout)
    logger.info("Read simulation complete")
    return True


def convert_sam_to_bam(sim_dir: str) -> bool:
    """
    Convert SAM files to sorted, indexed BAM files using pysam.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Converting SAM to BAM and indexing")
    logger.info("=" * 60)

    import pysam

    sim_path = Path(sim_dir)
    sam_files = list(sim_path.glob("*.sam"))

    if not sam_files:
        logger.error(f"No SAM files found in {sim_dir}")
        return False

    for sam_file in sam_files:
        timepoint = sam_file.stem  # e.g., "T1"
        bam_file = sim_path / f"{timepoint}.bam"
        unsorted_bam = sim_path / f"{timepoint}.unsorted.bam"

        logger.info(f"Converting {sam_file.name} -> {bam_file.name}")

        try:
            # Convert SAM to BAM
            with pysam.AlignmentFile(str(sam_file), "r") as sam_in:
                with pysam.AlignmentFile(str(unsorted_bam), "wb", header=sam_in.header) as bam_out:
                    for read in sam_in:
                        bam_out.write(read)

            # Sort BAM
            pysam.sort("-o", str(bam_file), str(unsorted_bam))

            # Index BAM
            pysam.index(str(bam_file))

            # Clean up unsorted BAM
            unsorted_bam.unlink()

            logger.info(f"  Created {bam_file.name} and {bam_file.name}.bai")

        except Exception as e:
            logger.error(f"BAM conversion failed for {sam_file.name}: {e}")
            return False

    logger.info("BAM conversion complete")
    return True


def create_vcf_from_truth(sim_dir: str) -> bool:
    """
    Create a VCF file suitable for strainphase from the truth VCF.

    The truth VCF has all SNV positions; we need to make it compatible
    with the strainphase input format.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Preparing VCF for strainphase")
    logger.info("=" * 60)

    sim_path = Path(sim_dir)
    truth_vcf = sim_path / "truth_snvs.vcf"
    output_vcf = sim_path / "variants.vcf"

    if not truth_vcf.exists():
        truth_vcf = sim_path / "truth_variants.vcf"

    if not truth_vcf.exists():
        logger.error(f"No truth VCF found in {sim_dir}")
        return False

    # Copy the truth VCF - strainphase can handle uncompressed VCF
    shutil.copy(truth_vcf, output_vcf)
    logger.info(f"Copied {truth_vcf.name} -> {output_vcf.name}")

    # Try to compress and index with pysam if available
    try:
        import pysam
        compressed_vcf = sim_path / "variants.vcf.gz"
        pysam.tabix_compress(str(output_vcf), str(compressed_vcf), force=True)
        pysam.tabix_index(str(compressed_vcf), preset="vcf", force=True)
        logger.info(f"Created compressed and indexed VCF: {compressed_vcf.name}")
    except Exception as e:
        logger.warning(f"Could not compress/index VCF (will use uncompressed): {e}")

    logger.info("VCF preparation complete")
    return True


def run_parameter_sweep(
    sim_dir: str,
    output_dir: str,
    max_configs: Optional[int] = None,
    max_contigs: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run parameter sweep on simulated data.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Running parameter sweep")
    logger.info("=" * 60)

    from parameter_sweep import run_parameter_sweep as sweep_func

    sim_path = Path(sim_dir)

    # Find BAM file (use first timepoint)
    bam_files = sorted(sim_path.glob("*.bam"))
    if not bam_files:
        logger.error(f"No BAM files found in {sim_dir}")
        return {}

    bam_path = str(bam_files[0])

    # Find VCF file
    vcf_path = sim_path / "variants.vcf.gz"
    if not vcf_path.exists():
        vcf_path = sim_path / "variants.vcf"
    if not vcf_path.exists():
        vcf_path = sim_path / "truth_snvs.vcf"
    if not vcf_path.exists():
        vcf_path = sim_path / "truth_variants.vcf"

    vcf_path = str(vcf_path)

    sweep_output = Path(output_dir) / "sweep_results"
    sweep_output.mkdir(parents=True, exist_ok=True)

    summary = sweep_func(
        bam_path=bam_path,
        vcf_path=vcf_path,
        output_dir=str(sweep_output),
        truth_dir=sim_dir,
        max_configs=max_configs,
        max_contigs=max_contigs,
        verbose=verbose
    )

    return summary


def generate_report(
    sweep_dir: str,
    output_dir: str,
    validation_dir: Optional[str] = None
) -> bool:
    """
    Generate HTML benchmark report.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Generating benchmark report")
    logger.info("=" * 60)

    from generate_report import generate_report as report_func

    report_output = Path(output_dir) / "report"
    report_output.mkdir(parents=True, exist_ok=True)

    report_path = report_func(
        results_dir=sweep_dir,
        output_dir=str(report_output),
        validation_dir=validation_dir
    )

    if report_path:
        logger.info(f"Report generated: {report_path}")
        return True
    else:
        logger.warning("Report generation failed or skipped")
        return False


def run_performance_benchmark(
    sim_dir: str,
    output_dir: str,
    quick: bool = True
) -> bool:
    """
    Run performance profiling benchmark.
    """
    logger.info("=" * 60)
    logger.info("STEP 6: Running performance benchmark")
    logger.info("=" * 60)

    script_path = Path(__file__).parent / "benchmark_performance.py"

    perf_output = Path(output_dir) / "performance"
    perf_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(script_path),
        "--output", str(perf_output),
    ]

    if quick:
        cmd.append("--quick")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Performance benchmark failed: {result.stderr}")
        return False

    logger.info(result.stdout)
    logger.info("Performance benchmark complete")
    return True


def run_full_benchmark(
    genomes_dir: str,
    output_dir: str,
    n_timepoints: int = 4,
    coverage: int = 30,
    snv_density: int = 10,
    error_rate: float = 0.001,
    seed: int = 42,
    max_strains: Optional[int] = None,
    max_configs: Optional[int] = None,
    max_contigs: Optional[int] = None,
    include_performance: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the complete benchmark pipeline.

    Returns summary dict with all results.
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("STRAINPHASE FULL BENCHMARK PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Input genomes: {genomes_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Timepoints: {n_timepoints}")
    logger.info(f"Coverage: {coverage}x")
    logger.info(f"Max parameter configs: {max_configs or 'all'}")
    logger.info("=" * 60)

    # Check dependencies
    if not check_dependencies():
        return {"error": "Missing dependencies"}

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Simulation output directory
    sim_dir = output_path / "simulated_data"

    results = {
        "genomes_dir": genomes_dir,
        "output_dir": output_dir,
        "parameters": {
            "n_timepoints": n_timepoints,
            "coverage": coverage,
            "snv_density": snv_density,
            "error_rate": error_rate,
            "seed": seed,
            "max_strains": max_strains,
            "max_configs": max_configs,
        },
        "steps": {}
    }

    # Step 1: Simulate reads
    step_start = time.time()
    success = simulate_reads(
        genomes_dir=genomes_dir,
        output_dir=str(sim_dir),
        n_timepoints=n_timepoints,
        coverage=coverage,
        snv_density=snv_density,
        error_rate=error_rate,
        seed=seed,
        max_strains=max_strains,
    )
    results["steps"]["simulate_reads"] = {
        "success": success,
        "duration_seconds": time.time() - step_start
    }
    if not success:
        logger.error("Simulation failed, aborting pipeline")
        return results

    # Step 2: Convert SAM to BAM
    step_start = time.time()
    success = convert_sam_to_bam(str(sim_dir))
    results["steps"]["sam_to_bam"] = {
        "success": success,
        "duration_seconds": time.time() - step_start
    }
    if not success:
        logger.error("BAM conversion failed, aborting pipeline")
        return results

    # Step 3: Prepare VCF
    step_start = time.time()
    success = create_vcf_from_truth(str(sim_dir))
    results["steps"]["prepare_vcf"] = {
        "success": success,
        "duration_seconds": time.time() - step_start
    }
    if not success:
        logger.error("VCF preparation failed, aborting pipeline")
        return results

    # Step 4: Parameter sweep
    step_start = time.time()
    sweep_summary = run_parameter_sweep(
        sim_dir=str(sim_dir),
        output_dir=output_dir,
        max_configs=max_configs,
        max_contigs=max_contigs,
        verbose=verbose
    )
    results["steps"]["parameter_sweep"] = {
        "success": bool(sweep_summary),
        "duration_seconds": time.time() - step_start,
        "summary": sweep_summary
    }

    # Step 5: Generate report
    step_start = time.time()
    sweep_dir = str(output_path / "sweep_results")
    success = generate_report(
        sweep_dir=sweep_dir,
        output_dir=output_dir,
        validation_dir=str(sim_dir)
    )
    results["steps"]["generate_report"] = {
        "success": success,
        "duration_seconds": time.time() - step_start
    }

    # Step 6: Performance benchmark (optional)
    if include_performance:
        step_start = time.time()
        success = run_performance_benchmark(
            sim_dir=str(sim_dir),
            output_dir=output_dir,
            quick=True
        )
        results["steps"]["performance_benchmark"] = {
            "success": success,
            "duration_seconds": time.time() - step_start
        }

    # Total time
    total_time = time.time() - start_time
    results["total_duration_seconds"] = total_time

    # Save results summary
    summary_file = output_path / "benchmark_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print final summary
    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total duration: {total_time:.1f} seconds")
    logger.info(f"Results saved to: {output_dir}")

    for step_name, step_info in results["steps"].items():
        status = "OK" if step_info.get("success") else "FAILED"
        duration = step_info.get("duration_seconds", 0)
        logger.info(f"  {step_name}: {status} ({duration:.1f}s)")

    if sweep_summary and "scenarios" in sweep_summary:
        logger.info("\nSweep Results:")
        for scenario, stats in sweep_summary.get("scenarios", {}).items():
            n_lin = stats.get("n_lineages", {})
            logger.info(f"  {scenario}: {n_lin.get('mean', 0):.1f} lineages (mean)")
            if stats.get("snv_f1") is not None:
                logger.info(f"    SNV F1: {stats['snv_f1']:.3f}")

    report_path = output_path / "report" / "benchmark_report.html"
    if report_path.exists():
        logger.info(f"\nView report: {report_path}")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete Strainphase benchmark pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument("--genomes", required=True,
                        help="Directory containing strain FASTA files")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for all results")

    # Simulation parameters
    parser.add_argument("--timepoints", type=int, default=4,
                        help="Number of timepoints to simulate")
    parser.add_argument("--coverage", type=int, default=30,
                        help="Read coverage per timepoint")
    parser.add_argument("--snv-density", type=int, default=10,
                        help="SNVs per 10kb to introduce")
    parser.add_argument("--error-rate", type=float, default=0.001,
                        help="Sequencing error rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-strains", type=int,
                        help="Limit number of strains to use")

    # Sweep parameters
    parser.add_argument("--max-configs", type=int,
                        help="Limit number of parameter configs to test")
    parser.add_argument("--max-contigs", type=int,
                        help="Limit number of contigs to process")

    # Optional steps
    parser.add_argument("--include-performance", action="store_true",
                        help="Include performance profiling benchmark")

    # Verbosity
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Reduce output verbosity")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    results = run_full_benchmark(
        genomes_dir=args.genomes,
        output_dir=args.output,
        n_timepoints=args.timepoints,
        coverage=args.coverage,
        snv_density=args.snv_density,
        error_rate=args.error_rate,
        seed=args.seed,
        max_strains=args.max_strains,
        max_configs=args.max_configs,
        max_contigs=args.max_contigs,
        include_performance=args.include_performance,
        verbose=not args.quiet
    )

    # Exit with error if any critical step failed
    critical_steps = ["simulate_reads", "sam_to_bam", "prepare_vcf"]
    for step in critical_steps:
        if not results.get("steps", {}).get(step, {}).get("success", False):
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
