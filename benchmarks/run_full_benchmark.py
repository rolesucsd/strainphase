#!/usr/bin/env python3
"""
Unified benchmark pipeline for Strainphase.

Orchestrates the complete benchmark workflow:
1. Simulate HiFi reads from user-provided bacterial genomes
2. Convert SAM to BAM and index
3. Run parameter sweep across configurations (with automatic validation)
4. Generate HTML report with figures
5. Run performance profiling (uses in-memory synthetic data for scalability testing)

This is the PRIMARY entry point for all benchmarking. All validation and
reporting modules are automatically called by this script.

Usage:
    # Full benchmark with synthetic strains (default)
    python benchmarks/run_full_benchmark.py \
        --genomes data/genomes/ \
        --output results/benchmark/ \
        --timepoints 4 \
        --coverage 30 \
        --max-strains 5

    # Full benchmark with real strains (each FASTA = distinct strain)
    python benchmarks/run_full_benchmark.py \
        --genomes data/real_strains/ \
        --output results/benchmark/ \
        --use-real-strains \
        --timepoints 4 \
        --coverage 30

Note: Performance profiling automatically runs and uses in-memory synthetic data
for scalability testing (results may differ from file-based benchmarks).
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
    use_real_strains: bool = False,
    snv_counts: Optional[str] = None,
    fixed_strains_per_genome: Optional[int] = None,
) -> bool:
    """
    Run read simulation from bacterial genomes.

    Uses validation/simulate_reads.py to generate synthetic data.
    
    Args:
        use_real_strains: If True, use FASTA files directly as distinct strains
                         (detect real SNVs instead of introducing synthetic ones)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Simulating reads from genomes")
    if use_real_strains:
        logger.info("Mode: Using real strains (each FASTA = distinct strain)")
    else:
        logger.info("Mode: Creating synthetic strains from genomes")
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
    
    if use_real_strains:
        cmd.append("--use-real-strains")
    if snv_counts:
        cmd.extend(["--snv-counts", snv_counts])
    if fixed_strains_per_genome:
        cmd.extend(["--fixed-strains-per-genome", str(fixed_strains_per_genome)])

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
    verbose: bool = True,
    mode: str = "grid",
    resume: bool = False,
    checkpoint_interval: int = 10,
    passes: int = 1,
    n_workers: int = 1,
    params_file: Optional[str] = None,
    coverage: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run parameter sweep on simulated data.

    Args:
        sim_dir: Directory with simulated data (BAM, VCF files)
        output_dir: Output directory for results
        max_configs: Limit number of configs (grid mode only)
        max_contigs: Limit number of contigs to process
        verbose: Print progress
        mode: "grid" for full sweep, "sequential" for coordinate descent
        resume: Resume from checkpoint if available
        checkpoint_interval: Save checkpoint every N configs
        passes: Number of optimization passes (sequential mode only)
        n_workers: Number of parallel workers for window processing
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Running parameter sweep")
    logger.info("=" * 60)
    logger.info(f"Mode: {mode}")
    if n_workers > 1:
        logger.info(f"Parallel workers: {n_workers}")

    from parameter_sweep import run_parameter_sweep as sweep_func

    sim_path = Path(sim_dir)

    # Find all BAM files (all timepoints)
    bam_files = sorted(sim_path.glob("*.bam"))
    if not bam_files:
        logger.error(f"No BAM files found in {sim_dir}")
        return {}

    # Extract timepoint IDs from BAM filenames (e.g., T1.bam -> T1)
    timepoints = []
    bam_paths = {}
    vcf_paths = {}
    
    for bam_file in bam_files:
        timepoint = bam_file.stem  # e.g., "T1"
        timepoints.append(timepoint)
        bam_paths[timepoint] = str(bam_file)
        
        # Find corresponding VCF file for this timepoint
        vcf_path = sim_path / f"{timepoint}.vcf.gz"
        if not vcf_path.exists():
            vcf_path = sim_path / f"{timepoint}.vcf"
        if not vcf_path.exists():
            # Fallback to shared variants.vcf
            vcf_path = sim_path / "variants.vcf.gz"
            if not vcf_path.exists():
                vcf_path = sim_path / "variants.vcf"
                if not vcf_path.exists():
                    vcf_path = sim_path / "truth_snvs.vcf"
                    if not vcf_path.exists():
                        vcf_path = sim_path / "truth_variants.vcf"
        
        if vcf_path.exists():
            vcf_paths[timepoint] = str(vcf_path)
        else:
            logger.warning(f"No VCF found for timepoint {timepoint}, skipping")
    
    if not timepoints:
        logger.error("No valid timepoints found")
        return {}
    
    logger.info(f"Found {len(timepoints)} timepoints: {timepoints}")
    
    # Find reference FASTA (needed for longitudinal processing)
    reference_path = sim_path / "combined_reference.fasta"
    if not reference_path.exists():
        # Try alternative names
        ref_files = list(sim_path.glob("*.fasta")) + list(sim_path.glob("*.fa"))
        if ref_files:
            reference_path = ref_files[0]
        else:
            logger.error(f"No reference FASTA found in {sim_dir}")
            return {}
    
    # Create .fai index if it doesn't exist (required by parse_reference_contigs)
    fai_path = Path(str(reference_path) + ".fai")
    if not fai_path.exists():
        logger.info(f"Creating FASTA index: {fai_path.name}")
        try:
            # Use pysam.faidx to create the index (no external samtools needed)
            import pysam
            pysam.faidx(str(reference_path))
            if not fai_path.exists():
                raise RuntimeError(f"Index file {fai_path.name} was not created")
            logger.info(f"Created FASTA index: {fai_path.name}")
        except ImportError:
            logger.error("pysam is required to create FASTA index")
            logger.error("Install with: pip install pysam")
            return {}
        except Exception as e:
            logger.error(f"Failed to create FASTA index: {e}")
            logger.error("You can create it manually with:")
            logger.error(f"  samtools faidx {reference_path}")
            logger.error("  or: python -c 'import pysam; pysam.faidx(\"{}\")'".format(reference_path))
            return {}
    
    reference_path = str(reference_path)

    sweep_output = Path(output_dir) / "sweep_results"
    sweep_output.mkdir(parents=True, exist_ok=True)

    summary = sweep_func(
        bam_paths=bam_paths if len(timepoints) > 1 else {timepoints[0]: bam_paths[timepoints[0]]},
        vcf_paths=vcf_paths if len(timepoints) > 1 else {timepoints[0]: vcf_paths[timepoints[0]]},
        reference_path=reference_path if len(timepoints) > 1 else None,
        timepoints=timepoints,
        output_dir=str(sweep_output),
        truth_dir=sim_dir,
        coverage=coverage,
        max_configs=max_configs,
        max_contigs=max_contigs,
        verbose=verbose,
        mode=mode,
        resume=resume,
        checkpoint_interval=checkpoint_interval,
        passes=passes,
        n_workers=n_workers,
        params_file=params_file,
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
    
    NOTE: This uses in-memory synthetic data (different from file-based pipeline)
    for quick performance profiling. Results may differ from file-based benchmarks.
    """
    logger.info("=" * 60)
    logger.info("STEP 6: Running performance benchmark")
    logger.info("=" * 60)
    logger.info("NOTE: Using in-memory synthetic data for performance profiling")

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
    use_real_strains: bool = False,
    snv_counts: Optional[str] = None,
    fixed_strains_per_genome: Optional[int] = None,
    max_configs: Optional[int] = None,
    max_contigs: Optional[int] = None,
    resume: bool = False,
    verbose: bool = True,
    mode: str = "grid",
    passes: int = 1,
    checkpoint_interval: int = 10,
    n_workers: int = 1,
    params_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete benchmark pipeline.

    Args:
        genomes_dir: Directory containing strain FASTA files
        output_dir: Output directory for all results
        n_timepoints: Number of timepoints to simulate
        coverage: Read coverage per timepoint
        snv_density: SNVs per 10kb to introduce (only for synthetic mode)
        error_rate: Sequencing error rate
        seed: Random seed
        max_strains: Limit number of strains (only for synthetic mode)
        use_real_strains: If True, use FASTA files directly as distinct strains
                         (detect real SNVs instead of introducing synthetic ones)
        snv_counts: Comma-separated SNV counts for strains[1:] (exact overrides density)
        fixed_strains_per_genome: Exact number of strains per genome (synthetic mode)
        max_configs: Limit number of configs (grid mode only)
        max_contigs: Limit number of contigs
        resume: Resume from checkpoint
        verbose: Print progress
        mode: "grid" for full sweep, "sequential" for coordinate descent
        passes: Number of optimization passes (sequential mode)
        checkpoint_interval: Save checkpoint every N configs
        n_workers: Number of parallel workers for window processing

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
    if use_real_strains:
        logger.info("Strain mode: Using real strains (each FASTA = distinct strain)")
    else:
        logger.info("Strain mode: Creating synthetic strains from genomes")
    logger.info(f"Sweep mode: {mode}")
    if mode == "sequential":
        logger.info(f"Optimization passes: {passes}")
    else:
        logger.info(f"Max parameter configs: {max_configs or 'all'}")
    if n_workers > 1:
        logger.info(f"Parallel workers: {n_workers}")
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
            "snv_counts": snv_counts,
            "fixed_strains_per_genome": fixed_strains_per_genome,
            "error_rate": error_rate,
            "seed": seed,
            "max_strains": max_strains,
            "max_configs": max_configs,
        },
        "steps": {}
    }

    # Step 1: Simulate reads
    step_start = time.time()
    if resume and sim_dir.exists():
        logger.info("Resume enabled: skipping read simulation")
        success = True
    else:
        success = simulate_reads(
            genomes_dir=genomes_dir,
            output_dir=str(sim_dir),
            n_timepoints=n_timepoints,
            coverage=coverage,
            snv_density=snv_density,
            error_rate=error_rate,
            seed=seed,
            max_strains=max_strains,
            use_real_strains=use_real_strains,
            snv_counts=snv_counts,
            fixed_strains_per_genome=fixed_strains_per_genome,
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
    bam_files = list(sim_dir.glob("*.bam")) if sim_dir.exists() else []
    if resume and bam_files:
        logger.info("Resume enabled: skipping SAM->BAM conversion")
        success = True
    else:
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
    vcf_candidates = [sim_dir / "variants.vcf.gz", sim_dir / "variants.vcf"]
    vcf_exists = any(p.exists() for p in vcf_candidates)
    if resume and vcf_exists:
        logger.info("Resume enabled: skipping VCF preparation")
        success = True
    else:
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
        verbose=verbose,
        mode=mode,
        resume=resume,
        checkpoint_interval=checkpoint_interval,
        passes=passes,
        n_workers=n_workers,
        params_file=params_file,
        coverage=coverage,
    )
    results["steps"]["parameter_sweep"] = {
        "success": bool(sweep_summary),
        "duration_seconds": time.time() - step_start,
        "summary": sweep_summary
    }

    # Ensure coverage metadata exists in sweep_results.json for reporting
    sweep_results_path = output_path / "sweep_results" / "sweep_results.json"
    if coverage is not None and sweep_results_path.exists():
        try:
            with open(sweep_results_path) as f:
                sweep_results_data = json.load(f)
            updated = False
            for r in sweep_results_data:
                if r.get("coverage") is None:
                    r["coverage"] = coverage
                    updated = True
            if updated:
                with open(sweep_results_path, "w") as f:
                    json.dump(sweep_results_data, f, indent=2, default=str)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not update sweep_results.json with coverage: {e}")

    # Step 5: Generate report
    step_start = time.time()
    sweep_dir = str(output_path / "sweep_results")
    success = generate_report(
        sweep_dir=sweep_dir,
        output_dir=output_dir,
        validation_dir=sweep_dir
    )
    results["steps"]["generate_report"] = {
        "success": success,
        "duration_seconds": time.time() - step_start
    }

    # Step 6: Performance benchmark (always run)
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
    parser.add_argument("--snv-counts", type=str, default=None,
                        help="Comma-separated SNV counts for strains[1:] (exact overrides density)")
    parser.add_argument("--error-rate", type=float, default=0.001,
                        help="Sequencing error rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-strains", type=int,
                        help="Limit number of strains to use (only for synthetic mode)")
    parser.add_argument("--use-real-strains", action="store_true",
                        help="Use FASTA files directly as distinct strains (detect real SNVs instead of introducing synthetic ones)")
    parser.add_argument("--fixed-strains-per-genome", type=int, default=None,
                        help="Use an exact number of strains per genome (synthetic mode only)")

    # Sweep parameters
    parser.add_argument("--max-configs", type=int,
                        help="Limit number of parameter configs to test (grid mode only)")
    parser.add_argument("--max-contigs", type=int,
                        help="Limit number of contigs to process")
    parser.add_argument("--params", dest="params_file",
                        help="Custom parameter grid JSON file (optional)")

    # Mode selection
    parser.add_argument("--mode", choices=["grid", "sequential"], default="grid",
                        help="Optimization mode: 'grid' for full sweep (13,824 configs), "
                             "'sequential' for coordinate descent (~27 configs)")

    # Checkpointing
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Save checkpoint every N configs")

    # Sequential mode options
    parser.add_argument("--passes", type=int, default=1,
                        help="Number of optimization passes (sequential mode only)")

    # Parallelization
    parser.add_argument("-j", "--workers", type=int, default=1,
                        help="Number of parallel workers for window processing (default: 1)")

    # Optional steps
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")

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
        use_real_strains=args.use_real_strains,
        snv_counts=args.snv_counts,
        fixed_strains_per_genome=args.fixed_strains_per_genome,
        max_configs=args.max_configs,
        max_contigs=args.max_contigs,
        resume=args.resume,
        verbose=not args.quiet,
        mode=args.mode,
        passes=args.passes,
        checkpoint_interval=args.checkpoint_interval,
        n_workers=args.workers,
        params_file=args.params_file,
    )

    # Exit with error if any critical step failed
    critical_steps = ["simulate_reads", "sam_to_bam", "prepare_vcf"]
    for step in critical_steps:
        if not results.get("steps", {}).get(step, {}).get("success", False):
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
