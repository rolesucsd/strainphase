#!/usr/bin/env python3
"""
Performance benchmarking script for Strainphase.

Measures runtime, memory usage, and scalability across different parameters.

Usage:
    python benchmarks/benchmark_performance.py --output results/
    python benchmarks/benchmark_performance.py --quick  # Fast benchmarks
"""

import argparse
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple
import json

from strainphase import HaplotyperConfig, process_window, link_windows
from strainphase.simulation import SyntheticDataGenerator, SimulationScenario

import numpy as np


class PerformanceBenchmark:
    """Track performance metrics."""

    def __init__(self):
        self.results = []

    def benchmark_scenario(
        self,
        name: str,
        scenario: SimulationScenario,
        config: HaplotyperConfig,
        n_reads: int,
        coverage: int
    ) -> Dict:
        """
        Benchmark a single scenario.

        Returns dict with timing and memory stats.
        """
        logging.info(f"Benchmarking: {name}")
        logging.info(f"  n_reads={n_reads}, coverage={coverage}x")

        generator = SyntheticDataGenerator(seed=config.random_seed or 42)

        # Generate window
        logging.info("  Generating synthetic data...")
        window = generator.generate_window(
            scenario=scenario,
            timepoint=scenario.timepoints[0],
            window_start=1,
            window_end=scenario.contig_length,
            n_reads=n_reads,
            coverage=coverage,
            read_length=10000,
            error_rate=0.001,
        )

        logging.info(f"  Window: {len(window.reads)} reads, {len(window.snv_pos)} SNVs")

        # Benchmark processing
        tracemalloc.start()
        start_time = time.time()

        result = process_window(window, config)

        elapsed = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Collect metrics
        metrics = {
            'name': name,
            'n_reads': len(window.reads),
            'n_snvs': len(window.snv_pos),
            'coverage': coverage,
            'n_haplotypes_detected': len(result.haplotypes),
            'elapsed_seconds': elapsed,
            'peak_memory_mb': peak_mem / 1024 / 1024,
            'converged': result.converged,
            'iterations': result.iterations,
            'log_likelihood': result.log_likelihood,
        }

        logging.info(f"  Time: {elapsed:.2f}s")
        logging.info(f"  Memory: {peak_mem/1024/1024:.1f} MB")
        logging.info(f"  Haplotypes: {len(result.haplotypes)}")

        self.results.append(metrics)
        return metrics

    def save_results(self, output_file: Path):
        """Save results to JSON."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logging.info(f"Saved results to {output_file}")

    def summary(self) -> str:
        """Get summary statistics."""
        if not self.results:
            return "No results"

        lines = [
            "\n" + "=" * 60,
            "PERFORMANCE BENCHMARK SUMMARY",
            "=" * 60,
        ]

        # Overall stats
        times = [r['elapsed_seconds'] for r in self.results]
        mems = [r['peak_memory_mb'] for r in self.results]

        lines.extend([
            f"Scenarios tested: {len(self.results)}",
            f"Mean runtime: {np.mean(times):.2f}s (±{np.std(times):.2f})",
            f"Mean memory: {np.mean(mems):.1f} MB (±{np.std(mems):.1f})",
            "=" * 60,
            "",
            "Per-scenario results:",
        ])

        # Per-scenario
        for r in self.results:
            lines.append(
                f"  {r['name']:30s} | "
                f"{r['n_reads']:4d} reads | "
                f"{r['n_snvs']:3d} SNVs | "
                f"{r['elapsed_seconds']:6.2f}s | "
                f"{r['peak_memory_mb']:6.1f} MB | "
                f"{r['n_haplotypes_detected']} haps"
            )

        lines.append("=" * 60)

        return "\n".join(lines)


def scalability_test(
    benchmark: PerformanceBenchmark,
    base_scenario: SimulationScenario,
    config: HaplotyperConfig,
    parameter: str,
    values: List,
):
    """
    Test scalability by varying a parameter.

    Parameters:
        parameter: 'n_reads', 'coverage', 'window_size', 'n_snvs'
        values: List of values to test
    """
    logging.info(f"\nScalability test: {parameter}")

    for value in values:
        if parameter == 'n_reads':
            name = f"scale_reads_{value}"
            benchmark.benchmark_scenario(
                name=name,
                scenario=base_scenario,
                config=config,
                n_reads=value,
                coverage=50
            )

        elif parameter == 'coverage':
            name = f"scale_coverage_{value}x"
            benchmark.benchmark_scenario(
                name=name,
                scenario=base_scenario,
                config=config,
                n_reads=200,
                coverage=value
            )

        elif parameter == 'window_size':
            name = f"scale_window_{value}"
            # Create new scenario with different contig length
            modified_scenario = SimulationScenario(
                name=f"{base_scenario.name}_L{value}",
                contig_id=base_scenario.contig_id,
                contig_length=value,
                snv_positions=base_scenario.snv_positions[:int(len(base_scenario.snv_positions) * value / base_scenario.contig_length)],
                ref_alleles=base_scenario.ref_alleles,
                true_haplotypes=base_scenario.true_haplotypes,
                timepoints=base_scenario.timepoints,
            )
            # Update config
            test_config = HaplotyperConfig(
                window_size=value,
                max_reads_per_window=config.max_reads_per_window,
                random_seed=config.random_seed,
            )
            benchmark.benchmark_scenario(
                name=name,
                scenario=modified_scenario,
                config=test_config,
                n_reads=200,
                coverage=50
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Strainphase performance"
    )
    parser.add_argument(
        "--output", "-o",
        default="benchmarks",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks (fewer tests)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create benchmark suite
    benchmark = PerformanceBenchmark()

    # Configuration
    config = HaplotyperConfig(
        window_size=10000,
        max_reads_per_window=500,
        random_seed=args.seed,
        validate_results=False,  # Faster without validation
    )

    # Create base scenario
    generator = SyntheticDataGenerator(seed=args.seed)
    base_scenario = generator.create_scenario(
        name="benchmark_scenario",
        contig_length=50000,
        n_snvs=100,
        n_haplotypes=3,
        n_timepoints=1,
        include_sweep=False,
    )

    # Scalability tests
    if args.quick:
        # Quick tests
        scalability_test(
            benchmark, base_scenario, config,
            parameter='n_reads',
            values=[50, 100, 200]
        )
    else:
        # Read count scalability
        scalability_test(
            benchmark, base_scenario, config,
            parameter='n_reads',
            values=[50, 100, 200, 300, 500, 1000]
        )

        # Coverage scalability
        scalability_test(
            benchmark, base_scenario, config,
            parameter='coverage',
            values=[10, 30, 50, 100, 200]
        )

        # Window size scalability
        scalability_test(
            benchmark, base_scenario, config,
            parameter='window_size',
            values=[2000, 5000, 10000, 20000, 50000]
        )

    # Save results
    results_file = output_dir / "performance_benchmarks.json"
    benchmark.save_results(results_file)

    # Print summary
    print(benchmark.summary())

    # Create performance plots (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        # Runtime vs. read count
        read_results = [r for r in benchmark.results if 'scale_reads' in r['name']]
        if read_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            reads = [r['n_reads'] for r in read_results]
            times = [r['elapsed_seconds'] for r in read_results]
            mems = [r['peak_memory_mb'] for r in read_results]

            ax1.plot(reads, times, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Reads', fontsize=12)
            ax1.set_ylabel('Runtime (seconds)', fontsize=12)
            ax1.set_title('Runtime Scalability', fontsize=14)
            ax1.grid(alpha=0.3)

            ax2.plot(reads, mems, 'o-', linewidth=2, markersize=8, color='orange')
            ax2.set_xlabel('Number of Reads', fontsize=12)
            ax2.set_ylabel('Peak Memory (MB)', fontsize=12)
            ax2.set_title('Memory Scalability', fontsize=14)
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plot_file = output_dir / "scalability_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logging.info(f"Saved plots to {plot_file}")

    except ImportError:
        logging.info("matplotlib not available, skipping plots")

    return 0


if __name__ == "__main__":
    sys.exit(main())
