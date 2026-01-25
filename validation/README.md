# Validation

Tools for validating strainphase against simulated data.

## Overview

This folder contains the validation modules used by the benchmarking pipeline.
All validation is orchestrated through `benchmarks/run_full_benchmark.py`.

## Core Validation Modules

- `simulate_reads.py` - Generates file-based synthetic HiFi reads from real genomes
- `validate_haplotypes.py` - Main validation entry point (precision, recall, F1, etc.)
- `validate_tracks.py` - Track/linking validation metrics (library module)
- `validate_lineages.py` - Lineage validation metrics (library module)

## Usage

### Full Benchmarking Pipeline (Recommended)

```bash
python benchmarks/run_full_benchmark.py \
    --genomes data/genomes/ \
    --output results/benchmark/
```

This automatically:
1. Simulates reads from genomes
2. Runs parameter sweeps
3. Validates results
4. Generates reports

### Standalone Validation

If you already have results and want to validate them:

```bash
python validation/validate_haplotypes.py \
    --detected results/lineages.tsv \
    --truth data/simulated/ \
    --output results/validation/
```

## Simulation Systems

This directory contains **file-based simulation** (`simulate_reads.py`) which generates
real BAM/VCF/FASTQ files from user-provided genomes. This is used by the main
benchmarking pipeline for validation.

**Note:** There is also an **in-memory simulation system** (`strainphase.simulation.SyntheticDataGenerator`)
which generates synthetic data without file I/O. This is used by `benchmark_performance.py`
for quick performance profiling. The two systems serve different purposes:
- **File-based** (`validation/simulate_reads.py`): Full pipeline testing with real file formats
- **In-memory** (`strainphase/simulation/`): Fast performance profiling without I/O overhead
