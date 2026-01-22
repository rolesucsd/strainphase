# Validation

Tools for validating strainphase against simulated data.

## Overview

This folder contains scripts for:
1. **Downloading** real bacterial genomes from NCBI RefSeq
2. **Validating** strainphase results against ground truth
3. **Generating** publication-quality figures

## Files

- `download_real_genomes.py` - Downloads bacterial genomes from NCBI for testing
- `validate_synthetic.py` - Validates strainphase against synthetic scenarios
- `test_and_figure.py` - Quick validation with figure generation
- `slurm_array.sh` - Example SLURM job script for HPC clusters

## Usage

### 1. Download Reference Genomes

```bash
python validation/download_real_genomes.py --output data/genomes/
```

### 2. Run Validation

```bash
python validation/validate_synthetic.py --output results/
python validation/validate_synthetic.py --quick  # Fast validation
```

### 3. Generate Figures

```bash
python validation/test_and_figure.py
```

## Note

The synthetic data generator in `strainphase.simulation` creates in-memory test
scenarios. For file-based validation with realistic data, use `download_real_genomes.py`
to obtain real bacterial genomes from NCBI.
