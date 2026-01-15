# Testing and Validation Tools

This directory contains all testing, validation, and development tools for StrainPhase.

## Directory Structure

```
testing_and_validation/
├── synthetic_community_tools/    # Generate synthetic metagenomic communities
│   ├── generate_synthetic_community.py
│   ├── test_synthetic_community.py
│   ├── download_real_genomes.py
│   └── [documentation files]
│
├── validation_scripts/           # Validation and benchmarking scripts
│   ├── validate_synthetic.py    # Validate pipeline on synthetic data
│   ├── benchmark_performance.py # Performance benchmarking
│   ├── test_synthetic_quick.py  # Quick validation tests
│   ├── test_and_figure.py       # Generate validation figures
│   └── add_high_complexity_scenario.py
│
├── results/                      # Test results and outputs
│   └── validation/
│       ├── validation_figure.png
│       └── validation_metrics.json
│
└── examples/                     # Example usage scripts
    └── slurm_array.sh           # SLURM job submission example
```

## Quick Start

### 1. Generate Synthetic Community

Create a synthetic metagenomic dataset with 40 species and 120 strains:

```bash
cd synthetic_community_tools
python generate_synthetic_community.py -o ../../synthetic_output
python test_synthetic_community.py ../../synthetic_output
```

See `synthetic_community_tools/README.md` for detailed documentation.

### 2. Validate Pipeline

Run validation tests on synthetic data:

```bash
cd validation_scripts
python validate_synthetic.py
```

### 3. Benchmark Performance

Test pipeline performance:

```bash
cd validation_scripts
python benchmark_performance.py
```

### 4. Generate Validation Figures

Create publication-quality validation figures:

```bash
cd validation_scripts
python test_and_figure.py
```

## Tools Overview

### Synthetic Community Tools

**Purpose:** Generate synthetic metagenomic communities with known ground truth for testing and validation.

**Key features:**
- Generate 40 species with 120 total strains
- Realistic abundance distributions
- Temporal dynamics across timepoints
- Ground truth metadata for validation

**Main scripts:**
- `generate_synthetic_community.py` - Generate synthetic data
- `test_synthetic_community.py` - Validate generated data
- `download_real_genomes.py` - Download real bacterial genomes (optional)

**Documentation:**
- `synthetic_community_tools/README.md` - Main documentation
- `synthetic_community_tools/SETUP_COMPLETE.md` - Setup guide
- `synthetic_community_tools/QUICKSTART.md` - Quick reference

### Validation Scripts

**Purpose:** Validate StrainPhase pipeline accuracy and performance.

**Scripts:**
- `validate_synthetic.py` - Full validation on synthetic data
- `benchmark_performance.py` - Performance benchmarking
- `test_synthetic_quick.py` - Quick validation tests
- `test_and_figure.py` - Generate validation figures

### Results

**Purpose:** Store validation results, figures, and metrics.

**Contents:**
- `validation/validation_figure.png` - Validation plots
- `validation/validation_metrics.json` - Quantitative metrics

### Examples

**Purpose:** Example scripts for common use cases.

**Contents:**
- `slurm_array.sh` - SLURM job submission template

## Common Workflows

### Full Validation Pipeline

```bash
# 1. Generate synthetic community
cd synthetic_community_tools
python generate_synthetic_community.py -o ../../test_data

# 2. Validate generated data
python test_synthetic_community.py ../../test_data

# 3. Run validation on synthetic data
cd ../validation_scripts
python validate_synthetic.py --input ../../test_data

# 4. Generate figures
python test_and_figure.py --results ../results/validation

# 5. Check results
ls -lh ../results/validation/
```

### Quick Test

```bash
cd validation_scripts
python test_synthetic_quick.py
```

### Benchmark Performance

```bash
cd validation_scripts
python benchmark_performance.py --n-species 40 --n-strains 120
```

### HPC Cluster Usage

```bash
cd examples
# Edit slurm_array.sh with your parameters
sbatch slurm_array.sh
```

## Documentation

### Synthetic Community Generation
- Full documentation: `synthetic_community_tools/README_synthetic_community.md`
- Quick start: `synthetic_community_tools/QUICKSTART.md`
- Setup guide: `synthetic_community_tools/SETUP_COMPLETE.md`

### Validation Scripts
- Overview: `validation_scripts/README.md`
- Individual script help: Run with `--help` flag

## File Locations

After running tools, outputs will be created at:

```
strainphase/
├── testing_and_validation/       # You are here
│   ├── synthetic_community_tools/
│   ├── validation_scripts/
│   ├── results/
│   └── examples/
│
├── synthetic_output/              # Generated synthetic data (from step 1)
├── test_data/                     # Test datasets
└── validation_results/            # Validation outputs
```

## Requirements

### Python Packages
- numpy
- pandas
- matplotlib (for figures)
- scipy (for validation)

### Optional
- ncbi-datasets-cli (for downloading real genomes)
- pbsim3 or InSilicoSeq (for read generation)

### Install
```bash
pip install -e .
```

## Testing Checklist

Before publication or major release:

- [ ] Generate synthetic community (40 species, 120 strains)
- [ ] Validate synthetic data integrity
- [ ] Run full validation pipeline
- [ ] Generate validation figures
- [ ] Check metrics are within expected ranges
- [ ] Test on HPC cluster (if applicable)
- [ ] Document any issues or limitations

## Getting Help

1. Check tool-specific documentation in subdirectories
2. Run scripts with `--help` flag
3. See main StrainPhase documentation in `docs/`
4. Check validation results in `results/`

## Contributing

When adding new tests or validation tools:

1. Add scripts to appropriate subdirectory
2. Update this README
3. Include usage examples
4. Add results to `results/` directory
5. Document in tool-specific README

## Notes

- All paths in examples assume you're running from the subdirectory
- Outputs are created in the parent directory to keep this folder clean
- Results are version controlled (small files only)
- Large datasets should be generated locally and are in `.gitignore`
