# Testing and Validation Tools

All testing, validation, and development tools have been organized in the `testing_and_validation/` folder.

## Location

```
strainphase/
└── testing_and_validation/
    ├── synthetic_community_tools/    # Generate synthetic communities
    │   ├── generate_synthetic_community.py
    │   ├── test_synthetic_community.py
    │   ├── download_real_genomes.py
    │   └── [documentation files]
    │
    ├── validation_scripts/           # Validation and benchmarking
    │   ├── validate_synthetic.py
    │   ├── benchmark_performance.py
    │   ├── test_synthetic_quick.py
    │   └── test_and_figure.py
    │
    ├── results/                      # Test results
    │   └── validation/
    │
    └── examples/                     # Example scripts
        └── slurm_array.sh
```

## Quick Start

### Generate Synthetic Community (40 species, 120 strains)

```bash
cd testing_and_validation/synthetic_community_tools

# Generate synthetic data
python generate_synthetic_community.py -o ../../synthetic_output

# Validate results
python test_synthetic_community.py ../../synthetic_output
```

### Run Validation Tests

```bash
cd testing_and_validation/validation_scripts

# Quick validation
python test_synthetic_quick.py

# Full validation
python validate_synthetic.py

# Generate figures
python test_and_figure.py
```

## Documentation

See the `testing_and_validation/` folder for complete documentation:
- `testing_and_validation/README.md` - Main overview
- `synthetic_community_tools/README.md` - Synthetic data generation
- `validation_scripts/README.md` - Validation tools
