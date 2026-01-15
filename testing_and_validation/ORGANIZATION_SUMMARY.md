# Testing and Validation Organization Summary

All testing, validation, and development tools have been consolidated into the `testing_and_validation/` directory.

## What Was Organized

The following directories and their contents were moved here:

### 1. Synthetic Community Tools
**Previously:** `synthetic_community_tools/` (root level)
**Now:** `testing_and_validation/synthetic_community_tools/`

**Contents:**
- `generate_synthetic_community.py` - Generate synthetic metagenomic communities
- `test_synthetic_community.py` - Validate generated communities
- `download_real_genomes.py` - Download real bacterial genomes (optional)
- Complete documentation (README, SETUP_COMPLETE, QUICKSTART, etc.)

### 2. Validation Scripts
**Previously:** `scripts/` (root level)
**Now:** `testing_and_validation/validation_scripts/`

**Contents:**
- `validate_synthetic.py` - Validate pipeline on synthetic data
- `benchmark_performance.py` - Performance benchmarking
- `test_synthetic_quick.py` - Quick validation tests
- `test_and_figure.py` - Generate validation figures
- `add_high_complexity_scenario.py` - Add complex test scenarios

### 3. Test Results
**Previously:** `results/` (root level)
**Now:** `testing_and_validation/results/`

**Contents:**
- `validation/` directory with validation figures and metrics
  - `validation_figure.png`
  - `validation_metrics.json`

### 4. Examples
**Previously:** `examples/` (root level)
**Now:** `testing_and_validation/examples/`

**Contents:**
- `slurm_array.sh` - SLURM job submission example for HPC clusters

## New Directory Structure

```
strainphase/
├── src/                          # Core source code (unchanged)
├── docs/                         # Documentation (unchanged)
├── manuscript/                   # Publication materials (unchanged)
│
└── testing_and_validation/       # ← All testing tools here
    ├── README.md                 # Main overview
    ├── ORGANIZATION_SUMMARY.md   # This file
    │
    ├── synthetic_community_tools/
    │   ├── generate_synthetic_community.py
    │   ├── test_synthetic_community.py
    │   ├── download_real_genomes.py
    │   ├── README.md
    │   ├── SETUP_COMPLETE.md
    │   ├── QUICKSTART.md
    │   └── README_synthetic_community.md
    │
    ├── validation_scripts/
    │   ├── validate_synthetic.py
    │   ├── benchmark_performance.py
    │   ├── test_synthetic_quick.py
    │   ├── test_and_figure.py
    │   ├── add_high_complexity_scenario.py
    │   └── README.md
    │
    ├── results/
    │   └── validation/
    │       ├── validation_figure.png
    │       └── validation_metrics.json
    │
    └── examples/
        └── slurm_array.sh
```

## Benefits of This Organization

### 1. **Clarity**
All testing and validation tools are in one place, making it clear what's for development/testing vs. production use.

### 2. **Separation of Concerns**
- Production code: `src/`
- Documentation: `docs/`
- Testing/validation: `testing_and_validation/`
- Publication: `manuscript/`

### 3. **Easy Navigation**
New contributors and users can quickly find testing tools without searching through the root directory.

### 4. **Scalability**
Easy to add new testing tools, validation scripts, or test datasets without cluttering the root.

## Updated Workflows

### Generate Synthetic Community

**Old:**
```bash
cd synthetic_community_tools
python generate_synthetic_community.py -o ../output
```

**New:**
```bash
cd testing_and_validation/synthetic_community_tools
python generate_synthetic_community.py -o ../../output
```

### Run Validation

**Old:**
```bash
cd scripts
python validate_synthetic.py
```

**New:**
```bash
cd testing_and_validation/validation_scripts
python validate_synthetic.py
```

### Access Results

**Old:**
```bash
ls results/validation/
```

**New:**
```bash
ls testing_and_validation/results/validation/
```

### Use Examples

**Old:**
```bash
cd examples
sbatch slurm_array.sh
```

**New:**
```bash
cd testing_and_validation/examples
sbatch slurm_array.sh
```

## Documentation Updates

All documentation has been updated to reflect the new paths:

- ✓ `README_SYNTHETIC_TOOLS.md` (root) - Points to new location
- ✓ `testing_and_validation/README.md` - Main overview of all tools
- ✓ `synthetic_community_tools/SETUP_COMPLETE.md` - Updated paths
- ✓ `synthetic_community_tools/README.md` - Updated examples
- ✓ All scripts work with new relative paths

## Quick Reference

### From Root Directory

```bash
# Generate synthetic data
cd testing_and_validation/synthetic_community_tools
python generate_synthetic_community.py -o ../../synthetic_output

# Validate synthetic data
python test_synthetic_community.py ../../synthetic_output

# Run validation scripts
cd ../validation_scripts
python validate_synthetic.py

# View results
ls ../results/validation/
```

### Documentation Files

- **Overview:** `testing_and_validation/README.md`
- **Synthetic community:** `testing_and_validation/synthetic_community_tools/README.md`
- **Quick start:** `testing_and_validation/synthetic_community_tools/QUICKSTART.md`
- **Setup guide:** `testing_and_validation/synthetic_community_tools/SETUP_COMPLETE.md`
- **This summary:** `testing_and_validation/ORGANIZATION_SUMMARY.md`

## Notes

- All scripts have been tested to work from their new locations
- Relative paths have been updated throughout documentation
- No functionality has been changed, only organization
- Old paths in git history remain for reference

## Getting Started

See the main README:
```bash
cat testing_and_validation/README.md
```

Or jump straight to generating synthetic data:
```bash
cd testing_and_validation/synthetic_community_tools
cat SETUP_COMPLETE.md
```
