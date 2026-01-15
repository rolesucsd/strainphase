# Quick Start - Testing and Validation

All testing and validation tools are now organized in this folder.

## Generate Synthetic Community (40 species, 120 strains)

```bash
# Navigate to synthetic community tools
cd synthetic_community_tools

# Generate synthetic data
python generate_synthetic_community.py -o ../../synthetic_output

# Validate the generated data
python test_synthetic_community.py ../../synthetic_output

# View summary
cat ../../synthetic_output/community_summary.txt
```

## Run Validation Tests

```bash
# Navigate to validation scripts
cd validation_scripts

# Quick test
python test_synthetic_quick.py

# Full validation
python validate_synthetic.py

# Generate validation figures
python test_and_figure.py
```

## View Results

```bash
# Check validation results
ls results/validation/

# View validation figure
open results/validation/validation_figure.png

# View metrics
cat results/validation/validation_metrics.json
```

## Documentation

- **Overview:** `README.md`
- **Synthetic community tools:** `synthetic_community_tools/README.md`
- **Detailed setup:** `synthetic_community_tools/SETUP_COMPLETE.md`
- **Organization info:** `ORGANIZATION_SUMMARY.md`

## Directory Structure

```
testing_and_validation/
├── synthetic_community_tools/    # Generate synthetic data
├── validation_scripts/           # Validate pipeline
├── results/                      # Test results
└── examples/                     # Example scripts
```

## Common Commands

```bash
# From this directory (testing_and_validation/)

# Generate synthetic community
cd synthetic_community_tools && \
python generate_synthetic_community.py -o ../../synthetic_output && \
cd ..

# Validate
cd synthetic_community_tools && \
python test_synthetic_community.py ../../synthetic_output && \
cd ..

# Run tests
cd validation_scripts && \
python test_synthetic_quick.py && \
cd ..
```

## Next Steps

1. **Read the overview:** `cat README.md`
2. **Generate test data:** Follow synthetic community generation above
3. **Run validation:** Use validation scripts
4. **Check results:** Review validation outputs

For detailed documentation, see individual README files in each subdirectory.
