# Scripts for Publication

This directory contains scripts for generating publication-quality results.

## Available Scripts

### 1. validate_synthetic.py

Validates Strainphase accuracy using synthetic data with known ground truth.

**Usage:**
```bash
# Quick validation (3 scenarios, ~5 minutes)
python scripts/validate_synthetic.py --quick --output results/validation/

# Full validation (4 scenarios, all timepoints, ~30 minutes)
python scripts/validate_synthetic.py --output results/validation/

# Custom seed for reproducibility
python scripts/validate_synthetic.py --seed 42 --output results/validation/
```

**Output:**
- `validation_summary.json` - Overall precision, recall, F1 scores
- `{scenario}_validation.json` - Per-scenario detailed results

**What it does:**
1. Generates synthetic scenarios with known haplotypes
2. Runs Strainphase on synthetic data
3. Compares detected haplotypes to ground truth
4. Computes precision, recall, F1, abundance accuracy

**Use in manuscript:** Table 1 (Section 3.1)

---

### 2. benchmark_performance.py

Measures runtime and memory usage across different data sizes.

**Usage:**
```bash
# Quick benchmarks (fewer tests, ~5 minutes)
python scripts/benchmark_performance.py --quick --output benchmarks/

# Full benchmarks (complete scalability tests, ~20 minutes)
python scripts/benchmark_performance.py --output benchmarks/

# Custom parameters
python scripts/benchmark_performance.py \
  --output benchmarks/ \
  --seed 42 \
  --log-level DEBUG
```

**Output:**
- `performance_benchmarks.json` - Runtime and memory data
- `scalability_plots.png` - Performance curves (if matplotlib available)

**What it does:**
1. Creates synthetic data of varying sizes
2. Measures runtime and peak memory for each
3. Tests scalability with read count, coverage, window size
4. Generates performance plots

**Use in manuscript:** Table 2 (Section 3.3)

---

### 3. plot_strain_dynamics.py (You'll create this)

Creates strain abundance visualization from real data.

**Template:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/real_data/lineages.tsv', sep='\t')

# Filter to one MAG
mag_df = df[df['mag'] == 'YOUR_MAG_NAME']

# Pivot and plot
pivot = mag_df.pivot_table(
    index='sample',
    columns='lineage_id',
    values='mean_weight',
    fill_value=0
)

pivot.plot(kind='area', stacked=True, figsize=(10, 6))
plt.xlabel('Timepoint')
plt.ylabel('Relative Abundance')
plt.title('Strain Dynamics')
plt.savefig('manuscript/figure2.png', dpi=600)
```

**Use in manuscript:** Figure 2 (Section 3.2)

---

## Typical Workflow for Publication

### Step 1: Generate Synthetic Validation Results

```bash
python scripts/validate_synthetic.py --output results/validation/
```

Copy F1 scores into manuscript Table 1.

### Step 2: Generate Performance Benchmarks

```bash
python scripts/benchmark_performance.py --output benchmarks/
```

Copy runtime/memory data into manuscript Table 2.

### Step 3: Analyze Real Data

```bash
strainphase longitudinal \
  --samples T1,T2,T3,T4 \
  --bams /path/to/{sample}.bam \
  --vcfs /path/to/{sample}.vcf.gz \
  --reference /path/to/ref.fasta \
  --output-dir results/real_data/
```

Write findings in manuscript Section 3.2.

### Step 4: Create Figures

```bash
# Figure 1: Use existing docs/haplotypes.pdf (algorithm schematic)
# Figure 2: Create strain dynamics plot from real data
python scripts/plot_strain_dynamics.py
```

---

## Expected Results

### Validation Metrics (Synthetic Data)

Based on algorithm design, expected performance:

- **Simple scenarios (2 haplotypes):** F1 > 0.90
- **Complex scenarios (4 haplotypes):** F1 > 0.80
- **Consensus accuracy:** > 99% on shared SNVs
- **Abundance MAE:** < 0.10 (10% error)

### Performance Benchmarks

Expected scalability:

| Reads | Runtime | Memory |
|-------|---------|--------|
| 100   | <5s     | <50 MB |
| 200   | <10s    | <100 MB|
| 500   | <30s    | <200 MB|
| 1000  | <60s    | <400 MB|

Memory usage should be approximately linear with read count.

---

## Troubleshooting

### Import Errors

```bash
# Make sure strainphase is installed
pip install -e ".[all]"

# Or just core dependencies
pip install -e .
pip install matplotlib  # for plots
```

### Memory Errors

Reduce synthetic data size:

```python
# In validate_synthetic.py, modify:
config = HaplotyperConfig(
    max_reads_per_window=100,  # Reduce from default 200
)
```

### Slow Performance

Use `--quick` flag for faster testing:

```bash
python scripts/validate_synthetic.py --quick
python scripts/benchmark_performance.py --quick
```

---

## Customization

### Adding New Scenarios

Edit `validate_synthetic.py`:

```python
# Add new scenario
scenarios.append(
    generator.create_scenario(
        name="my_custom_scenario",
        n_haplotypes=5,
        n_timepoints=3,
        include_sweep=True,
    )
)
```

### Different Parameter Grids

Edit `benchmark_performance.py`:

```python
# Test different parameters
scalability_test(
    benchmark, base_scenario, config,
    parameter='n_reads',
    values=[50, 100, 200, 500, 1000, 2000]  # Add more values
)
```

---

## Integration with CI/CD

These scripts can run in CI for continuous validation:

```yaml
# In .github/workflows/ci.yml
- name: Run validation
  run: |
    python scripts/validate_synthetic.py --quick --output ci_validation/

- name: Check performance
  run: |
    python scripts/benchmark_performance.py --quick --output ci_benchmarks/
```

---

## Questions?

See:
- Main README: `../README.md`
- Tutorial: `../docs/tutorials/TUTORIAL.md`
- Publication guide: `../QUICK_START_PUBLICATION.md`
- GitHub issues: https://github.com/roles/strainphase/issues
