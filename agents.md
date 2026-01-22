# Strainphase Simulation & Benchmarking Pipeline

## Overview

This document describes the simulation and benchmarking framework for validating strainphase's haplotype reconstruction accuracy and parameter sensitivity.

---

## 1. Simulation Design

### 1.1 Input Requirements

**User provides**: A folder containing individual bacterial genome FASTA files (one per strain).

```
user_genomes/
├── species_A_strain_1.fasta
├── species_A_strain_2.fasta
├── species_B_strain_1.fasta
└── ...
```

**Tool generates**: Synthetic HiFi reads with known ground truth.

### 1.2 Read Simulation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Read length | 15kb | 10-20kb | HiFi length distribution |
| Error rate | 0.1% | 0.1-0.5% | HiFi accuracy profile |
| Coverage | 30x | 10-50x | Per-sample depth |
| Error type | Substitutions | - | Substitutions > indels (HiFi characteristic) |

### 1.3 Community Complexity Presets

| Complexity | Species | Strains/Species | Total Strains |
|------------|---------|-----------------|---------------|
| Simple     | 5       | 2-3             | 10-15         |
| Medium     | 10      | 2-5             | 20-50         |
| Complex    | 20      | 1-10            | 20-200        |

### 1.4 SNV Introduction

The tool introduces SNVs to create strain diversity from user-provided genomes:

**Random sequencing error** (mimics real data):
- User-specified rate (default: 0.01%)
- Uniformly distributed across reads

**Biological SNVs** (strain-defining variants):
- Density: 1-50 SNVs per 10kb (user-configurable)
- Two patterns randomly assigned per strain:
  - **Sweeping strains**: Many SNVs with correlated frequency changes over time
  - **Fixed strains**: Few SNVs with stable frequencies

```
Example strain relationships:
Species A
├── Strain A1 (reference)
├── Strain A2 ─── ~2 SNVs/10kb  (fixed, stable)
├── Strain A3 ─── ~5 SNVs/10kb  (sweeping, dynamic)
└── Strain A4 ─── ~30 SNVs/10kb (fixed, stable)
```

### 1.5 Abundance Profiles

**Longitudinal dynamics** (4-6 timepoints):
- **Stable strains**: Abundance varies <2x across timepoints
- **Dynamic strains**: Abundance varies 5-10x
- **Selective sweep**: One strain increases >50%, another decreases >50%
- **Strain replacement**: One strain disappears, new strain emerges

**Abundance distribution**:
| Category | Relative Abundance | Detection Difficulty |
|----------|-------------------|---------------------|
| Dominant | 10-40% | Easy |
| Moderate | 1-10% | Medium |
| Rare | 0.1-1% | Hard |

---

## 2. Ground Truth Files

For each simulation, automatically generate:

| File | Contents |
|------|----------|
| `truth_strains.tsv` | Strain ID, species, genome file, SNV count |
| `truth_snvs.vcf` | All true SNV positions with strain assignments |
| `truth_abundances.tsv` | Strain abundances per timepoint |
| `truth_haplotypes.tsv` | Expected haplotype blocks with SNV alleles |
| `read_origins.tsv` | Which strain each simulated read came from |

---

## 3. Validation Metrics

### 3.1 Haplotype Detection

| Metric | Definition | Target |
|--------|------------|--------|
| **Precision** | Detected haplotypes that match a true haplotype | >90% |
| **Recall** | True haplotypes that were detected | >85% |
| **F1 Score** | Harmonic mean of precision and recall | >87% |

**Matching criteria**: A detected haplotype matches truth if:
- ≥90% of SNV alleles match
- Abundance estimate within 2x of true value

### 3.2 Abundance Accuracy

| Metric | Definition |
|--------|------------|
| **Pearson r** | Correlation between estimated and true abundances |
| **MAE** | Mean absolute error of abundance estimates |
| **Detection threshold** | Minimum true abundance reliably detected (>50% recall) |

### 3.3 SNV Accuracy

| Metric | Definition |
|--------|------------|
| **SNV precision** | Fraction of called SNVs that are true positives |
| **SNV recall** | Fraction of true SNVs that were called |
| **Phasing accuracy** | Fraction of SNVs correctly assigned to haplotypes |

---

## 4. Benchmarking Framework

Supports benchmarking on **simulated data** (with ground truth) and **real data** (metrics without truth comparison).

### 4.1 Parameters to Sweep

```python
PARAMETER_GRID = {
    # Clustering parameters
    'max_mismatch_frac': [0.005, 0.01, 0.02, 0.04],
    'min_shared_snvs_for_edge': [2, 3, 4, 5],

    # Quality filters
    'min_mapq': [10, 20, 30],
    'min_base_quality': [20, 30],

    # Merging thresholds
    'merge_distance_threshold': [0.005, 0.01, 0.02],

    # Abundance thresholds
    'min_weight_for_anchor': [0.05, 0.10, 0.15, 0.20],
    'rescued_min_weight': [0.01, 0.02, 0.05],
}
```

### 4.2 Benchmark Matrix

| Community | Strain Similarity | Abundance Skew | Challenge Level |
|-----------|-------------------|----------------|-----------------|
| Simple-easy | High (99%+) | Even | Baseline |
| Simple-hard | Medium (97%) | Skewed | Divergent strains |
| Complex-easy | High (99%+) | Even | Many species |
| Complex-hard | Mixed | Skewed + rare | Full challenge |

### 4.3 Output Metrics (JSON)

```json
{
  "params": {"max_mismatch_frac": 0.01, "min_mapq": 20, ...},
  "community": "complex-hard",
  "metrics": {
    "haplotype_precision": 0.92,
    "haplotype_recall": 0.85,
    "haplotype_f1": 0.88,
    "abundance_pearson_r": 0.95,
    "abundance_mae": 0.03,
    "snv_precision": 0.94,
    "snv_recall": 0.89,
    "runtime_seconds": 45.2,
    "memory_peak_mb": 512
  }
}
```

---

## 5. Output Figures

### 5.1 Validation Figures

| Figure | Description |
|--------|-------------|
| `haplotype_accuracy.png` | Bar plot of precision/recall/F1 by community complexity |
| `abundance_correlation.png` | Scatter plot: true vs estimated abundance with Pearson r |
| `detection_sensitivity.png` | Recall vs true abundance (shows detection threshold) |
| `confusion_matrix.png` | Haplotype assignment confusion matrix |

### 5.2 Benchmarking Figures

| Figure | Description |
|--------|-------------|
| `parameter_heatmap.png` | Heatmap of F1 score across parameter grid |
| `parameter_sensitivity.png` | Line plots showing metric vs each parameter |
| `complexity_comparison.png` | Grouped bar: metrics across community complexities |
| `runtime_scaling.png` | Runtime vs dataset size/complexity |
| `optimal_params.png` | Highlight best parameter combinations |

### 5.3 Summary Report

Generate `benchmark_report.html` containing:
- All figures embedded
- Best parameter recommendations per community type
- Failure mode analysis (when/why performance drops)
- Comparison tables

---

## 6. Implementation Plan

### Phase 1: Read Simulation
- [ ] `simulate_reads.py`: Generate HiFi reads from user-provided genomes
- [ ] Implement SNV spiking (random error + biological variants)
- [ ] Implement abundance mixing across timepoints
- [ ] Generate all ground truth files

### Phase 2: Validation Pipeline
- [ ] `validate_haplotypes.py`: Compare detected vs true haplotypes
- [ ] Implement matching algorithm (SNV overlap + abundance)
- [ ] Calculate all metrics (precision, recall, F1, abundance accuracy)
- [ ] Generate validation figures

### Phase 3: Benchmarking
- [ ] Update `parameter_sweep.py` to accept real BAM/VCF input
- [ ] `benchmark_communities.py`: Run across complexity levels
- [ ] `generate_report.py`: Create figures and HTML report

---

## 7. File Structure

```
strainphase/
├── validation/
│   ├── simulate_reads.py           # HiFi read simulation from user genomes
│   ├── validate_haplotypes.py      # Compare detected vs ground truth
│   └── run_validation.py           # Full validation pipeline
│
├── benchmarks/
│   ├── parameter_sweep.py          # Test parameter grid
│   ├── benchmark_communities.py    # Test across complexity levels
│   └── generate_report.py          # Create figures and HTML report
│
└── data/                           # Generated data (gitignored)
    ├── simulated/                  # Simulated reads + ground truth
    └── results/                    # Benchmark outputs + figures
```

---

## 8. Example Workflow

```bash
# 1. Simulate HiFi reads from your genomes
python validation/simulate_reads.py \
    --genomes /path/to/your/strain_genomes/ \
    --complexity medium \
    --snv-density 10 \
    --error-rate 0.001 \
    --timepoints 4 \
    --coverage 30 \
    --output data/simulated/

# 2. Run strainphase on simulated data
strainphase longitudinal \
    --samples T1,T2,T3,T4 \
    --bams data/simulated/{sample}.bam \
    --vcfs data/simulated/{sample}.vcf.gz \
    --reference data/simulated/combined_reference.fasta \
    --output-dir results/

# 3. Validate against ground truth
python validation/validate_haplotypes.py \
    --detected results/lineages.tsv \
    --truth data/simulated/truth_haplotypes.tsv \
    --output results/validation/

# 4. Run parameter sweep
python benchmarks/parameter_sweep.py \
    --bam data/simulated/T1.bam \
    --vcf data/simulated/T1.vcf.gz \
    --truth data/simulated/truth_haplotypes.tsv \
    --output benchmarks/sweep_results/

# 5. Generate benchmark report with figures
python benchmarks/generate_report.py \
    --results benchmarks/sweep_results/ \
    --output benchmarks/report/
```

---

## 9. CLI Interface Summary

### simulate_reads.py
```
--genomes PATH        Folder with strain FASTA files (required)
--complexity LEVEL    simple|medium|complex (default: medium)
--snv-density INT     SNVs per 10kb to introduce (default: 10)
--error-rate FLOAT    Random sequencing error rate (default: 0.001)
--timepoints INT      Number of timepoints (default: 4)
--coverage INT        Read coverage per sample (default: 30)
--sweep-fraction FLOAT Fraction of strains with sweeping dynamics (default: 0.3)
--output PATH         Output directory (required)
```

### validate_haplotypes.py
```
--detected PATH       Strainphase output (lineages.tsv)
--truth PATH          Ground truth haplotypes file
--output PATH         Output directory for metrics and figures
```

### parameter_sweep.py
```
--bam PATH            Input BAM file
--vcf PATH            Input VCF file
--truth PATH          Ground truth file (optional, for simulated data)
--params PATH         Custom parameter grid JSON (optional)
--output PATH         Output directory
```

### generate_report.py
```
--results PATH        Directory with benchmark JSON results
--output PATH         Output directory for figures and HTML report
```
