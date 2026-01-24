# Strainphase Simulation & Benchmarking Pipeline

## Overview

This document describes the simulation and benchmarking framework for validating strainphase's haplotype reconstruction accuracy and parameter sensitivity.

---

## Required Dependencies

The benchmarking and validation pipeline requires these Python packages at runtime:

- `pandas`
- `python-louvain` (imported as `community`)
- `pysam`

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

**Publication-grade stress tests to include (in addition to the default ranges):**
- **Low coverage**: 5-10x (rare-strain and SNV-sparsity stress)
- **High coverage**: 75-150x (subsampling and scalability stress)
- **Uneven coverage across timepoints** (e.g., 5x → 50x)

### 1.3 Community Complexity Presets

| Complexity | Species | Strains/Species | Total Strains |
|------------|---------|-----------------|---------------|
| Simple     | 5       | 2-3             | 10-15         |
| Medium     | 10      | 2-5             | 20-50         |
| Complex    | 20      | 1-10            | 20-200        |

**Within-species (same-MAG) stress preset (important for phasing/linking):**
- 1 species / MAG with **10-20 strains** spanning a range of divergence and abundances.

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

**Divergence regimes to include in benchmarks:**
- **Very close strains**: ~0.1-0.5% divergence (high risk of over-merging / 1-SNV edge cases)
- **Moderately diverged strains**: ~0.5-2% divergence (typical intra-species variation)
- **Highly diverged strains**: >2% divergence (graph separation easier; checks precision)

**SNV density regimes to include:**
- **Sparse**: ~1-3 SNVs per 10kb (hard for linking and EM stability)
- **Dense**: ~25-50 SNVs per 10kb (hard for clustering/merging; more evidence but more conflicts)

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

**Benchmark emphasis (publication):**
- Explicitly quantify performance for **rare strains** (0.1-1%) and **very-low** strains (e.g., 0.01-0.1% if feasible).
- Include scenarios where rare strains are **present but only detectable via longitudinal rescue** (bloom elsewhere).

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

**Additional ground truth (needed to validate track/linking + lineage behavior):**
| File | Contents |
|------|----------|
| `truth_tracks.tsv` | True track spans per strain (per contig) with expected window-chain membership |
| `truth_lineages.tsv` | Mapping of strain IDs ↔ expected cross-timepoint lineage IDs (per MAG/contig) |
| `truth_vcf_perturbations.json` | If VCF realism is simulated: which sites/fields were dropped/added/altered (FP/FN, missing AF/DP) |

**VCF STRAINS encoding (required):** use `|` to separate allele groups within the `STRAINS` INFO field.
Example: `STRAINS=A:strain_1,strain_2|G:strain_3`. Avoid `;` within the `STRAINS` value.

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

### 3.4 Track / Window-Linking Validation (Strainphase-specific)

Because Strainphase outputs **linked tracks** (not just per-window haplotypes), include metrics that capture linking quality:

| Metric | Definition |
|--------|------------|
| **Track fragmentation** | Mean/median number of inferred tracks per true strain per contig (lower is better) |
| **False link rate** | Fraction of links that join haplotypes from different true strains |
| **Missed link rate** | Fraction of true adjacent-window links not recovered (splitting) |
| **Track consensus error** | Mismatch fraction between inferred track consensus and truth at shared SNV sites |

**Linking rule (required):** For adjacent windows, evaluate all candidate haplotype pairs first, then select only the smallest-distance match. If a haplotype has more than one perfect match (or any distance tie for the minimum), do not link it to any option.

### 3.5 Longitudinal Lineage Validation (Strainphase-specific)

| Metric | Definition |
|--------|------------|
| **Lineage precision/recall** | Cluster correctness for inferred `lineage_id` vs truth mapping across timepoints |
| **Rescue gain (Δrecall)** | Recall improvement for low-abundance strains with rescue enabled vs disabled |
| **Abundance trajectory error** | Error between inferred and true abundance trajectories per lineage/strain |

---

## 4. Benchmarking Framework

Supports benchmarking on **simulated data** (with ground truth) and **real data** (metrics without truth comparison).

### 4.1 Parameters to Sweep

```python
PARAMETER_GRID = {
    # Windowing / subsampling
    # Minimum window_size is 10000 to ensure sufficient shared SNVs for reliable linking.
    # With ~1-2 SNVs/kb, smaller windows have too few SNVs in the 50% overlap region.
    'window_size': [10000, 20000, 50000, 100000],
    'max_reads_per_window': [100, 300, 600],

    # Clustering parameters
    'max_mismatch_frac': [0.005, 0.01, 0.02, 0.04],
    'min_shared_snvs_for_edge': [2, 3, 4, 5],

    # Quality filters
    'min_mapq': [10, 20, 30],
    'min_base_quality': [20, 30],

    # Junk model sensitivity (publication expansion)
    'junk_divergence_rate': [0.05, 0.10, 0.20],

    # Merging thresholds
    'merge_distance_threshold': [0.005, 0.01, 0.02],

    # Linking thresholds (publication expansion)
    'max_link_distance': [0.01, 0.02, 0.04],
    'min_shared_snvs_for_link': [2, 3, 5],

    # Abundance thresholds
    'min_weight_for_anchor': [0.05, 0.10, 0.15, 0.20],
    'rescued_min_weight': [0.01, 0.02, 0.05],

    # Rescue/lineage thresholds (publication expansion)
    'rescue_match_distance': [0.005, 0.01, 0.02],
    'lineage_merge_distance': [0.01, 0.02, 0.04],
}
```

**Important for publication:** even when you don’t sweep a parameter, always record the full effective configuration in results (all `HaplotyperConfig` fields), plus software versions and random seed.

### 4.2 Benchmark Matrix

| Community | Strain Similarity | Abundance Skew | Challenge Level |
|-----------|-------------------|----------------|-----------------|
| Simple-easy | High (99%+) | Even | Baseline |
| Simple-hard | Medium (97%) | Skewed | Divergent strains |
| Complex-easy | High (99%+) | Even | Many species |
| Complex-hard | Mixed | Skewed + rare | Full challenge |

**Add benchmark axes (so “complex-hard” is reproducible and reviewable):**
- **Strain count per MAG/contig** (e.g., 2, 4, 8, 12+)
- **Divergence** (very close vs moderate vs high; see §1.4)
- **SNV density** (sparse vs dense; see §1.4)
- **Coverage regime** (low vs typical vs high; even vs uneven over time)
- **VCF realism** (perfect truth VCF vs perturbed VCF: FP/FN, missing AF/DP)

### 4.3 Output Metrics (JSON)

```json
{
  "params": {"max_mismatch_frac": 0.01, "min_mapq": 20, "...": "..."},
  "config_full": {"... all HaplotyperConfig fields ..."},
  "community": "complex-hard",
  "seed": 42,
  "replicate": 1,
  "environment": {"python": "3.11.x", "platform": "linux|macos", "cpu": "...", "threads": 1},
  "metrics": {
    "haplotype_precision": 0.92,
    "haplotype_recall": 0.85,
    "haplotype_f1": 0.88,
    "abundance_pearson_r": 0.95,
    "abundance_mae": 0.03,
    "snv_precision": 0.94,
    "snv_recall": 0.89,
    "track_fragmentation": 1.3,
    "false_link_rate": 0.01,
    "missed_link_rate": 0.08,
    "lineage_precision": 0.90,
    "lineage_recall": 0.86,
    "rescue_delta_recall_rare": 0.12,
    "runtime_seconds": 45.2,
    "memory_peak_mb": 512
  }
}
```

**Ablation reporting (publication):** include a `mode` or `ablation` label in each record (e.g., `full`, `no_linking`, `no_rescue`, `no_junk`, `no_1snp_guard`) so plots can show the marginal value of each component.

---

## 5. Output Figures

### 5.1 Validation Figures

| Figure | Description |
|--------|-------------|
| `haplotype_accuracy.png` | Bar plot of precision/recall/F1 by community complexity |
| `abundance_correlation.png` | Scatter plot: true vs estimated abundance with Pearson r |
| `detection_sensitivity.png` | Recall vs true abundance (shows detection threshold) |
| `confusion_matrix.png` | Haplotype assignment confusion matrix |
| `track_fragmentation.png` | Tracks-per-truth distribution (fragmentation) across scenarios |
| `linking_errors.png` | False-link and missed-link rates across scenarios |
| `lineage_accuracy.png` | Lineage precision/recall across scenarios and timepoints |

### 5.2 Benchmarking Figures

| Figure | Description |
|--------|-------------|
| `parameter_heatmap.png` | Heatmap of F1 score across parameter grid |
| `parameter_sensitivity.png` | Line plots showing metric vs each parameter |
| `complexity_comparison.png` | Grouped bar: metrics across community complexities |
| `runtime_scaling.png` | Runtime vs dataset size/complexity |
| `optimal_params.png` | Highlight best parameter combinations |
| `ablation_summary.png` | Delta-metrics for key ablations (e.g., rescue, linking, 1-SNV guard) |
| `seed_sensitivity.png` | Metric variability across replicate seeds (subsampling sensitivity) |
| `vcf_robustness.png` | Performance under perturbed VCF conditions (FP/FN, missing AF/DP) |

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
- [ ] Add VCF realism toggles (optional but recommended): inject FP/FN sites and missing AF/DP fields + write `truth_vcf_perturbations.json`

### Phase 2: Validation Pipeline
- [ ] `validate_haplotypes.py`: Compare detected vs true haplotypes
- [ ] Implement matching algorithm (SNV overlap + abundance)
- [ ] Calculate all metrics (precision, recall, F1, abundance accuracy)
- [ ] Generate validation figures
- [ ] Add track/linking validation (`validate_tracks.py` or extend validator): fragmentation, false-link, missed-link, track consensus error
- [ ] Add lineage validation for longitudinal simulations: lineage clustering precision/recall and rescue Δrecall for rare strains

### Phase 3: Benchmarking
- [ ] Update `parameter_sweep.py` to accept real BAM/VCF input
- [ ] `benchmark_communities.py`: Run across complexity levels
- [ ] `generate_report.py`: Create figures and HTML report
- [ ] Add ablation runner (toggle rescue/linking/junk/1-SNV guard) and ensure output JSON includes `ablation` label
- [ ] Add scaling/profiling harness to collect runtime + peak memory consistently across scenarios

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
    --error-rate 0.01 \
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

# (Optional) Resume a full benchmark run if simulation/alignments already exist
python benchmarks/run_full_benchmark.py \
    --genomes /path/to/your/strain_genomes/ \
    --output results/full_benchmark/ \
    --resume
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
