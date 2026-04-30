# Strainphase

**Hybrid graph-probabilistic haplotype reconstruction for PacBio HiFi metagenomic data**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](https://opensource.org/license/bsd-3-clause/)

## Overview

Strainphase reconstructs distinct bacterial haplotypes (strain-specific SNV patterns) from mixed metagenomic reads. It uses a hybrid approach combining:

1. **Graph-based initialization** - Louvain clustering of read overlap networks
2. **Probabilistic EM refinement** - Quality-weighted soft assignments
3. **Window linking** - Track assembly across overlapping genomic windows
4. **Longitudinal rescue** - Cross-timepoint detection of low-abundance strains

## Installation

```bash
git clone https://github.com/rolesucsd/strainphase.git
cd strainphase
pip install -e .
```

**Dependencies:** `numpy`, `scipy`, `networkx`, `pandas`, `python-louvain`, `pysam`

## Quick Start

### Command Line Interface

```bash
# Process single contig
strainphase run \
    --bam sample.sorted.bam \
    --vcf clair3/pileup.vcf.gz \
    --contig MAG_01_contig_1 \
    --length 50000 \
    --output haplotypes.tsv

# Longitudinal analysis (multiple timepoints)
strainphase longitudinal \
    --samples T1,T2,T3,T4 \
    --bams mapping/{sample}.sorted.bam \
    --vcfs variants/{sample}/pileup.vcf.gz \
    --reference combined_bins.fasta \
    --output-dir results/ \
    --mags MAG_01
```

### Python API

```python
from strainphase import HaplotyperConfig, process_contig

config = HaplotyperConfig(
    window_size=3000,
    max_mismatch_frac=0.02,
    min_weight_for_anchor=0.15,
)

results = process_contig(
    bam_path="sample.bam",
    vcf_path="variants.vcf.gz",
    contig_id="MAG_01_contig_1",
    contig_length=50000,
    config=config,
)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `strainphase run` | Process a single contig |
| `strainphase longitudinal` | Multi-sample longitudinal analysis |
| `strainphase test` | Run unit test suite |
| `strainphase sweep` | Parameter sensitivity analysis (developer tool) |
| `strainphase version` | Show version |

### `strainphase run`

```
strainphase run --bam FILE --vcf FILE --contig ID --length INT [OPTIONS]

Required:
  --bam FILE          Input BAM file (sorted, indexed)
  --vcf FILE          Input VCF file (Clair3 format)
  --contig ID         Contig ID to process
  --length INT        Contig length in bp

Options:
  --sample ID         Sample identifier
  --output FILE       Output TSV file [default: haplotypes.tsv]
  --window-size INT   Analysis window size [default: 3000]
  --max-reads INT     Max reads per window [default: 300]
  --min-mapq INT      Minimum MAPQ [default: 20]
  --max-mismatch FLT  Max mismatch fraction [default: 0.02]
  --seed INT          Random seed for reproducibility
  --log-level LEVEL   Logging level [default: INFO]
```

### `strainphase longitudinal`

```
strainphase longitudinal --samples LIST --bams TPL --vcfs TPL --reference FILE --output-dir DIR [OPTIONS]

Required:
  --samples LIST      Comma-separated sample IDs (e.g., T1,T2,T3)
  --bams TPL          BAM path template with {sample} placeholder
  --vcfs TPL          VCF path template with {sample} placeholder
  --reference FILE    Reference FASTA (with .fai index)
  --output-dir DIR    Output directory

Options:
  --mags LIST              Comma-separated MAG names [default: all]
  --contig-filter F        File listing allowed contigs
  --window-size INT        Window size [default: 3000]
  --max-reads INT          Max reads per window [default: 300]
  --min-anchor-weight FLT  Minimum weight for anchor panel [default: 0.15]
  --rescued-min-weight FLT Minimum weight after rescue [default: 0.02]
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 3000 | Analysis window size (bp) |
| `max_mismatch_frac` | 0.02 | Max Hamming distance for graph edges |
| `min_shared_snvs_for_edge` | 3 | Min shared SNVs to connect reads |
| `merge_distance_threshold` | 0.01 | Distance threshold for merging haplotypes |
| `assign_confidence_threshold` | 0.90 | γ threshold for hard read assignment |
| `min_weight_for_anchor` | 0.20 | Min abundance for anchor panel |
| `rescued_min_weight` | 0.02 | Min weight after longitudinal rescue |
| `junk_divergence_rate` | 0.10 | Junk model divergence rate |

## Output Format

### haplotypes.tsv / lineages.tsv

| Column | Description |
|--------|-------------|
| `contig` | Contig ID |
| `sample` | Sample/timepoint ID |
| `track_id` | Linked haplotype track identifier |
| `lineage_id` | Cross-sample lineage cluster |
| `span_start` | Track start position |
| `span_end` | Track end position |
| `n_snvs` | Number of SNVs in consensus |
| `mean_weight` | Mean abundance (π) |
| `consensus` | SNV profile (pos:base pairs) |

## Algorithm Overview

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT: BAM + VCF → Overlapping windows (50% step)       │
├─────────────────────────────────────────────────────────────┤
│ 2. GRAPH: Build read overlap network → Louvain clustering   │
├─────────────────────────────────────────────────────────────┤
│ 3. EM: E-step (γ responsibilities) ↔ M-step (π, consensus) │
├─────────────────────────────────────────────────────────────┤
│ 4. POST: Merge similar haplotypes, validate 1-SNP diffs    │
├─────────────────────────────────────────────────────────────┤
│ 5. LINK: Connect haplotypes across windows → tracks        │
├─────────────────────────────────────────────────────────────┤
│ 6. RESCUE: Cross-timepoint anchor matching (longitudinal)  │
└─────────────────────────────────────────────────────────────┘
```

## Benchmarking

This section documents how the parameter sweep / benchmark suite is wired together: which scripts call which, what dimensions are tested, and how to extend them.

### How the suite is organized

```
run_cluster_benchmark.sh         # top-level submitter (SLURM or local)
 └─ cluster_benchmark.sh         # per-complexity worker (SLURM array tasks 1..3)
    └─ run_full_benchmark.py     # orchestrator: simulate → BAM → VCF → sweep
       ├─ validation/simulate_reads.py     # synthetic data generator (SNVs + indels)
       └─ benchmarks/parameter_sweep.py    # the sweep + metrics
          └─ ParameterSweep.REQUIRED_GRID  # parameter dimensions
```

| File | Role |
|------|------|
| `benchmarks/run_cluster_benchmark.sh` | Top-level entry point. Submits the SLURM array (3 tasks: simple/medium/complex) or runs the same three locally. Consolidates `consolidated_summary.json` after all jobs finish. |
| `benchmarks/cluster_benchmark.sh` | Per-complexity SLURM worker. Picks a base genome from `$GENOME_SOURCE`, sets `N_STRAINS`/`SNV_COUNTS` (overridable via env vars) based on the complexity level, invokes `run_full_benchmark.py`. |
| `benchmarks/run_full_benchmark.py` | Orchestrator. Steps: simulate reads → SAM→BAM → build truth VCF → run parameter sweep → optional performance benchmark. |
| `benchmarks/parameter_sweep.py` | Defines `ParameterSweep.REQUIRED_GRID` and runs **grid** (Cartesian product, optionally clipped by `--max-configs`) or **sequential** (coordinate descent) sweeps. Computes precision / recall / F1 for SNVs, haplotypes, and lineages, plus abundance MAE / Pearson when ground truth is available. |
| `benchmarks/best_params.json` | Trimmed grid focused on the most impactful parameters; used as `--params` in grid mode. Includes the `include_indels` axis. |
| `validation/simulate_reads.py` | Synthetic data generator. Generates strains, abundances, and reads. Now supports **indel injection** via `--indel-density` (deletions and insertions emit proper `D`/`I` CIGAR ops in the simulated BAM and matching records in the truth VCF). |

### How to run

**Default (SLURM, sequential coordinate-descent on 3 complexity levels):**
```bash
./benchmarks/run_cluster_benchmark.sh
```

**Locally, sequentially across complexity levels:**
```bash
./benchmarks/run_cluster_benchmark.sh --local
```

**Locally, all three complexity levels in parallel:**
```bash
./benchmarks/run_cluster_benchmark.sh --local --parallel
```

**Just one complexity level locally:**
```bash
bash benchmarks/cluster_benchmark.sh 2   # 1=simple/2-strain, 2=medium/4-strain, 3=complex/8-strain
```

**Consolidate results after jobs finish:**
```bash
./benchmarks/run_cluster_benchmark.sh --consolidate-only
```

**Test the indel pipeline (synthetic indels at 5/10kb, max 10 bp):**
```bash
INDEL_DENSITY=5 INDEL_MAX_SIZE=10 \
    bash benchmarks/cluster_benchmark.sh 2
```

**Sweep coverage (one job per coverage level):**
```bash
for cov in 10 30 50 100; do
    OUTPUT_BASE=results/cluster_benchmark_cov${cov} \
    COVERAGE=$cov \
    bash benchmarks/cluster_benchmark.sh 2
done
```

**Sweep number of timepoints:**
```bash
for tp in 2 4 8; do
    OUTPUT_BASE=results/cluster_benchmark_tp${tp} \
    TIMEPOINTS=$tp \
    bash benchmarks/cluster_benchmark.sh 2
done
```

**Decouple strain count from complexity-level defaults:**
```bash
# Run "complex" (8 strain) at simple-style SNV diversity (10000 only):
SNV_COUNTS=10000 N_STRAINS=8 \
    bash benchmarks/cluster_benchmark.sh 3
```

**Full env-var reference for `cluster_benchmark.sh`:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODE` | `sequential` | `sequential` (coordinate descent) or `grid` |
| `MAX_CONFIGS` | `4` | Cap on grid-mode configs |
| `PARAMS_FILE` | `best_params.json` | Custom grid file (grid mode only) |
| `PASSES` | `3` | Optimization passes (sequential mode) |
| `COVERAGE` | `30` | Read coverage per timepoint |
| `TIMEPOINTS` | `4` | Number of timepoints |
| `ERROR_RATE` | `0.001` | Sequencing error rate |
| `INDEL_DENSITY` | `0.0` | Indels per 10kb (`0` = SNV-only) |
| `INDEL_MAX_SIZE` | `10` | Maximum indel size in bp |
| `N_STRAINS` | per level | Override per-complexity default strain count |
| `SNV_COUNTS` | per level | Override per-complexity default SNV-count list |
| `WORKERS` | `SLURM_CPUS_PER_TASK` or 8 | Parallel worker threads |

### What is tested

**Complexity (number of strains in the mix; defaults — overridable via `N_STRAINS`):**

| Level | Strains | SNV counts swept across strains[1:] |
|-------|---------|-----|
| 1 (simple) | 2 | `10000` |
| 2 (medium) | 4 | `2500, 5000, 10000` |
| 3 (complex) | 8 | `500, 1000, 2000, 5000, 6000, 7000, 10000` |

**Parameters in `REQUIRED_GRID` (full grid mode):**

- `window_size`: `[500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]`
- `max_mismatch_frac`: `[0.005, 0.01, 0.02, 0.05, 0.1]`
- `min_shared_snvs_for_edge`: `[1, 2, 3, 4, 5, 6]`
- `junk_divergence_rate`: `[0.05, 0.10, 0.2]`
- `merge_distance_threshold`: `[0.005, 0.01, 0.02, 0.05, 0.1]`
- `min_shared_for_merge`: `[1, 2, 3, 4, 5, 6]`
- `max_link_distance`: `[0.005, 0.01, 0.02, 0.05, 0.1]`
- `min_shared_snvs_for_link`: `[1, 2, 3, 4, 5, 6]`
- `min_depth_site`: `[3, 5, 10, 20, 50]`
- `include_indels`: `[false, true]` *(new)*

Sequential mode visits these in `DEFAULT_PARAM_ORDER`, varying one at a time and locking the best value before moving on.

**`best_params.json` (recommended starting grid, ~144 configs):**

Focuses on the parameters most likely to move accuracy. Smaller domains for less-impactful axes. The `include_indels: [false, true]` axis verifies the new indel pipeline does not regress on SNV-only inputs, and exercises the CIGAR walk on indel-rich inputs.

**Metrics computed per config (when truth is available):**

- SNV-level: precision / recall / F1
- Haplotype-level: precision / recall / F1, abundance MAE, abundance Pearson r
- Lineage-level: precision / recall / F1, trajectory error
- Track / linking metrics
- Sweep detection (winner / loser identification)
- Convergence rate, mean confidence, runtime

### Recently fixed

These were gaps in the previous version of the suite and have been addressed:

- **Indel pipeline coverage.** `simulate_reads.py` now injects deletions and insertions when `--indel-density > 0`. Reads from strains carrying indels emit `D`/`I` CIGAR ops in the SAM/BAM. The truth VCF gets canonical left-anchored indel records. The sweep grid has an `include_indels` axis so the new code path is exercised.
- **Coverage and timepoints are now overridable** at the `cluster_benchmark.sh` level via the `COVERAGE` and `TIMEPOINTS` env vars. Sweep them by submitting once per value (see "How to run").
- **Default coverage is consistent.** Both `cluster_benchmark.sh` and `run_full_benchmark.py` default to `30x`.
- **Strain count and SNV diversity can be set independently** via `N_STRAINS` and `SNV_COUNTS` env vars. Defaults still follow the original complexity-level table.
- **Error rate is a knob.** `ERROR_RATE` env var (default `0.001`) controls the simulator's HiFi error rate.
- **`test_configs.json` removed.** It had no callers.
- **`best_params.json` rewritten** to focus on impactful axes only and to include the `include_indels` toggle.

### Remaining gaps

1. **Synthetic alignments are still "perfect" in homopolymer runs.** The simulator emits exact-position indels, while real long-read aligners (minimap2) place indels variably in repeats. Phasing accuracy on real data may differ from synthetic results. The `validation/prepare_isolate_mix.py` real-strain track tests against actual BAMs but, per `MEMORY.md`, has flawed ground truth.
2. **Read length / error rate are still fixed within a sweep.** They can be set per-run via env vars but are not Cartesian-swept by `cluster_benchmark.sh`. Same loop pattern as coverage works.
3. **Indel injection is uniform random.** Real microbial indel hot spots (tandem repeats, IS elements) are not modeled.

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.
