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
    window_size=20000,
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
  --window-size INT   Analysis window size [default: 20000]
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
  --window-size INT        Window size [default: 20000]
  --max-reads INT          Max reads per window [default: 300]
  --min-anchor-weight FLT  Minimum weight for anchor panel [default: 0.15]
  --rescued-min-weight FLT Minimum weight after rescue [default: 0.02]
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 20000 | Analysis window size (bp) |
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

A public benchmark suite lives in [`benchmark/`](benchmark/). It compares
strainphase against [Floria](https://github.com/bluenote-1577/floria) and
[Strainy](https://github.com/katerinakazantseva/strainy) on longitudinal
mixtures built from **real strain assemblies** — the genomes and their
differences are not simulated, only the abundances over time are.

```bash
conda env create -f benchmark/workflow/envs/spbench.yaml && conda activate spbench
cd benchmark && pip install -e .. -e .

$EDITOR config/config.yaml            # point `groups` at your assembly directories
snakemake --use-conda --cores 16      # or --profile <your-slurm-profile>
```

Output: `results/report/report.md`. It is a Snakemake workflow, so cluster
submission, resume and per-rule resources come from your existing profile.

### How a dataset is built

1. **Strains** — a group of closely related assemblies. One is drawn as the
   reference; the rest are the mixture. True haplotypes come from
   `minimap2 -cx asm5` of each assembly against that reference.
2. **Abundances** — simulated from named biological archetypes: `stable`,
   `bloom`, `colonisation` (starts at exactly 0, logistic growth), `decline`,
   and a crossing `sweep` pair.
3. **Reads** — Badread, run per strain at its abundance × coverage. Badread is
   what Strainy's own HiFi benchmark used.
4. **Alignment and calling** — the production workflow's own commands
   (`workflow/rules/pipeline.smk`), so every tool receives the BAMs and VCFs the
   real pipeline produces.

### How the comparison stays fair

strainphase uses several timepoints at once; Floria and Strainy phase one sample
in isolation. Three design choices keep that meaningful:

1. **Every tool is scored on its read partition**, with consensus haplotypes
   derived by the same code for all of them — so the comparison is not
   confounded by three different consensus callers.
2. **strainphase competes against itself.** `strainphase-single` (rescue
   disabled) is the row beside Floria and Strainy; the longitudinal claim is the
   gap to `strainphase-longitudinal`, one flag apart.
3. **Unclaimed columns read `n/a`, not zero.** Cross-timepoint identity is
   scored only for tools that claim it.

Detection sensitivity is reported stratified by true abundance *and* by absolute
strain depth (`abundance × coverage`), which separates "the method missed it"
from "the reads were not there".

See [`benchmark/README.md`](benchmark/README.md) for the metric definitions, the
abundance archetypes, how to add a tool, and the suite's known limitations.

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.
