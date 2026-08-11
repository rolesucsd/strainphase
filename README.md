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
    max_mismatch_frac=0.01,
    min_weight_for_anchor=0.20,
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
| `max_mismatch_frac` | 0.01 | Max Hamming distance for graph edges |
| `min_shared_snvs_for_edge` | 2 | Min shared SNVs to connect reads |
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

Benchmarking is **not distributed with the repository right now.** Both suites
are gitignored and live only in the working tree and in git history. Nothing
below is runnable from a fresh clone yet; this section records what exists and
what has to be true before it ships.

### `benchmark/` — the public tool comparison

A Snakemake workflow that compares strainphase against Floria and Strainy on
longitudinal mixtures built from **real strain assemblies**, with `spbench` as
the scoring library. Every tool is scored on its **read partition**, and the
harness derives consensus haplotypes from that partition with one code path for
all tools, so no tool is scored through another tool's output format.
`strainphase-single` (cross-timepoint rescue off) sits beside the external
tools; the longitudinal claim is the gap between it and
`strainphase-longitudinal`, which is the same code one flag apart.

```
benchmark/
├── config/config.yaml     strain groups, seeds, tools
├── workflow/              Snakefile + rules: simulate, pipeline, tools, evaluate
│   ├── envs/              one conda env per tool
│   └── scripts/           snoopy_to_vcf.py
└── spbench/               scoring library (metrics, truth, report)
```

Run it with `snakemake --use-conda --cores 16`, or `--profile <slurm-profile>`
on a cluster; Snakemake handles job submission, so there are no hand-written
sbatch scripts by design.

**Known blockers, in the order they have to be fixed:**

1. **`call_variants` cannot run.** The rule invokes `snoopy` under
   `workflow/envs/snoopy.yaml`, which does not exist and never has. The
   `spbench` env pins `clair3` instead, so the workflow and its environment
   disagree about the variant caller. Everything downstream of this rule is
   blocked.
2. **No way to obtain the input data.** `config/config.yaml` points `groups` at
   `assemblies/group3` and friends, and the README tells you to supply your own
   directories. There is no accession list and no fetch script, so a reader who
   clones this cannot reproduce a single number — which is the whole point of
   shipping it.
3. **The suite has never run end to end.** Badread, minimap2 and SNooPy are
   conda-only and the simulate → align → call → compare chain has only been
   dry-run. Treat every number as provisional until that changes.
4. **`naive-greedy` is cited but cannot be produced.** `spbench/report.py`
   explains that the row exists to expose the over-splitting failure mode, but
   no rule or adapter generates it. Either restore the baseline or drop the
   paragraph.

Comparators worth adding before submission: **devider** (same author as Floria,
the obvious "why not the current method?" question), a **naive greedy baseline**
so a reviewer can tell a real result from an easy dataset, and **whatshap** at
k=2 to answer "why not a standard diploid phaser?" with a measurement.

### `benchmarks/` + `validation/` — internal parameter sweeps

The older tooling: a SLURM-array parameter sweep (`parameter_sweep.py`), a
file-based read simulator with indel injection (`validation/simulate_reads.py`),
and Floria/Strainy comparison scripts. Still used —
`strainphase sweep` shells out to `benchmarks/parameter_sweep.py`, and
`tests/test_parsing.py` imports `validation.simulate_reads`, so a fresh clone
currently fails those tests. This is not superseded by `benchmark/`; the two
answer different questions.

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.
