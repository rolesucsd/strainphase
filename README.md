# Strainphase

**Hybrid graph-probabilistic haplotype reconstruction for PacBio HiFi metagenomic data**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-yellow.svg)](https://opensource.org/license/bsd-3-clause/)

## Overview

Strainphase phases strain-resolved haplotypes from PacBio HiFi metagenomes. It
divides each genome into overlapping windows and, within each window, clusters
reads that share alleles into candidate haplotypes, refines them with a
quality-weighted expectation–maximization model, and links haplotypes across
windows and timepoints into contig-spanning lineages. Indels and structural
variants are phased as alleles rather than discarded, and low-abundance
haplotypes below single-timepoint detection are recovered from the other
timepoints. It resolves intra-strain lineages differing by less than 0.5%.

## Installation

```bash
git clone https://github.com/rolesucsd/strainphase.git
cd strainphase
pip install -e .
```

**Dependencies:** `numpy`, `scipy`, `networkx`, `python-louvain`, `pysam`

## Quick Start

### Demo dataset

A small demo dataset is bundled in [`demo/`](demo/) so the tool can be run immediately
without supplying data. It is six timepoints of two strains sweeping past each other
over a 100 kb contig, and it runs in under a minute:

```bash
bash demo/run_demo.sh
```

This phases all six timepoints and prints the recovered trajectories beside the
bundled ground truth. See [`demo/README.md`](demo/README.md) for what the dataset
contains and what the output should look like.

### Command Line Interface

```bash
# Process single contig
# The BAM must be sorted and indexed; the VCF bgzip-compressed and tabix-indexed.
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
    identity_distance=0.02,
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
| `strainphase sv` | Structural-variant sidecar tools (`reconcile`, `verify`) |
| `strainphase test` | Run the unit test suite (source checkout only) |
| `strainphase version` | Show version |

### `strainphase run`

```
strainphase run --bam FILE --vcf FILE --contig ID --length INT [OPTIONS]

Required:
  --bam FILE          Input BAM file (sorted, indexed)
  --vcf FILE          Input VCF file (Clair3 format; bgzip-compressed, tabix-indexed)
  --contig ID         Contig ID to process
  --length INT        Contig length in bp

Options:
  --sample ID         Sample identifier
  --output FILE       Output TSV file [default: haplotypes.tsv]
  --window-size INT   Analysis window size [default: 20000]
  --max-reads INT     Max reads per window [default: 500]
  --min-mapq INT      Minimum MAPQ [default: 20]
  --identity-distance FLT  Max mismatch RATE at which two things are one entity [default: 0.02]
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
  --sv-sidecars TPL        SV sidecar TSV template with {sample}; co-phases structural variants
  -j, --workers INT        Worker processes for window processing [default: 1]
  --window-size INT        Window size [default: 20000]
  --max-reads INT          Max reads per window [default: 500]
  --min-anchor-weight FLT  Minimum weight for anchor panel [default: 0.15]
  --rescued-min-weight FLT Minimum weight after rescue [default: 0.02]
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 20000 | Analysis window size (bp) |
| `identity_distance` | 0.02 | Max mismatch RATE at which two things are one entity — read to read, haplotype to haplotype, and rescue |
| `min_shared_snvs_for_edge` | 1 | Shared SNVs before that rate means anything, read to read |
| `min_shared_markers` | 3 | Shared markers before it means anything, haplotype to haplotype |
| `track_merge_min_shared_markers` | 1 | Agreeing markers before the cross-sample merge joins two tracks |
| `link_window_reach` | 2 | Windows ahead step 1 may link; >1 links NON-overlapping windows on shared reads |
| `assign_confidence_threshold` | 0.90 | γ threshold for hard read assignment |
| `min_weight_for_anchor` | 0.15 | Min abundance for anchor panel |
| `rescued_min_weight` | 0.02 | Min weight after longitudinal rescue |
| `junk_divergence_rate` | 0.10 | Junk model divergence rate |

## Output Format

### `strainphase run` → `haplotypes.tsv`

One row per track (a haplotype linked across windows) on the contig.

| Column | Description |
|--------|-------------|
| `contig` | Contig ID |
| `sample` | Sample ID |
| `track_id` | Linked haplotype track identifier |
| `span_start`, `span_end`, `span_bp` | Track extent |
| `n_windows` | Windows the track covers |
| `n_snvs` | Number of sites in the consensus |
| `mean_weight` | Read-weighted abundance pooled over the track's windows; `NaN` where no window could measure one |
| `total_supporting_reads` | Reads confidently assigned to the track |
| `mean_confidence` | Mean assignment confidence |
| `consensus` | Allele profile, `pos:allele` pairs separated by `\|` |

### `strainphase longitudinal` → `--output-dir`

| File | One row per |
|------|-------------|
| `lineages.tsv` | (lineage, sample): the main output |
| `haplotypes.tsv` | window-haplotype, with its consensus |
| `windows_within_sample.tsv` | within-sample track |
| `windows_across_samples.tsv` | per-window group of samples judged one entity |

Diagnostics are written to `tmp/` beside them.

`lineages.tsv`:

| Column | Description |
|--------|-------------|
| `lineage_id` | Lineage identifier |
| `mag`, `contig`, `sample` | MAG, contig and sample/timepoint |
| `n_windows_lineage`, `n_samples_lineage` | Windows and samples the lineage covers in total |
| `window_start`, `window_end` | Lineage window extent |
| `marker_start`, `marker_end`, `marker_span_bp` | First and last marker position, and the distance between them |
| `n_windows` | Windows this sample contributes |
| `abundance` | Reads assigned / non-junk reads, pooled over the lineage's windows in this sample |
| `abundance_all_reads` | The same over all reads, junk included |
| `reads`, `total_reads`, `junk_reads` | The counts behind both abundances |
| `window_group_ids` | Per-window groups this sample contributes to |
| `haplotype_ids` | Window-haplotype IDs behind the row (join key to `haplotypes.tsv`) |

An `abundance` of 0 with non-zero `total_reads` is a measured zero. A (lineage, sample)
pair with no row was not measured there.

## Structural variants

`--sv-sidecars` co-phases structural variants with SNVs and indels. Each sample needs a
sidecar TSV listing its SV events and the reads supporting them; the format is
documented at the top of `strainphase/sv_encoding.py`. For Sniffles2 calls made with
`--output-rnames`, `scripts/sniffles_to_sidecar.py` in this repository writes one:

```bash
python scripts/sniffles_to_sidecar.py --sniffles T1.sv.vcf.gz --out-sidecar sv/T1.tsv
strainphase sv reconcile --sidecars sv/*.tsv --out-dir sv_reconciled/
```

`reconcile` gives one event the same ID and position in every sample; run it before
phasing.

## Algorithm Overview

```
1. WINDOWS   BAM + VCF → overlapping 20 kb windows, 10 kb step
2. GRAPH     read-similarity graph per window → Louvain communities
3. EM        γ responsibilities ↔ π weights and consensus, with a junk component
4. MERGE     within-window: merge haplotypes within identity_distance;
             a one-marker difference is adjudicated against a sequencing-error null
5. LINK      chain a sample's haplotypes across windows into tracks
6. RESCUE    recover low-abundance haplotypes from other timepoints (longitudinal)
7. LINEAGES  merge tracks across samples into contig-spanning lineages
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.
