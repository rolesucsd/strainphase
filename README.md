# Strainphase

**Hybrid graph-probabilistic haplotype reconstruction for PacBio HiFi metagenomic data**

[![PyPI version](https://badge.fury.io/py/strainphase.svg)](https://badge.fury.io/py/strainphase)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Strainphase reconstructs distinct bacterial haplotypes (strain-specific SNV patterns) from mixed metagenomic reads. It uses a hybrid approach combining:

1. **Graph-based initialization** - Louvain clustering of read overlap networks
2. **Probabilistic EM refinement** - Quality-weighted soft assignments
3. **Window linking** - Track assembly across overlapping genomic windows
4. **Longitudinal rescue** - Cross-timepoint detection of low-abundance strains

## Installation

### From PyPI (recommended)
```bash
pip install strainphase
```

### With optional dependencies
```bash
# Full installation (includes pysam for BAM/VCF I/O)
pip install strainphase[full]

# Development installation
pip install strainphase[dev]

# Everything
pip install strainphase[all]
```

### From source
```bash
git clone https://github.com/roles/strainphase.git
cd strainphase
pip install -e .
```

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

# Run tests
strainphase test

# Parameter sensitivity analysis
strainphase sweep --quick
```

### Python API

```python
from strainphase import HaplotyperConfig, process_contig, results_to_dataframe

# Configure
config = HaplotyperConfig(
    window_size=3000,
    max_mismatch_frac=0.02,
    min_weight_for_anchor=0.15,
)

# Process
results = process_contig(
    bam_path="sample.bam",
    vcf_path="variants.vcf.gz",
    contig_id="MAG_01_contig_1",
    contig_length=50000,
    config=config,
)

# Export
records = results_to_dataframe({"MAG_01_contig_1": results})
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `strainphase run` | Process a single contig |
| `strainphase longitudinal` | Multi-sample longitudinal analysis |
| `strainphase test` | Run unit test suite |
| `strainphase sweep` | Parameter sensitivity analysis |
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
  --mags LIST         Comma-separated MAG names [default: all]
  --contig-filter F   File listing allowed contigs
  --window-size INT   Window size [default: 3000]
  --max-reads INT     Max reads per window [default: 300]
  --min-anchor-weight Minimum weight for anchor panel [default: 0.15]
  --rescued-min-weight Minimum weight after rescue [default: 0.02]
```

### `strainphase sweep`

```
strainphase sweep [OPTIONS]

Options:
  --quick             Use reduced parameter grid (~2-5 min)
  --output-dir DIR    Output directory for results
  --max-configs INT   Limit number of configurations to test
  -q, --quiet         Suppress progress output
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

## Testing

```bash
# Run unit tests
strainphase test

# Run with verbose output
strainphase test -v

# Parameter sensitivity (quick)
strainphase sweep --quick

# Full parameter sweep (324 configs × 4 scenarios)
strainphase sweep --output-dir full_sweep_results/
```

### Test Scenarios

| Scenario | Haplotypes | Timepoints | Description |
|----------|------------|------------|-------------|
| `simple_2hap` | 2 | 3 | Clear separation, no sweep |
| `sweep_2hap` | 2 | 4 | Selective sweep dynamics |
| `complex_4hap` | 4 | 5 | Multiple related strains |
| `low_abundance` | 3 | 4 | Rare strain detection |

## Citation

If you use Strainphase in your research, please cite:

```bibtex
@software{strainphase,
  author = {Oles, Renee},
  title = {Strainphase: Hybrid graph-probabilistic haplotype reconstruction},
  year = {2025},
  url = {https://github.com/roles/strainphase}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
