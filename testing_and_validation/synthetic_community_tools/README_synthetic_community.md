# Synthetic Community Generator

Generate synthetic metagenomic communities with multiple species and strains for testing the StrainPhase pipeline.

## Quick Start

### Generate default community (40 species, 120 strains)

```bash
python generate_synthetic_community.py -o synthetic_data
```

This will create:
- 40 bacterial species with realistic names
- 120 total strains distributed across species (average 3 strains per species)
- 4 timepoints with varying abundances
- Reference genomes (FASTA files)
- VCF files with SNV positions
- Metadata files with ground truth strain abundances

## Custom Parameters

### Specify custom species/strain counts

```bash
python generate_synthetic_community.py \
  -o output_dir \
  --species 50 \
  --strains 200 \
  --timepoints 6
```

### Set SNV density

```bash
python generate_synthetic_community.py \
  -o output_dir \
  --snv-density 3.5  # SNVs per kb
```

### Use specific random seed for reproducibility

```bash
python generate_synthetic_community.py \
  -o output_dir \
  --seed 12345
```

## Output Structure

```
output_dir/
├── references/
│   ├── species_000.fasta
│   ├── species_001.fasta
│   └── ...
├── vcfs/
│   ├── species_000.vcf
│   ├── species_001.vcf
│   └── ...
├── strain_metadata.json         # Complete strain info with abundances
├── strain_abundances.tsv        # Abundance table (easy to import)
└── community_summary.txt        # Human-readable summary
```

## Understanding the Output

### strain_metadata.json
Contains complete information about each species and strain:
- Species ID and name
- Genome length
- Number of SNVs
- All strain IDs and their abundances at each timepoint

### strain_abundances.tsv
Tab-separated table with columns:
- `species_id`: Species identifier
- `strain_id`: Strain identifier
- `T1`, `T2`, ...: Abundance at each timepoint (0.0 to 1.0)

### Reference Genomes
FASTA files for each species' reference genome. These can be used with alignment tools like minimap2.

### VCF Files
Variant Call Format files listing all SNV positions for each species with reference and alternate alleles.

## Generating Reads

The script generates a manifest file but not actual sequencing reads. To generate reads, you'll need an external simulator:

### Using pbsim3 (PacBio HiFi)

```bash
# For each species reference genome
pbsim3 \
  --prefix output/species_000_reads \
  --depth 30 \
  --length-mean 10000 \
  --accuracy-mean 0.999 \
  references/species_000.fasta
```

### Using InSilicoSeq (Illumina or other platforms)

```bash
iss generate \
  --genomes references/species_000.fasta \
  --model hiseq \
  --output species_000_reads \
  --abundance_file abundances.txt
```

## Using with StrainPhase

Once you have generated reads and aligned them:

```bash
# 1. Align reads to reference
minimap2 -ax map-hifi reference.fasta reads.fastq | samtools sort > aligned.bam
samtools index aligned.bam

# 2. Call variants with Clair3
clair3.py --bam_fn aligned.bam --ref_fn reference.fasta --output vcf_output

# 3. Run StrainPhase
strainphase \
  --bam aligned.bam \
  --vcf clair3_output/merge_output.vcf.gz \
  --contig species_000 \
  --output results.tsv
```

## Example: Full Workflow

Here's a complete example from generation to analysis:

```bash
# 1. Generate synthetic community
python generate_synthetic_community.py -o synthetic_data

# 2. Check the summary
cat synthetic_data/community_summary.txt

# 3. View strain abundances
head synthetic_data/strain_abundances.tsv

# 4. Examine metadata
python -m json.tool synthetic_data/strain_metadata.json | head -50

# 5. Use reference genomes with read simulators
# (See "Generating Reads" section above)

# 6. Process with StrainPhase pipeline
# (See "Using with StrainPhase" section above)
```

## Features of the Synthetic Data

### Realistic Diversity
- Variable number of strains per species (some species have more diversity)
- Power-law abundance distribution (mimics real microbiomes)
- SNV positions with realistic clustering patterns

### Temporal Dynamics
- Abundance changes over timepoints
- Some species stable, some varying
- Realistic abundance distributions (uniform, dominant, or skewed per species)

### Strain Relationships
- Strains within species are related (derived from common ancestor)
- Variable divergence rates (2-10% SNP differences)
- First strain ~5% divergent from reference

## Validation

To validate the generated community:

```python
import json

# Load metadata
with open('synthetic_data/strain_metadata.json') as f:
    metadata = json.load(f)

# Check totals
print(f"Species: {metadata['n_species']}")
print(f"Total strains: {metadata['total_strains']}")

# Verify abundances sum to 1.0 for each timepoint
import pandas as pd
abund = pd.read_csv('synthetic_data/strain_abundances.tsv', sep='\t')
for tp in metadata['timepoints']:
    total = abund[tp].sum()
    print(f"Timepoint {tp}: {total:.6f}")
```

## Parameters Explained

### --species (default: 40)
Number of distinct species in the community. Each species has its own reference genome and independent SNV sites.

### --strains (default: 120)
Total number of strains across all species. These are distributed non-uniformly - some species will have more strains than others, representing higher intra-species diversity.

### --timepoints (default: 4)
Number of time points to simulate. Each strain has different abundances at each timepoint, allowing for temporal dynamics analysis.

### --snv-density (default: 2.0)
Number of SNVs per kilobase of genome. This controls how many variant sites exist in each species. Higher values create more complex haplotypes.

### --seed (default: 42)
Random seed for reproducibility. Using the same seed will generate identical communities.

## Troubleshooting

### Memory usage
If generating very large communities (>100 species), the script may use significant memory. Consider generating in batches or reducing genome lengths.

### File sizes
Each reference genome is 2-5 Mb. For 40 species, expect ~100-200 MB of FASTA files.

### Read generation
The script itself doesn't generate actual reads - you need external tools like pbsim3 or InSilicoSeq for that step.

## Citation

If you use this synthetic data generator in your research, please cite:
- StrainPhase: [Citation info]
- pbsim3 (if used): Ono, Y., et al. (2021)
- InSilicoSeq (if used): Gourlé, H., et al. (2019)
