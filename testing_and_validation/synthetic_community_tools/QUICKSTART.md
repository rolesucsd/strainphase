# Quick Start Guide

## Generate and Test Synthetic Community (40 species, 120 strains)

### 1. Generate the synthetic community

```bash
cd /Users/reneeoles/Desktop/strainphase

# Generate with default parameters (40 species, 120 strains, 4 timepoints)
python generate_synthetic_community.py -o synthetic_output

# This will take a few minutes and create:
# - 40 reference genomes (FASTA)
# - 40 VCF files with variant positions
# - Metadata and abundance tables
```

### 2. Validate the generated data

```bash
# Run validation
python test_synthetic_community.py synthetic_output

# View sample species details
python test_synthetic_community.py synthetic_output --show-samples
```

### 3. Explore the output

```bash
# View the summary
cat synthetic_output/community_summary.txt

# Check strain abundances
head -20 synthetic_output/strain_abundances.tsv

# View metadata (formatted)
python -m json.tool synthetic_output/strain_metadata.json | less

# List reference genomes
ls -lh synthetic_output/references/

# List VCF files
ls -lh synthetic_output/vcfs/
```

## Example Output

When you run the generation script, you'll see output like:

```
================================================================================
SYNTHETIC COMMUNITY GENERATOR
================================================================================
Output directory: synthetic_output
Species: 40
Total strains: 120
Timepoints: 4
SNV density: 2.0 per kb
Random seed: 42
================================================================================

Generating community with 40 species and 120 strains...
Strain distribution: min=1, max=6, mean=3.0
  Timepoint T1 total abundance: 1.000
  Timepoint T2 total abundance: 1.000
  Timepoint T3 total abundance: 1.000
  Timepoint T4 total abundance: 1.000

Writing reference genomes...
  Wrote synthetic_output/references/species_000.fasta
  Wrote synthetic_output/references/species_001.fasta
  ...

Writing VCF files...
  Wrote synthetic_output/vcfs/species_000.vcf
  ...

Writing metadata to synthetic_output/strain_metadata.json...
  Wrote metadata file

Writing abundance table to synthetic_output/strain_abundances.tsv...
  Wrote abundance table

Writing summary to synthetic_output/community_summary.txt...
  Wrote summary file

================================================================================
GENERATION COMPLETE!
================================================================================
```

## Customization Examples

### More species and strains

```bash
python generate_synthetic_community.py \
  -o large_community \
  --species 100 \
  --strains 500 \
  --timepoints 8
```

### Higher SNV density (more variants)

```bash
python generate_synthetic_community.py \
  -o high_diversity \
  --snv-density 5.0
```

### Reproducible generation with specific seed

```bash
python generate_synthetic_community.py \
  -o reproducible_output \
  --seed 12345
```

## Next Steps

### Option 1: Generate actual sequencing reads

Use a read simulator like pbsim3 or InSilicoSeq:

```bash
# Install pbsim3
# https://github.com/yukiteruono/pbsim3

# Generate reads for each species
for ref in synthetic_output/references/*.fasta; do
    species=$(basename $ref .fasta)
    pbsim3 \
        --prefix reads/${species} \
        --depth 30 \
        --length-mean 10000 \
        --accuracy-mean 0.999 \
        $ref
done
```

### Option 2: Use with StrainPhase directly

If you have reads from the synthetic data:

```bash
# 1. Align reads
minimap2 -ax map-hifi \
    synthetic_output/references/species_000.fasta \
    reads.fastq | \
    samtools sort -o aligned.bam
samtools index aligned.bam

# 2. Call variants
clair3.py \
    --bam_fn aligned.bam \
    --ref_fn synthetic_output/references/species_000.fasta \
    --output vcf_output \
    --platform hifi

# 3. Run StrainPhase
python src/strainphase/core.py \
    --bam aligned.bam \
    --vcf vcf_output/merge_output.vcf.gz \
    --contig species_000 \
    --length 4500000 \
    --output strainphase_results.tsv
```

## File Structure

```
synthetic_output/
├── references/
│   ├── species_000.fasta       # Reference genome for species 0
│   ├── species_001.fasta       # Reference genome for species 1
│   └── ...                     # (40 total)
│
├── vcfs/
│   ├── species_000.vcf         # Variant positions for species 0
│   ├── species_001.vcf         # Variant positions for species 1
│   └── ...                     # (40 total)
│
├── strain_metadata.json        # Complete metadata (JSON)
│   └── Contains:
│       - Species information
│       - Strain IDs and abundances
│       - Genome statistics
│
├── strain_abundances.tsv       # Abundance table (TSV)
│   └── Columns:
│       - species_id
│       - strain_id
│       - T1, T2, T3, T4 (abundance at each timepoint)
│
└── community_summary.txt       # Human-readable summary
    └── Contains:
        - Overview statistics
        - Species details
        - Strain lists
```

## Understanding the Data

### Species
- 40 distinct bacterial species
- Each has a unique reference genome (2-5 Mb)
- Genome lengths vary realistically
- SNV positions distributed across genome

### Strains
- 120 total strains across all species
- Distribution is non-uniform (some species have more strains)
- Range: 1-6 strains per species (average ~3)
- Strains within a species share ancestry

### Strain Relationships
- First strain per species is ~5% divergent from reference
- Subsequent strains are 2-10% divergent from first strain
- SNP differences create distinguishable haplotypes

### Abundances
- Vary across timepoints (temporal dynamics)
- Follow realistic distributions (power law)
- Sum to 1.0 at each timepoint
- Some strains increase/decrease over time

### Variant Positions (SNVs)
- Density: ~2 SNVs per kb (adjustable)
- Distributed with realistic clustering
- Both reference and alternate alleles defined
- Used to distinguish strains

## Troubleshooting

### Script fails to import strainphase modules

```bash
# Make sure you're in the correct directory
cd /Users/reneeoles/Desktop/strainphase

# Or install strainphase package
pip install -e .
```

### Memory issues with large communities

```bash
# Generate smaller batches
python generate_synthetic_community.py -o batch1 --species 20 --strains 60
python generate_synthetic_community.py -o batch2 --species 20 --strains 60
```

### Validation fails

```bash
# Check the error messages
python test_synthetic_community.py synthetic_output

# Common issues:
# - Missing files: Re-run generation
# - Abundances don't sum to 1.0: This is expected (numerical precision)
#   As long as they're close (0.999-1.001), it's fine
```

## Getting Help

For more detailed information, see:
- `README_synthetic_community.md` - Full documentation
- `generate_synthetic_community.py --help` - Command-line options
- `test_synthetic_community.py --help` - Validation options

## Success Indicators

You'll know everything worked correctly when:

1. ✓ Generation completes without errors
2. ✓ Validation passes all checks
3. ✓ Abundances sum to ~1.0 for each timepoint
4. ✓ File counts match species count (40 FASTA, 40 VCF)
5. ✓ Metadata shows 120 total strains
6. ✓ Summary file is readable and shows expected distributions
