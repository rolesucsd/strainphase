# Synthetic Community Tools

Tools for generating and testing synthetic metagenomic communities with multiple species and strains.

## Quick Start

### Generate synthetic community (40 species, 120 strains)

```bash
cd synthetic_community_tools

python generate_synthetic_community.py -o ../synthetic_output
```

### Validate the results

```bash
python test_synthetic_community.py ../synthetic_output

# View sample species details
python test_synthetic_community.py ../synthetic_output --show-samples
```

## What's Included

### Scripts

- **`generate_synthetic_community.py`** - Main generation script
  - Creates 40 species with 120 total strains
  - Generates reference genomes (FASTA), VCF files, and metadata
  - Fully customizable parameters

- **`test_synthetic_community.py`** - Validation script
  - Validates generated community data
  - Checks file integrity and statistics
  - Shows summary information

- **`download_real_genomes.py`** - Optional real genome downloader
  - Downloads actual bacterial genomes from NCBI RefSeq
  - **Note:** Don't run unless you want to download large files
  - Use this if you need real genomes instead of synthetic ones

### Documentation

- **`SETUP_COMPLETE.md`** - Overview and getting started guide
- **`README_synthetic_community.md`** - Complete detailed documentation
- **`QUICKSTART.md`** - Quick reference and examples

## Basic Usage

```bash
# Default: 40 species, 120 strains, 4 timepoints
python generate_synthetic_community.py -o output_dir

# Custom parameters
python generate_synthetic_community.py \
  -o output_dir \
  --species 50 \
  --strains 150 \
  --timepoints 6

# With specific seed for reproducibility
python generate_synthetic_community.py -o output_dir --seed 12345

# Get help
python generate_synthetic_community.py --help
```

## Output Structure

```
output_dir/
├── references/           # Reference genomes (FASTA)
├── vcfs/                # Variant call files
├── strain_metadata.json # Complete strain information
├── strain_abundances.tsv# Abundance table
└── community_summary.txt# Human-readable summary
```

## Documentation Files

- **Start here:** `SETUP_COMPLETE.md`
- **Quick commands:** `QUICKSTART.md`
- **Full details:** `README_synthetic_community.md`

## Requirements

- Python 3.7+
- NumPy
- Pandas (for validation script)

Optional (for real genome download):
- ncbi-datasets-cli

## Examples

### Generate and validate

```bash
# Generate
python generate_synthetic_community.py -o test_output

# Validate
python test_synthetic_community.py test_output

# View summary
cat test_output/community_summary.txt
```

### Large community

```bash
python generate_synthetic_community.py \
  -o large_community \
  --species 100 \
  --strains 500
```

### High diversity

```bash
python generate_synthetic_community.py \
  -o high_diversity \
  --snv-density 5.0
```

## What Gets Generated

### Synthetic Data (Default)
- **Artificial genomes:** Randomly generated sequences
- **120 strains:** Distributed across 40 species (average ~3 per species)
- **Temporal dynamics:** 4 timepoints with varying abundances
- **Strain relationships:** Simulated evolutionary relationships

### Key Features
- Realistic genome sizes (2-5 Mb)
- Power-law abundance distribution
- SNV positions with realistic clustering
- Ground truth metadata for validation

## Next Steps After Generation

1. **Validate** - Run the test script to verify data integrity
2. **Generate reads** - Use pbsim3 or InSilicoSeq to create sequencing reads
3. **Test pipeline** - Use with StrainPhase to validate haplotype detection
4. **Benchmark** - Compare detected strains against ground truth

## File Sizes

- 40 species, 120 strains: ~150-200 MB
- 100 species, 500 strains: ~400-600 MB
- 200 species, 1000 strains: ~800-1200 MB

## Support

See the documentation files in this folder:
- `SETUP_COMPLETE.md` - Setup and overview
- `README_synthetic_community.md` - Full documentation
- `QUICKSTART.md` - Quick reference guide

Or run with `--help`:
```bash
python generate_synthetic_community.py --help
python test_synthetic_community.py --help
```
