#!/usr/bin/env python3
"""
Test and validate synthetic community generation.

This script validates the generated synthetic community and provides
summary statistics.
"""

import json
import argparse
from pathlib import Path
import pandas as pd


def validate_community(output_dir: Path):
    """Validate the generated community data."""

    print("="*80)
    print("VALIDATING SYNTHETIC COMMUNITY")
    print("="*80)

    output_dir = Path(output_dir)

    # Check required files exist
    required_files = [
        'strain_metadata.json',
        'strain_abundances.tsv',
        'community_summary.txt'
    ]

    required_dirs = [
        'references',
        'vcfs'
    ]

    print("\n1. Checking required files and directories...")
    all_exist = True

    for file in required_files:
        path = output_dir / file
        if path.exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            all_exist = False

    for dir_name in required_dirs:
        path = output_dir / dir_name
        if path.exists() and path.is_dir():
            n_files = len(list(path.glob('*')))
            print(f"  ✓ {dir_name}/ ({n_files} files)")
        else:
            print(f"  ✗ {dir_name}/ - MISSING")
            all_exist = False

    if not all_exist:
        print("\n✗ Validation FAILED - missing required files/directories")
        return False

    # Load and validate metadata
    print("\n2. Validating metadata...")
    metadata_file = output_dir / 'strain_metadata.json'

    try:
        with open(metadata_file) as f:
            metadata = json.load(f)

        n_species = metadata['n_species']
        total_strains = metadata['total_strains']
        n_timepoints = metadata['n_timepoints']

        print(f"  ✓ Metadata loaded successfully")
        print(f"    - Species: {n_species}")
        print(f"    - Total strains: {total_strains}")
        print(f"    - Timepoints: {n_timepoints}")

        # Count actual strains in metadata
        actual_strains = sum(len(sp['strains']) for sp in metadata['species'])
        if actual_strains == total_strains:
            print(f"  ✓ Strain count matches: {actual_strains} == {total_strains}")
        else:
            print(f"  ✗ Strain count mismatch: {actual_strains} != {total_strains}")
            return False

    except Exception as e:
        print(f"  ✗ Error loading metadata: {e}")
        return False

    # Load and validate abundance table
    print("\n3. Validating abundance table...")
    abund_file = output_dir / 'strain_abundances.tsv'

    try:
        abund_df = pd.read_csv(abund_file, sep='\t')
        print(f"  ✓ Abundance table loaded: {len(abund_df)} rows")

        # Check row count matches strain count
        if len(abund_df) == total_strains:
            print(f"  ✓ Row count matches total strains: {len(abund_df)} == {total_strains}")
        else:
            print(f"  ✗ Row count mismatch: {len(abund_df)} != {total_strains}")
            return False

        # Validate abundances sum to 1.0 for each timepoint
        print(f"\n4. Validating abundance sums...")
        timepoints = metadata['timepoints']
        all_valid = True

        for tp in timepoints:
            total = abund_df[tp].sum()
            if abs(total - 1.0) < 0.001:
                print(f"  ✓ {tp}: {total:.6f} (≈ 1.0)")
            else:
                print(f"  ✗ {tp}: {total:.6f} (should be 1.0)")
                all_valid = False

        if not all_valid:
            print("\n  ⚠ WARNING: Abundances don't sum to 1.0")

    except Exception as e:
        print(f"  ✗ Error loading abundance table: {e}")
        return False

    # Validate reference files
    print("\n5. Validating reference genomes...")
    ref_dir = output_dir / 'references'
    ref_files = list(ref_dir.glob('*.fasta'))

    if len(ref_files) == n_species:
        print(f"  ✓ Reference count matches: {len(ref_files)} == {n_species}")
    else:
        print(f"  ✗ Reference count mismatch: {len(ref_files)} != {n_species}")
        return False

    # Check a sample reference file
    sample_ref = ref_files[0]
    with open(sample_ref) as f:
        first_line = f.readline()
        if first_line.startswith('>'):
            print(f"  ✓ Sample reference has valid FASTA header")
        else:
            print(f"  ✗ Sample reference missing FASTA header")
            return False

    # Validate VCF files
    print("\n6. Validating VCF files...")
    vcf_dir = output_dir / 'vcfs'
    vcf_files = list(vcf_dir.glob('*.vcf'))

    if len(vcf_files) == n_species:
        print(f"  ✓ VCF count matches: {len(vcf_files)} == {n_species}")
    else:
        print(f"  ✗ VCF count mismatch: {len(vcf_files)} != {n_species}")
        return False

    # Check a sample VCF file
    sample_vcf = vcf_files[0]
    with open(sample_vcf) as f:
        first_line = f.readline()
        if first_line.startswith('##fileformat=VCF'):
            print(f"  ✓ Sample VCF has valid header")
        else:
            print(f"  ✗ Sample VCF missing header")
            return False

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nSpecies distribution:")
    species_strains = [len(sp['strains']) for sp in metadata['species']]
    print(f"  Min strains per species: {min(species_strains)}")
    print(f"  Max strains per species: {max(species_strains)}")
    print(f"  Mean strains per species: {sum(species_strains)/len(species_strains):.2f}")

    print(f"\nAbundance ranges:")
    for tp in timepoints:
        abundances = abund_df[tp].values
        print(f"  {tp}:")
        print(f"    Min: {abundances.min():.6f}")
        print(f"    Max: {abundances.max():.6f}")
        print(f"    Mean: {abundances.mean():.6f}")
        print(f"    Median: {pd.Series(abundances).median():.6f}")

    print(f"\nGenome statistics:")
    genome_lengths = [sp['genome_length'] for sp in metadata['species']]
    snv_counts = [sp['n_snvs'] for sp in metadata['species']]

    print(f"  Genome lengths:")
    print(f"    Min: {min(genome_lengths):,} bp")
    print(f"    Max: {max(genome_lengths):,} bp")
    print(f"    Mean: {sum(genome_lengths)//len(genome_lengths):,} bp")

    print(f"  SNV counts:")
    print(f"    Min: {min(snv_counts)}")
    print(f"    Max: {max(snv_counts)}")
    print(f"    Mean: {sum(snv_counts)//len(snv_counts)}")

    # All checks passed
    print("\n" + "="*80)
    print("✓ ALL VALIDATION CHECKS PASSED")
    print("="*80)
    print(f"\nSynthetic community in {output_dir} is valid and ready to use!")

    return True


def show_sample_species(output_dir: Path, n_samples: int = 3):
    """Show details of a few sample species."""

    print("\n" + "="*80)
    print("SAMPLE SPECIES DETAILS")
    print("="*80)

    metadata_file = output_dir / 'strain_metadata.json'
    with open(metadata_file) as f:
        metadata = json.load(f)

    for i, species in enumerate(metadata['species'][:n_samples]):
        print(f"\n{i+1}. {species['name']}")
        print(f"   ID: {species['id']}")
        print(f"   Genome: {species['genome_length']:,} bp")
        print(f"   SNVs: {species['n_snvs']}")
        print(f"   Strains: {species['n_strains']}")

        print(f"\n   Strain details:")
        for strain in species['strains']:
            abundances = [f"{strain['abundances'][tp]:.4f}"
                         for tp in metadata['timepoints']]
            print(f"     {strain['id']}: [{', '.join(abundances)}]")
            print(f"       Variant positions: {strain['n_variant_positions']}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated synthetic community"
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory containing generated synthetic data'
    )

    parser.add_argument(
        '--show-samples',
        action='store_true',
        help='Show details of sample species'
    )

    args = parser.parse_args()

    # Validate
    valid = validate_community(args.output_dir)

    # Show samples if requested
    if valid and args.show_samples:
        show_sample_species(args.output_dir)

    # Exit with appropriate code
    exit(0 if valid else 1)


if __name__ == "__main__":
    main()
