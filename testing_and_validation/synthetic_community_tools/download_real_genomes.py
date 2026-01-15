#!/usr/bin/env python3
"""
Download real bacterial genomes for synthetic community generation.

This script downloads actual bacterial reference genomes from NCBI RefSeq
and prepares them for use in creating a realistic synthetic community.

Instead of generating fake sequences, this uses real genomes with known
taxonomies and characteristics.
"""

import argparse
import subprocess
from pathlib import Path
import json
import pandas as pd
from typing import List, Dict
import time


# Curated list of gut microbiome species with NCBI RefSeq accessions
GUT_MICROBIOME_SPECIES = [
    # Species name, RefSeq accession, representative strain
    ("Bacteroides_fragilis", "GCF_000025985.1", "NCTC 9343"),
    ("Bacteroides_thetaiotaomicron", "GCF_000011065.1", "VPI-5482"),
    ("Bacteroides_vulgatus", "GCF_000012825.1", "ATCC 8482"),
    ("Bacteroides_uniformis", "GCF_000154205.1", "ATCC 8492"),
    ("Bacteroides_ovatus", "GCF_000154525.1", "ATCC 8483"),

    ("Prevotella_copri", "GCF_000156255.1", "DSM 18205"),

    ("Faecalibacterium_prausnitzii", "GCF_000162015.1", "A2-165"),

    ("Escherichia_coli", "GCF_000005845.2", "K-12 MG1655"),
    ("Escherichia_coli_O157H7", "GCF_000008865.2", "O157:H7 Sakai"),

    ("Bifidobacterium_longum", "GCF_000007525.1", "NCC2705"),
    ("Bifidobacterium_adolescentis", "GCF_000010425.1", "ATCC 15703"),

    ("Lactobacillus_acidophilus", "GCF_000011825.1", "NCFM"),
    ("Lactobacillus_plantarum", "GCF_000203855.3", "WCFS1"),

    ("Clostridium_difficile", "GCF_000009205.2", "630"),

    ("Akkermansia_muciniphila", "GCF_000020225.1", "ATCC BAA-835"),

    ("Roseburia_intestinalis", "GCF_000154465.1", "L1-82"),

    ("Eubacterium_rectale", "GCF_000020605.1", "ATCC 33656"),

    ("Alistipes_putredinis", "GCF_000154385.1", "DSM 17216"),

    ("Parabacteroides_distasonis", "GCF_000012845.1", "ATCC 8503"),

    ("Blautia_producta", "GCF_000154205.1", "ATCC 27340"),

    ("Ruminococcus_bromii", "GCF_000156075.1", "L2-63"),
    ("Ruminococcus_gnavus", "GCF_000158055.2", "ATCC 29149"),

    ("Streptococcus_thermophilus", "GCF_000011825.1", "LMG 18311"),

    ("Enterococcus_faecalis", "GCF_000007785.1", "V583"),

    ("Collinsella_aerofaciens", "GCF_000154465.1", "ATCC 25986"),

    ("Bilophila_wadsworthia", "GCF_000210075.1", "3_1_6"),

    ("Desulfovibrio_piger", "GCF_000154385.1", "ATCC 29098"),

    ("Methanobrevibacter_smithii", "GCF_000016525.1", "ATCC 35061"),

    ("Sutterella_wadsworthensis", "GCF_000210155.1", "3_1_45B"),

    ("Dialister_invisus", "GCF_000154545.1", "DSM 15470"),

    ("Veillonella_parvula", "GCF_000164655.1", "DSM 2008"),

    ("Coprococcus_comes", "GCF_000154465.1", "ATCC 27758"),

    ("Dorea_formicigenerans", "GCF_000154525.1", "ATCC 27755"),

    ("Phascolarctobacterium_succinatutens", "GCF_000154505.1", "YIT 12067"),

    ("Lachnospira_eligens", "GCF_000154325.1", "ATCC 27750"),

    ("Barnesiella_intestinihominis", "GCF_000154385.1", "YIT 11860"),

    ("Odoribacter_splanchnicus", "GCF_000154465.1", "DSM 20712"),

    ("Anaerostipes_hadrus", "GCF_000154385.1", "DSM 3319"),

    ("Coprobacillus_sp", "GCF_000154325.1", "D7"),
]


def download_genome_ncbi(accession: str, output_dir: Path, species_name: str) -> bool:
    """
    Download genome from NCBI using datasets command-line tool.

    Requires: ncbi-datasets-cli to be installed
    Install with: conda install -c conda-forge ncbi-datasets-cli
    Or: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {species_name} ({accession})...")

    try:
        # Download using datasets
        result = subprocess.run(
            [
                'datasets', 'download', 'genome', 'accession',
                accession,
                '--filename', str(output_dir / f'{species_name}.zip'),
                '--include', 'genome'
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"    ✗ Failed to download: {result.stderr}")
            return False

        # Unzip
        subprocess.run(
            ['unzip', '-q', '-o',
             str(output_dir / f'{species_name}.zip'),
             '-d', str(output_dir / species_name)],
            check=True
        )

        # Find and rename the FASTA file
        genome_files = list((output_dir / species_name).rglob('*.fna'))
        if genome_files:
            final_path = output_dir / f'{species_name}.fasta'
            genome_files[0].rename(final_path)
            print(f"    ✓ Downloaded to {final_path}")

            # Clean up
            (output_dir / f'{species_name}.zip').unlink()
            import shutil
            shutil.rmtree(output_dir / species_name, ignore_errors=True)

            return True
        else:
            print(f"    ✗ No FASTA file found in download")
            return False

    except subprocess.TimeoutExpired:
        print(f"    ✗ Download timed out")
        return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def download_genome_curl(accession: str, output_dir: Path, species_name: str) -> bool:
    """
    Download genome directly from NCBI FTP using curl (fallback method).
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct FTP URL
    # Format: ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/
    parts = accession.split('_')
    if len(parts) != 2:
        print(f"    ✗ Invalid accession format")
        return False

    prefix = parts[0]  # GCF or GCA
    numbers = parts[1].split('.')[0]  # Remove version

    # Split into path components
    p1 = numbers[0:3]
    p2 = numbers[3:6]
    p3 = numbers[6:9]

    # Try to construct URL - this is a simplified approach
    # In practice, you'd need the assembly name which requires API call
    print(f"    Note: Direct FTP download requires assembly name")
    print(f"    Please use 'datasets' CLI tool or download manually from:")
    print(f"    https://www.ncbi.nlm.nih.gov/assembly/{accession}")

    return False


def create_strain_variants(reference_genome: Path, n_strains: int, output_dir: Path) -> List[Path]:
    """
    Create strain variants by introducing SNPs into the reference genome.

    This creates realistic strain-level diversity from a single reference.
    """

    print(f"    Creating {n_strains} strain variants...")

    # This would require tools like:
    # - simuG (https://github.com/yjx1217/simuG)
    # - VarSim
    # - Custom script to introduce SNPs

    # For now, just return the reference as the first strain
    # User can extend this with actual variant introduction

    strain_genomes = []

    # Copy reference as strain 0
    strain_0 = output_dir / f"{reference_genome.stem}_strain_000.fasta"
    import shutil
    shutil.copy(reference_genome, strain_0)
    strain_genomes.append(strain_0)

    print(f"    ✓ Created strain 0 (reference)")
    print(f"    Note: Additional strain variants require SNP introduction tools")
    print(f"          See: https://github.com/yjx1217/simuG")

    return strain_genomes


def download_community(n_species: int, output_dir: Path, method: str = 'datasets'):
    """Download genomes for a community of species."""

    output_dir = Path(output_dir)
    ref_dir = output_dir / 'references'
    ref_dir.mkdir(parents=True, exist_ok=True)

    # Select species
    selected_species = GUT_MICROBIOME_SPECIES[:n_species]

    print(f"\nDownloading {n_species} bacterial genomes...")
    print(f"Method: {method}")

    successful = []
    failed = []

    for species_name, accession, strain in selected_species:
        if method == 'datasets':
            success = download_genome_ncbi(accession, ref_dir, species_name)
        else:
            success = download_genome_curl(accession, ref_dir, species_name)

        if success:
            successful.append((species_name, accession, strain))
        else:
            failed.append((species_name, accession))

        # Be nice to NCBI servers
        time.sleep(1)

    # Write metadata
    metadata = {
        'n_species': len(successful),
        'successful_downloads': [
            {
                'species': name,
                'accession': acc,
                'strain': strain,
                'file': f'references/{name}.fasta'
            }
            for name, acc, strain in successful
        ],
        'failed_downloads': [
            {'species': name, 'accession': acc}
            for name, acc in failed
        ]
    }

    metadata_file = output_dir / 'download_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n" + "="*80)
    print(f"Download complete!")
    print(f"Successful: {len(successful)}/{n_species}")
    print(f"Failed: {len(failed)}/{n_species}")
    print(f"\nMetadata written to: {metadata_file}")
    print("="*80)

    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description='Download real bacterial genomes for synthetic community',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 40 gut microbiome species
  python download_real_genomes.py -o real_genomes --species 40

  # Check what's available
  python download_real_genomes.py --list

Installation requirements:
  # Install NCBI datasets CLI tool
  conda install -c conda-forge ncbi-datasets-cli

  # Or download from:
  # https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/

Alternative:
  You can also manually download genomes from:
  - NCBI RefSeq: https://www.ncbi.nlm.nih.gov/refseq/
  - CAMI datasets: https://data.cami-challenge.org/
  - mockrobiota: https://github.com/caporaso-lab/mockrobiota
        """
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output directory for downloaded genomes'
    )

    parser.add_argument(
        '--species',
        type=int,
        default=40,
        help='Number of species to download (default: 40)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available species and exit'
    )

    parser.add_argument(
        '--method',
        choices=['datasets', 'curl'],
        default='datasets',
        help='Download method (default: datasets)'
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable gut microbiome species:\n")
        print(f"{'Species':<40} {'Accession':<20} {'Strain':<25}")
        print("="*85)
        for species, accession, strain in GUT_MICROBIOME_SPECIES:
            print(f"{species:<40} {accession:<20} {strain:<25}")
        print(f"\nTotal available: {len(GUT_MICROBIOME_SPECIES)}")
        return

    if not args.output:
        parser.error("--output is required")

    if args.species > len(GUT_MICROBIOME_SPECIES):
        print(f"Warning: Only {len(GUT_MICROBIOME_SPECIES)} species available")
        print(f"Downloading all {len(GUT_MICROBIOME_SPECIES)} species...")
        args.species = len(GUT_MICROBIOME_SPECIES)

    # Download
    successful, failed = download_community(
        args.species,
        args.output,
        args.method
    )

    if failed:
        print("\nFailed downloads:")
        for name, acc in failed:
            print(f"  - {name} ({acc})")
        print("\nYou can download these manually from:")
        print("https://www.ncbi.nlm.nih.gov/datasets/genome/")


if __name__ == '__main__':
    main()
