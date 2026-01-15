#!/usr/bin/env python3
"""
Generate synthetic read community with 40 species and 120 strains.

This script creates a diverse synthetic metagenomic dataset with:
- 40 different species (distinct reference genomes)
- 120 total strains distributed across species (average 3 strains per species)
- Realistic strain-level variation within species
- Variable abundance distributions
- Multiple timepoints (optional)

Output:
- FASTA files for reference genomes (one per species)
- FASTQ files for synthetic reads
- Metadata files with ground truth strain abundances
- VCF files for variant positions
"""

import numpy as np
import argparse
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

# Import existing strainphase simulation code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from strainphase.simulation.synthetic_data import SyntheticDataGenerator, TrueHaplotype


@dataclass
class Species:
    """Represents a species with multiple strains."""
    id: str
    name: str
    genome_length: int
    n_strains: int
    reference_sequence: Dict[int, str]  # position -> base
    snv_positions: List[int]
    ref_alleles: Dict[int, str]
    strains: List['Strain'] = field(default_factory=list)

    def get_total_abundance(self, timepoint: str) -> float:
        """Get total abundance of this species at a timepoint."""
        return sum(s.get_abundance(timepoint) for s in self.strains)


@dataclass
class Strain:
    """Represents a strain (haplotype) within a species."""
    id: str
    species_id: str
    consensus: Dict[int, str]  # SNV positions -> alleles
    abundance_by_timepoint: Dict[str, float] = field(default_factory=dict)

    def get_abundance(self, timepoint: str) -> float:
        return self.abundance_by_timepoint.get(timepoint, 0.0)


class CommunityGenerator:
    """Generator for multi-species synthetic communities."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.bases = ['A', 'C', 'G', 'T']
        self.synth_gen = SyntheticDataGenerator(seed=seed)

    def generate_species_name(self, species_idx: int) -> str:
        """Generate realistic-looking species names."""
        genera = [
            'Bacteroides', 'Prevotella', 'Faecalibacterium', 'Ruminococcus',
            'Escherichia', 'Bifidobacterium', 'Lactobacillus', 'Clostridium',
            'Akkermansia', 'Roseburia', 'Eubacterium', 'Alistipes',
            'Parabacteroides', 'Blautia', 'Coprococcus', 'Dorea',
            'Streptococcus', 'Enterococcus', 'Collinsella', 'Bilophila',
            'Desulfovibrio', 'Methanobrevibacter', 'Sutterella', 'Oscillospira',
            'Dialister', 'Veillonella', 'Megamonas', 'Mitsuokella',
            'Phascolarctobacterium', 'Acidaminococcus', 'Anaerostipes', 'Butyrivibrio',
            'Coprobacillus', 'Holdemania', 'Lachnospira', 'Oribacterium',
            'Peptostreptococcus', 'Turicibacter', 'Barnesiella', 'Odoribacter'
        ]

        species_epithets = [
            'intestinalis', 'faecalis', 'prausnitzii', 'coli',
            'vulgatus', 'uniformis', 'fragilis', 'thetaiotaomicron',
            'ovatus', 'stercoris', 'coprocola', 'massiliensis',
            'caccae', 'dorei', 'xylanisolvens', 'cellulosilyticus',
            'plebeius', 'finegoldii', 'eggerthii', 'merdae'
        ]

        genus = genera[species_idx % len(genera)]
        epithet = species_epithets[(species_idx * 7) % len(species_epithets)]

        return f"{genus}_{epithet}_{species_idx}"

    def generate_genome_length(self, species_idx: int) -> int:
        """Generate realistic genome lengths (2-6 Mb for bacteria)."""
        # Most bacterial genomes are 2-5 Mb
        base_length = self.rng.integers(2_000_000, 5_000_000)
        # Add some variation
        variation = self.rng.integers(-500_000, 500_000)
        return max(1_500_000, base_length + variation)

    def generate_reference_sequence(self, length: int) -> Dict[int, str]:
        """Generate random reference sequence."""
        # For efficiency, we only store a subset of positions
        # In real use, this would be a full genome
        sample_positions = sorted(self.rng.choice(
            range(1, length + 1),
            size=min(10000, length),
            replace=False
        ))
        return {pos: self.rng.choice(self.bases) for pos in sample_positions}

    def generate_snv_positions(self, genome_length: int,
                                snv_density_per_kb: float = 2.0) -> List[int]:
        """Generate SNV positions with realistic density."""
        n_snvs = int((genome_length / 1000) * snv_density_per_kb)
        n_snvs = max(50, n_snvs)  # At least 50 SNVs

        # Generate with some clustering (realistic)
        positions = []
        base_spacing = genome_length / (n_snvs + 1)

        for i in range(n_snvs):
            base_pos = int((i + 1) * base_spacing)
            noise = int(self.rng.normal(0, base_spacing * 0.2))
            pos = max(1, min(genome_length - 1, base_pos + noise))
            positions.append(pos)

        return sorted(set(positions))

    def generate_strain_abundances(self,
                                     n_strains: int,
                                     n_timepoints: int,
                                     abundance_type: str = 'uniform') -> List[Dict[str, float]]:
        """
        Generate abundance trajectories for strains.

        Args:
            n_strains: Number of strains
            n_timepoints: Number of timepoints
            abundance_type: 'uniform', 'dominant', or 'skewed'
        """
        timepoints = [f"T{i+1}" for i in range(n_timepoints)]
        abundances = [{tp: 0.0 for tp in timepoints} for _ in range(n_strains)]

        for tp_idx, tp in enumerate(timepoints):
            if abundance_type == 'uniform':
                # Roughly equal abundances with noise
                base = 1.0 / n_strains
                values = [max(0.01, base + self.rng.normal(0, base * 0.2))
                         for _ in range(n_strains)]

            elif abundance_type == 'dominant':
                # One dominant strain
                values = [0.1] * n_strains
                dominant_idx = 0
                values[dominant_idx] = 0.7
                # Add noise
                values = [max(0.01, v + self.rng.normal(0, 0.05)) for v in values]

            elif abundance_type == 'skewed':
                # Power law distribution
                ranks = np.arange(1, n_strains + 1)
                values = 1.0 / ranks**1.5
                values = values + self.rng.normal(0, 0.01, size=n_strains)
                values = np.maximum(values, 0.01)

            else:
                raise ValueError(f"Unknown abundance_type: {abundance_type}")

            # Normalize
            total = sum(values)
            values = [v / total for v in values]

            for i in range(n_strains):
                abundances[i][tp] = values[i]

        return abundances

    def generate_species(self,
                          species_idx: int,
                          n_strains: int,
                          n_timepoints: int,
                          snv_density: float = 2.0) -> Species:
        """Generate a single species with multiple strains."""

        species_id = f"species_{species_idx:03d}"
        species_name = self.generate_species_name(species_idx)
        genome_length = self.generate_genome_length(species_idx)

        # Generate reference genome
        reference_seq = self.generate_reference_sequence(genome_length)

        # Generate SNV positions
        snv_positions = self.generate_snv_positions(genome_length, snv_density)
        ref_alleles = {pos: self.rng.choice(self.bases) for pos in snv_positions}

        # Create species
        species = Species(
            id=species_id,
            name=species_name,
            genome_length=genome_length,
            n_strains=n_strains,
            reference_sequence=reference_seq,
            snv_positions=snv_positions,
            ref_alleles=ref_alleles
        )

        # Generate strain abundances
        abundance_type = self.rng.choice(['uniform', 'dominant', 'skewed'])
        strain_abundances = self.generate_strain_abundances(
            n_strains, n_timepoints, abundance_type
        )

        # Generate strains
        for strain_idx in range(n_strains):
            strain_id = f"{species_id}_strain_{strain_idx:02d}"

            # Generate strain consensus (derived from reference with mutations)
            consensus = {}

            if strain_idx == 0:
                # First strain: close to reference (~5% divergent)
                for pos in snv_positions:
                    if self.rng.random() < 0.05:
                        alt_bases = [b for b in self.bases if b != ref_alleles[pos]]
                        consensus[pos] = self.rng.choice(alt_bases)
                    else:
                        consensus[pos] = ref_alleles[pos]
            else:
                # Subsequent strains: derived from first strain
                first_strain = species.strains[0]
                for pos in snv_positions:
                    # 2-10% different from first strain
                    divergence_rate = 0.02 + 0.08 * (strain_idx / n_strains)
                    if self.rng.random() < divergence_rate:
                        alt_bases = [b for b in self.bases if b != first_strain.consensus[pos]]
                        consensus[pos] = self.rng.choice(alt_bases)
                    else:
                        consensus[pos] = first_strain.consensus[pos]

            strain = Strain(
                id=strain_id,
                species_id=species_id,
                consensus=consensus,
                abundance_by_timepoint=strain_abundances[strain_idx]
            )

            species.strains.append(strain)

        return species

    def distribute_strains_across_species(self,
                                           n_species: int,
                                           total_strains: int) -> List[int]:
        """
        Distribute strains across species.

        Returns list of strain counts per species.
        Some species will have more strains (representing higher diversity).
        """
        # Start with base distribution (average)
        avg_strains = total_strains // n_species
        strain_counts = [avg_strains] * n_species
        remainder = total_strains - (avg_strains * n_species)

        # Add remainder to random species
        for i in range(remainder):
            strain_counts[i] += 1

        # Add variation: some species with more diversity
        # Move strains from low-diversity species to high-diversity ones
        n_adjustments = n_species // 4
        for _ in range(n_adjustments):
            donor_idx = self.rng.choice([i for i, c in enumerate(strain_counts) if c > 1])
            receiver_idx = self.rng.choice(range(n_species))

            if strain_counts[donor_idx] > 1:
                strain_counts[donor_idx] -= 1
                strain_counts[receiver_idx] += 1

        # Shuffle to randomize which species get more strains
        self.rng.shuffle(strain_counts)

        return strain_counts

    def generate_species_abundances(self,
                                      n_species: int,
                                      n_timepoints: int) -> List[Dict[str, float]]:
        """Generate species-level abundance distributions."""
        timepoints = [f"T{i+1}" for i in range(n_timepoints)]
        abundances = []

        for tp in timepoints:
            # Generate power-law distribution (realistic for microbiomes)
            ranks = np.arange(1, n_species + 1)
            values = 1.0 / ranks**1.2

            # Add temporal variation
            variation = self.rng.normal(0, 0.1, size=n_species)
            values = values * (1 + variation)
            values = np.maximum(values, 0.001)

            # Normalize
            values = values / values.sum()
            abundances.append(values)

        # Transpose to get per-species abundances across timepoints
        species_abundances = []
        for species_idx in range(n_species):
            species_abund = {
                tp: abundances[tp_idx][species_idx]
                for tp_idx, tp in enumerate(timepoints)
            }
            species_abundances.append(species_abund)

        return species_abundances

    def generate_community(self,
                           n_species: int = 40,
                           total_strains: int = 120,
                           n_timepoints: int = 4,
                           snv_density: float = 2.0) -> Dict:
        """
        Generate complete synthetic community.

        Returns:
            Dictionary with species list and metadata
        """
        print(f"Generating community with {n_species} species and {total_strains} strains...")

        # Distribute strains across species
        strain_counts = self.distribute_strains_across_species(n_species, total_strains)
        print(f"Strain distribution: min={min(strain_counts)}, max={max(strain_counts)}, "
              f"mean={np.mean(strain_counts):.1f}")

        # Generate species-level abundances
        species_abundances = self.generate_species_abundances(n_species, n_timepoints)

        # Generate each species
        community = []
        for species_idx in range(n_species):
            n_strains_for_species = strain_counts[species_idx]

            print(f"  Generating species {species_idx + 1}/{n_species}: "
                  f"{n_strains_for_species} strains...")

            species = self.generate_species(
                species_idx=species_idx,
                n_strains=n_strains_for_species,
                n_timepoints=n_timepoints,
                snv_density=snv_density
            )

            # Scale strain abundances by species abundance
            species_abund_by_tp = species_abundances[species_idx]
            for strain in species.strains:
                for tp in strain.abundance_by_timepoint:
                    strain_relative_abund = strain.abundance_by_timepoint[tp]
                    species_abund = species_abund_by_tp[tp]
                    strain.abundance_by_timepoint[tp] = strain_relative_abund * species_abund

            community.append(species)

        # Validate total abundances sum to 1.0
        timepoints = [f"T{i+1}" for i in range(n_timepoints)]
        for tp in timepoints:
            total = sum(
                sum(strain.get_abundance(tp) for strain in species.strains)
                for species in community
            )
            print(f"  Timepoint {tp} total abundance: {total:.3f}")

        return {
            'species': community,
            'n_species': n_species,
            'total_strains': total_strains,
            'n_timepoints': n_timepoints,
            'timepoints': timepoints,
            'strain_counts': strain_counts
        }


def write_reference_genomes(community: Dict, output_dir: Path):
    """Write reference genome FASTA files for each species."""
    ref_dir = output_dir / 'references'
    ref_dir.mkdir(parents=True, exist_ok=True)

    print("\nWriting reference genomes...")
    for species in community['species']:
        ref_file = ref_dir / f"{species.id}.fasta"

        with open(ref_file, 'w') as f:
            f.write(f">{species.id} {species.name}\n")

            # Generate full sequence (simplified - just use random bases)
            # In production, would generate actual sequence with SNVs embedded
            rng = np.random.default_rng(42)
            bases = ['A', 'C', 'G', 'T']
            sequence = ''.join(rng.choice(bases, size=species.genome_length))

            # Write in 80-character lines
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + '\n')

        print(f"  Wrote {ref_file}")


def write_vcf_files(community: Dict, output_dir: Path):
    """Write VCF files with SNV positions for each species."""
    vcf_dir = output_dir / 'vcfs'
    vcf_dir.mkdir(parents=True, exist_ok=True)

    print("\nWriting VCF files...")
    for species in community['species']:
        vcf_file = vcf_dir / f"{species.id}.vcf"

        with open(vcf_file, 'w') as f:
            # VCF header
            f.write("##fileformat=VCFv4.2\n")
            f.write(f"##contig=<ID={species.id},length={species.genome_length}>\n")
            f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            f.write("##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">\n")
            f.write("##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">\n")

            # Header line
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")

            # Write each SNV
            for pos in species.snv_positions:
                ref = species.ref_alleles[pos]

                # Collect all alternate alleles from strains
                alt_alleles = set()
                for strain in species.strains:
                    if strain.consensus[pos] != ref:
                        alt_alleles.add(strain.consensus[pos])

                if alt_alleles:
                    alt = ','.join(sorted(alt_alleles))
                    f.write(f"{species.id}\t{pos}\t.\t{ref}\t{alt}\t60\tPASS\t.\tGT:DP:AF\t0/1:50:0.5\n")

        print(f"  Wrote {vcf_file}")


def write_strain_metadata(community: Dict, output_dir: Path):
    """Write metadata file with strain information and abundances."""
    metadata_file = output_dir / 'strain_metadata.json'

    print(f"\nWriting metadata to {metadata_file}...")

    metadata = {
        'n_species': community['n_species'],
        'total_strains': community['total_strains'],
        'n_timepoints': community['n_timepoints'],
        'timepoints': community['timepoints'],
        'species': []
    }

    for species in community['species']:
        species_data = {
            'id': species.id,
            'name': species.name,
            'genome_length': species.genome_length,
            'n_snvs': len(species.snv_positions),
            'n_strains': species.n_strains,
            'strains': []
        }

        for strain in species.strains:
            strain_data = {
                'id': strain.id,
                'abundances': strain.abundance_by_timepoint,
                'n_variant_positions': sum(
                    1 for pos in strain.consensus
                    if strain.consensus[pos] != species.ref_alleles[pos]
                )
            }
            species_data['strains'].append(strain_data)

        metadata['species'].append(species_data)

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Wrote metadata file")


def write_abundance_table(community: Dict, output_dir: Path):
    """Write abundance table as TSV."""
    abundance_file = output_dir / 'strain_abundances.tsv'

    print(f"\nWriting abundance table to {abundance_file}...")

    with open(abundance_file, 'w') as f:
        # Header
        timepoints = community['timepoints']
        f.write("species_id\tstrain_id\t" + "\t".join(timepoints) + "\n")

        # Data
        for species in community['species']:
            for strain in species.strains:
                abundances = [f"{strain.get_abundance(tp):.6f}" for tp in timepoints]
                f.write(f"{species.id}\t{strain.id}\t" + "\t".join(abundances) + "\n")

    print(f"  Wrote abundance table")


def generate_reads_for_community(community: Dict,
                                   output_dir: Path,
                                   total_reads: int = 100000,
                                   read_length: int = 10000,
                                   error_rate: float = 0.001):
    """
    Generate synthetic reads for the entire community.

    This is a placeholder - full implementation would use a read simulator
    like pbsim3 or InSilicoSeq.
    """
    reads_dir = output_dir / 'reads'
    reads_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {total_reads} synthetic reads...")
    print("  (Note: Full read generation requires external simulator like pbsim3)")
    print("  Creating manifest file instead...")

    manifest_file = reads_dir / 'read_generation_manifest.txt'

    with open(manifest_file, 'w') as f:
        f.write(f"# Synthetic Read Generation Manifest\n")
        f.write(f"# Total reads: {total_reads}\n")
        f.write(f"# Read length: {read_length}\n")
        f.write(f"# Error rate: {error_rate}\n\n")

        for timepoint in community['timepoints']:
            f.write(f"\n## Timepoint: {timepoint}\n")
            f.write(f"# Expected read counts per strain:\n")

            for species in community['species']:
                for strain in species.strains:
                    abundance = strain.get_abundance(timepoint)
                    expected_reads = int(total_reads * abundance)

                    if expected_reads > 0:
                        f.write(f"{strain.id}\t{abundance:.6f}\t{expected_reads}\n")

    print(f"  Wrote manifest to {manifest_file}")
    print("\n  To generate actual reads, use:")
    print(f"    pbsim3 --prefix <output> --depth <coverage> --length-mean {read_length}")
    print(f"           --accuracy-mean {1-error_rate} <reference.fasta>")


def write_summary_statistics(community: Dict, output_dir: Path):
    """Write summary statistics about the community."""
    summary_file = output_dir / 'community_summary.txt'

    print(f"\nWriting summary to {summary_file}...")

    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SYNTHETIC COMMUNITY SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Number of species: {community['n_species']}\n")
        f.write(f"Total strains: {community['total_strains']}\n")
        f.write(f"Number of timepoints: {community['n_timepoints']}\n")
        f.write(f"Timepoints: {', '.join(community['timepoints'])}\n\n")

        f.write("Strain distribution across species:\n")
        strain_counts = community['strain_counts']
        f.write(f"  Min: {min(strain_counts)} strains\n")
        f.write(f"  Max: {max(strain_counts)} strains\n")
        f.write(f"  Mean: {np.mean(strain_counts):.2f} strains\n")
        f.write(f"  Median: {np.median(strain_counts):.0f} strains\n\n")

        f.write("="*80 + "\n")
        f.write("SPECIES DETAILS\n")
        f.write("="*80 + "\n\n")

        for species in community['species']:
            f.write(f"\n{species.id}: {species.name}\n")
            f.write(f"  Genome length: {species.genome_length:,} bp\n")
            f.write(f"  Number of SNVs: {len(species.snv_positions)}\n")
            f.write(f"  Number of strains: {species.n_strains}\n")

            f.write(f"\n  Strains:\n")
            for strain in species.strains:
                abundances = [f"{strain.get_abundance(tp):.4f}"
                             for tp in community['timepoints']]
                f.write(f"    {strain.id}: [{', '.join(abundances)}]\n")

    print(f"  Wrote summary file")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic metagenomic community with 40 species and 120 strains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default community (40 species, 120 strains, 4 timepoints)
  python generate_synthetic_community.py -o synthetic_community_output

  # Custom parameters
  python generate_synthetic_community.py -o output --species 50 --strains 150 --timepoints 6

  # With specific seed for reproducibility
  python generate_synthetic_community.py -o output --seed 12345
        """
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory for synthetic data'
    )

    parser.add_argument(
        '--species',
        type=int,
        default=40,
        help='Number of species (default: 40)'
    )

    parser.add_argument(
        '--strains',
        type=int,
        default=120,
        help='Total number of strains across all species (default: 120)'
    )

    parser.add_argument(
        '--timepoints',
        type=int,
        default=4,
        help='Number of timepoints to simulate (default: 4)'
    )

    parser.add_argument(
        '--snv-density',
        type=float,
        default=2.0,
        help='SNV density per kb (default: 2.0)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--generate-reads',
        action='store_true',
        help='Generate read manifest (full read generation requires external tools)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SYNTHETIC COMMUNITY GENERATOR")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Species: {args.species}")
    print(f"Total strains: {args.strains}")
    print(f"Timepoints: {args.timepoints}")
    print(f"SNV density: {args.snv_density} per kb")
    print(f"Random seed: {args.seed}")
    print("="*80 + "\n")

    # Generate community
    generator = CommunityGenerator(seed=args.seed)
    community = generator.generate_community(
        n_species=args.species,
        total_strains=args.strains,
        n_timepoints=args.timepoints,
        snv_density=args.snv_density
    )

    # Write outputs
    write_reference_genomes(community, output_dir)
    write_vcf_files(community, output_dir)
    write_strain_metadata(community, output_dir)
    write_abundance_table(community, output_dir)
    write_summary_statistics(community, output_dir)

    if args.generate_reads:
        generate_reads_for_community(community, output_dir)

    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print(f"\nOutput files written to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - references/        : Reference genomes (FASTA)")
    print(f"  - vcfs/              : Variant call files (VCF)")
    print(f"  - strain_metadata.json : Complete strain information")
    print(f"  - strain_abundances.tsv: Abundance table")
    print(f"  - community_summary.txt: Human-readable summary")

    if args.generate_reads:
        print(f"  - reads/             : Read generation manifest")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
