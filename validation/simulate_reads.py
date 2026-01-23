#!/usr/bin/env python3
"""
Simulate HiFi reads from user-provided bacterial genomes.

Creates synthetic metagenomic data with known ground truth for validating
strainphase haplotype reconstruction.

Usage:
    python validation/simulate_reads.py \
        --genomes /path/to/strain_fastas/ \
        --output data/simulated/ \
        --complexity medium \
        --timepoints 4 \
        --coverage 30
"""

import argparse
import logging
import os
import random
import gzip
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Strain:
    """Represents a bacterial strain."""
    id: str
    genome_file: str
    contigs: Dict[str, str] = field(default_factory=dict)  # contig_id -> sequence
    total_length: int = 0
    snvs: Dict[str, Dict[int, str]] = field(default_factory=dict)  # contig -> {pos -> alt_base}
    is_sweeping: bool = False  # Dynamic abundance pattern


@dataclass
class SimulationConfig:
    """Configuration for read simulation."""
    # Read parameters
    mean_read_length: int = 15000
    read_length_std: int = 3000
    min_read_length: int = 5000
    max_read_length: int = 25000
    error_rate: float = 0.001  # 0.1% HiFi error rate

    # SNV parameters
    snv_density: int = 10  # SNVs per 10kb

    # Coverage and timepoints
    coverage: int = 30
    n_timepoints: int = 4

    # Abundance dynamics
    sweep_fraction: float = 0.3  # Fraction of strains with sweeping dynamics

    # Random seed
    seed: int = 42


@dataclass
class GroundTruth:
    """Ground truth data for validation."""
    strains: List[Strain]
    abundances: Dict[str, Dict[str, float]]  # timepoint -> {strain_id -> abundance}
    snv_positions: Dict[str, List[int]]  # contig -> [positions]
    read_origins: List[Dict]  # List of {read_id, strain_id, contig, start, end}


# =============================================================================
# Genome loading
# =============================================================================

def load_genomes(genome_dir: str) -> List[Strain]:
    """Load all FASTA files from directory as strains."""
    genome_path = Path(genome_dir)
    strains = []

    fasta_files = list(genome_path.glob("*.fa")) + \
                  list(genome_path.glob("*.fasta")) + \
                  list(genome_path.glob("*.fna"))

    if not fasta_files:
        raise ValueError(f"No FASTA files found in {genome_dir}")

    logger.info(f"Found {len(fasta_files)} genome files")

    for fasta_file in sorted(fasta_files):
        strain_id = fasta_file.stem
        strain = Strain(id=strain_id, genome_file=str(fasta_file))

        # Parse FASTA
        current_contig = None
        current_seq = []

        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_contig:
                        strain.contigs[current_contig] = ''.join(current_seq)
                    current_contig = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line.upper())

            if current_contig:
                strain.contigs[current_contig] = ''.join(current_seq)

        strain.total_length = sum(len(seq) for seq in strain.contigs.values())
        strains.append(strain)
        logger.debug(f"Loaded {strain_id}: {len(strain.contigs)} contigs, {strain.total_length:,} bp")

    return strains


# =============================================================================
# SNV introduction
# =============================================================================

def introduce_snvs(strains: List[Strain], config: SimulationConfig, rng: np.random.Generator) -> Dict[str, List[int]]:
    """
    Introduce SNVs to create strain diversity.

    Returns dict of contig -> [snv_positions] for ground truth.
    """
    bases = ['A', 'C', 'G', 'T']
    all_snv_positions = defaultdict(set)

    # First strain is reference (no SNVs)
    reference = strains[0]
    logger.info(f"Using {reference.id} as reference strain")

    # Assign sweeping vs fixed pattern to each strain
    n_sweeping = int(len(strains) * config.sweep_fraction)
    sweeping_indices = set(rng.choice(range(1, len(strains)), size=min(n_sweeping, len(strains)-1), replace=False))

    for i, strain in enumerate(strains[1:], 1):
        strain.is_sweeping = i in sweeping_indices

        # Determine SNV density for this strain
        if strain.is_sweeping:
            # Sweeping strains: moderate SNV count
            density = config.snv_density * rng.uniform(0.5, 1.5)
        else:
            # Fixed strains: variable density
            density = config.snv_density * rng.uniform(0.2, 2.0)

        snvs_to_add = int(strain.total_length * density / 10000)
        logger.debug(f"{strain.id}: adding ~{snvs_to_add} SNVs (sweeping={strain.is_sweeping})")

        # Introduce SNVs based on reference contigs
        for contig_id, ref_seq in reference.contigs.items():
            if contig_id not in strain.contigs:
                # Copy reference contig if strain doesn't have it
                strain.contigs[contig_id] = ref_seq

            # Calculate SNVs for this contig proportional to length
            contig_snvs = int(snvs_to_add * len(ref_seq) / strain.total_length)

            # Select random positions (avoiding first/last 100bp)
            valid_positions = list(range(100, len(ref_seq) - 100))
            if len(valid_positions) < contig_snvs:
                contig_snvs = len(valid_positions)

            snv_positions = sorted(rng.choice(valid_positions, size=contig_snvs, replace=False))

            # Create SNVs
            strain.snvs[contig_id] = {}
            seq_list = list(strain.contigs[contig_id])

            for pos in snv_positions:
                ref_base = ref_seq[pos]
                alt_bases = [b for b in bases if b != ref_base]
                alt_base = rng.choice(alt_bases)

                strain.snvs[contig_id][pos] = alt_base
                seq_list[pos] = alt_base
                all_snv_positions[contig_id].add(pos)

            strain.contigs[contig_id] = ''.join(seq_list)

    # Convert to sorted lists
    return {contig: sorted(positions) for contig, positions in all_snv_positions.items()}


# =============================================================================
# Abundance profiles
# =============================================================================

def generate_abundances(strains: List[Strain], config: SimulationConfig, rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
    """
    Generate abundance profiles across timepoints.

    Returns: {timepoint -> {strain_id -> relative_abundance}}
    """
    n_strains = len(strains)
    timepoints = [f"T{i+1}" for i in range(config.n_timepoints)]
    abundances = {tp: {} for tp in timepoints}

    for strain in strains:
        if strain.is_sweeping:
            # Sweeping pattern: increase or decrease over time
            direction = rng.choice([-1, 1])
            start_abund = rng.uniform(0.05, 0.3)

            for i, tp in enumerate(timepoints):
                progress = i / (len(timepoints) - 1) if len(timepoints) > 1 else 0
                change = direction * progress * rng.uniform(0.2, 0.4)
                abund = max(0.01, min(0.5, start_abund + change))
                abundances[tp][strain.id] = abund
        else:
            # Fixed pattern: relatively stable
            base_abund = rng.uniform(0.02, 0.2)

            for tp in timepoints:
                noise = rng.normal(0, 0.02)
                abund = max(0.01, base_abund + noise)
                abundances[tp][strain.id] = abund

    # Normalize to sum to 1.0 for each timepoint
    for tp in timepoints:
        total = sum(abundances[tp].values())
        for strain_id in abundances[tp]:
            abundances[tp][strain_id] /= total

    return abundances


# =============================================================================
# Read simulation
# =============================================================================

def simulate_read(
    strain: Strain,
    contig_id: str,
    start: int,
    length: int,
    error_rate: float,
    rng: np.random.Generator
) -> Tuple[str, List[int], str]:
    """
    Simulate a single HiFi read with errors.

    Returns: (sequence, quality_scores, cigar)
    """
    seq = strain.contigs[contig_id]
    end = min(start + length, len(seq))
    read_seq = list(seq[start:end])
    quals = []

    bases = ['A', 'C', 'G', 'T']

    for i in range(len(read_seq)):
        if rng.random() < error_rate:
            # Introduce error (mostly substitutions for HiFi)
            if rng.random() < 0.9:  # 90% substitutions
                alt_bases = [b for b in bases if b != read_seq[i]]
                read_seq[i] = rng.choice(alt_bases)
                quals.append(15)  # Lower quality for errors
            else:  # 10% indels - skip for simplicity
                quals.append(30)
        else:
            quals.append(30 + rng.integers(-5, 6))  # Q25-35

    return ''.join(read_seq), quals, f"{len(read_seq)}M"


def simulate_reads_for_timepoint(
    strains: List[Strain],
    abundances: Dict[str, float],
    config: SimulationConfig,
    timepoint: str,
    rng: np.random.Generator
) -> Tuple[List[Dict], List[Dict]]:
    """
    Simulate all reads for one timepoint.

    Returns: (reads, read_origins)
    """
    reads = []
    read_origins = []

    # Calculate total bases needed for coverage
    total_genome_length = sum(s.total_length for s in strains)
    avg_genome_length = total_genome_length / len(strains)
    total_bases_needed = int(avg_genome_length * config.coverage)

    read_count = 0
    bases_generated = 0

    while bases_generated < total_bases_needed:
        # Select strain based on abundance
        strain_probs = [abundances.get(s.id, 0) for s in strains]
        strain_idx = rng.choice(len(strains), p=strain_probs)
        strain = strains[strain_idx]

        # Select contig weighted by length
        contig_ids = list(strain.contigs.keys())
        contig_lengths = [len(strain.contigs[c]) for c in contig_ids]
        contig_probs = np.array(contig_lengths) / sum(contig_lengths)
        contig_id = rng.choice(contig_ids, p=contig_probs)
        contig_seq = strain.contigs[contig_id]

        # Generate read length
        read_length = int(rng.normal(config.mean_read_length, config.read_length_std))
        read_length = max(config.min_read_length, min(config.max_read_length, read_length))

        # Select start position
        if len(contig_seq) <= read_length:
            start = 0
            read_length = len(contig_seq)
        else:
            start = rng.integers(0, len(contig_seq) - read_length)

        # Simulate read
        seq, quals, cigar = simulate_read(strain, contig_id, start, read_length, config.error_rate, rng)

        read_id = f"{timepoint}_read_{read_count:08d}"
        read_count += 1
        bases_generated += len(seq)

        reads.append({
            'id': read_id,
            'seq': seq,
            'quals': quals,
            'contig': contig_id,
            'start': start,
            'end': start + len(seq),
            'mapq': 60,
            'cigar': cigar,
        })

        read_origins.append({
            'read_id': read_id,
            'strain_id': strain.id,
            'contig': contig_id,
            'start': start,
            'end': start + len(seq),
        })

    logger.info(f"{timepoint}: Generated {read_count:,} reads ({bases_generated:,} bp)")
    return reads, read_origins


# =============================================================================
# Output generation
# =============================================================================

def write_fastq(reads: List[Dict], output_path: str):
    """Write reads to FASTQ file."""
    with open(output_path, 'w') as f:
        for read in reads:
            qual_str = ''.join(chr(q + 33) for q in read['quals'])
            f.write(f"@{read['id']}\n")
            f.write(f"{read['seq']}\n")
            f.write("+\n")
            f.write(f"{qual_str}\n")


def write_sam(reads: List[Dict], reference_contigs: Dict[str, str], output_path: str):
    """Write reads to SAM file."""
    with open(output_path, 'w') as f:
        # Header
        f.write("@HD\tVN:1.6\tSO:coordinate\n")
        for contig_id, seq in reference_contigs.items():
            f.write(f"@SQ\tSN:{contig_id}\tLN:{len(seq)}\n")
        f.write("@PG\tID:simulate_reads\tPN:strainphase_sim\tVN:1.0\n")

        # Reads
        for read in reads:
            flag = 0  # Forward strand, mapped
            f.write(f"{read['id']}\t{flag}\t{read['contig']}\t{read['start']+1}\t{read['mapq']}\t{read['cigar']}\t*\t0\t0\t{read['seq']}\t{''.join(chr(q+33) for q in read['quals'])}\n")


def write_vcf(snv_positions: Dict[str, List[int]], strains: List[Strain], reference: Strain, output_path: str):
    """Write ground truth VCF with all SNV positions in proper VCF format with sample column."""
    with open(output_path, 'w') as f:
        # Header
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=strainphase_simulate\n")
        for contig_id, seq in reference.contigs.items():
            f.write(f"##contig=<ID={contig_id},length={len(seq)}>\n")
        f.write('##INFO=<ID=STRAINS,Number=.,Type=String,Description="Strains with alt allele">\n')
        f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n')
        f.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n')
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n')
        f.write('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")

        # Variants
        for contig_id, positions in sorted(snv_positions.items()):
            ref_seq = reference.contigs.get(contig_id, "")

            for pos in positions:
                ref_base = ref_seq[pos] if pos < len(ref_seq) else 'N'

                # Find all alt alleles and which strains have them
                alt_alleles = {}
                for strain in strains[1:]:
                    if contig_id in strain.snvs and pos in strain.snvs[contig_id]:
                        alt = strain.snvs[contig_id][pos]
                        if alt not in alt_alleles:
                            alt_alleles[alt] = []
                        alt_alleles[alt].append(strain.id)

                if alt_alleles:
                    alt_str = ','.join(sorted(alt_alleles.keys()))
                    strain_info = ';'.join(f"{alt}:{','.join(sids)}" for alt, sids in alt_alleles.items())
                    # Add fake depth and AF for strainphase compatibility
                    info = f"DP=50;AF=0.5;STRAINS={strain_info}"
                    # Add sample genotype (0/1 = heterozygous indicating mixed population)
                    gt_format = "GT:DP:AD"
                    gt_sample = "0/1:50:25,25"
                    f.write(f"{contig_id}\t{pos+1}\t.\t{ref_base}\t{alt_str}\t30\tPASS\t{info}\t{gt_format}\t{gt_sample}\n")


def write_reference_fasta(reference: Strain, output_path: str):
    """Write reference genome FASTA."""
    with open(output_path, 'w') as f:
        for contig_id, seq in reference.contigs.items():
            f.write(f">{contig_id}\n")
            # Write in 80-char lines
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def _infer_species(strain_id: str) -> str:
    """Infer species name from strain identifier."""
    parts = strain_id.split("_")
    if "strain" in parts:
        idx = parts.index("strain")
        species = "_".join(parts[:idx]).strip()
        return species or strain_id
    return parts[0] if parts else strain_id


def write_ground_truth(ground_truth: GroundTruth, output_dir: str):
    """Write all ground truth files."""
    os.makedirs(output_dir, exist_ok=True)

    # Strain info
    with open(os.path.join(output_dir, "truth_strains.tsv"), 'w') as f:
        f.write("strain_id\tspecies\tgenome_file\ttotal_length\tsnv_count\tis_sweeping\n")
        for strain in ground_truth.strains:
            snv_count = sum(len(snvs) for snvs in strain.snvs.values())
            species = _infer_species(strain.id)
            f.write(f"{strain.id}\t{species}\t{strain.genome_file}\t{strain.total_length}\t{snv_count}\t{strain.is_sweeping}\n")

    # Abundances
    with open(os.path.join(output_dir, "truth_abundances.tsv"), 'w') as f:
        timepoints = sorted(ground_truth.abundances.keys())
        strain_ids = [s.id for s in ground_truth.strains]
        f.write("strain_id\t" + "\t".join(timepoints) + "\n")
        for strain_id in strain_ids:
            abunds = [f"{ground_truth.abundances[tp].get(strain_id, 0):.6f}" for tp in timepoints]
            f.write(f"{strain_id}\t" + "\t".join(abunds) + "\n")

    # Read origins
    with open(os.path.join(output_dir, "read_origins.tsv"), 'w') as f:
        f.write("read_id\tstrain_id\tcontig\tstart\tend\n")
        for origin in ground_truth.read_origins:
            f.write(f"{origin['read_id']}\t{origin['strain_id']}\t{origin['contig']}\t{origin['start']}\t{origin['end']}\n")

    # SNV positions
    with open(os.path.join(output_dir, "truth_snv_positions.tsv"), 'w') as f:
        f.write("contig\tposition\n")
        for contig, positions in sorted(ground_truth.snv_positions.items()):
            for pos in positions:
                f.write(f"{contig}\t{pos}\n")

    # Haplotype blocks (SNV alleles per strain/contig)
    haplotypes_path = os.path.join(output_dir, "truth_haplotypes.tsv")
    with open(haplotypes_path, 'w') as f:
        f.write("strain_id\tcontig\tsnv_alleles\n")
        for strain in ground_truth.strains:
            for contig, positions in sorted(ground_truth.snv_positions.items()):
                seq = strain.contigs.get(contig, "")
                alleles = []
                for pos in positions:
                    if pos < len(seq):
                        alleles.append(f"{pos}:{seq[pos]}")
                allele_str = ",".join(alleles) if alleles else "."
                f.write(f"{strain.id}\t{contig}\t{allele_str}\n")

    logger.info(f"Wrote ground truth files to {output_dir}")


# =============================================================================
# Main simulation pipeline
# =============================================================================

def run_simulation(
    genome_dir: str,
    output_dir: str,
    config: SimulationConfig,
    max_strains: Optional[int] = None
) -> GroundTruth:
    """Run the full simulation pipeline."""

    rng = np.random.default_rng(config.seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load genomes
    logger.info(f"Loading genomes from {genome_dir}")
    strains = load_genomes(genome_dir)
    logger.info(f"Loaded {len(strains)} strains")

    # Limit strains if requested
    if max_strains and len(strains) > max_strains:
        strains = strains[:max_strains]
        logger.info(f"Limited to {max_strains} strains")

    # Introduce SNVs
    logger.info("Introducing SNVs between strains")
    snv_positions = introduce_snvs(strains, config, rng)
    total_snvs = sum(len(pos) for pos in snv_positions.values())
    logger.info(f"Introduced {total_snvs:,} SNV positions")

    # Generate abundance profiles
    logger.info("Generating abundance profiles")
    abundances = generate_abundances(strains, config, rng)

    # Write reference genome (first strain)
    reference = strains[0]
    ref_path = os.path.join(output_dir, "reference.fasta")
    write_reference_fasta(reference, ref_path)
    logger.info(f"Wrote reference to {ref_path}")

    # Write ground truth VCF (primary + legacy name)
    vcf_path = os.path.join(output_dir, "truth_snvs.vcf")
    write_vcf(snv_positions, strains, reference, vcf_path)
    logger.info(f"Wrote truth VCF to {vcf_path}")

    legacy_vcf_path = os.path.join(output_dir, "truth_variants.vcf")
    write_vcf(snv_positions, strains, reference, legacy_vcf_path)
    logger.info(f"Wrote legacy truth VCF to {legacy_vcf_path}")

    # Simulate reads for each timepoint
    all_read_origins = []
    timepoints = [f"T{i+1}" for i in range(config.n_timepoints)]

    for tp in timepoints:
        logger.info(f"Simulating reads for {tp}")
        reads, origins = simulate_reads_for_timepoint(strains, abundances[tp], config, tp, rng)
        all_read_origins.extend(origins)

        # Write FASTQ
        fastq_path = os.path.join(output_dir, f"{tp}.fastq")
        write_fastq(reads, fastq_path)

        # Write SAM
        sam_path = os.path.join(output_dir, f"{tp}.sam")
        write_sam(reads, reference.contigs, sam_path)

        logger.info(f"Wrote {tp}.fastq and {tp}.sam")

    # Create ground truth object
    ground_truth = GroundTruth(
        strains=strains,
        abundances=abundances,
        snv_positions=snv_positions,
        read_origins=all_read_origins
    )

    # Write ground truth files
    write_ground_truth(ground_truth, output_dir)

    # Write config
    config_path = os.path.join(output_dir, "simulation_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            'mean_read_length': config.mean_read_length,
            'error_rate': config.error_rate,
            'snv_density': config.snv_density,
            'coverage': config.coverage,
            'n_timepoints': config.n_timepoints,
            'sweep_fraction': config.sweep_fraction,
            'seed': config.seed,
            'n_strains': len(strains),
            'n_snv_positions': total_snvs,
        }, f, indent=2)

    logger.info(f"Simulation complete! Output in {output_dir}")
    return ground_truth


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulate HiFi reads from bacterial genomes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--genomes", required=True, help="Directory with strain FASTA files")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--complexity", choices=["simple", "medium", "complex"], default="medium",
                        help="Community complexity (affects strain selection)")
    parser.add_argument("--snv-density", type=int, default=10, help="SNVs per 10kb to introduce")
    parser.add_argument("--error-rate", type=float, default=0.001, help="Sequencing error rate")
    parser.add_argument("--coverage", type=int, default=30, help="Read coverage per timepoint")
    parser.add_argument("--timepoints", type=int, default=4, help="Number of timepoints")
    parser.add_argument("--sweep-fraction", type=float, default=0.3, help="Fraction of sweeping strains")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-strains", type=int, default=None, help="Limit number of strains to use")

    args = parser.parse_args()

    # Create config
    config = SimulationConfig(
        snv_density=args.snv_density,
        error_rate=args.error_rate,
        coverage=args.coverage,
        n_timepoints=args.timepoints,
        sweep_fraction=args.sweep_fraction,
        seed=args.seed,
    )

    # Adjust max strains based on complexity
    complexity_limits = {"simple": 15, "medium": 50, "complex": None}
    max_strains = args.max_strains or complexity_limits.get(args.complexity)

    # Run simulation
    run_simulation(args.genomes, args.output, config, max_strains=max_strains)


if __name__ == "__main__":
    main()
