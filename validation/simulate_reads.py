#!/usr/bin/env python3
"""
Simulate HiFi reads from user-provided bacterial genomes.

Creates synthetic metagenomic data with known ground truth for validating
strainphase haplotype reconstruction.

Two modes:
1. Synthetic mode (default): Creates multiple strains from each genome file by
   introducing SNVs. Each genome file generates 2-max_strains synthetic strains.
   
2. Real strains mode (--use-real-strains): Uses FASTA files directly as distinct
   strains. SNVs are detected from real differences between strains.

Usage (synthetic mode):
    python validation/simulate_reads.py \
        --genomes /path/to/genome_fastas/ \
        --output data/simulated/ \
        --timepoints 4 \
        --coverage 30 \
        --max-strains 5

Usage (real strains mode):
    python validation/simulate_reads.py \
        --genomes /path/to/strain_fastas/ \
        --output data/simulated/ \
        --use-real-strains \
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
    snv_counts_per_strain: Optional[List[int]] = None  # Exact SNV counts for strains[1:]

    # Coverage and timepoints
    coverage: int = 30
    n_timepoints: int = 4

    # Abundance dynamics
    sweep_fraction: float = 0.3  # Fraction of strains with sweeping dynamics

    # Strain handling mode
    use_real_strains: bool = False  # If True, use FASTA files directly as strains (no SNV introduction)
    fixed_strains_per_genome: Optional[int] = None  # If set, use exact count per genome in synthetic mode
    # When use_real_strains=True:
    # - Each FASTA file is treated as a distinct strain
    # - SNVs are detected from real differences between strains
    # - No synthetic strain duplication or SNV introduction

    # VCF realism (optional)
    vcf_realism: bool = False  # If True, inject FP/FN sites and missing AF/DP
    vcf_fp_rate: float = 0.01  # False positive rate (fraction of sites to add)
    vcf_fn_rate: float = 0.01  # False negative rate (fraction of sites to drop)
    vcf_missing_af_rate: float = 0.05  # Fraction of sites with missing AF
    vcf_missing_dp_rate: float = 0.05  # Fraction of sites with missing DP

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
# SNV detection/introduction
# =============================================================================

def detect_snvs_from_real_strains(strains: List[Strain], reference: Strain) -> Dict[str, List[int]]:
    """
    Detect SNVs from real differences between strains and a reference.

    Compares each strain to the reference and identifies variant positions.
    Returns dict of contig -> [snv_positions] for ground truth.

    Uses numpy vectorization for performance (>100x faster than naive loop).
    """
    all_snv_positions = defaultdict(set)

    logger.info(f"Detecting SNVs from real strain differences (reference: {reference.id})")

    # Get all contigs present in any strain
    all_contigs = set(reference.contigs.keys())
    for strain in strains:
        all_contigs.update(strain.contigs.keys())

    # Count non-reference strains for progress
    non_ref_strains = [s for s in strains if s.id != reference.id]
    total_strains = len(non_ref_strains)
    logger.info(f"Comparing {total_strains} strains against reference across {len(all_contigs)} contigs")

    for strain_idx, strain in enumerate(non_ref_strains):
        logger.info(f"  Processing strain {strain_idx + 1}/{total_strains}: {strain.id}")

        # Initialize snvs dict for this strain if needed
        if not hasattr(strain, 'snvs') or strain.snvs is None:
            strain.snvs = {}

        strain_snv_count = 0

        for contig_idx, contig_id in enumerate(sorted(all_contigs)):
            ref_seq = reference.contigs.get(contig_id, "")
            strain_seq = strain.contigs.get(contig_id, "")

            if not ref_seq or not strain_seq:
                continue

            # Initialize contig dict if needed
            if contig_id not in strain.snvs:
                strain.snvs[contig_id] = {}

            # Use numpy for fast vectorized comparison
            min_len = min(len(ref_seq), len(strain_seq))

            # Convert sequences to numpy arrays of bytes
            ref_arr = np.frombuffer(ref_seq[:min_len].upper().encode('ascii'), dtype=np.uint8)
            strain_arr = np.frombuffer(strain_seq[:min_len].upper().encode('ascii'), dtype=np.uint8)

            # Find positions where bases differ
            diff_mask = ref_arr != strain_arr

            # Valid bases: A=65, C=67, G=71, T=84
            valid_bases = np.array([65, 67, 71, 84], dtype=np.uint8)
            ref_valid = np.isin(ref_arr, valid_bases)
            strain_valid = np.isin(strain_arr, valid_bases)

            # SNV positions: different AND both are valid ACGT bases
            snv_mask = diff_mask & ref_valid & strain_valid
            snv_positions = np.where(snv_mask)[0]

            # Record SNVs
            for pos in snv_positions:
                pos = int(pos)
                strain.snvs[contig_id][pos] = strain_seq[pos].upper()
                all_snv_positions[contig_id].add(pos)

            strain_snv_count += len(snv_positions)

            # Log progress for large contigs
            if len(ref_seq) > 1_000_000 and (contig_idx + 1) % 10 == 0:
                logger.debug(f"    Processed {contig_idx + 1}/{len(all_contigs)} contigs")

        logger.info(f"    Found {strain_snv_count:,} SNVs in {strain.id}")

    # Convert to sorted lists
    result = {contig: sorted(positions) for contig, positions in all_snv_positions.items()}
    total_snvs = sum(len(positions) for positions in result.values())
    logger.info(f"Detected {total_snvs:,} total SNV positions from real strain differences")

    return result


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

        if config.snv_counts_per_strain is not None:
            if len(config.snv_counts_per_strain) < len(strains) - 1:
                raise ValueError(
                    f"snv_counts_per_strain has {len(config.snv_counts_per_strain)} entries "
                    f"but {len(strains) - 1} strains require counts"
                )
            snvs_to_add = int(config.snv_counts_per_strain[i - 1])
        else:
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
        chosen_by_contig = {}
        total_added = 0
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
            chosen_by_contig[contig_id] = set(snv_positions)
            total_added += len(snv_positions)

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

        # If exact counts were requested and proportional allocation under-shot, top up.
        if config.snv_counts_per_strain is not None and total_added < snvs_to_add:
            remaining = snvs_to_add - total_added
            remaining_positions = []
            for contig_id, ref_seq in reference.contigs.items():
                valid_positions = range(100, len(ref_seq) - 100)
                already = chosen_by_contig.get(contig_id, set())
                for pos in valid_positions:
                    if pos not in already:
                        remaining_positions.append((contig_id, pos))
            if remaining_positions:
                pick_count = min(remaining, len(remaining_positions))
                extra = rng.choice(len(remaining_positions), size=pick_count, replace=False)
                for idx in extra:
                    contig_id, pos = remaining_positions[idx]
                    ref_seq = reference.contigs[contig_id]
                    ref_base = ref_seq[pos]
                    alt_bases = [b for b in bases if b != ref_base]
                    alt_base = rng.choice(alt_bases)
                    strain.snvs.setdefault(contig_id, {})
                    seq_list = list(strain.contigs[contig_id])
                    strain.snvs[contig_id][pos] = alt_base
                    seq_list[pos] = alt_base
                    strain.contigs[contig_id] = ''.join(seq_list)
                    all_snv_positions[contig_id].add(pos)

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


def write_vcf(
    snv_positions: Dict[str, List[int]], 
    strains: List[Strain], 
    reference: Strain, 
    output_path: str,
    config: Optional[SimulationConfig] = None,
    rng: Optional[np.random.Generator] = None
):
    """
    Write ground truth VCF with all SNV positions in proper VCF format with sample column.
    
    If config.vcf_realism is True, injects FP/FN sites and missing AF/DP fields.
    """
    perturbations = {
        'fp_sites': [],  # (contig, pos) added
        'fn_sites': [],  # (contig, pos) dropped
        'missing_af_sites': [],  # (contig, pos) with missing AF
        'missing_dp_sites': [],  # (contig, pos) with missing DP
    }
    
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
        all_positions = list(snv_positions.items())
        
        # Apply VCF realism if enabled
        if config and config.vcf_realism and rng:
            # Drop some sites (false negatives)
            n_fn = int(len(all_positions) * config.vcf_fn_rate)
            fn_indices = set(rng.choice(len(all_positions), size=n_fn, replace=False))
            filtered_positions = [(c, p) for i, (c, ps) in enumerate(all_positions) 
                                 for p in ps if i not in fn_indices]
            
            # Add false positive sites (random positions not in truth)
            n_fp = int(len(filtered_positions) * config.vcf_fp_rate)
            fp_sites = []
            for contig_id, ref_seq in reference.contigs.items():
                valid_positions = [p for p in range(100, len(ref_seq) - 100) 
                                 if (contig_id, p) not in filtered_positions]
                if valid_positions:
                    fp_positions = rng.choice(valid_positions, 
                                            size=min(n_fp, len(valid_positions)), 
                                            replace=False)
                    fp_sites.extend([(contig_id, int(p)) for p in fp_positions])
                    perturbations['fp_sites'] = fp_sites
            
            # Mark sites for missing AF/DP
            all_sites = filtered_positions + fp_sites
            n_missing_af = int(len(all_sites) * config.vcf_missing_af_rate)
            n_missing_dp = int(len(all_sites) * config.vcf_missing_dp_rate)
            missing_af_indices = set(rng.choice(len(all_sites), size=n_missing_af, replace=False))
            missing_dp_indices = set(rng.choice(len(all_sites), size=n_missing_dp, replace=False))
        else:
            filtered_positions = [(c, p) for c, ps in all_positions for p in ps]
            fp_sites = []
            missing_af_indices = set()
            missing_dp_indices = set()
        
        # Write variants
        all_sites_to_write = filtered_positions + fp_sites
        for site_idx, (contig_id, pos) in enumerate(all_sites_to_write):
            ref_seq = reference.contigs.get(contig_id, "")
            # pos is 0-indexed from snv_positions dict, convert to 1-indexed for VCF
            vcf_pos = pos + 1
            ref_base = ref_seq[pos] if pos < len(ref_seq) else 'N'
            
            # Check if this is a false positive (not in truth)
            is_fp = (contig_id, pos) in fp_sites
            
            if is_fp:
                # Generate random alt allele for FP
                bases = ['A', 'C', 'G', 'T']
                alt_base = rng.choice([b for b in bases if b != ref_base]) if rng else 'A'
                strain_info = f"{alt_base}:fake_strain"  # Fake strain assignment
            else:
                # Find all alt alleles and which strains have them
                alt_alleles = {}
                # Find all alt alleles and which strains have them
                # pos is 0-indexed from snv_positions dict
                for strain in strains:
                    if strain.id == reference.id:
                        continue  # Skip reference
                    if contig_id in strain.snvs and pos in strain.snvs[contig_id]:
                        alt = strain.snvs[contig_id][pos]
                        if alt not in alt_alleles:
                            alt_alleles[alt] = []
                        alt_alleles[alt].append(strain.id)
                
                if not alt_alleles:
                    continue  # Skip if no alt alleles (shouldn't happen for truth sites)
                
                alt_base = sorted(alt_alleles.keys())[0]
                strain_info = '|'.join(f"{alt}:{','.join(sids)}" for alt, sids in alt_alleles.items())
            
            # Build INFO field
            info_parts = []
            if site_idx not in missing_dp_indices:
                info_parts.append("DP=50")
            if site_idx not in missing_af_indices:
                info_parts.append("AF=0.5")
            info_parts.append(f"STRAINS={strain_info}")
            info = ';'.join(info_parts)
            
            # Build FORMAT/SAMPLE fields
            gt_format = "GT:DP:AD"
            if site_idx in missing_dp_indices:
                gt_sample = "0/1:.:25,25"  # Missing DP
            else:
                gt_sample = "0/1:50:25,25"
            
            f.write(f"{contig_id}\t{vcf_pos}\t.\t{ref_base}\t{alt_base}\t30\tPASS\t{info}\t{gt_format}\t{gt_sample}\n")
            
            # Track perturbations
            if is_fp:
                perturbations['fp_sites'].append((contig_id, pos))
            if site_idx in missing_af_indices:
                perturbations['missing_af_sites'].append((contig_id, pos))
            if site_idx in missing_dp_indices:
                perturbations['missing_dp_sites'].append((contig_id, pos))
        
        # Track false negatives
        if config and config.vcf_realism and rng:
            fn_sites = [(c, p) for i, (c, ps) in enumerate(all_positions) 
                       for p in ps if i in fn_indices]
            perturbations['fn_sites'] = fn_sites
    
    # Save perturbations JSON if realism enabled
    if config and config.vcf_realism:
        perturbations_path = output_path.replace('.vcf', '_perturbations.json').replace('truth_', 'truth_vcf_')
        import json
        import os
        # Convert tuples to lists for JSON serialization
        perturbations_json = {
            'fp_sites': [{'contig': c, 'pos': p} for c, p in perturbations['fp_sites']],
            'fn_sites': [{'contig': c, 'pos': p} for c, p in perturbations['fn_sites']],
            'missing_af_sites': [{'contig': c, 'pos': p} for c, p in perturbations['missing_af_sites']],
            'missing_dp_sites': [{'contig': c, 'pos': p} for c, p in perturbations['missing_dp_sites']],
        }
        # Write to same directory as VCF
        perturbations_path = os.path.join(os.path.dirname(output_path), 'truth_vcf_perturbations.json')
        with open(perturbations_path, 'w') as f:
            json.dump(perturbations_json, f, indent=2)
        logger.info(f"Wrote VCF perturbations to {perturbations_path}")


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

    # Track spans per strain (per contig) - for track/linking validation
    # Each strain should have one continuous track per contig (including reference strain)
    tracks_path = os.path.join(output_dir, "truth_tracks.tsv")
    with open(tracks_path, 'w') as f:
        f.write("strain_id\tcontig\tstart\tend\twindow_chain\n")
        for strain in ground_truth.strains:
            for contig, positions in sorted(ground_truth.snv_positions.items()):
                # Include all strains that have this contig (including reference strain)
                # Reference strain has reference alleles at all SNV positions, which is informative
                if contig in strain.contigs:
                    # Track spans the entire contig (or at least where SNVs exist)
                    contig_length = len(strain.contigs[contig])
                    # Window chain is just a placeholder - actual windows depend on window_size
                    # Format: "w0,w1,w2" where windows overlap by 50%
                    # For now, we'll mark it as "full" and validation will compute actual windows
                    f.write(f"{strain.id}\t{contig}\t1\t{contig_length}\tfull\n")
    
    logger.info(f"Wrote truth_tracks.tsv to {tracks_path}")

    # Lineage mapping (strain IDs ↔ cross-timepoint lineage IDs)
    # For simulation: each strain gets a unique lineage ID (since strains persist across timepoints)
    lineages_path = os.path.join(output_dir, "truth_lineages.tsv")
    with open(lineages_path, 'w') as f:
        f.write("strain_id\tlineage_id\tcontig\n")
        for strain in ground_truth.strains:
            # Create a unique lineage ID per strain (can be same as strain_id or derived)
            # For simulation, lineage_id = strain_id since each strain is a distinct lineage
            lineage_id = strain.id
            # Each strain appears in all contigs where it has SNVs
            for contig in sorted(set(ground_truth.snv_positions.keys())):
                if contig in strain.contigs:
                    f.write(f"{strain.id}\t{lineage_id}\t{contig}\n")
    
    logger.info(f"Wrote truth_lineages.tsv to {lineages_path}")

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
    genome_strains = load_genomes(genome_dir)
    logger.info(f"Loaded {len(genome_strains)} genome file(s)")

    if config.use_real_strains:
        # Mode: Use FASTA files directly as distinct strains
        logger.info("Using real strains mode: each FASTA file represents a distinct strain")
        
        if len(genome_strains) < 2:
            raise ValueError(f"Real strains mode requires at least 2 FASTA files (found {len(genome_strains)})")
        
        # Use loaded genomes directly as strains
        strains = genome_strains
        
        # First strain is the reference
        reference = strains[0]
        logger.info(f"Using {reference.id} as reference strain")
        
        # Detect SNVs from real differences
        logger.info("Detecting SNVs from real strain differences")
        snv_positions = detect_snvs_from_real_strains(strains, reference)
        total_snvs = sum(len(pos) for pos in snv_positions.values())
        logger.info(f"Detected {total_snvs:,} SNV positions from real differences")
        
        # Assign sweeping patterns randomly (for abundance dynamics)
        n_sweeping = int(len(strains) * config.sweep_fraction)
        sweeping_indices = set(rng.choice(range(1, len(strains)), size=min(n_sweeping, len(strains)-1), replace=False))
        for i, strain in enumerate(strains[1:], 1):
            strain.is_sweeping = i in sweeping_indices
        
    else:
        # Mode: Create synthetic strains from genomes (original behavior)
        # CRITICAL RULE: For EVERY SINGLE genome file provided, randomly create between 2 and max_strains strains
        # This ensures SNVs can be introduced (introduce_snvs only processes strains[1:], so we need >= 2 strains)
        if config.fixed_strains_per_genome is not None:
            if config.fixed_strains_per_genome < 2:
                raise ValueError("fixed_strains_per_genome must be >= 2")
            min_strains_per_genome = config.fixed_strains_per_genome
            max_strains_per_genome = config.fixed_strains_per_genome
            logger.info(f"Using fixed strains per genome: {config.fixed_strains_per_genome}")
        else:
            max_strains_per_genome = max_strains if max_strains and max_strains >= 2 else 2
            min_strains_per_genome = 2
        
        logger.info(f"Creating between {min_strains_per_genome} and {max_strains_per_genome} strains per genome file (randomly chosen)")

        # Expand each genome file into multiple strains
        strains = []
        for genome_strain in genome_strains:
            # Randomly choose number of strains for this genome file (between 2 and max_strains)
            n_strains_for_this_genome = rng.integers(min_strains_per_genome, max_strains_per_genome + 1)
            logger.info(f"  {genome_strain.id}: creating {n_strains_for_this_genome} strains")
            
            # First strain is the reference (no SNVs introduced)
            reference_strain = Strain(
                id=f"{genome_strain.id}_ref",
                genome_file=genome_strain.genome_file,
                contigs=dict(genome_strain.contigs),  # Deep copy
                total_length=genome_strain.total_length
            )
            strains.append(reference_strain)
            
            # Create additional variant strains from this genome (strains[1:] will get SNVs)
            for i in range(1, n_strains_for_this_genome):
                variant_strain = Strain(
                    id=f"{genome_strain.id}_var_{i}",
                    genome_file=genome_strain.genome_file,
                    contigs=dict(genome_strain.contigs),  # Deep copy
                    total_length=genome_strain.total_length
                )
                strains.append(variant_strain)
        
        logger.info(f"Created {len(strains)} total strains ({len(genome_strains)} references + {len(strains) - len(genome_strains)} variants)")

        # Introduce SNVs synthetically
        logger.info("Introducing SNVs between strains")
        reference = strains[0]
        snv_positions = introduce_snvs(strains, config, rng)
        total_snvs = sum(len(pos) for pos in snv_positions.values())
        logger.info(f"Introduced {total_snvs:,} SNV positions")

    # Generate abundance profiles
    logger.info("Generating abundance profiles")
    abundances = generate_abundances(strains, config, rng)

    # Write reference genome (first strain)
    # Reference is already set above for both modes
    ref_path = os.path.join(output_dir, "reference.fasta")
    write_reference_fasta(reference, ref_path)
    logger.info(f"Wrote reference to {ref_path}")

    # Write ground truth VCF (primary + legacy name)
    vcf_path = os.path.join(output_dir, "truth_snvs.vcf")
    write_vcf(snv_positions, strains, reference, vcf_path, config=config, rng=rng)
    logger.info(f"Wrote truth VCF to {vcf_path}")
    
    # If VCF realism enabled, also write perturbations JSON
    if config.vcf_realism:
        perturbations_path = os.path.join(output_dir, "truth_vcf_perturbations.json")
        # This is written inside write_vcf, but we log it here too
        logger.info(f"VCF realism enabled - perturbations logged")

    legacy_vcf_path = os.path.join(output_dir, "truth_variants.vcf")
    write_vcf(snv_positions, strains, reference, legacy_vcf_path, config=config, rng=rng)
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
            'snv_counts_per_strain': config.snv_counts_per_strain,
            'coverage': config.coverage,
            'n_timepoints': config.n_timepoints,
            'sweep_fraction': config.sweep_fraction,
            'seed': config.seed,
            'fixed_strains_per_genome': config.fixed_strains_per_genome,
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
    parser.add_argument("--snv-density", type=int, default=10, help="SNVs per 10kb to introduce (only for synthetic mode)")
    parser.add_argument("--snv-counts", type=str, default=None,
                        help="Comma-separated SNV counts for strains[1:] (exact overrides density)")
    parser.add_argument("--error-rate", type=float, default=0.001, help="Sequencing error rate")
    parser.add_argument("--coverage", type=int, default=30, help="Read coverage per timepoint")
    parser.add_argument("--timepoints", type=int, default=4, help="Number of timepoints")
    parser.add_argument("--sweep-fraction", type=float, default=0.3, help="Fraction of sweeping strains")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-strains", type=int, default=None, help="Limit number of strains to use")
    parser.add_argument("--fixed-strains-per-genome", type=int, default=None,
                        help="Use an exact number of strains per genome (synthetic mode only)")
    parser.add_argument("--use-real-strains", action="store_true",
                        help="Use FASTA files directly as distinct strains (detect real SNVs instead of introducing synthetic ones)")

    args = parser.parse_args()

    # Create config
    config = SimulationConfig(
        snv_density=args.snv_density,
        error_rate=args.error_rate,
        coverage=args.coverage,
        n_timepoints=args.timepoints,
        sweep_fraction=args.sweep_fraction,
        seed=args.seed,
        snv_counts_per_strain=[int(x) for x in args.snv_counts.split(",")] if args.snv_counts else None,
        fixed_strains_per_genome=args.fixed_strains_per_genome,
        use_real_strains=args.use_real_strains,
    )

    # Adjust max strains based on complexity
    complexity_limits = {"simple": 15, "medium": 50, "complex": None}
    max_strains = args.max_strains or complexity_limits.get(args.complexity)

    # Run simulation
    run_simulation(args.genomes, args.output, config, max_strains=max_strains)


if __name__ == "__main__":
    main()
