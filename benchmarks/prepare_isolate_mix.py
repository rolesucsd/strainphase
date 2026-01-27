#!/usr/bin/env python3
"""
Prepare real isolate BAMs for strainphase benchmarking.

Takes individual isolate BAMs (each with reads from one strain) and creates
mixed samples simulating a metagenomic scenario where multiple strains are
present at different abundances.

Usage:
    python benchmarks/prepare_isolate_mix.py \
        --bams isolate1.bam isolate2.bam isolate3.bam \
        --reference reference.fasta \
        --output output_dir/ \
        --timepoints 4 \
        --target-coverage 30

This will:
1. Subsample reads from each isolate BAM according to target abundances
2. Merge into mixed BAM files (one per timepoint)
3. Call variants on merged data
4. Generate ground truth files for validation
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json

import numpy as np

try:
    import pysam
except ImportError:
    print("Error: pysam is required. Install with: pip install pysam")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_bam_stats(bam_path: str) -> Dict:
    """Get basic statistics from a BAM file."""
    bam = pysam.AlignmentFile(bam_path, "rb")

    total_reads = 0
    total_bases = 0

    for read in bam.fetch(until_eof=True):
        if not read.is_unmapped:
            total_reads += 1
            total_bases += read.query_length

    bam.close()

    # Get reference length
    bam = pysam.AlignmentFile(bam_path, "rb")
    ref_length = sum(bam.lengths)
    bam.close()

    coverage = total_bases / ref_length if ref_length > 0 else 0

    return {
        "total_reads": total_reads,
        "total_bases": total_bases,
        "ref_length": ref_length,
        "coverage": coverage,
    }


def generate_abundance_profiles(
    n_strains: int,
    n_timepoints: int,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Generate abundance profiles across timepoints.

    Returns: {timepoint -> {strain_id -> relative_abundance}}
    """
    rng = np.random.default_rng(seed)

    timepoints = [f"T{i+1}" for i in range(n_timepoints)]
    abundances = {tp: {} for tp in timepoints}

    # Generate base abundances with some dynamics
    base_abundances = rng.dirichlet(np.ones(n_strains) * 2)  # Start with random proportions

    for tp_idx, tp in enumerate(timepoints):
        # Add some temporal dynamics
        noise = rng.normal(0, 0.05, n_strains)
        tp_abundances = base_abundances + noise * (tp_idx / n_timepoints)
        tp_abundances = np.clip(tp_abundances, 0.01, None)
        tp_abundances = tp_abundances / tp_abundances.sum()  # Normalize

        for strain_idx in range(n_strains):
            abundances[tp][f"strain_{strain_idx+1}"] = float(tp_abundances[strain_idx])

    return abundances


def subsample_bam(
    input_bam: str,
    output_bam: str,
    fraction: float,
    seed: int = 42
) -> int:
    """
    Subsample reads from a BAM file.

    Returns number of reads written.
    """
    rng = random.Random(seed)

    bam_in = pysam.AlignmentFile(input_bam, "rb")
    bam_out = pysam.AlignmentFile(output_bam, "wb", header=bam_in.header)

    n_written = 0
    for read in bam_in.fetch(until_eof=True):
        if rng.random() < fraction:
            bam_out.write(read)
            n_written += 1

    bam_in.close()
    bam_out.close()

    return n_written


def merge_bams(input_bams: List[str], output_bam: str):
    """Merge multiple BAM files into one."""
    if len(input_bams) == 1:
        # Just copy
        import shutil
        shutil.copy(input_bams[0], output_bam)
        return

    # Use pysam merge
    pysam.merge("-f", output_bam, *input_bams)


def call_variants(
    bam_path: str,
    reference_path: str,
    output_vcf: str,
    min_depth: int = 5,
    min_af: float = 0.01
):
    """
    Call variants from a BAM file using pysam pileup.

    Simple variant caller that identifies positions with multiple alleles.
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    ref = pysam.FastaFile(reference_path)

    variants = []

    for contig in bam.references:
        contig_len = bam.get_reference_length(contig)
        logger.info(f"  Scanning {contig} ({contig_len:,} bp)")

        for pileup_col in bam.pileup(contig, min_base_quality=20, min_mapping_quality=20):
            pos = pileup_col.pos
            ref_base = ref.fetch(contig, pos, pos + 1).upper()

            if ref_base not in "ACGT":
                continue

            # Count alleles
            base_counts = defaultdict(int)
            for pileup_read in pileup_col.pileups:
                if pileup_read.is_del or pileup_read.is_refskip:
                    continue
                base = pileup_read.alignment.query_sequence[pileup_read.query_position].upper()
                if base in "ACGT":
                    base_counts[base] += 1

            total_depth = sum(base_counts.values())
            if total_depth < min_depth:
                continue

            # Check for variants
            for alt_base, count in base_counts.items():
                if alt_base != ref_base:
                    af = count / total_depth
                    if af >= min_af:
                        variants.append({
                            "contig": contig,
                            "pos": pos + 1,  # 1-indexed for VCF
                            "ref": ref_base,
                            "alt": alt_base,
                            "depth": total_depth,
                            "af": af,
                        })

    bam.close()
    ref.close()

    # Write VCF
    with open(output_vcf, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=prepare_isolate_mix\n")

        # Add contig headers
        ref = pysam.FastaFile(reference_path)
        for contig in ref.references:
            f.write(f"##contig=<ID={contig},length={ref.get_reference_length(contig)}>\n")
        ref.close()

        f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n')
        f.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n')
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")

        for var in variants:
            info = f"DP={var['depth']};AF={var['af']:.3f}"
            f.write(f"{var['contig']}\t{var['pos']}\t.\t{var['ref']}\t{var['alt']}\t30\tPASS\t{info}\tGT:DP\t0/1:{var['depth']}\n")

    logger.info(f"  Called {len(variants)} variants")
    return variants


def write_ground_truth(
    output_dir: str,
    strain_names: List[str],
    abundances: Dict[str, Dict[str, float]],
    reference_path: str,
    variants_by_strain: Dict[str, List[Dict]]
):
    """Write ground truth files for validation."""

    # truth_strains.tsv
    with open(os.path.join(output_dir, "truth_strains.tsv"), "w") as f:
        f.write("strain_id\tspecies\tgenome_file\ttotal_length\tsnv_count\tis_sweeping\n")
        for strain in strain_names:
            snv_count = len(variants_by_strain.get(strain, []))
            f.write(f"{strain}\tunknown\t{reference_path}\t0\t{snv_count}\tFalse\n")

    # truth_abundances.tsv
    timepoints = sorted(abundances.keys())
    with open(os.path.join(output_dir, "truth_abundances.tsv"), "w") as f:
        f.write("strain_id\t" + "\t".join(timepoints) + "\n")
        for strain in strain_names:
            abunds = [f"{abundances[tp].get(strain, 0):.6f}" for tp in timepoints]
            f.write(f"{strain}\t" + "\t".join(abunds) + "\n")

    # truth_haplotypes.tsv - SNV alleles per strain
    # Get all SNV positions
    all_snv_positions = defaultdict(set)
    for strain, variants in variants_by_strain.items():
        for var in variants:
            all_snv_positions[var["contig"]].add(var["pos"])

    with open(os.path.join(output_dir, "truth_haplotypes.tsv"), "w") as f:
        f.write("strain_id\tcontig\tsnv_alleles\n")
        for strain in strain_names:
            strain_vars = {(v["contig"], v["pos"]): v["alt"] for v in variants_by_strain.get(strain, [])}
            for contig, positions in sorted(all_snv_positions.items()):
                alleles = []
                for pos in sorted(positions):
                    if (contig, pos) in strain_vars:
                        alleles.append(f"{pos}:{strain_vars[(contig, pos)]}")
                    else:
                        # Reference allele (would need to look up)
                        alleles.append(f"{pos}:REF")
                allele_str = ",".join(alleles) if alleles else "."
                f.write(f"{strain}\t{contig}\t{allele_str}\n")

    logger.info(f"Wrote ground truth files to {output_dir}")


def load_isolate_vcfs(
    vcf_paths: List[str],
    strain_names: List[str]
) -> Tuple[Dict[str, List[Dict]], Dict[str, set]]:
    """
    Load variants from per-isolate VCF files.

    Returns:
        variants_by_strain: {strain_id -> [variant_dicts]}
        all_positions: {contig -> set of positions}
    """
    variants_by_strain = {strain: [] for strain in strain_names}
    all_positions = defaultdict(set)

    for strain, vcf_path in zip(strain_names, vcf_paths):
        if not os.path.exists(vcf_path):
            logger.warning(f"VCF not found for {strain}: {vcf_path}")
            continue

        try:
            vcf = pysam.VariantFile(vcf_path)
            for record in vcf.fetch():
                # Skip indels, only process SNVs
                if len(record.ref) != 1 or any(len(alt) != 1 for alt in record.alts):
                    continue

                var = {
                    "contig": record.chrom,
                    "pos": record.pos,  # 1-indexed from VCF
                    "ref": record.ref,
                    "alt": record.alts[0],
                }
                variants_by_strain[strain].append(var)
                all_positions[record.chrom].add(record.pos)

            vcf.close()
            logger.info(f"  {strain}: {len(variants_by_strain[strain])} SNVs from {os.path.basename(vcf_path)}")
        except Exception as e:
            logger.warning(f"Error reading VCF for {strain}: {e}")

    return variants_by_strain, all_positions


def create_combined_vcf(
    all_positions: Dict[str, set],
    variants_by_strain: Dict[str, List[Dict]],
    reference_path: str,
    output_vcf: str
):
    """
    Create a combined VCF with all variant positions from all strains.
    """
    ref = pysam.FastaFile(reference_path)

    # Build lookup: (contig, pos) -> {strain -> alt_allele}
    position_alleles = defaultdict(dict)
    for strain, variants in variants_by_strain.items():
        for var in variants:
            key = (var["contig"], var["pos"])
            position_alleles[key][strain] = var["alt"]

    # Write combined VCF
    with open(output_vcf, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=prepare_isolate_mix\n")

        # Add contig headers
        for contig in ref.references:
            f.write(f"##contig=<ID={contig},length={ref.get_reference_length(contig)}>\n")

        f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n')
        f.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n')
        f.write('##INFO=<ID=STRAINS,Number=.,Type=String,Description="Strains with this variant">\n')
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")

        # Sort positions and write
        all_vars = []
        for contig, positions in all_positions.items():
            for pos in sorted(positions):
                ref_base = ref.fetch(contig, pos - 1, pos).upper()  # pos is 1-indexed
                strains_at_pos = position_alleles.get((contig, pos), {})

                if strains_at_pos:
                    # Get the alt allele (should be same across strains for same position)
                    alt_base = list(strains_at_pos.values())[0]
                    strain_list = ",".join(sorted(strains_at_pos.keys()))

                    all_vars.append((contig, pos, ref_base, alt_base, strain_list))

        # Sort by contig and position
        all_vars.sort(key=lambda x: (x[0], x[1]))

        for contig, pos, ref_base, alt_base, strain_list in all_vars:
            info = f"DP=50;AF=0.5;STRAINS={strain_list}"
            f.write(f"{contig}\t{pos}\t.\t{ref_base}\t{alt_base}\t30\tPASS\t{info}\tGT:DP\t0/1:50\n")

    ref.close()
    logger.info(f"  Created combined VCF with {len(all_vars)} SNV positions")

    return len(all_vars)


def write_ground_truth_from_vcfs(
    output_dir: str,
    strain_names: List[str],
    abundances: Dict[str, Dict[str, float]],
    reference_path: str,
    variants_by_strain: Dict[str, List[Dict]],
    all_positions: Dict[str, set]
):
    """Write accurate ground truth files using per-isolate VCF data."""

    ref = pysam.FastaFile(reference_path)
    ref_length = sum(ref.lengths)

    # truth_strains.tsv
    with open(os.path.join(output_dir, "truth_strains.tsv"), "w") as f:
        f.write("strain_id\tspecies\tgenome_file\ttotal_length\tsnv_count\tis_sweeping\n")
        for strain in strain_names:
            snv_count = len(variants_by_strain.get(strain, []))
            f.write(f"{strain}\tunknown\t{reference_path}\t{ref_length}\t{snv_count}\tFalse\n")

    # truth_abundances.tsv
    timepoints = sorted(abundances.keys())
    with open(os.path.join(output_dir, "truth_abundances.tsv"), "w") as f:
        f.write("strain_id\t" + "\t".join(timepoints) + "\n")
        for strain in strain_names:
            abunds = [f"{abundances[tp].get(strain, 0):.6f}" for tp in timepoints]
            f.write(f"{strain}\t" + "\t".join(abunds) + "\n")

    # truth_haplotypes.tsv - SNV alleles per strain per contig
    # Build lookup for each strain's variants
    strain_var_lookup = {}
    for strain, variants in variants_by_strain.items():
        strain_var_lookup[strain] = {(v["contig"], v["pos"]): v["alt"] for v in variants}

    with open(os.path.join(output_dir, "truth_haplotypes.tsv"), "w") as f:
        f.write("strain_id\tcontig\tsnv_alleles\n")
        for strain in strain_names:
            lookup = strain_var_lookup.get(strain, {})
            for contig in sorted(all_positions.keys()):
                positions = sorted(all_positions[contig])
                alleles = []
                for pos in positions:
                    if (contig, pos) in lookup:
                        # This strain has the alt allele
                        alleles.append(f"{pos}:{lookup[(contig, pos)]}")
                    else:
                        # This strain has the reference allele
                        ref_base = ref.fetch(contig, pos - 1, pos).upper()
                        alleles.append(f"{pos}:{ref_base}")
                allele_str = ",".join(alleles) if alleles else "."
                f.write(f"{strain}\t{contig}\t{allele_str}\n")

    # truth_snv_positions.tsv
    with open(os.path.join(output_dir, "truth_snv_positions.tsv"), "w") as f:
        f.write("contig\tposition\n")
        for contig in sorted(all_positions.keys()):
            for pos in sorted(all_positions[contig]):
                f.write(f"{contig}\t{pos}\n")

    ref.close()
    logger.info(f"  Wrote ground truth files to {output_dir}")


def prepare_isolate_mix(
    bam_paths: List[str],
    reference_path: str,
    output_dir: str,
    vcf_paths: Optional[List[str]] = None,
    n_timepoints: int = 4,
    target_coverage: int = 30,
    seed: int = 42
):
    """
    Main function to prepare isolate BAMs for strainphase benchmarking.

    Args:
        bam_paths: List of BAM files, one per isolate
        reference_path: Reference FASTA file
        output_dir: Output directory
        vcf_paths: Optional list of VCF files, one per isolate (for accurate ground truth)
        n_timepoints: Number of timepoints to create
        target_coverage: Target coverage per timepoint
        seed: Random seed
    """
    os.makedirs(output_dir, exist_ok=True)

    n_strains = len(bam_paths)
    strain_names = [f"strain_{i+1}" for i in range(n_strains)]

    logger.info("=" * 60)
    logger.info("PREPARE ISOLATE MIX FOR STRAINPHASE")
    logger.info("=" * 60)
    logger.info(f"Input BAMs: {n_strains}")
    logger.info(f"Timepoints: {n_timepoints}")
    logger.info(f"Target coverage: {target_coverage}x")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Step 1: Get BAM statistics
    logger.info("\nStep 1: Analyzing input BAMs")
    bam_stats = {}
    for i, bam_path in enumerate(bam_paths):
        logger.info(f"  {strain_names[i]}: {os.path.basename(bam_path)}")
        stats = get_bam_stats(bam_path)
        bam_stats[strain_names[i]] = stats
        logger.info(f"    Reads: {stats['total_reads']:,}, Coverage: {stats['coverage']:.1f}x")

    # Step 2: Generate abundance profiles
    logger.info("\nStep 2: Generating abundance profiles")
    abundances = generate_abundance_profiles(n_strains, n_timepoints, seed)
    for tp, tp_abunds in sorted(abundances.items()):
        logger.info(f"  {tp}: " + ", ".join(f"{s}={a:.2f}" for s, a in tp_abunds.items()))

    # Step 3: Create mixed BAMs for each timepoint
    logger.info("\nStep 3: Creating mixed BAM files")

    ref = pysam.FastaFile(reference_path)
    ref_length = sum(ref.lengths)
    ref.close()

    timepoints = sorted(abundances.keys())

    for tp in timepoints:
        logger.info(f"\n  Creating {tp} mixed BAM:")

        temp_bams = []
        total_target_bases = ref_length * target_coverage

        for i, (strain, bam_path) in enumerate(zip(strain_names, bam_paths)):
            abundance = abundances[tp][strain]
            strain_target_bases = total_target_bases * abundance

            # Calculate subsample fraction
            stats = bam_stats[strain]
            if stats["total_bases"] > 0:
                fraction = min(1.0, strain_target_bases / stats["total_bases"])
            else:
                fraction = 0

            logger.info(f"    {strain}: abundance={abundance:.2f}, subsample={fraction:.3f}")

            # Subsample
            temp_bam = os.path.join(output_dir, f"temp_{tp}_{strain}.bam")
            n_reads = subsample_bam(bam_path, temp_bam, fraction, seed=seed + i + hash(tp))
            temp_bams.append(temp_bam)
            logger.info(f"      Kept {n_reads:,} reads")

        # Merge subsampled BAMs
        merged_bam = os.path.join(output_dir, f"{tp}.unsorted.bam")
        merge_bams(temp_bams, merged_bam)

        # Sort and index
        sorted_bam = os.path.join(output_dir, f"{tp}.bam")
        pysam.sort("-o", sorted_bam, merged_bam)
        pysam.index(sorted_bam)

        # Cleanup temp files
        for temp_bam in temp_bams:
            os.remove(temp_bam)
        os.remove(merged_bam)

        logger.info(f"    Created {tp}.bam")

    # Step 4: Load per-isolate VCFs or call variants
    logger.info("\nStep 4: Processing variants")

    if vcf_paths and all(os.path.exists(p) for p in vcf_paths):
        logger.info("  Using provided per-isolate VCF files")
        variants_by_strain, all_positions = load_isolate_vcfs(vcf_paths, strain_names)

        # Create combined VCF
        combined_vcf = os.path.join(output_dir, "variants.vcf")
        n_variants = create_combined_vcf(all_positions, variants_by_strain, reference_path, combined_vcf)
    else:
        logger.info("  No per-isolate VCFs provided, calling variants from merged BAM")
        first_bam = os.path.join(output_dir, f"{timepoints[0]}.bam")
        combined_vcf = os.path.join(output_dir, "variants.vcf")
        variants = call_variants(first_bam, reference_path, combined_vcf)
        n_variants = len(variants)
        variants_by_strain = {strain: [] for strain in strain_names}
        all_positions = defaultdict(set)
        for var in variants:
            all_positions[var["contig"]].add(var["pos"])

    # Compress and index VCF
    vcf_gz = combined_vcf + ".gz"
    pysam.tabix_compress(combined_vcf, vcf_gz, force=True)
    pysam.tabix_index(vcf_gz, preset="vcf", force=True)
    logger.info(f"  Created variants.vcf.gz with {n_variants} SNV sites")

    # Step 5: Copy reference
    import shutil
    ref_dest = os.path.join(output_dir, "reference.fasta")
    shutil.copy(reference_path, ref_dest)
    pysam.faidx(ref_dest)
    logger.info(f"\nStep 5: Copied reference to {ref_dest}")

    # Step 6: Write ground truth
    logger.info("\nStep 6: Writing ground truth files")
    if vcf_paths and all(os.path.exists(p) for p in vcf_paths):
        write_ground_truth_from_vcfs(output_dir, strain_names, abundances, reference_path,
                                      variants_by_strain, all_positions)
    else:
        write_ground_truth(output_dir, strain_names, abundances, reference_path, variants_by_strain)

    # Save config
    config = {
        "n_strains": n_strains,
        "n_timepoints": n_timepoints,
        "target_coverage": target_coverage,
        "seed": seed,
        "strain_bams": {strain: bam for strain, bam in zip(strain_names, bam_paths)},
        "strain_vcfs": {strain: vcf for strain, vcf in zip(strain_names, vcf_paths)} if vcf_paths else None,
        "abundances": abundances,
        "variants_per_strain": {strain: len(variants_by_strain.get(strain, [])) for strain in strain_names},
        "total_snv_positions": sum(len(pos) for pos in all_positions.values()),
    }
    with open(os.path.join(output_dir, "mix_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mixed BAMs: {', '.join(f'{tp}.bam' for tp in timepoints)}")
    logger.info(f"Variants: variants.vcf.gz")
    logger.info(f"Reference: reference.fasta")
    logger.info("\nYou can now run strainphase benchmarking with:")
    logger.info(f"  python benchmarks/run_full_benchmark.py \\")
    logger.info(f"      --genomes {output_dir} \\")
    logger.info(f"      --output results/real_strains/ \\")
    logger.info(f"      --use-real-strains --resume")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare isolate BAMs for strainphase benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--bams", nargs="+", required=True,
                        help="Input BAM files (one per isolate)")
    parser.add_argument("--vcfs", nargs="+", default=None,
                        help="Input VCF files (one per isolate, same order as BAMs). "
                             "Provides accurate ground truth for strain-specific variants.")
    parser.add_argument("--reference", required=True,
                        help="Reference FASTA file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory")
    parser.add_argument("--timepoints", type=int, default=4,
                        help="Number of timepoints to create")
    parser.add_argument("--target-coverage", type=int, default=30,
                        help="Target coverage per timepoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Validate VCF count matches BAM count
    if args.vcfs and len(args.vcfs) != len(args.bams):
        parser.error(f"Number of VCFs ({len(args.vcfs)}) must match number of BAMs ({len(args.bams)})")

    prepare_isolate_mix(
        bam_paths=args.bams,
        reference_path=args.reference,
        output_dir=args.output,
        vcf_paths=args.vcfs,
        n_timepoints=args.timepoints,
        target_coverage=args.target_coverage,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
