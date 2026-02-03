#!/usr/bin/env python3
"""
Generate ground truth files for strainphase validation from strain FASTA files.

Compares each strain FASTA to the reference to identify true SNVs, then writes
ground truth files in the format expected by validation/validate_haplotypes.py.

Usage:
    python benchmarks/generate_ground_truth.py \
        --reference reference.fasta \
        --strains strainA.fasta strainB.fasta \
        --output-dir output/ \
        --abundances output/abundances.tsv
"""

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_fasta(path: str) -> Dict[str, str]:
    """Parse a FASTA file into {contig_name: sequence}."""
    contigs = {}
    current_name = None
    current_seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    contigs[current_name] = "".join(current_seq).upper()
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_name is not None:
        contigs[current_name] = "".join(current_seq).upper()

    return contigs


def find_snvs(ref_seq: str, strain_seq: str) -> List[Tuple[int, str, str]]:
    """
    Compare two sequences and return SNV positions (0-indexed).

    Returns list of (position, ref_base, alt_base).
    Only considers substitutions at positions where both sequences have ACGT.
    """
    snvs = []
    min_len = min(len(ref_seq), len(strain_seq))

    for i in range(min_len):
        ref_base = ref_seq[i]
        alt_base = strain_seq[i]

        if ref_base == alt_base:
            continue
        if ref_base not in "ACGT" or alt_base not in "ACGT":
            continue

        snvs.append((i, ref_base, alt_base))

    return snvs


def load_abundances(path: str) -> Dict[str, Dict[str, float]]:
    """
    Load abundance profiles from TSV.

    Expects columns: strain_idx, timepoint, abundance
    Or the truth format: strain_id, T1, T2, ...

    Returns: {timepoint -> {strain_name -> abundance}}
    """
    abundances = defaultdict(dict)

    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames

        if "strain_idx" in headers:
            # Format: strain_idx, timepoint, abundance
            for row in reader:
                # We'll map strain_idx later
                abundances[row["timepoint"]][row["strain_idx"]] = float(row["abundance"])
        elif "strain_id" in headers:
            # Format: strain_id, T1, T2, ...
            for row in reader:
                strain_id = row["strain_id"]
                for key, val in row.items():
                    if key != "strain_id" and key.startswith("T"):
                        abundances[key][strain_id] = float(val)

    return dict(abundances)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth files from strain FASTA files"
    )
    parser.add_argument("--reference", required=True, help="Reference FASTA file")
    parser.add_argument(
        "--strains", nargs="+", required=True, help="Strain FASTA files"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--abundances",
        default=None,
        help="Abundances TSV file (strain_idx/timepoint/abundance)",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load reference ──
    logger.info(f"Loading reference: {args.reference}")
    ref_contigs = parse_fasta(args.reference)
    logger.info(f"  {len(ref_contigs)} contigs, {sum(len(s) for s in ref_contigs.values()):,} bp")

    # ── Load strains and find SNVs ──
    strain_names = []
    strain_snvs = {}  # {strain_name -> {contig -> [(pos_0idx, ref, alt)]}}
    all_snv_positions = defaultdict(set)  # {contig -> set of 1-indexed positions}

    for strain_path in args.strains:
        strain_name = os.path.splitext(os.path.basename(strain_path))[0]
        strain_names.append(strain_name)

        logger.info(f"Loading strain: {strain_name} ({strain_path})")
        strain_contigs = parse_fasta(strain_path)

        strain_snvs[strain_name] = {}
        total_snvs = 0

        for contig_name, ref_seq in ref_contigs.items():
            if contig_name not in strain_contigs:
                # Try matching by order if names differ
                logger.warning(
                    f"  Contig {contig_name} not found in {strain_name}"
                )
                continue

            strain_seq = strain_contigs[contig_name]
            snvs = find_snvs(ref_seq, strain_seq)

            if snvs:
                strain_snvs[strain_name][contig_name] = snvs
                for pos, ref_base, alt_base in snvs:
                    all_snv_positions[contig_name].add(pos + 1)  # 1-indexed
                total_snvs += len(snvs)

        logger.info(f"  {total_snvs} SNVs vs reference")

    # If contig names don't match, try matching by position/order
    if not any(strain_snvs[s] for s in strain_names):
        logger.warning("No SNVs found with contig name matching. Trying positional matching...")
        ref_contig_names = list(ref_contigs.keys())

        for strain_name, strain_path in zip(strain_names, args.strains):
            strain_contigs = parse_fasta(strain_path)
            strain_contig_names = list(strain_contigs.keys())

            strain_snvs[strain_name] = {}
            total_snvs = 0

            for i, ref_name in enumerate(ref_contig_names):
                if i >= len(strain_contig_names):
                    break
                strain_name_contig = strain_contig_names[i]
                ref_seq = ref_contigs[ref_name]
                strain_seq = strain_contigs[strain_name_contig]

                snvs = find_snvs(ref_seq, strain_seq)
                if snvs:
                    # Use reference contig name for consistency
                    strain_snvs[strain_name][ref_name] = snvs
                    for pos, ref_base, alt_base in snvs:
                        all_snv_positions[ref_name].add(pos + 1)
                    total_snvs += len(snvs)

            logger.info(f"  {strain_name}: {total_snvs} SNVs (positional matching)")

    # ── Load or generate abundances ──
    n_strains = len(strain_names)

    if args.abundances and os.path.exists(args.abundances):
        raw_abundances = load_abundances(args.abundances)
        # Map strain indices to names if needed
        abundances = {}
        for tp, tp_abunds in raw_abundances.items():
            tp_key = f"T{tp}" if not tp.startswith("T") else tp
            abundances[tp_key] = {}
            for key, val in tp_abunds.items():
                if key.isdigit():
                    idx = int(key)
                    if idx < n_strains:
                        abundances[tp_key][strain_names[idx]] = val
                else:
                    abundances[tp_key][key] = val
    else:
        # Default: equal abundances, 4 timepoints
        abundances = {}
        for tp_idx in range(1, 5):
            tp_key = f"T{tp_idx}"
            abundances[tp_key] = {s: 1.0 / n_strains for s in strain_names}

    timepoints = sorted(abundances.keys())

    # ── Write truth_strains.tsv ──
    strains_file = os.path.join(args.output_dir, "truth_strains.tsv")
    ref_length = sum(len(s) for s in ref_contigs.values())
    with open(strains_file, "w") as f:
        f.write("strain_id\tspecies\tgenome_file\ttotal_length\tsnv_count\tis_sweeping\n")
        for strain_name, strain_path in zip(strain_names, args.strains):
            snv_count = sum(len(v) for v in strain_snvs[strain_name].values())
            f.write(f"{strain_name}\tunknown\t{strain_path}\t{ref_length}\t{snv_count}\tFalse\n")
    logger.info(f"Wrote {strains_file}")

    # ── Write truth_abundances.tsv ──
    abund_file = os.path.join(args.output_dir, "truth_abundances.tsv")
    with open(abund_file, "w") as f:
        f.write("strain_id\t" + "\t".join(timepoints) + "\n")
        for strain_name in strain_names:
            abunds = [f"{abundances[tp].get(strain_name, 0):.6f}" for tp in timepoints]
            f.write(f"{strain_name}\t" + "\t".join(abunds) + "\n")
    logger.info(f"Wrote {abund_file}")

    # ── Write truth_snvs.vcf ──
    vcf_file = os.path.join(args.output_dir, "truth_snvs.vcf")
    with open(vcf_file, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=generate_ground_truth\n")

        for contig_name, seq in ref_contigs.items():
            f.write(f"##contig=<ID={contig_name},length={len(seq)}>\n")

        f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n')
        f.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n')
        f.write(
            '##INFO=<ID=STRAINS,Number=.,Type=String,Description="Strains with alt allele">\n'
        )
        f.write(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        )
        f.write(
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">\n'
        )
        f.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
        )

        # Build per-position strain/allele lookup
        # position_info: {(contig, pos_1idx) -> {strain -> alt_base}}
        position_info = defaultdict(dict)
        for strain_name in strain_names:
            for contig, snvs in strain_snvs[strain_name].items():
                for pos_0, ref_base, alt_base in snvs:
                    position_info[(contig, pos_0 + 1)][strain_name] = alt_base

        # Write sorted VCF records
        for (contig, pos), strain_alleles in sorted(position_info.items()):
            ref_base = ref_contigs[contig][pos - 1]

            # Group strains by allele
            allele_strains = defaultdict(list)
            for strain, allele in strain_alleles.items():
                allele_strains[allele].append(strain)

            # Pick main alt (most common)
            alt_base = max(allele_strains, key=lambda a: len(allele_strains[a]))

            # Format STRAINS info
            strains_parts = []
            for allele, strains in sorted(allele_strains.items()):
                strains_parts.append(f"{allele}:{','.join(sorted(strains))}")
            strains_info = "|".join(strains_parts)

            info = f"DP=50;AF=0.5;STRAINS={strains_info}"
            f.write(
                f"{contig}\t{pos}\t.\t{ref_base}\t{alt_base}\t30\tPASS\t{info}\tGT:DP\t0/1:50\n"
            )

    logger.info(f"Wrote {vcf_file} ({sum(len(p) for p in all_snv_positions.values())} positions)")

    # ── Write truth_snv_positions.tsv ──
    pos_file = os.path.join(args.output_dir, "truth_snv_positions.tsv")
    with open(pos_file, "w") as f:
        f.write("contig\tposition\n")
        for contig in sorted(all_snv_positions.keys()):
            for pos in sorted(all_snv_positions[contig]):
                f.write(f"{contig}\t{pos}\n")
    logger.info(f"Wrote {pos_file}")

    # ── Write truth_haplotypes.tsv ──
    hap_file = os.path.join(args.output_dir, "truth_haplotypes.tsv")
    with open(hap_file, "w") as f:
        f.write("strain_id\tcontig\tsnv_alleles\n")
        for strain_name in strain_names:
            strain_var_lookup = {}
            for contig, snvs in strain_snvs[strain_name].items():
                for pos_0, ref_base, alt_base in snvs:
                    strain_var_lookup[(contig, pos_0 + 1)] = alt_base

            for contig in sorted(all_snv_positions.keys()):
                positions = sorted(all_snv_positions[contig])
                alleles = []
                for pos in positions:
                    if (contig, pos) in strain_var_lookup:
                        alleles.append(f"{pos}:{strain_var_lookup[(contig, pos)]}")
                    else:
                        # Reference allele at this position
                        ref_base = ref_contigs[contig][pos - 1]
                        alleles.append(f"{pos}:{ref_base}")
                allele_str = ",".join(alleles) if alleles else "."
                f.write(f"{strain_name}\t{contig}\t{allele_str}\n")
    logger.info(f"Wrote {hap_file}")

    # ── Write truth_tracks.tsv ──
    tracks_file = os.path.join(args.output_dir, "truth_tracks.tsv")
    with open(tracks_file, "w") as f:
        f.write("strain_id\tcontig\tstart\tend\twindow_chain\n")
        for strain_name in strain_names:
            for contig_name, seq in ref_contigs.items():
                f.write(f"{strain_name}\t{contig_name}\t1\t{len(seq)}\tfull\n")
    logger.info(f"Wrote {tracks_file}")

    # ── Write truth_lineages.tsv ──
    lineages_file = os.path.join(args.output_dir, "truth_lineages.tsv")
    with open(lineages_file, "w") as f:
        f.write("strain_id\tlineage_id\tcontig\n")
        for strain_name in strain_names:
            for contig_name in ref_contigs.keys():
                f.write(f"{strain_name}\t{strain_name}\t{contig_name}\n")
    logger.info(f"Wrote {lineages_file}")

    # ── Summary ──
    total_snv_positions = sum(len(p) for p in all_snv_positions.values())
    logger.info("")
    logger.info("=" * 60)
    logger.info("GROUND TRUTH SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Reference contigs:    {len(ref_contigs)}")
    logger.info(f"  Reference length:     {ref_length:,} bp")
    logger.info(f"  Strains:              {n_strains}")
    logger.info(f"  Total SNV positions:  {total_snv_positions}")
    logger.info(f"  Timepoints:           {len(timepoints)}")
    for strain_name in strain_names:
        n_snvs = sum(len(v) for v in strain_snvs[strain_name].values())
        logger.info(f"    {strain_name}: {n_snvs} SNVs")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
