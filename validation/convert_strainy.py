#!/usr/bin/env python3
"""
Convert Strainy output to strainphase lineages.tsv format for validation.

Reads Strainy's per-contig SNP_pos.tsv files and read cluster assignments,
and produces a lineages.tsv that can be fed into strainphase's run_validation().

Strainy output structure (--stage phase):
    out_strainy/
        {contig}/
            SNP_pos.tsv          <- SNP positions with per-cluster alleles
            reads_cl{N}.lst      <- reads assigned to cluster N

Usage:
    python validation/convert_strainy.py \
        --strainy-dir results/strainy_T1 \
        --vcf results/variants.vcf \
        --sample T1 \
        --output-dir results/strainy_T1/validation_input
"""

import argparse
import csv
import glob
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_vcf(vcf_path: str) -> dict[tuple[str, int], tuple[str, list[str]]]:
    """
    Parse VCF to get ref/alt alleles at each position.

    Returns: {(contig, 1-indexed position) -> (ref_allele, [alt_alleles])}
    """
    variants = {}
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            contig = parts[0]
            pos = int(parts[1])  # VCF is 1-indexed
            ref = parts[3]
            alts = parts[4].split(",")
            variants[(contig, pos)] = (ref, alts)

    logger.info(f"Parsed {len(variants)} variants from VCF")
    return variants


def parse_snp_pos_tsv(snp_pos_path: str) -> list[dict]:
    """
    Parse Strainy's SNP_pos.tsv for a contig.

    The file has columns: Pos, Ref, Alt, cluster_0, cluster_1, ...
    Cluster columns contain either the actual allele (A/C/G/T) or a
    numeric index (0=ref, 1=alt1, ...).

    Returns list of dicts:
    [
        {
            'pos': int (1-indexed),
            'ref': str,
            'alt': str,
            'clusters': {cluster_name: allele_str, ...}
        },
        ...
    ]
    """
    snps = []
    with open(snp_pos_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames or []

        # Identify cluster columns (anything not Pos/Ref/Alt)
        cluster_cols = [
            h for h in headers
            if h.lower() not in ("pos", "ref", "alt", "position", "reference", "alternative")
        ]

        for row in reader:
            pos_key = None
            for k in ("Pos", "pos", "Position", "position"):
                if k in row:
                    pos_key = k
                    break
            if pos_key is None:
                continue

            ref_key = None
            for k in ("Ref", "ref", "Reference", "reference"):
                if k in row:
                    ref_key = k
                    break

            alt_key = None
            for k in ("Alt", "alt", "Alternative", "alternative"):
                if k in row:
                    alt_key = k
                    break

            try:
                pos = int(row[pos_key])
            except (ValueError, TypeError):
                continue

            ref = row.get(ref_key, "") if ref_key else ""
            alt = row.get(alt_key, "") if alt_key else ""

            clusters = {}
            for col in cluster_cols:
                val = row.get(col, "").strip()
                if not val or val in (".", "-", "?", "N"):
                    continue

                # Check if the value is a numeric index (0=ref, 1=alt)
                if val.isdigit():
                    idx = int(val)
                    if idx == 0:
                        clusters[col] = ref
                    else:
                        alt_alleles = alt.split(",")
                        if idx - 1 < len(alt_alleles):
                            clusters[col] = alt_alleles[idx - 1]
                elif val.upper() in ("A", "C", "G", "T"):
                    clusters[col] = val.upper()

            if clusters:
                snps.append({
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "clusters": clusters,
                })

    return snps


def count_cluster_reads(contig_dir: str) -> dict[str, int]:
    """
    Count reads per cluster from reads_cl{N}.lst files.

    Returns: {cluster_name: read_count}
    """
    counts = {}
    contig_path = Path(contig_dir)

    # Look for reads_cl*.lst, reads_strain_*.lst, cluster_*.lst patterns
    for pattern in ["reads_cl*.lst", "reads_strain_*.lst", "cluster_*.lst"]:
        for lst_file in sorted(contig_path.glob(pattern)):
            # Extract cluster name from filename
            stem = lst_file.stem
            # Match patterns like reads_cl0, reads_strain_1, cluster_2
            match = re.search(r"(\d+)$", stem)
            if match:
                cluster_idx = match.group(1)
                cluster_name = f"cluster_{cluster_idx}"
            else:
                cluster_name = stem

            n_reads = 0
            with open(lst_file) as f:
                for line in f:
                    if line.strip():
                        n_reads += 1
            counts[cluster_name] = n_reads

    return counts


def find_contig_dirs(strainy_dir: str) -> list[tuple[str, str]]:
    """
    Find contig subdirectories in strainy output.

    Returns: [(contig_name, contig_dir_path), ...]
    """
    strainy_path = Path(strainy_dir)
    contig_dirs = []

    for d in sorted(strainy_path.iterdir()):
        if not d.is_dir():
            continue
        # Skip non-contig directories (like logs, tmp, etc.)
        if d.name.startswith(".") or d.name in ("tmp", "log", "logs"):
            continue
        # Check if directory has SNP_pos.tsv or similar SNP file
        has_snp_file = any(
            (d / name).exists()
            for name in ["SNP_pos.tsv", "snp_pos.tsv", "SNP_pos.csv"]
        )
        has_read_lists = bool(list(d.glob("reads_cl*.lst")) or list(d.glob("reads_strain_*.lst")))

        if has_snp_file or has_read_lists:
            contig_dirs.append((d.name, str(d)))

    return contig_dirs


def convert_strainy_to_lineages(
    strainy_dir: str,
    vcf_path: str,
    sample_id: str,
    output_dir: str,
) -> str:
    """
    Convert Strainy output to strainphase lineages.tsv format.

    Returns path to the output lineages.tsv.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parse VCF for allele lookup
    variants = parse_vcf(vcf_path)

    # Find contig directories
    contig_dirs = find_contig_dirs(strainy_dir)
    if not contig_dirs:
        logger.error(f"No contig directories with SNP data found in {strainy_dir}")
        return ""

    logger.info(f"Found {len(contig_dirs)} contig directories in {strainy_dir}")

    all_records = []
    global_cluster_idx = 0

    for contig_name, contig_dir in contig_dirs:
        # Parse SNP positions and cluster alleles
        snp_file = None
        for name in ["SNP_pos.tsv", "snp_pos.tsv", "SNP_pos.csv"]:
            candidate = os.path.join(contig_dir, name)
            if os.path.exists(candidate):
                snp_file = candidate
                break

        if snp_file is None:
            logger.warning(f"No SNP_pos.tsv in {contig_dir}")
            continue

        snps = parse_snp_pos_tsv(snp_file)
        if not snps:
            logger.warning(f"No SNP data parsed from {snp_file}")
            continue

        # Get all cluster names from SNP data
        cluster_names = set()
        for snp in snps:
            cluster_names.update(snp["clusters"].keys())
        cluster_names = sorted(cluster_names)

        if not cluster_names:
            logger.warning(f"No clusters found for {contig_name}")
            continue

        # Count reads per cluster
        read_counts = count_cluster_reads(contig_dir)

        # Build per-cluster allele strings
        for cluster_name in cluster_names:
            snv_alleles = []
            for snp in snps:
                allele = snp["clusters"].get(cluster_name)
                if allele is None:
                    continue
                # Use VCF position if available, otherwise use the position from SNP_pos.tsv
                pos = snp["pos"]  # Already 1-indexed from strainy
                snv_alleles.append(f"{pos}:{allele}")

            if not snv_alleles:
                continue

            n_reads = read_counts.get(cluster_name, 0)
            # Try matching by index if the cluster name mapping doesn't work
            if n_reads == 0:
                match = re.search(r"(\d+)", cluster_name)
                if match:
                    idx = match.group(1)
                    for key in read_counts:
                        if key.endswith(idx):
                            n_reads = read_counts[key]
                            break

            hap_id = f"strainy_{global_cluster_idx}"
            global_cluster_idx += 1

            record = {
                "lineage_id": hap_id,
                "sample": sample_id,
                "contig": contig_name,
                "track_id": hap_id,
                "supporting_reads": n_reads,
                "total_reads": n_reads,
                "snv_alleles": ",".join(snv_alleles),
            }
            all_records.append(record)

            logger.info(
                f"  {hap_id} ({cluster_name}) on {contig_name}: "
                f"{len(snv_alleles)} SNVs, {n_reads} reads"
            )

    # Write lineages.tsv
    output_path = os.path.join(output_dir, "lineages.tsv")
    fieldnames = [
        "lineage_id", "sample", "contig", "track_id",
        "supporting_reads", "total_reads", "snv_alleles",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_records)

    logger.info(f"Wrote {len(all_records)} records to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Strainy output to strainphase lineages.tsv for validation"
    )
    parser.add_argument(
        "--strainy-dir", required=True,
        help="Strainy output directory (contains contig subdirectories)",
    )
    parser.add_argument(
        "--vcf", required=True,
        help="VCF file used for the Strainy run",
    )
    parser.add_argument(
        "--sample", default="T1",
        help="Sample/timepoint name (default: T1)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for lineages.tsv",
    )
    parser.add_argument(
        "--truth-dir",
        help="If provided, also run validation against this truth directory",
    )
    parser.add_argument(
        "--validation-output-dir",
        help="Output directory for validation results (requires --truth-dir)",
    )
    args = parser.parse_args()

    # Convert Strainy output
    lineages_path = convert_strainy_to_lineages(
        args.strainy_dir, args.vcf, args.sample, args.output_dir
    )

    if not lineages_path:
        logger.error("Conversion failed")
        return

    # Optionally run validation
    if args.truth_dir:
        validation_output = args.validation_output_dir or os.path.join(
            args.output_dir, "validation"
        )
        logger.info(f"\nRunning validation against {args.truth_dir}...")

        from validation.validate_haplotypes import run_validation

        result = run_validation(
            detected_file=lineages_path,
            truth_dir=args.truth_dir,
            output_dir=validation_output,
        )

        logger.info(f"\n{'='*60}")
        logger.info("STRAINY VALIDATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"  Haplotype precision: {result.haplotype_precision:.3f}")
        logger.info(f"  Haplotype recall:    {result.haplotype_recall:.3f}")
        logger.info(f"  Haplotype F1:        {result.haplotype_f1:.3f}")
        logger.info(f"  SNV precision:       {result.snv_precision:.3f}")
        logger.info(f"  SNV recall:          {result.snv_recall:.3f}")
        if result.abundance_pearson_r is not None:
            logger.info(f"  Abundance Pearson r: {result.abundance_pearson_r:.3f}")
        logger.info(f"  False negatives:     {len(result.false_negatives or [])}")
        logger.info(f"  False positives:     {len(result.false_positives or [])}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
