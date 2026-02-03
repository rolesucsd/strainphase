#!/usr/bin/env python3
"""
Convert Floria output to strainphase lineages.tsv format for validation.

Reads Floria's vartig_info.txt and the VCF used for the run, and produces
a lineages.tsv that can be fed into strainphase's run_validation().

Usage:
    python validation/convert_floria.py \
        --floria-dir results/floria_10x \
        --vcf results/mixed_samples_10x/variants.vcf \
        --sample T1 \
        --output-dir results/floria_10x/validation_input

    # Then validate:
    python -m validation.validate_haplotypes \
        --detected results/floria_10x/validation_input/lineages.tsv \
        --truth-dir results/truth \
        --output-dir results/floria_10x/validation
"""

import argparse
import csv
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


def parse_vartig_info(vartig_info_path: str) -> list[dict]:
    """
    Parse Floria's vartig_info.txt.

    Returns list of haploset dicts:
    [
        {
            'hap_id': 'HAP0',
            'contig': 'contig_1',
            'snp_range': (start, end),
            'snps': [(0-indexed base_pos, allele_index, support_str), ...]
        },
        ...
    ]
    """
    haplosets = []
    current_hap = None

    with open(vartig_info_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            if line.startswith(">"):
                # Header line: >HAP0.results/floria/floria_10x/contig_1\tSNPRANGE:35-189
                # Extract HAP id and contig
                parts = line.split("\t")
                header = parts[0][1:]  # Remove '>'

                # Parse HAP id (e.g. "HAP0.results/floria/floria_10x/contig_1")
                hap_match = re.match(r"(HAP\d+)\.", header)
                hap_id = hap_match.group(1) if hap_match else header.split(".")[0]

                # Parse contig from the end of the header
                contig = header.split("/")[-1]

                # Parse SNP range
                snp_range = None
                for p in parts[1:]:
                    if p.startswith("SNPRANGE:"):
                        rng = p.split(":")[1]
                        start, end = rng.split("-")
                        snp_range = (int(start), int(end))

                current_hap = {
                    "hap_id": hap_id,
                    "contig": contig,
                    "snp_range": snp_range,
                    "snps": [],
                }
                haplosets.append(current_hap)
            else:
                # SNP line: "35:2335079\t0\t0:4\t"
                parts = line.split("\t")
                if len(parts) >= 2 and ":" in parts[0]:
                    snp_pos_str = parts[0]
                    snp_idx, base_pos_0idx = snp_pos_str.split(":")
                    allele_index = int(parts[1])
                    support = parts[2] if len(parts) > 2 else ""

                    current_hap["snps"].append(
                        (int(base_pos_0idx), allele_index, support.strip())
                    )

    logger.info(f"Parsed {len(haplosets)} haplosets from vartig_info.txt")
    return haplosets


def parse_haplosets(haplosets_path: str) -> dict[str, dict]:
    """
    Parse Floria's .haplosets file to get read counts per haploset.

    Returns: {hap_id -> {'n_reads': int, 'base_range': (start, end), 'coverage': float}}
    """
    hap_info = {}
    current_hap_id = None
    current_reads = 0
    current_base_range = None
    current_cov = 0.0

    with open(haplosets_path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                # Save previous haploset
                if current_hap_id is not None:
                    hap_info[current_hap_id] = {
                        "n_reads": current_reads,
                        "base_range": current_base_range,
                        "coverage": current_cov,
                    }

                # Parse new header
                parts = line.split("\t")
                header = parts[0][1:]
                hap_match = re.match(r"(HAP\d+)\.", header)
                current_hap_id = hap_match.group(1) if hap_match else header.split(".")[0]
                current_reads = 0

                for p in parts[1:]:
                    if p.startswith("BASERANGE:"):
                        rng = p.split(":")[1]
                        start, end = rng.split("-")
                        current_base_range = (int(start), int(end))
                    elif p.startswith("COV:"):
                        current_cov = float(p.split(":")[1])
            else:
                current_reads += 1

        # Save last haploset
        if current_hap_id is not None:
            hap_info[current_hap_id] = {
                "n_reads": current_reads,
                "base_range": current_base_range,
                "coverage": current_cov,
            }

    return hap_info


def convert_floria_to_lineages(
    floria_dir: str,
    vcf_path: str,
    sample_id: str,
    output_dir: str,
) -> str:
    """
    Convert Floria output to strainphase lineages.tsv format.

    Returns path to the output lineages.tsv.
    """
    floria_path = Path(floria_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Parse VCF
    variants = parse_vcf(vcf_path)

    # Find all contig directories
    contig_dirs = [d for d in floria_path.iterdir() if d.is_dir()]
    if not contig_dirs:
        logger.error(f"No contig directories found in {floria_dir}")
        return ""

    all_records = []

    for contig_dir in sorted(contig_dirs):
        contig_name = contig_dir.name

        # Parse vartig_info.txt
        vartig_info_path = contig_dir / "vartig_info.txt"
        if not vartig_info_path.exists():
            logger.warning(f"No vartig_info.txt in {contig_dir}")
            continue

        haplosets = parse_vartig_info(str(vartig_info_path))

        # Parse haplosets for read counts
        haplosets_path = contig_dir / f"{contig_name}.haplosets"
        hap_read_info = {}
        if haplosets_path.exists():
            hap_read_info = parse_haplosets(str(haplosets_path))

        # Convert each haploset to a lineages.tsv record
        for hap in haplosets:
            hap_id = hap["hap_id"]
            contig = hap["contig"]

            # Convert SNP alleles: Floria 0-indexed positions -> 1-indexed + allele lookup
            snv_alleles = []
            for base_pos_0idx, allele_index, _support in hap["snps"]:
                pos_1idx = base_pos_0idx + 1  # Convert to 1-indexed
                key = (contig, pos_1idx)

                if key in variants:
                    ref, alts = variants[key]
                    if allele_index == 0:
                        allele = ref
                    elif allele_index <= len(alts):
                        allele = alts[allele_index - 1]
                    else:
                        logger.warning(
                            f"Allele index {allele_index} out of range for {contig}:{pos_1idx}"
                        )
                        continue
                else:
                    logger.warning(
                        f"Position {contig}:{pos_1idx} not found in VCF (0-idx={base_pos_0idx})"
                    )
                    continue

                snv_alleles.append(f"{pos_1idx}:{allele}")

            if not snv_alleles:
                logger.warning(f"No valid SNV alleles for {hap_id} on {contig}")
                continue

            # Get read count info
            read_info = hap_read_info.get(hap_id, {})
            n_reads = read_info.get("n_reads", 0)

            record = {
                "lineage_id": hap_id,
                "sample": sample_id,
                "contig": contig,
                "track_id": hap_id,
                "supporting_reads": n_reads,
                "total_reads": n_reads,
                "snv_alleles": ",".join(snv_alleles),
            }
            all_records.append(record)

            logger.info(
                f"  {hap_id} on {contig}: {len(snv_alleles)} SNVs, {n_reads} reads"
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
        description="Convert Floria output to strainphase lineages.tsv for validation"
    )
    parser.add_argument(
        "--floria-dir", required=True,
        help="Floria output directory (contains contig subdirectories)",
    )
    parser.add_argument(
        "--vcf", required=True,
        help="VCF file used for the Floria run",
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

    # Convert Floria output
    lineages_path = convert_floria_to_lineages(
        args.floria_dir, args.vcf, args.sample, args.output_dir
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
        logger.info("FLORIA VALIDATION RESULTS")
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
