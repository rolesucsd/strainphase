#!/usr/bin/env python3
"""
Convert SNooPy variant output into a strainphase-ready VCF.

SNooPy plays the same role as Clair3 in the strainphase pipeline: it nominates
candidate polymorphic sites that ``strainphase run`` / ``strainphase longitudinal``
consume via ``--vcf`` / ``--vcfs``. But raw SNooPy output cannot be fed directly,
for two reasons this script fixes:

1. **SNooPy reports changes in BLOCKS.** A single SNooPy record can span several
   reference bases, e.g. ``POS  TACG  CACC`` (an equal-length multi-base
   substitution / MNP) or a length-changing indel block. strainphase's
   ``load_snvs`` (see strainphase/core.py) **silently skips every equal-length
   multi-base record** ("MNP ... skip; not handled") and collapses indels. On a
   real cohort ~a third of SNooPy's records are MNP blocks, and they hide the
   *majority* of its SNVs. So feeding raw SNooPy throws away most of the signal.

   Fix: **atomize** each block. An equal-length block at POS becomes one SNV per
   offset ``i`` where ``REF[i] != ALT[i]`` (identical positions inside the block
   are not variants and are dropped). This needs no reference — the per-base REF
   and ALT come straight from the block.

2. **SNooPy's per-contig ``tmp/*.vcf`` files are bare VCF bodies** (no header, no
   ``##contig`` lines, unsorted, not indexed). strainphase opens the VCF with
   pysam and calls ``fetch(contig=...)``, which requires a bgzipped +
   tabix-indexed VCF with proper ``##contig`` header lines. This script writes a
   valid ``##fileformat=VCFv4.2`` header (contig lengths sourced from the
   reference FASTA, so names/lengths match the BAM), sorts, bgzips, and tabixes.

**All mutation types are ALWAYS kept** — SNVs, atomized MNP→SNVs, and indels —
so nothing snoopy called is dropped. There is no flag to disable indels
(invariant, matching strainphase core; see ``docs/MUTATION_HANDLING.md``).
Length-changing blocks are emitted as a single left-anchored
(shared-prefix-trimmed) del/ins record. Some snoopy indel blocks are *complex*
(substitution + length change with no shared anchor base); those are emitted
as-is and are canonicalized by piping this tool's output through
``bcftools norm -a -f REF`` downstream (the Snakemake ``snoopy_sample_vcf``
rule does exactly this).

Input (``--snoopy``) may be either:
  * a directory of SNooPy per-window/per-contig VCFs (e.g. the ``tmp/`` dir), or
  * a single aggregated SNooPy VCF (e.g. ``variants.vcf``).

Contig lengths for the header come from ``--ref`` (a FASTA; its ``.fai`` is built
if missing) or, for quick local checks, from ``--lengths-from`` pointing at a
BAM/BCF/VCF whose header carries ``@SQ`` / ``##contig`` records.

Usage
-----
    python -m strainphase.snoopy_to_vcf \
        --snoopy path/to/results/snoopy/SAMPLE \
        --ref combined_bins.fasta \
        --sample LS_3_13_16 \
        --output results/snoopy_vcf/variable/LS_3_13_16.vcf.gz
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from collections import defaultdict

import pysam

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("snoopy_to_vcf")

_BASES = set("ACGTNacgtn")


def _collect_snoopy_files(snoopy_path: str) -> list[str]:
    """Resolve a --snoopy argument to a list of SNooPy VCF files.

    Handles the layouts the pipeline actually produces:
      * a single VCF file (aggregated ``variants.vcf``);
      * a per-MAG output dir with ``variants.vcf`` (+ ``tmp/*.vcf`` fallback if
        the aggregate is empty because SNooPy didn't finish);
      * a per-SAMPLE dir ``results/snoopy/{sample}/`` containing one ``{mag}/``
        subdir per MAG (each with ``variants.vcf`` / ``tmp/``) — all MAGs are
        pooled into the one per-sample VCF strainphase's ``--vcfs`` expects.
    """
    if os.path.isfile(snoopy_path):
        return [snoopy_path]
    if not os.path.isdir(snoopy_path):
        raise SystemExit(f"--snoopy path not found: {snoopy_path}")

    def _dir_files(d: str) -> list[str]:
        agg = os.path.join(d, "variants.vcf")
        if os.path.isfile(agg) and os.path.getsize(agg) > 0:
            # Prefer the aggregate, but if it has no variant rows fall back to tmp.
            with open(agg) as fh:
                if any(ln.strip() and not ln.startswith("#") for ln in fh):
                    return [agg]
        tmp = sorted(glob.glob(os.path.join(d, "tmp", "*.vcf")))
        if tmp:
            return tmp
        return sorted(glob.glob(os.path.join(d, "*.vcf")))

    files = _dir_files(snoopy_path)
    if not files:
        # Treat as a per-sample dir: pool every immediate MAG subdirectory.
        for sub in sorted(glob.glob(os.path.join(snoopy_path, "*"))):
            if os.path.isdir(sub):
                files.extend(_dir_files(sub))
    if not files:
        raise SystemExit(f"No SNooPy VCF records found under {snoopy_path}")
    return files


def iter_snoopy_records(snoopy_path: str):
    """Yield ``(contig, pos, ref, alt, info)`` tuples from SNooPy output.

    Header lines (starting with ``#``) and blank lines are skipped. SNooPy
    bodies are tab-separated ``CHROM POS ID REF ALT QUAL FILTER INFO`` with no
    sample columns.
    """
    files = _collect_snoopy_files(snoopy_path)

    for fpath in files:
        with open(fpath) as fh:
            for line in fh:
                if not line.strip() or line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue
                contig, pos, ref, alt = p[0], int(p[1]), p[3], p[4]
                info = p[7] if len(p) > 7 else "."
                yield contig, pos, ref, alt, info


def _parse_dp(info: str) -> int | None:
    """Pull DP from a SNooPy INFO string (e.g. ``DP=6``)."""
    for field in info.split(";"):
        if field.startswith("DP="):
            try:
                return int(field[3:])
            except ValueError:
                return None
    return None


def atomize(contig, pos, ref, alt, dp):
    """Decompose one SNooPy record into atomic ``(pos, ref, alt, dp)`` variants.

    EVERY mutation type is kept — SNVs, MNP->SNVs, and indels (invariant; there
    is no flag to drop indels, matching strainphase core):

    * 1bp REF / 1bp ALT  -> passed through unchanged.
    * equal-length block -> one SNV per differing offset (identical bases dropped).
    * length-changing    -> a single left-anchored del/ins (shared prefix trimmed
                            to one anchor base; complex blocks with no shared
                            anchor are emitted as-is for ``bcftools norm -a``
                            downstream).
    """
    lr, la = len(ref), len(alt)
    out = []
    if lr == 1 and la == 1:
        if ref in _BASES and alt in _BASES and ref.upper() != alt.upper():
            out.append((pos, ref.upper(), alt.upper(), dp))
    elif lr == la:  # MNP: split into per-position SNVs
        for i in range(lr):
            rb, ab = ref[i], alt[i]
            if rb.upper() != ab.upper() and rb in _BASES and ab in _BASES:
                out.append((pos + i, rb.upper(), ab.upper(), dp))
    else:  # length-changing block -> indel (always kept)
        # Trim the common leading bases so the record is left-anchored, matching
        # the VCF indel convention strainphase's CIGAR matcher expects.
        shift = 0
        while shift < min(lr, la) and ref[shift].upper() == alt[shift].upper():
            shift += 1
        # Keep one anchor base before the indel (VCF requires a shared anchor).
        anchor = max(0, shift - 1)
        new_ref = ref[anchor:].upper()
        new_alt = alt[anchor:].upper()
        if new_ref != new_alt and set(new_ref) <= _BASES and set(new_alt) <= _BASES:
            out.append((pos + anchor, new_ref, new_alt, dp))
    return out


def load_contig_lengths(ref: str | None, lengths_from: str | None) -> dict[str, int]:
    """Return ``{contig: length}`` from a FASTA (.fai) or a BAM/BCF/VCF header."""
    lengths: dict[str, int] = {}
    if ref:
        fai = ref + ".fai"
        if not os.path.exists(fai):
            logger.info("Indexing reference (%s) ...", ref)
            pysam.faidx(ref)
        with open(fai) as fh:
            for line in fh:
                name, length = line.split("\t")[:2]
                lengths[name] = int(length)
    elif lengths_from:
        if lengths_from.endswith((".bam", ".cram")):
            with pysam.AlignmentFile(lengths_from, "rb") as bam:
                lengths = dict(zip(bam.references, bam.lengths))  # noqa: B905
        else:  # BCF / VCF header ##contig lines
            with pysam.VariantFile(lengths_from) as vf:
                lengths = {name: c.length for name, c in vf.header.contigs.items()}
    else:
        raise SystemExit("Provide either --ref or --lengths-from for contig lengths")
    return lengths


def convert(
    snoopy_path: str,
    output: str,
    contig_lengths: dict[str, int],
    check_ref: str | None = None,
) -> dict:
    """Convert SNooPy output to a bgzipped, tabix-indexed strainphase VCF.

    Keeps every mutation type (SNVs, MNP->SNVs, indels); nothing is dropped.
    """
    # (contig, pos) -> dict of {(ref, alt): dp}. A position with more than one
    # distinct ALT is emitted as a multi-allelic record; strainphase's load_snvs
    # keeps multi-allelic sites (invariant), and `bcftools norm -m -` downstream
    # splits them into biallelic records anyway.
    sites: dict[tuple[str, int], dict[tuple[str, str], int]] = defaultdict(dict)

    stats = defaultdict(int)
    unknown_contigs: set[str] = set()

    fasta = pysam.FastaFile(check_ref) if check_ref else None

    for contig, pos, ref, alt, info in iter_snoopy_records(snoopy_path):
        stats["raw_records"] += 1
        if contig not in contig_lengths:
            unknown_contigs.add(contig)
            continue
        dp = _parse_dp(info)
        for apos, aref, aalt, adp in atomize(contig, pos, ref, alt, dp):
            if apos < 1 or apos > contig_lengths[contig]:
                stats["out_of_bounds"] += 1
                continue
            if fasta is not None and len(aref) == 1:
                actual = fasta.fetch(contig, apos - 1, apos).upper()
                if actual and actual != aref:
                    stats["ref_mismatch"] += 1
            sites[(contig, apos)][(aref, aalt)] = (
                adp if adp is not None else sites[(contig, apos)].get((aref, aalt), 0)
            )
            stats["atomic_variants"] += 1

    if fasta is not None:
        fasta.close()

    # Order contigs as in the reference header, positions ascending.
    contig_order = {name: i for i, name in enumerate(contig_lengths)}
    ordered = sorted(sites, key=lambda k: (contig_order[k[0]], k[1]))

    # Emit only the contigs we actually wrote, in reference order.
    used_contigs = [c for c in contig_lengths if any(k[0] == c for k in sites)]

    raw_path = output[:-3] if output.endswith(".gz") else output + ".raw"
    header = [
        "##fileformat=VCFv4.2",
        "##source=snoopy_to_vcf.py",
    ]
    for c in used_contigs:
        header.append(f"##contig=<ID={c},length={contig_lengths[c]}>")
    header += [
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth (from SNooPy)">',
        '##FILTER=<ID=PASS,Description="All filters passed">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]

    n_written = 0
    n_multiallelic = 0
    with open(raw_path, "w") as out:
        out.write("\n".join(header) + "\n")
        for key in ordered:
            contig, apos = key
            alleles = sites[key]
            refs = {r for r, _ in alleles}
            if len(refs) != 1:
                # Inconsistent REF across atomized alleles at one position; skip
                # (should be vanishingly rare — indicates overlapping blocks).
                stats["ref_conflict"] += 1
                continue
            ref = next(iter(refs))
            alts = sorted({a for _, a in alleles})
            dp = max(v for v in alleles.values() if v is not None) if any(
                v is not None for v in alleles.values()
            ) else None
            if len(alts) > 1:
                n_multiallelic += 1
            info = f"DP={dp}" if dp is not None else "."
            out.write(
                f"{contig}\t{apos}\t.\t{ref}\t{','.join(alts)}\t.\tPASS\t{info}\n"
            )
            n_written += 1

    pysam.tabix_compress(raw_path, output, force=True)
    pysam.tabix_index(output, preset="vcf", force=True)
    os.remove(raw_path)

    stats["written_records"] = n_written
    stats["multiallelic_records"] = n_multiallelic
    stats["contigs_written"] = len(used_contigs)
    if unknown_contigs:
        stats["unknown_contigs"] = len(unknown_contigs)
        logger.warning(
            "%d contig(s) in SNooPy output not found in reference header (skipped), "
            "e.g. %s",
            len(unknown_contigs),
            sorted(unknown_contigs)[:3],
        )
    return dict(stats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snoopy", required=True, help="SNooPy tmp/ dir or aggregated VCF")
    ap.add_argument("--output", "-o", required=True, help="Output .vcf.gz path")
    ap.add_argument("--ref", help="Reference FASTA (source of contig lengths; .fai built if absent)")
    ap.add_argument("--lengths-from", help="Alt length source: BAM/CRAM or BCF/VCF header")
    ap.add_argument("--sample", help="Sample label (informational; not written into the VCF)")
    ap.add_argument("--check-ref", help="FASTA to validate SNooPy REF bases against (QC; reports mismatches)")
    args = ap.parse_args(argv)

    contig_lengths = load_contig_lengths(args.ref, args.lengths_from)
    logger.info("Loaded %d contig lengths", len(contig_lengths))

    stats = convert(
        args.snoopy,
        args.output,
        contig_lengths,
        check_ref=args.check_ref,
    )

    logger.info("Conversion summary for %s:", args.sample or args.snoopy)
    for k in (
        "raw_records",
        "atomic_variants",
        "written_records",
        "multiallelic_records",
        "contigs_written",
        "out_of_bounds",
        "ref_conflict",
        "ref_mismatch",
        "unknown_contigs",
    ):
        if k in stats:
            logger.info("  %-22s %d", k, stats[k])
    logger.info("Wrote %s (+ .tbi)", args.output)


if __name__ == "__main__":
    main()
