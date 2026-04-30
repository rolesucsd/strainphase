"""Helpers for building tiny VCFs, FASTAs, and BAMs in unit tests.

Each helper writes a minimal, valid file under a caller-provided ``tmp_path``
(typically a ``pytest`` ``tmp_path`` fixture or a ``tempfile.TemporaryDirectory``)
and returns the path. Files are indexed where applicable (``.fai``, ``.bai``,
``.tbi``) so they're ready to be opened by ``pysam``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pysam


# ---------------------------------------------------------------------------
# FASTA
# ---------------------------------------------------------------------------


def write_fasta(tmp_path, contigs: dict[str, str], name: str = "ref.fa") -> str:
    """Write a FASTA with the given ``{contig: sequence}`` and index it."""
    path = Path(tmp_path) / name
    with open(path, "w") as fh:
        for contig, seq in contigs.items():
            fh.write(f">{contig}\n{seq}\n")
    pysam.faidx(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# VCF
# ---------------------------------------------------------------------------


def write_vcf(
    tmp_path,
    contig: str,
    records: list[dict],
    samples: list[str] | None = None,
    name: str = "test.vcf.gz",
    contig_length: int = 100_000,
) -> str:
    """Write a tiny VCF (bgzipped + tabix-indexed) and return its path.

    Parameters
    ----------
    contig
        The contig name to put in the header.
    records
        Each record is a dict with keys: ``pos`` (1-based), ``ref``, ``alt``
        (string or list of strings), and optional ``filter`` (default
        ``"PASS"``), ``info`` (dict), ``samples_fmt`` (per-sample dict).
    samples
        Sample names. If ``None``, no FORMAT/sample columns are emitted.
    """
    path = Path(tmp_path) / name
    samples = samples or []

    header_lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={contig},length={contig_length}>",
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">',
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">',
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##FILTER=<ID=LowQual,Description="Low quality">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read depth">',
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">',
    ]
    cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
    if samples:
        cols += ["FORMAT"] + list(samples)
    header_lines.append("\t".join(cols))

    body = []
    for r in records:
        alts = r["alt"] if isinstance(r["alt"], list) else [r["alt"]]
        info_parts = []
        info = r.get("info", {})
        if "DP" in info:
            info_parts.append(f"DP={info['DP']}")
        if "AF" in info:
            af_val = info["AF"]
            if isinstance(af_val, (list, tuple)):
                info_parts.append("AF=" + ",".join(str(a) for a in af_val))
            else:
                info_parts.append(f"AF={af_val}")
        info_str = ";".join(info_parts) if info_parts else "."

        row = [
            contig,
            str(r["pos"]),
            r.get("id", "."),
            r["ref"],
            ",".join(alts),
            r.get("qual", "."),
            r.get("filter", "PASS"),
            info_str,
        ]
        if samples:
            sfmt = r.get("samples_fmt", {})
            fmt_keys = ["GT"]
            for sample in samples:
                if sample in sfmt and "DP" in sfmt[sample]:
                    if "DP" not in fmt_keys:
                        fmt_keys.append("DP")
                if sample in sfmt and "AD" in sfmt[sample]:
                    if "AD" not in fmt_keys:
                        fmt_keys.append("AD")
            row.append(":".join(fmt_keys))
            for sample in samples:
                vals = []
                sd = sfmt.get(sample, {})
                for key in fmt_keys:
                    if key == "GT":
                        vals.append(sd.get("GT", "1"))
                    elif key == "DP":
                        vals.append(str(sd.get("DP", ".")))
                    elif key == "AD":
                        ad = sd.get("AD", ".")
                        if isinstance(ad, (list, tuple)):
                            vals.append(",".join(str(v) for v in ad))
                        else:
                            vals.append(str(ad))
                row.append(":".join(vals))
        body.append("\t".join(row))

    raw = Path(tmp_path) / (name.removesuffix(".gz") if name.endswith(".gz") else name + ".raw")
    with open(raw, "w") as fh:
        fh.write("\n".join(header_lines) + "\n")
        if body:
            fh.write("\n".join(body) + "\n")
    pysam.tabix_compress(str(raw), str(path), force=True)
    pysam.tabix_index(str(path), preset="vcf", force=True)
    os.unlink(raw)
    return str(path)


# ---------------------------------------------------------------------------
# BAM
# ---------------------------------------------------------------------------


# Pysam CIGAR op codes
_OP = {"M": 0, "I": 1, "D": 2, "N": 3, "S": 4, "H": 5, "P": 6, "=": 7, "X": 8}


def cigar_str_to_tuples(cigar: str) -> list[tuple[int, int]]:
    """Parse a CIGAR like '5M2D3M' into pysam (op, length) tuples."""
    tuples = []
    num = ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            if ch not in _OP:
                raise ValueError(f"Unknown CIGAR op: {ch!r}")
            tuples.append((_OP[ch], int(num)))
            num = ""
    return tuples


def write_bam(
    tmp_path,
    contig: str,
    contig_length: int,
    reads: list[dict],
    name: str = "test.bam",
) -> str:
    """Write a tiny coordinate-sorted, indexed BAM.

    Each read dict supports:
        name, start (0-based), cigar (str), seq, mapq (default 60),
        quals (list[int] | None; default Q30 per base), flag (default 0).
    """
    path = Path(tmp_path) / name
    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"LN": contig_length, "SN": contig}],
    }

    # Sort by start so the BAM is coordinate-sorted on first write.
    reads_sorted = sorted(reads, key=lambda r: r["start"])

    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for i, r in enumerate(reads_sorted):
            seg = pysam.AlignedSegment(bam.header)
            seg.query_name = r.get("name", f"r{i}")
            seg.flag = r.get("flag", 0)
            seg.reference_id = 0
            seg.reference_start = r["start"]
            seg.mapping_quality = r.get("mapq", 60)
            seg.cigartuples = cigar_str_to_tuples(r["cigar"])
            seg.query_sequence = r["seq"]
            quals = r.get("quals")
            if quals is None:
                quals = [30] * len(r["seq"])
            seg.query_qualities = pysam.qualitystring_to_array(
                "".join(chr(33 + q) for q in quals)
            )
            bam.write(seg)

    pysam.index(str(path))
    return str(path)
