#!/usr/bin/env python3
"""
Encode Sniffles2 structural variants as phaseable pseudo-SNVs for strainphase.

A structural variant (MGE/prophage insertion, HGT junction, large deletion,
duplication, inversion, breakend) is reduced to a *biallelic* marker at its
breakpoint anchor: a read either ``present`` (carries the event) or ``absent``
(spans the locus reference-like). strainphase then co-phases this marker with
real SNVs using its existing indel alphabet — ``present`` reuses the
``"INS"``/``"DEL"`` state, ``absent`` is the matched reference base. No change
to the model alphabet is required.

The per-read ``present`` decision is driven by **Sniffles' supporting-read
list** (``RNAMES``, requires running Sniffles with ``--output-rnames``), not by
CIGAR matching. Large SVs manifest as split reads (soft-clip + supplementary
alignment), which the strainphase CIGAR scan skips; trusting Sniffles' read
list sidesteps that entirely. See ``future_work.md`` for the full rationale and
caveats.

Outputs
-------
1. A **sidecar TSV** (the source consumed by ``process_contig``):

       #contig  pos  svtype  svlen  af  dr  dv  support_reads

   ``svtype`` is normalized to ``ins`` or ``del`` (the phasing token);
   ``support_reads`` is a comma-separated list of supporting read names.

2. An optional **synthetic VCF** (``--out-vcf``) for inspection only, with each
   SV as a biallelic INS/DEL record carrying ``INFO/SVMARK=1``, ``DP``, ``AF``.

CLI
---
    python -m strainphase.sv_encoding \
        --sniffles sniffles.vcf.gz \
        --out-sidecar sv_sidecar.tsv \
        [--out-vcf sv_markers.vcf] \
        [--min-support 3] [--min-af 0.05] [--max-af 0.95] \
        [--svtypes INS,DEL,DUP,INV,BND] [--contigs contigs.txt]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field

# svtype -> phasing token. The token is only a distinct non-ACGT allele state;
# it carries no size/orientation meaning (see caveat 6 in future_work.md).
_SVTYPE_TO_TOKEN = {
    "INS": "ins",
    "DUP": "ins",
    "INV": "ins",
    "BND": "ins",
    "DEL": "del",
}
_DEFAULT_SVTYPES = ("INS", "DEL", "DUP", "INV", "BND")


@dataclass
class SVRecord:
    """One structural variant reduced to a phaseable pseudo-site."""

    contig: str
    pos: int  # 1-based breakpoint anchor (VCF POS)
    svtype: str  # normalized phasing token: "ins" or "del"
    raw_svtype: str  # original Sniffles SVTYPE (INS/DEL/DUP/INV/BND)
    svlen: int
    af: float
    dr: int
    dv: int
    support_reads: set[str] = field(default_factory=set)


# ------------------------------------------------------------------ parsing --#


def _get_info(record, key, default=None):
    """Fetch an INFO value from a pysam VariantRecord, unwrapping 1-tuples."""
    if key not in record.info:
        return default
    val = record.info[key]
    if isinstance(val, tuple):
        return val[0] if len(val) == 1 else val
    return val


def _sample_dr_dv(record):
    """Return (DR, DV) from the first sample's FORMAT fields, or (None, None)."""
    if not record.samples:
        return None, None
    smp = record.samples[list(record.samples)[0]]
    dr = smp.get("DR")
    dv = smp.get("DV")
    if isinstance(dr, tuple):
        dr = dr[0] if dr else None
    if isinstance(dv, tuple):
        dv = dv[0] if dv else None
    return dr, dv


def parse_sniffles(
    vcf_path: str,
    min_support: int = 3,
    min_af: float = 0.0,
    max_af: float = 1.0,
    svtypes: tuple[str, ...] = _DEFAULT_SVTYPES,
    allowed_contigs: set[str] | None = None,
) -> list[SVRecord]:
    """Parse a Sniffles2 VCF into a list of :class:`SVRecord`.

    Requires ``pysam``. The VCF should be produced with ``--output-rnames`` so
    ``INFO/RNAMES`` is populated; records without RNAMES are skipped with a
    warning count (there is nothing to assign per-read).
    """
    try:
        import pysam
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("pysam is required to parse Sniffles VCFs") from exc

    svtypes = tuple(s.upper() for s in svtypes)
    records: list[SVRecord] = []
    n_no_rnames = 0
    n_filtered = 0

    vcf = pysam.VariantFile(vcf_path)
    for rec in vcf.fetch():
        if rec.filter.keys() and "PASS" not in rec.filter.keys():
            n_filtered += 1
            continue

        raw_svtype = _get_info(rec, "SVTYPE")
        if raw_svtype is None or raw_svtype.upper() not in svtypes:
            continue
        token = _SVTYPE_TO_TOKEN.get(raw_svtype.upper())
        if token is None:
            continue

        contig = rec.contig
        if allowed_contigs is not None and contig not in allowed_contigs:
            continue

        svlen = _get_info(rec, "SVLEN", 0)
        try:
            svlen = abs(int(svlen)) if svlen is not None else 0
        except (TypeError, ValueError):
            svlen = 0

        dr, dv = _sample_dr_dv(rec)
        if dv is None:
            dv = _get_info(rec, "SUPPORT", 0) or 0
        if dr is None:
            dr = 0
        dr, dv = int(dr), int(dv)

        if dv < min_support:
            continue

        denom = dr + dv
        af = (dv / denom) if denom > 0 else 0.0
        if not (min_af <= af <= max_af):
            continue

        rnames = _get_info(rec, "RNAMES")
        if rnames is None:
            n_no_rnames += 1
            continue
        if isinstance(rnames, str):
            rnames = rnames.split(",")
        support = {r for r in rnames if r}
        if not support:
            n_no_rnames += 1
            continue

        records.append(
            SVRecord(
                contig=contig,
                pos=int(rec.pos),
                svtype=token,
                raw_svtype=raw_svtype.upper(),
                svlen=svlen,
                af=af,
                dr=dr,
                dv=dv,
                support_reads=support,
            )
        )

    vcf.close()

    if n_no_rnames:
        logging.warning(
            "%d SV records skipped for missing/empty RNAMES — run Sniffles with "
            "--output-rnames",
            n_no_rnames,
        )
    if n_filtered:
        logging.info("%d non-PASS SV records skipped", n_filtered)
    logging.info("Parsed %d structural variants from %s", len(records), vcf_path)
    return records


# ------------------------------------------------------------------ writing --#

_SIDECAR_HEADER = "#contig\tpos\tsvtype\tsvlen\taf\tdr\tdv\tsupport_reads"


def write_sidecar(records: list[SVRecord], path: str) -> None:
    """Write SV pseudo-sites to a strainphase sidecar TSV."""
    with open(path, "w") as fh:
        fh.write(_SIDECAR_HEADER + "\n")
        for r in sorted(records, key=lambda x: (x.contig, x.pos)):
            fh.write(
                "\t".join(
                    [
                        r.contig,
                        str(r.pos),
                        r.svtype,
                        str(r.svlen),
                        f"{r.af:.4f}",
                        str(r.dr),
                        str(r.dv),
                        ",".join(sorted(r.support_reads)),
                    ]
                )
                + "\n"
            )
    logging.info("Wrote %d SV pseudo-sites to sidecar %s", len(records), path)


def write_synthetic_vcf(records: list[SVRecord], path: str, reference: str | None = None) -> None:
    """Write an inspection-only synthetic VCF (one biallelic INS/DEL per SV).

    Sequence content is a placeholder; the phasing allele is taken from the
    sidecar, not from these REF/ALT strings. Included so SV markers can be
    viewed/indexed alongside the SNV VCF (``INFO/SVMARK=1``).
    """
    contigs = sorted({r.contig for r in records})
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write('##INFO=<ID=SVMARK,Number=0,Type=Flag,Description="strainphase SV pseudo-site">\n')
        fh.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Original Sniffles SVTYPE">\n')
        fh.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="DR+DV at breakpoint">\n')
        fh.write('##INFO=<ID=AF,Number=1,Type=Float,Description="DV/(DR+DV)">\n')
        for c in contigs:
            fh.write(f"##contig=<ID={c}>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for r in sorted(records, key=lambda x: (x.contig, x.pos)):
            # Placeholder alleles that classify as ins/del by length.
            if r.svtype == "del":
                ref, alt = "NN", "N"
            else:
                ref, alt = "N", "NN"
            dp = r.dr + r.dv
            info = f"SVMARK;SVTYPE={r.raw_svtype};DP={dp};AF={r.af:.4f}"
            fh.write(
                f"{r.contig}\t{r.pos}\tsv_{r.contig}_{r.pos}\t{ref}\t{alt}\t.\tPASS\t{info}\n"
            )
    logging.info("Wrote %d SV markers to synthetic VCF %s", len(records), path)


# ----------------------------------------------------------------- loading --#

# Cache parsed sidecars so per-(sample,contig) process_contig calls don't
# re-read the whole file. Keyed by sidecar path.
_SIDECAR_CACHE: dict[str, dict[str, list[SVRecord]]] = {}


def _load_sidecar_grouped(path: str) -> dict[str, list[SVRecord]]:
    """Parse a sidecar TSV into {contig: [SVRecord, ...]} (cached)."""
    if path in _SIDECAR_CACHE:
        return _SIDECAR_CACHE[path]
    by_contig: dict[str, list[SVRecord]] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            contig, pos, svtype, svlen, af, dr, dv, support = parts[:8]
            rec = SVRecord(
                contig=contig,
                pos=int(pos),
                svtype=svtype,
                raw_svtype=svtype.upper(),
                svlen=int(svlen) if svlen else 0,
                af=float(af) if af else 0.0,
                dr=int(dr) if dr else 0,
                dv=int(dv) if dv else 0,
                support_reads={s for s in support.split(",") if s},
            )
            by_contig.setdefault(contig, []).append(rec)
    _SIDECAR_CACHE[path] = by_contig
    return by_contig


def load_sv_sidecar_for_contig(
    path: str, contig_id: str
) -> tuple[list[int], dict[int, str], dict[int, str], dict[int, set[str]]]:
    """Load SV pseudo-sites for one contig, ready to merge into the variant set.

    Returns
    -------
    sv_pos
        Sorted list of anchor positions.
    sv_ref_alleles
        Placeholder REF base per position (``"N"``).
    sv_site_type
        Per-position ``"sv_ins"`` / ``"sv_del"``.
    sv_support
        Per-position set of supporting read names.
    """
    grouped = _load_sidecar_grouped(path)
    recs = grouped.get(contig_id, [])
    sv_pos: list[int] = []
    sv_ref_alleles: dict[int, str] = {}
    sv_site_type: dict[int, str] = {}
    sv_support: dict[int, set[str]] = {}
    for r in sorted(recs, key=lambda x: x.pos):
        sv_pos.append(r.pos)
        sv_ref_alleles[r.pos] = "N"
        sv_site_type[r.pos] = "sv_del" if r.svtype == "del" else "sv_ins"
        sv_support[r.pos] = r.support_reads
    return sv_pos, sv_ref_alleles, sv_site_type, sv_support


# --------------------------------------------------------------------- CLI --#


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="strainphase.sv_encoding",
        description="Encode Sniffles2 SVs as phaseable pseudo-SNVs for strainphase.",
    )
    p.add_argument("--sniffles", required=True, help="Sniffles2 VCF (run with --output-rnames)")
    p.add_argument("--out-sidecar", required=True, help="Output sidecar TSV")
    p.add_argument("--out-vcf", help="Optional inspection-only synthetic VCF")
    p.add_argument("--min-support", type=int, default=3, help="Min supporting reads (DV) [3]")
    p.add_argument("--min-af", type=float, default=0.05, help="Min AF = DV/(DR+DV) [0.05]")
    p.add_argument("--max-af", type=float, default=0.95, help="Max AF [0.95]")
    p.add_argument(
        "--svtypes",
        default=",".join(_DEFAULT_SVTYPES),
        help="Comma-separated SVTYPEs to keep [INS,DEL,DUP,INV,BND]",
    )
    p.add_argument("--contigs", help="Optional file listing contigs to keep (one per line)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    allowed = None
    if args.contigs:
        with open(args.contigs) as fh:
            allowed = {ln.strip() for ln in fh if ln.strip()}

    svtypes = tuple(s.strip().upper() for s in args.svtypes.split(",") if s.strip())

    records = parse_sniffles(
        args.sniffles,
        min_support=args.min_support,
        min_af=args.min_af,
        max_af=args.max_af,
        svtypes=svtypes,
        allowed_contigs=allowed,
    )

    write_sidecar(records, args.out_sidecar)
    if args.out_vcf:
        write_synthetic_vcf(records, args.out_vcf)

    return 0


if __name__ == "__main__":
    sys.exit(main())
