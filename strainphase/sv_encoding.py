#!/usr/bin/env python3
"""
Encode Sniffles2 structural variants as phaseable pseudo-SNVs for strainphase.

Each structural variant becomes a pseudo-site at its breakpoint anchor. Unlike a
plain SNV, the "present" allele is the event's UNIQUE ID (Sniffles' merged VCF
ID), not a generic INS/DEL token. This matters for lineage clustering: two reads
are grouped only if they carry the *identical* event. Collapsing different events
(or different sizes/types) to a shared token would let genuinely different
structural changes look the same and falsely merge into one lineage — so we do
NOT collapse. A read that spans the locus reference-like votes the matched
reference base ("absent"); otherwise it is a no-call.

Because the token is the event ID, a locus with two distinct events is handled
as a genuinely multi-allelic site (read supports event A -> "A", event B -> "B",
neither -> ref base). strainphase compares alleles by string equality, so
arbitrary event-ID tokens work without any alphabet change.

The per-read "present" decision is driven by Sniffles' supporting-read list
(``RNAMES``, requires ``--output-rnames``), not CIGAR matching — large SVs are
split reads that the strainphase CIGAR scan skips, and breakpoints are imprecise.

VERIFICATIONS built in
----------------------
1. RNAMES fail-fast: ``parse_sniffles`` errors if the VCF has no RNAMES at all
   (the ``--output-rnames`` flag silently didn't take effect), instead of
   emitting an empty sidecar.
2. Cross-sample consistency: ``verify`` subcommand asserts each event ID maps to
   a single (contig, pos) across all per-sample sidecars — required so the same
   event clusters into one lineage across timepoints.

CLI
---
    # Convert one Sniffles VCF -> one sidecar
    python -m strainphase.sv_encoding \
        --sniffles sample.sv.vcf.gz --out-sidecar sample.sv_sidecar.tsv \
        [--out-vcf markers.vcf] [--min-support 3] [--min-af 0.05] [--max-af 0.95] \
        [--svtypes INS,DEL,DUP,INV,BND] [--contigs contigs.txt]

    # Verify event-ID consistency across all sidecars (verification #2)
    python -m strainphase.sv_encoding verify --sidecars s1.tsv s2.tsv ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field

# svtype -> "del-like" or "ins-like", used ONLY to classify placeholder REF/ALT
# in the optional inspection VCF. It does NOT define the phasing allele (that is
# the unique event ID). All svtypes are retained as distinct events.
_SVTYPE_IS_DEL = {"DEL": True}
_DEFAULT_SVTYPES = ("INS", "DEL", "DUP", "INV", "BND")


@dataclass
class SVRecord:
    """One structural variant reduced to a phaseable, uniquely-labeled pseudo-site."""

    contig: str
    pos: int  # 1-based breakpoint anchor (VCF POS)
    event_id: str  # UNIQUE, cross-sample-stable event label = the phasing allele
    svtype: str  # original Sniffles SVTYPE (INS/DEL/DUP/INV/BND), for interpretation
    svlen: int
    af: float
    dr: int
    dv: int
    support_reads: set[str] = field(default_factory=set)


# ------------------------------------------------------------------ parsing --#


def _get_info(record, key, default=None):
    if key not in record.info:
        return default
    val = record.info[key]
    if isinstance(val, tuple):
        return val[0] if len(val) == 1 else val
    return val


def _sample_dr_dv(record):
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
    """Parse a Sniffles2 VCF into :class:`SVRecord`s (one per SV, uniquely labeled).

    Requires ``pysam`` and a VCF produced with ``--output-rnames``. Raises
    ``RuntimeError`` if RNAMES is entirely absent (verification #1).
    """
    try:
        import pysam
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("pysam is required to parse Sniffles VCFs") from exc

    svtypes = tuple(s.upper() for s in svtypes)
    records: list[SVRecord] = []
    n_records = 0
    n_no_rnames = 0
    n_filtered = 0

    vcf = pysam.VariantFile(vcf_path)
    header_has_rnames = "RNAMES" in vcf.header.info

    for rec in vcf.fetch():
        if rec.filter.keys() and "PASS" not in rec.filter.keys():
            n_filtered += 1
            continue

        raw_svtype = _get_info(rec, "SVTYPE")
        if raw_svtype is None or raw_svtype.upper() not in svtypes:
            continue
        raw_svtype = raw_svtype.upper()
        n_records += 1

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

        # Unique, cross-sample-stable label. After Sniffles population merge +
        # --genotype-vcf, rec.id is inherited from the merged VCF, so the SAME
        # event carries the SAME id in every sample (verified by `verify`).
        event_id = rec.id if rec.id and rec.id != "." else f"{raw_svtype}.{contig}.{rec.pos}.{svlen}"

        records.append(
            SVRecord(
                contig=contig,
                pos=int(rec.pos),
                event_id=event_id,
                svtype=raw_svtype,
                svlen=svlen,
                af=af,
                dr=dr,
                dv=dv,
                support_reads=support,
            )
        )

    vcf.close()

    # Verification #1: RNAMES must actually be present.
    if n_records > 0 and not header_has_rnames:
        raise RuntimeError(
            f"{vcf_path}: no RNAMES INFO field in header. Re-run Sniffles with "
            "--output-rnames (and, for --genotype-vcf, confirm your Sniffles "
            "version emits per-sample RNAMES at forced sites)."
        )
    if n_records > 0 and not records and n_no_rnames == n_records:
        raise RuntimeError(
            f"{vcf_path}: every SV record lacked RNAMES. Re-run Sniffles with "
            "--output-rnames; per-read SV assignment is impossible without it."
        )

    if n_no_rnames:
        logging.warning("%d SV records skipped for missing/empty RNAMES", n_no_rnames)
    if n_filtered:
        logging.info("%d non-PASS SV records skipped", n_filtered)
    logging.info("Parsed %d structural variants from %s", len(records), vcf_path)
    return records


# ------------------------------------------------------------------ writing --#

_SIDECAR_HEADER = "#contig\tpos\tevent_id\tsvtype\tsvlen\taf\tdr\tdv\tsupport_reads"


def write_sidecar(records: list[SVRecord], path: str) -> None:
    """Write SV pseudo-sites to a strainphase sidecar TSV."""
    with open(path, "w") as fh:
        fh.write(_SIDECAR_HEADER + "\n")
        for r in sorted(records, key=lambda x: (x.contig, x.pos, x.event_id)):
            fh.write(
                "\t".join(
                    [
                        r.contig,
                        str(r.pos),
                        r.event_id,
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


def write_synthetic_vcf(records: list[SVRecord], path: str) -> None:
    """Inspection-only synthetic VCF (one biallelic record per SV). Sequence
    content is a placeholder; the phasing allele is the event ID in the sidecar."""
    contigs = sorted({r.contig for r in records})
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write('##INFO=<ID=SVMARK,Number=0,Type=Flag,Description="strainphase SV pseudo-site">\n')
        fh.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Sniffles SVTYPE">\n')
        fh.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="DR+DV at breakpoint">\n')
        fh.write('##INFO=<ID=AF,Number=1,Type=Float,Description="DV/(DR+DV)">\n')
        for c in contigs:
            fh.write(f"##contig=<ID={c}>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for r in sorted(records, key=lambda x: (x.contig, x.pos, x.event_id)):
            ref, alt = ("NN", "N") if _SVTYPE_IS_DEL.get(r.svtype) else ("N", "NN")
            info = f"SVMARK;SVTYPE={r.svtype};DP={r.dr + r.dv};AF={r.af:.4f}"
            fh.write(f"{r.contig}\t{r.pos}\t{r.event_id}\t{ref}\t{alt}\t.\tPASS\t{info}\n")
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
            if len(parts) < 9:
                continue
            contig, pos, event_id, svtype, svlen, af, dr, dv, support = parts[:9]
            rec = SVRecord(
                contig=contig,
                pos=int(pos),
                event_id=event_id,
                svtype=svtype.upper(),
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
) -> tuple[list[int], dict[int, str], dict[int, str], dict[int, dict[str, set[str]]]]:
    """Load SV pseudo-sites for one contig, ready to merge into the variant set.

    Returns
    -------
    sv_pos
        Sorted, de-duplicated list of anchor positions.
    sv_ref_alleles
        Placeholder REF base per position (``"N"``).
    sv_site_type
        Per-position type marker, always ``"sv"``.
    sv_support
        Per-position map ``{event_id: {read_name, ...}}``. Multiple events at one
        anchor are preserved (multi-allelic site).
    """
    grouped = _load_sidecar_grouped(path)
    recs = grouped.get(contig_id, [])
    sv_ref_alleles: dict[int, str] = {}
    sv_site_type: dict[int, str] = {}
    sv_support: dict[int, dict[str, set[str]]] = {}
    for r in sorted(recs, key=lambda x: (x.pos, x.event_id)):
        sv_ref_alleles[r.pos] = "N"
        sv_site_type[r.pos] = "sv"
        sv_support.setdefault(r.pos, {})[r.event_id] = r.support_reads
    sv_pos = sorted(sv_support.keys())
    return sv_pos, sv_ref_alleles, sv_site_type, sv_support


# -------------------------------------------------------------- verification -#


def check_event_consistency(sidecar_paths: list[str]) -> list[str]:
    """Verification #2: each event ID must map to exactly ONE (contig, pos)
    across all sidecars, else the same event won't cluster into one lineage
    across timepoints. Returns a list of human-readable violation messages
    (empty if consistent)."""
    loci_by_event: dict[str, set[tuple[str, int]]] = {}
    for path in sidecar_paths:
        for contig, recs in _load_sidecar_grouped(path).items():
            for r in recs:
                loci_by_event.setdefault(r.event_id, set()).add((contig, r.pos))
    violations = []
    for event_id, loci in sorted(loci_by_event.items()):
        if len(loci) > 1:
            locs = ", ".join(f"{c}:{p}" for c, p in sorted(loci))
            violations.append(f"event {event_id} maps to multiple loci: {locs}")
    return violations


# --------------------------------------------------------------------- CLI --#


def _convert_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="strainphase.sv_encoding")
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


def _verify_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="strainphase.sv_encoding verify")
    p.add_argument("--sidecars", required=True, nargs="+", help="Sidecar TSVs to cross-check")
    p.add_argument("--out", help="Optional file to touch on success (Snakemake sentinel)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    violations = check_event_consistency(args.sidecars)
    if violations:
        for v in violations:
            logging.error("SV consistency: %s", v)
        logging.error(
            "%d event(s) drift across samples — anchors are not harmonized; "
            "check the Sniffles population-merge / genotype step.",
            len(violations),
        )
        return 1
    logging.info("SV consistency OK across %d sidecars", len(args.sidecars))
    if args.out:
        open(args.out, "w").close()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "verify":
        return _verify_main(argv[1:])
    return _convert_main(argv)


if __name__ == "__main__":
    sys.exit(main())
