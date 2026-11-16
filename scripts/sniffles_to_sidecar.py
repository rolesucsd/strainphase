#!/usr/bin/env python3
"""Adapter: Sniffles2 VCF -> strainphase SV sidecar TSV.

Not part of the installed strainphase package: the package is caller-agnostic and
only defines and consumes the sidecar format. This adapter is one way (Sniffles2)
to produce a conforming sidecar; swap it for a cuteSV/SVIM adapter without touching
strainphase. It imports only the format API (SVRecord, write_sidecar) from the
package, so the sidecar spec stays authoritative there.

Requires the VCF to be produced with `--output-rnames` (per-read support is what
makes a read "present" at a pseudo-site). Errors loudly if RNAMES is absent, and
skips records that pysam cannot parse (rather than dying on a malformed VCF).

    python scripts/sniffles_to_sidecar.py --sniffles s.sv.vcf.gz --out-sidecar s.tsv \
        [--contigs keep.txt] [--min-support 3] [--min-af 0.05] [--max-af 0.95] \
        [--svtypes INS,DEL,DUP,INV,BND]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter

from strainphase.sv_encoding import SVRecord, write_sidecar


class Ledger:
    """Every record that does not become a sidecar row, by reason."""

    def __init__(self) -> None:
        self.reasons: Counter = Counter()

    def drop(self, reason, contig, pos, svtype, event_id, detail="") -> None:
        self.reasons[reason] += 1
        logging.debug("drop %s\t%s\t%s\t%s\t%s\t%s", reason, contig, pos, svtype, event_id, detail)

    @property
    def total(self) -> int:
        return sum(self.reasons.values())

    def report(self) -> None:
        for reason, n in sorted(self.reasons.items()):
            logging.info("  not written: %-28s %d", reason, n)

_DEFAULT_SVTYPES = ("INS", "DEL", "DUP", "INV", "BND")


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
    dr, dv = smp.get("DR"), smp.get("DV")
    if isinstance(dr, tuple):
        dr = dr[0] if dr else None
    if isinstance(dv, tuple):
        dv = dv[0] if dv else None
    return dr, dv


def parse_sniffles(vcf_path, led, min_support=3, min_af=0.0, max_af=1.0,
                   svtypes=_DEFAULT_SVTYPES, allowed_contigs=None):
    """Sniffles VCF -> SVRecord list. EVERY record that does not become a sidecar
    row is counted in the ledger with a reason (logged per record at DEBUG), and the
    run fails if records seen != rows written + records accounted for."""
    import pysam

    svtypes = tuple(s.upper() for s in svtypes)
    records, n_records, n_no_rnames, n_bad = [], 0, 0, 0
    n_seen = 0
    vcf = pysam.VariantFile(vcf_path)
    header_has_rnames = "RNAMES" in vcf.header.info

    it = vcf.fetch()
    while True:
        try:
            rec = next(it)
        except StopIteration:
            break
        except (OSError, ValueError) as exc:  # malformed record — skip, don't die
            n_bad += 1
            led.drop("unparseable_vcf_record", ".", ".", ".", ".", str(exc)[:120])
            logging.warning("skipping unparseable VCF record: %s", exc)
            continue

        n_seen += 1
        if rec.filter.keys() and "PASS" not in rec.filter.keys():
            led.drop("sniffles_filter_not_PASS", rec.contig, rec.pos,
                     _get_info(rec, "SVTYPE") or ".", rec.id or ".",
                     ",".join(rec.filter.keys()))
            continue
        raw = _get_info(rec, "SVTYPE")
        if raw is None or raw.upper() not in svtypes:
            led.drop("svtype_not_requested", rec.contig, rec.pos, str(raw),
                     rec.id or ".", f"allowed={','.join(svtypes)}")
            continue
        raw = raw.upper()
        n_records += 1
        contig = rec.contig
        if allowed_contigs is not None and contig not in allowed_contigs:
            led.drop("contig_not_in_keep_list", contig, rec.pos, raw, rec.id or ".", "")
            continue

        svlen = _get_info(rec, "SVLEN", 0)
        try:
            svlen = abs(int(svlen)) if svlen is not None else 0
        except (TypeError, ValueError):
            svlen = 0

        dr, dv = _sample_dr_dv(rec)
        dv = int(dv if dv is not None else (_get_info(rec, "SUPPORT", 0) or 0))
        dr = int(dr if dr is not None else 0)
        if dv < min_support:
            led.drop("below_min_support", contig, rec.pos, raw, rec.id or ".",
                     f"dv={dv} min_support={min_support}")
            continue
        denom = dr + dv
        af = (dv / denom) if denom > 0 else 0.0
        if not (min_af <= af <= max_af):
            led.drop("af_outside_band", contig, rec.pos, raw, rec.id or ".",
                     f"af={af:.4f} band=[{min_af},{max_af}]")
            continue

        rnames = _get_info(rec, "RNAMES")
        if rnames is None:
            n_no_rnames += 1
            led.drop("no_RNAMES", contig, rec.pos, raw, rec.id or ".", f"dv={dv}")
            continue
        if isinstance(rnames, str):
            rnames = rnames.split(",")
        support = {r for r in rnames if r}
        if not support:
            n_no_rnames += 1
            led.drop("empty_RNAMES", contig, rec.pos, raw, rec.id or ".", f"dv={dv}")
            continue

        event_id = rec.id if rec.id and rec.id != "." else f"{raw}.{contig}.{rec.pos}.{svlen}"
        records.append(SVRecord(contig, int(rec.pos), event_id, raw, svlen, af, dr, dv, support))
    vcf.close()

    if n_records > 0 and not header_has_rnames:
        raise RuntimeError(
            f"{vcf_path}: no RNAMES INFO field. Re-run Sniffles with --output-rnames "
            "(and confirm your version emits per-sample RNAMES if you force-genotype)."
        )
    if n_records > 0 and not records and n_no_rnames == n_records:
        raise RuntimeError(f"{vcf_path}: every SV record lacked RNAMES — cannot assign per-read support.")
    if n_bad:
        logging.warning("%d malformed VCF records skipped in %s", n_bad, vcf_path)
    if n_no_rnames:
        logging.warning("%d records skipped for missing/empty RNAMES", n_no_rnames)
    logging.info("Parsed %d structural variants from %s", len(records), vcf_path)
    if n_seen + n_bad != len(records) + led.total:
        raise RuntimeError(
            f"{vcf_path}: {n_seen + n_bad} records read but {len(records)} written + "
            f"{led.total} accounted for"
        )
    return records


def main(argv=None):
    p = argparse.ArgumentParser(prog="sniffles_to_sidecar")
    p.add_argument("--sniffles", required=True)
    p.add_argument("--out-sidecar", required=True)
    p.add_argument("--min-support", type=int, default=3)
    # No AF band by default: the SNV path deliberately keeps the full 0..1 range
    # so sweep-to-fixation is preserved; SVs must match, or an SV that fixes
    # (AF>0.95) or drops out (AF<0.05) at a timepoint would vanish from the sidecar.
    p.add_argument("--min-af", type=float, default=0.0)
    p.add_argument("--max-af", type=float, default=1.0)
    p.add_argument("--svtypes", default=",".join(_DEFAULT_SVTYPES))
    p.add_argument("--contigs")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    allowed = None
    if args.contigs:
        with open(args.contigs) as fh:
            allowed = {ln.strip() for ln in fh if ln.strip()}
    svtypes = tuple(s.strip().upper() for s in args.svtypes.split(",") if s.strip())

    led = Ledger()
    records = parse_sniffles(args.sniffles, led, min_support=args.min_support,
                             min_af=args.min_af, max_af=args.max_af,
                             svtypes=svtypes, allowed_contigs=allowed)
    write_sidecar(records, args.out_sidecar)
    led.report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
