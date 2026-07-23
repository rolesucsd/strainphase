#!/usr/bin/env python3
"""
Structural-variant input for strainphase: the SV SIDECAR format.

strainphase's SV interface is ONE documented, caller-agnostic format — the
sidecar TSV. strainphase consumes it via ``load_sv_sidecar_for_contig``; how you
produce it is up to you. Caller-specific readers (e.g. Sniffles2) live in the
PIPELINE, not here — the package stays tool-agnostic. Any caller (Sniffles,
cuteSV, SVIM, hand-made) is fine as long as the output conforms to the format
below; a reference Sniffles adapter ships in the pipeline as
``scripts/sniffles_to_sidecar.py`` and imports ``SVRecord`` / ``write_sidecar``
from here so this spec stays authoritative.

SIDECAR FORMAT (tab-separated, one row per event per sample)
------------------------------------------------------------
    #contig  pos  event_id  svtype  svlen  af  dr  dv  support_reads

  contig         reference contig name (must match the phasing BAM/reference)
  pos            1-based breakpoint anchor
  event_id       UNIQUE, cross-sample-STABLE label for the event = the phasing
                 allele. Two reads cluster into one lineage iff they carry the
                 SAME event_id, so the SAME real event MUST carry the SAME id (and
                 the SAME pos) in every sample. Distinct events MUST differ.
  svtype         INS/DEL/DUP/INV/BND (kept for interpretation; NOT the allele)
  svlen          event length (0 for BND)
  af, dr, dv     allele freq, ref-support, variant-support (metadata)
  support_reads  comma-separated read names that carry the event; MUST match the
                 BAM query names. This is what makes a read "present" at the site.

Encoding rationale: the allele is the event ID (never a generic INS/DEL token),
so a locus with two distinct events is a genuinely multi-allelic site and
different structural changes never falsely merge. A read that spans the anchor
reference-like votes the matched base ("absent"); one that doesn't span → no-call.

TOOLS (all operate on the standard sidecar; only --sniffles is caller-specific)
-------------------------------------------------------------------------------
  reconcile  Harmonize breakpoint DRIFT: cluster events within +/-pos_tol with
             concordant type/length into one canonical (id, pos), so one real
             event isn't split into several alleles across samples. Never merges
             two events from the same sample. Run this BEFORE phasing.
  verify     Assert each event_id maps to a single (contig, pos) across sidecars
             (guaranteed to pass after reconcile).

CLI (both operate on the standard sidecar; producing sidecars is the pipeline's job)
------------------------------------------------------------------------------------
    python -m strainphase.sv_encoding reconcile --sidecars *.tsv --out-dir recon/ \
        [--pos-tol 50] [--len-tol 0.25]
    python -m strainphase.sv_encoding verify --sidecars recon/*.tsv [--out ok]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field

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


# ------------------------------------------------------------- reconciliation -#
# Allele identity == event-ID identity, so breakpoint drift (one real event
# called with slightly different anchors / IDs across samples) makes strainphase
# see several alleles and under-cluster. reconcile harmonizes drifted events
# into one canonical (id, pos) BEFORE phasing. It operates on the standard
# sidecar format alone — caller-agnostic.


def _svtypes_concordant(t1: str, l1: int, t2: str, l2: int, len_tol_frac: float) -> bool:
    """Same SV type, and (for length-bearing types) concordant SVLEN."""
    if t1 != t2:
        return False
    if t1 in ("BND", "INV"):  # length not meaningful
        return True
    return abs(l1 - l2) <= len_tol_frac * max(l1, l2, 1)


# A reconciled cluster's total position span is capped at this many bp. It MUST
# stay <= core._SV_ANCHOR_PAD (the phasing span-bracket tolerance): with the cap,
# the canonical (median) anchor is within max_span of every member, so a read
# that bracketed a member still brackets the canonical anchor — reconcile only
# DECLINES to merge when the cap would be exceeded; it NEVER drops a read.
_RECONCILE_MAX_SPAN = 50


def reconcile_events(
    sidecar_paths: list[str],
    pos_tol: int = 50,
    len_tol_frac: float = 0.25,
    max_span: int = _RECONCILE_MAX_SPAN,
) -> tuple[dict[str, tuple[str, int]], dict[str, int]]:
    """Cluster drifted events across sidecars into a canonical (id, pos).

    Two events reconcile iff: same contig, |Δpos| <= pos_tol, same SVTYPE, and
    concordant SVLEN. A merge is DECLINED (events kept separate) when it would:
      - collapse two events that co-occur in the SAME sample (a real
        multi-allelic site), or
      - grow the cluster's total position span beyond ``max_span``.
    The span cap makes reconciliation SAFE-BY-CONSTRUCTION: the canonical anchor
    (median) then sits within max_span (<= the phasing pad) of every member, so
    canonicalizing a member's position never pushes it outside a supporting
    read's span bracket. Reconcile therefore only ever *declines to merge* — it
    never drops a read. Events it declines to merge simply stay distinct alleles.

    Returns
    -------
    mapping : {original_event_id: (canonical_id, canonical_pos)}
    stats   : {"events", "clusters", "merged", "declined_span", "declined_sample"}
    """
    import statistics

    rows = []  # (contig, pos, svtype, svlen, event_id, sample, dv)
    samples_of: dict[str, set[str]] = {}
    dv_of: dict[str, int] = {}
    pos_of: dict[str, list[int]] = {}
    range_of: dict[str, tuple[int, int]] = {}  # root -> (min_pos, max_pos)
    for path in sidecar_paths:
        sample = os.path.basename(path)
        for contig, recs in _load_sidecar_grouped(path).items():
            for r in recs:
                rows.append((contig, r.pos, r.svtype, r.svlen, r.event_id, sample, r.dv))
                samples_of.setdefault(r.event_id, set()).add(sample)
                dv_of[r.event_id] = max(dv_of.get(r.event_id, 0), r.dv)
                pos_of.setdefault(r.event_id, []).append(r.pos)

    for e, ps in pos_of.items():
        range_of[e] = (min(ps), max(ps))

    parent: dict[str, str] = {e: e for e in samples_of}
    declined = {"span": 0, "sample": 0}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if samples_of[ra] & samples_of[rb]:
            declined["sample"] += 1
            return  # would collapse a same-sample multi-allelic site
        lo = min(range_of[ra][0], range_of[rb][0])
        hi = max(range_of[ra][1], range_of[rb][1])
        if hi - lo > max_span:
            declined["span"] += 1
            return  # cluster would span wider than the phasing pad -> keep apart
        parent[rb] = ra
        samples_of[ra] |= samples_of[rb]
        range_of[ra] = (lo, hi)

    rows.sort(key=lambda x: (x[0], x[1]))
    n = len(rows)
    for i in range(n):
        ci, pi, ti, li, ei = rows[i][0], rows[i][1], rows[i][2], rows[i][3], rows[i][4]
        for j in range(i + 1, n):
            cj, pj = rows[j][0], rows[j][1]
            if cj != ci or pj - pi > pos_tol:
                break
            ej = rows[j][4]
            if ei != ej and _svtypes_concordant(ti, li, rows[j][2], rows[j][3], len_tol_frac):
                union(ei, ej)

    clusters: dict[str, list[str]] = {}
    for e in parent:
        clusters.setdefault(find(e), []).append(e)

    mapping: dict[str, tuple[str, int]] = {}
    merged = 0
    for members in clusters.values():
        canon_id = max(members, key=lambda e: (dv_of[e], e))
        all_pos = [p for e in members for p in pos_of[e]]
        canon_pos = int(statistics.median(all_pos))
        for e in members:
            mapping[e] = (canon_id, canon_pos)
        if len(members) > 1:
            merged += len(members) - 1

    stats = {
        "events": len(samples_of),
        "clusters": len(clusters),
        "merged": merged,
        "declined_span": declined["span"],
        "declined_sample": declined["sample"],
    }
    return mapping, stats


def write_reconciled(
    sidecar_paths: list[str], mapping: dict[str, tuple[str, int]], out_dir: str
) -> list[str]:
    """Rewrite each sidecar into out_dir with canonical event IDs + positions."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for path in sidecar_paths:
        out = os.path.join(out_dir, os.path.basename(path))
        new_recs = []
        for contig, recs in _load_sidecar_grouped(path).items():
            for r in recs:
                cid, cpos = mapping.get(r.event_id, (r.event_id, r.pos))
                new_recs.append(
                    SVRecord(r.contig, cpos, cid, r.svtype, r.svlen, r.af, r.dr, r.dv, r.support_reads)
                )
        write_sidecar(new_recs, out)
        written.append(out)
    return written


# --------------------------------------------------------------------- CLI --#


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


def _reconcile_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="strainphase.sv_encoding reconcile")
    p.add_argument("--sidecars", required=True, nargs="+", help="Per-sample sidecar TSVs")
    p.add_argument("--out-dir", required=True, help="Directory for reconciled sidecars")
    p.add_argument("--pos-tol", type=int, default=50, help="Max breakpoint drift (bp) [50]")
    p.add_argument("--len-tol", type=float, default=0.25, help="Max SVLEN fractional diff [0.25]")
    p.add_argument(
        "--max-span", type=int, default=_RECONCILE_MAX_SPAN,
        help=f"Max total cluster span (bp); keep <= the phasing pad [{_RECONCILE_MAX_SPAN}]",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    mapping, stats = reconcile_events(
        args.sidecars, pos_tol=args.pos_tol, len_tol_frac=args.len_tol, max_span=args.max_span
    )
    write_reconciled(args.sidecars, mapping, args.out_dir)
    logging.info(
        "Reconciled %d events -> %d canonical (%d merged; declined %d span / %d same-sample) into %s",
        stats["events"], stats["clusters"], stats["merged"],
        stats["declined_span"], stats["declined_sample"], args.out_dir,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "verify":
        return _verify_main(argv[1:])
    if argv and argv[0] == "reconcile":
        return _reconcile_main(argv[1:])
    sys.stderr.write(
        "usage: python -m strainphase.sv_encoding {reconcile,verify} ...\n"
        "The package is caller-agnostic and consumes the sidecar TSV format; "
        "produce sidecars with your own adapter (e.g. the pipeline's "
        "sniffles_to_sidecar.py).\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
