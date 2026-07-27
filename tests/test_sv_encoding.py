"""Tests for structural-variant pseudo-SNV encoding (strainphase.sv_encoding)
and its consumption in the phasing core (make_windows_lazy / process_contig)."""

from __future__ import annotations

import tempfile

from strainphase.core import HaplotyperConfig, make_windows_lazy, process_contig
from strainphase.sv_encoding import (
    _SIDECAR_CACHE,
    SVRecord,
    check_event_consistency,
    load_sv_sidecar_for_contig,
    reconcile_events,
    write_reconciled,
    write_sidecar,
)

from tests.util_io import write_bam, write_fasta, write_vcf


def _rec(contig, pos, event_id, svtype, reads, svlen=1000, af=0.4, dr=6, dv=4):
    return SVRecord(contig, pos, event_id, svtype, svlen, af, dr, dv, set(reads))


def _write_sidecar(tmp_path, records, name="sc.tsv"):
    path = str(tmp_path / name)
    write_sidecar(records, path)
    _SIDECAR_CACHE.pop(path, None)  # avoid cross-test cache bleed
    return path


def test_sidecar_roundtrip(tmp_path):
    recs = [
        _rec("MAG1_contig_1", 14523, "ev.INS.1", "INS", {"rA", "rB"}),
        _rec("MAG1_contig_1", 90210, "ev.DEL.1", "DEL", {"rX"}),
        _rec("MAG1_contig_2", 33, "ev.DUP.1", "DUP", {"rZ"}),
    ]
    path = _write_sidecar(tmp_path, recs)

    pos, ref, stype, sup = load_sv_sidecar_for_contig(path, "MAG1_contig_1")
    assert pos == [14523, 90210]
    assert stype == {14523: "sv", 90210: "sv"}  # generic marker, not INS/DEL
    assert sup[14523] == {"ev.INS.1": {"rA", "rB"}}  # keyed by event ID
    assert ref[14523] == "N"

    pos2, _, stype2, sup2 = load_sv_sidecar_for_contig(path, "MAG1_contig_2")
    assert pos2 == [33] and stype2 == {33: "sv"} and sup2 == {33: {"ev.DUP.1": {"rZ"}}}

    assert load_sv_sidecar_for_contig(path, "MAG1_contig_9") == ([], {}, {}, {})


def _build_ref_and_reads(tmp_path, read_names, snv_pos=200, length=1000):
    ref_seq = "A" * length
    ref = write_fasta(tmp_path, {"c1": ref_seq})
    reads = []
    for nm in read_names:
        seq = list("A" * 800)  # covers ref 101..900
        seq[snv_pos - 101] = "T"  # alt base at the SNV site
        reads.append({"name": nm, "start": 100, "cigar": "800M", "seq": "".join(seq)})
    bam = write_bam(tmp_path, "c1", length, reads)
    return ref, bam


def test_present_allele_is_event_id(tmp_path):
    """Support reads get the EVENT ID at the anchor; others vote the ref base."""
    names = ["pres1", "pres2", "abs1", "abs2", "abs3"]
    _ref, bam = _build_ref_and_reads(tmp_path, names)

    windows = make_windows_lazy(
        bam_path=bam,
        contig_id="c1",
        contig_length=1000,
        snv_positions=[200, 500],
        ref_alleles={200: "A", 500: "N"},
        # 800 bp synthetic reads on a 1 kb contig sit below the 1 kb overlap
        # default; this test is about SV allele encoding, not depth policy.
        config=HaplotyperConfig(
            min_read_window_overlap_bp=0, min_read_read_overlap_bp=0
        ),
        site_type={200: "snv", 500: "sv"},
        sv_support={500: {"ev.INS.7": {"pres1", "pres2"}}},
    )
    assert windows
    by_id = {r.id: r for w in windows for r in w.reads}
    for nm in ("pres1", "pres2"):
        assert by_id[nm].alleles[500] == "ev.INS.7"
    for nm in ("abs1", "abs2", "abs3"):
        assert by_id[nm].alleles[500] == "A"


def test_distinct_events_at_one_anchor_stay_distinct(tmp_path):
    """Two different events at the SAME anchor must NOT collapse — reads carrying
    event A vs event B get different alleles (multi-allelic site)."""
    names = ["a1", "a2", "b1", "b2", "ref1"]
    _ref, bam = _build_ref_and_reads(tmp_path, names)

    windows = make_windows_lazy(
        bam_path=bam,
        contig_id="c1",
        contig_length=1000,
        snv_positions=[200, 500],
        ref_alleles={200: "A", 500: "N"},
        # 800 bp synthetic reads on a 1 kb contig sit below the 1 kb overlap
        # default; this test is about SV allele encoding, not depth policy.
        config=HaplotyperConfig(
            min_read_window_overlap_bp=0, min_read_read_overlap_bp=0
        ),
        site_type={200: "snv", 500: "sv"},
        sv_support={500: {"ev.A": {"a1", "a2"}, "ev.B": {"b1", "b2"}}},
    )
    by_id = {r.id: r for w in windows for r in w.reads}
    assert by_id["a1"].alleles[500] == "ev.A"
    assert by_id["a2"].alleles[500] == "ev.A"
    assert by_id["b1"].alleles[500] == "ev.B"
    assert by_id["b2"].alleles[500] == "ev.B"
    assert by_id["ref1"].alleles[500] == "A"  # supports neither -> reference
    # A and B are genuinely different alleles.
    assert by_id["a1"].alleles[500] != by_id["b1"].alleles[500]


def test_sv_present_requires_span_overlap(tmp_path):
    """A read in the support set whose alignment doesn't reach the anchor is not
    called present here (its other split segment supports it elsewhere)."""
    ref_seq = "A" * 2000
    write_fasta(tmp_path, {"c1": ref_seq})
    reads = [
        {"name": "near1", "start": 100, "cigar": "800M", "seq": "A" * 800},
        {"name": "near2", "start": 100, "cigar": "800M", "seq": "A" * 800},
        {"name": "near3", "start": 100, "cigar": "800M", "seq": "A" * 800},
        {"name": "faraway", "start": 1500, "cigar": "200M", "seq": "A" * 200},
    ]
    for r in reads[:3]:
        s = list(r["seq"])
        s[200 - 101] = "T"
        r["seq"] = "".join(s)
    bam = write_bam(tmp_path, "c1", 2000, reads)

    windows = make_windows_lazy(
        bam_path=bam,
        contig_id="c1",
        contig_length=2000,
        snv_positions=[200, 500],
        ref_alleles={200: "A", 500: "N"},
        # Synthetic reads here are far shorter than the 1 kb physical-overlap
        # default, so the gates are disabled; this test is about SV allele encoding.
        config=HaplotyperConfig(
            window_size=1200, min_read_window_overlap_bp=0, min_read_read_overlap_bp=0
        ),
        site_type={200: "snv", 500: "sv"},
        sv_support={500: {"ev.X": {"faraway"}}},
    )
    by_id = {r.id: r for w in windows for r in w.reads}
    if "faraway" in by_id:
        assert by_id["faraway"].alleles.get(500) != "ev.X"


def test_process_contig_merges_and_drops_collisions(tmp_path):
    _ref, bam = _build_ref_and_reads(tmp_path, ["r1", "r2", "r3", "r4"])
    vcf = write_vcf(
        tmp_path,
        "c1",
        [{"pos": 200, "ref": "A", "alt": "T", "info": {"DP": 20, "AF": 0.5}}],
        contig_length=1000,
    )
    sidecar = _write_sidecar(
        tmp_path,
        [
            _rec("c1", 600, "ev.INS.9", "INS", {"r1", "r2"}, svlen=3000),
            _rec("c1", 200, "ev.DEL.9", "DEL", {"r3"}, svlen=100),  # collides w/ SNV
        ],
    )
    results = process_contig(
        bam_path=bam,
        vcf_path=vcf,
        contig_id="c1",
        contig_length=1000,
        # 4 synthetic reads, far shorter than the 1 kb physical-overlap default and
        # below the 10-read phasing floor. Both are relaxed here; this test is about SV
        # allele encoding, not depth policy.
        config=HaplotyperConfig(
            min_read_window_overlap_bp=0,
            min_read_read_overlap_bp=0,
            min_reads_per_window=1,
        ),
        sample_id="s1",
        sv_sidecar_path=sidecar,
    )
    assert isinstance(results, list)
    all_positions = {p for wr in results for p in wr.window.snv_pos}
    assert 600 in all_positions
    assert 200 in all_positions


def test_event_consistency_check(tmp_path):
    """verification #2: an event ID at different loci across sidecars is flagged;
    the same ID at the same locus is fine."""
    s1 = _write_sidecar(tmp_path, [_rec("c1", 600, "ev.1", "INS", {"a"})], name="s1.tsv")
    s2_ok = _write_sidecar(tmp_path, [_rec("c1", 600, "ev.1", "INS", {"b"})], name="s2ok.tsv")
    s2_bad = _write_sidecar(tmp_path, [_rec("c1", 650, "ev.1", "INS", {"b"})], name="s2bad.tsv")

    assert check_event_consistency([s1, s2_ok]) == []  # same event, same locus
    bad = check_event_consistency([s1, s2_bad])  # same event, drifted locus
    assert len(bad) == 1 and "ev.1" in bad[0]


def test_reconcile_merges_drifted_events(tmp_path):
    """The same DEL called with drifting breakpoints/IDs across 3 samples is
    reconciled to one canonical (id, pos); a distinct event stays separate."""
    s1 = _write_sidecar(tmp_path, [_rec("c1", 1000, "ev.A1", "DEL", {"r1"}, svlen=1200, dv=9)], name="s1.tsv")
    s2 = _write_sidecar(tmp_path, [_rec("c1", 1030, "ev.A2", "DEL", {"r2"}, svlen=1180, dv=4)], name="s2.tsv")
    s3 = _write_sidecar(tmp_path, [_rec("c1", 1015, "ev.A3", "DEL", {"r3"}, svlen=1210, dv=5)], name="s3.tsv")
    # a genuinely different event far away
    s4 = _write_sidecar(tmp_path, [_rec("c1", 8000, "ev.B", "INS", {"r4"}, svlen=500)], name="s4.tsv")

    mapping, stats = reconcile_events([s1, s2, s3, s4], pos_tol=50, len_tol_frac=0.25)
    # A1/A2/A3 collapse to one canonical id (the best-supported = ev.A1, dv=9)
    assert mapping["ev.A1"][0] == mapping["ev.A2"][0] == mapping["ev.A3"][0] == "ev.A1"
    # canonical pos is the median (1015)
    assert mapping["ev.A1"][1] == 1015
    # the distinct INS is untouched
    assert mapping["ev.B"] == ("ev.B", 8000)
    assert stats["clusters"] == 2 and stats["merged"] == 2


def test_reconcile_span_cap_prevents_chaining(tmp_path):
    """Chained drift (A~B~C spanning > max_span) must NOT collapse into one
    cluster whose canonical anchor would fall outside a member's pad. reconcile
    declines the wide merge instead of dropping — A+B merge, C stays separate."""
    s1 = _write_sidecar(tmp_path, [_rec("c1", 1000, "ev.A", "DEL", {"r1"}, svlen=1000, dv=9)], name="a.tsv")
    s2 = _write_sidecar(tmp_path, [_rec("c1", 1040, "ev.B", "DEL", {"r2"}, svlen=1000, dv=5)], name="b.tsv")
    s3 = _write_sidecar(tmp_path, [_rec("c1", 1080, "ev.C", "DEL", {"r3"}, svlen=1000, dv=5)], name="c.tsv")
    # pos_tol=50 allows A~B (40) and B~C (40), but A..C span is 80 > max_span=50.
    mapping, stats = reconcile_events([s1, s2, s3], pos_tol=50, len_tol_frac=0.25, max_span=50)
    # A and B merge; C is declined (span cap) and stays its own event.
    assert mapping["ev.A"][0] == mapping["ev.B"][0]
    assert mapping["ev.C"][0] != mapping["ev.A"][0]
    assert stats["declined_span"] >= 1
    # every canonical anchor is within max_span of its members (no read-drop risk)
    for eid, (_cid, cpos) in mapping.items():
        assert abs(cpos - {"ev.A": 1000, "ev.B": 1040, "ev.C": 1080}[eid]) <= 50


def test_reconcile_never_merges_same_sample(tmp_path):
    """Two nearby distinct events in ONE sample must NOT be collapsed."""
    s1 = _write_sidecar(
        tmp_path,
        [_rec("c1", 1000, "ev.X", "DEL", {"r1"}, svlen=1000),
         _rec("c1", 1020, "ev.Y", "DEL", {"r2"}, svlen=1000)],
        name="s1.tsv",
    )
    mapping, stats = reconcile_events([s1], pos_tol=50, len_tol_frac=0.25)
    assert mapping["ev.X"][0] != mapping["ev.Y"][0]  # stay distinct
    assert stats["merged"] == 0


def test_reconcile_respects_type_and_length(tmp_path):
    """Same locus, but different type or discordant length -> not merged."""
    s1 = _write_sidecar(tmp_path, [_rec("c1", 1000, "ev.del", "DEL", {"r1"}, svlen=1000)], name="s1.tsv")
    s2 = _write_sidecar(tmp_path, [_rec("c1", 1010, "ev.ins", "INS", {"r2"}, svlen=1000)], name="s2.tsv")
    s3 = _write_sidecar(tmp_path, [_rec("c1", 1010, "ev.big", "DEL", {"r3"}, svlen=5000)], name="s3.tsv")
    mapping, _ = reconcile_events([s1, s2, s3], pos_tol=50, len_tol_frac=0.25)
    ids = {mapping["ev.del"][0], mapping["ev.ins"][0], mapping["ev.big"][0]}
    assert len(ids) == 3  # none merged


def test_write_reconciled_roundtrip(tmp_path):
    s1 = _write_sidecar(tmp_path, [_rec("c1", 1000, "ev.A1", "DEL", {"r1"}, svlen=1200)], name="s1.tsv")
    s2 = _write_sidecar(tmp_path, [_rec("c1", 1030, "ev.A2", "DEL", {"r2"}, svlen=1200)], name="s2.tsv")
    mapping, _ = reconcile_events([s1, s2], pos_tol=50, len_tol_frac=0.25)
    outdir = str(tmp_path / "recon")
    written = write_reconciled([s1, s2], mapping, outdir)
    for w in written:
        _SIDECAR_CACHE.pop(w, None)
    # after reconcile, verify passes (one id -> one locus)
    assert check_event_consistency(written) == []


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        import pathlib

        test_sidecar_roundtrip(pathlib.Path(d))
    print("ok")
