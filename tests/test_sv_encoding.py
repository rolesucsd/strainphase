"""Tests for structural-variant pseudo-SNV encoding (strainphase.sv_encoding)
and its consumption in the phasing core (make_windows_lazy / process_contig)."""

from __future__ import annotations

import os
import tempfile

import pytest

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


def _write_sidecar(tmp_path, records, name="sc.tsv", subdir=None):
    """Write a sidecar and clear its cache entry.

    ``subdir`` puts it in a per-sample directory, which is the shape that made file
    basename an unusable sample identity (S1/sv.tsv and S2/sv.tsv are two samples).
    """
    directory = tmp_path if subdir is None else tmp_path / subdir
    directory.mkdir(parents=True, exist_ok=True)
    path = str(directory / name)
    write_sidecar(records, path)
    _SIDECAR_CACHE.pop(path, None)  # avoid cross-test cache bleed
    return path


def _canon(mapping, event_id, contig="c1"):
    """Look an event up in a reconcile mapping.

    The mapping is keyed on ``(contig, event_id)``, never on the id alone: an id reused
    on a second contig must not be able to import the other contig's samples and
    positions into a well-formed cluster here.
    """
    return mapping[(contig, event_id)]


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
    called present here (its other split segment supports it elsewhere).

    The far reads must genuinely REACH the window, or the assertion is on a read that
    was never built and cannot fail. An earlier version of this fixture produced no
    windows at all (four reads, below the phasing floor) and guarded the only assertion
    behind a membership test on the resulting empty dict.
    """
    ref_seq = "A" * 2000
    write_fasta(tmp_path, {"c1": ref_seq})
    reads = []
    for i in range(3):
        s = list("A" * 800)
        s[200 - 101] = "T"  # alt base at the SNV site
        reads.append({"name": f"near{i}", "start": 100, "cigar": "800M", "seq": "".join(s)})
    for i in range(3):
        # Aligned entirely to the RIGHT of the SV anchor at 500, but carrying a marker
        # of their own so they survive into the window.
        s = list("A" * 400)
        s[1600 - 1501] = "T"
        reads.append({"name": f"far{i}", "start": 1500, "cigar": "400M", "seq": "".join(s)})
    bam = write_bam(tmp_path, "c1", 2000, reads)

    windows = make_windows_lazy(
        bam_path=bam,
        contig_id="c1",
        contig_length=2000,
        snv_positions=[200, 500, 1600],
        ref_alleles={200: "A", 500: "N", 1600: "A"},
        # Synthetic reads here are far shorter than the 1 kb physical-overlap
        # default, so the gates are disabled; this test is about SV allele encoding.
        config=HaplotyperConfig(
            window_size=2000, min_read_window_overlap_bp=0, min_read_read_overlap_bp=0
        ),
        site_type={200: "snv", 500: "sv", 1600: "snv"},
        sv_support={500: {"ev.X": {"far0", "far1", "far2"}}},
    )
    by_id = {r.id: r for w in windows for r in w.reads}
    assert {"far0", "far1", "far2"} <= set(by_id), "the far reads must reach the window"
    for nm in ("far0", "far1", "far2"):
        assert 500 not in by_id[nm].alleles, "an unspanned anchor is a hole, not a call"
    # ...while a read that does span it votes there.
    assert by_id["near0"].alleles[500] == "A"


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
    assert (
        _canon(mapping, "ev.A1")[0]
        == _canon(mapping, "ev.A2")[0]
        == _canon(mapping, "ev.A3")[0]
        == "ev.A1"
    )
    # canonical pos is the median (1015)
    assert _canon(mapping, "ev.A1")[1] == 1015
    # the distinct INS is untouched
    assert _canon(mapping, "ev.B") == ("ev.B", 8000)
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
    assert _canon(mapping, "ev.A")[0] == _canon(mapping, "ev.B")[0]
    assert _canon(mapping, "ev.C")[0] != _canon(mapping, "ev.A")[0]
    assert stats["declined_span"] >= 1
    # every canonical anchor is within max_span of its members (no read-drop risk)
    for (_contig, eid), (_cid, cpos) in mapping.items():
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
    assert _canon(mapping, "ev.X")[0] != _canon(mapping, "ev.Y")[0]  # stay distinct
    assert stats["merged"] == 0


def test_reconcile_respects_type_and_length(tmp_path):
    """Same locus, but different type or discordant length -> not merged."""
    s1 = _write_sidecar(tmp_path, [_rec("c1", 1000, "ev.del", "DEL", {"r1"}, svlen=1000)], name="s1.tsv")
    s2 = _write_sidecar(tmp_path, [_rec("c1", 1010, "ev.ins", "INS", {"r2"}, svlen=1000)], name="s2.tsv")
    s3 = _write_sidecar(tmp_path, [_rec("c1", 1010, "ev.big", "DEL", {"r3"}, svlen=5000)], name="s3.tsv")
    mapping, _ = reconcile_events([s1, s2, s3], pos_tol=50, len_tol_frac=0.25)
    ids = {_canon(mapping, e)[0] for e in ("ev.del", "ev.ins", "ev.big")}
    assert len(ids) == 3  # none merged


def test_negative_svlen_still_reconciles(tmp_path):
    """SVLEN is compared on MAGNITUDE (R1-18).

    VCF INFO/SVLEN is negative for a deletion by convention, and a cohort assembled
    from several callers can mix the two conventions. Scaling the tolerance by the
    SIGNED maximum floors it at 1 bp for any negative pair, so every merge of two
    deletions is refused - and the refusal is counted in no statistic, so the run log
    reads "0 merged; declined 0 of 0" and looks like a clean run with nothing nearby.
    """
    s1 = _write_sidecar(tmp_path, [_rec("c1", 1000, "ev.n1", "DEL", {"r1"}, svlen=-1200)], name="s1.tsv")
    s2 = _write_sidecar(tmp_path, [_rec("c1", 1030, "ev.n2", "DEL", {"r2"}, svlen=-1180)], name="s2.tsv")
    mapping, stats = reconcile_events([s1, s2], pos_tol=50, len_tol_frac=0.25)
    assert _canon(mapping, "ev.n1")[0] == _canon(mapping, "ev.n2")[0]
    assert stats["merged"] == 1

    # ...and genuinely discordant magnitudes are still kept apart.
    s3 = _write_sidecar(tmp_path, [_rec("c1", 1030, "ev.n3", "DEL", {"r3"}, svlen=-6000)], name="s3.tsv")
    apart, apart_stats = reconcile_events([s1, s3], pos_tol=50, len_tol_frac=0.25)
    assert _canon(apart, "ev.n1")[0] != _canon(apart, "ev.n3")[0]
    assert apart_stats["declined_type"] == 1, "a near neighbour kept apart must be counted"


def test_an_id_reused_on_another_contig_cannot_damage_this_one(tmp_path):
    """R-C: events are keyed on (contig, event_id), never on the id alone.

    The id is the phasing allele and the spec requires it to be globally unique, but a
    sidecar that breaks that must only spoil its own records. Under id-only keys the
    duplicate on ctg2 imports its sample into ev.B's cluster - so the legitimate 20 bp
    ctg1 merge is refused and logged as the documented multi-allelic protection - and
    imports its POSITION, dragging ev.B's ctg1 anchor hundreds of kb away, out of every
    supporting read's span bracket.
    """
    a = _write_sidecar(
        tmp_path,
        [_rec("c1", 10000, "ev.A", "DEL", {"r1"}, svlen=1000),
         _rec("c2", 900000, "ev.B", "DEL", {"r9"}, svlen=1000)],  # id reused here
        name="a.tsv",
    )
    b = _write_sidecar(tmp_path, [_rec("c1", 10020, "ev.B", "DEL", {"r2"}, svlen=1000)], name="b.tsv")

    mapping, stats = reconcile_events([a, b], pos_tol=50, len_tol_frac=0.25)
    assert _canon(mapping, "ev.A")[0] == _canon(mapping, "ev.B")[0], (
        "the well-formed 20 bp merge on c1 must not be vetoed by a reuse on c2"
    )
    assert stats["merged"] == 1 and stats["declined_sample"] == 0
    # c1's anchor stays where its reads are; c2's record is untouched.
    assert abs(_canon(mapping, "ev.B")[1] - 10020) <= 50
    assert _canon(mapping, "ev.B", contig="c2")[1] == 900000


def test_per_sample_directories_are_distinct_samples(tmp_path):
    """R1-9 + R1-10: sample identity is the sidecar's PATH, not its basename.

    Per-sample directories are the normal layout, and under basename identity every
    one of them is the same sample "sv.tsv": the same-sample veto then refuses every
    legitimate cross-timepoint merge and logs it as multi-allelic protection. The
    output side of the same mistake is worse - all of them are written to one file,
    the caller gets a path per input, and every timepoint is phased against whichever
    sample happened to be written last.
    """
    s1 = _write_sidecar(tmp_path, [_rec("c1", 1000, "ev.A1", "DEL", {"r1"}, svlen=1200)],
                        name="sv.tsv", subdir="S1")
    s2 = _write_sidecar(tmp_path, [_rec("c1", 1030, "ev.A2", "DEL", {"r2"}, svlen=1180)],
                        name="sv.tsv", subdir="S2")

    mapping, stats = reconcile_events([s1, s2], pos_tol=50, len_tol_frac=0.25)
    # Both sidecars are read: under basename identity the second is de-duplicated away
    # as "the same file", so its events never reach the clustering at all.
    assert stats["events"] == 2
    assert stats["clusters"] == 1 and stats["declined_sample"] == 0
    assert _canon(mapping, "ev.A1")[0] == _canon(mapping, "ev.A2")[0]

    written = write_reconciled([s1, s2], mapping, str(tmp_path / "recon"))
    for w in written:
        _SIDECAR_CACHE.pop(w, None)
    assert len(set(written)) == 2, "one output path per input sidecar, all distinct"
    assert all(os.path.exists(w) for w in written)
    canon = _canon(mapping, "ev.A1")[0]
    for w in written:
        pos, _ref, _stype, sup = load_sv_sidecar_for_contig(w, "c1")
        assert pos == [_canon(mapping, "ev.A1")[1]]
        assert set(sup[pos[0]]) == {canon}, "each sample must keep its own record"


def test_one_sample_split_across_sidecars_is_named_explicitly(tmp_path):
    """The one case a path cannot express: several sidecars, one sample.

    Path identity fails OPEN there - it reads the two files as two samples and
    collapses a real multi-allelic site - so ``samples`` names them, and a list that
    does not correspond one-to-one is an error rather than a silent zip truncation.
    """
    a = _write_sidecar(tmp_path, [_rec("c1", 1000, "ev.P", "DEL", {"r1"}, svlen=1000)], name="a.tsv")
    b = _write_sidecar(tmp_path, [_rec("c1", 1020, "ev.Q", "DEL", {"r2"}, svlen=1000)], name="b.tsv")

    naive, _ = reconcile_events([a, b], pos_tol=50, len_tol_frac=0.25)
    assert _canon(naive, "ev.P")[0] == _canon(naive, "ev.Q")[0], "two paths read as two samples"

    named, stats = reconcile_events([a, b], pos_tol=50, len_tol_frac=0.25, samples=["S1", "S1"])
    assert _canon(named, "ev.P")[0] != _canon(named, "ev.Q")[0]
    assert stats["declined_sample"] == 1

    with pytest.raises(ValueError, match="one-to-one"):
        reconcile_events([a, b], samples=["S1"])


def test_an_already_spread_event_is_left_where_it_is(tmp_path):
    """R1-11: the span cap applies to the finished cluster, singletons included.

    An id arriving with two positions further apart than the pad is malformed input no
    merge decision can see. Relocating it to their median would push the anchor outside
    both supporting reads' brackets - and would launder the violation past a later
    ``verify``, which is how it would stay unnoticed.
    """
    spread = _write_sidecar(
        tmp_path,
        [_rec("c1", 10000, "ev.S", "DEL", {"r1"}, svlen=1000),
         _rec("c1", 500000, "ev.S", "DEL", {"r2"}, svlen=1000)],
        name="spread.tsv",
    )
    mapping, stats = reconcile_events([spread], pos_tol=50, len_tol_frac=0.25)
    assert stats["declined_relocate"] == 1
    assert _canon(mapping, "ev.S") == ("ev.S", None), "None means 'leave every record alone'"

    written = write_reconciled([spread], mapping, str(tmp_path / "recon"))
    for w in written:
        _SIDECAR_CACHE.pop(w, None)
    pos, _ref, _stype, _sup = load_sv_sidecar_for_contig(written[0], "c1")
    assert pos == [10000, 500000], "both records keep the position they came in with"
    assert check_event_consistency(written), "the violation must still be visible to verify"


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
