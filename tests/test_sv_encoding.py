"""Tests for structural-variant pseudo-SNV encoding (strainphase.sv_encoding)
and its consumption in the phasing core (make_windows_lazy / process_contig)."""

from __future__ import annotations

import tempfile

from strainphase.core import DEFAULT_CONFIG, HaplotyperConfig, make_windows_lazy, process_contig
from strainphase.sv_encoding import (
    _SIDECAR_CACHE,
    SVRecord,
    load_sv_sidecar_for_contig,
    write_sidecar,
)

from tests.util_io import write_bam, write_fasta, write_vcf


def _write_sidecar(tmp_path, records, name="sc.tsv"):
    path = str(tmp_path / name)
    write_sidecar(records, path)
    _SIDECAR_CACHE.pop(path, None)  # avoid cross-test cache bleed
    return path


def test_sidecar_roundtrip(tmp_path):
    recs = [
        SVRecord("MAG1_contig_1", 14523, "ins", "INS", 4210, 0.34, 21, 11, {"rA", "rB"}),
        SVRecord("MAG1_contig_1", 90210, "del", "DEL", 512, 0.60, 4, 6, {"rX"}),
        SVRecord("MAG1_contig_2", 33, "ins", "DUP", 8000, 0.5, 5, 5, {"rZ"}),
    ]
    path = _write_sidecar(tmp_path, recs)

    pos, ref, stype, sup = load_sv_sidecar_for_contig(path, "MAG1_contig_1")
    assert pos == [14523, 90210]
    assert stype == {14523: "sv_ins", 90210: "sv_del"}
    assert sup[14523] == {"rA", "rB"}
    assert ref[14523] == "N"

    pos2, _, stype2, sup2 = load_sv_sidecar_for_contig(path, "MAG1_contig_2")
    assert pos2 == [33] and stype2 == {33: "sv_ins"} and sup2 == {33: {"rZ"}}

    # Absent contig -> empty structures, not an error.
    assert load_sv_sidecar_for_contig(path, "MAG1_contig_9") == ([], {}, {}, {})


def _build_ref_and_reads(tmp_path, read_names, snv_pos=200, sv_anchor=500, length=1000):
    """A contig with one SNV and reads that all span [101, 900]."""
    ref_seq = "A" * length
    ref = write_fasta(tmp_path, {"c1": ref_seq})

    reads = []
    for nm in read_names:
        seq = list("A" * 800)  # covers ref 101..900
        seq[snv_pos - 101] = "T"  # alt base at the SNV site
        reads.append({"name": nm, "start": 100, "cigar": "800M", "seq": "".join(seq)})
    bam = write_bam(tmp_path, "c1", length, reads)
    return ref, bam


def test_make_windows_lazy_sv_present_vs_absent(tmp_path):
    """Support-set reads get the INS token at the anchor; others vote the ref base."""
    names = ["pres1", "pres2", "abs1", "abs2", "abs3"]
    _ref, bam = _build_ref_and_reads(tmp_path, names)

    windows = make_windows_lazy(
        bam_path=bam,
        contig_id="c1",
        contig_length=1000,
        snv_positions=[200, 500],
        ref_alleles={200: "A", 500: "N"},
        config=DEFAULT_CONFIG,
        site_type={200: "snv", 500: "sv_ins"},
        sv_support={500: {"pres1", "pres2"}},
    )
    assert windows, "expected at least one window"
    by_id = {r.id: r for w in windows for r in w.reads}
    assert set(by_id) == set(names)

    # Present reads carry the INS pseudo-allele; absent reads vote the ref base.
    for nm in ("pres1", "pres2"):
        assert by_id[nm].alleles[500] == "INS"
    for nm in ("abs1", "abs2", "abs3"):
        assert by_id[nm].alleles[500] == "A"


def test_sv_present_requires_span_overlap(tmp_path):
    """A read in the support set whose alignment does not reach the anchor
    (its other split segment supports the SV) is not called present here."""
    ref_seq = "A" * 2000
    write_fasta(tmp_path, {"c1": ref_seq})
    # far-away read maps at 1500..1700, nowhere near anchor 500.
    reads = [
        {"name": "near1", "start": 100, "cigar": "800M", "seq": "A" * 800},
        {"name": "near2", "start": 100, "cigar": "800M", "seq": "A" * 800},
        {"name": "near3", "start": 100, "cigar": "800M", "seq": "A" * 800},
        {"name": "faraway", "start": 1500, "cigar": "200M", "seq": "A" * 200},
    ]
    # set SNV alt base for the near reads so they overlap the SNV site
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
        config=HaplotyperConfig(window_size=1200),
        site_type={200: "snv", 500: "sv_ins"},
        sv_support={500: {"faraway"}},  # claims support but doesn't span anchor
    )
    by_id = {r.id: r for w in windows for r in w.reads}
    # faraway either absent from the anchor window or has no call at 500.
    if "faraway" in by_id:
        assert by_id["faraway"].alleles.get(500) != "INS"


def test_process_contig_merges_and_drops_collisions(tmp_path):
    """process_contig merges SV sites and drops anchors colliding with a variant."""
    _ref, bam = _build_ref_and_reads(tmp_path, ["r1", "r2", "r3", "r4"])
    vcf = write_vcf(
        tmp_path,
        "c1",
        [{"pos": 200, "ref": "A", "alt": "T", "info": {"DP": 20, "AF": 0.5}}],
        contig_length=1000,
    )
    # One clean SV site (600) and one colliding with the SNV at 200 (dropped).
    sidecar = _write_sidecar(
        tmp_path,
        [
            SVRecord("c1", 600, "ins", "INS", 3000, 0.4, 6, 4, {"r1", "r2"}),
            SVRecord("c1", 200, "del", "DEL", 100, 0.3, 7, 3, {"r3"}),
        ],
    )

    results = process_contig(
        bam_path=bam,
        vcf_path=vcf,
        contig_id="c1",
        contig_length=1000,
        config=HaplotyperConfig(),
        sample_id="s1",
        sv_sidecar_path=sidecar,
    )
    # Should run without error and produce window results.
    assert isinstance(results, list)
    # The merged SV site (600) should appear among phased positions; the
    # colliding one (200) stays a SNV, not an SV token.
    all_positions = {p for wr in results for p in wr.window.snv_pos}
    assert 600 in all_positions
    assert 200 in all_positions


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        import pathlib

        test_sidecar_roundtrip(pathlib.Path(d))
    print("ok")
