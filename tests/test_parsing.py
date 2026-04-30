"""Tests for VCF/BAM parsing: ``load_snvs`` and the CIGAR walk in
``make_windows_lazy``.

These cover the indel-handling code paths that are not exercised by the
existing test suite. Synthetic VCFs and BAMs are built on the fly via
``tests.util_io``.
"""

from __future__ import annotations

import logging

import pytest

from strainphase.core import (
    DEFAULT_CONFIG,
    HaplotyperConfig,
    LogProbCache,
    _select_log_prob_cache,
    load_snvs,
    make_windows_lazy,
)

from tests.util_io import write_bam, write_fasta, write_vcf


CONTIG = "chr1"
CONTIG_LEN = 10_000


# ============================================================================
# Priority 1 — VCF loader
# ============================================================================


@pytest.fixture
def base_config():
    """Permissive config that lets every record reach the type-classification step."""
    return HaplotyperConfig(
        min_depth_site=1,
        af_range=None,
        require_biallelic=True,
        include_indels=True,
        min_snvs_per_window=1,
        min_reads_per_window=1,
    )


def _record(pos, ref, alt, dp=20, **kw):
    """Build a VCF record dict with sensible defaults."""
    rec = {"pos": pos, "ref": ref, "alt": alt, "info": {"DP": dp}}
    rec.update(kw)
    return rec


def test_loader_snv(tmp_path, base_config):
    """A single SNV produces 'snv' type, no del_span, no ins_len."""
    vcf = write_vcf(tmp_path, CONTIG, [_record(100, "A", "G")])
    pos, refs, depth, af, st, ds, il = load_snvs(vcf, CONTIG, config=base_config)

    assert pos == [100]
    assert refs[100] == "A"
    assert st[100] == "snv"
    assert 100 not in ds
    assert 100 not in il


def test_loader_simple_deletion_footprint(tmp_path, base_config):
    """REF=AGCT, ALT=A at pos 100 -> del_span[100] = (101, 103).

    This is the regression test for the deletion-anchor-vs-footprint bug.
    """
    vcf = write_vcf(tmp_path, CONTIG, [_record(100, "AGCT", "A")])
    pos, refs, _, _, st, ds, il = load_snvs(vcf, CONTIG, config=base_config)

    assert st[100] == "del"
    assert ds[100] == (101, 103)
    assert 100 not in il
    assert refs[100] == "AGCT"


def test_loader_multibase_alt_deletion(tmp_path, base_config):
    """REF=AGCT, ALT=AG at pos 100 -> del_span[100] = (102, 103).

    Catches an implementation that assumes ALT is always a single base.
    """
    vcf = write_vcf(tmp_path, CONTIG, [_record(100, "AGCT", "AG")])
    _, _, _, _, st, ds, _ = load_snvs(vcf, CONTIG, config=base_config)

    assert st[100] == "del"
    assert ds[100] == (102, 103)


def test_loader_simple_insertion(tmp_path, base_config):
    """REF=A, ALT=ACGT at pos 100 -> ins_len[100] = 3."""
    vcf = write_vcf(tmp_path, CONTIG, [_record(100, "A", "ACGT")])
    _, _, _, _, st, ds, il = load_snvs(vcf, CONTIG, config=base_config)

    assert st[100] == "ins"
    assert il[100] == 3
    assert 100 not in ds


def test_loader_mnp_skipped(tmp_path, base_config):
    """Same-length multi-base records (MNPs) are skipped."""
    vcf = write_vcf(
        tmp_path,
        CONTIG,
        [
            _record(100, "AT", "GC"),  # MNP -> skip
            _record(200, "A", "G"),  # SNV -> keep
        ],
    )
    pos, _, _, _, st, _, _ = load_snvs(vcf, CONTIG, config=base_config)

    assert pos == [200]
    assert st[200] == "snv"


def test_loader_multiallelic_dropped_when_biallelic_required(tmp_path, base_config):
    """A record with two ALT alleles is skipped when require_biallelic=True."""
    vcf = write_vcf(tmp_path, CONTIG, [_record(100, "A", ["G", "C"])])
    pos, _, _, _, _, _, _ = load_snvs(vcf, CONTIG, config=base_config)
    assert pos == []


def test_loader_af_range_none_keeps_fixed_sites(tmp_path):
    """With af_range=None, sites at AF=1.0 are kept (the longitudinal use case)."""
    cfg = HaplotyperConfig(min_depth_site=1, af_range=None)
    vcf = write_vcf(
        tmp_path, CONTIG, [_record(100, "A", "G", info={"DP": 20, "AF": 1.0})]
    )
    pos, _, _, _, _, _, _ = load_snvs(vcf, CONTIG, config=cfg)
    assert pos == [100]


def test_loader_af_range_strict_drops_fixed_sites(tmp_path):
    """With af_range=(0.05, 0.95), AF=1.0 sites are dropped."""
    cfg = HaplotyperConfig(min_depth_site=1, af_range=(0.05, 0.95))
    vcf = write_vcf(
        tmp_path,
        CONTIG,
        [
            _record(100, "A", "G", info={"DP": 20, "AF": 1.0}),  # filtered
            _record(200, "A", "G", info={"DP": 20, "AF": 0.5}),  # kept
        ],
    )
    pos, _, _, _, _, _, _ = load_snvs(vcf, CONTIG, config=cfg)
    assert pos == [200]


def test_loader_include_indels_false(tmp_path):
    """When include_indels=False, only SNVs come through."""
    cfg = HaplotyperConfig(min_depth_site=1, af_range=None, include_indels=False)
    vcf = write_vcf(
        tmp_path,
        CONTIG,
        [
            _record(100, "A", "G"),  # SNV -> keep
            _record(200, "AGCT", "A"),  # DEL -> skip
            _record(300, "A", "ACGT"),  # INS -> skip
        ],
    )
    pos, _, _, _, st, ds, il = load_snvs(vcf, CONTIG, config=cfg)
    assert pos == [100]
    assert st[100] == "snv"
    assert ds == {}
    assert il == {}


def test_loader_non_pass_filter_skipped(tmp_path, base_config):
    """Records with non-PASS FILTER are skipped."""
    vcf = write_vcf(
        tmp_path,
        CONTIG,
        [
            _record(100, "A", "G", filter="LowQual"),
            _record(200, "A", "G", filter="PASS"),
        ],
    )
    pos, _, _, _, _, _, _ = load_snvs(vcf, CONTIG, config=base_config)
    assert pos == [200]


def test_loader_min_depth_filter(tmp_path):
    """Sites below min_depth_site are dropped."""
    cfg = HaplotyperConfig(min_depth_site=10, af_range=None)
    vcf = write_vcf(
        tmp_path,
        CONTIG,
        [
            _record(100, "A", "G", dp=5),  # filtered
            _record(200, "A", "G", dp=20),  # kept
        ],
    )
    pos, _, _, _, _, _, _ = load_snvs(vcf, CONTIG, config=cfg)
    assert pos == [200]


def test_loader_multisample_without_sample_name_raises(tmp_path, base_config):
    """Multi-sample VCFs require an explicit sample_name."""
    vcf = write_vcf(
        tmp_path,
        CONTIG,
        [
            {
                "pos": 100,
                "ref": "A",
                "alt": "G",
                "info": {"DP": 20},
                "samples_fmt": {
                    "S1": {"GT": "1", "DP": 20},
                    "S2": {"GT": "1", "DP": 20},
                },
            }
        ],
        samples=["S1", "S2"],
    )
    with pytest.raises(ValueError, match="sample_name"):
        load_snvs(vcf, CONTIG, sample_name=None, config=base_config)


def test_loader_empty_vcf(tmp_path, base_config):
    """A VCF with no records returns empty containers."""
    vcf = write_vcf(tmp_path, CONTIG, [])
    pos, refs, depth, af, st, ds, il = load_snvs(vcf, CONTIG, config=base_config)
    assert pos == []
    assert refs == {}
    assert depth == {}
    assert af == {}
    assert st == {}
    assert ds == {}
    assert il == {}


# ============================================================================
# Priority 2 — CIGAR walk via make_windows_lazy
# ============================================================================


@pytest.fixture
def cigar_config():
    """Config tuned for tiny synthetic windows."""
    return HaplotyperConfig(
        window_size=200,
        min_snvs_per_window=1,
        min_reads_per_window=1,
        min_mapq=0,
        min_base_quality=0,
        af_range=None,
        include_indels=True,
        max_reads_per_window=1000,
    )


def _run_window(tmp_path, cigar_config, vcf_records, reads, ref_seq=None):
    """Build VCF + BAM and run make_windows_lazy. Returns the list of windows."""
    if ref_seq is None:
        ref_seq = "A" * CONTIG_LEN
    write_fasta(tmp_path, {CONTIG: ref_seq})
    vcf = write_vcf(tmp_path, CONTIG, vcf_records, contig_length=CONTIG_LEN)
    bam = write_bam(tmp_path, CONTIG, CONTIG_LEN, reads)

    snv_pos, refs, _, _, st, ds, il = load_snvs(vcf, CONTIG, config=cigar_config)
    return make_windows_lazy(
        bam,
        CONTIG,
        CONTIG_LEN,
        snv_pos,
        refs,
        cigar_config,
        site_type=st,
        del_span=ds,
        ins_len=il,
    )


def _read_alleles(windows, read_name):
    """Find a read by name across all windows; return its allele dict."""
    for w in windows:
        for r in w.reads:
            if r.id == read_name:
                return r.alleles
    return None


def test_cigar_del_exact_match(tmp_path, cigar_config):
    """Read with exact D op of footprint length at footprint start -> 'DEL'."""
    # VCF deletion at pos 100, REF=AGCT ALT=A; deleted footprint = [101, 103].
    vcf_recs = [_record(100, "AGCT", "A")]
    # Read aligned starting at ref 0 (0-based), with 100 M then 3 D then 50 M.
    # pysam reference_start=0, so M op covers ref 0..99 (1-based 1..100); the
    # D op at 1-based [101, 103] matches the footprint exactly.
    reads = [
        {
            "name": "del_carrier",
            "start": 0,
            "cigar": "100M3D50M",
            "seq": "A" * 150,
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "del_carrier")
    assert alleles is not None
    assert alleles[100] == "DEL"


def test_cigar_del_wrong_size_no_call(tmp_path, cigar_config):
    """D op of the wrong length at the footprint start does NOT call DEL.

    Under exact-match semantics, a read with a 5-bp deletion at a site whose
    footprint is 3 bp is treated as not carrying the variant.
    """
    vcf_recs = [_record(100, "AGCT", "A")]  # footprint length 3
    reads = [
        {
            "name": "wrong_size",
            "start": 0,
            "cigar": "100M5D50M",  # 5-bp deletion, not 3
            "seq": "A" * 150,
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "wrong_size")
    assert alleles is not None
    # Anchor base recorded as the ref-like vote; not "DEL".
    assert alleles[100] != "DEL"


def test_cigar_del_off_by_one_no_call(tmp_path, cigar_config):
    """D op anchored one base off does NOT call DEL (no fuzz)."""
    vcf_recs = [_record(100, "AGCT", "A")]
    reads = [
        {
            "name": "off_by_one",
            "start": 0,
            "cigar": "101M3D49M",  # D starts at 1-based 102, not 101
            "seq": "A" * 150,
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "off_by_one")
    assert alleles is not None
    assert alleles[100] != "DEL"


def test_cigar_no_deletion_records_matched_base(tmp_path, cigar_config):
    """Read fully spans the deletion site with continuous M -> matched base.

    Regression test for the order-of-ops bug: if the M op at the anchor
    incorrectly added the position to a 'called' set, a later D op at the
    correct footprint would silently skip.
    """
    vcf_recs = [_record(100, "AGCT", "A")]
    # 'A' only at the anchor position — base is "A" matching ref.
    seq = "C" * 99 + "A" + "C" * 50
    reads = [
        {
            "name": "no_del",
            "start": 0,
            "cigar": "150M",  # no deletion
            "seq": seq,
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "no_del")
    assert alleles is not None
    assert alleles[100] == "A"  # anchor base, not "DEL"


def test_cigar_ins_exact_match(tmp_path, cigar_config):
    """Read with I op anchored exactly at the VCF anchor -> 'INS'."""
    vcf_recs = [_record(100, "A", "ACGT")]
    # M op of length 100 covers ref 1..100 (0-based 0..99), then I op anchored
    # at 1-based ref position 100 (==ref_cursor at that point).
    reads = [
        {
            "name": "ins_carrier",
            "start": 0,
            "cigar": "100M3I50M",
            "seq": "A" * 153,  # 100 + 3 inserted + 50
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "ins_carrier")
    assert alleles is not None
    assert alleles[100] == "INS"


def test_cigar_ins_off_by_one_no_call(tmp_path, cigar_config):
    """I op anchored one base off does NOT call INS (no fuzz)."""
    vcf_recs = [_record(100, "A", "ACGT")]
    reads = [
        {
            "name": "ins_off",
            "start": 0,
            "cigar": "99M3I51M",  # anchor at 99, not 100
            "seq": "A" * 153,
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "ins_off")
    assert alleles is not None
    assert alleles[100] != "INS"


def test_cigar_no_insertion_records_matched_base(tmp_path, cigar_config):
    """Read covering an INS site with continuous M -> matched anchor base."""
    vcf_recs = [_record(100, "A", "ACGT")]
    seq = "C" * 99 + "A" + "C" * 50
    reads = [
        {
            "name": "no_ins",
            "start": 0,
            "cigar": "150M",
            "seq": seq,
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "no_ins")
    assert alleles is not None
    assert alleles[100] == "A"


def test_cigar_read_carries_both_del_and_ins(tmp_path, cigar_config):
    """A read can carry a DEL at one site and an INS at another."""
    vcf_recs = [
        _record(100, "AGCT", "A"),  # DEL at 100
        _record(200, "A", "ACG"),  # INS at 200
    ]
    # 100 M (covers 1..100) + 3 D (deletes 101..103) + 96 M (covers 104..199)
    # + 2 I (insertion anchored at 200... wait need to reach pos 200)
    # Actually: 100 M + 3 D + 96 M = ref consumed 100 + 3 + 96 = 199. Need
    # one more M to reach ref position 200. Then I at anchor 200.
    # 100 M + 3 D + 100 M + 2 I + 50 M:
    #   M1: ref [1..100], query [1..100]
    #   D:  ref [101..103]
    #   M2: ref [104..203], query [101..200]
    # That overshoots — anchor would be 203, not 200.
    # Use 100 M + 3 D + 97 M + 2 I + 50 M:
    #   M1: ref 1..100
    #   D:  ref 101..103
    #   M2: ref 104..200
    #   I:  anchor at 200 ✓
    reads = [
        {
            "name": "both",
            "start": 0,
            "cigar": "100M3D97M2I50M",
            "seq": "A" * (100 + 97 + 2 + 50),
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "both")
    assert alleles is not None
    assert alleles[100] == "DEL"
    assert alleles[200] == "INS"


def test_cigar_snv_alongside_indel(tmp_path, cigar_config):
    """SNV and indel sites in the same window are independently extracted."""
    vcf_recs = [
        _record(50, "A", "G"),  # SNV at 50
        _record(100, "AGCT", "A"),  # DEL at 100
    ]
    # Read: G at ref 50, deletion at 101..103, otherwise A.
    seq = list("A" * 150)
    seq[49] = "G"  # 0-based index 49 == 1-based pos 50
    reads = [
        {
            "name": "mixed",
            "start": 0,
            "cigar": "100M3D50M",
            "seq": "".join(seq),
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "mixed")
    assert alleles is not None
    assert alleles[50] == "G"
    assert alleles[100] == "DEL"


def test_cigar_soft_clip_then_deletion(tmp_path, cigar_config):
    """Leading soft-clip does not throw off the CIGAR cursor for the deletion."""
    vcf_recs = [_record(100, "AGCT", "A")]
    # 10 S (soft-clip, consumes query only) + 100 M + 3 D + 50 M.
    # reference_start is the position of the first MATCHED base, so M starts
    # at 0-based 0, deletion footprint at 1-based [101, 103] is matched.
    reads = [
        {
            "name": "softclip",
            "start": 0,
            "cigar": "10S100M3D50M",
            "seq": "A" * (10 + 100 + 50),
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "softclip")
    assert alleles is not None
    assert alleles[100] == "DEL"


def test_cigar_hard_clip_does_not_break_cursor(tmp_path, cigar_config):
    """Leading hard-clip (consumes neither ref nor query) is handled correctly."""
    vcf_recs = [_record(100, "AGCT", "A")]
    reads = [
        {
            "name": "hardclip",
            "start": 0,
            "cigar": "5H100M3D50M",
            "seq": "A" * 150,  # H does not consume query; seq matches M+D footprint
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "hardclip")
    assert alleles is not None
    assert alleles[100] == "DEL"


def test_cigar_read_does_not_cover_site(tmp_path, cigar_config):
    """If the read does not cover the indel site, no entry is recorded."""
    vcf_recs = [_record(500, "AGCT", "A")]
    reads = [
        {
            "name": "noncover",
            "start": 0,
            "cigar": "100M",
            "seq": "A" * 100,
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    # Window of size 200 contains pos 500 in window starting at 401.
    # If the read doesn't reach there, it just shouldn't appear in that window.
    for w in windows:
        for r in w.reads:
            if r.id == "noncover":
                assert 500 not in r.alleles


def test_cigar_low_quality_anchor_skipped(tmp_path):
    """Matched anchor base with qual < min_base_quality is not recorded."""
    cfg = HaplotyperConfig(
        window_size=200,
        min_snvs_per_window=1,
        min_reads_per_window=1,
        min_mapq=0,
        min_base_quality=20,  # threshold above the read's qual
        af_range=None,
        include_indels=True,
    )
    vcf_recs = [_record(100, "A", "ACGT")]  # INS site
    reads = [
        {
            "name": "lowq",
            "start": 0,
            "cigar": "150M",
            "seq": "A" * 150,
            "quals": [10] * 150,  # below min_base_quality=20
        }
    ]
    write_fasta(tmp_path, {CONTIG: "A" * CONTIG_LEN})
    vcf = write_vcf(tmp_path, CONTIG, vcf_recs, contig_length=CONTIG_LEN)
    bam = write_bam(tmp_path, CONTIG, CONTIG_LEN, reads)
    snv_pos, refs, _, _, st, ds, il = load_snvs(vcf, CONTIG, config=cfg)
    windows = make_windows_lazy(
        bam, CONTIG, CONTIG_LEN, snv_pos, refs, cfg,
        site_type=st, del_span=ds, ins_len=il,
    )
    # No matched-base call should be recorded; the read may have no alleles
    # at all, in which case it isn't kept by the loader.
    for w in windows:
        for r in w.reads:
            if r.id == "lowq":
                assert 100 not in r.alleles


def test_cigar_indel_at_window_boundary(tmp_path, cigar_config):
    """An indel site near the window boundary is still resolved correctly."""
    # Window size is 200; place the indel at pos 199 to test the boundary.
    vcf_recs = [_record(199, "AGC", "A")]  # footprint [200, 201]
    reads = [
        {
            "name": "boundary",
            "start": 0,
            "cigar": "199M2D50M",
            "seq": "A" * 249,
        }
    ]
    windows = _run_window(tmp_path, cigar_config, vcf_recs, reads)
    alleles = _read_alleles(windows, "boundary")
    assert alleles is not None
    assert alleles[199] == "DEL"


# ============================================================================
# Priority 4 — Sanity checks on the alphabet-aware cache
# ============================================================================


class TestLogProbCacheAlphabet:
    """Verify the n_alleles parameter changes mismatch probability correctly."""

    def test_match_probability_independent_of_alphabet(self):
        """Match log-prob should not depend on n_alleles."""
        c4 = LogProbCache(n_alleles=4)
        c6 = LogProbCache(n_alleles=6)
        for q in (10, 20, 30, 40):
            assert c4.log_prob_base("A", "A", q) == pytest.approx(
                c6.log_prob_base("A", "A", q)
            )

    def test_mismatch_probability_scales_with_alphabet(self):
        """Mismatch log-prob spreads p_err across n_alleles - 1 states."""
        import numpy as np

        c4 = LogProbCache(n_alleles=4)
        c6 = LogProbCache(n_alleles=6)
        q = 30
        # Difference should be log(3) - log(5) = log(3/5).
        diff = c4.log_prob_base("A", "C", q) - c6.log_prob_base("A", "C", q)
        assert diff == pytest.approx(np.log(5.0 / 3.0), rel=1e-3)

    def test_indel_states_work_in_six_allele_cache(self):
        """DEL/INS strings are valid alleles with the 6-allele cache."""
        c6 = LogProbCache(n_alleles=6)
        # match
        assert c6.log_prob_base("DEL", "DEL", 30) == c6.log_prob_base("A", "A", 30)
        # mismatch
        assert c6.log_prob_base("DEL", "A", 30) == c6.log_prob_base("A", "C", 30)

    def test_invalid_alphabet_raises(self):
        with pytest.raises(ValueError):
            LogProbCache(n_alleles=1)

    def test_select_log_prob_cache_picks_4_when_indels_disabled(self):
        cfg = HaplotyperConfig(include_indels=False)
        assert _select_log_prob_cache(cfg).n_alleles == 4

    def test_select_log_prob_cache_picks_6_when_indels_enabled(self):
        cfg = HaplotyperConfig(include_indels=True)
        assert _select_log_prob_cache(cfg).n_alleles == 6


# ============================================================================
# Sanity: no crashes on empty inputs
# ============================================================================


def test_empty_vcf_yields_no_windows(tmp_path, cigar_config, caplog):
    """Empty VCF produces no windows and a warning."""
    write_fasta(tmp_path, {CONTIG: "A" * CONTIG_LEN})
    vcf = write_vcf(tmp_path, CONTIG, [], contig_length=CONTIG_LEN)
    bam = write_bam(
        tmp_path,
        CONTIG,
        CONTIG_LEN,
        [{"name": "r", "start": 0, "cigar": "150M", "seq": "A" * 150}],
    )
    snv_pos, refs, _, _, st, ds, il = load_snvs(vcf, CONTIG, config=cigar_config)
    assert snv_pos == []

    with caplog.at_level(logging.WARNING):
        windows = make_windows_lazy(
            bam, CONTIG, CONTIG_LEN, snv_pos, refs, cigar_config,
            site_type=st, del_span=ds, ins_len=il,
        )
    assert windows == []


def test_default_config_is_indel_aware():
    """The default config enables indels and uses no AF filter."""
    assert DEFAULT_CONFIG.include_indels is True
    assert DEFAULT_CONFIG.af_range is None


# ============================================================================
# Simulator: indel injection round-trip
# ============================================================================


def test_simulator_emits_indel_cigar():
    """``simulate_read`` emits D and I CIGAR ops when the strain carries indels."""
    import numpy as np

    from validation.simulate_reads import Strain, simulate_read

    s = Strain(id="S", genome_file="-")
    ref = "ACGTACGTACGT" * 20  # 240 bp
    s.contigs["c"] = ref
    s.deletions["c"] = {50: 3}  # delete 3 bases after pos 50
    s.insertions["c"] = {100: "NNN"}  # insert NNN after pos 100

    rng = np.random.default_rng(0)
    seq, quals, cigar = simulate_read(s, "c", 0, 200, 0.0, rng)

    assert "3D" in cigar
    assert "3I" in cigar
    # Read sequence: 200 ref positions - 3 deleted + 3 inserted = 200 bp
    assert len(seq) == 200
    assert len(quals) == 200


def test_simulator_writes_canonical_indel_vcf(tmp_path):
    """``write_vcf`` emits indels in canonical left-anchored form."""
    from validation.simulate_reads import Strain, write_vcf

    ref = Strain(id="ref", genome_file="-")
    ref.contigs["c"] = "ACGT" * 60  # 240 bp

    s1 = Strain(id="s1", genome_file="-")
    s1.contigs["c"] = ref.contigs["c"]
    s1.deletions["c"] = {100: 3}
    s1.insertions["c"] = {150: "TTT"}

    out = tmp_path / "truth.vcf"
    write_vcf({"c": []}, [ref, s1], ref, str(out))

    body = [l for l in out.read_text().splitlines() if not l.startswith("#")]
    # Two records: one DEL and one INS
    assert len(body) == 2

    rows = [l.split("\t") for l in body]
    by_pos = {int(r[1]): r for r in rows}

    # DEL: anchor 0-based 100 -> VCF POS 101; REF len = 1+3 = 4; ALT len = 1
    del_row = by_pos[101]
    assert len(del_row[3]) == 4
    assert len(del_row[4]) == 1

    # INS: anchor 0-based 150 -> VCF POS 151; REF len = 1; ALT len = 1+3 = 4
    ins_row = by_pos[151]
    assert len(ins_row[3]) == 1
    assert len(ins_row[4]) == 4
