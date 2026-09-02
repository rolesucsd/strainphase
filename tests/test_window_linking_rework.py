#!/usr/bin/env python3
"""Tests for the window-linking rework.

Covers the identity gate stack, the marker set (positions that actually vary), the
abundance coherence test, the QC gate, read-overlap linking, and the zero-leak fix.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import os

import numpy as np
import pytest

from strainphase.coherence import abundance_coherent, qc_flags
from strainphase.core import (
    HaplotyperConfig,
    Read,
    compare_consensus,
    unique_best_matches,
    variable_marker_positions,
)
from strainphase.longitudinal import _window_conditional_abundance
from strainphase.window_groups import WindowHaplotype


def cfg(**kw) -> HaplotyperConfig:
    """Config sized for small synthetic markers rather than 20 kb windows."""
    base = {
        "min_entity_overlap_bp": 0,
        "min_cosupported_span_frac": 0.0,
        "min_shared_markers": 3,
    }
    base.update(kw)
    return HaplotyperConfig(**base)


# --------------------------------------------------------------------------- #
# Marker set
# --------------------------------------------------------------------------- #


def test_marker_set_keeps_only_variable_positions():
    """A position where every haplotype agrees carries no identity information."""
    consensuses = [
        {10: "A", 20: "C", 30: "G"},
        {10: "A", 20: "T", 30: "G"},  # only pos 20 differs
    ]
    assert variable_marker_positions(consensuses) == {20}


def test_sv_sites_are_identity_markers_by_default():
    """SVs are kept as identity markers by default: an inversion is a marker like any
    other, and its flip shows up as two entities trading frequency. The exclusion path
    stays reachable (``exclude_sv_from_identity``) so the effect can be measured, but is
    off by default.
    """
    consensuses = [{10: "A", 99: "ev.INV.1"}, {10: "T", 99: "ev.INV.2"}]
    site_type = {10: "snv", 99: "sv"}
    assert variable_marker_positions(consensuses, site_type) == {10, 99}
    dropped = variable_marker_positions(
        consensuses, site_type, cfg(exclude_sv_from_identity=True)
    )
    assert dropped == {10}


def test_marker_set_is_empty_for_a_clonal_locus():
    """A clonal sample has no variable sites, so its marker set is empty."""
    consensuses = [{10: "A", 20: "C"}, {10: "A", 20: "C"}]
    assert variable_marker_positions(consensuses) == set()


# --------------------------------------------------------------------------- #
# Gate stack
# --------------------------------------------------------------------------- #


def test_no_discriminating_markers_means_no_verdict():
    """With no discriminating markers the verdict is no-evidence, not a link. Invariant
    positions cannot disagree, so comparing them would only dilute a real difference;
    the cost is that genuinely clonal loci no longer link on absence of evidence.
    """
    a = {10: "A", 20: "C", 30: "G"}
    b = {10: "A", 20: "C", 30: "G"}
    gate = compare_consensus(a, b, markers=set(), config=cfg())
    assert not gate.passed
    assert gate.reason == "failed_no_evidence"
    assert gate.n_shared == 0



def test_failure_reason_distinguishes_dropout_from_disagreement():
    """The lineage layer needs to tell a measurement hole from a genotypic wall."""
    a = {10: "A", 20: "C", 30: "G"}
    thin = {10: "A"}  # only 1 shared position -> no evidence either way
    assert compare_consensus(a, thin, set(), cfg()).reason == "failed_no_evidence"

    differing = {10: "T", 20: "A", 30: "C"}
    markers = variable_marker_positions([a, differing])
    assert compare_consensus(a, differing, markers, cfg()).reason == "failed_mismatch"


def test_min_entity_overlap_is_an_explicit_non_merge():
    """Below the physical-overlap floor the verdict is NON-MERGE, not 'unknown'."""
    a = {10: "A", 11: "C", 12: "G"}
    b = {10: "A", 11: "C", 12: "G"}
    gate = compare_consensus(a, b, set(), cfg(min_entity_overlap_bp=1000))
    assert not gate.passed
    assert gate.reason == "failed_no_evidence"


def test_overlap_gate_ignores_where_the_markers_sit():
    """The overlap gate measures how much sequence was compared, not how spread out the
    markers are. Clustered variation (recombination tracts, hypervariable loci) is
    exactly where markers bunch up, so measuring their spread would penalise the most
    informative sites.
    """
    # Both cover 1000..7000 (6 kb of shared sequence); the only two markers are the
    # adjacent positions 3000/3190, 190 bp apart.
    a = {1000: "A", 3000: "G", 3190: "G", 7000: "T"}
    b = {1000: "A", 3000: "G", 3190: "G", 7000: "T"}
    gate = compare_consensus(a, b, {3000, 3190}, cfg(min_entity_overlap_bp=1000),
                             min_shared=2)
    assert gate.passed, "6 kb of shared sequence must not be rejected for tight markers"

    # ...and the gate still bites when the FOOTPRINTS barely overlap, even though the
    # pair clears min_shared. Only the sequence overlap decides this.
    c = {1000: "A", 2950: "G", 3000: "C"}
    d = {2950: "G", 3000: "C", 9000: "T"}
    gate = compare_consensus(c, d, {2950, 3000}, cfg(min_entity_overlap_bp=1000),
                             min_shared=2)
    assert not gate.passed and gate.n_shared == 2, "50 bp of overlap must be rejected"


def test_cosupported_span_fraction_gate():
    """Two haplotypes touching only at the edge of a shared region are rejected."""
    a = {p: "A" for p in (100, 101, 102)}
    b = {p: "A" for p in (100, 101, 102)}
    marks = {100, 101, 102}      # real markers: the gate needs a verdict to gate ON
    config = cfg(min_cosupported_span_frac=0.25, min_entity_overlap_bp=0)
    # co-supported span is 2 bp inside a 10 kb region -> far below 25%
    assert not compare_consensus(a, b, marks, config, region=(1, 10001)).passed
    # same pair with no region constraint passes
    assert compare_consensus(a, b, marks, config).passed


# --------------------------------------------------------------------------- #
# unique_best_matches
# --------------------------------------------------------------------------- #


def test_tie_contributes_no_edge():
    """A tied best match is ambiguous and yields no edge, which bounds entity size."""
    assert unique_best_matches({0: [(0.0, 1)]}) == {0: 1}
    assert unique_best_matches({0: [(0.0, 1), (0.0, 2)]}) == {}
    assert unique_best_matches({0: [(0.0, 1), (0.5, 2)]}) == {0: 1}


# --------------------------------------------------------------------------- #
# Window-haplotype fixture
# --------------------------------------------------------------------------- #


def _hap(sample, hid, consensus, window_start=1, reads=20, total=40):
    return WindowHaplotype(
        sample=sample,
        contig="c1",
        window_start=window_start,
        window_end=window_start + 20000,
        haplotype_id=hid,
        consensus=consensus,
        reads=reads,
        total_reads=total,
    )










# --------------------------------------------------------------------------- #
# Abundance coherence -- SINGLE TIMEPOINT ONLY
# --------------------------------------------------------------------------- #


def test_coherent_when_counts_agree():
    assert abundance_coherent([(20, 40), (22, 40), (19, 40)]).coherent


def test_incoherent_when_counts_disagree_with_depth_to_prove_it():
    result = abundance_coherent([(2, 100), (95, 100)])
    assert not result.coherent
    assert result.min_p < 0.01


def test_low_depth_windows_are_excluded_not_failed():
    """At 3 reads the test has no power; including such windows would dilute the result
    rather than inform it."""
    result = abundance_coherent([(1, 3), (3, 3)])
    assert result.coherent
    assert result.n_tested == 0


def test_sampling_noise_alone_does_not_reject():
    """At ~12 reads two windows at the SAME true frequency routinely differ by 0.3+, so
    a fixed threshold would reject real merges. A likelihood test must not."""
    assert abundance_coherent([(6, 12), (9, 12)]).coherent


# --------------------------------------------------------------------------- #
# QC gate
# --------------------------------------------------------------------------- #


def test_qc_flags_detect_too_many_members_per_cell():
    members = [
        {"sample": "t0", "window_start": 1, "reads": 10, "total_reads": 40}
        for _ in range(3)
    ]
    assert qc_flags("E1", members).too_many_per_cell


def test_qc_flags_clean_entity_passes():
    members = [
        {"sample": f"t{i}", "window_start": 1, "reads": 20, "total_reads": 40}
        for i in range(6)
    ]
    assert not qc_flags("E1", members).failed


def test_qc_flags_detect_horizontal_occupancy():
    """An over-merged entity piles many windows across timepoints without any single
    timepoint holding them all."""
    members = [
        {"sample": f"t{i}", "window_start": i * 10000, "reads": 20, "total_reads": 40}
        for i in range(9)
    ]
    assert qc_flags("E1", members).horizontal_occupancy


# --------------------------------------------------------------------------- #
# Zero-leak
# --------------------------------------------------------------------------- #


def test_unmeasurable_window_returns_none_not_zero():
    """A junk-only or pi-less window has NO measurement; it is not a measurement of 0."""
    assert _window_conditional_abundance(None, 0) is None
    assert _window_conditional_abundance(np.array([1.0]), 0) is None  # pi_junk == 1
    assert _window_conditional_abundance(np.array([0.5, 0.5]), 5) is None  # short vector


def test_measurable_window_is_conditioned_on_non_junk():
    got = _window_conditional_abundance(np.array([0.4, 0.2, 0.4]), 0)
    assert got == pytest.approx(0.4 / 0.6)


# --------------------------------------------------------------------------- #
# Read coordinates
# --------------------------------------------------------------------------- #


def test_read_overlap_bp():
    a = Read(id="a", contig="c1", mapq=60, ref_start=1000, ref_end=5000)
    b = Read(id="b", contig="c1", mapq=60, ref_start=4000, ref_end=9000)
    assert a.overlap_bp(b) == 1000
    far = Read(id="c", contig="c1", mapq=60, ref_start=8000, ref_end=9000)
    assert a.overlap_bp(far) == 0


def test_read_overlap_unknown_is_minus_one_not_zero():
    """Unknown spans must skip the gate, not silently reject every pair."""
    a = Read(id="a", contig="c1", mapq=60)
    b = Read(id="b", contig="c1", mapq=60, ref_start=1, ref_end=100)
    assert a.overlap_bp(b) == -1


# --------------------------------------------------------------------------- #
# SV site types must survive from the VCF to the identity code
# --------------------------------------------------------------------------- #


def test_window_carries_site_type():
    """Window must expose ``site_type`` as a real field. The SV-exclusion rule reads it
    off the Window, and a ``getattr`` default of {} could not tell "no SVs here" from
    "the field is missing", so the exclusion would silently no-op.
    """
    from strainphase.core import Window

    w = Window(contig="c1", start=1, end=100)
    assert hasattr(w, "site_type")
    assert w.site_type == {}

    w.site_type = {10: "snv", 50: "sv"}
    consensuses = [{10: "A", 50: "ev.INV.1"}, {10: "T", 50: "ev.INV.2"}]
    # Exercised with the flag forced on, since the shipping default keeps SVs as markers
    # (see test_sv_sites_are_identity_markers).
    assert variable_marker_positions(
        consensuses, w.site_type, cfg(exclude_sv_from_identity=True)
    ) == {10}
    assert variable_marker_positions(consensuses, w.site_type) == {10, 50}


# --------------------------------------------------------------------------- #
# Window-group fixtures for the cross-sample table tests
# --------------------------------------------------------------------------- #


def _grp(gid, wstart, members):
    """Build a WindowGroup, stamping its members with the ids the pipeline would emit.

    The window coordinates and haplotype id belong to the group, so they are set here
    rather than guessed by ``_mem``; a window-unique id keeps two groups' members from
    colliding.
    """
    from strainphase.longitudinal import _window_haplotype_id
    from strainphase.window_groups import WindowGroup

    per_sample: Counter = Counter()
    for m in members:
        m.window_start = wstart
        m.window_end = wstart + 20000
        m.haplotype_id = _window_haplotype_id(m.sample, "c1", wstart, per_sample[m.sample])
        per_sample[m.sample] += 1
    return WindowGroup(group_id=gid, contig="c1", window_start=wstart,
                       members=members, window_end=wstart + 20000)


def _mem(sample, consensus, reads=30, total=60, wsid="T1", n_read_ids=8, reads_key=None):
    """``wsid`` is the within-sample track this haplotype belongs to.

    Read ids are derived from (sample, wsid): two members of the same track in one
    sample carry the same reads and so continue into each other, while members of
    different tracks share none. ``reads_key`` decouples the two when a test needs
    members that share reads but sit in different tracks. The id and window coordinates
    are placeholders that ``_grp`` overwrites with the group's own.
    """
    return WindowHaplotype(sample=sample, contig="c1", window_start=0, window_end=20000,
                           haplotype_id="", consensus=consensus,
                           reads=reads, total_reads=total, abundance=reads / total,
                           within_sample_id=wsid,
                           read_ids=frozenset(f"{sample}:{reads_key or wsid}:r{i}"
                                              for i in range(n_read_ids)))


def _hid_of(group, sample, idx=0):
    """The id `_grp` gave this sample's ``idx``-th member of ``group``."""
    return [m for m in group.members if m.sample == sample][idx].haplotype_id


def _lcfg(**kw):
    base = {"window_size": 20000, "min_shared_markers": 3,
            "min_entity_overlap_bp": 0, "min_cosupported_span_frac": 0.0}
    base.update(kw)
    return HaplotyperConfig(**base)


















# --------------------------------------------------------------------------- #
# Split molecules: re-assembly and the BREAK marker
# --------------------------------------------------------------------------- #


def _seg(name, rs, re_, alleles):
    from strainphase.core import Read
    r = Read(id=name, contig="c1", mapq=60, ref_start=rs, ref_end=re_)
    r.alleles = dict(alleles)
    r.quals = dict.fromkeys(alleles, 30)
    return r


def test_split_molecule_becomes_one_read_carrying_both_sides():
    """A molecule the aligner split is one observation of one strain, so its segments
    must merge into a single read. Kept apart, the two halves phase into two haplotypes
    and split one strain's weight; merged, the read spans the break.
    """
    from strainphase.core import _merge_split_reads

    merged, breaks = _merge_split_reads([
        _seg("mol1", 1000, 2000, {1100: "A", 1900: "C"}),
        _seg("mol1", 3000, 4000, {3100: "G", 3900: "T"}),
    ])
    assert len(merged) == 1, "two segments of one molecule must not stay two reads"
    r = merged[0]
    assert r.ref_start == 1000 and r.ref_end == 4000, "span must cover the whole molecule"
    for p in (1100, 1900, 3100, 3900):
        assert p in r.alleles, "alleles from BOTH sides must survive the merge"
    assert breaks == {1999}, "break anchors at the last aligned base of the left segment"


def test_break_is_a_discriminating_allele_not_a_hole():
    """BRK<resume> vs CONT is what makes the SV trackable rather than missing data."""
    from strainphase.core import BREAK_PREFIX, CONTINUOUS, _merge_split_reads

    split = [_seg("split", 1000, 2000, {1100: "A"}), _seg("split", 3000, 4000, {3100: "G"})]
    whole = _seg("whole", 1000, 4000, {1100: "A", 3100: "G"})
    merged, breaks = _merge_split_reads([*split, whole])

    by_id = {r.id: r for r in merged}
    bp = breaks.pop()
    assert by_id["split"].alleles[bp] == f"{BREAK_PREFIX}3000"
    # the unbroken read spans the position, so it is evidence AGAINST the event - without
    # this the marker cannot discriminate, since only one side would carry a call
    assert by_id["whole"].alleles[bp] == CONTINUOUS


def test_unsplit_reads_are_untouched_and_overlapping_segments_make_no_break():
    from strainphase.core import _merge_split_reads

    merged, breaks = _merge_split_reads([_seg("solo", 1000, 2000, {1100: "A"})])
    assert len(merged) == 1 and breaks == set()

    # segments that overlap have no unplaceable gap between them
    merged, breaks = _merge_split_reads([
        _seg("m", 1000, 2500, {1100: "A"}),
        _seg("m", 2000, 3000, {2900: "G"}),
    ])
    assert len(merged) == 1 and breaks == set(), "overlapping segments are not a breakpoint"


def test_step1_records_mismatches_but_not_dropouts():
    """Only a genuine allele disagreement is recorded, not a dropout. A dropout is a
    measurement hole, and logging every one buries the real disagreements."""
    from strainphase.core import Haplotype, Window, WindowResult, link_windows

    def wr(start, cons_list):
        w = Window(contig="c1", start=start, end=start + 20000)
        w.snv_pos = sorted({p for c in cons_list for p in c})
        haps = [Haplotype(consensus=dict(c), supporting_reads=20) for c in cons_list]
        return WindowResult(window=w, haplotypes=haps, gamma=np.zeros((1, len(haps) + 1)),
                            pi=np.zeros(len(haps) + 1), log_likelihood=0.0,
                            assignments=[], converged=True, iterations=1)

    shared = {12000: "A", 14000: "C", 16000: "G", 18000: "T"}
    flipped = {p: {"A": "T", "C": "G", "G": "C", "T": "A"}[b] for p, b in shared.items()}
    results = link_windows([wr(1, [shared]), wr(10001, [flipped])], cfg())

    rec = [m for r in results for m in r.link_mismatches]
    assert rec, "a genuine disagreement must reach the output"
    assert rec[0]["n_diff"] > 0 and rec[0]["n_shared"] >= 3

    # identical haplotypes link and record nothing
    quiet = link_windows([wr(1, [shared]), wr(10001, [dict(shared)])], cfg())
    assert not [m for r in quiet for m in r.link_mismatches]




def test_step1_refuses_to_link_incompatible_abundances():
    """Step 1 uses abundance only as an eliminator. Two adjacent windows in one sample
    are the same timepoint, so a genome cannot sit at 95% in one and 5% in the next;
    identical alleles are not enough to link such a pair. Agreement never scores, and
    the test runs on raw counts because the derived abundance is quantised by a small
    median denominator.
    """
    from strainphase.core import Haplotype, Window, WindowResult, link_windows

    shared = {12000: "A", 14000: "C", 16000: "G", 18000: "T"}

    # A decoy second haplotype so the marker set is non-empty: markers are positions
    # that vary across a sample's haplotypes, and one identical haplotype per window
    # would leave nothing to compare. Real multi-strain data always supplies this.
    decoy = {p: "G" for p in shared}

    def wr(start, reads, n_junk=0, n_total=100):
        w = Window(contig="c1", start=start, end=start + 20000)
        w.snv_pos = sorted(shared)
        g = np.zeros((n_total, 3))
        g[:n_total - n_junk, 0] = 1.0
        g[n_total - n_junk:, 2] = 1.0
        return WindowResult(
            window=w,
            haplotypes=[Haplotype(consensus=dict(shared), supporting_reads=reads),
                        Haplotype(consensus=dict(decoy), supporting_reads=1)],
            gamma=g, pi=np.array([1.0, 0.0, 0.0]), log_likelihood=0.0,
            assignments=[], converged=True, iterations=1)

    # 95/100 next to 5/100 - alleles identical, shares incompatible
    a = link_windows([wr(1, 95), wr(10001, 5)], cfg())
    assert len({r.haplotypes[0].track_id for r in a}) == 2, \
        "incompatible shares must not be linked"

    # same alleles, compatible shares -> linked
    b = link_windows([wr(1, 95), wr(10001, 93)], cfg())
    assert len({r.haplotypes[0].track_id for r in b}) == 1




# --------------------------------------------------------------------------- #
# Two members of one sample in one group
# --------------------------------------------------------------------------- #
# Most fixtures here put one member per (sample, window). The pipeline also produces
# two members per cell routinely: merge_similar_haplotypes declines some 1-SNV pairs
# and both halves survive as a real strain split. coherence.py flags three per cell,
# not two.


def _split_grp(gid, wstart, samples, reads_each, total, wsid="T1"):
    """A group where every sample contributes two near-identical members.

    ``reads_each`` is a per-member pair; the two halves of a split are rarely equal. The
    denominator is one window's non-junk total, carried identically on every member of
    that (sample, window) cell, since it is a property of the window, not the member.
    """
    members = []
    for s in samples:
        for k, reads in enumerate(reads_each):
            members.append(_mem(s, {12000: "A", 15000: "C", 18000: "G"},
                                reads=reads, total=total, wsid=f"{wsid}{k}"))
    return _grp(gid, wstart, members)














def test_load_snvs_is_cached_across_samples():
    """A longitudinal run calls process_contig once per sample, and under a cohort union
    VCF every call parses the identical file, so ``load_snvs`` caches the result. The
    cache is keyed on the settings that change what is kept, so a config change re-parses.
    """
    from strainphase import core

    calls = {"n": 0}
    real = core._load_snvs_uncached

    # The real loader returns eight tables, so the stub must too.
    def counting(*a, **k):
        calls["n"] += 1
        return ([1], {1: "A"}, {1: 9}, {1: 0.5}, {1: "snv"}, {1: frozenset({"alt"})},
                {1: {(1, 2)}}, {1: {3}})

    core._SNV_CACHE.clear()
    core._load_snvs_uncached = counting
    try:
        for _ in range(5):
            core.load_snvs("dummy.vcf.gz", "c1", None, cfg())
        assert calls["n"] == 1, "identical calls must parse once"

        core.load_snvs("dummy.vcf.gz", "c1", None, cfg(min_depth_site=99))
        assert calls["n"] == 2, "a gate change must re-parse"

        core.load_snvs("dummy.vcf.gz", "c2", None, cfg())
        assert calls["n"] == 3, "a different contig must re-parse"
    finally:
        core._load_snvs_uncached = real
        core._SNV_CACHE.clear()


def test_a_cached_load_snvs_caller_cannot_reach_the_next_callers_tables():
    """Each ``load_snvs`` call gets its own containers, hit or miss. process_contig
    appends SV pseudo-sites to the returned tables, so handing back the cached objects
    would let one sample's SV anchors reach the next sample under the same union VCF,
    where they read as collisions and its own SV sites are dropped.
    """
    from strainphase import core

    real = core._load_snvs_uncached

    def stub(*a, **k):
        return ([100], {100: "A"}, {100: 30}, {100: 0.5}, {100: "snv"},
                {100: frozenset({"T"})}, {}, {})

    core._SNV_CACHE.clear()
    core._load_snvs_uncached = stub
    try:
        first = core.load_snvs("union.vcf.gz", "c1", None, cfg())
        # Sample 1 does what process_contig does: append an SV anchor and claim the site.
        first[0].append(500)
        first[1][500] = "N"
        first[4][500] = "sv"

        second = core.load_snvs("union.vcf.gz", "c1", None, cfg())
        assert second[0] == [100], "sample 2 sees sample 1's appended SV anchor"
        assert 500 not in second[1] and 500 not in second[4]
        assert all(a is not b for a, b in zip(first, second) if isinstance(a, (list, dict)))
    finally:
        core._load_snvs_uncached = real
        core._SNV_CACHE.clear()




def test_a_run_writes_lineages_tsv_and_puts_diagnostics_in_tmp(tmp_path):
    """End to end: ``build_window_tables`` -> ``write_window_tables`` writes the
    deliverable tables in output_dir and the diagnostics in output_dir/tmp. Guards the
    wiring that produces the lineage output, not the algorithm.
    """
    import numpy as np

    from strainphase.core import Haplotype, Window, WindowResult
    from strainphase.longitudinal import build_window_tables, write_window_tables

    shared = {12000: "A", 15000: "C", 18000: "G"}

    def wr(start, sample, extra):
        w = Window(contig="c1", start=start, end=start + 20000)
        w.snv_pos = sorted({**shared, **extra})
        g = np.zeros((20, 2))
        g[:18, 0] = 1.0
        g[18:, 1] = 1.0
        h = Haplotype(consensus={**shared, **extra}, supporting_reads=9)
        h.track_id = "T0001"
        return WindowResult(window=w, haplotypes=[h], gamma=g,
                            pi=np.array([0.9, 0.1]), log_likelihood=0.0,
                            assignments=[], converged=True, iterations=1)

    all_results = {"MAG1": {s: {"c1": [wr(1, s, {3000: "T"}), wr(10001, s, {25000: "T"})]}
                            for s in ("t0", "t1", "t2")}}
    out = str(tmp_path / "run")
    rows = build_window_tables(out, all_results, _lcfg(), sample_order=["t0", "t1", "t2"])
    write_window_tables(rows[0], rows[1], rows[2], rows[3], out, rows[4], rows[6])

    assert os.path.exists(os.path.join(out, "lineages.tsv")), "the deliverable must exist"
    for name in ("haplotypes.tsv", "windows_within_sample.tsv",
                 "windows_across_samples.tsv"):
        assert os.path.exists(os.path.join(out, name))
    for name in ("mismatches_within_sample.tsv", "mismatches_across_samples.tsv"):
        assert os.path.exists(os.path.join(out, "tmp", name)), f"{name} belongs in tmp/"
        assert not os.path.exists(os.path.join(out, name)), f"{name} must NOT be a deliverable"

    with open(os.path.join(out, "lineages.tsv")) as f:
        head, *body = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    assert body, "lineages.tsv must not be empty"
    for col in ("lineage_id", "sample", "abundance", "abundance_all_reads",
                "reads", "total_reads", "junk_reads", "haplotype_ids"):
        assert col in head, f"lineages.tsv is missing {col}"
    # one row per (lineage, sample); all three samples represented
    assert {r[head.index("sample")] for r in body} == {"t0", "t1", "t2"}








# ---------------------------------------------------------------------------
# Read-overlap threading
#
# link_windows joins haplotypes across windows on the reads they share. These
# fixtures cover the behaviours that matter: an over-split strain is rejoined,
# the consensus gate still vetoes first, and no shared-read evidence means no join.
# ---------------------------------------------------------------------------

_OVERLAP_REGION = range(5000, 8000, 50)   # inside the [0,10000]/[5000,15000] overlap


def _ro_hap(sample, window, hap_id, read_ids, consensus=None, reads=50):
    from strainphase.window_groups import WindowHaplotype

    return WindowHaplotype(
        sample=sample, contig="c1", window_start=window, window_end=window + 10000,
        haplotype_id=hap_id,
        consensus=dict(consensus or {p: "A" for p in _OVERLAP_REGION}),
        reads=reads, total_reads=100, junk_reads=0, abundance=reads / 100,
        within_sample_id="", read_ids=frozenset(read_ids),
    )


def _ro_group(group_id, window, members):
    from strainphase.window_groups import WindowGroup

    return WindowGroup(group_id=group_id, contig="c1", window_start=window,
                       window_end=window + 10000, members=members)




def _ro_contested_groups():
    """One strain over-split into two groups at w=0, both feeding one group at w=5000."""
    shared = {f"r{i}" for i in range(1, 9)}
    return [
        _ro_group("g_A1", 0, [_ro_hap("S1", 0, "hA1", {"r1", "r2", "r3", "r4"})]),
        _ro_group("g_A2", 0, [_ro_hap("S1", 0, "hA2", {"r5", "r6", "r7", "r8"})]),
        _ro_group("g_B", 5000, [_ro_hap("S1", 5000, "hB", shared, reads=100)]),
    ]








def test_offload_preserves_read_assignments():
    """offload_heavy drops reads but must keep the assignments they produced."""
    from strainphase.core import WindowResult

    wr = WindowResult.__new__(WindowResult)
    wr.assignments = [{"read_id": "r1", "hap_id": 0}]
    wr.heavy_offloaded = False
    wr.gamma = None
    wr.n_reads_total = wr.n_junk_reads = -1

    class _W:
        reads = ["read-object"]
        _pos_sets = None
    wr.window = _W()

    wr.offload_heavy()
    assert wr.window.reads == []
    assert wr.assignments == [{"read_id": "r1", "hap_id": 0}]


def test_consistent_subsampling_preserves_overlap_between_windows():
    """The read cap must not destroy the shared reads the linker joins on.

    Adjacent windows overlap by 50%, so they share half their molecules. Drawing an
    independent random subsample in each keeps a shared read in BOTH only with
    probability (cap/N)^2, which at a 200-read cap over a few thousand reads leaves
    ~1-3 shared reads where the biology has hundreds - read-overlap linking then
    reports a sampling artefact as a linking failure. Selecting on a stable hash of
    the read id keeps the loss linear.
    """
    import numpy as np

    from strainphase.core import _read_sort_hash

    cap, n, seed = 200, 6000, 42
    rng = np.random.default_rng(0)
    window_a = [f"read{i}" for i in range(n)]
    window_b = [f"read{i}" for i in range(n // 2, n // 2 + n)]   # 50% overlap

    independent = len(
        set(np.array(window_a)[rng.permutation(n)[:cap]])
        & set(np.array(window_b)[rng.permutation(n)[:cap]])
    )
    consistent = len(
        set(sorted(window_a, key=lambda r: _read_sort_hash(r, seed))[:cap])
        & set(sorted(window_b, key=lambda r: _read_sort_hash(r, seed))[:cap])
    )
    assert independent < 10           # the artefact being fixed
    assert consistent > cap // 4      # enough shared reads to link on
    assert consistent > independent * 5


def test_read_sort_hash_is_stable_and_seed_dependent():
    """Must not use hash(): str hashing is salted per process, so the subsample
    would differ between runs and between workers of the same run."""
    from strainphase.core import _read_sort_hash

    assert _read_sort_hash("read1", 42) == _read_sort_hash("read1", 42)
    assert _read_sort_hash("read1", 42) != _read_sort_hash("read1", 7)



def test_assign_reads_records_best_hap_below_confidence_threshold():
    """Linking needs the argmax haplotype even when it is not confidently called.
    ``hap_id`` is withheld below ``assign_confidence_threshold``, but read-overlap
    linking only asks whether a molecule sits in both windows, and near-identical
    strains leave most reads at gamma 0.6-0.89, so ``best_hap`` is recorded regardless.
    """
    import numpy as np

    from strainphase.core import DEFAULT_CONFIG, PostProcessor

    post = PostProcessor.__new__(PostProcessor)
    post.config = replace(DEFAULT_CONFIG,
                          assign_confidence_threshold=0.90)

    class _R:
        def __init__(self, rid):
            self.id = rid

    # read0 confident on hap0, read1 ambiguous between hap0/hap1, read2 junk
    gamma = np.array([
        [0.97, 0.02, 0.01],
        [0.70, 0.28, 0.02],
        [0.05, 0.05, 0.90],
    ])
    out = post.assign_reads([_R("r0"), _R("r1"), _R("r2")], gamma)
    by_id = {a["read_id"]: a for a in out}

    assert by_id["r0"]["hap_id"] == 0 and by_id["r0"]["best_hap"] == 0
    # the case that matters: no confident call, but the argmax is still recorded
    assert by_id["r1"]["hap_id"] is None
    assert by_id["r1"]["is_ambiguous"] is True
    assert by_id["r1"]["best_hap"] == 0
    # junk contributes no link
    assert by_id["r2"]["is_junk"] is True
    assert by_id["r2"]["best_hap"] is None


# --- forward-strict best target, many-to-one still allowed --------------------


























# ---------------------------------------------------------------------------
# Redundant parallel lineages (observed on B. fragilis 000089747_1, upeY locus)
#
# One real run reported four lineages (LIN000290/284/285/286) all >=98.8%
# identical, spanning the same ~1.37-1.41 Mb interval and co-occurring in 14-32
# of the same samples. They are one strain reported four times: once a strain is
# split across parallel groups at the same window, nothing rejoins them.
# ---------------------------------------------------------------------------


def _par_groups():
    """Two parallel, consensus-identical chains over the same two windows."""
    reads = {"A1": {"r1", "r2", "r3", "r4"}, "A2": {"r5", "r6", "r7", "r8"}}
    return [
        _ro_group("A1", 0, [_ro_hap("S1", 0, "a1", reads["A1"])]),
        _ro_group("A2", 0, [_ro_hap("S1", 0, "a2", reads["A2"])]),
        _ro_group("B1", 5000, [_ro_hap("S1", 5000, "b1", reads["A1"])]),
        _ro_group("B2", 5000, [_ro_hap("S1", 5000, "b2", reads["A2"])]),
    ]






def _group_consensus_of(lineage):
    """Majority allele per position across every member of every group."""
    from collections import Counter, defaultdict

    votes = defaultdict(Counter)
    for g in lineage.groups:
        for m in g.members:
            for pos, base in m.consensus.items():
                votes[pos][base] += 1
    return {p: c.most_common(1)[0][0] for p, c in votes.items()}


# ---------------------------------------------------------------------------
# How two pure step-1 tracks end up in one over-merged lineage
#
# Observed on div0050_k2 T3 (truth 0.663/0.337): step 1 produced two pure tracks,
# and both landed in one lineage of purity 0.653, taking assigned_correct_fraction
# from 0.973 to 0.000. The test below pins the compare_consensus behaviour that
# would let that happen, and asserts the current (fixed) behaviour.
# ---------------------------------------------------------------------------

_FUSION_DISCRIM = [5000, 5500]                      # where two strains differ
_FUSION_INVAR = list(range(6000, 20000, 100))       # identical in both


def _fusion_haps():
    a = {**{p: "A" for p in _FUSION_DISCRIM}, **{p: "G" for p in _FUSION_INVAR}}
    b = {**{p: "T" for p in _FUSION_DISCRIM}, **{p: "G" for p in _FUSION_INVAR}}
    return a, b


def test_a_real_difference_is_not_diluted_by_uninformative_positions():
    """Two haplotypes differing at every discriminating marker they share must not be
    declared identical. Only the discriminating markers are compared; the removed clonal
    fallback used to pad with invariant co-covered positions, diluting 2 real differences
    over 2 markers into 2 over 142 and clearing the 2% gate.
    """
    from strainphase.core import compare_consensus

    a, b = _fusion_haps()
    markers = set(_FUSION_DISCRIM)
    config = cfg()

    gate = compare_consensus(a, b, markers, config)

    # Only the 2 DISCRIMINATING markers are compared, and they both disagree, so there
    # is no verdict rather than a link. The removed fallback re-scored this as 2
    # differences over 142 co-covered positions (rate 0.014) and called it identical.
    assert gate.reason == "failed_no_evidence"
    assert gate.n_shared == 2






# ---------------------------------------------------------------------------
# Read-supported marker set
# ---------------------------------------------------------------------------


def test_supported_markers_keep_a_swept_position():
    """A swept position is FIXED within every sample - all one allele before the
    sweep, all the other after - so a within-sample minor-allele test finds no
    polymorphism and would discard exactly the event being tracked. Each allele is
    therefore counted across samples INDEPENDENTLY.
    """
    from strainphase.core import supported_marker_positions

    obs = [("S1", {10: "A"}, 20), ("S2", {10: "A"}, 20),      # allele A, 2 samples
           ("S3", {10: "T"}, 20), ("S4", {10: "T"}, 20)]      # allele T, 2 samples
    assert supported_marker_positions(obs, None, cfg()) == frozenset({10})


def test_supported_markers_drop_unreplicated_and_invariant():
    """Read support AND replication: one shallow miscall must not make a marker."""
    from strainphase.core import supported_marker_positions

    obs = [
        ("S1", {20: "C", 30: "G"}, 20),
        ("S2", {20: "C", 30: "G"}, 20),
        ("S3", {20: "C", 30: "A"}, 1),    # 30: minor allele, 1 read, 1 sample
    ]
    markers = supported_marker_positions(obs, None, cfg())
    assert 20 not in markers, "invariant position is not a marker"
    assert 30 not in markers, "one unreplicated read is not a marker"


def test_supported_markers_need_two_samples_per_allele():
    from strainphase.core import supported_marker_positions

    obs = [("S1", {10: "A"}, 20), ("S2", {10: "A"}, 20), ("S3", {10: "T"}, 20)]
    assert supported_marker_positions(obs, None, cfg()) == frozenset()
    obs.append(("S4", {10: "T"}, 20))
    assert supported_marker_positions(obs, None, cfg()) == frozenset({10})


# ---------------------------------------------------------------------------
# Step-1 abundance refusals carry over into the cross-sample merge
# ---------------------------------------------------------------------------






# --------------------------------------------------------------------------- #
# Step-1 read reach (link_window_reach)
# --------------------------------------------------------------------------- #


def _reach_windows():
    """Two NON-overlapping windows, A=[1,20001) and C=[20001,40001).

    They call disjoint positions, so consensus agreement between them is not merely
    unknown - it is undefined, and no amount of marker comparison can produce a verdict.
    Reads carried by a haplotype in both windows are the only evidence there is.

    Read layout pairs A0 with C0 and A1 with C1, and gives each pair a clear margin over
    the crossing alternative so the best-match linker has a unique winner to choose.
    """
    from strainphase.core import Haplotype, Window, WindowResult

    def wr(start, pos, reads_by_hap):
        w = Window(contig="c1", start=start, end=start + 20000)
        w.snv_pos = sorted(pos)
        g = np.zeros((100, 3))
        g[:, 0] = 1.0
        haps = [
            Haplotype(consensus=dict(pos), supporting_reads=50),
            Haplotype(consensus={p: "T" for p in pos}, supporting_reads=50),
        ]
        assignments = [
            {"best_hap": h, "read_id": r}
            for h, rs in enumerate(reads_by_hap)
            for r in rs
        ]
        return WindowResult(
            window=w, haplotypes=haps, gamma=g, pi=np.array([0.5, 0.5, 0.0]),
            log_likelihood=0.0, assignments=assignments, converged=True, iterations=1,
        )

    a = wr(1, {2000: "A", 4000: "C", 6000: "G"},
           [["r1", "r2", "r3", "r4"], ["r5", "r6", "r7", "r8"]])
    c = wr(20001, {22000: "A", 24000: "C", 26000: "G"},
           [["r1", "r2", "r3", "r9"], ["r5", "r6", "r7", "r10"]])
    return a, c


def test_reach_one_restores_the_overlap_only_rule():
    """Reach 1 restricts linking to overlapping windows, the escape hatch from the
    read-reach default. A disjoint pair must be refused outright rather than falling
    through to the read path.
    """
    from strainphase.core import link_windows

    a, c = _reach_windows()
    out = link_windows([a, c], cfg(link_window_reach=1))
    ids = {h.track_id for wr_ in out for h in wr_.haplotypes}
    assert len(ids) == 4, "reach 1 must not link a non-overlapping pair"


def test_read_reach_is_on_by_default():
    """The default ``link_window_reach`` is 2, reaching one window past the overlap.
    Pinned because a silent revert to 1 would look like the read path simply finding
    nothing.
    """
    from strainphase.core import HaplotyperConfig, link_windows

    assert HaplotyperConfig().link_window_reach == 2

    a, c = _reach_windows()
    out = link_windows([a, c], cfg())
    tracks = [[h.track_id for h in wr_.haplotypes] for wr_ in out]
    assert tracks[0][0] == tracks[1][0] and tracks[0][1] == tracks[1][1]


def test_read_reach_links_across_a_non_overlapping_gap():
    """With reach 2, shared reads link A to C even though no position is shared.

    This is the case the overlap rule could not express: `link_windows` stopped at
    `next.start >= curr.end`, so with 20 kb windows on a 10 kb step it never reached
    past its immediate neighbour - a 10 kb horizon against reads that reach 30 kb.
    """
    from strainphase.core import link_windows

    a, c = _reach_windows()
    out = link_windows([a, c], cfg(link_window_reach=2, link_min_shared_reads=2))
    tracks = [[h.track_id for h in wr_.haplotypes] for wr_ in out]

    assert tracks[0][0] == tracks[1][0], "A0 and C0 share 3 reads and must link"
    assert tracks[0][1] == tracks[1][1], "A1 and C1 share 3 reads and must link"
    assert tracks[0][0] != tracks[0][1], "the two strains must stay apart"


def test_read_reach_respects_the_shared_read_threshold():
    """Below ``link_min_shared_reads`` the pair is not linked. Consensus cannot veto a
    disjoint pair, so this threshold is the evidence bar between read chaining and
    fusing two strains on a single mis-assigned read.
    """
    from strainphase.core import link_windows

    a, c = _reach_windows()
    out = link_windows([a, c], cfg(link_window_reach=2, link_min_shared_reads=4))
    ids = {h.track_id for wr_ in out for h in wr_.haplotypes}
    assert len(ids) == 4, "3 shared reads must not clear a threshold of 4"


def test_read_reach_rejects_invalid_settings():
    from strainphase.core import HaplotyperConfig

    with pytest.raises(ValueError, match="link_window_reach"):
        HaplotyperConfig(link_window_reach=0)
    with pytest.raises(ValueError, match="link_min_shared_reads"):
        HaplotyperConfig(link_min_shared_reads=0)


# --------------------------------------------------------------------------- #
# The 1-SNV validator: it has to RUN, and evidence has to beat the floor
# --------------------------------------------------------------------------- #


def _snv_pair(minor_weight, minor_reads, n_shared, n_timepoints=1, depth=130):
    """Two haplotypes differing at exactly ONE of `n_shared` called positions.

    Mirrors the real shape: a dominant haplotype and a minority splitting off a few
    reads. `gamma` is built so the minority has exactly `minor_reads` confidently
    assigned, which is what the validator counts.
    """
    from strainphase.core import Haplotype, PostProcessor, Window

    pos = [1000 + 10 * i for i in range(n_shared)]
    major = {p: "A" for p in pos}
    minor = dict(major)
    minor[pos[0]] = "G"                      # the single difference

    w = Window(contig="c1", start=1, end=20001)
    w.snv_pos = list(pos)
    w.reads = []

    g = np.zeros((depth, 2))
    g[:depth - minor_reads, 0] = 1.0
    g[depth - minor_reads:, 1] = 1.0

    h_major = Haplotype(consensus=major, supporting_reads=depth - minor_reads)
    h_minor = Haplotype(consensus=minor, supporting_reads=minor_reads)
    h_major.weight = 1.0 - minor_weight
    h_minor.weight = minor_weight

    hp = PostProcessor.__new__(PostProcessor)
    hp.config = cfg(marker_min_frac=0.10, marker_min_reads=3, marker_min_samples=2)
    return hp, h_major, h_minor, w, g, n_timepoints


def test_1snv_validator_runs_when_the_rate_would_reject():
    """7 shared positions with 1 difference is a 14% rate, above identity_distance, yet
    the 1-SNV validator still runs and judges the pair on read counts rather than the
    rate. The shape is 3 reads of 130 splitting off a dominant haplotype.
    """
    hp, maj, minor, w, g, ntp = _snv_pair(minor_weight=0.023, minor_reads=3, n_shared=7)
    assert 1 / 7 > hp.config.identity_distance, "the rate must reject, or this proves nothing"
    # 3 reads is not < marker_min_reads, so read support passes; 1 timepoint sends it
    # to the binomial, and with no reads recorded the guard merges.
    assert hp.should_merge_1snp_pair(maj, minor, 0, 1, w, g, ntp) is True


def test_1snv_thin_read_support_still_merges():
    """2 reads is below marker_min_reads: still noise, still merged."""
    hp, maj, minor, w, g, ntp = _snv_pair(minor_weight=0.30, minor_reads=2, n_shared=7)
    assert hp.should_merge_1snp_pair(maj, minor, 0, 1, w, g, ntp) is True


def test_1snv_persistent_minor_allele_is_kept_apart():
    """A credible minor strain survives: >=3 reads, seen across timepoints. Persistence
    short-circuits before the binomial, and the frequency floor does not fire because
    0.30 > marker_min_frac.
    """
    hp, maj, minor, w, g, ntp = _snv_pair(minor_weight=0.30, minor_reads=45,
                                          n_shared=52, n_timepoints=3)
    assert hp.should_merge_1snp_pair(maj, minor, 0, 1, w, g, ntp) is False


def test_1snv_floor_is_a_backstop_not_a_pre_emption():
    """A sub-10% haplotype with real read support reaches the later tests. The frequency
    floor still applies, but only after read support, persistence, and the binomial have
    spoken, so a persistent sub-10% allele is judged rather than assumed to be noise.
    """
    hp, maj, minor, w, g, ntp = _snv_pair(minor_weight=0.048, minor_reads=8,
                                          n_shared=52, n_timepoints=3)
    # persistence is satisfied, so the run reaches the floor at the END and merges
    assert hp.should_merge_1snp_pair(maj, minor, 0, 1, w, g, ntp) is True
    # ...whereas the same allele above the floor is kept apart
    hp2, maj2, minor2, w2, g2, ntp2 = _snv_pair(minor_weight=0.12, minor_reads=8,
                                                n_shared=52, n_timepoints=3)
    assert hp2.should_merge_1snp_pair(maj2, minor2, 0, 1, w2, g2, ntp2) is False


def test_supported_markers_keep_a_low_abundance_allele_with_read_support():
    """An allele qualifies as a marker on read support (``n >= marker_min_reads``) even
    when far below ``marker_min_frac``. The rule is reads OR frequency, not AND, so a
    strain at 2-5% can still contribute a marker: 10 reads at 2% is a better-supported
    call than 3 reads at 15%.
    """
    from strainphase.core import supported_marker_positions

    # position 40: major allele at 490 reads, minor at 10 (2%) in two samples
    obs = [
        ("S1", {40: "C"}, 490), ("S1", {40: "T"}, 10),
        ("S2", {40: "C"}, 490), ("S2", {40: "T"}, 10),
    ]
    assert 40 in supported_marker_positions(obs, None, cfg()), \
        "2% with 10 reads in 2 samples must qualify on read support"


def test_supported_markers_still_reject_thin_low_abundance_alleles():
    """OR is not a free pass: below marker_min_reads AND below the frequency floor
    an allele still fails, which is what keeps single stray reads out."""
    from strainphase.core import supported_marker_positions

    obs = [
        ("S1", {50: "C"}, 490), ("S1", {50: "T"}, 2),   # 2 reads, 0.4%
        ("S2", {50: "C"}, 490), ("S2", {50: "T"}, 2),
    ]
    assert 50 not in supported_marker_positions(obs, None, cfg()), \
        "2 reads at 0.4% clears neither bar and must not qualify"


# --------------------------------------------------------------------------- #
# The within-window merge compares over INFORMATIVE positions, without a rate
# --------------------------------------------------------------------------- #


def _window_haps(pairs, n_invariant, depth=100):
    """Haplotypes differing at ``pairs`` positions, padded with ``n_invariant`` agreeing
    ones. The padding is what ``window.snv_pos`` accumulates as coverage rises, and what
    used to dilute the merge rate.
    """
    from strainphase.core import Haplotype, PostProcessor, Window

    var = {1000 + 10 * i: ("A", "G") for i in range(pairs)}
    inv = {5000 + 10 * i: "C" for i in range(n_invariant)}
    a = {**{p: v[0] for p, v in var.items()}, **inv}
    b = {**{p: v[1] for p, v in var.items()}, **inv}
    w = Window(contig="c1", start=1, end=20001)
    w.snv_pos = sorted(set(a) | set(b))
    w.reads = []
    h1, h2 = Haplotype(consensus=a, supporting_reads=depth // 2), \
             Haplotype(consensus=b, supporting_reads=depth // 2)
    h1.weight = h2.weight = 0.5
    g = np.zeros((depth, 2))
    g[: depth // 2, 0] = 1.0
    g[depth // 2:, 1] = 1.0
    hp = PostProcessor.__new__(PostProcessor)
    hp.config = cfg(min_shared_markers=2)
    return hp, [h1, h2], g, np.array([0.5, 0.5]), w


def test_window_merge_is_not_diluted_by_agreeing_positions():
    """Two real differences survive however many agreeing sites surround them, because
    the merge compares informative positions rather than a rate over ``window.snv_pos``.
    The old rate diluted 2 differences among 400 agreeing sites to 0.005, below
    identity_distance, and merged harder as coverage added more sites.
    """
    for n_invariant in (10, 100, 400):
        hp, haps, g, pi, w = _window_haps(pairs=2, n_invariant=n_invariant)
        out, _, _ = hp.merge_similar_haplotypes(haps, g, pi, w)
        assert len(out) == 2, (
            f"2 real differences must not merge with {n_invariant} agreeing sites; "
            "the decision must not depend on how many positions were called"
        )


def test_window_merge_still_collapses_identical_haplotypes():
    """Byte-identical on informative positions still merges - the rule is not just
    'never merge'. With no differing site there is nothing to keep apart."""
    hp, haps, g, pi, w = _window_haps(pairs=0, n_invariant=50)
    out, _, _ = hp.merge_similar_haplotypes(haps, g, pi, w)
    assert len(out) == 1, "identical haplotypes must still collapse"


def test_window_merge_defers_one_difference_to_the_validator():
    """Exactly ONE differing position is the validator's case, not the rate's."""
    hp, haps, g, pi, w = _window_haps(pairs=1, n_invariant=100, depth=130)
    haps[1].weight = 0.02          # 2% minor: below marker_min_frac
    haps[0].weight = 0.98
    out, _, _ = hp.merge_similar_haplotypes(haps, g, pi, w)
    assert len(out) == 1, "a 1-SNV difference on a sub-threshold minor must merge"
