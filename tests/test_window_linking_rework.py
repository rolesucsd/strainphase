#!/usr/bin/env python3
"""Tests for the window-linking rework.

Covers the identity gate stack, the marker set (positions that actually vary; a
naive implementation gets wrong), both cross-sample grouping shapes, the abundance
coherence test, the QC gate, and the zero-leak fix.
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
    """AUTHOR'S DECISION: SVs are NEVER excluded from identity. Capturing the trajectory
    of a flip is a goal of the analysis, so an inversion is a marker like any other. The
    flip then shows up as two entities trading frequency, which IS the trajectory.

    The exclusion path is still reachable so the effect can be measured, but it must not
    be the default - it was set True in code while diagnosis §6 #16 was still open.
    """
    consensuses = [{10: "A", 99: "ev.INV.1"}, {10: "T", 99: "ev.INV.2"}]
    site_type = {10: "snv", 99: "sv"}
    assert variable_marker_positions(consensuses, site_type) == {10, 99}
    dropped = variable_marker_positions(
        consensuses, site_type, cfg(exclude_sv_from_identity=True)
    )
    assert dropped == {10}


def test_marker_set_is_empty_for_a_clonal_locus():
    """The precondition for the fallback below: a clonal sample has NO variable sites."""
    consensuses = [{10: "A", 20: "C"}, {10: "A", 20: "C"}]
    assert variable_marker_positions(consensuses) == set()


# --------------------------------------------------------------------------- #
# Gate stack
# --------------------------------------------------------------------------- #


def test_no_discriminating_markers_means_no_verdict():
    """With no discriminating markers there is no verdict - not a link.

    A "clonal fallback" used to compare all co-covered positions here instead, so two
    haplotypes with nothing informative in common were declared identical. That is
    what let two distinct strains merge: positions invariant across every haplotype
    cannot disagree, so padding with them only ever dilutes a real difference. The
    cost of removing it is that genuinely clonal loci no longer link on absence of
    evidence, which is the honest reading.
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
    """REGRESSION: the overlap gate asks how much SEQUENCE was compared, never how
    spread out the markers are.

    It used to measure ``max(shared) - min(shared)`` over the MARKER subset. On
    000066952_0 window 1,880,001 that rejected all 703 pairs - each overlapped by
    6,126 bp, but its only 2 markers sat 190 bp apart - and emitted a 2-genotype locus
    as 38 singleton groups. Clustered variation (recombination tracts, hypervariable
    loci) is exactly where markers bunch up, so the old rule penalised the most
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
    """A tied best match is ambiguous and must yield nothing -- this is what bounds
    entity size by construction rather than by tuning."""
    assert unique_best_matches({0: [(0.0, 1)]}) == {0: 1}
    assert unique_best_matches({0: [(0.0, 1), (0.0, 2)]}) == {}
    assert unique_best_matches({0: [(0.0, 1), (0.5, 2)]}) == {0: 1}


# --------------------------------------------------------------------------- #
# Cross-sample grouping
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
    """REGRESSION: Window must expose site_type.

    The SV-exclusion rule reads it off the Window. When the field did not exist, the
    lookup used a getattr default and silently returned {} - so the exclusion no-opped
    and inversions were being used as identity markers after all. A getattr with a
    default cannot distinguish "no SVs here" from "the plumbing is missing".
    """
    from strainphase.core import Window

    w = Window(contig="c1", start=1, end=100)
    assert hasattr(w, "site_type")
    assert w.site_type == {}

    w.site_type = {10: "snv", 50: "sv"}
    consensuses = [{10: "A", 50: "ev.INV.1"}, {10: "T", 50: "ev.INV.2"}]
    # The plumbing must REACH the exclusion path - a getattr default returning {} could
    # not be told apart from "no SVs here". Exercised with the flag forced on, since the
    # shipping default keeps SVs as markers (see test_sv_sites_are_identity_markers).
    assert variable_marker_positions(
        consensuses, w.site_type, cfg(exclude_sv_from_identity=True)
    ) == {10}
    assert variable_marker_positions(consensuses, w.site_type) == {10, 50}


# --------------------------------------------------------------------------- #
# STEP 3: chaining window groups into lineages
# --------------------------------------------------------------------------- #


def _grp(gid, wstart, members):
    """Build a WindowGroup, giving its members the ids the pipeline would emit.

    The window coordinates and the haplotype id belong to the GROUP, so they are stamped
    here rather than guessed by `_mem`. Before that, `_mem` spelled every id
    `sample|h` with no window component, so two groups' members collided: a veto set of
    `frozenset(("t0|h", "t0|h"))` is the degenerate ONE-element frozenset, which is a
    state the pipeline cannot produce and which made the step-1 veto test pass on the
    collision instead of on the veto.
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
    """`wsid` is the step-1 chain this haplotype belongs to.

    Since 2026-08-30 step 3 links on SHARED READS, not on chain votes, so the read ids
    are derived from (sample, wsid): two members of the same chain in the same sample
    carry the SAME reads and therefore continue into each other, which is exactly what
    `wsid` meant before. Members of different chains share none.

    `reads_key` decouples the two when a test needs groups that LINK (shared reads) but
    sit in DIFFERENT step-1 tracks - which is also the real case, since a read spanning
    a window seam belongs to both windows whatever track they ended up in. Needed to
    exercise the abundance checks, which the track-preserving union outranks.

    The id and window coordinates are placeholders; `_grp` overwrites them with the
    group's own, so a member always carries an id unique to its (sample, window, index).
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
# SPLIT MOLECULES: re-assembly + the BREAK marker (troubleshooting U1)
# --------------------------------------------------------------------------- #


def _seg(name, rs, re_, alleles):
    from strainphase.core import Read
    r = Read(id=name, contig="c1", mapq=60, ref_start=rs, ref_end=re_)
    r.alleles = dict(alleles)
    r.quals = dict.fromkeys(alleles, 30)
    return r


def test_split_molecule_becomes_one_read_carrying_both_sides():
    """The whole point: a molecule the aligner split is ONE observation of ONE strain.

    Kept apart, its two halves phase into two haplotypes and the window's mixture weight
    is divided between fragments of the same strain. Merged, the read spans the break and
    the halves cannot come apart.
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
    """Only a genuine allele disagreement is reported. A dropout is a measurement hole -
    recording those is what made the full comparison log unaffordable, and it buries the
    finding."""
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
    """ABUNDANCE AS AN ELIMINATOR AT STEP 1 (author's rule, 2026-07-28).

    Within ONE sample two adjacent windows are the same timepoint, so a genome cannot sit
    at two different frequencies across them. Identical alleles are therefore not enough:
    a haplotype at 95% in one window and 5% in the next is not the same entity.

    Eliminator only - agreement never scores, and the test runs on RAW COUNTS because the
    derived abundance is quantised onto unit fractions by a median denominator of 9.
    """
    from strainphase.core import Haplotype, Window, WindowResult, link_windows

    shared = {12000: "A", 14000: "C", 16000: "G", 18000: "T"}

    # A DECOY second haplotype so the marker set is non-empty. Markers are positions
    # that VARY across a sample's haplotypes; with one identical haplotype per window
    # nothing varies, there is nothing informative to compare, and no verdict is
    # possible - which is the real situation in a clonal locus, not what this test is
    # about. Real multi-strain data always has the variation this supplies.
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
# TWO MEMBERS OF ONE SAMPLE IN ONE GROUP
# --------------------------------------------------------------------------- #
# Every other fixture in this file puts one member per (sample, window), and that is
# the ONE shape in which summing a shared denominator, collapsing a sample's members,
# and handing them to Fisher as separate windows are all indistinguishable from
# correct. The pipeline produces the two-member shape routinely:
# merge_similar_haplotypes deliberately declines some 1-SNP pairs and both halves then
# clear the step-2 gate - which is a real strain split, i.e. exactly the event the
# analysis exists to find. coherence.py:157 flags three per cell, not two.


def _split_grp(gid, wstart, samples, reads_each, total, wsid="T1"):
    """A group where every sample contributes TWO near-identical members.

    ``reads_each`` is a per-member pair; the two halves of a split are rarely equal, and
    equal halves are the one case that hides feeding them to Fisher as separate windows.
    The denominator is one window's non-junk total, carried identically on every member
    of that (sample, window) cell - it is a property of the window, not of the member.
    """
    members = []
    for s in samples:
        for k, reads in enumerate(reads_each):
            members.append(_mem(s, {12000: "A", 15000: "C", 18000: "G"},
                                reads=reads, total=total, wsid=f"{wsid}{k}"))
    return _grp(gid, wstart, members)














def test_load_snvs_is_cached_across_samples():
    """A longitudinal run calls process_contig once PER SAMPLE, and under a cohort union
    VCF every call parses the identical file. On 000066952_0 that was the same 76,988
    records re-read 146 times, once per sample, for a result that cannot differ.

    The cache is keyed on the settings that change what is kept, so a config change still
    re-parses.
    """
    from strainphase import core

    calls = {"n": 0}
    real = core._load_snvs_uncached

    # The real loader returns EIGHT tables; a shorter stub was accepted here for a while
    # and nothing noticed, which is how the aliasing contract below stayed untested.
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
    """REGRESSION (R1-4): each call gets its OWN containers, hit or miss.

    process_contig APPENDS SV pseudo-sites to snv_pos and writes their anchor bases and
    site types; iter_windows_lazy registers split-read breakpoints the same way. Handing
    back the cached objects meant sample 1's SV anchors were already present when sample
    2 ran under the same union VCF, so sample 2 saw them as collisions with a called
    variant and dropped its own SV sites - one timepoint kept the event, every other
    timepoint lost it, and the loss was logged as a legitimate collision.

    Asserting the call count alone cannot see this: the cache was already working.
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
    """END TO END: build_window_tables -> write_window_tables produces the deliverables in
    output_dir and the diagnostics in output_dir/tmp.

    Guards the wiring, not the algorithm: step 3 was previously never called by the
    pipeline at all, so a run produced no lineage output.
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
# Read-overlap threading (the step-3 linker)
#
# Measured on div0050_k4: the reciprocal-best-match linker terminated 152 chains
# and HALF of those had a byte-identical partner it refused as failed_not_mutual,
# 67 of them because that partner preferred an equally identical rival. These
# tests pin the three behaviours that matter: the over-split is rejoined, the
# 2% consensus gate still vetoes first, and no read evidence means no join.
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

    hap_id is withheld below assign_confidence_threshold, which is correct for
    CALLING a read's haplotype. Read-overlap threading only asks whether the same
    molecule sits in both windows, and two strains differing at a couple of markers
    leave most reads at gamma 0.6-0.89 - so gating on hap_id discarded exactly the
    evidence that near-identical strains depend on.
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
# Four lineages (LIN000290/284/285/286) came out of one real run all >=98.8%
# identical to each other, spanning the same ~1.37-1.41 Mb interval, and
# co-occurring in 14-32 of the same samples. They are one strain reported four
# times.
#
# The 2% consensus gate is NOT what let this through - the pieces agree far
# inside it. The cause is structural: build_lineages only ever compares a group
# at window W with a group at W+step. Two groups sitting at the SAME window are
# never candidates for each other, so once a strain is split across parallel
# groups, each half chains forward independently and nothing downstream can
# rejoin them. Step 2 is what should have grouped them in the first place.
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
# REPRODUCTION: how two pure step-1 tracks end up in one over-merged lineage
#
# Observed on div0050_k2 sample T3 (truth 0.663/0.337): step 1 produced two
# PURE tracks - 12,029 reads at purity 1.000 (strain2) and 6,392 at purity
# 1.000 (strain1) - and BOTH landed entirely in one lineage of purity 0.653,
# taking assigned_correct_fraction from 0.973 (single) to 0.000. No track was
# split; two tracks that should be separate lineages were welded together.
#
# These tests pin the two links in that chain. They assert the CURRENT
# behaviour, so they will fail loudly when either is fixed.
# ---------------------------------------------------------------------------

_FUSION_DISCRIM = [5000, 5500]                      # where two strains differ
_FUSION_INVAR = list(range(6000, 20000, 100))       # identical in both


def _fusion_haps():
    a = {**{p: "A" for p in _FUSION_DISCRIM}, **{p: "G" for p in _FUSION_INVAR}}
    b = {**{p: "T" for p in _FUSION_DISCRIM}, **{p: "G" for p in _FUSION_INVAR}}
    return a, b


def test_a_real_difference_is_not_diluted_by_uninformative_positions():
    """ROOT CAUSE, now fixed. Two haplotypes differing at EVERY discriminating marker
    they share must not be declared identical.

    compare_consensus falls back to all co-covered positions when fewer than
    min_shared_markers DISCRIMINATING markers are shared. Those extra positions are
    invariant - they cannot disagree - so they only ever dilute. Here 2 real
    differences over 2 markers (rate 1.000) become 2 over 142 (rate 0.014), which
    clears the 2% gate.
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
# Read-supported marker set (shared by steps 2 and 3)
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
# Step-1 abundance verdicts propagate into step 3 (hybrid)
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
    the crossing alternative so reciprocal best match has something to choose on.
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
    """Reach 1 is the pre-2026-08-31 rule: only OVERLAPPING windows link.

    It is the escape hatch, so it has to keep working even though it is no longer the
    default - a disjoint pair must be refused outright rather than falling through to
    the read path, which an earlier `(k - i) > reach` bound got wrong for the immediate
    neighbour.
    """
    from strainphase.core import link_windows

    a, c = _reach_windows()
    out = link_windows([a, c], cfg(link_window_reach=1))
    ids = {h.track_id for wr_ in out for h in wr_.haplotypes}
    assert len(ids) == 4, "reach 1 must not link a non-overlapping pair"


def test_read_reach_is_on_by_default():
    """The default reaches one window past the overlap.

    Pinned because the default is what every pipeline run actually uses: a silent revert
    to 1 would look like the read path simply finding nothing, which is exactly how it
    fails when read assignments are missing.
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
    """Below `link_min_shared_reads` the pair is not linked.

    Consensus cannot veto a disjoint pair, so this threshold and reciprocal best match
    are the entire evidence bar - it is the one knob standing between read chaining and
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
    """7 shared positions, 1 difference -> rate is 14%, far above identity_distance.

    The validator used to sit INSIDE the rate gate, so this pair was split without the
    count-based test that exists to judge it. This is the LS_11_5_16 shape: 3 reads of
    130 splitting off a dominant haplotype.
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
    """A credible minor strain survives: >=3 reads, seen across timepoints.

    30% with 45 reads over many timepoints is the case the tool exists to find - it
    must NOT be merged away. Persistence short-circuits before the binomial, and the
    frequency floor (now last) does not fire because 0.30 > marker_min_frac.
    """
    hp, maj, minor, w, g, ntp = _snv_pair(minor_weight=0.30, minor_reads=45,
                                          n_shared=52, n_timepoints=3)
    assert hp.should_merge_1snp_pair(maj, minor, 0, 1, w, g, ntp) is False


def test_1snv_floor_is_a_backstop_not_a_pre_emption():
    """A sub-10% haplotype with real read support reaches the later tests.

    It used to be dismissed on the frequency floor before read support, persistence or
    the binomial were consulted - 622 of 1,174 pairs on 000089747_1 contig_2 were
    decided that way. The floor still applies, but only after the evidence has spoken,
    so a persistent sub-10% allele is now judged rather than assumed to be noise.
    """
    hp, maj, minor, w, g, ntp = _snv_pair(minor_weight=0.048, minor_reads=8,
                                          n_shared=52, n_timepoints=3)
    # persistence is satisfied, so the run reaches the floor at the END and merges
    assert hp.should_merge_1snp_pair(maj, minor, 0, 1, w, g, ntp) is True
    # ...whereas the same allele above the floor is kept apart
    hp2, maj2, minor2, w2, g2, ntp2 = _snv_pair(minor_weight=0.12, minor_reads=8,
                                                n_shared=52, n_timepoints=3)
    assert hp2.should_merge_1snp_pair(maj2, minor2, 0, 1, w2, g2, ntp2) is False
