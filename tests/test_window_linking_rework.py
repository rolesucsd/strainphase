#!/usr/bin/env python3
"""Tests for the window-linking rework.

Covers the identity gate stack, the marker set (including the clonal-fallback that a
naive implementation gets wrong), both cross-sample grouping shapes, the abundance
coherence test, the QC gate, and the zero-leak fix.
"""

from __future__ import annotations

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
from strainphase.longitudinal import _window_conditional_abundance, _weighted_median
from strainphase.window_groups import WindowHaplotype, group_window_across_samples


def cfg(**kw) -> HaplotyperConfig:
    """Config sized for small synthetic markers rather than 20 kb windows."""
    base = {
        "min_entity_overlap_bp": 0,
        "min_cosupported_span_frac": 0.0,
        "min_shared_for_lineage": 3,
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


def test_clonal_fallback_still_links():
    """REGRESSION: with no discriminating markers, absence of evidence of difference is
    not evidence of difference.

    85% of windows hold a single haplotype, so a clonal sample can have almost no
    variable positions at all. Restricting the comparison to markers would then make
    every comparison impossible and shatter a real lineage into singletons.
    """
    a = {10: "A", 20: "C", 30: "G"}
    b = {10: "A", 20: "C", 30: "G"}
    gate = compare_consensus(a, b, markers=set(), config=cfg())
    assert gate.passed
    assert gate.used_fallback
    assert gate.n_shared == 3


def test_absolute_cap_binds_where_the_rate_does_not():
    """The rate is a floor, so at large n_shared it tolerates many mismatches; the
    absolute cap is what actually binds there."""
    a = {i: "A" for i in range(1000)}
    b = dict(a)
    for i in (5, 15, 25):  # 3 mismatches out of 1000 -> rate 0.003, under 0.01
        b[i] = "T"
    # All 1000 positions are markers: in a real run the marker set is computed across
    # every sample on the contig, so a position can vary somewhere in the cohort while
    # these two particular haplotypes happen to agree on it.
    markers = set(range(1000))
    permissive = compare_consensus(a, b, markers, cfg(max_num_diff=10))
    assert permissive.passed, "rate alone admits 3 mismatches at n_shared=1000"
    strict = compare_consensus(a, b, markers, cfg(max_num_diff=1))
    assert not strict.passed
    assert strict.reason == "failed_mismatch"


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
    config = cfg(min_cosupported_span_frac=0.25, min_entity_overlap_bp=0)
    # co-supported span is 2 bp inside a 10 kb region -> far below 25%
    assert not compare_consensus(a, b, set(), config, region=(1, 10001)).passed
    # same pair with no region constraint passes
    assert compare_consensus(a, b, set(), config).passed


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


@pytest.mark.parametrize("method", ["clique", "reciprocal"])
def test_identical_haplotypes_group_together(method):
    cons = {100: "A", 2000: "C", 5000: "G"}
    haps = [_hap(f"t{i}", f"h{i}", dict(cons)) for i in range(4)]
    groups, edges = group_window_across_samples(
        haps, markers=set(), config=cfg(cross_sample_method=method)
    )
    assert len(groups) == 1
    assert groups[0].n_samples == 4
    assert all(e.reason == "linked" for e in edges)


@pytest.mark.parametrize("method", ["clique", "reciprocal"])
def test_divergent_haplotypes_stay_separate(method):
    a = {100: "A", 2000: "C", 5000: "G"}
    b = {100: "T", 2000: "A", 5000: "C"}
    haps = [_hap("t0", "h0", a), _hap("t1", "h1", b)]
    groups, edges = group_window_across_samples(
        haps, markers={100, 2000, 5000}, config=cfg(cross_sample_method=method)
    )
    assert len(groups) == 2
    assert {e.reason for e in edges} == {"failed_mismatch"}


def test_clique_refuses_to_chain():
    """A and C differ, but both match B. Single linkage would chain them into one group;
    complete linkage must not -- this is the accretion being removed."""
    a = {100: "A", 2000: "A", 5000: "A", 8000: "A"}
    b = {100: "A", 2000: "A", 5000: "A", 8000: "T"}  # 1 diff from a
    c = {100: "A", 2000: "A", 5000: "T", 8000: "T"}  # 1 diff from b, 2 from a
    haps = [_hap("t0", "ha", a), _hap("t1", "hb", b), _hap("t2", "hc", c)]
    markers = variable_marker_positions([a, b, c])
    groups, _ = group_window_across_samples(
        haps, markers, cfg(cross_sample_method="clique", max_num_diff=1)
    )
    labels = {m.haplotype_id: g.group_id for g in groups for m in g.members}
    assert labels["ha"] != labels["hc"], "a and c differ by 2 and must not share a group"


def test_edges_record_every_comparison_including_failures():
    """A discarded comparison is indistinguishable from one never made."""
    a = {100: "A", 2000: "C", 5000: "G"}
    b = {100: "T", 2000: "A", 5000: "C"}
    haps = [_hap("t0", "h0", a), _hap("t1", "h1", b), _hap("t2", "h2", dict(a))]
    _, edges = group_window_across_samples(haps, {100, 2000, 5000}, cfg())
    assert len(edges) == 3  # all pairs, not just successes
    assert {e.reason for e in edges} == {"linked", "failed_mismatch"}


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


def test_zero_leak_would_have_dragged_the_median_down():
    """Why None matters: the aggregate is a weighted MEDIAN, a selection operator that
    returns one input verbatim, so an injected 0.0 can become the reported value."""
    real = [0.8, 0.82, 0.79]
    weights = [30.0, 30.0, 30.0]
    assert _weighted_median(real, weights) == pytest.approx(0.8, abs=0.03)
    leaked = _weighted_median([0.0] + real, [90.0] + weights)
    assert leaked == 0.0


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
    from strainphase.window_groups import WindowGroup
    return WindowGroup(group_id=gid, contig="c1", window_start=wstart,
                       window_end=wstart + 20000, members=members)


def _mem(sample, consensus, reads=30, total=60):
    return WindowHaplotype(sample=sample, contig="c1", window_start=0, window_end=20000,
                           haplotype_id=f"{sample}|h", consensus=consensus,
                           reads=reads, total_reads=total, abundance=reads / total)


def _lcfg(**kw):
    base = {"window_size": 20000, "min_shared_for_lineage": 3,
            "min_entity_overlap_bp": 0, "min_cosupported_span_frac": 0.0}
    base.update(kw)
    return HaplotyperConfig(**base)


def test_lineage_chains_matching_groups_across_windows():
    """Two groups agreeing in the 50% overlap interval become one lineage."""
    from strainphase.lineages import build_lineages
    shared = {12000: "A", 15000: "C", 18000: "G"}
    a = _grp("A", 1, [_mem(f"t{i}", {**shared, 3000: "T"}) for i in range(4)])
    b = _grp("B", 10001, [_mem(f"t{i}", {**shared, 25000: "T"}) for i in range(4)])
    lins, edges = build_lineages([a, b], _lcfg())
    assert len(lins) == 1
    assert lins[0].n_windows == 2
    assert [e.reason for e in edges if e.reason == "linked"]


def test_lineage_does_not_chain_divergent_groups():
    from strainphase.lineages import build_lineages
    a = _grp("A", 1, [_mem(f"t{i}", {12000: "A", 15000: "C", 18000: "G"}) for i in range(4)])
    b = _grp("B", 10001, [_mem(f"t{i}", {12000: "T", 15000: "A", 18000: "C"}) for i in range(4)])
    lins, edges = build_lineages([a, b], _lcfg())
    assert len(lins) == 2
    assert any(e.reason == "failed_mismatch" for e in edges)


def test_abundance_eliminates_a_join_that_identity_would_accept():
    """The ELIMINATOR: identical markers, but the shares genuinely disagree."""
    from strainphase.lineages import build_lineages
    shared = {12000: "A", 15000: "C", 18000: "G"}
    a = _grp("A", 1, [_mem(f"t{i}", shared, reads=95, total=100) for i in range(5)])
    b = _grp("B", 10001, [_mem(f"t{i}", shared, reads=5, total=100) for i in range(5)])
    lins, edges = build_lineages([a, b], _lcfg())
    assert any(e.reason == "failed_abundance" for e in edges)
    assert len(lins) == 2, "incompatible shares must not merge"


def test_abundance_agreement_does_not_rescue_a_failed_identity():
    """Abundance is an eliminator, NOT an indicator: matching shares earn no credit."""
    from strainphase.lineages import build_lineages
    a = _grp("A", 1, [_mem(f"t{i}", {12000: "A", 15000: "C", 18000: "G"}) for i in range(5)])
    b = _grp("B", 10001, [_mem(f"t{i}", {12000: "T", 15000: "A", 18000: "C"}) for i in range(5)])
    lins, _ = build_lineages([a, b], _lcfg())          # identical abundances throughout
    assert len(lins) == 2


def test_ambiguous_continuation_contributes_no_edge():
    """RECIPROCAL BEST, not greedy: two equally good successors -> neither is chosen."""
    from strainphase.lineages import build_lineages
    shared = {12000: "A", 15000: "C", 18000: "G"}
    a = _grp("A", 1, [_mem(f"t{i}", shared) for i in range(4)])
    b1 = _grp("B1", 10001, [_mem(f"t{i}", dict(shared)) for i in range(4)])
    b2 = _grp("B2", 10001, [_mem(f"t{i}", dict(shared)) for i in range(4)])
    lins, edges = build_lineages([a, b1, b2], _lcfg())
    assert len(lins) == 3, "a tie must stop the chain, not pick a winner"
    assert any(e.reason == "failed_not_mutual" for e in edges)


def test_lineage_length_is_bounded_by_the_window_count():
    """Reciprocity makes each component a PATH, so a lineage cannot accrete."""
    from strainphase.lineages import build_lineages
    shared_pat = {"a": "A", "b": "C", "c": "G"}
    gs = []
    for i in range(5):
        w = 1 + i * 10000
        cons = {w + 12000: shared_pat["a"], w + 15000: shared_pat["b"], w + 18000: shared_pat["c"],
                w + 2000: shared_pat["a"], w + 5000: shared_pat["b"], w + 8000: shared_pat["c"]}
        gs.append(_grp(f"G{i}", w, [_mem(f"t{j}", cons) for j in range(4)]))
    lins, _ = build_lineages(gs, _lcfg())
    assert max(x.n_windows for x in lins) <= 5


def test_marker_span_reports_what_was_actually_resolved():
    from strainphase.lineages import build_lineages
    shared = {12000: "A", 15000: "C", 18000: "G"}
    a = _grp("A", 1, [_mem(f"t{i}", shared) for i in range(4)])
    lins, _ = build_lineages([a], _lcfg())
    assert lins[0].marker_span == (12000, 18000)


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
