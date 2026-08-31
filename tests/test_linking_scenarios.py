#!/usr/bin/env python3
"""Linking scenarios drawn from observed runs, scored on FRAGMENTATION vs ERROR.

Every scenario here is a reduction of something measured on real data, with the
CORRECT answer written down. They are not all expected to pass: the point is to
score a candidate rule on both axes at once, because every change we have tried so
far moved one at the expense of the other.

    fragmentation   the answer is split into more pieces than the truth has
    error           things the truth keeps apart are merged

Run the benchmark table with:

    pytest tests/test_linking_scenarios.py -s -k report

The individual tests assert the CORRECT behaviour. Ones the current code fails are
marked ``xfail(strict=False)`` so the suite stays green while still recording the
gap - remove the marker when a change fixes one, and it becomes a regression test.
"""
from __future__ import annotations

import numpy as np
import pytest

from strainphase.core import HaplotyperConfig
from strainphase.lineages import build_lineages
from strainphase.window_groups import (
    WindowGroup,
    WindowHaplotype,
    group_window_across_samples,
)

W, STEP = 20000, 10000


def _cfg(**kw):
    base = dict(window_size=W, min_shared_reads_for_link=1)
    base.update(kw)
    return HaplotyperConfig(**base)


def _hap(sample, consensus, wsid="", reads=100, total=200, read_key=None, hid=""):
    """One window-haplotype. ``read_key`` names the physical reads it holds, so two
    haplotypes of the same strain in adjacent windows can share reads."""
    return WindowHaplotype(
        sample=sample, contig="c1", window_start=0, window_end=W, haplotype_id=hid,
        consensus=dict(consensus), reads=reads, total_reads=total, junk_reads=0,
        abundance=reads / total, within_sample_id=wsid,
        read_ids=frozenset(f"{sample}:{read_key or wsid}:r{i}" for i in range(8)))


def _grp(gid, wstart, members):
    for i, m in enumerate(members):
        m.window_start, m.window_end = wstart, wstart + W
        m.haplotype_id = f"{m.sample}_c1_{wstart}_H{i}"
    return WindowGroup(group_id=gid, contig="c1", window_start=wstart,
                       window_end=wstart + W, members=members)


# --------------------------------------------------------------------------- #
# Consensus fixtures. Markers are positions that VARY; invariant positions carry
# no information and are the ones the removed clonal fallback used to pad with.
# --------------------------------------------------------------------------- #
MARKERS = list(range(1000, 70000, 500))
STRAIN_A = {p: "A" for p in MARKERS}
STRAIN_B = {p: "T" for p in MARKERS}

# a window where the two strains happen to share only TWO discriminating markers
THIN_DISCRIM = [5000, 5500]
THIN_INVAR = list(range(6000, 20000, 100))
THIN_A = {**{p: "A" for p in THIN_DISCRIM}, **{p: "G" for p in THIN_INVAR}}
THIN_B = {**{p: "T" for p in THIN_DISCRIM}, **{p: "G" for p in THIN_INVAR}}
# ...and the same window for ONE strain seen in two samples: nothing varies at all
CLONAL = {p: "G" for p in THIN_INVAR}


def _lineage_sets(lineages):
    return sorted((frozenset(g.group_id for g in lin.groups) for lin in lineages),
                  key=lambda s: sorted(s))


def score(groups, expected, **cfg_kw):
    """Return (n_lineages, n_expected, fragmentation, error) for one scenario.

    fragmentation: emitted pieces per expected lineage, 1.0 is perfect.
    error:         fraction of expected-DIFFERENT group pairs placed together.
    """
    lineages, _ = build_lineages(groups, _cfg(**cfg_kw), step=STEP,
                                 transitive_abundance_check=False)
    got = _lineage_sets(lineages)
    place = {}
    for i, s in enumerate(got):
        for gid in s:
            place[gid] = i
    wrong = total = 0
    exp_of = {}
    for i, s in enumerate(expected):
        for gid in s:
            exp_of[gid] = i
    ids = sorted(exp_of)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if exp_of[a] != exp_of[b]:
                total += 1
                if place.get(a) == place.get(b):
                    wrong += 1
    return len(got), len(expected), len(got) / len(expected), (wrong / total if total else 0.0)


# --------------------------------------------------------------------------- #
# SCENARIOS
# --------------------------------------------------------------------------- #
def s_two_pure_tracks_of_different_strains():
    """div0050_k2 T3. Step 1 produced two PURE tracks - 12,029 reads at purity 1.000
    (strain2) and 6,392 at purity 1.000 (strain1). Both landed in ONE lineage of
    purity 0.653; assigned_correct_fraction went 0.973 (single) -> 0.000. They must
    stay two lineages."""
    groups = [
        _grp("A0", 1, [_hap("S1", STRAIN_A, wsid="TA", reads=130)]),
        _grp("A1", 1 + STEP, [_hap("S1", STRAIN_A, wsid="TA", reads=130)]),
        _grp("B0", 1, [_hap("S1", STRAIN_B, wsid="TB", reads=70)]),
        _grp("B1", 1 + STEP, [_hap("S1", STRAIN_B, wsid="TB", reads=70)]),
    ]
    return groups, [{"A0", "A1"}, {"B0", "B1"}]


def s_one_track_spanning_many_windows():
    """div0025_k2 T6. Step 1 produced ONE track spanning the whole 4.87 Mb contig;
    the lineages built over it came out in 12 pieces capped at 1.85 Mb. A read is
    labelled by its lineage, so the intact chain was thrown away. One track must give
    one lineage."""
    groups = [
        _grp(f"W{w}", 1 + w * STEP, [_hap("S1", STRAIN_A, wsid="ONE", reads=130)])
        for w in range(6)
    ]
    return groups, [{f"W{w}" for w in range(6)}]


def s_oversplit_strain_rejoined():
    """The contested over-split: one strain appears as TWO groups at a window, both
    continuing into one group. Reciprocity sees a tie and links neither. They are one
    strain and belong in one lineage."""
    groups = [
        _grp("A1", 1, [_hap("S1", STRAIN_A, wsid="TA", reads=65, read_key="X")]),
        _grp("A2", 1, [_hap("S2", STRAIN_A, wsid="TB", reads=65, read_key="Y")]),
        _grp("B", 1 + STEP, [_hap("S1", STRAIN_A, wsid="TA", reads=130, read_key="X"),
                             _hap("S2", STRAIN_A, wsid="TB", reads=130, read_key="Y")]),
    ]
    return groups, [{"A1", "A2", "B"}]


def s_genuine_strain_boundary():
    """Two different strains in adjacent windows must never join - the control that
    keeps every other scenario honest."""
    groups = [
        _grp("A", 1, [_hap("S1", STRAIN_A, wsid="TA", reads=130)]),
        _grp("B", 1 + STEP, [_hap("S1", STRAIN_B, wsid="TB", reads=70)]),
    ]
    return groups, [{"A"}, {"B"}]


SCENARIOS = {
    "two_pure_tracks_different_strains": s_two_pure_tracks_of_different_strains,
    "one_track_spanning_many_windows": s_one_track_spanning_many_windows,
    "oversplit_strain_rejoined": s_oversplit_strain_rejoined,
    "genuine_strain_boundary": s_genuine_strain_boundary,
}


def test_report_fragmentation_vs_error():
    """Benchmark table. Not a gate - it prints and always passes."""
    print(f"\n  {'scenario':36s}{'lineages':>10}{'ideal':>7}{'fragmentation':>15}{'error':>8}")
    for name, build in SCENARIOS.items():
        groups, expected = build()
        n, e, frag, err = score(groups, expected)
        flag = "" if (abs(frag - 1) < 1e-9 and err == 0) else "   <-- WRONG"
        print(f"  {name:36s}{n:>10}{e:>7}{frag:>15.2f}{err:>8.2f}{flag}")


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_is_reconstructed_exactly(name):
    """Each scenario's CORRECT answer. xfail marks the ones the current rules miss."""
    known_bad = {
        "oversplit_strain_rejoined": (
            "reciprocity sees a tie on the target's side and links neither half. The "
            "track-preserving union only rescues this when a step-1 chain crossed the "
            "boundary in some sample; here the two halves are in different tracks, "
            "which is exactly the case forward-only best-match used to handle."
        ),
    }
    groups, expected = SCENARIOS[name]()
    n, e, frag, err = score(groups, expected)
    if name in known_bad:
        pytest.xfail(known_bad[name])
    assert err == 0.0, f"{name}: merged things the truth keeps apart"
    assert n == e, f"{name}: emitted {n} lineages, truth has {e}"


# --------------------------------------------------------------------------- #
# STEP 2 scenarios - where the fusion actually originates
# --------------------------------------------------------------------------- #
def _w2(sample, cons):
    return WindowHaplotype(
        sample=sample, contig="c1", window_start=0, window_end=W,
        haplotype_id=f"{sample}_h", consensus=dict(cons), reads=100, total_reads=200,
        junk_reads=0, abundance=0.5, within_sample_id=f"T_{sample}")


def test_step2_must_not_group_two_strains_that_differ_at_every_shared_marker():
    """THE FUSION'S ORIGIN. Two strains sharing only 2 discriminating markers, and
    differing at BOTH. One group here is a single node that both strains' chains run
    through, so every lineage built on it inherits the fusion - for every sample,
    including ones the phasing had separated perfectly."""
    groups, _, _ = group_window_across_samples(
        [_w2("S1", THIN_A), _w2("S2", THIN_B)], set(THIN_DISCRIM), _cfg(),
        group_prefix="c1_")
    assert len(groups) == 2, "two strains must not share a group"


@pytest.mark.xfail(strict=False, reason=(
    "the clonal fallback that used to link this was removed because it also merged "
    "DIFFERENT strains by dilution; a rule that links here without linking the "
    "scenario above is what we are looking for"))
def test_step2_should_group_one_strain_seen_in_two_samples_at_a_clonal_locus():
    """THE COST OF THE FIX. One strain, two samples, a locus with NO variable
    position - so no marker, no verdict, no group. Measured: 25% of candidate pairs
    on the divergent panel and 99% on the near-clonal isolates share fewer than 3
    markers, so this is the common case on clonal genomes, not a corner."""
    groups, _, _ = group_window_across_samples(
        [_w2("S1", CLONAL), _w2("S2", CLONAL)], set(), _cfg(), group_prefix="c1_")
    assert len(groups) == 1, "one strain in two samples is one group"
