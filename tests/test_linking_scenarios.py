#!/usr/bin/env python3
"""Linking scenarios reduced from observed runs, each paired with the correct answer.

Every scenario is a reduction of something measured on real data. Two axes matter,
and a change that fixes one often worsens the other:

    fragmentation   the answer is split into more pieces than the truth has
    error           things the truth keeps apart are merged

These are the scenario definitions used to build and check candidate linking rules.
"""
from __future__ import annotations


from strainphase.window_groups import WindowGroup, WindowHaplotype

W, STEP = 20000, 10000




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
    """The contested over-split: one strain appears as two groups at a window, both
    continuing into one group. The linker sees a tie and links neither. They are one
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






# --------------------------------------------------------------------------- #
# Cross-sample scenarios - where the fusion actually originates
# --------------------------------------------------------------------------- #
def _w2(sample, cons):
    return WindowHaplotype(
        sample=sample, contig="c1", window_start=0, window_end=W,
        haplotype_id=f"{sample}_h", consensus=dict(cons), reads=100, total_reads=200,
        junk_reads=0, abundance=0.5, within_sample_id=f"T_{sample}")




