"""Merging step-1 tracks across samples, in place of cross-sample window grouping."""

from __future__ import annotations

from dataclasses import replace

import pytest

from strainphase.core import DEFAULT_CONFIG
from strainphase.track_merge import build_lineages_from_tracks, track_consensus, tracks_of
from strainphase.window_groups import WindowHaplotype

MARKERS = frozenset({100, 200, 300, 400})


def _hap(sample, track, window, alleles, reads=20, hid=None):
    return WindowHaplotype(
        sample=sample, contig="c1", window_start=window, window_end=window + 10000,
        haplotype_id=hid or f"{sample}_{track}_{window}", consensus=dict(alleles),
        reads=reads, total_reads=reads * 2, junk_reads=0, abundance=0.5,
        within_sample_id=track,
    )


def _cfg(**kw):
    return replace(DEFAULT_CONFIG, **kw)


def test_tracks_group_by_sample_and_chain():
    haps = [_hap("S1", "T1", 0, {100: "A"}), _hap("S1", "T1", 5000, {200: "A"}),
            _hap("S2", "T1", 0, {100: "A"})]
    tracks = tracks_of(haps)
    assert len(tracks) == 2, "a track is per (sample, chain), not per chain id"
    assert len(tracks[("S1", "T1")]) == 2


def test_unlinked_haplotype_becomes_its_own_track():
    """A haplotype step 1 never placed still has to be mergeable, not dropped."""
    haps = [_hap("S1", "", 0, {100: "A"}), _hap("S1", "", 5000, {200: "A"})]
    assert len(tracks_of(haps)) == 2


def test_track_consensus_weights_by_reads_and_drops_ties():
    a = _hap("S1", "T1", 0, {100: "A"}, reads=30)
    b = _hap("S1", "T1", 5000, {100: "C"}, reads=10)
    assert track_consensus([a, b], MARKERS) == {100: "A"}      # 30 reads beat 10
    b.reads = 30
    assert track_consensus([a, b], MARKERS) == {}, "a tie emits no call"


def test_track_consensus_keeps_only_markers():
    h = _hap("S1", "T1", 0, {100: "A", 999: "T"})
    assert track_consensus([h], MARKERS) == {100: "A"}


def test_identical_tracks_from_different_samples_merge():
    haps = [_hap("S1", "T1", 0, {100: "A", 200: "C"}),
            _hap("S2", "T1", 0, {100: "A", 200: "C"})]
    lineages = build_lineages_from_tracks(haps, _cfg(), markers=MARKERS)
    assert len(lineages) == 1
    assert {m.sample for g in lineages[0].groups for m in g.members} == {"S1", "S2"}


def test_a_single_disagreement_prevents_the_merge():
    """Byte-for-byte: no rate can dilute one real difference into agreement."""
    haps = [_hap("S1", "T1", 0, {100: "A", 200: "C"}),
            _hap("S2", "T1", 0, {100: "A", 200: "T"})]
    assert len(build_lineages_from_tracks(haps, _cfg(), markers=MARKERS)) == 2



def test_step1_veto_refuses_a_merge_identity_would_allow():
    haps = [_hap("S1", "T1", 0, {100: "A", 200: "C"}),
            _hap("S2", "T1", 0, {100: "A", 200: "C"})]
    forbidden = {frozenset((("S1", "T1"), ("S2", "T1")))}
    assert len(build_lineages_from_tracks(
        haps, _cfg(), markers=MARKERS, cannot_link=forbidden)) == 2


def test_entity_is_split_back_into_one_group_per_window():
    """Downstream reads WindowGroups, so a merged entity has to present as them."""
    haps = [_hap("S1", "T1", 0, {100: "A"}), _hap("S1", "T1", 5000, {200: "C"}),
            _hap("S2", "T1", 0, {100: "A"}), _hap("S2", "T1", 5000, {200: "C"})]
    lineages = build_lineages_from_tracks(haps, _cfg(), markers=MARKERS)
    assert len(lineages) == 1
    assert [g.window_start for g in lineages[0].groups] == [0, 5000]
    assert all(len(g.members) == 2 for g in lineages[0].groups)


def test_markers_must_be_supplied():
    with pytest.raises(ValueError):
        build_lineages_from_tracks([_hap("S1", "T1", 0, {100: "A"})], _cfg())


def test_markerless_tracks_are_reported_not_dropped():
    """No evidence to merge is not evidence of absence.

    A track carrying no identity marker cannot be merged with anything, but on a real
    MAG that is 47% of tracks - dropping them would silently discard nearly half the
    phased sequence. They come back as singleton lineages.
    """
    haps = [_hap("S1", "T1", 0, {999: "A"}),      # 999 is not in MARKERS
            _hap("S2", "T1", 0, {999: "A"})]
    lineages = build_lineages_from_tracks(haps, _cfg(), markers=MARKERS)
    assert len(lineages) == 2, "unmergeable is not the same as absent"
    assert {m.sample for lin in lineages for g in lin.groups for m in g.members} == {"S1", "S2"}


def test_markerless_and_mergeable_tracks_coexist():
    haps = [_hap("S1", "T1", 0, {100: "A", 200: "C"}),
            _hap("S2", "T1", 0, {100: "A", 200: "C"}),   # merges with S1
            _hap("S3", "T9", 0, {999: "G"})]             # no marker -> singleton
    lineages = build_lineages_from_tracks(haps, _cfg(), markers=MARKERS)
    assert len(lineages) == 2
    sizes = sorted(len({m.sample for g in lin.groups for m in g.members}) for lin in lineages)
    assert sizes == [1, 2]


def test_a_conflict_blocks_a_transitive_merge():
    """Byte-identity is NOT transitive across different marker subsets.

    A and B agree on {100}, B and C agree on {200}, A and C disagree at {300}. Merging
    on agreement alone fuses A and C through B without ever comparing them - which is
    what union-find did. A conflict is recorded, not merely skipped, and is checked
    component-to-component.
    """
    haps = [_hap("S1", "T1", 0, {100: "A", 300: "G"}),
            _hap("S2", "T1", 0, {100: "A", 200: "C"}),
            _hap("S3", "T1", 0, {200: "C", 300: "T"})]      # 300 disagrees with S1
    lineages = build_lineages_from_tracks(haps, _cfg(), markers=MARKERS)
    got = [{m.sample for g in lin.groups for m in g.members} for lin in lineages]
    assert {"S1", "S3"} not in got, "a recorded conflict must survive transitivity"
    assert sum(len(x) for x in got) == 3


def test_strongest_evidence_merges_first():
    """Pairs are taken in descending agreeing-marker count, so a one-marker
    coincidence can only join what the strong evidence already built."""
    haps = [_hap("S1", "T1", 0, {100: "A", 200: "C", 300: "G"}),
            _hap("S2", "T1", 0, {100: "A", 200: "C", 300: "G"}),   # 3 markers with S1
            _hap("S3", "T1", 0, {100: "A", 400: "T"})]             # 1 marker with S1
    lineages = build_lineages_from_tracks(haps, _cfg(), markers=MARKERS)
    assert len(lineages) == 1
    assert len(lineages[0].groups[0].members) == 3


def test_same_sample_tracks_more_than_one_window_apart_get_an_abundance_check():
    """link_windows never puts distant windows side by side, so its abundance verdict
    does not exist for them and is taken at merge time instead."""
    far = _hap("S1", "T2", 60000, {100: "A"}, reads=1)
    far.total_reads = 500                       # 1/500 vs 20/40 - incoherent shares
    haps = [_hap("S1", "T1", 0, {100: "A"}, reads=20), far,
            _hap("S2", "T1", 0, {100: "A"}, reads=20)]
    lineages = build_lineages_from_tracks(haps, _cfg(), markers=MARKERS)
    joined = [{(m.sample, m.within_sample_id) for g in lin.groups for m in g.members}
              for lin in lineages]
    assert not any({("S1", "T1"), ("S1", "T2")} <= j for j in joined)
