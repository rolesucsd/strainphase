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


def test_min_shared_markers_governs_how_much_evidence_a_merge_needs():
    """One agreeing marker is enough at the default and not at 2 - the single
    threshold the merge has."""
    haps = [_hap("S1", "T1", 0, {100: "A"}), _hap("S2", "T1", 0, {100: "A", 200: "C"})]
    assert len(build_lineages_from_tracks(
        haps, _cfg(track_merge_min_shared_markers=1), markers=MARKERS)) == 1
    assert len(build_lineages_from_tracks(
        haps, _cfg(track_merge_min_shared_markers=2), markers=MARKERS)) == 2


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
