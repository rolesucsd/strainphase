"""Merge step-1 tracks across samples into lineages, replacing step 2.

WHY THIS REPLACES CROSS-SAMPLE WINDOW GROUPING
    Step 2 grouped haplotypes at ONE fixed window across samples, and step 3 then
    chained those groups along the genome. That split one question - "are these the
    same strain?" - across two passes with different units, and produced three
    structural problems measured on B. fragilis 000089747_1:

      * Two groups at the SAME window are never a candidate pair in step 3, so a
        recent mutation that step 2 correctly separated could be re-fused
        transitively and no pairwise veto could reach it.
      * 78% of step-3 comparisons returned `failed_no_evidence`, and complete
        linkage turns each one into an explicit non-merge. 51 haplotypes that were
        >=98% identical to one strain shattered into 11 groups on 82 uncomparable
        pairs against 2 that genuinely disagreed.
      * The machinery that repairs that shattering - the track-preserving union -
        then had to run last, after the coherence check, or be undone by it.

    A step-1 track is already a within-sample chain built from that sample's own
    reads, and it is never chimeric in practice: of 31 tracks spanning both
    positions of a real sweep, zero carried both genotypes. So the remaining work
    is only to merge tracks ACROSS samples, which is one clustering rather than
    1,884 windows of grouping plus a chaining pass.

THE MERGE RULE
    Pass 1 is BYTE-FOR-BYTE IDENTITY over the shared identity markers, with no rate
    and no threshold to tune except how much evidence a merge needs
    (``track_merge_min_shared_markers``). Exact agreement cannot fuse two genotypes
    that disagree anywhere both called, which is what a rate gate does when a
    handful of real differences are diluted by thousands of identical sites.

    Measured sweep of that one parameter on contig_2 (7,858 tracks carrying at
    least one marker):

        min_shared   entities   mean size   sweep genotypes fused?   upeY duplicates
                 1        118       66.6     YES                     0
                 3      4,474       1.76     YES                     0
                10      6,009       1.31     no                      0
                50      6,819       1.15     no                      0

    At 1 a single agreeing marker merges, so identity stops discriminating. The
    default is 1 because a permissive merge is the intended FIRST pass - splitting
    passes come after it - but it is exposed so the trade-off can be measured.

CANNOT-LINK
    Step 1's own refusals carry over: two tracks holding haplotypes step 1 refused
    on a genuine allele disagreement are never merged. Note that byte-identity
    already rejects any pair that disagrees where both called, so these constrain
    only a looser pass.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from strainphase.core import DEFAULT_CONFIG, HaplotyperConfig
from strainphase.lineages import Lineage
from strainphase.window_groups import WindowGroup, WindowHaplotype

__all__ = ["build_lineages_from_tracks", "track_consensus", "tracks_of"]


def tracks_of(haps: list[WindowHaplotype]) -> dict[tuple[str, str], list[WindowHaplotype]]:
    """``(sample, within_sample_id) -> its window-haplotypes``, genomic order.

    A haplotype ``link_windows`` never placed in a track has no identity beyond its
    one window; it becomes a track of its own so it can still be merged, rather than
    being dropped.
    """
    out: dict[tuple[str, str], list[WindowHaplotype]] = defaultdict(list)
    for h in haps:
        key = (h.sample, h.within_sample_id or f"_unlinked_{h.haplotype_id}")
        out[key].append(h)
    for members in out.values():
        members.sort(key=lambda m: m.window_start)
    return dict(out)


def track_consensus(
    members: list[WindowHaplotype], markers: frozenset[int] | set[int]
) -> dict[int, str]:
    """Majority allele per marker across a track's windows, weighted by reads.

    Weighted by reads rather than counted per window because adjacent windows
    overlap by 50%: a position covered by two windows would otherwise vote twice on
    the strength of the same molecules. A TIE emits no call - the position simply
    stops being shared, which costs at worst a comparison, where resolving it by
    member order would make the result depend on input ordering.
    """
    votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for m in members:
        for pos, base in m.consensus.items():
            if pos in markers:
                votes[pos][base] += m.reads
    consensus: dict[int, str] = {}
    for pos, counts in votes.items():
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue
        consensus[pos] = ranked[0][0]
    return consensus


def build_lineages_from_tracks(
    haps: list[WindowHaplotype],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    markers: frozenset[int] | set[int] | None = None,
    cannot_link: set[frozenset[str]] | None = None,
    lineage_prefix: str = "LIN",
) -> list[Lineage]:
    """Merge tracks across samples, then emit one Lineage per merged entity.

    ``cannot_link`` holds ``frozenset((track_key_a, track_key_b))`` pairs step 1
    refused. Each entity is split back into one ``WindowGroup`` per window it
    occupies, so everything downstream - lineage rows, pooled abundance - reads the
    same structures it always has.
    """
    if markers is None:
        raise ValueError("markers must be supplied; steps must judge identity on one set")
    tracks = tracks_of(haps)
    cons = {k: track_consensus(v, markers) for k, v in tracks.items()}
    # A track carrying no marker cannot be MERGED - there is nothing to agree on - but it
    # must still be REPORTED. On B. fragilis 000089747_1, 7,120 of 14,978 tracks (47%)
    # carry no marker under the read-supported set, and dropping them would silently
    # discard nearly half the phased sequence rather than emitting it as unmergeable.
    # They become singleton lineages: no evidence to merge is not evidence of absence.
    keys = [k for k, c in cons.items() if c]
    unmergeable = [k for k, c in cons.items() if not c]

    parent: dict[tuple[str, str], tuple[str, str]] = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Candidates are tracks sharing at least one marker, so tracks that cannot
    # possibly agree are never compared. Pairs already in one component are skipped,
    # which is what keeps this near-linear on real input despite being pairwise.
    by_marker: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for k in keys:
        for pos in cons[k]:
            by_marker[pos].append(k)

    forbidden = cannot_link or set()
    min_shared = max(1, config.track_merge_min_shared_markers)
    n_merged = n_refused = 0
    tested: set[frozenset] = set()
    for k in keys:
        candidates: set[tuple[str, str]] = set()
        for pos in cons[k]:
            candidates.update(by_marker[pos])
        ck = cons[k]
        for j in candidates:
            if j == k:
                continue
            pair = frozenset((k, j))
            if pair in tested:
                continue
            tested.add(pair)
            if find(k) == find(j):
                continue
            if pair in forbidden:
                n_refused += 1
                continue
            cj = cons[j]
            shared = ck.keys() & cj.keys()
            if len(shared) < min_shared:
                continue
            if any(ck[p] != cj[p] for p in shared):
                continue                      # BYTE-IDENTICAL only
            parent[find(k)] = find(j)
            n_merged += 1

    entities: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for k in keys:
        entities[find(k)].append(k)
    for k in unmergeable:
        entities[k].append(k)

    lineages: list[Lineage] = []
    for i, member_keys in enumerate(sorted(entities.values(), key=lambda v: -len(v))):
        by_window: dict[tuple[str, int], list[WindowHaplotype]] = defaultdict(list)
        for key in member_keys:
            for m in tracks[key]:
                by_window[(m.contig, m.window_start)].append(m)
        groups = [
            WindowGroup(group_id=f"{lineage_prefix}{i:06d}_{w}", contig=c,
                        window_start=w, window_end=max(m.window_end for m in ms),
                        members=ms)
            for (c, w), ms in sorted(by_window.items())
        ]
        if groups:
            lineages.append(Lineage(f"{lineage_prefix}{i:06d}", groups[0].contig, groups))

    logging.info(
        f"  track merge: {len(tracks)} step-1 tracks ({len(unmergeable)} carry no "
        f"marker and stay singletons) -> "
        f"{len(lineages)} lineages | {n_merged} merges, {n_refused} refused by a "
        f"step-1 veto, min_shared_markers={min_shared}")
    return lineages
