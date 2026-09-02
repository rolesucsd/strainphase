"""Merge within-sample tracks across samples into contig-spanning lineages.

A track is a sample's own haplotype chained across windows from that sample's
reads; it is not chimeric in practice. The only remaining work is to merge tracks
that are the same strain across samples, which this module does as one clustering
rather than a per-window grouping and a separate chaining pass.

Two tracks merge when they agree byte-for-byte at every identity marker they both
call, over at least ``track_merge_min_shared_markers`` shared markers. Exact
agreement, rather than a mismatch rate, keeps a handful of real differences from
being diluted by thousands of identical sites at high coverage. The default of one
shared marker makes this a permissive first pass; later passes split.

A refusal recorded in step 1 — two tracks whose haplotypes disagreed at a called
allele — carries over as a cannot-link and is never merged.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from strainphase.coherence import abundance_coherent
from strainphase.core import DEFAULT_CONFIG, HaplotyperConfig
from strainphase.lineages import Lineage
from strainphase.window_groups import WindowGroup, WindowHaplotype

__all__ = ["build_lineages_from_tracks", "track_consensus", "tracks_of"]


def tracks_of(haps: list[WindowHaplotype]) -> dict[tuple[str, str], list[WindowHaplotype]]:
    """``(sample, within_sample_id) -> its window-haplotypes``, genomic order.

    A haplotype ``link_windows`` never placed in a track becomes a track of its own,
    so it can still be merged rather than being dropped.
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

    Weighted by reads, not counted per window, because adjacent windows overlap by 50%.
    A tie emits no call rather than resolving by member order. See
    docs/design/track_merge.md.
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


def _edges_and_conflicts(
    keys, cons
) -> tuple[dict[frozenset, int], set[frozenset]]:
    """Every track pair's agreeing marker count, and every pair that disagrees.

    Built by bucketing tracks on ``(position, allele)``, so one pass over the markers
    gives both the edge weights and the cannot-link set and never compares a pair with
    no marker in common. A single disagreement is disqualifying, so the returned sets
    can overlap; the caller must treat any conflict as final. See
    docs/design/track_merge.md.
    """
    by_pos_allele: dict[tuple[int, str], list] = defaultdict(list)
    for k in keys:
        for pos, base in cons[k].items():
            by_pos_allele[(pos, base)].append(k)

    weight: dict[frozenset, int] = defaultdict(int)
    conflict: set[frozenset] = set()
    by_pos: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for key in by_pos_allele:
        by_pos[key[0]].append(key)
    for pos, allele_keys in by_pos.items():
        for ak in allele_keys:
            bucket = by_pos_allele[ak]
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    weight[frozenset((bucket[i], bucket[j]))] += 1
        # different alleles at the SAME position: a genuine disagreement
        for i in range(len(allele_keys)):
            for j in range(i + 1, len(allele_keys)):
                for a in by_pos_allele[allele_keys[i]]:
                    for b in by_pos_allele[allele_keys[j]]:
                        conflict.add(frozenset((a, b)))
    return weight, conflict


def build_lineages_from_tracks(
    haps: list[WindowHaplotype],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    markers: frozenset[int] | set[int] | None = None,
    cannot_link: set[frozenset[str]] | None = None,
    lineage_prefix: str = "LIN",
) -> list[Lineage]:
    """Merge tracks across samples on a weighted agreement graph; emit one Lineage each.

    Track pairs sharing at least ``track_merge_min_shared_markers`` markers are edges
    weighted by agreeing-marker count; a pair disagreeing at any shared marker is a
    recorded cannot-link. Edges are taken in descending weight and joined only if no
    forbidden pair spans the two components, no shared sample fails step 1's gates, and
    (for tracks step 1 never compared) the abundance test passes. Each merged entity is
    split back into one ``WindowGroup`` per window. ``cannot_link`` carries step 1's own
    refusals as ``frozenset((track_key_a, track_key_b))``.

    Why a graph and not union-find, and why per-component gating: docs/design/track_merge.md.
    """
    if markers is None:
        raise ValueError("markers must be supplied; steps must judge identity on one set")
    tracks = tracks_of(haps)
    cons = {k: track_consensus(v, markers) for k, v in tracks.items()}
    # A track carrying no marker cannot be merged - there is nothing to agree on - but it
    # must still be reported. On a real MAG that is ~47% of tracks; dropping them would
    # silently discard nearly half the phased sequence instead of emitting it as
    # unmergeable.
    keys = [k for k, c in cons.items() if c]
    unmergeable = [k for k, c in cons.items() if not c]

    weight, conflict = _edges_and_conflicts(keys, cons)
    forbidden_pairs = set(conflict) | set(cannot_link or set())

    # per-track facts the gates below need
    span = {k: (min(m.window_start for m in v), max(m.window_start for m in v))
            for k, v in tracks.items()}
    reads = {k: sum(m.reads for m in v) for k, v in tracks.items()}
    total = {k: max((m.total_reads for m in v), default=0) for k, v in tracks.items()}
    step_bp = config.window_size // 2

    comp: dict = {k: k for k in keys}

    def find(x):
        while comp[x] != x:
            comp[x] = comp[comp[x]]
            x = comp[x]
        return x

    members: dict = {k: [k] for k in keys}
    forbidden: dict = defaultdict(set)
    for pair in forbidden_pairs:
        a, b = tuple(pair) if len(pair) == 2 else (None, None)
        if a in comp and b in comp:
            forbidden[a].add(b)
            forbidden[b].add(a)

    def same_sample_ok(xs, ys) -> bool:
        """Step 1's gates, applied to every sample both components hold."""
        by_sample: dict[str, list] = defaultdict(list)
        for k in xs:
            by_sample[k[0]].append(k)
        for k in ys:
            if k[0] not in by_sample:
                continue
            for other in by_sample[k[0]]:
                # a refusal step 1 already recorded stands
                if frozenset((k, other)) in forbidden_pairs:
                    return False
                # More than one window apart: link_windows never put these side by side,
                # so its abundance verdict does not exist and is taken now, on the same
                # config step 1 reads (--abundance-coherence-alpha,
                # --min-reads-for-coherence), so a pair is not judged differently by
                # distance alone.
                lo_a, hi_a = span[other]
                lo_b, hi_b = span[k]
                gap = max(lo_b - hi_a, lo_a - hi_b)
                if gap > step_bp and total[k] and total[other]:
                    if not abundance_coherent(
                        [(reads[other], total[other]), (reads[k], total[k])], config
                    ).coherent:
                        return False
        return True

    ordered = sorted(weight.items(), key=lambda kv: -kv[1])
    n_merged = n_refused_link = n_refused_sample = n_too_weak = 0
    for idx, (pair, w) in enumerate(ordered):
        if w < config.track_merge_min_shared_markers:
            # Weights descend, so no pair after this one clears the bar either.
            n_too_weak = len(ordered) - idx
            break
        a, b = tuple(pair)
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if pair in forbidden_pairs or (members[rb] and forbidden[ra] & set(members[rb])):
            n_refused_link += 1
            continue
        if not same_sample_ok(members[ra], members[rb]):
            n_refused_sample += 1
            continue
        comp[ra] = rb
        members[rb] = members[rb] + members[ra]
        forbidden[rb] |= forbidden[ra]
        n_merged += 1

    entities: dict = defaultdict(list)
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
        f"  track merge: {len(tracks)} step-1 tracks ({len(unmergeable)} carry no marker "
        f"and stay singletons) -> {len(lineages)} lineages | {len(weight)} candidate "
        f"edges, {len(conflict)} conflicting pairs | {n_merged} merged, "
        f"{n_refused_link} refused by cannot-link, {n_refused_sample} refused by a "
        f"same-sample gate, {n_too_weak} below "
        f"track_merge_min_shared_markers={config.track_merge_min_shared_markers}")
    return lineages
