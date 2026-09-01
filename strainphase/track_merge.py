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

THE MERGE RULE (2026-08-31)
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

from strainphase.coherence import abundance_coherent
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


def _edges_and_conflicts(
    keys, cons
) -> tuple[dict[frozenset, int], set[frozenset]]:
    """Every track pair's AGREEING marker count, and every pair that disagrees.

    Built by bucketing tracks on ``(position, allele)`` rather than comparing whole
    consensuses: two tracks in one bucket agree at that position, and two tracks in
    different buckets AT THE SAME POSITION disagree. One pass over the markers gives
    both the edge weights and the cannot-link set, and never compares a pair that has
    no marker in common.

    A single disagreement is disqualifying, so the returned sets can overlap - a pair
    may agree at 40 markers and disagree at one. The caller must treat the conflict as
    final; that is the whole point of byte-for-byte identity.
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
    """Merge tracks across samples, then emit one Lineage per merged entity.

    THE GRAPH. Every track pair sharing at least one marker is an edge weighted by how
    many markers they AGREE on. A pair that disagrees at even one shared marker is not
    an edge but a CANNOT-LINK: recorded, not merely skipped. That distinction is the
    reason this is not union-find over agreeing pairs. Byte-identity is not transitive
    across different marker subsets - A and B can agree on {1,2}, B and C on {3,4}, and
    A and C disagree on {5} - so merging on agreement alone fuses A and C through B
    without ever comparing them.

    THE MERGE. Pairs are taken in DESCENDING weight, so the strongest evidence merges
    first and a one-marker coincidence can only join whatever the strong evidence has
    already built. Before two components are joined:

      1. component-to-component cannot-link - any forbidden pair ACROSS the two
         components refuses the merge, not just the pair on the edge. Checking only the
         edge endpoints would reopen the transitive hole one level up.
      2. for every sample present in BOTH components, the two tracks it contributes are
         put through step 1's own gates: a recorded step-1 refusal refuses the merge.
      3. where step 1 never compared them - tracks more than one window apart, which
         link_windows never puts side by side - the abundance test is run now, because
         a genome cannot sit at two frequencies in one sample at one moment.

    ``cannot_link`` holds ``frozenset((track_key_a, track_key_b))`` pairs step 1 already
    refused. Each entity is split back into one ``WindowGroup`` per window it occupies,
    so everything downstream reads the structures it always has.
    """
    if markers is None:
        raise ValueError("markers must be supplied; steps must judge identity on one set")
    tracks = tracks_of(haps)
    cons = {k: track_consensus(v, markers) for k, v in tracks.items()}
    # A track carrying no marker cannot be MERGED - there is nothing to agree on - but it
    # must still be REPORTED. On B. fragilis 000089747_1, 7,120 of 14,978 tracks (47%)
    # carry no marker under the read-supported set, and dropping them would silently
    # discard nearly half the phased sequence rather than emitting it as unmergeable.
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
                # more than one window apart: link_windows never put these side by
                # side, so its abundance verdict does not exist and is taken now
                lo_a, hi_a = span[other]
                lo_b, hi_b = span[k]
                gap = max(lo_b - hi_a, lo_a - hi_b)
                if gap > step_bp and total[k] and total[other]:
                    if not abundance_coherent(
                        [(reads[other], total[other]), (reads[k], total[k])], config
                    ).coherent:
                        return False
        return True

    n_merged = n_refused_link = n_refused_sample = 0
    for pair, w in sorted(weight.items(), key=lambda kv: -kv[1]):
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
        f"same-sample gate")
    return lineages
