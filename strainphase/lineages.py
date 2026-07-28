#!/usr/bin/env python3
"""Lineages: chain cross-sample window groups along the genome.

This is the third and last linking step. The first two are settled:

    step 1  link_windows              within ONE sample, across adjacent windows
    step 2  window_groups.group_all_windows
                                      across samples, at ONE fixed window
    step 3  THIS MODULE               chain step-2 groups across adjacent windows

A 20 kb window is a computational tile, not a biological boundary — a strain extends
across it — so the entities produced by step 2 have to be chained into something that
spans the genome. Because step-2 groups already span samples, a lineage built here has one
identity across every sample at once; only its genomic *extent* can vary by timepoint.

Three rules, all deliberate:

**Same identity gates as everywhere else.** Adjacent windows overlap by 50%, so two groups
are compared on the markers falling in that shared interval, through the same
``compare_consensus`` gate stack link_windows and step 2 use. No bespoke thresholds.

**Abundance is an ELIMINATOR, not an INDICATOR.** A strain sits at the same frequency in
every window it occupies, so two groups whose within-window shares genuinely disagree
cannot be one biological unit — that is a sound veto. But *agreeing* on frequency is weak
evidence in favour, because many haplotypes sit at similar frequencies: measured on the
union run, 57% of groups had more than one abundance-compatible partner, so scoring on it
only manufactures ambiguity. It therefore vetoes joins and never scores them.

**Reciprocal best match, never greedy.** Each group's best continuation is taken only when
the choice is mutual and unambiguous on both sides — the same ``unique_best_matches`` rule
as step 1 and step 2. A tie contributes no edge. This is what keeps a lineage from
accreting: every node has at most one predecessor and one successor, so a chain is a PATH
and its length is bounded by the number of windows rather than by a threshold.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import networkx as nx

from strainphase.coherence import abundance_coherent
from strainphase.core import (
    DEFAULT_CONFIG,
    HaplotyperConfig,
    compare_consensus,
    unique_best_matches,
)
from strainphase.window_groups import WindowGroup

__all__ = ["Lineage", "LineageEdge", "build_lineages"]


@dataclass
class Lineage:
    """A chain of window groups along one contig."""

    lineage_id: str
    contig: str
    groups: list[WindowGroup] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len({g.window_start for g in self.groups})

    @property
    def window_start(self) -> int:
        return min(g.window_start for g in self.groups)

    @property
    def window_end(self) -> int:
        return max(g.window_end for g in self.groups)

    @property
    def samples(self) -> set[str]:
        return {m.sample for g in self.groups for m in g.members}

    @property
    def marker_span(self) -> tuple[int, int]:
        """First and last MARKER position, which is what the lineage actually resolves —
        as opposed to the tiles it nominally covers."""
        pos = [p for g in self.groups for m in g.members for p in m.consensus]
        return (min(pos), max(pos)) if pos else (0, 0)


@dataclass
class LineageEdge:
    """One attempted continuation, kept whether or not it was accepted.

    ``reason`` distinguishes why a join failed, which downstream needs in order to tell a
    measurement hole from a genuine genotypic wall (a candidate recombination breakpoint):

      ``linked``              accepted
      ``failed_no_evidence``  too few shared markers / too little overlap in the shared interval
      ``failed_mismatch``     enough evidence, the alleles genuinely disagree
      ``failed_abundance``    identity passed, but the within-window shares are incompatible
      ``failed_not_mutual``   best match, but not reciprocated (or a tie on either side)
    """

    contig: str
    window_a: int
    window_b: int
    group_a: str
    group_b: str
    reason: str
    rate: float
    n_shared: int
    n_diff: int
    n_samples_tested: int
    n_samples_incompatible: int


def _group_consensus(group: WindowGroup) -> dict[int, str]:
    """Majority allele per position across the group's members."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for m in group.members:
        for pos, base in m.consensus.items():
            votes[pos][base] += 1
    return {p: c.most_common(1)[0][0] for p, c in votes.items()}


def _group_counts(group: WindowGroup) -> dict[str, tuple[int, int]]:
    """sample -> (supporting reads, non-junk reads). RAW counts: the abundance veto is a
    likelihood test on these, never on the derived (already quantised) ``abundance``."""
    return {m.sample: (m.reads, m.total_reads) for m in group.members}


def _abundance_incompatible(
    counts_a: dict[str, tuple[int, int]],
    counts_b: dict[str, tuple[int, int]],
    config: HaplotyperConfig,
    max_bad_frac: float,
    min_tested: int,
) -> tuple[bool, int, int]:
    """Do these two groups' shares genuinely disagree? (the ELIMINATOR)

    Per sample where both are observed with enough depth, Fisher's exact test on the raw
    counts. Too few testable samples means we cannot eliminate, which is NOT the same as
    passing — it returns no veto, and the identity gates carry the decision alone.
    """
    tested = bad = 0
    for s in counts_a.keys() & counts_b.keys():
        res = abundance_coherent([counts_a[s], counts_b[s]], config)
        if res.n_tested == 0:
            continue
        tested += 1
        if not res.coherent:
            bad += 1
    if tested < min_tested:
        return False, tested, bad
    return (bad / tested) > max_bad_frac, tested, bad


def build_lineages(
    groups: list[WindowGroup],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    step: int | None = None,
    max_bad_frac: float = 0.30,
    min_samples_for_veto: int = 3,
    lineage_prefix: str = "LIN",
) -> tuple[list[Lineage], list[LineageEdge]]:
    """Chain window groups into lineages by reciprocal best match.

    ``step`` is the window stride (defaults to ``window_size // 2``, i.e. 50% overlap).
    ``max_bad_frac`` is the share of testable samples that may disagree on abundance
    before the join is vetoed.

    Returns the lineages and EVERY attempted continuation with its outcome.
    """
    if step is None:
        step = config.window_size // 2

    by_key: dict[tuple[str, int], list[WindowGroup]] = defaultdict(list)
    for g in groups:
        by_key[(g.contig, g.window_start)].append(g)

    cons = {g.group_id: _group_consensus(g) for g in groups}
    counts = {g.group_id: _group_counts(g) for g in groups}
    edges: list[LineageEdge] = []
    graph = nx.Graph()
    graph.add_nodes_from(g.group_id for g in groups)

    for (contig, w), left in sorted(by_key.items()):
        right = by_key.get((contig, w + step))
        if not right:
            continue
        # markers must fall in the interval the two windows SHARE
        region = (w + step, min(g.window_end for g in left) - 1)

        forward: dict[str, list[tuple[float, str]]] = {}
        backward: dict[str, list[tuple[float, str]]] = {}
        pending: dict[tuple[str, str], LineageEdge] = {}

        for ga in left:
            for gb in right:
                gate = compare_consensus(
                    cons[ga.group_id], cons[gb.group_id], set(cons[ga.group_id]),
                    config, min_shared=config.min_shared_for_lineage, region=region,
                    min_cospan_frac=0.0,       # the region IS the constraint here
                )
                e = LineageEdge(contig, w, w + step, ga.group_id, gb.group_id,
                                gate.reason, round(gate.rate, 6), gate.n_shared,
                                gate.n_diff, 0, 0)
                if not gate.passed:
                    edges.append(e)
                    continue
                veto, tested, bad = _abundance_incompatible(
                    counts[ga.group_id], counts[gb.group_id], config,
                    max_bad_frac, min_samples_for_veto)
                e.n_samples_tested, e.n_samples_incompatible = tested, bad
                if veto:
                    e.reason = "failed_abundance"
                    edges.append(e)
                    continue
                # score is the identity distance ONLY - abundance never scores
                forward.setdefault(ga.group_id, []).append((gate.rate, gb.group_id))
                backward.setdefault(gb.group_id, []).append((gate.rate, ga.group_id))
                pending[(ga.group_id, gb.group_id)] = e

        best_f = unique_best_matches(forward)
        best_b = unique_best_matches(backward)
        for a, b in best_f.items():
            e = pending[(a, b)]
            if best_b.get(b) == a:
                e.reason = "linked"
                graph.add_edge(a, b)
            else:
                e.reason = "failed_not_mutual"
            edges.append(e)
        for (a, b), e in pending.items():
            if best_f.get(a) != b:
                e.reason = "failed_not_mutual"
                edges.append(e)

    gmap = {g.group_id: g for g in groups}
    lineages: list[Lineage] = []
    for i, comp in enumerate(nx.connected_components(graph)):
        members = [gmap[x] for x in comp]
        lineages.append(Lineage(f"{lineage_prefix}{i:06d}", members[0].contig,
                                sorted(members, key=lambda g: g.window_start)))

    reasons = Counter(e.reason for e in edges)
    logging.info(
        f"  lineages: {len(lineages)} from {len(groups)} window groups | "
        f"continuations: {dict(reasons)}"
    )
    return lineages, edges
