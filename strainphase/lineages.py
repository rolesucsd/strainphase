#!/usr/bin/env python3
"""Lineages: chain cross-sample window groups along the genome.

This is the third and last linking step:

    step 1  link_windows                     within ONE sample, across adjacent windows
    step 2  window_groups.group_all_windows  across samples, at ONE fixed window
    step 3  THIS MODULE                      chain step-2 groups along the genome

A 20 kb window is a computational tile, not a biological boundary, so the entities step 2
produces have to be chained into something that spans the genome. Because step-2 groups
already span samples, a lineage built here has one identity across every sample at once;
only its genomic *extent* can vary by timepoint.

⚠️ THE STANDING CAVEAT — everything here assumes step 2 got the unit right
=========================================================================
This module chains step-2 GROUPS. If two haplotypes at one window should have been one
group and were not, that error is invisible from here and propagates into every lineage
built on them. Known causes, none fully resolved:

  * haplotypes with disjoint marker footprints are never comparable, so complete linkage
    scores them distance 1.0 - an explicit NON-merge - having shown nothing to differ
  * a strain carrying a divergent segment used to be split across it entirely; merging
    split molecules at load time fixes the common case, not necessarily all of it

The symptom to watch for is a lineage that *ought* to hold two groups at one window. This
module makes that structurally impossible (see below), which is correct if step 2 is right
and a silent loss if it is not. Troubleshoot at the step that owns it - see S2-6 in
FIGURE4 STRAINPHASE_TROUBLESHOOTING.md - not by loosening the rules here.

THE RULES
=========

**Candidacy is PROXIMITY.** Every (group at W, group at W+step) pair is a candidate because
the windows are adjacent. Nothing has to pass a genotype test to be considered.

**Identity is a VETO, never the evidence for.** ``failed_mismatch`` refuses the join
outright - one sample is enough, and it outranks any number of votes. ``failed_no_evidence``
refuses nothing: 46% of windows carry no discriminating position in the forward overlap, so
gating on it would discard those joins wholesale. A veto is computed on real markers with
the clonal fallback DISABLED, because a negative verdict must not rest on positions that
were incapable of disagreeing.

**Abundance is an ELIMINATOR, not an INDICATOR.** A strain sits at the same frequency in
every window it occupies, so genuinely disagreeing shares cannot be one unit. But *agreeing*
is weak evidence in favour - 57% of groups had more than one abundance-compatible partner -
so it vetoes and never scores.

**Step-1 votes are the only score.** A vote is a sample whose ``link_windows`` chain holds a
member of BOTH groups: direct read-level evidence that they continue into each other. Zero
votes is not a join. Sample count, identity distance and abundance never score.

**Reciprocal best match, never greedy.** A join is kept only when each side's best partner
is the other and the best is a strict winner. A tie contributes NO edge.

Reciprocity is what bounds the result. Every node has at most one partner in each
direction, so a component is a PATH; every edge advances exactly one window, so a lineage
holds exactly ONE group per window and ``|lineage| <= n_windows`` by construction rather
than by tuning. That also satisfies step 2's cannot-link constraints for free - measured on
000066952_0, zero unions refused against 16,412 constraints - because a strain has one
haplotype at a locus, so two groups of one lineage at one window would be a contradiction
between the two steps.

The alternative considered and not taken was a vote floor with constrained union-find
(§5.2 of the troubleshooting doc): it links more (29.1% vs 21.5% multi-window) but produces
48-group components, needs the constraints actively enforced, and its size bound is
empirical rather than structural.

BOTH EXISTING TABLES ARE USED AS EVIDENCE; NEITHER IS RECOMPUTED
   step 1  ``windows_within_sample.tsv`` -> which haplotypes a sample's own reads chained
           across adjacent windows. Reaches this module through
           ``WindowHaplotype.within_sample_id`` and becomes the join SCORE.
           ``mismatches_within_sample.tsv`` -> the pairs it found to genuinely disagree,
           passed in as ``step1_mismatches`` and used as an absolute veto.
   step 2  ``windows_across_samples.tsv`` -> the groups being chained, i.e. the nodes.
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
    consensus_footprint,
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
    # samples whose step-1 (within-sample) chain contains a haplotype from BOTH groups —
    # direct read-level evidence that these two continue into each other
    n_link_votes: int = 0


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
    markers: set[int] | None = None,
    step1_mismatches: set[frozenset[str]] | None = None,
) -> tuple[list[Lineage], list[LineageEdge]]:
    """Chain window groups into lineages by reciprocal best match on step-1 votes.

    ``markers`` is the identity marker set, computed at the WIDEST scope available (all
    haplotypes on the contig, every sample - the same set step 2 used). Passing None
    reproduces the old behaviour of comparing on every co-covered position, which made
    the marker restriction a no-op.

    ``step1_mismatches`` are ``{haplotype_id_a, haplotype_id_b}`` pairs that link_windows
    compared and found to GENUINELY DISAGREE. One is enough to veto a join: a sample's own
    reads saying two haplotypes are not the same genome outranks any number of votes.

    Returns the lineages and EVERY attempted continuation with its outcome.
    """
    if step is None:
        step = config.window_size // 2
    if markers is None:
        markers = {p for g in groups for m in g.members for p in m.consensus}
    mismatched = step1_mismatches or set()

    by_key: dict[tuple[str, int], list[WindowGroup]] = defaultdict(list)
    for g in groups:
        by_key[(g.contig, g.window_start)].append(g)

    # ---- step-1 evidence: which groups does a within-sample chain already connect? ----
    # link_windows chained haplotypes across adjacent windows inside each sample. Step 2
    # then assigned each of those haplotypes to a cross-sample group. So a sample whose
    # chain holds a member of group A at W and a member of group B at W+step is a direct
    # read-level vote that A continues into B. Both existing tables are used as evidence;
    # nothing is recomputed.
    chain_pos: dict[tuple[str, str, str], dict[int, str]] = defaultdict(dict)
    for g in groups:
        for m in g.members:
            if m.within_sample_id:
                chain_pos[(m.sample, g.contig, m.within_sample_id)][g.window_start] = g.group_id
    link_votes: Counter = Counter()
    for (_sample, _contig, _eid), wins in chain_pos.items():
        for w, ga in wins.items():
            gb = wins.get(w + step)
            if gb is not None and gb != ga:
                link_votes[(ga, gb)] += 1

    cons = {g.group_id: _group_consensus(g) for g in groups}
    counts = {g.group_id: _group_counts(g) for g in groups}
    hap_ids = {g.group_id: {m.haplotype_id for m in g.members} for g in groups}
    edges: list[LineageEdge] = []
    graph = nx.Graph()
    graph.add_nodes_from(g.group_id for g in groups)

    for (contig, w), left in sorted(by_key.items()):
        right = by_key.get((contig, w + step))
        if not right:
            continue
        # markers must fall in the interval the two windows SHARE
        region = (w + step, min(g.window_end for g in left) - 1)

        span = {g.group_id: consensus_footprint(cons[g.group_id], region)
                for g in (*left, *right)}
        forward: dict[str, list[tuple[float, str]]] = {}
        backward: dict[str, list[tuple[float, str]]] = {}
        pending: dict[tuple[str, str], LineageEdge] = {}

        for ga in left:
            for gb in right:
                # CANDIDACY IS PROXIMITY ALONE. The identity comparison below is a VETO,
                # not the gate that admits a pair: `failed_no_evidence` must not block,
                # because 46% of windows carry no discriminating position in the forward
                # overlap and gating on it discards those joins outright.
                gate = compare_consensus(
                    cons[ga.group_id], cons[gb.group_id], markers,
                    config, min_shared=config.min_shared_for_lineage, region=region,
                    min_cospan_frac=0.0,       # the region IS the constraint here
                    allow_fallback=False,      # a veto may not rest on padded evidence
                    a_span=span[ga.group_id], b_span=span[gb.group_id],
                )
                e = LineageEdge(contig, w, w + step, ga.group_id, gb.group_id,
                                gate.reason, round(gate.rate, 6), gate.n_shared,
                                gate.n_diff, 0, 0)
                if gate.reason == "failed_mismatch":
                    edges.append(e)
                    continue
                # A single sample whose OWN reads disagree across this boundary vetoes the
                # join outright, however many other samples vote for it.
                if mismatched and any(
                    frozenset((a, b)) in mismatched
                    for a in hap_ids[ga.group_id] for b in hap_ids[gb.group_id]
                ):
                    e.reason = "failed_mismatch"
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
                # SCORE: step-1 link votes and nothing else. Identity has already had its
                # say as a veto, abundance only eliminates, and sample count never scores.
                # No votes means no read ever chained these two - not a join.
                votes = link_votes.get((ga.group_id, gb.group_id), 0)
                e.n_link_votes = votes
                if votes == 0:
                    e.reason = "failed_no_votes"
                    edges.append(e)
                    continue
                score = -float(votes)   # unique_best_matches takes lower-is-better
                forward.setdefault(ga.group_id, []).append((score, gb.group_id))
                backward.setdefault(gb.group_id, []).append((score, ga.group_id))
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

    # Reciprocity makes every component a PATH, and every edge advances one window, so a
    # lineage can never hold two groups at the same window. That is what satisfies the
    # step-2 cannot-link constraints for free (measured: 0 unions refused against 16,412
    # constraints on 000066952_0) - a strain has ONE haplotype at a locus, so two groups
    # of one lineage at one window would mean a contradiction between the two steps.
    for lin in lineages:
        per_window = Counter(g.window_start for g in lin.groups)
        if per_window and max(per_window.values()) > 1:  # pragma: no cover - invariant
            raise AssertionError(
                f"{lin.lineage_id} holds {max(per_window.values())} groups at one window; "
                "reciprocal best match should make this impossible"
            )

    linked = [e for e in edges if e.reason == "linked"]
    reasons = Counter(e.reason for e in edges)
    if linked:
        logging.info(
            f"  step-1 backing: {sum(1 for e in linked if e.n_link_votes)}/{len(linked)} "
            f"accepted joins carry a within-sample chain link"
        )
    logging.info(
        f"  lineages: {len(lineages)} from {len(groups)} window groups | "
        f"continuations: {dict(reasons)}"
    )
    return lineages, edges
