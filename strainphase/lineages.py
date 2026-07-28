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


THE FULL RULE SET
=================

Every gate below must pass for two groups to be joined. They are applied in this order,
and the first failure is recorded as the edge's ``reason``.

1. WHAT COUNTS AS A COMPARABLE POSITION
   marker set            a position is an identity marker only if >=2 distinct alleles are
                         observed across the whole contig, across all samples. A position
                         where everything agrees carries no identity information yet still
                         dilutes the mismatch rate (42.6% of emitted positions were
                         invariant MAG-wide on 000089747_1).
   SV exclusion          ``exclude_sv_from_identity=True``. Structural variants are loaded,
                         phased and reported, but never used as identity markers: an
                         invertible promoter at af~0.5 flips independently of strain
                         background, so it would split a lineage every time it flips.
   clonal fallback       if fewer than ``min_shared`` MARKERS are shared, fall back to all
                         co-covered positions. A clonal locus genuinely has no variable
                         sites; absence of discriminating evidence is not evidence of
                         difference, and without this every clonal lineage shatters into
                         singletons (85% of windows hold one haplotype).

2. IS THERE ENOUGH EVIDENCE TO COMPARE AT ALL?  -> ``failed_no_evidence``
   min overlapping positions
                         ``min_shared_for_lineage``, default **3**. Fewer shared markers
                         than this and the pair is not compared. (Runs to date pass 2.)
   min physical overlap  ``min_entity_overlap_bp``, default **1000 bp** between the first
                         and last shared marker. Below it the verdict is an explicit
                         NON-MERGE, not "unknown" — Strainy's ``I = 1000``. (Runs pass 500.)
   shared interval       adjacent windows overlap by 50% (``step = window_size // 2``,
                         i.e. 10 kb of a 20 kb window). ONLY markers inside that interval
                         are eligible; the region is passed explicitly, which is why the
                         co-supported-span fraction (``min_cosupported_span_frac``, 0.25
                         at step 1) is set to 0 here — the region already IS the constraint.

3. DO THE ALLELES AGREE?  -> ``failed_mismatch``
   max absolute mismatches
                         ``max_num_diff``. HARD CAP: no more than this many differing
                         positions regardless of how long the comparison is. Library
                         default **1**; runs to date pass **3**. This is the gate that
                         binds at high evidence — the rate below is applied as a FLOOR
                         (``int(rate * n_shared)``), so at n_shared=1172 a 1% rate alone
                         would tolerate 11 mismatches. The two guard opposite ends: the
                         rate forces 0 mismatches below n_shared=100, the cap takes over
                         above n_shared=200.
   max mismatch rate     ``lineage_merge_distance``, default **0.01** (1%).

4. DO THE ABUNDANCES ALLOW IT?  -> ``failed_abundance``   (ELIMINATOR ONLY)
   test                  Fisher's exact on the RAW counts ``[[k_a, n_a-k_a], [k_b, n_b-k_b]]``
                         per sample where both groups are observed. Never on the derived
                         ``abundance``, which is already quantised by
                         ``pi_k / (1 - pi_junk)``.
   significance          ``abundance_coherence_alpha``, default **0.01**.
   min depth to test     ``min_reads_for_coherence``, default **10** non-junk reads on BOTH
                         sides. A likelihood test rather than a fixed threshold, so the
                         rule self-tightens with depth instead of rejecting real merges at
                         low coverage.
   veto threshold        ``max_bad_frac``, default **0.30** — the join is vetoed when more
                         than 30% of testable samples disagree.
   min testable samples  ``min_samples_for_veto``, default **3**. Below this the veto does
                         NOT fire and the identity gates decide alone: an eliminator must
                         not block on absence of evidence.

5. IS THE CHOICE UNAMBIGUOUS?  -> ``failed_not_mutual``
   reciprocal best       each side's best partner must be the other, and the best must be
                         a strict winner. A TIE CONTRIBUTES NO EDGE.
   score                 STEP-1 LINK VOTES first, identity mismatch rate as the tiebreak,
                         encoded as one "lower is better" number
                         (``-votes * 1000 + rate``) so an exact tie on both still yields
                         no edge. A vote is a sample whose link_windows chain contains a
                         member of BOTH groups — direct read-level evidence that the two
                         continue into each other. Abundance never scores and sample count
                         never scores. With no votes anywhere the score degrades to the
                         identity rate alone, which is the sensible fallback.
                         Length is preferred implicitly: a chain extends as far as the
                         evidence unambiguously supports and stops at the first ambiguity,
                         never guessing between two candidates.

BOTH EXISTING TABLES ARE USED AS EVIDENCE; NEITHER IS RECOMPUTED
   step 1  ``windows_within_sample.tsv`` -> which haplotypes a sample's own reads already
           chained across adjacent windows. Reaches this module through
           ``WindowHaplotype.within_sample_id`` and becomes the join SCORE.
   step 2  ``windows_across_samples.tsv`` -> which haplotypes are the same entity across
           samples at one window. These groups are the nodes being chained.
   So both ends are already joined, and step 3 only has to decide which pairing is the
   consistent one.

UPSTREAM GATES THAT SHAPE THE INPUT (not applied here, but they decide what exists)
   window_size 20000, step 10000        50% overlap so adjacent windows share markers
   min_reads_per_window 10              reads needed to PHASE a window de novo
   min_reads_for_rescue 5               reads needed to BUILD one, so rescue can fill it
   max_reads_per_window 500             subsample cap
   min_read_window_overlap_bp 1000      a read must cover this much of a window to count
   min_read_read_overlap_bp 1000        two reads must overlap this much to be compared
   min_shared_snvs_for_edge 3           read-read graph, seeds the EM
   min_shared_snvs_for_link 3           step 1: window-level shared SNV positions
   min_shared_calls_for_link 3          step 1: haplotype-level shared actual calls
   max_link_distance 0.01               step 1: mismatch rate
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
                # SCORE: step-1 link votes first (direct read-level evidence that these
                # two continue into each other), identity distance as the tiebreak.
                # Abundance never scores - it only eliminates. Sample count never scores.
                # Encoded as one "lower is better" number so an exact tie on both still
                # yields no edge.
                votes = link_votes.get((ga.group_id, gb.group_id), 0)
                e.n_link_votes = votes
                score = -votes * 1000.0 + gate.rate
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
