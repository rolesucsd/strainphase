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

ABUNDANCE: POOLED COUNTS, TWO DENOMINATORS
   ``Lineage.abundance_by_sample()`` returns, per sample, Sum(reads) / Sum(denominator)
   over the windows this lineage occupies in that sample - the same estimator step 1 uses
   for a track, inherited one level up rather than redefined.

   Never a mean or median of the per-window abundances. Each window carries its own
   denominator (median 9 phased reads, varying ~4x across one sample's windows), 46% of
   windows hold a single haplotype whose abundance is 1.000 by construction, and the
   window set changes between timepoints - so an average over it moves with the window set
   rather than with the biology. That is what produced the sawtooth.

   Both denominators are reported. ``abundance`` divides by the reads that PHASED, which
   renormalises away how much of a window resolved; ``abundance_all_reads`` divides by
   every read at those loci, so a poorly-resolving window pulls the estimate down instead
   of being rescaled up to look like a good one.

   CAVEAT: adjacent windows overlap by 50%, so a read spanning the overlap is counted in
   both. Numerator and denominator inflate together and the ratio holds, but the
   overlapping region is effectively double-weighted.

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

__all__ = ["Lineage", "LineageEdge", "PooledAbundance", "build_lineages"]


@dataclass
class PooledAbundance:
    """One lineage's pooled read counts in one sample, with both denominators.

    Counts are pooled, never averaged: Sum(reads) / Sum(denominator) over the windows the
    lineage occupies in that sample. A mean or median of the per-window abundances would
    be wrong because each window carries its own denominator (median 9 phased reads,
    varying ~4x across one sample's windows), 46% of windows hold a single haplotype whose
    abundance is 1.000 by construction, and the window set changes between timepoints.
    """

    abundance: float            # reads / phased reads
    abundance_all_reads: float  # reads / (phased + junk)
    reads: int
    total_reads: int
    junk_reads: int
    n_windows: int


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

    def abundance_by_sample(self) -> dict[str, "PooledAbundance"]:
        """Pooled abundance per sample, both denominators.

        POOLED COUNTS, never an average of per-window ratios. ``Sum(supporting reads) /
        Sum(non-junk reads)`` over the windows this lineage occupies IN THAT SAMPLE - the
        same estimator step 1 uses for a track, inherited one level up.

        Averaging the per-window abundances instead would be wrong for three measured
        reasons: each window carries its own denominator (median 9 non-junk reads, varying
        ~4x across one sample's windows); 46% of windows hold a single haplotype whose
        abundance is 1.000 by construction; and the window set a lineage occupies changes
        between timepoints, so any average over it moves with the window set rather than
        with the biology. That last one is what produced the sawtooth.

        TWO denominators are returned because the choice is a real one.
        ``abundance`` divides by the reads that PHASED, which renormalises away how much
        of the window resolved - a window where 10% of reads phased scores the same as one
        where 90% did. ``abundance_all_reads`` divides by every read at those loci, so a
        poorly-resolving window pulls the estimate down instead of being rescaled up.

        ``n_windows`` is returned so a consumer can require a minimum, and the raw counts
        so a ratio of small numbers is visible as such rather than taken at face value.

        CAVEAT: adjacent windows overlap by 50%, so a read spanning the overlap is counted
        in both. Numerator and denominator inflate together and the ratio holds, but the
        overlapping region is effectively double-weighted.
        """
        acc: dict[str, list[int]] = {}
        # The denominators are a per-(sample, window) constant - one WindowResult's
        # non-junk and junk totals, copied onto every haplotype of that window - so they
        # are accumulated once per window while the numerator is summed over members. A
        # window where a strain split into two near-identical haplotypes would otherwise
        # count its own denominator twice and report half the abundance, i.e. an error
        # correlated with exactly the event the split represents.
        counted: set[tuple[str, int]] = set()
        for g in self.groups:
            for m in g.members:
                a = acc.setdefault(m.sample, [0, 0, 0, 0])
                a[0] += m.reads
                cell = (m.sample, g.window_start)
                if cell in counted:
                    continue
                counted.add(cell)
                a[1] += m.total_reads
                a[2] += m.junk_reads
                a[3] += 1
        out = {}
        for smp, (k, n, j, w) in acc.items():
            out[smp] = PooledAbundance(
                abundance=(k / n) if n else float("nan"),
                abundance_all_reads=(k / (n + j)) if (n + j) else float("nan"),
                reads=k, total_reads=n, junk_reads=j, n_windows=w)
        return out

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
    """Majority allele per position across the group's members. A TIE emits no call.

    ``Counter.most_common`` breaks a tie on first-increment order, which here is member
    order, which follows the order the samples were given on the command line - so the
    same BAMs listed as ``A,B`` and ``B,A`` produced different consensus bases and, once
    one of them fell in a forward overlap, different lineages. Dropping the position
    instead is both deterministic and the honest reading: this consensus is only ever
    used as a VETO, and a veto may not rest on a coin flip. A dropped position simply
    stops being shared, which costs at worst a ``failed_no_evidence`` - and that refuses
    nothing.
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    for m in group.members:
        for pos, base in m.consensus.items():
            votes[pos][base] += 1
    out: dict[int, str] = {}
    for p, c in votes.items():
        top = c.most_common(2)
        if len(top) > 1 and top[1][1] == top[0][1]:
            continue
        out[p] = top[0][0]
    return out


def _group_counts(group: WindowGroup) -> dict[str, tuple[int, int]]:
    """sample -> (supporting reads, non-junk reads). RAW counts: the abundance veto is a
    likelihood test on these, never on the derived (already quantised) ``abundance``.

    One sample can legitimately contribute TWO members to a group -
    ``merge_similar_haplotypes`` deliberately declines some 1-SNP pairs and both halves
    then clear the step-2 gate - so the supporting reads are summed across them. The
    denominator is not: it is one window's non-junk total, carried identically on every
    member, so it is taken once.
    """
    acc: dict[str, list[int]] = {}
    for m in group.members:
        a = acc.setdefault(m.sample, [0, m.total_reads])
        a[0] += m.reads
    return {s: (k, n) for s, (k, n) in acc.items()}


def _shares_incompatible(
    per_sample: dict[str, list[tuple[int, int]]],
    config: HaplotyperConfig,
    max_bad_frac: float,
    min_tested: int,
) -> tuple[bool, int, int]:
    """Do these per-sample share observations genuinely disagree? (the ELIMINATOR)

    ``per_sample`` maps a sample to the raw ``(supporting_reads, non_junk_reads)``
    counts being compared in it — two of them for one candidate join, N of them for
    a whole lineage. Fisher's exact test per sample via ``abundance_coherent``;
    samples where the test has no power (windows below ``min_reads_for_coherence``)
    report ``n_tested == 0`` and are skipped rather than failed.

    Too few testable samples means we cannot eliminate, which is NOT the same as
    passing — it returns no veto, and the identity gates carry the decision alone.

    This is the ONE place the tally-and-threshold lives; both the pairwise gate and
    the end-to-end chain check are thin adapters over it, so they can never drift
    apart.
    """
    tested = bad = 0
    for counts in per_sample.values():
        if len(counts) < 2:
            continue
        res = abundance_coherent(counts, config)
        if res.n_tested == 0:
            continue
        tested += 1
        if not res.coherent:
            bad += 1
    if tested < min_tested:
        return False, tested, bad
    return (bad / tested) > max_bad_frac, tested, bad


def _abundance_incompatible(
    counts_a: dict[str, tuple[int, int]],
    counts_b: dict[str, tuple[int, int]],
    config: HaplotyperConfig,
    max_bad_frac: float,
    min_tested: int,
) -> tuple[bool, int, int]:
    """PAIRWISE adapter: do these two groups' shares disagree, per sample?"""
    per_sample = {
        s: [counts_a[s], counts_b[s]] for s in counts_a.keys() & counts_b.keys()
    }
    return _shares_incompatible(per_sample, config, max_bad_frac, min_tested)


def _lineage_abundance_incompatible(
    groups: list[WindowGroup],
    config: HaplotyperConfig,
    max_bad_frac: float,
    min_tested: int,
) -> tuple[bool, int, int]:
    """END-TO-END adapter: do a WHOLE chain's windows disagree, per sample?

    The pairwise gate only ever compares two ADJACENT groups, so A-B and B-C can
    each pass while A and C are never compared and drift accumulates along the
    chain. On 000089747_1 that showed up as a lineage's own windows disagreeing by
    more than the noise floor at 34.2% of shared timepoints, while the median
    pairwise disagreement was ~0. Pooling every window of the chain into one
    per-sample comparison catches what no adjacent pair could.

    One pair per WINDOW, which is what ``abundance_coherent`` is contracted on. Two
    members of one sample inside one group are one strain that split into two
    near-identical haplotypes, not two windows disagreeing; handing them to Fisher as
    if they were would make any lineage carrying such a split incoherent at every cut
    and recurse it down to singletons.
    """
    per_window: dict[tuple[str, int], list[int]] = {}
    for g in groups:
        for m in g.members:
            cell = per_window.setdefault((m.sample, g.window_start), [0, m.total_reads])
            cell[0] += m.reads
    per_sample: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for (sample, _w), (k, n) in sorted(per_window.items()):
        per_sample[sample].append((k, n))
    return _shares_incompatible(per_sample, config, max_bad_frac, min_tested)


def _split_incoherent_lineage(
    groups: list[WindowGroup],
    votes: dict[tuple[str, str], int],
    config: HaplotyperConfig,
    max_bad_frac: float,
    min_tested: int,
) -> list[list[WindowGroup]]:
    """Cut an end-to-end-incoherent chain at its weakest link, then re-test.

    Reciprocal best match makes every component a PATH (asserted below), so the
    chain has a well-defined ordering and cutting it is unambiguous: remove the
    internal edge with the fewest step-1 link votes — the least-supported join —
    and re-test both halves. Recurses until every piece is coherent or is a
    single window.

    Splitting, rather than discarding, is deliberate: the windows are still real
    observations, and a chain that drifts is two lineages mis-joined, not noise.
    """
    ordered = sorted(groups, key=lambda g: g.window_start)
    if len(ordered) < 2:
        return [ordered]
    bad, _, _ = _lineage_abundance_incompatible(ordered, config, max_bad_frac, min_tested)
    if not bad:
        return [ordered]

    cut = min(
        range(len(ordered) - 1),
        key=lambda i: votes.get((ordered[i].group_id, ordered[i + 1].group_id), 0),
    )
    left, right = ordered[: cut + 1], ordered[cut + 1:]
    return (
        _split_incoherent_lineage(left, votes, config, max_bad_frac, min_tested)
        + _split_incoherent_lineage(right, votes, config, max_bad_frac, min_tested)
    )


def build_lineages(
    groups: list[WindowGroup],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    step: int | None = None,
    # ZERO TOLERANCE, matching step 1 (author's choice, 2026-07-30). One sample
    # whose shares genuinely disagree refuses the join, however many others vote
    # for it — the same rule link_windows already applied within a sample.
    #
    # Measured on 000089747_1, sweeping this value:
    #     0.30  1,256 lineages  316 multi-window  largest 16 windows  32.6% disagree
    #     0.20  1,299           344               largest 12          27.6%
    #     0.10  1,380           340               largest  7          21.6%
    #     0.00  1,425           337               largest  4          22.5%
    # Multi-window lineages are NOT lost by tightening (316 -> 337); long chains
    # are. The 16-window chain at 0.30 survived only because 30% of its samples
    # were allowed to disagree at every link.
    #
    # ⚠️ ~22% of shared timepoints still disagree by more than the noise floor at
    # ZERO tolerance, so this cannot be closed from here. The residual is upstream
    # in step-2 grouping (S2-6), not in how step 3 chains.
    max_bad_frac: float = 0.0,
    # Was 3. Any testable sample may now veto: requiring three shared samples
    # before a disagreement could count made the veto unreachable for a third of
    # accepted joins.
    #
    # There is deliberately NO matching rule on the accept side: a join with
    # `tested == 0` still proceeds. `tested == 0` means the test had no POWER —
    # both windows below `min_reads_for_coherence` (10) — not that the join is
    # doubtful, and 46% of real windows hold a single haplotype, so a shallow
    # window reading 1.000 is the normal case rather than a disagreement.
    # Rejecting on it would discard ~42 of ~563 accepted joins on 000089747_1,
    # concentrated exactly where coverage is thinnest. Tried as a flag and
    # removed rather than left as dead configuration (S3-5).
    min_samples_for_veto: int = 1,
    transitive_abundance_check: bool = True,
    lineage_prefix: str = "LIN",
    markers: set[int] | None = None,
    step1_mismatches: dict[frozenset[str], set[str]] | set[frozenset[str]] | None = None,
    # A step-1 within-sample mismatch vetoes only when >= this many DISTINCT
    # timepoints flag the group-to-group join. See config.step1_veto_min_timepoints.
    min_mismatch_timepoints: int = 2,
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
    # Normalise to {pair: {timepoints that flagged it}}. A legacy set (no timepoint
    # info) treats each pair as its own timepoint, so an explicit dict is required to
    # exercise the >=2-timepoint rule; a bare set of one pair no longer vetoes.
    if isinstance(step1_mismatches, dict):
        mismatched = step1_mismatches
    elif step1_mismatches:
        mismatched = {pair: {f"_tp{i}"} for i, pair in enumerate(step1_mismatches)}
    else:
        mismatched = {}

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
                # A within-sample mismatch vetoes the join only when >= this many
                # DISTINCT timepoints flag it. One timepoint's per-window EM miscall
                # (~0.03%/site) trips the zero-tolerance link gate over a short overlap
                # but is not corroborated; a genuine strain difference is flagged in
                # every timepoint the strains co-occur. Corroboration separates them.
                if mismatched:
                    veto_tps: set[str] = set()
                    for a in hap_ids[ga.group_id]:
                        for b in hap_ids[gb.group_id]:
                            flagged = mismatched.get(frozenset((a, b)))
                            if flagged:
                                veto_tps |= flagged
                    if len(veto_tps) >= min_mismatch_timepoints:
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
    # Each component is a chain of pairwise-approved joins. Re-test each one END TO
    # END and cut any that drifts; a chain of individually-fine links can still be
    # incoherent overall (see _lineage_abundance_incompatible).
    chains: list[list[WindowGroup]] = []
    n_split = 0
    for comp in nx.connected_components(graph):
        members = [gmap[x] for x in comp]
        if transitive_abundance_check and len(members) > 1:
            pieces = _split_incoherent_lineage(
                members, link_votes, config, max_bad_frac, min_samples_for_veto)
            if len(pieces) > 1:
                n_split += 1
            chains.extend(pieces)
        else:
            chains.append(sorted(members, key=lambda g: g.window_start))
    if n_split:
        logging.info(
            f"  transitive abundance check split {n_split} chain(s) into "
            f"{len(chains)} piece(s)")

    lineages: list[Lineage] = []
    for i, members in enumerate(chains):
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
