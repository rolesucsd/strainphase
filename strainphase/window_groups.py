#!/usr/bin/env python3
"""Cross-sample grouping of haplotypes at a fixed genomic window.

This is the "vertical" linking axis. It is the counterpart to
:func:`strainphase.core.link_windows`, which is the "horizontal" axis:

    rows = samples (time), columns = windows (genome position)

                    window W0       window W1       window W2
      sample T0      h ------------- h ------------- h      -.
                     |               |               |        |  HORIZONTAL
      sample T1      h ------------- h ------------- h        |  = link_windows
                     |               |               |        |    (within one sample,
      sample T2      h ------------- h ------------- h      -'     across windows)

                     `---------------^---------------'
                        VERTICAL = this module (across samples, at ONE window)

Why the vertical axis runs on raw per-window haplotypes rather than on assembled
within-sample entities: windows are FIXED coordinate tiles
(``make_windows_lazy`` steps by ``window_size // 2``), so window ``W`` is the same
interval in every sample. Comparing at a fixed window therefore means every comparison
has an identical footprint - there is no span to gate, nothing to expand via min/max,
and no imputation gap. Comparing assembled entities instead means comparing objects with
different genomic extents, which is what previously required span-gap gating and let a
10 kb gate grow into a 4 Mb entity.

Two identity shapes are implemented. The choice between them is still open, so both are
available and produce identical output schemas:

``clique``
    Complete linkage: a group is a clique - every member passes the gates against every
    other member. No time axis at all, so it is immune to irregular timepoint spacing
    and to sample-ordering mistakes. Chaining is impossible by construction.

``reciprocal``
    Unique-best-on-both-sides plus mutual agreement between consecutive samples, with a
    per-haplotype dropout skip (``t -> t+2`` attempted ONLY for haplotypes that found no
    partner at ``t+1``, and only when the merge would not put two haplotypes of one
    sample in one group). Requires a genuinely chronological sample order to mean
    anything.

Failed comparisons are retained, not discarded, with the reason attached
(``failed_no_evidence`` vs ``failed_mismatch``). Downstream consumers need that
distinction to tell a measurement dropout from a real genotypic difference; a comparison
that simply returns nothing cannot be told apart from one that was never attempted.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from strainphase.core import (
    DEFAULT_CONFIG,
    HaplotyperConfig,
    compare_consensus,
    consensus_footprint,
    unique_best_matches,
    supported_marker_positions,
)

__all__ = [
    "WindowHaplotype",
    "WindowGroup",
    "GroupEdge",
    "group_window_across_samples",
    "group_all_windows",
]


@dataclass
class WindowHaplotype:
    """One haplotype at one window in one sample - the unit being grouped."""

    sample: str
    contig: str
    window_start: int
    window_end: int
    haplotype_id: str
    consensus: dict[int, str]
    reads: int = 0
    total_reads: int = 0        # reads that PHASED in this window
    junk_reads: int = 0         # reads that did not; carried so the denominator is a choice
    abundance: float = 0.0
    # The step-1 entity this haplotype belongs to (link_windows' track id, unique within a
    # sample+contig). Carried through so step 3 can use the WITHIN-sample chaining as
    # direct evidence that two step-2 groups continue into each other: if a sample's
    # link_windows entity contains a haplotype from group A at window W and one from group
    # B at W+step, that sample is a vote for joining A and B.
    within_sample_id: str = ""
    # Ids of the reads confidently assigned to this haplotype, populated only under
    # Step 3 joins
    # two window-groups when the same physical reads sit in both - direct evidence
    # that they continue into each other, and unlike a consensus comparison it does
    # not have to re-establish identity from a string at every window boundary.
    read_ids: frozenset = frozenset()


@dataclass
class WindowGroup:
    """A set of haplotypes at one window judged to be the same entity across samples."""

    group_id: str
    contig: str
    window_start: int
    window_end: int
    members: list[WindowHaplotype] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len({m.sample for m in self.members})

    @property
    def n_members(self) -> int:
        return len(self.members)


@dataclass
class GroupEdge:
    """One recorded pairwise comparison, whether or not it passed.

    Kept for BOTH outcomes on purpose. ``failed_no_evidence`` (a dropout - too few shared
    markers or too little overlap, nothing shown to differ) and ``failed_mismatch``
    (enough evidence, alleles genuinely disagree - a candidate recombination breakpoint)
    have to be distinguishable downstream, and a discarded comparison is indistinguishable
    from one never made.
    """

    contig: str
    window_start: int
    sample_a: str
    sample_b: str
    haplotype_a: str
    haplotype_b: str
    reason: str
    rate: float
    n_shared: int
    n_diff: int


def _pairwise(
    haps: list[WindowHaplotype],
    markers: set[int],
    config: HaplotyperConfig,
) -> tuple[dict[tuple[int, int], object], list[GroupEdge], "Counter"]:
    """Compare every pair once.

    Returns the gate results, the MISMATCH edges only, and a count of every outcome.

    Only mismatches are materialised. They are the sole verdict written out, and holding
    all of them cost ~2.4 GB of GroupEdge objects for a single MAG (3,988,701 comparisons
    on 000066952_0) which were then copied into dicts and filtered at write time. The
    outcome counts preserve everything the logs reported.
    """
    gates: dict[tuple[int, int], object] = {}
    edges: list[GroupEdge] = []
    counts: Counter = Counter()
    # Footprints are per-haplotype, so they are computed ONCE here rather than inside the
    # O(n^2) loop - this is the difference between O(n^2 * positions) and O(n * positions).
    spans = [consensus_footprint(h.consensus) for h in haps]
    for i in range(len(haps)):
        for j in range(i + 1, len(haps)):
            gate = compare_consensus(haps[i].consensus, haps[j].consensus, markers, config,
                                     a_span=spans[i], b_span=spans[j])
            gates[(i, j)] = gate
            counts[gate.reason] += 1
            if gate.reason != "failed_mismatch":
                continue
            edges.append(
                GroupEdge(
                    contig=haps[i].contig,
                    window_start=haps[i].window_start,
                    sample_a=haps[i].sample,
                    sample_b=haps[j].sample,
                    haplotype_a=haps[i].haplotype_id,
                    haplotype_b=haps[j].haplotype_id,
                    reason=gate.reason,
                    rate=round(gate.rate, 6),
                    n_shared=gate.n_shared,
                    n_diff=gate.n_diff,
                )
            )
    return gates, edges, counts


def _labels_clique(
    haps: list[WindowHaplotype],
    gates: dict[tuple[int, int], object],
    config: HaplotyperConfig = DEFAULT_CONFIG,
) -> list[int]:
    """Complete-linkage clustering: every member within threshold of every other.

    Complete linkage rather than connected components is the whole point - single linkage
    would let A join C through B without A and C ever being compared, which is exactly
    the accretion this replaces.
    """
    n = len(haps)
    if n == 1:
        return [0]
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gate = gates[(i, j)]
            # A failed gate is an explicit NON-MERGE verdict (distance 1.0), not
            # "unknown" - matching Strainy's treatment of insufficient overlap.
            d = gate.rate if gate.passed else 1.0
            dist[i][j] = dist[j][i] = d
    condensed = squareform(dist, checks=False)
    linkage_matrix = linkage(condensed, method="complete")
    # fcluster is inclusive at t, so nudge below 1.0 to keep explicit non-merges apart.
    # The threshold must come from the CALLER's config, not from DEFAULT_CONFIG: the gate
    # above already ran at `config.identity_distance`, and clustering at a different
    # value meant a raised --lineage-merge-distance admitted pairs the gate called linked
    # and then split them anyway - steps 2 and 3 running at two different identities.
    # A pair still
    # needs n_diff <= 1, so beyond 1/min_shared the absolute cap binds, not the rate.
    threshold = min(config.identity_distance, 0.999)
    return [int(x) for x in fcluster(linkage_matrix, t=threshold, criterion="distance")]


def _labels_reciprocal(
    haps: list[WindowHaplotype],
    gates: dict[tuple[int, int], object],
    sample_order: list[str],
) -> list[int]:
    """Unique-best + mutual between consecutive samples, with a per-haplotype skip gate."""
    by_sample: dict[str, list[int]] = {}
    for idx, hap in enumerate(haps):
        by_sample.setdefault(hap.sample, []).append(idx)

    def gate_of(i: int, j: int):
        return gates[(i, j)] if i < j else gates[(j, i)]

    def matched_pairs(left: list[int], right: list[int]) -> list[tuple[int, int]]:
        forward: dict[int, list[tuple[float, int]]] = {}
        backward: dict[int, list[tuple[float, int]]] = {}
        for a in left:
            for b in right:
                gate = gate_of(a, b)
                if gate.passed:
                    forward.setdefault(a, []).append((gate.rate, b))
                    backward.setdefault(b, []).append((gate.rate, a))
        best_a = unique_best_matches(forward)
        best_b = unique_best_matches(backward)
        return [(a, b) for a, b in best_a.items() if best_b.get(b) == a]

    graph = nx.Graph()
    graph.add_nodes_from(range(len(haps)))
    present = [s for s in sample_order if s in by_sample]

    # Per-haplotype dropout skip: `t -> t+2` is attempted ONLY for a haplotype that found
    # no partner at `t+1`. Gating per SAMPLE PAIR instead would make a haplotype that
    # legitimately needs a skip lose its link whenever some other haplotype in the same
    # sample happened to find a t+1 partner.
    skips: list[tuple[int, int]] = []
    for t in range(len(present) - 1):
        left = by_sample[present[t]]
        pairs = matched_pairs(left, by_sample[present[t + 1]])
        for a, b in pairs:
            graph.add_edge(a, b)
        linked = {a for a, _ in pairs}
        if t + 2 < len(present):
            skips.extend(
                (a, b)
                for a, b in matched_pairs(left, by_sample[present[t + 2]])
                if a not in linked
            )

    # Skip edges are applied LAST, and only when the merge they cause keeps one haplotype
    # per sample. The consecutive-sample edges above cannot break that on their own -
    # unique_best_matches is injective on both sides, so a component is a path visiting
    # samples in order - and per-haplotype gating rules out transitive TRIANGLES, since
    # the source of a skip edge has no t+1 edge by definition. Neither argument covers a
    # longer cycle, and the labels below come from connected components, i.e. single
    # linkage: a1~b, b~c and a skip a2~c close a 4-node cycle that puts BOTH of one
    # sample's haplotypes in one group. That contradicts the premise the pipeline rests
    # on - a strain has ONE haplotype at a locus - so the premise is checked directly
    # rather than argued from the edge pattern. Deferring the skips is what makes the
    # check meaningful: they are a repair for a dropout, so the structure they are being
    # tested against has to be the finished one.
    for a, b in skips:
        merged = (nx.node_connected_component(graph, a)
                  | nx.node_connected_component(graph, b))
        if len({haps[i].sample for i in merged}) < len(merged):
            continue
        graph.add_edge(a, b)

    labels = [0] * len(haps)
    for k, component in enumerate(nx.connected_components(graph)):
        for idx in component:
            labels[idx] = k
    return labels


def group_window_across_samples(
    haps: list[WindowHaplotype],
    markers: set[int],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    sample_order: list[str] | None = None,
    group_prefix: str = "G",
) -> tuple[list[WindowGroup], list[GroupEdge], Counter]:
    """Group the haplotypes at ONE window across samples.

    Returns the groups, the MISMATCH comparisons, and a count of every outcome.
    """
    if not haps:
        return [], [], Counter()

    gates, edges, counts = _pairwise(haps, markers, config)

    if config.cross_sample_method == "clique":
        labels = _labels_clique(haps, gates, config)
    elif config.cross_sample_method == "reciprocal":
        order = sample_order or sorted({h.sample for h in haps})
        labels = _labels_reciprocal(haps, gates, order)
    else:  # pragma: no cover - validated in HaplotyperConfig.__post_init__
        raise ValueError(f"unknown cross_sample_method {config.cross_sample_method!r}")

    by_label: dict[int, WindowGroup] = {}
    for hap, label in zip(haps, labels):  # noqa: B905
        group = by_label.get(label)
        if group is None:
            group = WindowGroup(
                group_id=f"{group_prefix}{hap.window_start}_H{label}",
                contig=hap.contig,
                window_start=hap.window_start,
                window_end=hap.window_end,
                members=[],
            )
            by_label[label] = group
        group.members.append(hap)

    return list(by_label.values()), edges, counts


def group_all_windows(
    haps: list[WindowHaplotype],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    sample_order: list[str] | None = None,
    site_type: dict[int, str] | None = None,
    markers_by_contig: dict[str, frozenset[int]] | None = None,
) -> tuple[list[WindowGroup], list[GroupEdge], Counter]:
    """Group every window of every contig, across samples.

    The marker set is computed ONCE PER CONTIG over every haplotype in every sample -
    the widest scope available, and the reason construction no longer prunes. Computing
    it per window would reintroduce a local definition of "variable".

    ``markers_by_contig`` lets the caller supply that set instead, so steps 2 and 3 judge
    identity on exactly the same positions. Passing it is the normal path; computing it
    here is the fallback for direct callers and tests.
    """
    by_contig: dict[str, list[WindowHaplotype]] = {}
    for hap in haps:
        by_contig.setdefault(hap.contig, []).append(hap)

    groups: list[WindowGroup] = []
    edges: list[GroupEdge] = []
    counts: Counter = Counter()

    for contig, contig_haps in sorted(by_contig.items()):
        markers = (markers_by_contig or {}).get(contig)
        if markers is None:
            markers = supported_marker_positions(
                ((h.sample, h.consensus, h.reads) for h in contig_haps), site_type, config
            )
        logging.info(
            f"  {contig}: {len(contig_haps)} window-haplotypes, "
            f"{len(markers)} identity markers, method={config.cross_sample_method}"
        )

        by_window: dict[int, list[WindowHaplotype]] = {}
        for hap in contig_haps:
            by_window.setdefault(hap.window_start, []).append(hap)

        for window_start in sorted(by_window):
            window_groups, window_edges, window_counts = group_window_across_samples(
                by_window[window_start],
                markers,
                config,
                sample_order=sample_order,
                group_prefix=f"{contig}_",
            )
            groups.extend(window_groups)
            edges.extend(window_edges)
            counts.update(window_counts)

    return groups, edges, counts
