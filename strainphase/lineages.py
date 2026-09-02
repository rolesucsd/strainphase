#!/usr/bin/env python3
"""Lineage containers: one merged entity, and its pooled per-sample abundance.

The construction logic (``build_lineages`` and the cross-window chaining pass) was
removed once ``track_merge.build_lineages_from_tracks`` replaced steps 2 and 3; what
remains is the data these produce.

A lineage's per-sample abundance is pooled - ``sum(reads) / sum(total_reads)`` over
its windows in that sample - never the mean of the per-window ratios, which would
inflate the estimate. See docs/design/lineages.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field


from strainphase.window_groups import WindowGroup

__all__ = ["Lineage", "PooledAbundance"]

# Empty read-set sentinel for a group with no recorded reads. Not currently referenced
# elsewhere in this module.
_NO_READS: frozenset = frozenset()


@dataclass
class PooledAbundance:
    """One lineage's pooled read counts in one sample, with both denominators.

    Counts are pooled, never averaged: sum(reads) / sum(denominator) over the windows
    the lineage occupies in that sample. See docs/design/lineages.md.
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

        Pooled, not averaged: ``sum(supporting reads) / sum(non-junk reads)`` over the
        windows this lineage occupies in that sample - the same estimator step 1 uses
        for a track. ``abundance`` divides by the reads that phased;
        ``abundance_all_reads`` divides by every read at those loci, so a
        poorly-resolving window pulls it down. ``n_windows`` and raw counts are returned
        so a consumer can require a minimum. See docs/design/lineages.md.
        """
        acc: dict[str, list[int]] = {}
        # Denominators are a per-(sample, window) constant, accumulated once per window
        # while the numerator sums over members; counting per haplotype would halve the
        # abundance where a strain split into two near-identical haplotypes.
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
        """First and last marker position - what the lineage actually resolves, not the
        tiles it nominally covers."""
        pos = [p for g in self.groups for m in g.members for p in m.consensus]
        return (min(pos), max(pos)) if pos else (0, 0)
