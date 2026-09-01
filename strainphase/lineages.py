#!/usr/bin/env python3
"""Lineage containers: one merged entity, and its pooled per-sample abundance.

The construction logic that used to live here - ``build_lineages``, the cross-window
chaining pass, its veto stack and its ``lineage_max_bad_frac`` tolerance - was removed
once ``track_merge.build_lineages_from_tracks`` replaced steps 2 and 3. What remains is
the data these produce.

WHY ABUNDANCE IS POOLED, NOT AVERAGED. A lineage's per-sample abundance is
``sum(reads) / sum(total_reads)`` over its windows in that sample, never the mean of
the per-window ratios. The distinction is not cosmetic: a deep window where the strain
is rare (5/100) and a shallow one where it takes everything (3/3) average to 0.53,
while the reads actually say 8/103 = 0.078. The shallow window's 1.000 is an artefact
of nothing else resolving there, and 46% of real windows hold exactly one haplotype.
"""

from __future__ import annotations

from dataclasses import dataclass, field


from strainphase.window_groups import WindowGroup

__all__ = ["Lineage", "PooledAbundance"]

# Default for a group with no recorded reads. A bare () would raise TypeError on the
# set intersection below rather than reporting "no shared reads".
_NO_READS: frozenset = frozenset()


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
