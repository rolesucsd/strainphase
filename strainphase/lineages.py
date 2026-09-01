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

ONE RULE SET, SHARED WITH STEP 1 (2026-08-30)
=============================================
Steps 1 and 3 answer the same question at two scopes - "do these two continue into each
other?" - and used to answer it with different rules: step 1 linked on consensus identity
under a reciprocal best match and refused a pair it could not judge, step 3 linked on
shared reads under a forward-only best match and let an unjudgeable pair through. They now
run the same stack, in this order:

**Candidacy is PROXIMITY.** Every (group at W, group at W+step) pair is a candidate because
the windows are adjacent. Nothing has to pass a genotype test to be considered.

**Evidence GATES.** Too few shared markers to judge the pair (``failed_no_evidence``)
refuses it. This is step 1's rule, adopted here in 2026-08. Measured over 353k candidate
pairs it costs 0.9% of joins on the divergent panel but blocks 9,470 near-clonal joins
(99.999% ANI) that rested on ZERO discriminating markers - the population that cost 12-21
points of read-placement accuracy. A pair we cannot judge is not a pair we may join.

**Identity is a VETO, never the evidence for.** ``failed_mismatch`` refuses the join
outright - one sample is enough. It is computed on real markers only: positions that are
invariant across every haplotype cannot disagree, so including them would dilute a real
difference rather than measure one.

**Abundance is an ELIMINATOR, not an INDICATOR.** A strain sits at the same frequency in
every window it occupies, so genuinely disagreeing shares cannot be one unit. But *agreeing*
is weak evidence in favour - 57% of groups had more than one abundance-compatible partner -
so it vetoes and never scores. One tolerance (``lineage_max_bad_frac``) governs both steps;
within a sample there is one testable unit, so any value below 1.0 behaves identically.

**Shared reads SCORE.** Among admissible candidates, rank by how many physical reads sit in
both. Identity has already had its say, and where two candidates are byte-identical it
cannot discriminate - shared reads can. A floor on shared reads guards against
coincidence, not the gate.

**Reciprocal best match, never greedy.** A join is kept only when each side's best partner
is the other and the best is a strict winner. A tie contributes NO edge. Measured on 5
datasets with exact truth, reciprocal had lower cross-strain error than forward-only at
equal recall everywhere (div0025_k6: 0.0018 vs 0.0034 for -0.2% recall).

**A lineage may never fragment a step-1 track.** If two groups hold members of one
within-sample chain, that sample's own reads already called them one strain, and they are
unioned even where reciprocity refused the edge - unless the mismatch veto refused it, which
always wins. This is what makes longitudinal >= single structurally: without it, a sample
whose step-1 track spanned a whole contig could still be handed a fragmented lineage, and
because a read is labelled by its lineage, the intact chain was discarded. It is also what
makes reciprocity affordable: a continuation lost to a tie is recovered wherever any
sample's own chain crossed that boundary.

A lineage may therefore hold more than one group at a window, so the old one-group-per-window
invariant no longer holds; doubled cells are counted and logged instead of asserted away.

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
