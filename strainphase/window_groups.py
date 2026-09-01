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

from dataclasses import dataclass, field



__all__ = [
    "WindowHaplotype",
    "WindowGroup",
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
