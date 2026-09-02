#!/usr/bin/env python3
"""Data containers for cross-sample haplotype merging.

A ``WindowHaplotype`` is one haplotype at one window in one sample; a ``WindowGroup``
is the set of window-haplotypes at one window judged the same strain across samples.
The merging is done in :mod:`strainphase.track_merge`.

Windows are fixed coordinate tiles (``make_windows_lazy`` steps by
``window_size // 2``), so a window is the same interval in every sample and every
cross-sample comparison has an identical footprint. See docs/design/window_groups.md.
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
    # The step-1 track this haplotype belongs to (link_windows' chain id, unique within a
    # sample+contig); a haplotype link_windows never chained becomes a track of its own.
    within_sample_id: str = ""
    # Reads confidently assigned to this haplotype, from the per-window EM. Populated
    # during longitudinal integration; the cross-sample merge compares consensus markers
    # instead and does not read it back.
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
