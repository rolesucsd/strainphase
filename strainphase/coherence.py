#!/usr/bin/env python3
"""Abundance-coherence checking and the over-merge QC gate.

Single timepoint only. A genome cannot hold two frequencies at one locus at one
time, so two windows merged into one entity within one sample must agree on
abundance to within sampling error; applied across timepoints this would reject
real dynamics, so it never compares different samples.

The test is Fisher's exact on ``[[k1, n1-k1], [k2, n2-k2]]`` over the raw counts
(``k`` = supporting reads, ``n`` = non-junk reads), not the derived ``abundance``.
A likelihood test rather than a fixed cutoff so it self-tightens as depth grows.
See docs/design/coherence.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import fisher_exact

from strainphase.core import DEFAULT_CONFIG, HaplotyperConfig

__all__ = [
    "CoherenceResult",
    "abundance_coherent",
    "qc_flags",
    "QCFlags",
]


@dataclass
class CoherenceResult:
    coherent: bool
    n_tested: int
    n_incoherent: int
    min_p: float


def abundance_coherent(
    counts: list[tuple[int, int]],
    config: HaplotyperConfig = DEFAULT_CONFIG,
) -> CoherenceResult:
    """Are these ``(supporting_reads, non_junk_reads)`` pairs mutually compatible?

    ``counts`` are per-window raw counts for one entity in one sample. Windows below
    ``min_reads_for_coherence`` are excluded, not failed: at low depth the test has no
    power. Coherent unless more than half the tested pairs are incoherent; at exactly
    half it still merges, so a single unlucky pair cannot veto a merge. See
    docs/design/coherence.md.
    """
    usable = [(k, n) for k, n in counts if n >= config.min_reads_for_coherence]
    if len(usable) < 2:
        # Not enough evidence to reject; absence of a test is not a failure.
        return CoherenceResult(True, 0, 0, 1.0)

    n_tested = 0
    n_incoherent = 0
    min_p = 1.0
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            k1, n1 = usable[i]
            k2, n2 = usable[j]
            table = [[k1, max(n1 - k1, 0)], [k2, max(n2 - k2, 0)]]
            p = float(fisher_exact(table)[1])
            n_tested += 1
            min_p = min(min_p, p)
            if p < config.abundance_coherence_alpha:
                n_incoherent += 1

    coherent = n_incoherent <= n_tested / 2
    return CoherenceResult(coherent, n_tested, n_incoherent, min_p)


@dataclass
class QCFlags:
    """Per-entity over-merge detector. It flags over-merge for QC; it does not repair it."""

    entity_id: str
    too_many_per_cell: bool  # G1
    abundance_incoherent: bool  # G2
    horizontal_occupancy: bool  # G3
    max_members_per_cell: int
    n_windows: int
    n_samples: int

    @property
    def failed(self) -> bool:
        return self.too_many_per_cell or self.abundance_incoherent or self.horizontal_occupancy


def qc_flags(
    entity_id: str,
    members: list[dict],
    config: HaplotyperConfig = DEFAULT_CONFIG,
) -> QCFlags:
    """Run the three over-merge gates on one entity.

    ``members`` are dicts with at least ``sample``, ``window_start``, ``reads`` and
    ``total_reads``.

    G1  more than 2 members in one (sample, window) cell - a genome cannot be in three
        states at one locus at one time.
    G2  co-timepoint abundance incoherence (see ``abundance_coherent``), single
        timepoint only.
    G3  occupancy shape: a real entity traces a diagonal, an over-merged one fills a
        horizontal band. See docs/design/coherence.md.
    """
    cells: dict[tuple[str, int], int] = {}
    for m in members:
        key = (m["sample"], m["window_start"])
        cells[key] = cells.get(key, 0) + 1
    max_per_cell = max(cells.values()) if cells else 0

    by_sample: dict[str, list[tuple[int, int]]] = {}
    windows_by_sample: dict[str, set[int]] = {}
    for m in members:
        by_sample.setdefault(m["sample"], []).append((m["reads"], m["total_reads"]))
        windows_by_sample.setdefault(m["sample"], set()).add(m["window_start"])

    incoherent_samples = sum(
        0 if abundance_coherent(counts, config).coherent else 1
        for counts in by_sample.values()
    )

    n_windows = len({m["window_start"] for m in members})
    n_samples = len(by_sample)

    per_sample_windows = sorted(len(w) for w in windows_by_sample.values())
    if per_sample_windows:
        mid = len(per_sample_windows) // 2
        median_occ = (
            per_sample_windows[mid]
            if len(per_sample_windows) % 2
            else (per_sample_windows[mid - 1] + per_sample_windows[mid]) / 2
        )
    else:
        median_occ = 0
    horizontal = n_samples >= 5 and n_windows > 0 and (median_occ / n_windows) < (1 / 3)

    return QCFlags(
        entity_id=entity_id,
        too_many_per_cell=max_per_cell > 2,
        abundance_incoherent=incoherent_samples > 0,
        horizontal_occupancy=horizontal,
        max_members_per_cell=max_per_cell,
        n_windows=n_windows,
        n_samples=n_samples,
    )
