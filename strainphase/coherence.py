#!/usr/bin/env python3
"""Abundance-coherence checking and the over-merge QC gate.

**Scope, which is the whole point of this module: SINGLE TIMEPOINT ONLY.**

A genome cannot hold two different frequencies at one locus at one time, so two windows
merged into the same entity *within one sample* must agree on abundance to within
sampling error. This is a *window-merging* check. It is never a cross-timepoint
comparison - real biology changes between timepoints, so applying it across time would
reject true dynamics.

Two design rules that are easy to get wrong:

1. **Test the RAW COUNTS, never the derived ``abundance``.** ``abundance`` is
   ``pi_k / (1 - pi_junk)``, already quantised onto unit fractions by low denominators.
   Fisher's exact test on ``[[k1, n1-k1], [k2, n2-k2]]`` with ``k`` = supporting reads and
   ``n`` = non-junk reads uses the evidence directly.

2. **Use a likelihood test, not a fixed threshold.** At a median of ~12 reads two windows
   at the *same* true frequency routinely differ by 0.3 or more from sampling alone. A
   fixed cutoff would reject real merges at low coverage and accept fake ones at high
   coverage; a test self-tightens as depth grows. (Comparator precedent uses the same
   shape: Floria's Poisson coverage-compatibility test, Strainy's depth-ratio link
   deletion - both likelihood tests, neither a distance.)
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

    ``counts`` are the per-window raw counts for ONE entity in ONE sample. Windows below
    ``min_reads_for_coherence`` are excluded from the test rather than failing it - at
    very low depth the test has no power and would wave everything through, so including
    them would dilute the result.

    Returns coherent=True unless MORE than half of the tested pairs are incoherent - at
    exactly half it still merges. The tolerance is the point: a single unlucky pair
    should not veto a merge, and one outlier among m windows produces m-1 incoherent
    pairs out of m(m-1)/2, i.e. exactly half at m=4. Excluding the half case would make
    a lone outlier veto at four windows and not at five, which is a threshold artefact
    rather than a rule.
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
    """Per-entity over-merge detector. Not a fix - a detector, so that over-merge cannot
    pass silently whatever the linking algorithm does."""

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

    ``members`` are dicts carrying at least ``sample``, ``window_start``, ``reads`` and
    ``total_reads``.

    G1  more than 2 members in a single (sample, window) cell - a genome cannot be in
        three states at one locus at one time.
    G2  co-timepoint abundance incoherence (see ``abundance_coherent``), single
        timepoint only.
    G3  occupancy shape. A real entity traces a DIAGONAL: it occupies roughly its full
        window set at each timepoint, walking along the contig. An over-merged one fills
        a HORIZONTAL band - many windows piled up across timepoints without any single
        timepoint holding them all.
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
