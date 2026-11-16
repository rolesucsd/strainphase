#!/usr/bin/env python3
"""Abundance-coherence checking.

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
