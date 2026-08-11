"""Read-partition agreement between a tool's output and the truth.

Every tool in this benchmark, whatever it calls its output, ultimately decides
which reads belong together. Comparing those partitions to the true
read-to-strain assignment is the only comparison that needs no charitable
reinterpretation of any tool's output, so it is the benchmark's primary surface.

All four scores below are computed from a single contingency table. They are
implemented here rather than pulled from scikit-learn so that the benchmark has
no heavyweight dependency and so a reviewer can check the arithmetic against the
formulas in the docstrings.

Reported alongside every score is ``assigned_fraction``. A tool that places 5%
of the reads perfectly must not look better than one that places all of them
well, and the scores below are computed over placed reads only.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.special import gammaln

from spbench.formats import UNASSIGNED


def contingency(labels_true: list[str], labels_pred: list[str]) -> np.ndarray:
    """Contingency matrix ``n[i, j]`` = reads with true class i and cluster j."""
    true_index = {label: i for i, label in enumerate(sorted(set(labels_true)))}
    pred_index = {label: i for i, label in enumerate(sorted(set(labels_pred)))}
    matrix = np.zeros((len(true_index), len(pred_index)), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred, strict=True):
        matrix[true_index[t], pred_index[p]] += 1
    return matrix


def _comb2(values: np.ndarray) -> np.ndarray:
    return values * (values - 1) / 2.0


def adjusted_rand_index(matrix: np.ndarray) -> float:
    """ARI = (I - E) / (M - E) over pair counts, where ``I`` is the observed
    pair agreement, ``E`` its expectation under a fixed-marginal null and ``M``
    the maximum. 0 is chance, 1 is a perfect partition."""
    n = matrix.sum()
    if n < 2:
        return float("nan")
    sum_ij = _comb2(matrix.astype(float)).sum()
    a = _comb2(matrix.sum(axis=1).astype(float)).sum()
    b = _comb2(matrix.sum(axis=0).astype(float)).sum()
    total_pairs = n * (n - 1) / 2.0
    expected = a * b / total_pairs
    maximum = 0.5 * (a + b)
    if math.isclose(maximum, expected):
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


def _entropies(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return ``(H_true, H_pred, mutual_information)`` in nats."""
    n = matrix.sum()
    if n == 0:
        return 0.0, 0.0, 0.0
    p_ij = matrix / n
    p_i = p_ij.sum(axis=1)
    p_j = p_ij.sum(axis=0)

    def entropy(p: np.ndarray) -> float:
        nz = p[p > 0]
        return float(-(nz * np.log(nz)).sum())

    nz = p_ij > 0
    outer = np.outer(p_i, p_j)
    mi = float((p_ij[nz] * np.log(p_ij[nz] / outer[nz])).sum())
    return entropy(p_i), entropy(p_j), max(0.0, mi)


def homogeneity_completeness_v(matrix: np.ndarray) -> tuple[float, float, float]:
    """Homogeneity (no cluster mixes strains), completeness (no strain is split
    across clusters), and their harmonic mean.

    The pair is more diagnostic than either alone: over-splitting a strain into
    many pure clusters scores homogeneity 1.0 and completeness far below it,
    while merging two strains does the reverse. Long-read strain methods fail in
    both directions and the report shows both columns.
    """
    h_true, h_pred, mi = _entropies(matrix)
    homogeneity = 1.0 if h_true == 0 else mi / h_true
    completeness = 1.0 if h_pred == 0 else mi / h_pred
    if homogeneity + completeness == 0:
        v_measure = 0.0
    else:
        v_measure = 2 * homogeneity * completeness / (homogeneity + completeness)
    return float(homogeneity), float(completeness), float(v_measure)


def _expected_mutual_information(matrix: np.ndarray) -> float:
    """Expected MI under a hypergeometric null with the observed marginals.

    Standard Vinh, Epps & Bailey (2010) formulation, evaluated in log space.
    """
    n = int(matrix.sum())
    if n == 0:
        return 0.0
    a = matrix.sum(axis=1).astype(np.int64)
    b = matrix.sum(axis=0).astype(np.int64)
    log_n = math.log(n)
    emi = 0.0
    gl = gammaln
    for ai in a:
        if ai == 0:
            continue
        for bj in b:
            if bj == 0:
                continue
            start = max(1, int(ai) + int(bj) - n)
            stop = min(int(ai), int(bj))
            if start > stop:
                continue
            nij = np.arange(start, stop + 1, dtype=np.float64)
            term1 = nij / n * (np.log(nij) + log_n - math.log(ai) - math.log(bj))
            log_hyper = (
                gl(ai + 1)
                + gl(bj + 1)
                + gl(n - ai + 1)
                + gl(n - bj + 1)
                - gl(n + 1)
                - gl(nij + 1)
                - gl(ai - nij + 1)
                - gl(bj - nij + 1)
                - gl(n - ai - bj + nij + 1)
            )
            emi += float((term1 * np.exp(log_hyper)).sum())
    return emi


def adjusted_mutual_information(matrix: np.ndarray) -> float:
    """Chance-corrected mutual information.

    ARI and AMI disagree in a way that is informative here: ARI is dominated by
    the largest clusters, so a tool can score well on ARI while losing every
    rare strain. AMI is less forgiving of that. Both are reported.
    """
    h_true, h_pred, mi = _entropies(matrix)
    emi = _expected_mutual_information(matrix)
    denominator = max(h_true, h_pred) - emi
    if abs(denominator) < 1e-12:
        return 1.0 if mi > 0 else 0.0
    return float((mi - emi) / denominator)


def partition_metrics(
    truth_labels: dict[tuple[str, str], str],
    predicted_labels: dict[tuple[str, str], str],
    sample: str | None = None,
) -> dict:
    """Score one tool's read partition against the truth.

    Parameters
    ----------
    truth_labels
        ``(sample, read_id) -> true strain id`` for every simulated read.
    predicted_labels
        ``(sample, read_id) -> cluster id`` for reads the tool placed. Reads
        absent from this mapping, or mapped to ``UNASSIGNED``, count against
        ``assigned_fraction`` and are excluded from the scores.
    sample
        Restrict to one timepoint. ``None`` pools all timepoints, which is only
        meaningful for tools that carry cluster identity across samples.
    """
    keys = [k for k in truth_labels if sample is None or k[0] == sample]
    n_total = len(keys)
    if n_total == 0:
        return {"n_reads": 0, "assigned_fraction": float("nan")}

    pairs = [
        (truth_labels[k], predicted_labels[k])
        for k in keys
        if predicted_labels.get(k, UNASSIGNED) != UNASSIGNED
    ]
    assigned_fraction = len(pairs) / n_total

    base = {
        "n_reads": n_total,
        "n_reads_assigned": len(pairs),
        "assigned_fraction": assigned_fraction,
    }
    if len(pairs) < 2:
        return {
            **base,
            "ari": float("nan"),
            "ami": float("nan"),
            "homogeneity": float("nan"),
            "completeness": float("nan"),
            "v_measure": float("nan"),
            "n_clusters": 0,
            "n_true_classes": 0,
        }

    labels_true = [t for t, _ in pairs]
    labels_pred = [p for _, p in pairs]
    matrix = contingency(labels_true, labels_pred)
    homogeneity, completeness, v_measure = homogeneity_completeness_v(matrix)

    return {
        **base,
        "ari": adjusted_rand_index(matrix),
        "ami": adjusted_mutual_information(matrix),
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "n_clusters": len(set(labels_pred)),
        "n_true_classes": len(set(labels_true)),
        "largest_cluster_fraction": max(Counter(labels_pred).values()) / len(pairs),
    }
