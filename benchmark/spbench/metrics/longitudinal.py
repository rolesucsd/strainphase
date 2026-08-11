"""Detection sensitivity across abundance, and identity across timepoints.

This module is where strainphase's actual claim lives, and it is deliberately
split into two parts with very different fairness properties.

**Detection sensitivity by abundance** is a fair comparison for every tool.
Recall is stratified by the true abundance of each strain in each timepoint. A
single-sample method is expected to lose strains as abundance falls; the
question the benchmark asks is whether using the other timepoints recovers them,
and every tool's curve is computed the same way from the same truth table. No
tool is disadvantaged by the metric - only, possibly, by the task.

**Cross-timepoint identity** is not a fair comparison and is not presented as
one. Floria, Strainy and devider phase one sample at a time and never claim that
``HAP3`` in T1 is the same organism as ``HAP3`` in T2. Scoring them on it would
manufacture a win. Tools declare ``supports_cross_sample_ids`` in their adapter;
for those that do not, these columns are ``n/a`` in the report, never 0.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from spbench.formats import Haplotype, Truth
from spbench.metrics.haplotype import match_haplotypes
from spbench.metrics.partition import adjusted_rand_index, contingency

#: Abundance strata. The lowest two bins are where single-sample and
#: multi-sample methods should visibly separate; the top bin is where any
#: working method should score near 1.0 and a low score means something is
#: broken rather than merely hard.
ABUNDANCE_BINS: list[tuple[str, float, float]] = [
    ("lt_1pct", 0.0, 0.01),
    ("1_5pct", 0.01, 0.05),
    ("5_20pct", 0.05, 0.20),
    ("gt_20pct", 0.20, 1.01),
]


def detection_table(
    predicted: list[Haplotype],
    truth: Truth,
    contig: str,
    match_threshold: float = 0.99,
    min_shared_sites: int = 10,
) -> list[dict]:
    """One row per (sample, true strain): was it recovered, and at what abundance?

    This per-strain table is written to disk alongside the summary so that the
    abundance-vs-detection figure can be redrawn without re-running any tool,
    and so a reviewer can check any individual claim.
    """
    rows: list[dict] = []
    by_sample: dict[str, list[Haplotype]] = defaultdict(list)
    for hap in predicted:
        if hap.contig == contig:
            by_sample[hap.sample].append(hap)

    strain_ids = [sid for sid, hap in truth.strains.items() if hap.contig == contig]

    for sample in truth.samples:
        matches = match_haplotypes(by_sample.get(sample, []), truth, contig, min_shared_sites)
        recovered = {
            sid: (frac, n_shared)
            for _, sid, frac, n_shared in matches
            if frac >= match_threshold
        }
        best = {sid: frac for _, sid, frac, _ in matches}
        for sid in strain_ids:
            true_abundance = truth.abundance.get((sample, sid), 0.0)
            rows.append(
                {
                    "sample": sample,
                    "contig": contig,
                    "strain_id": sid,
                    "true_abundance": true_abundance,
                    "abundance_bin": _bin_for(true_abundance),
                    "detected": int(sid in recovered),
                    "best_agreement": best.get(sid, 0.0),
                    "n_shared_sites": recovered.get(sid, (0.0, 0))[1],
                }
            )
    return rows


def _bin_for(abundance: float) -> str:
    for name, low, high in ABUNDANCE_BINS:
        if low <= abundance < high:
            return name
    return ABUNDANCE_BINS[-1][0]


def longitudinal_metrics(
    predicted: list[Haplotype],
    truth: Truth,
    contig: str,
    supports_cross_sample_ids: bool,
    match_threshold: float = 0.99,
    min_shared_sites: int = 10,
) -> tuple[dict, list[dict]]:
    """Summary metrics plus the per-strain detection table backing them."""
    rows = detection_table(predicted, truth, contig, match_threshold, min_shared_sites)

    summary: dict = {"contig": contig}
    for name, _, _ in ABUNDANCE_BINS:
        in_bin = [r for r in rows if r["abundance_bin"] == name and r["true_abundance"] > 0]
        summary[f"recall_{name}"] = (
            float(np.mean([r["detected"] for r in in_bin])) if in_bin else float("nan")
        )
        summary[f"n_{name}"] = len(in_bin)

    present = [r for r in rows if r["true_abundance"] > 0]
    summary["recall_overall"] = (
        float(np.mean([r["detected"] for r in present])) if present else float("nan")
    )

    # Persistence: of the timepoints where a strain is genuinely present, in how
    # many is it found? Averaged over strains, so a strain seen at every
    # timepoint and one seen at none weigh equally.
    per_strain: dict[str, list[int]] = defaultdict(list)
    for row in present:
        per_strain[row["strain_id"]].append(row["detected"])
    summary["mean_persistence_recall"] = (
        float(np.mean([np.mean(v) for v in per_strain.values()])) if per_strain else float("nan")
    )
    summary["n_strains_never_detected"] = sum(
        1 for values in per_strain.values() if not any(values)
    )
    summary["n_strains_present"] = len(per_strain)

    summary.update(
        _cross_sample_metrics(
            predicted, truth, contig, supports_cross_sample_ids, match_threshold, min_shared_sites
        )
    )
    return summary, rows


def _cross_sample_metrics(
    predicted: list[Haplotype],
    truth: Truth,
    contig: str,
    supports_cross_sample_ids: bool,
    match_threshold: float,
    min_shared_sites: int,
) -> dict:
    """Does the tool give the same organism the same name at every timepoint?

    Reported as ARI between the tool's own grouping of its haplotypes (by
    ``hap_id``) and the true strain each of those haplotypes matched. A tool
    that emits fresh per-sample identifiers scores near 0 by construction, which
    is why this is gated on the adapter's declaration rather than computed for
    everyone.
    """
    if not supports_cross_sample_ids:
        return {
            "cross_sample_id_ari": float("nan"),
            "cross_sample_id_supported": False,
            "n_cross_sample_haps": 0,
        }

    by_sample: dict[str, list[Haplotype]] = defaultdict(list)
    for hap in predicted:
        if hap.contig == contig:
            by_sample[hap.sample].append(hap)

    labels_true: list[str] = []
    labels_pred: list[str] = []
    for sample in truth.samples:
        for hap, sid, frac, _ in match_haplotypes(
            by_sample.get(sample, []), truth, contig, min_shared_sites
        ):
            if frac >= match_threshold:
                labels_true.append(sid)
                labels_pred.append(hap.hap_id)

    if len(labels_true) < 2:
        return {
            "cross_sample_id_ari": float("nan"),
            "cross_sample_id_supported": True,
            "n_cross_sample_haps": len(labels_true),
        }

    return {
        "cross_sample_id_ari": adjusted_rand_index(contingency(labels_true, labels_pred)),
        "cross_sample_id_supported": True,
        "n_cross_sample_haps": len(labels_true),
    }
