"""Scoring a prediction against the truth.

Nothing here is tool-aware. It receives a :class:`~spbench.formats.Prediction`
and a :class:`~spbench.formats.Truth` and emits tidy rows, so the same code path
scores every tool and every representation.
"""

from __future__ import annotations

import logging

from spbench.adapters.base import ToolInfo
from spbench.dataset import Dataset
from spbench.formats import Haplotype, Prediction, Truth
from spbench.metrics import haplotype_metrics, longitudinal_metrics, partition_metrics

logger = logging.getLogger(__name__)


def evaluate(
    prediction: Prediction,
    truth: Truth,
    dataset: Dataset,
    info: ToolInfo,
    representation: str = "derived",
    match_threshold: float = 0.99,
    min_shared_sites: int = 10,
) -> dict[str, list[dict]]:
    """Score one tool on one dataset.

    Returns a dict of table name -> rows, ready to be concatenated across runs.
    """
    common = {
        "dataset": dataset.name,
        "tool": info.name,
        "representation": representation,
        "multi_sample": info.multi_sample,
        "supports_cross_sample_ids": info.supports_cross_sample_ids,
        "status": prediction.status,
    }

    if prediction.status != "ok":
        return {
            "per_sample": [],
            "longitudinal": [],
            "detection": [],
            "runs": [{**common, "message": prediction.message}],
        }

    haplotypes = prediction.haplotypes

    per_sample: list[dict] = []
    for sample in dataset.samples:
        partition = (
            partition_metrics(truth.read_origins, prediction.read_assignments, sample)
            if representation == "derived"
            else {}
        )
        for contig in dataset.contigs:
            row = {
                **common,
                "sample": sample,
                **partition,
                **haplotype_metrics(
                    haplotypes,
                    truth,
                    sample,
                    contig,
                    match_threshold=match_threshold,
                    min_shared_sites=min_shared_sites,
                ),
            }
            per_sample.append(row)

    longitudinal: list[dict] = []
    detection: list[dict] = []
    for contig in dataset.contigs:
        summary, rows = longitudinal_metrics(
            haplotypes,
            truth,
            contig,
            supports_cross_sample_ids=info.supports_cross_sample_ids,
            match_threshold=match_threshold,
            min_shared_sites=min_shared_sites,
        )
        longitudinal.append({**common, **summary})
        detection.extend({**common, **row} for row in rows)

    # Pooling reads across all timepoints only means something for a tool that
    # claims one identity per organism across samples. For the others it would
    # score them on a promise they never made.
    pooled = (
        partition_metrics(truth.read_origins, prediction.read_assignments, None)
        if info.supports_cross_sample_ids and representation == "derived"
        else {}
    )

    runs = [
        {
            **common,
            "message": prediction.message,
            "wall_seconds": prediction.wall_seconds,
            "peak_rss_mb": prediction.peak_rss_mb,
            "n_haplotypes": len(haplotypes),
            "n_reads_assigned": len(
                [v for v in prediction.read_assignments.values() if v != "UNASSIGNED"]
            ),
            "designed_for": info.designed_for,
            "citation": info.citation,
            **{f"pooled_{k}": v for k, v in pooled.items()},
        }
    ]

    return {
        "per_sample": per_sample,
        "longitudinal": longitudinal,
        "detection": detection,
        "runs": runs,
    }


def evaluate_native(
    native: list[Haplotype],
    truth: Truth,
    dataset: Dataset,
    info: ToolInfo,
    **kwargs,
) -> dict[str, list[dict]]:
    """Score a tool's own haplotype output, as a clearly separate row.

    Only strainphase currently supplies one. It is reported so the paper can
    quote its real output, and kept out of the headline table so no reviewer has
    to wonder whether the comparison used a strainphase-shaped yardstick.
    """
    stand_in = Prediction(tool=info.name, haplotypes=native, status="ok")
    return evaluate(stand_in, truth, dataset, info, representation="native", **kwargs)
