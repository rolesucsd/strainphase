"""Adapters for strainphase itself, in two configurations.

The two rows exist because the comparison against single-sample tools is only
honest if strainphase is also measured with one hand tied behind its back.

``strainphase-single``
    Each timepoint processed independently, no cross-timepoint rescue. This is
    the row that belongs next to Floria, Strainy and devider: same input, same
    information, same task.

``strainphase-longitudinal``
    All timepoints processed together with rescue enabled. This row is *not*
    comparable to the single-sample tools and the report never presents it as
    though it were. Its comparator is ``strainphase-single``; the difference
    between the two is the entire longitudinal claim, measured against the
    method's own ablation rather than against a tool that never attempted the
    task.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from spbench.adapters.base import Adapter, ToolInfo
from spbench.dataset import Dataset
from spbench.formats import UNASSIGNED, Haplotype, decode_alleles

logger = logging.getLogger(__name__)


def _config_from(overrides: dict | None, threads: int):
    """Build a HaplotyperConfig, seeded by default.

    strainphase's Louvain initialisation and read subsampling are both random.
    Left unseeded, two runs on identical input give different numbers, and a
    benchmark that cannot be re-run to the same answer is not evidence. The
    adapter therefore pins ``random_seed`` unless the config sets one, and the
    value used is recorded in the results.
    """
    from strainphase.core import HaplotyperConfig

    settings = dict(overrides or {})
    settings.setdefault("random_seed", 0)
    config = HaplotyperConfig(**settings)
    if hasattr(config, "n_workers"):
        config.n_workers = threads
    return config


def _assign_reads(
    window_results: list,
    sample: str,
    cluster_of_track,
    confidence_threshold: float,
) -> dict[tuple[str, str], tuple[str, float]]:
    """Best-confidence read assignment across overlapping windows.

    Windows overlap by 50%, so most reads are assigned twice. Taking the
    assignment from the window where the posterior is highest is the reading
    most favourable to strainphase, and it is the same rule the other adapters
    get where their outputs also overlap.
    """
    best: dict[tuple[str, str], tuple[str, float]] = {}
    for result in window_results:
        gamma = result.gamma
        if gamma is None or gamma.size == 0 or not result.haplotypes:
            continue
        n_haps = len(result.haplotypes)
        if gamma.shape[1] < n_haps:
            continue
        hap_gamma = gamma[:, :n_haps]
        best_k = np.argmax(hap_gamma, axis=1)
        best_p = hap_gamma[np.arange(hap_gamma.shape[0]), best_k]

        for i, read in enumerate(result.window.reads):
            if i >= len(best_k):
                break
            confidence = float(best_p[i])
            if confidence < confidence_threshold:
                continue
            track_id = result.haplotypes[int(best_k[i])].track_id
            if not track_id:
                continue
            key = (sample, read.id)
            cluster = cluster_of_track(sample, result.window.contig, track_id)
            if key not in best or confidence > best[key][1]:
                best[key] = (cluster, confidence)
    return best


class StrainphaseSingleAdapter(Adapter):
    info = ToolInfo(
        name="strainphase-single",
        designed_for=(
            "Ablation of strainphase with cross-timepoint rescue disabled: "
            "graph initialisation plus quality-weighted EM within one sample."
        ),
        supports_cross_sample_ids=False,
        multi_sample=False,
    )

    def __init__(self, config: dict | None = None, confidence_threshold: float = 0.9) -> None:
        self.config_overrides = config
        self.confidence_threshold = confidence_threshold
        self._native: list[Haplotype] = []

    def available(self) -> tuple[bool, str]:
        try:
            import strainphase  # noqa: F401
        except ImportError as exc:
            return False, f"strainphase not importable: {exc}"
        return True, ""

    def partition(
        self, dataset: Dataset, workdir: Path, threads: int
    ) -> dict[tuple[str, str], str]:
        from strainphase.core import process_contig

        config = _config_from(self.config_overrides, threads)
        assignments: dict[tuple[str, str], str] = {}

        for sample in dataset.samples:
            for contig, length in dataset.contigs.items():
                results = process_contig(
                    bam_path=str(dataset.bams[sample]),
                    vcf_path=str(dataset.vcfs[sample]),
                    contig_id=contig,
                    contig_length=length,
                    config=config,
                    sample_id=sample,
                )
                if not results:
                    continue
                best = _assign_reads(
                    results,
                    sample,
                    lambda samp, ctg, track: f"{samp}:{ctg}:{track}",
                    self.confidence_threshold,
                )
                for key, (cluster, _) in best.items():
                    assignments[key] = cluster

                # Reads the model saw but would not commit to are recorded as
                # explicitly unassigned, so assigned_fraction reflects reality.
                for result in results:
                    for read in result.window.reads:
                        assignments.setdefault((sample, read.id), UNASSIGNED)

        return assignments


class StrainphaseLongitudinalAdapter(Adapter):
    info = ToolInfo(
        name="strainphase-longitudinal",
        designed_for=(
            "Longitudinal strain reconstruction: all timepoints jointly, with "
            "cross-timepoint rescue of strains that fall below the "
            "single-sample detection floor, and stable lineage identity across "
            "samples."
        ),
        supports_cross_sample_ids=True,
        multi_sample=True,
    )

    def __init__(self, config: dict | None = None, confidence_threshold: float = 0.9) -> None:
        self.config_overrides = config
        self.confidence_threshold = confidence_threshold
        self._native: list[Haplotype] = []

    def available(self) -> tuple[bool, str]:
        try:
            import strainphase  # noqa: F401
        except ImportError as exc:
            return False, f"strainphase not importable: {exc}"
        return True, ""

    def partition(
        self, dataset: Dataset, workdir: Path, threads: int
    ) -> dict[tuple[str, str], str]:
        from strainphase.longitudinal import build_lineage_table, process_mag_longitudinal

        config = _config_from(self.config_overrides, threads)

        all_results, _integrator = process_mag_longitudinal(
            mag_name=dataset.name,
            mag_contigs=dict(dataset.contigs),
            samples=list(dataset.samples),
            bam_paths={s: str(p) for s, p in dataset.bams.items()},
            vcf_paths={s: str(p) for s, p in dataset.vcfs.items()},
            config=config,
        )

        lineage_records, _hap_records = build_lineage_table({dataset.name: all_results}, config)
        lineage_of: dict[tuple[str, str, str], str] = {
            (rec["sample"], rec["contig"], rec["track_id"]): rec["lineage_id"]
            for rec in lineage_records
            if rec.get("track_id")
        }
        self._native = self._native_haplotypes(lineage_records)

        def cluster_of_track(sample: str, contig: str, track_id: str) -> str:
            # Fall back to the per-sample track when a track never made it into
            # a lineage. Silently dropping those reads would inflate the
            # partition scores by discarding the hardest cases.
            return lineage_of.get((sample, contig, track_id), f"{sample}:{contig}:{track_id}")

        assignments: dict[tuple[str, str], str] = {}
        for sample, contig_results in all_results.items():
            for _contig, results in contig_results.items():
                best = _assign_reads(
                    results, sample, cluster_of_track, self.confidence_threshold
                )
                for key, (cluster, _) in best.items():
                    assignments[key] = cluster
                for result in results:
                    for read in result.window.reads:
                        assignments.setdefault((sample, read.id), UNASSIGNED)

        return assignments

    def native_haplotypes(self, dataset: Dataset, workdir: Path) -> list[Haplotype]:
        return self._native

    @staticmethod
    def _native_haplotypes(lineage_records: list[dict]) -> list[Haplotype]:
        """strainphase's own lineages.tsv output, scored as a separate row.

        Kept apart from the derived-consensus row so the headline cross-tool
        table stays free of any strainphase-specific representation.
        """
        haplotypes = []
        for rec in lineage_records:
            alleles = decode_alleles(rec.get("consensus", ""))
            if not alleles:
                continue
            abundance = rec.get("abundance")
            haplotypes.append(
                Haplotype(
                    hap_id=str(rec["lineage_id"]),
                    sample=str(rec["sample"]),
                    contig=str(rec["contig"]),
                    alleles=alleles,
                    start=int(rec.get("span_start") or 0),
                    end=int(rec.get("span_end") or 0),
                    abundance=float(abundance) if abundance not in (None, "") else None,
                )
            )
        return haplotypes
