"""Score one tool on one dataset.

Reads only the common format: a read partition, the BAMs and VCFs the tool was
given, and the truth tables. It has no idea which tool produced the partition,
which is what lets the report claim every tool was scored the same way.

Two steps, in this order:

1. **Consensus, derived here rather than taken from the tool.** Floria emits
   vartigs, Strainy emits assembly graph paths, strainphase emits window-linked
   tracks. Scoring those native representations against each other would compare
   three consensus callers as much as three phasing algorithms. So the partition
   is turned into haplotypes by one function, identically for everyone.
2. **Metrics**, from :mod:`spbench.metrics`.

A tool that also emits a native haplotype set (strainphase does) is scored on it
separately, tagged ``representation=native``, and kept out of the headline table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from spbench.formats import Haplotype, Truth, decode_alleles, read_table, write_table
from spbench.metrics import haplotype_metrics, longitudinal_metrics, partition_metrics
from spbench.reads import consensus_from_partition, load_sites, read_alleles
from spbench.tools import TOOLS, ToolInfo

logger = logging.getLogger(__name__)

TABLES = ("per_sample", "longitudinal", "detection", "runs")


def load_partition(path: str | Path) -> tuple[dict, dict]:
    """``read_assignments.tsv`` -> ``(assignments, confidences)``."""
    assignments: dict[tuple[str, str], str] = {}
    confidences: dict[tuple[str, str], float] = {}
    for row in read_table(path):
        key = (row["sample"], row["read_id"])
        assignments[key] = row["hap_id"]
        conf = (row.get("confidence") or "").strip()
        if conf:
            confidences[key] = float(conf)
    return assignments, confidences


def derive_haplotypes(
    assignments: dict[tuple[str, str], str],
    bams: dict[str, str],
    vcfs: dict[str, str],
    contigs: list[str],
    tool: str,
) -> list[Haplotype]:
    """Consensus per cluster, by the same code for every tool."""
    haplotypes: list[Haplotype] = []
    for sample, bam in sorted(bams.items()):
        per_sample = {
            read_id: cluster for (samp, read_id), cluster in assignments.items() if samp == sample
        }
        if not per_sample:
            continue
        for contig in contigs:
            sites = load_sites(vcfs[sample], contig)
            if not sites:
                continue
            alleles = read_alleles(bam, contig, sites)
            haplotypes.extend(
                consensus_from_partition(per_sample, alleles, sample, contig, tool)
            )
    return haplotypes


def score(
    tool: str,
    partition_path: str | Path,
    dataset_dir: str | Path,
    outdir: str | Path,
    native_path: str | Path | None = None,
    match_threshold: float = 0.99,
    min_shared_sites: int = 10,
    wall_seconds: float | None = None,
    peak_rss_mb: float | None = None,
) -> Path:
    """Score one (dataset, tool) pair. Writes the four tidy tables."""
    dataset_dir = Path(dataset_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    samples = list(manifest["samples"])
    contigs = list(manifest["contigs"])
    bams = {s: str(dataset_dir / "bam" / f"{s}.bam") for s in samples}
    vcfs = {s: str(dataset_dir / "variants" / f"{s}.vcf.gz") for s in samples}

    truth = Truth.read(dataset_dir / "truth")
    info = TOOLS.get(tool, ToolInfo(name=tool, designed_for=""))

    assignments, _confidences = load_partition(partition_path)
    haplotypes = derive_haplotypes(assignments, bams, vcfs, contigs, tool)
    write_table(
        outdir / "haplotypes.tsv",
        ["sample", "contig", "hap_id", "start", "end", "abundance", "n_sites", "alleles"],
        (hap.to_row() for hap in haplotypes),
    )

    stamp = {
        "dataset": manifest["name"],
        "tool": tool,
        "multi_sample": info.multi_sample,
        "supports_cross_sample_ids": info.supports_cross_sample_ids,
        "status": "ok",
        "seed": manifest["settings"].get("seed"),
        "n_strains": len(manifest.get("strains", [])),
        "reference_strain": manifest.get("reference_strain", ""),
        "coverage": manifest["settings"].get("coverage"),
        "n_timepoints": manifest["settings"].get("n_timepoints"),
        "n_variant_sites": manifest.get("n_variant_sites"),
    }

    tables = _score_representation(
        haplotypes, assignments, truth, samples, contigs, info, stamp,
        "derived", match_threshold, min_shared_sites,
    )

    if native_path and Path(native_path).exists():
        native = [
            Haplotype(
                hap_id=row["hap_id"],
                sample=row["sample"],
                contig=row["contig"],
                alleles=decode_alleles(row.get("alleles", "")),
                start=int(row.get("start") or 0),
                end=int(row.get("end") or 0),
                abundance=float(row["abundance"]) if (row.get("abundance") or "") else None,
            )
            for row in read_table(native_path)
        ]
        if native:
            for name, rows in _score_representation(
                native, {}, truth, samples, contigs, info, stamp,
                "native", match_threshold, min_shared_sites,
            ).items():
                tables[name].extend(rows)

    tables["runs"][0].update(
        {"wall_seconds": wall_seconds, "peak_rss_mb": peak_rss_mb,
         "designed_for": info.designed_for, "citation": info.citation}
    )

    for name, rows in tables.items():
        if not rows:
            continue
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        write_table(outdir / f"{name}.tsv", columns, rows)

    logger.info(
        "%s on %s: %d haplotypes from %d reads",
        tool, manifest["name"], len(haplotypes), len(assignments),
    )
    return outdir


def _score_representation(
    haplotypes, assignments, truth, samples, contigs, info, stamp,
    representation, match_threshold, min_shared_sites,
) -> dict[str, list[dict]]:
    common = {**stamp, "representation": representation}
    tables: dict[str, list[dict]] = {name: [] for name in TABLES}

    for sample in samples:
        # The read partition belongs to the tool, not to a representation of its
        # haplotypes, so it is scored once on the derived rows only.
        partition = (
            partition_metrics(truth.read_origins, assignments, sample)
            if representation == "derived"
            else {}
        )
        for contig in contigs:
            tables["per_sample"].append({
                **common,
                "sample": sample,
                **partition,
                **haplotype_metrics(
                    haplotypes, truth, sample, contig,
                    match_threshold=match_threshold,
                    min_shared_sites=min_shared_sites,
                ),
            })

    for contig in contigs:
        summary, rows = longitudinal_metrics(
            haplotypes, truth, contig,
            supports_cross_sample_ids=info.supports_cross_sample_ids,
            match_threshold=match_threshold,
            min_shared_sites=min_shared_sites,
        )
        tables["longitudinal"].append({**common, **summary})
        tables["detection"].extend({**common, **row} for row in rows)

    # Pooling reads across timepoints only means something for a tool claiming
    # one identity per organism. For the others it would score a promise never
    # made.
    pooled = (
        partition_metrics(truth.read_origins, assignments, None)
        if info.supports_cross_sample_ids and representation == "derived"
        else {}
    )
    tables["runs"].append({
        **common,
        "n_haplotypes": len(haplotypes),
        **{f"pooled_{k}": v for k, v in pooled.items()},
    })
    return tables
