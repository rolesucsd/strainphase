"""Read simulation, and running strainphase.

Both live here because both are the parts Snakemake cannot express as a plain
shell rule: read simulation needs the per-strain plan and exact read renaming,
and strainphase's read partition comes from the EM posteriors rather than from a
file it writes.

Everything else in the pipeline — alignment, variant calling, Floria, Strainy —
is a shell command and lives in the Snakemake rules where it belongs.
"""

from __future__ import annotations

import gzip
import logging
import subprocess
from pathlib import Path

import numpy as np

from spbench.formats import READ_ORIGINS_COLUMNS, write_table

logger = logging.getLogger(__name__)

#: Badread is what Strainy's own HiFi benchmark used. Placeholders are filled by
#: name from the plan row plus the read-length settings.
DEFAULT_READS_CMD = (
    "badread simulate --reference {assembly} --quantity {coverage}x "
    "--error_model pacbio2021 --qscore_model pacbio2021 "
    "--identity 30,3 --length {mean_length},{length_sd} --seed {seed}"
)


def simulate_reads(
    plan_path: str | Path,
    sample: str,
    fastq_out: str | Path,
    origins_out: str | Path,
    reads_cmd: str = DEFAULT_READS_CMD,
    mean_length: int = 15_000,
    length_sd: int = 4_000,
    seed: int = 0,
) -> int:
    """Simulate one timepoint's reads, one invocation per strain.

    Reads are renamed ``{sample}|{strain}|{n}`` before being concatenated, so
    every read carries its true origin in its name and ``read_origins.tsv`` is a
    fact rather than an inference. This is what keeps ground truth exact through
    a real aligner.
    """
    from spbench.formats import read_table

    rows = [r for r in read_table(plan_path) if r["sample"] == sample]
    fastq_out = Path(fastq_out)
    fastq_out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash((seed, sample))) % (2**31))

    origins: list[dict] = []
    with gzip.open(fastq_out, "wt") as out:
        for row in rows:
            command = reads_cmd.format(
                assembly=row["assembly"],
                coverage=row["coverage"],
                mean_length=mean_length,
                length_sd=length_sd,
                seed=int(rng.integers(1, 2**31 - 1)),
            )
            logger.info("$ %s", command)
            proc = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=False
            )
            if proc.returncode != 0:
                tail = "\n".join(proc.stderr.splitlines()[-20:])
                raise RuntimeError(f"read simulation failed:\n{command}\n{tail}")

            strain_id = row["strain_id"]
            n = 0
            for i, line in enumerate(proc.stdout.splitlines()):
                if i % 4 == 0:
                    n += 1
                    read_id = f"{sample}|{strain_id}|{n:07d}"
                    out.write(f"@{read_id}\n")
                    origins.append(
                        {
                            "sample": sample,
                            "read_id": read_id,
                            "contig": "",
                            "strain_id": strain_id,
                            "start": "",
                            "end": "",
                        }
                    )
                else:
                    out.write(line + "\n")
            logger.info("  %s / %s: %d reads at %sx", sample, strain_id, n, row["coverage"])

    write_table(origins_out, READ_ORIGINS_COLUMNS, origins)
    return len(origins)


# --------------------------------------------------------------------------- #
# strainphase
# --------------------------------------------------------------------------- #


def _config(overrides: dict | None, threads: int):
    """A stock HaplotyperConfig, seeded.

    strainphase's Louvain initialisation and read subsampling are both random;
    unseeded, two runs on identical input give different numbers, and a
    benchmark that cannot be re-run to the same answer is not evidence. Only
    ``random_seed`` is forced — everything else comes from the config file, and
    the benchmark's own config sets only window size and max reads to match the
    production workflow.
    """
    from strainphase.core import HaplotyperConfig

    settings = dict(overrides or {})
    settings.setdefault("random_seed", 0)
    config = HaplotyperConfig(**settings)
    if hasattr(config, "n_workers"):
        config.n_workers = threads
    return config


def _assign_reads(window_results, sample, cluster_of_track, confidence_threshold):
    """Best-confidence read assignment across overlapping windows.

    Windows overlap 50%, so most reads are assigned twice; taking the window
    where the posterior is highest is the reading most favourable to
    strainphase. Reads the model saw but would not commit to are recorded as
    unassigned rather than dropped, so assigned_fraction reflects reality.
    """
    best: dict[tuple[str, str], tuple[str, float]] = {}
    seen: set[tuple[str, str]] = set()

    for result in window_results:
        gamma = result.gamma
        for read in result.window.reads:
            seen.add((sample, read.id))
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
    return best, seen


def run_strainphase(
    mode: str,
    reference: str | Path,
    bams: dict[str, str],
    vcfs: dict[str, str],
    contigs: dict[str, int],
    name: str = "dataset",
    config_overrides: dict | None = None,
    threads: int = 1,
    confidence_threshold: float = 0.9,
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], float], list]:
    """Run strainphase and return ``(assignments, confidences, native_haplotypes)``.

    ``mode="single"`` processes each timepoint independently with no
    cross-timepoint rescue — the like-for-like row. ``mode="longitudinal"``
    processes them jointly and uses lineage ids as cluster labels, which is the
    only configuration that claims identity across samples.
    """
    from spbench.formats import UNASSIGNED, Haplotype, decode_alleles

    config = _config(config_overrides, threads)
    assignments: dict[tuple[str, str], str] = {}
    confidences: dict[tuple[str, str], float] = {}
    native: list[Haplotype] = []

    if mode == "single":
        from strainphase.core import process_contig

        for sample in sorted(bams):
            for contig, length in contigs.items():
                results = process_contig(
                    bam_path=bams[sample],
                    vcf_path=vcfs[sample],
                    contig_id=contig,
                    contig_length=length,
                    config=config,
                    sample_id=sample,
                )
                if not results:
                    continue
                best, seen = _assign_reads(
                    results,
                    sample,
                    lambda s, c, t: f"{s}:{c}:{t}",
                    confidence_threshold,
                )
                for key, (cluster, confidence) in best.items():
                    assignments[key] = cluster
                    confidences[key] = confidence
                for key in seen:
                    assignments.setdefault(key, UNASSIGNED)

    elif mode == "longitudinal":
        from strainphase.longitudinal import build_lineage_table, process_mag_longitudinal

        all_results, _ = process_mag_longitudinal(
            mag_name=name,
            mag_contigs=dict(contigs),
            samples=sorted(bams),
            bam_paths=dict(bams),
            vcf_paths=dict(vcfs),
            config=config,
        )
        lineage_records, _ = build_lineage_table({name: all_results}, config)
        lineage_of = {
            (r["sample"], r["contig"], r["track_id"]): r["lineage_id"]
            for r in lineage_records
            if r.get("track_id")
        }

        def cluster_of_track(sample: str, contig: str, track_id: str) -> str:
            # Fall back to the per-sample track when a track never entered a
            # lineage; silently dropping those reads would inflate the partition
            # scores by discarding the hardest cases.
            return lineage_of.get((sample, contig, track_id), f"{sample}:{contig}:{track_id}")

        for sample, contig_results in all_results.items():
            for results in contig_results.values():
                best, seen = _assign_reads(
                    results, sample, cluster_of_track, confidence_threshold
                )
                for key, (cluster, confidence) in best.items():
                    assignments[key] = cluster
                    confidences[key] = confidence
                for key in seen:
                    assignments.setdefault(key, UNASSIGNED)

        for record in lineage_records:
            alleles = decode_alleles(record.get("consensus", ""))
            if not alleles:
                continue
            abundance = record.get("abundance")
            native.append(
                Haplotype(
                    hap_id=str(record["lineage_id"]),
                    sample=str(record["sample"]),
                    contig=str(record["contig"]),
                    alleles=alleles,
                    start=int(record.get("span_start") or 0),
                    end=int(record.get("span_end") or 0),
                    abundance=float(abundance) if abundance not in (None, "") else None,
                )
            )
    else:
        raise ValueError(f"mode must be 'single' or 'longitudinal', got {mode!r}")

    return assignments, confidences, native
