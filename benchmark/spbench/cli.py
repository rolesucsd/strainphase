"""Command-line entry points, one per Snakemake rule that needs Python.

Everything that is a shell command — alignment, variant calling, Floria,
Strainy — lives in the Snakemake rules, not here. This module holds only the
steps Snakemake cannot express: building ground truth, simulating reads with
exact provenance, running strainphase (whose partition comes from the EM
posteriors rather than a file), parsing each tool's native output, scoring, and
writing the report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from spbench import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spbench",
        description="Scoring library for the strainphase benchmark workflow",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--version", action="version", version=f"spbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("truth", help="Assemblies -> reference, ground truth, read plan")
    p.add_argument("--assemblies", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timepoints", type=int, default=6)
    p.add_argument("--coverage", type=float, default=60.0)
    p.add_argument("--asm-preset", default="asm5")

    p = sub.add_parser("simulate-reads", help="Simulate one timepoint's reads")
    p.add_argument("--plan", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--fastq", required=True)
    p.add_argument("--origins", required=True)
    p.add_argument("--reads-cmd", default=None)
    p.add_argument("--mean-length", type=int, default=15_000)
    p.add_argument("--length-sd", type=int, default=4_000)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("run-strainphase", help="Run strainphase and emit its read partition")
    p.add_argument("--mode", choices=["single", "longitudinal"], required=True)
    p.add_argument("--dataset", required=True, help="Dataset directory")
    p.add_argument("--out", required=True, help="read_assignments.tsv")
    p.add_argument("--native", help="Also write strainphase's own haplotypes here")
    p.add_argument("--config", help="JSON dict of HaplotyperConfig overrides")
    p.add_argument("--threads", type=int, default=1)

    p = sub.add_parser("parse", help="A tool's native output -> a read partition")
    p.add_argument("--tool", required=True, choices=["floria", "strainy"])
    p.add_argument("--indir", required=True, help="Tool output directory")
    p.add_argument("--sample", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("merge-partitions", help="Concatenate per-sample partitions")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("score", help="Score one tool on one dataset")
    p.add_argument("--tool", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--native")
    p.add_argument("--match-threshold", type=float, default=0.99)
    p.add_argument("--min-shared-sites", type=int, default=10)

    p = sub.add_parser("report", help="Combine scored results into the report")
    p.add_argument("--inputs", nargs="+", required=True, help="Per-(dataset,tool) result dirs")
    p.add_argument("--outdir", required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return _DISPATCH[args.command](args)


def _cmd_truth(args) -> int:
    from spbench.truth import build

    build(
        assemblies=args.assemblies,
        outdir=args.outdir,
        seed=args.seed,
        n_timepoints=args.timepoints,
        coverage=args.coverage,
        asm_preset=args.asm_preset,
    )
    return 0


def _cmd_simulate_reads(args) -> int:
    from spbench.simulate import DEFAULT_READS_CMD, simulate_reads

    n = simulate_reads(
        plan_path=args.plan,
        sample=args.sample,
        fastq_out=args.fastq,
        origins_out=args.origins,
        reads_cmd=args.reads_cmd or DEFAULT_READS_CMD,
        mean_length=args.mean_length,
        length_sd=args.length_sd,
        seed=args.seed,
    )
    print(f"{args.sample}: {n} reads")
    return 0


def _cmd_run_strainphase(args) -> int:
    from spbench.formats import write_table
    from spbench.parsers import write_partition
    from spbench.simulate import run_strainphase

    dataset = Path(args.dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    samples = list(manifest["samples"])

    assignments, confidences, native = run_strainphase(
        mode=args.mode,
        reference=dataset / "reference.fasta",
        bams={s: str(dataset / "bam" / f"{s}.bam") for s in samples},
        vcfs={s: str(dataset / "variants" / f"{s}.vcf.gz") for s in samples},
        contigs=dict(manifest["contigs"]),
        name=manifest["name"],
        config_overrides=json.loads(args.config) if args.config else None,
        threads=args.threads,
    )
    write_partition(args.out, assignments, confidences)

    if args.native:
        write_table(
            args.native,
            ["sample", "contig", "hap_id", "start", "end", "abundance", "n_sites", "alleles"],
            (hap.to_row() for hap in native),
        )
    print(f"{len(assignments)} reads, {len(native)} native haplotypes")
    return 0


def _cmd_parse(args) -> int:
    from spbench.parsers import PARSERS, write_partition

    assignments = PARSERS[args.tool](args.indir, args.sample)
    write_partition(args.out, assignments)
    print(f"{args.tool} / {args.sample}: {len(assignments)} reads")
    return 0


def _cmd_merge_partitions(args) -> int:
    """Concatenate per-sample partitions into one file for scoring.

    Single-sample tools are run once per timepoint, so their partitions arrive
    in pieces; scoring wants the whole timecourse at once.
    """
    from spbench.formats import read_table, write_table
    from spbench.parsers import READ_ASSIGNMENT_COLUMNS

    rows = [row for path in args.inputs for row in read_table(path)]
    write_table(args.out, READ_ASSIGNMENT_COLUMNS, rows)
    print(f"{len(rows)} assignments from {len(args.inputs)} files")
    return 0


def _cmd_score(args) -> int:
    from spbench.score import score

    score(
        tool=args.tool,
        partition_path=args.partition,
        dataset_dir=args.dataset,
        outdir=args.outdir,
        native_path=args.native,
        match_threshold=args.match_threshold,
        min_shared_sites=args.min_shared_sites,
    )
    return 0


def _cmd_report(args) -> int:
    from spbench.formats import read_table, write_table
    from spbench.report import write_figures, write_report
    from spbench.score import TABLES

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        rows = []
        for indir in args.inputs:
            path = Path(indir) / f"{name}.tsv"
            if path.exists():
                rows.extend(read_table(path))
        if not rows:
            continue
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        write_table(outdir / f"{name}.tsv", columns, rows)

    _write_provenance(outdir, args.inputs)
    report = write_report(outdir)
    write_figures(outdir)
    print(report)
    return 0


def _write_provenance(outdir: Path, inputs: list[str]) -> None:
    """Record what produced these numbers, in enough detail to rerun them."""
    import platform
    import subprocess
    from datetime import datetime, timezone

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=Path(__file__).resolve().parent,
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance is best-effort
            return "unknown"

    versions = {}
    for module in ("strainphase", "numpy", "scipy", "pysam", "mappy", "pandas"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001
            versions[module] = "not installed"

    (outdir / "provenance.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "spbench_version": __version__,
                "git_commit": git("rev-parse", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain")),
                "python": sys.version,
                "platform": platform.platform(),
                "package_versions": versions,
                "n_result_dirs": len(inputs),
            },
            indent=2,
        )
        + "\n"
    )


_DISPATCH = {
    "truth": _cmd_truth,
    "simulate-reads": _cmd_simulate_reads,
    "run-strainphase": _cmd_run_strainphase,
    "parse": _cmd_parse,
    "merge-partitions": _cmd_merge_partitions,
    "score": _cmd_score,
    "report": _cmd_report,
}


if __name__ == "__main__":
    sys.exit(main())
