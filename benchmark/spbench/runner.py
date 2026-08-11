"""Top-level orchestration: simulate, run every tool, score, write tidy tables.

The three stages are separable on purpose. ``simulate`` is deterministic and
cacheable, so re-running a single tool never re-rolls the data underneath it;
``run`` writes each tool's prediction to disk in the common format, so scoring
can be redone with a changed threshold without re-running anything; ``evaluate``
reads only those files. On a cluster the middle stage is the one worth
parallelising, and it can be, because each (dataset, tool) pair is independent.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from spbench import __version__
from spbench.adapters import build
from spbench.config import BenchmarkConfig
from spbench.dataset import Dataset
from spbench.evaluate import evaluate, evaluate_native
from spbench.formats import Prediction, Truth, write_table
from spbench.simulate import simulate

logger = logging.getLogger(__name__)

TABLES = ("per_sample", "longitudinal", "detection", "runs")


def simulate_all(config: BenchmarkConfig, workdir: Path, force: bool = False) -> list[Path]:
    """Simulate every (dataset, seed). Existing datasets with a matching config
    fingerprint are reused."""
    roots = []
    for sim_config in config.expand():
        root = workdir / "datasets" / sim_config.name
        manifest = root / "manifest.json"
        if not force and manifest.exists():
            existing = json.loads(manifest.read_text()).get("config_fingerprint")
            if existing == sim_config.fingerprint():
                logger.info("reusing %s", root)
                roots.append(root)
                continue
            logger.info("config changed for %s; regenerating", sim_config.name)
        simulate(sim_config, root, threads=config.threads)
        roots.append(root)
    return roots


def run_tools(
    config: BenchmarkConfig, workdir: Path, roots: list[Path], only: list[str] | None = None
) -> None:
    """Run each tool on each dataset and write predictions in common format."""
    for root in roots:
        dataset = Dataset.load(root)
        for spec in config.tools:
            if only and spec.name not in only:
                continue
            outdir = workdir / "predictions" / dataset.name / spec.name
            adapter = build(spec.name, **spec.options)
            logger.info("=== %s on %s ===", spec.name, dataset.name)
            prediction = adapter.run(dataset, workdir / "work" / dataset.name / spec.name,
                                     threads=config.threads)
            prediction.write(outdir)
            (outdir / "status.json").write_text(
                json.dumps(
                    {
                        "tool": spec.name,
                        "options": spec.options,
                        "status": prediction.status,
                        "message": prediction.message,
                        "wall_seconds": prediction.wall_seconds,
                        "peak_rss_mb": prediction.peak_rss_mb,
                        "designed_for": adapter.info.designed_for,
                        "citation": adapter.info.citation,
                        "multi_sample": adapter.info.multi_sample,
                        "supports_cross_sample_ids": adapter.info.supports_cross_sample_ids,
                    },
                    indent=2,
                )
                + "\n"
            )
            native = adapter.native_haplotypes(dataset, outdir)
            if native:
                Prediction(tool=spec.name, haplotypes=native).write(outdir / "native")


def evaluate_all(config: BenchmarkConfig, workdir: Path, roots: list[Path]) -> dict[str, list[dict]]:
    """Score every written prediction. Reads only the common format."""
    tables: dict[str, list[dict]] = {name: [] for name in TABLES}

    for root in roots:
        dataset = Dataset.load(root)
        truth = Truth.read(dataset.truth_dir)
        for spec in config.tools:
            outdir = workdir / "predictions" / dataset.name / spec.name
            status_path = outdir / "status.json"
            if not status_path.exists():
                logger.warning("no prediction for %s on %s", spec.name, dataset.name)
                continue
            status = json.loads(status_path.read_text())
            info = _info_from_status(spec.name, status)

            prediction = Prediction.read(outdir, spec.name)
            prediction.status = status["status"]
            prediction.message = status.get("message", "")
            prediction.wall_seconds = status.get("wall_seconds")
            prediction.peak_rss_mb = status.get("peak_rss_mb")

            result = evaluate(
                prediction,
                truth,
                dataset,
                info,
                match_threshold=config.match_threshold,
                min_shared_sites=config.min_shared_sites,
            )
            for name in TABLES:
                tables[name].extend(_stamp(result[name], dataset, spec))

            native_dir = outdir / "native"
            if native_dir.exists() and prediction.status == "ok":
                native = Prediction.read(native_dir, spec.name).haplotypes
                if native:
                    native_result = evaluate_native(
                        native,
                        truth,
                        dataset,
                        info,
                        match_threshold=config.match_threshold,
                        min_shared_sites=config.min_shared_sites,
                    )
                    for name in TABLES:
                        tables[name].extend(_stamp(native_result[name], dataset, spec))

    return tables


def _info_from_status(name: str, status: dict):
    from spbench.adapters.base import ToolInfo

    return ToolInfo(
        name=name,
        designed_for=status.get("designed_for", ""),
        citation=status.get("citation", ""),
        supports_cross_sample_ids=bool(status.get("supports_cross_sample_ids")),
        multi_sample=bool(status.get("multi_sample")),
    )


def _stamp(rows: list[dict], dataset: Dataset, spec) -> list[dict]:
    """Attach the dataset's own parameters to every row.

    Every number the report makes carries its own provenance: strain count,
    coverage, seed and the tool options it was produced under. That is what lets
    the results table be sliced without cross-referencing a config file.
    """
    sim = dataset.manifest.get("config", {})
    extra = {
        "seed": sim.get("seed"),
        "n_strains": len(dataset.manifest.get("strains", [])),
        "reference_strain": dataset.manifest.get("reference_strain", ""),
        "coverage": sim.get("coverage"),
        "n_timepoints": sim.get("n_timepoints"),
        "mean_read_length": sim.get("mean_read_length"),
        "n_variant_sites": dataset.manifest.get("n_variant_sites"),
        "tool_options": json.dumps(spec.options, sort_keys=True) if spec.options else "",
    }
    return [{**row, **extra} for row in rows]


def write_tables(tables: dict[str, list[dict]], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if not rows:
            continue
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        write_table(outdir / f"{name}.tsv", columns, rows)
        logger.info("wrote %s (%d rows)", outdir / f"{name}.tsv", len(rows))


def write_provenance(config: BenchmarkConfig, outdir: Path) -> None:
    """Record what produced these numbers, in enough detail to rerun them."""
    outdir.mkdir(parents=True, exist_ok=True)

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance is best-effort
            return "unknown"

    versions = {}
    for module in ("strainphase", "numpy", "scipy", "pysam", "networkx", "pandas"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001
            versions[module] = "not installed"

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "spbench_version": __version__,
        "config": config.name,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": _cpu_count(),
        "package_versions": versions,
        "tools": [{"name": t.name, "options": t.options} for t in config.tools],
        "seeds": config.seeds,
        "match_threshold": config.match_threshold,
        "min_shared_sites": config.min_shared_sites,
    }
    (outdir / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")


def _cpu_count() -> int:
    try:
        return len(__import__("os").sched_getaffinity(0))
    except AttributeError:
        return __import__("os").cpu_count() or 1


# --------------------------------------------------------------------------- #
# Work units, for cluster execution
# --------------------------------------------------------------------------- #
#
# A (dataset, tool) pair is the natural unit of parallelism: independent, and
# the only stage that costs real time. Enumerating them lets a SLURM array map
# one task per pair without the harness knowing anything about SLURM.
#
# The enumeration is deterministic - sorted by dataset then by the tool's order
# in the config - so an array index means the same thing on resubmission. A task
# that failed can be rerun by index alone.


def plan_units(config: BenchmarkConfig) -> list[dict]:
    """Every (dataset, tool) pair, in a stable order."""
    units = []
    for sim_config in config.expand():
        for spec in config.tools:
            units.append(
                {
                    "index": len(units),
                    "dataset": sim_config.name,
                    "tool": spec.name,
                    "options": spec.options,
                }
            )
    return units


def simulate_only(config: BenchmarkConfig, workdir: Path, force: bool = False) -> list[Path]:
    """Simulation as a standalone stage.

    On a cluster this must run to completion before the array starts. Letting
    array tasks simulate on demand would have several of them writing the same
    dataset directory at once; the fix is a dependency, not a lock.
    """
    return simulate_all(config, workdir, force=force)


def run_unit(config: BenchmarkConfig, workdir: Path, index: int) -> Path:
    """Run exactly one (dataset, tool) pair. The body of a SLURM array task."""
    units = plan_units(config)
    if not 0 <= index < len(units):
        raise IndexError(f"unit index {index} out of range (0..{len(units) - 1})")
    unit = units[index]

    root = Path(workdir) / "datasets" / unit["dataset"]
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(
            f"dataset {unit['dataset']} has not been simulated yet at {root}. "
            f"Run `spbench simulate` first - on a cluster, as a job the array "
            f"depends on."
        )

    spec = next(t for t in config.tools if t.name == unit["tool"])
    run_tools(config, Path(workdir), [root], only=[spec.name])
    return Path(workdir) / "predictions" / unit["dataset"] / spec.name


def run_all(
    config_path: str | Path,
    workdir: str | Path,
    force_simulate: bool = False,
    only_tools: list[str] | None = None,
    skip_run: bool = False,
) -> Path:
    config = BenchmarkConfig.load(config_path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    roots = simulate_all(config, workdir, force=force_simulate)
    if not skip_run:
        run_tools(config, workdir, roots, only=only_tools)
    tables = evaluate_all(config, workdir, roots)

    results = workdir / "results"
    write_tables(tables, results)
    write_provenance(config, results)
    return results


