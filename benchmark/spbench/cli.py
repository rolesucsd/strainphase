"""Command-line entry point: ``spbench <subcommand>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from spbench import __version__


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", required=True, help="Benchmark config YAML")
    parser.add_argument("--workdir", "-w", default="results", help="Output directory")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="spbench",
        description="Reproducible benchmark of long-read strain phasing tools",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--version", action="version", version=f"spbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("run", help="Simulate, run every tool, score, and report")
    _add_common(p_all)
    p_all.add_argument("--force-simulate", action="store_true", help="Regenerate datasets")
    p_all.add_argument("--only", nargs="+", help="Run only these tools")
    p_all.add_argument("--skip-run", action="store_true", help="Re-score existing predictions")

    p_sim = sub.add_parser("simulate", help="Generate datasets only")
    _add_common(p_sim)
    p_sim.add_argument("--force", action="store_true")

    p_eval = sub.add_parser("evaluate", help="Score existing predictions and write the report")
    _add_common(p_eval)

    p_plan = sub.add_parser(
        "plan", help="List the (dataset, tool) work units, for cluster submission"
    )
    p_plan.add_argument("--config", "-c", required=True)
    p_plan.add_argument(
        "--count", action="store_true", help="Print only the number of units"
    )

    p_one = sub.add_parser("run-one", help="Run a single work unit by index (SLURM array task)")
    _add_common(p_one)
    p_one.add_argument("--index", "-i", type=int, required=True)

    p_report = sub.add_parser("report", help="Rebuild the report from an existing results dir")
    p_report.add_argument("--results", "-r", default="results/results")
    p_report.add_argument("--output", "-o")

    p_check = sub.add_parser(
        "check-tools", help="Report which tools are installed and runnable here"
    )
    p_check.add_argument("--config", "-c", help="Only check tools named in this config")

    p_verify = sub.add_parser(
        "verify", help="Compare results against a stored expectations file"
    )
    p_verify.add_argument("--results", "-r", default="results/results")
    p_verify.add_argument("--expected", "-e", required=True)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "simulate":
        return _cmd_simulate(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "run-one":
        return _cmd_run_one(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "check-tools":
        return _cmd_check_tools(args)
    if args.command == "verify":
        return _cmd_verify(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _cmd_run(args) -> int:
    from spbench.report import write_figures, write_report
    from spbench.runner import run_all

    results = run_all(
        args.config,
        args.workdir,
        force_simulate=args.force_simulate,
        only_tools=args.only,
        skip_run=args.skip_run,
    )
    report = write_report(results)
    figures = write_figures(results)
    print(f"\nResults: {results}")
    print(f"Report:  {report}")
    for figure in figures:
        print(f"Figure:  {figure}")
    return 0


def _cmd_simulate(args) -> int:
    from spbench.config import BenchmarkConfig
    from spbench.runner import simulate_all

    config = BenchmarkConfig.load(args.config)
    roots = simulate_all(config, Path(args.workdir), force=args.force)
    for root in roots:
        print(root)
    return 0


def _cmd_evaluate(args) -> int:
    from spbench.report import write_figures, write_report
    from spbench.runner import run_all

    results = run_all(args.config, args.workdir, skip_run=True)
    write_report(results)
    write_figures(results)
    print(f"Results: {results}")
    return 0


def _cmd_plan(args) -> int:
    """Enumerate work units. `--count` gives a SLURM array upper bound."""
    from spbench.config import BenchmarkConfig
    from spbench.runner import plan_units

    units = plan_units(BenchmarkConfig.load(args.config))
    if args.count:
        print(len(units))
        return 0
    try:
        print("index\tdataset\ttool")
        for unit in units:
            print(f"{unit['index']}\t{unit['dataset']}\t{unit['tool']}")
        sys.stdout.flush()
    except BrokenPipeError:
        # `spbench plan | head` is the obvious way to look at a large config.
        # Closing the fd keeps the interpreter from re-reporting the broken pipe
        # on shutdown, which would look like a crash.
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return 0


def _cmd_run_one(args) -> int:
    from spbench.config import BenchmarkConfig
    from spbench.runner import run_unit

    outdir = run_unit(BenchmarkConfig.load(args.config), Path(args.workdir), args.index)
    print(outdir)
    return 0


def _cmd_report(args) -> int:
    from spbench.report import write_figures, write_report

    output = write_report(args.results, args.output)
    write_figures(args.results)
    print(output)
    return 0


def _cmd_check_tools(args) -> int:
    """Report installation status for every registered tool.

    Run this before committing cluster time. A tool that is missing here will
    produce a ``skipped`` row in the results rather than an error, which is easy
    to miss in a long run.
    """
    from spbench.adapters import REGISTRY, build

    names = sorted(REGISTRY)
    if getattr(args, "config", None):
        from spbench.config import BenchmarkConfig

        names = [t.name for t in BenchmarkConfig.load(args.config).tools]

    all_ok = True
    width = max(len(n) for n in names)
    for name in names:
        adapter = build(name)
        ok, reason = adapter.available()
        status = "available" if ok else f"NOT AVAILABLE ({reason})"
        if not ok:
            all_ok = False
        print(f"  {name:<{width}}  {status}")
        print(f"  {'':<{width}}  {adapter.info.designed_for}")
        if adapter.info.citation:
            print(f"  {'':<{width}}  cite: {adapter.info.citation}")
        print()

    if not all_ok:
        print(
            "Missing tools are reported as 'skipped' rows, not failures - the "
            "benchmark still runs. See benchmark/envs/ for install recipes."
        )
    return 0


def _cmd_verify(args) -> int:
    """Check results against stored expectations.

    This is what CI runs. It guards the *harness*, not the tools: if a change to
    the simulator or the metrics moves the baseline's score outside tolerance,
    that is a bug in this directory, and it should fail loudly rather than
    quietly rebase the published numbers.
    """
    import pandas as pd

    expected = json.loads(Path(args.expected).read_text())
    per_sample = pd.read_csv(Path(args.results) / "per_sample.tsv", sep="\t")
    per_sample = per_sample[per_sample["representation"] == "derived"]

    failures = []
    for check in expected["checks"]:
        subset = per_sample[per_sample["tool"] == check["tool"]]
        if "dataset_contains" in check:
            subset = subset[subset["dataset"].str.contains(check["dataset_contains"])]
        if subset.empty:
            failures.append(f"{check['tool']}: no rows matched")
            continue
        value = float(subset[check["metric"]].mean())
        low, high = check["range"]
        mark = "ok " if low <= value <= high else "FAIL"
        if mark == "FAIL":
            failures.append(
                f"{check['tool']}.{check['metric']} = {value:.4f}, expected [{low}, {high}]"
            )
        print(f"  [{mark}] {check['tool']:<26} {check['metric']:<22} {value:.4f}  "
              f"expected [{low}, {high}]")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
