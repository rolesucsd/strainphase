"""Turn the tidy results tables into a report a reviewer can read.

The report is opinionated about presentation in one respect: it never puts a
multi-sample method and a single-sample method in the same table without saying
so. The headline table is single-sample tools plus ``strainphase-single``; the
longitudinal result is a separate section whose comparator is the ablation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

#: (column, label, higher-is-better)
HEADLINE_METRICS: list[tuple[str, str, bool]] = [
    ("ari", "Read ARI", True),
    ("v_measure", "V-measure", True),
    ("assigned_fraction", "Reads placed", True),
    ("hap_f1", "Haplotype F1", True),
    ("hamming_error_rate", "Allele err.", False),
    ("switch_error_rate", "Switch err.", False),
    ("k_error", "k error", False),
]

LONGITUDINAL_METRICS: list[tuple[str, str]] = [
    ("recall_lt_1pct", "<1%"),
    ("recall_1_5pct", "1-5%"),
    ("recall_5_20pct", "5-20%"),
    ("recall_gt_20pct", ">20%"),
    ("recall_overall", "overall"),
    ("mean_persistence_recall", "persistence"),
]


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _mean_ci(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    """Mean and half-width of a 95% CI **across replicate seeds**.

    The two-step aggregation matters. Rows within one seed are timepoints and
    contigs of the same simulated mixture, so they are not independent
    observations; treating them as such would report a confidence interval
    several times narrower than the truth. Each seed is collapsed to one value
    first, and the interval is taken over those. With a single seed there is no
    interval to report and the column shows the mean alone.
    """
    if column not in frame:
        return float("nan"), float("nan")
    values = frame[column].dropna()
    if values.empty:
        return float("nan"), float("nan")
    if "seed" not in frame:
        return float(values.mean()), float("nan")

    per_seed = frame.groupby("seed")[column].mean().dropna()
    if per_seed.empty:
        return float("nan"), float("nan")
    mean = float(per_seed.mean())
    if len(per_seed) < 2:
        return mean, float("nan")
    return mean, float(1.96 * per_seed.std(ddof=1) / math.sqrt(len(per_seed)))


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(results_dir: str | Path) -> str:
    results_dir = Path(results_dir)
    per_sample = pd.read_csv(results_dir / "per_sample.tsv", sep="\t")
    runs = pd.read_csv(results_dir / "runs.tsv", sep="\t")
    longitudinal = (
        pd.read_csv(results_dir / "longitudinal.tsv", sep="\t")
        if (results_dir / "longitudinal.tsv").exists()
        else pd.DataFrame()
    )

    parts: list[str] = ["# strainphase benchmark report", ""]
    parts.append(_provenance_section(results_dir))
    parts.append(_tools_section(runs))
    parts.append(_headline_section(per_sample))
    parts.append(_longitudinal_section(longitudinal, per_sample))
    parts.append(_depth_section(results_dir))
    parts.append(_resources_section(runs))
    parts.append(_caveats_section(per_sample))
    return "\n\n".join(p for p in parts if p)


def _provenance_section(results_dir: Path) -> str:
    path = results_dir / "provenance.json"
    if not path.exists():
        return ""
    import json

    prov = json.loads(path.read_text())
    dirty = " (uncommitted changes present)" if prov.get("git_dirty") else ""
    versions = ", ".join(f"{k} {v}" for k, v in sorted(prov.get("package_versions", {}).items()))
    return (
        "## Provenance\n\n"
        f"- Generated: {prov.get('generated_utc')}\n"
        f"- Config: `{prov.get('config')}`, seeds {prov.get('seeds')}\n"
        f"- Commit: `{prov.get('git_commit', '')[:12]}`{dirty}\n"
        f"- Match threshold: {prov.get('match_threshold')} agreement over "
        f"≥{prov.get('min_shared_sites')} shared sites\n"
        f"- Platform: {prov.get('platform')}, {prov.get('cpu_count')} CPUs\n"
        f"- Versions: {versions}\n"
    )


def _tools_section(runs: pd.DataFrame) -> str:
    if runs.empty:
        return ""
    rows = []
    for tool, group in runs.groupby("tool"):
        first = group.iloc[0]
        statuses = sorted(set(group["status"].astype(str)))
        note = ""
        if "skipped" in statuses or "failed" in statuses:
            messages = sorted({str(m) for m in group.get("message", []) if str(m) not in ("", "nan")})
            note = "; ".join(messages)[:160]
        rows.append(
            [
                f"`{tool}`",
                "yes" if first.get("multi_sample") else "no",
                "yes" if first.get("supports_cross_sample_ids") else "no",
                ", ".join(statuses),
                str(first.get("designed_for", ""))[:180] + (f" — {note}" if note else ""),
            ]
        )
    return (
        "## Tools compared\n\n"
        "`Multi-sample` means the tool sees every timepoint in one invocation. "
        "`Stable IDs` means it claims the same identifier for the same organism "
        "across timepoints. Columns that a tool does not claim are reported as "
        "`n/a`, never as zero.\n\n"
        + _table(rows, ["Tool", "Multi-sample", "Stable IDs", "Status", "Designed for / notes"])
    )


def _headline_section(per_sample: pd.DataFrame) -> str:
    """Single-sample tools only, so every row saw exactly the same information."""
    df = per_sample[
        (per_sample["representation"] == "derived")
        & (~per_sample["multi_sample"].astype(bool))
        & (per_sample["status"] == "ok")
    ]
    if df.empty:
        return "## Like-for-like comparison\n\n_No single-sample tool produced results._"

    sections = ["## Like-for-like comparison (single-sample tools)", "",
                "Every tool below received one timepoint at a time, the same BAM "
                "and the same VCF. `strainphase-single` is strainphase with "
                "cross-timepoint rescue disabled, which is the only strainphase "
                "configuration that belongs in this table. Values are means over "
                "replicate seeds; ± is a 95% CI where more than one seed was run."]

    for dataset_group, group in _group_by_condition(df):
        rows = []
        for tool, tool_group in group.groupby("tool"):
            row = [f"`{tool}`"]
            for column, _label, _higher in HEADLINE_METRICS:
                mean, ci = _mean_ci(tool_group, column)
                row.append(_fmt(mean) + ("" if math.isnan(ci) else f" ±{ci:.3f}"))
            rows.append(row)
        header = ["Tool", *[label for _c, label, _h in HEADLINE_METRICS]]
        sections.append(f"### {dataset_group}\n\n" + _table(rows, header))

    sections.append(
        "_Read ARI and V-measure score the read partition, which every tool "
        "genuinely produces. Haplotype F1 counts a strain recovered only when a "
        "predicted haplotype agrees with it above the match threshold over "
        "enough shared sites. `k error` is predicted minus true strain count: "
        "positive means over-splitting, negative means merging._"
    )
    return "\n\n".join(sections)


def _group_by_condition(df: pd.DataFrame):
    """Group results by simulated condition, not by individual seed."""
    keys = [k for k in ("n_strains_config", "coverage") if k in df.columns]
    if not keys:
        yield "All datasets", df
        return
    for values, group in df.groupby(keys):
        values = values if isinstance(values, tuple) else (values,)
        label = ", ".join(
            f"{'strains' if k == 'n_strains_config' else k}={v}"
            for k, v in zip(keys, values, strict=True)
        )
        yield label, group


def _longitudinal_section(longitudinal: pd.DataFrame, per_sample: pd.DataFrame) -> str:
    if longitudinal.empty:
        return ""
    df = longitudinal[
        (longitudinal["representation"] == "derived") & (longitudinal["status"] == "ok")
    ]
    if df.empty:
        return ""

    # Recall alone is gameable: a method that emits fifty fragments per sample
    # will match a rare strain with one of them by luck. The haplotype count is
    # printed in the same table so recall is never read without its cost.
    predicted = (
        per_sample[
            (per_sample["representation"] == "derived") & (per_sample["status"] == "ok")
        ]
        .groupby("tool")["n_predicted"]
        .mean()
        if "n_predicted" in per_sample
        else pd.Series(dtype=float)
    )

    rows = []
    for tool, group in df.groupby("tool"):
        row = [f"`{tool}`"]
        for column, _label in LONGITUDINAL_METRICS:
            mean, ci = _mean_ci(group, column)
            row.append(_fmt(mean) + ("" if math.isnan(ci) else f" ±{ci:.3f}"))
        row.append(
            _fmt(float(predicted.get(tool)), 1) if tool in predicted.index else "n/a"
        )
        if "cross_sample_id_ari" in group and bool(group.iloc[0].get("cross_sample_id_supported")):
            mean, _ = _mean_ci(group, "cross_sample_id_ari")
            row.append(_fmt(mean))
        else:
            row.append("n/a")
        rows.append(row)

    header = [
        "Tool",
        *[f"Recall {label}" for _c, label in LONGITUDINAL_METRICS[:5]],
        "Persistence",
        "Haps/sample",
        "Cross-TP ID ARI",
    ]
    return (
        "## Detection sensitivity by true abundance\n\n"
        "Recall of true strains, stratified by the abundance they actually had "
        "in that timepoint. This is the comparison the longitudinal claim has to "
        "win, and it is computed identically for every tool from the same truth "
        "table.\n\n"
        "The comparison that carries weight is `strainphase-longitudinal` versus "
        "`strainphase-single`: identical code, identical data, one flag apart. A "
        "gap in the low-abundance bins that does not appear in the >20% bin is "
        "evidence for cross-timepoint rescue specifically, rather than for the "
        "method in general.\n\n"
        "Single-sample tools appear here for context. They are not being faulted "
        "for low scores in the rare bins - they were never given the information "
        "those bins require.\n\n"
        "**Read recall against the `Haps/sample` column.** Matching is one-to-one, "
        "so a method that shatters the sample into many fragments will match a "
        "rare strain with one of them by chance and score high recall while "
        "reconstructing nothing. `naive-greedy` exists in this table to make that "
        "failure mode visible: it typically posts strong recall and an absurd "
        "haplotype count. A recall number is only meaningful next to a haplotype "
        "count near the true strain count.\n\n"
        + _table(rows, header)
        + "\n\n_Cross-TP ID ARI is `n/a` for tools that assign fresh identifiers "
        "per sample; scoring them on it would be measuring a claim they do not "
        "make._"
    )


#: Absolute per-strain depth bins, in x. The boundaries are where the physics
#: changes, not round numbers: below ~1x a strain is not covered end to end by
#: any read set, 1-3x is where a single sample has evidence a strain exists but
#: not enough to phase it, and above ~10x single-sample phasing should work.
DEPTH_BINS: list[tuple[str, float, float]] = [
    ("<1x", 0.0, 1.0),
    ("1-3x", 1.0, 3.0),
    ("3-10x", 3.0, 10.0),
    (">10x", 10.0, float("inf")),
]


def _depth_section(results_dir: Path) -> str:
    """Recall stratified by the strain's absolute read depth.

    Abundance fraction is the wrong axis to compare across coverage levels: a 1%
    strain at 300x has 3x of reads and is recoverable, while a 1% strain at 20x
    has 0.2x and is not present in the data in any useful sense. Stratifying by
    ``abundance x coverage`` puts every dataset on one axis and separates "the
    method missed it" from "the reads were not there".
    """
    path = results_dir / "detection.tsv"
    if not path.exists():
        return ""
    detection = pd.read_csv(path, sep="\t")
    if detection.empty or "coverage" not in detection:
        return ""
    detection = detection[
        (detection["representation"] == "derived")
        & (detection["status"] == "ok")
        & (detection["true_abundance"] > 0)
    ].copy()
    if detection.empty:
        return ""
    detection["strain_depth"] = detection["true_abundance"] * detection["coverage"]

    rows = []
    for tool, group in detection.groupby("tool"):
        row = [f"`{tool}`"]
        for label, low, high in DEPTH_BINS:
            in_bin = group[(group["strain_depth"] >= low) & (group["strain_depth"] < high)]
            row.append(
                f"{in_bin['detected'].mean():.3f} (n={len(in_bin)})" if len(in_bin) else "n/a"
            )
        rows.append(row)

    return (
        "## Detection sensitivity by absolute strain depth\n\n"
        "The same recall, re-binned by how many reads a strain actually had "
        "(`abundance x coverage`). A strain below 1x is not meaningfully present "
        "in a single sample, so a single-sample method missing it is not an "
        "error - it is the correct answer given its input. The 1-3x band is "
        "where cross-timepoint information can change the outcome, and it is the "
        "band to look at when judging the longitudinal claim.\n\n"
        + _table(rows, ["Tool", *[label for label, _l, _h in DEPTH_BINS]])
    )


def _resources_section(runs: pd.DataFrame) -> str:
    df = runs[(runs["status"] == "ok") & (runs["representation"] == "derived")]
    if df.empty or "wall_seconds" not in df:
        return ""
    rows = []
    for tool, group in df.groupby("tool"):
        rows.append(
            [
                f"`{tool}`",
                _fmt(group["wall_seconds"].mean(), 1),
                _fmt(group["peak_rss_mb"].mean(), 0) if "peak_rss_mb" in group else "n/a",
                str(int(group["n_haplotypes"].mean())) if "n_haplotypes" in group else "n/a",
            ]
        )
    return (
        "## Resource use\n\n"
        "Wall time is per dataset (all timepoints), measured in-process; it "
        "includes each tool's own I/O but not the shared consensus-derivation "
        "step, which is charged to no one.\n\n"
        + _table(rows, ["Tool", "Mean wall (s)", "Mean peak RSS (MB)", "Mean haplotypes"])
    )


def _caveats_section(per_sample: pd.DataFrame) -> str:
    """Limitations, phrased against the read model these results actually used.

    Printing "alignments are exact" under results generated with minimap2 would
    be worse than printing nothing, so this section reads the run rather than
    asserting a fixed set of caveats.
    """
    models = (
        sorted({str(m) for m in per_sample["read_model"].dropna().unique()})
        if "read_model" in per_sample
        else []
    )
    has_exact = "exact" in models or not models
    has_hifi = "hifi" in models

    lines = ["## What this benchmark does not show", ""]

    if has_hifi:
        lines.append(
            "- **Reads were aligned, not placed.** Datasets using `read_model: "
            "hifi` give reads homopolymer-concentrated indel error plus "
            "substitutions, sequence them from both strands, and align them back "
            "with minimap2, so placement ambiguity, soft clipping and mapping "
            "quality are real. What is still missing is a true CCS error model: "
            "the error parameters are literature-shaped approximations, not "
            "fitted to an instrument. For higher fidelity, simulate per-strain "
            "reads with PBSIM3 and feed those in."
        )
    if has_exact:
        lines.append(
            "- **Some datasets use exact alignments.** Rows with `read_model: "
            "exact` emit reads at their true coordinates with exact CIGARs and "
            "substitution-only error, so no tool pays for alignment error or "
            "indel placement ambiguity. Those numbers are optimistic for every "
            "tool and are not directly comparable to the `hifi` rows."
        )
    if has_exact and has_hifi:
        lines.append(
            "- **The gap between the two is itself a result.** Comparing the same "
            "condition under both read models measures how much of a tool's score "
            "came from the simulation being clean. Slice `per_sample.tsv` on "
            "`read_model` to see it."
        )

    lines.append(
        "- **One reference per dataset.** Cross-species mismapping, a major "
        "source of false haplotypes in real metagenomes, is absent by "
        "construction.\n"
        "- **Consensus haplotypes are derived by the harness**, identically for "
        "every tool, from the tool's read partition. This is deliberate - it "
        "removes four different consensus callers as a confounder - but it means "
        "these numbers are not each tool's native output quality. Rows labelled "
        "`native` in `per_sample.tsv` report native output where a tool supplies "
        "it.\n"
        "- **Tools were run at published defaults.** No per-tool tuning was done "
        "for any tool, including strainphase. Tuning strainphase alone would "
        "invalidate the comparison; tuning all of them fairly is a larger "
        "exercise than this suite performs.\n"
        "- **No real-data track with independent ground truth.** A sequenced "
        "mock community or defined isolate mixture would test what simulation "
        "cannot. This is the largest remaining gap."
    )
    return "\n".join(lines)


def write_report(results_dir: str | Path, output: str | Path | None = None) -> Path:
    results_dir = Path(results_dir)
    output = Path(output) if output else results_dir / "report.md"
    output.write_text(build_report(results_dir))
    return output


def write_figures(results_dir: str | Path) -> list[Path]:
    """Detection-versus-abundance figure. Skipped silently without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    results_dir = Path(results_dir)
    detection_path = results_dir / "detection.tsv"
    if not detection_path.exists():
        return []

    detection = pd.read_csv(detection_path, sep="\t")
    detection = detection[
        (detection["representation"] == "derived")
        & (detection["status"] == "ok")
        & (detection["true_abundance"] > 0)
    ]
    if detection.empty:
        return []

    bins = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.01]
    detection["bin"] = pd.cut(detection["true_abundance"], bins=bins)
    curves: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (tool, abundance_bin), group in detection.groupby(["tool", "bin"], observed=True):
        curves[tool].append((abundance_bin.mid, group["detected"].mean()))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tool, points in sorted(curves.items()):
        points.sort()
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            label=tool,
            linewidth=2 if "strainphase" in tool else 1.4,
        )
    ax.set_xscale("log")
    ax.set_xlabel("True strain abundance in timepoint")
    ax.set_ylabel("Fraction of strains recovered")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Detection sensitivity versus abundance")
    fig.tight_layout()

    out = results_dir / "detection_vs_abundance.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return [out]
