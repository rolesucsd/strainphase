#!/usr/bin/env python3
"""Generate comparison data showing strainphase rescue vs Floria for figure creation.

Produces two TSV files:
- rescue_comparison.tsv: per-lineage comparison across tools
- rescued_reads_detail.tsv: per-read rescue detail (strainphase only)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def load_truth_abundances(truth_dir: str) -> pd.DataFrame:
    """Load truth abundances into long format: (strain_id, timepoint, true_abundance)."""
    path = os.path.join(truth_dir, "truth_abundances.tsv")
    df = pd.read_csv(path, sep="\t")
    # Melt from wide to long
    timepoint_cols = [c for c in df.columns if c != "strain_id"]
    long = df.melt(id_vars="strain_id", value_vars=timepoint_cols,
                   var_name="timepoint", value_name="true_abundance")
    return long


def find_best_config_dir(sweep_dir: str) -> str:
    """Find the best config directory by reading validation metrics."""
    configs_dir = os.path.join(sweep_dir, "configs")
    best_f1 = -1.0
    best_dir = None

    for name in os.listdir(configs_dir):
        config_path = os.path.join(configs_dir, name)
        if not os.path.isdir(config_path):
            continue
        metrics_file = os.path.join(config_path, "validation", "validation_metrics.json")
        if not os.path.exists(metrics_file):
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
        f1 = metrics.get("f1", metrics.get("haplotype_f1", -1))
        # Enforce minimum window size of 5000 based on config naming convention
        try:
            ws = int(name.split("_ws")[-1])
        except Exception:
            ws = None
        if ws is None or ws < 5000:
            continue
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_dir = config_path

    if best_dir is None:
        raise FileNotFoundError(f"No valid config directories found in {configs_dir}")
    print(f"Best strainphase config (F1={best_f1:.4f}): {os.path.basename(best_dir)}")
    return best_dir


def load_lineage_details(validation_dir: str) -> pd.DataFrame:
    """Load lineage_details.tsv."""
    path = os.path.join(validation_dir, "lineage_details.tsv")
    return pd.read_csv(path, sep="\t")


def load_lineages(lineages_path: str) -> pd.DataFrame:
    """Load lineages.tsv."""
    return pd.read_csv(lineages_path, sep="\t")


def load_rescued_reads(validation_dir: str) -> pd.DataFrame:
    """Load rescued_reads.tsv if it exists."""
    path = os.path.join(validation_dir, "rescued_reads.tsv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return df
    # Rename 'sample' to 'timepoint' for consistency
    df = df.rename(columns={"sample": "timepoint"})
    return df


def load_rescue_statistics(validation_dir: str) -> pd.DataFrame:
    """Load rescue_statistics.tsv if it exists."""
    path = os.path.join(validation_dir, "rescue_statistics.tsv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    return df


def determine_rescue_status(details: pd.DataFrame,
                            rescue_stats: pd.DataFrame,
                            rescued_reads: pd.DataFrame) -> pd.DataFrame:
    """Add was_rescued and donor_timepoint columns to lineage details.

    A lineage is marked as rescued only if rescued reads at its position
    match its detected_abundance (via rescued_haplotype_weight). This avoids
    incorrectly marking high-abundance lineages as rescued when a different
    low-abundance lineage at the same position was the actual rescue target.
    """
    details = details.copy()
    details["was_rescued"] = False
    details["donor_timepoint"] = ""

    if rescued_reads.empty:
        return details

    # For each (timepoint, contig, window) group of rescued reads, find the
    # lineage whose detected_abundance best matches rescued_haplotype_weight
    for idx, row in details.iterrows():
        tp = row["timepoint"]
        contig = row["contig"]
        start = row["start_pos"]
        end = row["end_pos"]
        det_abund = row["detected_abundance"]

        # Find rescued reads at this timepoint+contig that overlap
        mask = (
            (rescued_reads["timepoint"] == tp) &
            (rescued_reads["contig"] == contig) &
            (rescued_reads["window_start"] < end) &
            (rescued_reads["window_end"] > start)
        )
        matching = rescued_reads[mask]
        if matching.empty:
            continue

        # Check if the rescued_haplotype_weight matches this lineage's
        # detected_abundance. Multiple lineages can overlap the same window;
        # only the one whose abundance matches the rescue weight is the
        # actual rescue target.
        rescue_weight = matching["rescued_haplotype_weight"].iloc[0]

        # Find all lineages overlapping this same window at this timepoint
        candidates = details[
            (details["timepoint"] == tp) &
            (details["contig"] == contig) &
            (details["start_pos"] <= end) &
            (details["end_pos"] >= start)
        ]

        if len(candidates) <= 1:
            # Only one lineage here, it must be the rescued one
            details.at[idx, "was_rescued"] = True
            donors = matching["donor_timepoint"].unique()
            details.at[idx, "donor_timepoint"] = ",".join(sorted(donors))
        else:
            # Multiple lineages overlap -- pick the one closest to rescue weight
            best_idx = (candidates["detected_abundance"] - rescue_weight).abs().idxmin()
            if best_idx == idx:
                details.at[idx, "was_rescued"] = True
                donors = matching["donor_timepoint"].unique()
                details.at[idx, "donor_timepoint"] = ",".join(sorted(donors))

    return details


def build_comparison_df(tool: str,
                        details: pd.DataFrame,
                        lineages: pd.DataFrame,
                        truth_abund: pd.DataFrame,
                        is_rescued_details: bool = False) -> pd.DataFrame:
    """Build the comparison dataframe for one tool.

    Joins lineage_details with lineages.tsv to get read counts,
    and with truth abundances.
    """
    # lineage_details columns: lineage_id, matched_strain, timepoint, contig,
    #   start_pos, end_pos, n_snvs_detected, n_snvs_true, n_shared_snvs,
    #   n_matching_snvs, n_different_snvs, snv_distance, detected_abundance,
    #   true_abundance, abundance_diff, track_id

    # lineages columns vary by format:
    #   New strainphase: lineage_id, sample, contig, track_id, abundance, total_reads, snv_alleles
    #   Old/Floria: lineage_id, sample, contig, track_id, supporting_reads, total_reads, snv_alleles

    # Aggregate lineages.tsv per (lineage_id, sample, contig) since a lineage
    # can have multiple tracks (fragments) producing multiple rows.
    # Handle both old (supporting_reads) and new (abundance) column names
    has_abundance_col = "abundance" in lineages.columns
    has_supporting_reads = "supporting_reads" in lineages.columns

    agg_dict = {
        "total_reads": ("total_reads", "sum"),
        "n_tracks": ("track_id", "count"),
    }
    if has_abundance_col:
        # New format: average abundance across tracks
        agg_dict["abundance"] = ("abundance", "mean")
    if has_supporting_reads:
        # Old format: sum supporting reads
        agg_dict["supporting_reads"] = ("supporting_reads", "sum")

    lineages_slim = lineages.groupby(
        ["lineage_id", "sample", "contig"], as_index=False
    ).agg(**agg_dict)
    lineages_slim = lineages_slim.rename(columns={"sample": "timepoint"})

    merged = details.merge(
        lineages_slim,
        on=["lineage_id", "timepoint", "contig"],
        how="left"
    )

    # If supporting_reads missing/zero, infer from detected_abundance * total_reads
    if "supporting_reads" not in merged.columns:
        merged["supporting_reads"] = 0.0
    if "total_reads" not in merged.columns:
        merged["total_reads"] = 0.0
    merged["supporting_reads"] = merged["supporting_reads"].fillna(0).astype(float)
    merged["total_reads"] = merged["total_reads"].fillna(0).astype(float)
    inferred = merged["supporting_reads"] <= 0
    merged.loc[inferred, "supporting_reads"] = (
        merged.loc[inferred, "detected_abundance"].fillna(0) *
        merged.loc[inferred, "total_reads"].fillna(0)
    )

    # Get supporting_reads: use from old format, or 0 for new format
    if "supporting_reads" in merged.columns:
        supporting_reads = merged["supporting_reads"].fillna(0).astype(int)
    else:
        # New abundance format - supporting_reads not meaningful
        supporting_reads = 0

    # Select and rename columns for output
    result = pd.DataFrame({
        "tool": tool,
        "lineage_id": merged["lineage_id"],
        "matched_strain": merged["matched_strain"],
        "timepoint": merged["timepoint"],
        "contig": merged["contig"],
        "start_pos": merged["start_pos"],
        "end_pos": merged["end_pos"],
        "supporting_reads": supporting_reads,
        "total_reads": merged["total_reads"].fillna(0).astype(int),
        "detected_abundance": merged["detected_abundance"],
        "true_abundance": merged["true_abundance"],
        "n_snvs_detected": merged["n_snvs_detected"],
        "n_snvs_matching": merged["n_matching_snvs"],
        "n_snvs_different": merged["n_different_snvs"],
        "snv_distance": merged["snv_distance"],
        "was_rescued": merged.get("was_rescued", False),
        "donor_timepoint": merged.get("donor_timepoint", ""),
        "is_detected": True,
    })

    return result


def add_undetected_rows(comparison: pd.DataFrame,
                        truth_abund: pd.DataFrame,
                        tool: str) -> pd.DataFrame:
    """Add rows for truth strains present but not detected by this tool.

    For each (strain, timepoint) with true_abundance > 0, if no lineage
    matched it, add an is_detected=False row.
    """
    tool_df = comparison[comparison["tool"] == tool]
    present = truth_abund[truth_abund["true_abundance"] > 0]

    undetected_rows = []
    for _, row in present.iterrows():
        strain = row["strain_id"]
        tp = row["timepoint"]
        abundance = row["true_abundance"]

        # Check if any lineage matched this strain at this timepoint
        matched = tool_df[
            (tool_df["matched_strain"] == strain) &
            (tool_df["timepoint"] == tp)
        ]
        if matched.empty:
            undetected_rows.append({
                "tool": tool,
                "lineage_id": "",
                "matched_strain": strain,
                "timepoint": tp,
                "contig": "",
                "start_pos": 0,
                "end_pos": 0,
                "supporting_reads": 0,
                "total_reads": 0,
                "detected_abundance": 0.0,
                "true_abundance": abundance,
                "n_snvs_detected": 0,
                "n_snvs_matching": 0,
                "n_snvs_different": 0,
                "snv_distance": float("nan"),
                "was_rescued": False,
                "donor_timepoint": "",
                "is_detected": False,
            })

    if undetected_rows:
        return pd.concat([comparison, pd.DataFrame(undetected_rows)],
                         ignore_index=True)
    return comparison


def build_rescued_reads_detail(rescued_reads: pd.DataFrame,
                               details: pd.DataFrame) -> pd.DataFrame:
    """Build detailed rescued reads output with matched_strain info.

    Matches rescued reads to lineages by checking which lineage's genomic
    region overlaps the read's window at the same timepoint.
    """
    if rescued_reads.empty:
        return pd.DataFrame()

    result_rows = []
    for _, rr in rescued_reads.iterrows():
        tp = rr["timepoint"]
        contig = rr["contig"]
        ws = rr["window_start"]
        we = rr["window_end"]

        # Find lineage overlapping this window
        mask = (
            (details["timepoint"] == tp) &
            (details["contig"] == contig) &
            (details["start_pos"] <= we) &
            (details["end_pos"] >= ws)
        )
        matching_lineages = details[mask]

        matched_strain = "unknown"
        lineage_id = ""
        if not matching_lineages.empty:
            # If there are rescued lineages (was_rescued=True), prefer those
            rescued_matches = matching_lineages[
                matching_lineages.get("was_rescued", False) == True  # noqa: E712
            ]
            if not rescued_matches.empty:
                best = rescued_matches.iloc[0]
            else:
                best = matching_lineages.iloc[0]
            matched_strain = best["matched_strain"]
            lineage_id = best["lineage_id"]

        result_rows.append({
            "read_name": rr["read_name"],
            "lineage_id": lineage_id,
            "timepoint": tp,
            "contig": contig,
            "window_start": ws,
            "window_end": we,
            "donor_timepoint": rr["donor_timepoint"],
            "n_snps_agree": rr["n_snps_agree"],
            "n_snps_disagree": rr["n_snps_disagree"],
            "n_snps_total": rr["n_snps_total"],
            "agreement_rate": rr["agreement_rate"],
            "matched_strain": matched_strain,
        })

    return pd.DataFrame(result_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate rescue figure comparison data"
    )
    parser.add_argument("--results-dir", required=True,
                        help="Root results directory")
    parser.add_argument("--coverage", required=True,
                        help="Coverage label (e.g. '10x')")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for TSVs")
    parser.add_argument("--strainphase-subdir", default=None,
                        help="Strainphase subdirectory name "
                        "(default: test_real_strains_{coverage})")
    parser.add_argument("--floria-subdir", default=None,
                        help="Floria subdirectory name "
                        "(default: floria_{coverage})")
    parser.add_argument("--strainphase-config", default=None,
                        help="Strainphase config directory path "
                        "(overrides auto-detection)")
    args = parser.parse_args()

    results_dir = args.results_dir
    coverage = args.coverage
    sp_subdir = args.strainphase_subdir or f"test_real_strains_{coverage}"
    fl_subdir = args.floria_subdir or f"floria_{coverage}"

    # Set up paths
    sp_dir = os.path.join(results_dir, sp_subdir)
    fl_dir = os.path.join(results_dir, fl_subdir)
    truth_dir = os.path.join(results_dir, "truth")

    # Load truth
    print("Loading truth abundances...")
    truth_abund = load_truth_abundances(truth_dir)

    # --- Strainphase ---
    print("Loading strainphase data...")
    if args.strainphase_config:
        sp_config_dir = args.strainphase_config
    else:
        sp_config_dir = find_best_config_dir(
            os.path.join(sp_dir, "sweep_results")
        )

    sp_validation_dir = os.path.join(sp_config_dir, "validation")
    sp_details = load_lineage_details(sp_validation_dir)
    sp_lineages = load_lineages(os.path.join(sp_config_dir, "lineages.tsv"))
    sp_rescued_reads = load_rescued_reads(sp_validation_dir)
    sp_rescue_stats = load_rescue_statistics(sp_validation_dir)

    # Determine rescue status
    sp_details = determine_rescue_status(
        sp_details, sp_rescue_stats, sp_rescued_reads
    )

    sp_comparison = build_comparison_df(
        "strainphase", sp_details, sp_lineages, truth_abund,
        is_rescued_details=True
    )

    # --- Floria ---
    print("Loading Floria data...")
    fl_validation_dir = os.path.join(fl_dir, "validation")
    fl_details = load_lineage_details(fl_validation_dir)
    fl_lineages = load_lineages(os.path.join(fl_dir, "lineages.tsv"))

    # Floria has no rescue mechanism
    fl_details["was_rescued"] = False
    fl_details["donor_timepoint"] = ""

    fl_comparison = build_comparison_df(
        "floria", fl_details, fl_lineages, truth_abund
    )

    # Combine
    comparison = pd.concat([sp_comparison, fl_comparison], ignore_index=True)

    # Add undetected strain rows
    comparison = add_undetected_rows(comparison, truth_abund, "strainphase")
    comparison = add_undetected_rows(comparison, truth_abund, "floria")

    # Sort for readability
    comparison = comparison.sort_values(
        ["tool", "matched_strain", "timepoint", "contig", "start_pos"]
    ).reset_index(drop=True)

    # --- Rescued reads detail ---
    print("Building rescued reads detail...")
    rescued_detail = build_rescued_reads_detail(sp_rescued_reads, sp_details)

    # --- Write outputs ---
    os.makedirs(args.output_dir, exist_ok=True)

    comp_path = os.path.join(args.output_dir, "rescue_comparison.tsv")
    comparison.to_csv(comp_path, sep="\t", index=False)
    print(f"Wrote {len(comparison)} rows to {comp_path}")

    detail_path = os.path.join(args.output_dir, "rescued_reads_detail.tsv")
    if not rescued_detail.empty:
        rescued_detail.to_csv(detail_path, sep="\t", index=False)
        print(f"Wrote {len(rescued_detail)} rows to {detail_path}")
    else:
        print("No rescued reads found; skipping rescued_reads_detail.tsv")

    # Print summary
    print("\n--- Summary ---")
    for tool in ["strainphase", "floria"]:
        tool_df = comparison[comparison["tool"] == tool]
        detected = tool_df[tool_df["is_detected"]]
        undetected = tool_df[~tool_df["is_detected"]]
        matched = detected[detected["matched_strain"] != "UNMATCHED"]
        rescued = detected[detected["was_rescued"]]
        perfect = matched[matched["snv_distance"] == 0.0]

        print(f"\n{tool}:")
        print(f"  Detected lineages: {len(detected)}")
        print(f"  Matched to truth:  {len(matched)}")
        print(f"  Perfect SNV match: {len(perfect)}")
        print(f"  Rescued:           {len(rescued)}")
        print(f"  Undetected strains (strain x timepoint): {len(undetected)}")


if __name__ == "__main__":
    main()
