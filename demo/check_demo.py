#!/usr/bin/env python3
"""Compare demo output against the bundled ground truth.

Prints the recovered lineage trajectories beside the true strain frequencies. The
two strains sweep past each other, so a correct run shows one full-length lineage
falling and one rising, each tracking its strain to within a few percent.
"""
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "output")

lin = pd.read_csv(out / "lineages.tsv", sep="\t")
truth = pd.read_csv(HERE / "truth" / "abundance.tsv", sep="\t")
samples = sorted(truth["sample"].unique())

# Full-length lineages are the ones that span every window; the short fragments are
# unmerged tails and are expected.
wide = lin.pivot_table(index="lineage_id", columns="sample", values="abundance")
wide = wide.reindex(columns=samples)
n_win = lin.groupby("lineage_id")["n_windows_lineage"].first()
full = wide.loc[n_win[n_win == n_win.max()].index]

print("True strain frequencies")
print(truth.pivot_table(index="strain_id", columns="sample",
                        values="abundance").reindex(columns=samples).round(3).to_string())
print(f"\nRecovered full-length lineages ({len(full)} of {len(wide)} total; "
      f"the rest are short fragments)")
print(full.round(3).to_string())

# Match each full-length lineage to the strain it tracks and report the error.
print("\nMean absolute error per matched lineage")
tr = truth.pivot_table(index="strain_id", columns="sample", values="abundance").reindex(columns=samples)
used = set()
for lid, row in full.iterrows():
    errs = {s: (row.fillna(0) - tr.loc[s]).abs().mean() for s in tr.index if s not in used}
    if not errs:
        break
    best = min(errs, key=errs.get)
    used.add(best)
    print(f"  {lid}  ->  {best}   MAE = {errs[best]:.3f}")

# A measured zero (the lineage was looked for at depth and not found) is written as
# abundance 0 with a non-zero denominator, and is distinct from an absent row.
zeros = lin[(lin.abundance == 0) & (lin.total_reads > 0)]
print(f"\nMeasured zeros (abundance 0 with reads at the locus): {len(zeros)}")
if len(zeros):
    print(zeros[["lineage_id", "sample", "abundance", "reads", "total_reads"]]
          .head().to_string(index=False))
