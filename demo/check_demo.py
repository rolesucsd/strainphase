#!/usr/bin/env python3
"""Compare demo output against the bundled ground truth.

Prints the recovered lineage trajectories beside the true strain frequencies. The
two strains sweep past each other, so a correct run shows one full-length lineage
falling and one rising, each tracking its strain to within a few percent.

Standard library only: this runs straight after `pip install strainphase`, which
does not pull in a dataframe library.
"""
import csv
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _table(rows, cols, label):
    """Fixed-width table: one line per row key, one column per sample."""
    w = max(len(label), max((len(r) for r in rows), default=0))
    out = [label.ljust(w) + "".join(s.rjust(8) for s in cols)]
    for key in rows:
        cells = "".join(
            ("" if rows[key].get(c) is None else f"{rows[key][c]:.3f}").rjust(8)
            for c in cols
        )
        out.append(key.ljust(w) + cells)
    return "\n".join(out)


def main(out_dir):
    lineages = _read(out_dir / "lineages.tsv")
    truth = _read(HERE / "truth" / "abundance.tsv")
    samples = sorted({r["sample"] for r in truth})

    true_freq = {}
    for r in truth:
        true_freq.setdefault(r["strain_id"], {})[r["sample"]] = float(r["abundance"])

    observed, n_windows = {}, {}
    for r in lineages:
        lid = r["lineage_id"]
        observed.setdefault(lid, {})[r["sample"]] = float(r["abundance"])
        n_windows[lid] = int(r["n_windows_lineage"])

    print("True strain frequencies")
    print(_table(true_freq, samples, "strain"))

    # Full-length lineages span every window; the short ones are unmerged fragments.
    longest = max(n_windows.values())
    full = {k: v for k, v in observed.items() if n_windows[k] == longest}
    print(f"\nRecovered full-length lineages ({len(full)} of {len(observed)} total; "
          f"the rest are short fragments)")
    print(_table(full, samples, "lineage"))

    print("\nMean absolute error per matched lineage")
    used = set()
    for lid in sorted(full):
        errs = {}
        for strain, freqs in true_freq.items():
            if strain in used:
                continue
            errs[strain] = sum(abs(full[lid].get(s, 0.0) - freqs[s])
                               for s in samples) / len(samples)
        if not errs:
            break
        best = min(errs, key=errs.get)
        used.add(best)
        print(f"  {lid}  ->  {best}   MAE = {errs[best]:.3f}")

    # A measured zero (looked for at depth, not found) is written as abundance 0
    # with a non-zero denominator, and is distinct from an absent row.
    zeros = [r for r in lineages
             if float(r["abundance"]) == 0 and int(r["total_reads"]) > 0]
    print(f"\nMeasured zeros (abundance 0 with reads at the locus): {len(zeros)}")
    for r in zeros[:5]:
        print(f"  {r['lineage_id']}  {r['sample']}  "
              f"reads={r['reads']}  total_reads={r['total_reads']}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "output")
