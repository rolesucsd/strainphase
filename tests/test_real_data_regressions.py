"""Regressions captured from real cohort output (B. fragilis MAG 000089747_1).

Every test here is built from data a real run actually produced, and every one
currently FAILS. They are specifications, not descriptions: each states a
property the output should have, so that whoever fixes the underlying defect
gets an unambiguous signal. `xfail(strict=True)` means a fix flips the test to
passing AND that removing the defect without removing the marker is itself an
error, so these cannot rot silently.

The fixtures are deliberately small extracts, not whole tables:
  tests/data/bfragilis_upey_lineages.tsv.gz   4 lineages at the upeY locus
"""

from __future__ import annotations

import csv
import gzip
import itertools
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


def _load_lineages(name):
    """(lineage_id -> {marker_start, marker_end, samples:set, consensus:dict})."""
    out = {}
    with gzip.open(DATA / name, "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            consensus = {}
            for item in row["consensus"].split("|"):
                pos, _, allele = item.partition(":")
                if allele:
                    consensus[int(pos)] = allele
            out[row["lineage_id"]] = {
                "start": int(row["marker_start"]),
                "end": int(row["marker_end"]),
                "samples": set(row["samples"].split(",")),
                "consensus": consensus,
            }
    return out


def _agreement(a, b):
    shared = a.keys() & b.keys()
    if not shared:
        return None, 0
    return sum(1 for p in shared if a[p] == b[p]) / len(shared), len(shared)


def _overlaps(a, b):
    return min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0


# --------------------------------------------------------------------------
# 1. upeY: one strain reported as four lineages
# --------------------------------------------------------------------------

def test_upey_fixture_documents_the_redundancy():
    """The premise, asserted so the xfail below cannot be blamed on bad data.

    All four lineages agree well inside the 2% identity gate, occupy the same
    interval, and are seen in overlapping sets of samples. By the pipeline's own
    rule for what makes two haplotypes the same entity, they are one thing.
    """
    lin = _load_lineages("bfragilis_upey_lineages.tsv.gz")
    assert len(lin) == 4
    for a, b in itertools.combinations(sorted(lin), 2):
        frac, shared = _agreement(lin[a]["consensus"], lin[b]["consensus"])
        assert shared >= 200, (a, b, shared)
        assert frac >= 0.98, (a, b, frac)          # inside the 2% gate
        assert _overlaps(lin[a], lin[b]), (a, b)   # same stretch of genome
        assert lin[a]["samples"] & lin[b]["samples"], (a, b)  # co-occur in samples


@pytest.mark.xfail(strict=True, reason=(
    "Known defect: one strain at the upeY locus is reported as four lineages "
    "(LIN000290/284/285/286 on 000089747_1). They are >=98% identical, span the "
    "same ~1.37-1.41 Mb interval and share 14-32 samples each. Nothing merges "
    "lineages that occupy the same windows, and step-2 grouping had already "
    "split them. Flip to a plain assert when same-window merging lands."))
def test_upey_should_be_one_lineage():
    """No two reported lineages should be this similar, co-located and co-occurring."""
    lin = _load_lineages("bfragilis_upey_lineages.tsv.gz")
    redundant = [
        (a, b)
        for a, b in itertools.combinations(sorted(lin), 2)
        if (_agreement(lin[a]["consensus"], lin[b]["consensus"])[0] or 0) >= 0.98
        and _overlaps(lin[a], lin[b])
        and lin[a]["samples"] & lin[b]["samples"]
    ]
    assert redundant == [], f"{len(redundant)} redundant lineage pairs: {redundant}"


# --------------------------------------------------------------------------
# 2. Temporal reach: a strain observed in 35 timepoints collapses to 5
# --------------------------------------------------------------------------

#: Observed on 000089747_1. The old run reported LIN000088 (upaY locus, a
#: has_sweep lineage, delta_rel_freq +0.99) across 35 samples. The unified run
#: still finds it -- LIN000076 matches its consensus at 99.88% over 859 markers,
#: so identity is preserved -- but reports it in only 5 samples, built from a
#: SINGLE window group holding 5 members. The sweep is therefore invisible: it
#: is not over-merged, it is under-observed.
UPAY_SWEEP_OLD_SAMPLES = 35
UPAY_SWEEP_NEW_SAMPLES = 5


@pytest.mark.xfail(strict=True, reason=(
    "Known defect: a sweeping strain (old LIN000088, upaY) survives with its "
    "identity intact (99.88% over 859 markers) but is reported in 5 of its 35 "
    "timepoints, because step-2 grouped only 5 samples' haplotypes together. A "
    "sweep cannot be seen from 14% of its observations. No fix has landed yet, "
    "so this records the target rather than a reproduction."))
def test_sweeping_strain_retains_its_timepoints():
    """A strain whose consensus is unchanged should keep most of its observations."""
    retained = UPAY_SWEEP_NEW_SAMPLES / UPAY_SWEEP_OLD_SAMPLES
    assert retained >= 0.5, (
        f"retained only {retained:.0%} of timepoints "
        f"({UPAY_SWEEP_NEW_SAMPLES}/{UPAY_SWEEP_OLD_SAMPLES})"
    )
