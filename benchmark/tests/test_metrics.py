"""Unit tests for the metric implementations.

The metrics decide what the paper claims, so they are tested against cases whose
correct answers are known independently of this code: a perfect partition, a
random one, a deliberately chimeric haplotype. Without these, a scoring bug
would be indistinguishable from a result.
"""

from __future__ import annotations

import math

import pytest

from spbench.formats import Haplotype, Truth, decode_alleles, encode_alleles
from spbench.metrics.haplotype import agreement, haplotype_metrics, match_haplotypes, switch_count
from spbench.metrics.partition import (
    adjusted_mutual_information,
    adjusted_rand_index,
    contingency,
    homogeneity_completeness_v,
    partition_metrics,
)

# --------------------------------------------------------------------------- #
# Allele encoding
# --------------------------------------------------------------------------- #


def test_allele_round_trip():
    alleles = {17: "A", 3: "GT", 250: "C"}
    assert decode_alleles(encode_alleles(alleles)) == alleles


def test_decode_tolerates_junk():
    assert decode_alleles("") == {}
    assert decode_alleles("bad,10:A,,x:y") == {10: "A"}


# --------------------------------------------------------------------------- #
# Partition metrics
# --------------------------------------------------------------------------- #


def test_perfect_partition_scores_one():
    truth = ["a"] * 50 + ["b"] * 50
    pred = ["x"] * 50 + ["y"] * 50
    matrix = contingency(truth, pred)
    assert adjusted_rand_index(matrix) == pytest.approx(1.0)
    assert adjusted_mutual_information(matrix) == pytest.approx(1.0, abs=1e-6)
    h, c, v = homogeneity_completeness_v(matrix)
    assert (h, c, v) == pytest.approx((1.0, 1.0, 1.0))


def test_single_cluster_has_zero_ari():
    """Lumping everything together is chance-level, not a partial credit."""
    truth = ["a"] * 50 + ["b"] * 50
    matrix = contingency(truth, ["x"] * 100)
    assert adjusted_rand_index(matrix) == pytest.approx(0.0, abs=1e-9)


def test_over_splitting_separates_homogeneity_from_completeness():
    """Each true class shattered into pure singleton-ish clusters."""
    truth = ["a"] * 20 + ["b"] * 20
    pred = [f"c{i // 2}" for i in range(40)]
    h, c, _v = homogeneity_completeness_v(contingency(truth, pred))
    assert h == pytest.approx(1.0)  # no cluster mixes classes
    assert c < 0.6  # but each class is scattered


def test_random_partition_ari_near_zero():
    import random

    rng = random.Random(0)
    truth = [rng.choice("abc") for _ in range(600)]
    pred = [rng.choice("xyz") for _ in range(600)]
    assert abs(adjusted_rand_index(contingency(truth, pred))) < 0.05
    assert abs(adjusted_mutual_information(contingency(truth, pred))) < 0.05


def test_unassigned_reads_lower_assigned_fraction_not_the_score():
    truth = {("T1", f"r{i}"): ("a" if i < 20 else "b") for i in range(40)}
    # Half the reads placed, and placed correctly.
    pred = {("T1", f"r{i}"): ("x" if i < 10 else "y") for i in range(40) if i < 10 or i >= 30}
    metrics = partition_metrics(truth, pred, "T1")
    assert metrics["assigned_fraction"] == pytest.approx(0.5)
    assert metrics["ari"] == pytest.approx(1.0)
    assert metrics["n_reads"] == 40


def test_partition_metrics_handles_empty_input():
    assert partition_metrics({}, {}, "T1")["n_reads"] == 0


# --------------------------------------------------------------------------- #
# Haplotype metrics
# --------------------------------------------------------------------------- #


def _truth_with(strains: dict[str, dict[int, str]], samples=("T1",), abundance=0.5) -> Truth:
    truth = Truth()
    for sid, alleles in strains.items():
        truth.strains[sid] = Haplotype(hap_id=sid, sample="", contig="c1", alleles=alleles)
        for sample in samples:
            truth.abundance[(sample, sid)] = abundance
    truth.samples = list(samples)
    return truth


def test_agreement_uses_shared_positions_only():
    frac, n_agree, n_shared = agreement({1: "A", 2: "C", 9: "T"}, {1: "A", 2: "G"})
    assert (n_agree, n_shared) == (1, 2)
    assert frac == pytest.approx(0.5)
    assert agreement({}, {1: "A"}) == (0.0, 0, 0)


def test_matching_is_global_not_greedy():
    """A greedy matcher would give strain_a to the long haplotype and leave
    strain_b unmatched; the optimal assignment matches both."""
    strains = {
        "strain_a": {i: "A" for i in range(1, 41)},
        "strain_b": {i: ("A" if i < 21 else "C") for i in range(1, 41)},
    }
    truth = _truth_with(strains)
    predicted = [
        Haplotype("h1", "T1", "c1", {i: "A" for i in range(1, 41)}),
        Haplotype("h2", "T1", "c1", {i: ("A" if i < 21 else "C") for i in range(1, 41)}),
    ]
    matches = {sid: hap.hap_id for hap, sid, _f, _n in match_haplotypes(predicted, truth, "c1")}
    assert matches == {"strain_a": "h1", "strain_b": "h2"}


def test_short_fragments_below_min_shared_never_match():
    truth = _truth_with({"strain_a": {i: "A" for i in range(1, 41)}})
    fragment = Haplotype("h1", "T1", "c1", {1: "A", 2: "A", 3: "A"})
    assert match_haplotypes([fragment], truth, "c1", min_shared_sites=10) == []


def test_switch_count_flags_a_chimera():
    """A haplotype that is strain_a in its first half and strain_b in its second
    must register exactly one switch."""
    strains = {
        "strain_a": {i: "A" for i in range(1, 41)},
        "strain_b": {i: "C" for i in range(1, 41)},
    }
    truth = _truth_with(strains)
    chimera = {i: ("A" if i <= 20 else "C") for i in range(1, 41)}
    switches, opportunities = switch_count(chimera, truth, "c1")
    assert switches == 1
    assert opportunities == 39


def test_switch_count_zero_for_a_faithful_haplotype():
    truth = _truth_with({"strain_a": {i: "A" for i in range(1, 41)}})
    switches, _ = switch_count({i: "A" for i in range(1, 41)}, truth, "c1")
    assert switches == 0


def test_perfect_reconstruction_scores_f1_one():
    strains = {
        "strain_a": {i: "A" for i in range(1, 61)},
        "strain_b": {i: ("A" if i % 2 else "G") for i in range(1, 61)},
    }
    truth = _truth_with(strains)
    predicted = [
        Haplotype("h1", "T1", "c1", dict(strains["strain_a"]), 1, 60, 0.5),
        Haplotype("h2", "T1", "c1", dict(strains["strain_b"]), 1, 60, 0.5),
    ]
    metrics = haplotype_metrics(predicted, truth, "T1", "c1")
    assert metrics["hap_f1"] == pytest.approx(1.0)
    assert metrics["k_error"] == 0
    assert metrics["hamming_error_rate"] == pytest.approx(0.0)
    assert metrics["abundance_mae"] == pytest.approx(0.0)


def test_missing_a_strain_costs_recall_not_precision():
    strains = {
        "strain_a": {i: "A" for i in range(1, 61)},
        "strain_b": {i: ("A" if i % 2 else "G") for i in range(1, 61)},
    }
    truth = _truth_with(strains)
    predicted = [Haplotype("h1", "T1", "c1", dict(strains["strain_a"]), 1, 60, 1.0)]
    metrics = haplotype_metrics(predicted, truth, "T1", "c1")
    assert metrics["hap_precision"] == pytest.approx(1.0)
    assert metrics["hap_recall"] == pytest.approx(0.5)
    assert metrics["k_error"] == -1


def test_over_splitting_costs_precision():
    strains = {"strain_a": {i: "A" for i in range(1, 61)}}
    truth = _truth_with(strains)
    predicted = [
        Haplotype(f"h{k}", "T1", "c1", dict(strains["strain_a"]), 1, 60, 0.25)
        for k in range(4)
    ]
    metrics = haplotype_metrics(predicted, truth, "T1", "c1")
    assert metrics["hap_precision"] == pytest.approx(0.25)
    assert metrics["hap_recall"] == pytest.approx(1.0)
    assert metrics["k_error"] == 3


def test_abundance_is_nan_when_the_tool_reports_none():
    truth = _truth_with({"strain_a": {i: "A" for i in range(1, 61)}})
    predicted = [Haplotype("h1", "T1", "c1", {i: "A" for i in range(1, 61)}, 1, 60, None)]
    metrics = haplotype_metrics(predicted, truth, "T1", "c1")
    assert metrics["reports_abundance"] is False
    assert math.isnan(metrics["abundance_mae"])
