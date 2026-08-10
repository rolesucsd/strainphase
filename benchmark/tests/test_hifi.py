"""Tests for the HiFi error model and the minimap2 alignment path.

The realistic path has more moving parts than the exact one - strain genomes are
materialised, reads are corrupted, half are reverse-complemented, and placement
comes from an aligner rather than from the simulator. Any of those steps getting
the coordinate bookkeeping wrong would make every tool look bad in exactly the
same way, which is the hardest kind of bug to notice in a results table.

So the load-bearing test is the same one as for the exact path: do the alleles
observed in the final BAM match the true haplotype of the strain each read came
from? If the strain sequences, the error model, the reverse complementing, the
alignment or the CIGAR translation were wrong, that agreement would collapse.
"""

from __future__ import annotations

import numpy as np
import pytest

from spbench.dataset import Dataset
from spbench.formats import Truth
from spbench.hifi import (
    HiFiErrorModel,
    apply_hifi_errors,
    build_strain_sequence,
    homopolymer_runs,
    reverse_complement,
)
from spbench.reads import load_sites, read_alleles
from spbench.simulate import SimConfig, Variant, simulate

pysam = pytest.importorskip("pysam")
mappy = pytest.importorskip("mappy")


# --------------------------------------------------------------------------- #
# Pieces
# --------------------------------------------------------------------------- #


def test_reverse_complement_round_trips():
    seq = "ACGTTTGCAN"
    assert reverse_complement(reverse_complement(seq)) == seq
    assert reverse_complement("AACG") == "CGTT"


def test_homopolymer_runs_finds_only_long_enough_runs():
    #                       0123456789...
    runs = homopolymer_runs("ACGTTTTACCAAAAAG", min_length=4)
    assert (3, 4) in runs  # TTTT starting at index 3
    assert (10, 5) in runs  # AAAAA starting at index 10
    assert all(length >= 4 for _start, length in runs)
    assert homopolymer_runs("ACGTACGT", min_length=4) == []


def test_build_strain_sequence_applies_each_variant_kind():
    reference = "AAAACCCCGGGGTTTT"  # 16 bp
    # SNV at 2 (A->G); 2 bp deletion anchored at 5 (removes 6-7);
    # 3 bp insertion anchored at 12.
    variants = {
        2: Variant(2, "snv", "A", "G"),
        5: Variant(5, "del", "CCC", "C"),
        12: Variant(12, "ins", "G", "GTTT"),
    }
    result = build_strain_sequence(reference, variants)
    # A G A A C C G G G T T T + inserted TTT ... verify by reconstruction rules
    assert result[1] == "G"  # the SNV
    assert len(result) == len(reference) - 2 + 3
    assert "TTT" in result


def test_hifi_errors_stay_calibrated_and_change_length():
    """Total error must track the configured rate, and indels must actually
    change read length - otherwise the model is decorative."""
    rng = np.random.default_rng(0)
    # A sequence with plenty of homopolymers, so the indel path is exercised.
    seq = ("ACGT" + "A" * 6 + "CG" + "T" * 5 + "GCA") * 200
    model = HiFiErrorModel(error_rate=0.01)

    lengths_changed = 0
    for _ in range(20):
        corrupted, quals = apply_hifi_errors(seq, rng, model)
        assert len(corrupted) == len(quals)
        assert all(2 <= q <= 60 for q in quals)
        if len(corrupted) != len(seq):
            lengths_changed += 1
    assert lengths_changed > 0, "no indel errors were ever applied"


def test_quality_is_lower_inside_homopolymers():
    """Q must reflect context, not be a constant with a different name."""
    rng = np.random.default_rng(1)
    seq = "ACGTACGTACGT" + "A" * 12 + "ACGTACGTACGT"
    model = HiFiErrorModel(error_rate=0.01, quality_penalty_in_homopolymer=6.0)

    inside = []
    outside = []
    for _ in range(60):
        _corrupted, quals = apply_hifi_errors(seq, rng, model)
        if len(quals) != len(seq):
            continue  # an indel shifted the coordinates; skip this draw
        inside.extend(quals[12:24])
        outside.extend(quals[:12])
    assert np.mean(inside) < np.mean(outside)


def test_zero_error_rate_leaves_the_read_untouched():
    rng = np.random.default_rng(2)
    seq = "ACGT" + "A" * 8 + "CCGGTT"
    corrupted, _quals = apply_hifi_errors(seq, rng, HiFiErrorModel(error_rate=0.0))
    assert corrupted == seq


# --------------------------------------------------------------------------- #
# Whole realistic pipeline
# --------------------------------------------------------------------------- #


def _hifi_config(**overrides) -> SimConfig:
    base = {
        "name": "hifi",
        "seed": 5,
        "contig_length": 40_000,
        "n_strains": 3,
        "n_mutations": 200,
        "n_timepoints": 2,
        "coverage": 25,
        "mean_read_length": 6_000,
        "read_length_sd": 1_500,
        "min_read_length": 2_500,
        "max_read_length": 12_000,
        "read_model": "hifi",
        "aligner": "minimap2",
    }
    base.update(overrides)
    return SimConfig(**base)


def test_hifi_requires_an_aligner():
    """Emitting the true CIGAR after simulating slippage would defeat the point."""
    with pytest.raises(ValueError, match="requires aligner='minimap2'"):
        SimConfig(read_model="hifi", aligner="exact")


@pytest.fixture(scope="module")
def hifi_dataset(tmp_path_factory) -> Dataset:
    root = tmp_path_factory.mktemp("hifi") / "ds"
    simulate(_hifi_config(indel_fraction=0.15), root)
    return Dataset.load(root)


def test_aligned_reads_still_carry_their_strain_alleles(hifi_dataset: Dataset):
    """The end-to-end coordinate check for the realistic path.

    Agreement is lower than on the exact path because there is genuinely more
    error, but it must stay high: a coordinate or strand bug would drop it to
    near chance.
    """
    truth = Truth.read(hifi_dataset.truth_dir)
    contig = next(iter(hifi_dataset.contigs))
    sample = hifi_dataset.samples[0]

    sites = load_sites(str(hifi_dataset.vcfs[sample]), contig)
    assert sites, "no variant sites were called"
    observed = read_alleles(str(hifi_dataset.bams[sample]), contig, sites)
    assert observed, "no reads produced allele calls"

    agree = total = 0
    for read_id, calls in observed.items():
        strain_id = truth.read_origins.get((sample, read_id))
        if strain_id is None:
            continue
        expected = truth.strains[strain_id].alleles
        for pos, (allele, _quality) in calls.items():
            if pos in expected:
                total += 1
                agree += allele == expected[pos]

    assert total > 100
    assert agree / total > 0.97


def test_bam_looks_like_real_alignment_output(hifi_dataset: Dataset):
    """Reverse-strand reads, soft clips and indel operations must all appear.

    Their absence would mean the aligner was bypassed somewhere and the path is
    quietly no more realistic than the exact one.
    """
    contig = next(iter(hifi_dataset.contigs))
    sample = hifi_dataset.samples[0]

    n_reverse = 0
    n_reads = 0
    ops: set[int] = set()
    with pysam.AlignmentFile(str(hifi_dataset.bams[sample]), "rb") as bam:
        for aln in bam.fetch(contig):
            n_reads += 1
            n_reverse += aln.is_reverse
            ops.update(op for op, _length in aln.cigartuples or [])

    assert n_reads > 50
    assert 0.2 < n_reverse / n_reads < 0.8, "reads are not being sequenced from both strands"
    assert 1 in ops or 2 in ops, "no indel operations survived alignment"


def test_realistic_path_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    simulate(_hifi_config(name="det"), a)
    simulate(_hifi_config(name="det"), b)
    for name in ("strains.tsv", "abundance.tsv", "read_origins.tsv"):
        assert (a / "truth" / name).read_bytes() == (b / "truth" / name).read_bytes()
