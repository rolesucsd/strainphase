"""End-to-end checks on the simulator.

Everything the benchmark reports rests on the simulated data being internally
consistent: the reads in the BAM must actually carry the alleles the truth table
says their strain has, the VCF must describe the same variants, and two runs
with the same seed must produce the same thing. A silent inconsistency here
would look like a tool failure in every result table, so these are the most
load-bearing tests in the suite.
"""

from __future__ import annotations

import json

import pytest

from spbench.dataset import Dataset
from spbench.formats import Truth
from spbench.reads import load_sites, read_alleles
from spbench.simulate import SimConfig, simulate

pysam = pytest.importorskip("pysam")


def _tiny(**overrides) -> SimConfig:
    base = {
        "name": "tiny",
        "seed": 7,
        "contig_length": 20_000,
        "n_strains": 3,
        "n_mutations": 120,
        "n_timepoints": 2,
        "coverage": 20,
        "mean_read_length": 4_000,
        "read_length_sd": 800,
        "min_read_length": 1_500,
        "max_read_length": 8_000,
    }
    base.update(overrides)
    return SimConfig(**base)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Dataset:
    root = tmp_path_factory.mktemp("sim") / "tiny"
    simulate(_tiny(), root)
    return Dataset.load(root)


def test_expected_files_exist(dataset: Dataset):
    assert dataset.reference.exists()
    for sample in dataset.samples:
        assert dataset.bams[sample].exists()
        assert dataset.bams[sample].with_suffix(".bam.bai").exists()
        assert dataset.vcfs[sample].exists()
    assert dataset.union_vcf.exists()
    for name in ("sites.tsv", "strains.tsv", "abundance.tsv", "read_origins.tsv"):
        assert (dataset.truth_dir / name).exists()


def test_abundances_sum_to_one_per_timepoint(dataset: Dataset):
    truth = Truth.read(dataset.truth_dir)
    for sample in dataset.samples:
        total = sum(v for (s, _), v in truth.abundance.items() if s == sample)
        assert total == pytest.approx(1.0, abs=1e-6)


def test_reads_carry_their_strain_alleles(dataset: Dataset):
    """The core consistency check.

    For every read, compare the alleles observed in the BAM against the truth
    haplotype of the strain the read came from. Agreement must be near-perfect;
    the small shortfall is exactly the simulated sequencing error, so the test
    bounds it rather than requiring 100%.
    """
    truth = Truth.read(dataset.truth_dir)
    contig = next(iter(dataset.contigs))
    sample = dataset.samples[0]

    sites = load_sites(str(dataset.vcfs[sample]), contig)
    assert sites, "simulated VCF has no sites"
    observed = read_alleles(str(dataset.bams[sample]), contig, sites)
    assert observed, "no reads produced allele calls"

    agree = 0
    total = 0
    for read_id, calls in observed.items():
        strain_id = truth.read_origins.get((sample, read_id))
        if strain_id is None:
            continue
        expected = truth.strains[strain_id].alleles
        for pos, (allele, _quality) in calls.items():
            if pos not in expected:
                continue  # false-positive call site: no truth allele to compare
            total += 1
            agree += allele == expected[pos]

    assert total > 100, "not enough comparable calls to be meaningful"
    # Simulated error rate is 1e-3, so >99% agreement is required; anything
    # materially below that means the CIGAR walk or the truth table is wrong.
    assert agree / total > 0.99


def test_vcf_records_match_truth_sites(dataset: Dataset):
    truth = Truth.read(dataset.truth_dir)
    contig = next(iter(dataset.contigs))
    truth_sites = truth.sites[contig]

    with pysam.VariantFile(str(dataset.vcfs[dataset.samples[0]])) as vcf:
        called = {rec.pos: (rec.ref, rec.alts[0]) for rec in vcf.fetch(contig)}

    # Every called site that is a true site must agree on REF/ALT. Sites present
    # in truth but absent here are the simulated caller's false negatives, which
    # is the intended behaviour, not a failure.
    overlap = 0
    for pos, (ref, alt) in called.items():
        if pos in truth_sites:
            overlap += 1
            assert (ref, alt[0] if isinstance(alt, tuple) else alt) == (
                truth_sites[pos][0],
                truth_sites[pos][1][0],
            )
    assert overlap > 0.5 * len(called), "most called sites should be real"


def test_rare_strain_is_present_but_shallow(dataset: Dataset):
    """The rescue scenario must actually be set up: one strain that blooms at a
    single timepoint and sits at low depth at the others."""
    truth = Truth.read(dataset.truth_dir)
    trajectories: dict[str, list[float]] = {}
    for (sample, strain_id), value in truth.abundance.items():
        trajectories.setdefault(strain_id, []).append(value)

    config = _tiny()
    trough = config.resolved_rare_abundance
    assert any(
        min(values) <= trough * 1.5 and max(values) >= config.bloom_abundance * 0.8
        for values in trajectories.values()
    ), "no strain follows the rare-then-blooming trajectory"


def test_simulation_is_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    simulate(_tiny(name="det"), a)
    simulate(_tiny(name="det"), b)
    for name in ("sites.tsv", "strains.tsv", "abundance.tsv", "read_origins.tsv"):
        assert (a / "truth" / name).read_bytes() == (b / "truth" / name).read_bytes()
    assert (
        json.loads((a / "manifest.json").read_text())["config_fingerprint"]
        == json.loads((b / "manifest.json").read_text())["config_fingerprint"]
    )


def test_indels_produce_cigar_operations(tmp_path):
    """With indel variants enabled, the BAM must contain I/D operations and the
    allele extractor must resolve them - otherwise the indel axis silently tests
    nothing."""
    root = tmp_path / "indel"
    simulate(_tiny(name="indel", indel_fraction=0.3, n_mutations=200), root)
    dataset = Dataset.load(root)
    contig = next(iter(dataset.contigs))
    sample = dataset.samples[0]

    ops = set()
    with pysam.AlignmentFile(str(dataset.bams[sample]), "rb") as bam:
        for aln in bam.fetch(contig):
            ops.update(op for op, _length in aln.cigartuples or [])
    assert 1 in ops or 2 in ops, "no insertion or deletion CIGAR operations emitted"

    sites = load_sites(str(dataset.vcfs[sample]), contig)
    indel_sites = {
        pos for pos, (ref, alt, kind) in sites.items() if kind in ("del", "ins")
    }
    assert indel_sites, "no indel records in the simulated VCF"

    observed = read_alleles(str(dataset.bams[sample]), contig, sites)
    calls_at_indels = sum(
        1 for calls in observed.values() for pos in calls if pos in indel_sites
    )
    assert calls_at_indels > 0, "indel sites produced no read-level calls"
