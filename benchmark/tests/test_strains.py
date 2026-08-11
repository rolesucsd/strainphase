"""Tests for deriving ground truth from real assemblies.

The truth tables are now read off a `minimap2 -cx asm5` alignment of each
assembly against the designated reference, so a bug here silently mis-scores
every tool in the same direction. These tests build small assemblies with known
differences and check the recovered variants are exactly those differences.
"""

from __future__ import annotations

import numpy as np
import pytest

from spbench.strains import (
    build_group,
    choose_reference,
    derive_variants,
    discover_assemblies,
    haplotype_alleles,
    parse_cs,
    read_fasta,
    union_sites,
    write_fasta,
)

pytest.importorskip("mappy")


def _backbone(seed: int = 0, length: int = 40_000) -> str:
    """A pseudo-random but reproducible backbone.

    Real assemblies are the benchmark's input; these fixtures exist only to test
    the variant-derivation arithmetic, which does not care about composition.
    """
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=length).tolist())


def _apply(seq: str, snvs: dict[int, str], deletions=(), insertions=()) -> str:
    """Build a variant genome. Positions are 1-based on ``seq``."""
    chars = list(seq)
    for pos, base in snvs.items():
        chars[pos - 1] = base
    out = "".join(chars)
    # Apply from the right so earlier coordinates stay valid.
    for pos, length in sorted(deletions, reverse=True):
        out = out[:pos] + out[pos + length :]
    for pos, inserted in sorted(insertions, reverse=True):
        out = out[:pos] + inserted + out[pos:]
    return out


@pytest.fixture
def group_dir(tmp_path):
    """Three 'strains': the reference, one with SNVs, one with SNVs and indels."""
    backbone = _backbone()

    def mutate(positions: list[int]) -> dict[int, str]:
        """Pick, at each position, a base that actually differs from the
        backbone — otherwise the fixture would silently test nothing."""
        return {
            pos: next(b for b in "ACGT" if b != backbone[pos - 1]) for pos in positions
        }

    snvs_a = mutate([5000, 12000, 23000, 31000])
    snvs_b = mutate([5000, 17000, 28000])

    write_fasta(tmp_path / "strain_ref.fasta", {"chr": backbone})
    write_fasta(tmp_path / "strain_a.fasta", {"chr": _apply(backbone, snvs_a)})
    write_fasta(
        tmp_path / "strain_b.fasta",
        {"chr": _apply(backbone, snvs_b, deletions=[(20000, 6)], insertions=[(9000, "ACGTAC")])},
    )
    return tmp_path, backbone, snvs_a, snvs_b


# --------------------------------------------------------------------------- #
# cs tag parsing
# --------------------------------------------------------------------------- #


def test_parse_cs_substitution():
    assert parse_cs(":100*ac:50", 0) == [(101, "A", "C")]


def test_parse_cs_insertion_and_deletion():
    records = parse_cs(":10+acgt:10-tt:5", 0)
    assert (10, "", "ACGT") in records
    assert (20, "TT", "") in records


def test_parse_cs_respects_alignment_start():
    assert parse_cs(":5*ac", 1000) == [(1006, "A", "C")]


def test_parse_cs_handles_spelled_out_matches():
    assert parse_cs("=ACGT*ac", 0) == [(5, "A", "C")]


# --------------------------------------------------------------------------- #
# Variant derivation
# --------------------------------------------------------------------------- #


def test_snvs_are_recovered_exactly(group_dir):
    tmp_path, backbone, snvs_a, _ = group_dir
    variants = derive_variants({"chr": backbone}, tmp_path / "strain_a.fasta")
    recovered = variants["chr"]
    for pos, alt in snvs_a.items():
        assert pos in recovered, f"missed SNV at {pos}"
        assert recovered[pos] == (backbone[pos - 1], alt)
    assert len(recovered) == len(snvs_a), "spurious variants recovered"


def test_indels_are_left_anchored(group_dir):
    tmp_path, backbone, _, _ = group_dir
    variants = derive_variants({"chr": backbone}, tmp_path / "strain_b.fasta")["chr"]

    indels = {p: v for p, v in variants.items() if len(v[0]) != len(v[1])}
    assert indels, "no indels recovered"
    for pos, (ref, alt) in indels.items():
        # VCF left-anchoring: both alleles start with the reference base at pos.
        assert ref[0] == alt[0] == backbone[pos - 1]
        assert len(ref) != len(alt)


def test_reference_strain_has_no_variants(group_dir):
    tmp_path, backbone, _, _ = group_dir
    variants = derive_variants({"chr": backbone}, tmp_path / "strain_ref.fasta")
    assert sum(len(v) for v in variants.values()) == 0


# --------------------------------------------------------------------------- #
# Group assembly
# --------------------------------------------------------------------------- #


def test_discover_and_build_group(group_dir):
    tmp_path, backbone, snvs_a, snvs_b = group_dir
    assemblies = discover_assemblies(tmp_path)
    assert set(assemblies) == {"strain_ref", "strain_a", "strain_b"}

    rng = np.random.default_rng(0)
    group = build_group("test", assemblies, rng)
    assert group.reference_id in assemblies
    assert group.variants[group.reference_id] == {}

    sites = union_sites(group)
    assert sites, "no polymorphic sites found across the group"


def test_reference_choice_is_seeded(group_dir):
    tmp_path, _, _, _ = group_dir
    ids = sorted(discover_assemblies(tmp_path))
    a = choose_reference(ids, np.random.default_rng(42))
    b = choose_reference(ids, np.random.default_rng(42))
    assert a == b
    assert a in ids


def test_haplotypes_agree_with_the_reference_where_a_strain_has_no_variant(group_dir):
    """A strain with no variant at a site must be scored as carrying REF there,
    not as having no call — otherwise every shared site drops out of the
    comparison and the metrics see far less evidence than exists."""
    tmp_path, backbone, _, _ = group_dir
    assemblies = discover_assemblies(tmp_path)

    # Force strain_ref to be the reference so the expected alleles are known.
    rng = np.random.default_rng(0)
    group = build_group("test", assemblies, rng)
    group.reference_id = "strain_ref"
    group.variants["strain_ref"] = {}
    group.variants["strain_a"] = derive_variants(
        {"chr": backbone}, tmp_path / "strain_a.fasta"
    )
    group.variants["strain_b"] = derive_variants(
        {"chr": backbone}, tmp_path / "strain_b.fasta"
    )

    sites = union_sites(group)
    alleles = haplotype_alleles(group, sites)

    ref_alleles = alleles["strain_ref"]["chr"]
    a_alleles = alleles["strain_a"]["chr"]
    assert len(ref_alleles) == len(sites["chr"])
    # The reference strain carries REF at every site by definition.
    for pos, (ref, _alt) in sites["chr"].items():
        assert ref_alleles[pos] == ref
    # And strain_a differs from it at its own variants.
    differing = [p for p in sites["chr"] if a_alleles.get(p) != ref_alleles[p]]
    assert differing, "strain_a should differ from the reference somewhere"


def test_fasta_round_trip(tmp_path):
    contigs = {"c1": "ACGT" * 100, "c2": "GGGGTTTT" * 50}
    write_fasta(tmp_path / "x.fasta", contigs)
    assert read_fasta(tmp_path / "x.fasta") == contigs
