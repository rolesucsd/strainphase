"""Real strain assemblies in, ground truth out.

No mutations are simulated. A strain group is a set of closely related
assemblies of the same species; one of them is designated the reference and the
rest are the strains present in the mixture. Their true haplotypes come from
aligning each assembly to that reference and reading the differences — real
strain variation, including whatever indel and repeat structure the organisms
actually have.

The alignment is ``minimap2 -cx asm5`` through the ``mappy`` bindings, and the
variants are read from minimap2's ``cs`` tag. ``paftools.js call`` on the same
alignment is the equivalent standard route; the ``cs`` parse is used here only
because it avoids a ``k8`` dependency, and it produces the same left-anchored
VCF-convention records.

Designating the reference by drawing from the group is deliberate. Every real
analysis phases against an assembly that is itself one strain's genome (or a MAG
close to one), so the reference is never equidistant from the members — one
strain always matches it exactly. Picking a "neutral" consensus reference would
be a more flattering setup than anything that happens in practice.
"""

from __future__ import annotations

import gzip
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FASTA_SUFFIXES = (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")


def read_fasta(path: str | Path) -> dict[str, str]:
    """Read a FASTA into ``{contig_name: sequence}``. Handles gzip."""
    contigs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    contigs[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.upper())
    if name is not None:
        contigs[name] = "".join(chunks)
    return contigs


def write_fasta(path: str | Path, contigs: dict[str, str], width: int = 60) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for name, seq in contigs.items():
            handle.write(f">{name}\n")
            for i in range(0, len(seq), width):
                handle.write(seq[i : i + width] + "\n")


@dataclass
class StrainGroup:
    """A set of closely related assemblies of one species."""

    name: str
    assemblies: dict[str, Path]  # strain_id -> FASTA path
    reference_id: str = ""
    #: ``strain_id -> {contig -> {pos: (ref_allele, alt_allele)}}`` in reference
    #: coordinates, 1-based, VCF left-anchoring.
    variants: dict[str, dict[str, dict[int, tuple[str, str]]]] = field(default_factory=dict)

    @property
    def strain_ids(self) -> list[str]:
        return sorted(self.assemblies)

    @property
    def reference_path(self) -> Path:
        return self.assemblies[self.reference_id]


def discover_assemblies(pattern_or_dir: str | Path) -> dict[str, Path]:
    """Find assemblies from a directory or a glob. Strain id is the file stem."""
    target = Path(pattern_or_dir)
    if target.is_dir():
        paths = [p for p in sorted(target.iterdir()) if _is_fasta(p)]
    else:
        paths = sorted(Path().glob(str(pattern_or_dir)))
        paths = [p for p in paths if _is_fasta(p)]
    if not paths:
        raise FileNotFoundError(f"no FASTA assemblies found at {pattern_or_dir!r}")
    return {_strain_id(p): p for p in paths}


def _is_fasta(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and any(name.endswith(s) for s in FASTA_SUFFIXES)


def _strain_id(path: Path) -> str:
    name = path.name
    for suffix in sorted(FASTA_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def choose_reference(strain_ids: list[str], rng: np.random.Generator) -> str:
    """Pick one assembly to serve as the reference, reproducibly."""
    return str(rng.choice(sorted(strain_ids)))


# --------------------------------------------------------------------------- #
# Variant extraction from assembly-to-reference alignment
# --------------------------------------------------------------------------- #

_CS_TOKEN = re.compile(r"([:=*+\-~])([A-Za-z0-9]+)")


def parse_cs(cs: str, ref_start: int) -> list[tuple[int, str, str]]:
    """Turn a minimap2 ``cs`` tag into ``(pos, ref_allele, alt_allele)`` records.

    ``pos`` is 1-based on the reference. SNVs are single-base REF/ALT. Indels are
    returned with an empty anchor here and are left-anchored by the caller, which
    is the only place the reference sequence is available.

    ``cs`` operators: ``:N`` identical run of length N, ``=SEQ`` identical run
    spelled out, ``*ab`` substitution ref a to alt b, ``+seq`` insertion to the
    reference, ``-seq`` deletion from the reference, ``~`` intron (not expected
    for bacterial assembly alignment, skipped).
    """
    records: list[tuple[int, str, str]] = []
    pos = ref_start  # 0-based cursor on the reference

    for op, value in _CS_TOKEN.findall(cs):
        if op == ":":
            pos += int(value)
        elif op == "=":
            pos += len(value)
        elif op == "*":
            # value is two bases: ref then alt, lowercase in the cs spec.
            if len(value) >= 2:
                records.append((pos + 1, value[0].upper(), value[1].upper()))
            pos += 1
        elif op == "+":
            # Insertion sits between reference bases; anchor is the base before.
            records.append((pos, "", value.upper()))
        elif op == "-":
            # Deletion of `value` starting at pos+1.
            records.append((pos, value.upper(), ""))
            pos += len(value)
        elif op == "~":
            logger.debug("skipping intron operator in cs tag")
    return records


def _left_anchor(
    records: list[tuple[int, str, str]], reference: str
) -> dict[int, tuple[str, str]]:
    """Convert raw cs records into VCF-convention REF/ALT keyed by 1-based pos."""
    variants: dict[int, tuple[str, str]] = {}
    for pos, ref_allele, alt_allele in records:
        if ref_allele and alt_allele:
            variants[pos] = (ref_allele, alt_allele)  # SNV
            continue
        if pos < 1 or pos > len(reference):
            continue
        anchor = reference[pos - 1]
        if anchor not in "ACGT":
            continue
        if alt_allele and not ref_allele:  # insertion
            variants[pos] = (anchor, anchor + alt_allele)
        elif ref_allele and not alt_allele:  # deletion
            variants[pos] = (anchor + ref_allele, anchor)
    return variants


def derive_variants(
    reference_contigs: dict[str, str],
    query_path: Path,
    preset: str = "asm5",
    min_alignment_length: int = 1000,
) -> dict[str, dict[int, tuple[str, str]]]:
    """Align one assembly to the reference and return its variants.

    Only primary alignments longer than ``min_alignment_length`` are used.
    Regions of the reference that no query alignment covers simply produce no
    variants — the strain is treated as having no call there rather than as
    matching the reference, which matters for accessory genome and for large
    rearrangements that asm5 will not span.
    """
    import mappy

    by_contig: dict[str, dict[int, tuple[str, str]]] = {}
    query = read_fasta(query_path)

    for ref_name, ref_seq in reference_contigs.items():
        aligner = mappy.Aligner(seq=ref_seq, preset=preset, n_threads=1)
        if not aligner:
            raise RuntimeError(f"failed to index reference contig {ref_name}")

        records: list[tuple[int, str, str]] = []
        for _query_name, query_seq in query.items():
            for hit in aligner.map(query_seq, cs=True):
                if not hit.is_primary or hit.blen < min_alignment_length:
                    continue
                records.extend(parse_cs(hit.cs, hit.r_st))

        if records:
            by_contig[ref_name] = _left_anchor(records, ref_seq)

    return by_contig


def build_group(
    name: str,
    assemblies: dict[str, Path],
    rng: np.random.Generator,
    preset: str = "asm5",
) -> StrainGroup:
    """Choose a reference and derive every other strain's true haplotype."""
    group = StrainGroup(name=name, assemblies=assemblies)
    group.reference_id = choose_reference(group.strain_ids, rng)
    reference_contigs = read_fasta(group.reference_path)

    logger.info(
        "group %s: %d strains, reference = %s (%d contigs, %d bp)",
        name,
        len(assemblies),
        group.reference_id,
        len(reference_contigs),
        sum(len(s) for s in reference_contigs.values()),
    )

    for strain_id in group.strain_ids:
        if strain_id == group.reference_id:
            # The reference strain carries the reference allele everywhere by
            # definition. Its haplotype is filled in once the union of variant
            # sites across the group is known.
            group.variants[strain_id] = {}
            continue
        group.variants[strain_id] = derive_variants(
            reference_contigs, assemblies[strain_id], preset=preset
        )
        n_variants = sum(len(v) for v in group.variants[strain_id].values())
        logger.info("  %s: %d variants vs reference", strain_id, n_variants)

    return group


def union_sites(group: StrainGroup) -> dict[str, dict[int, tuple[str, str]]]:
    """Every position polymorphic across the group, with its REF/ALT.

    Where two strains carry different alternates at the same position the first
    seen wins and the other strain is recorded as having no call there; that is
    rare between close relatives and pretending otherwise would silently
    misreport one of them.
    """
    sites: dict[str, dict[int, tuple[str, str]]] = {}
    for strain_id in sorted(group.variants):
        for contig, variants in group.variants[strain_id].items():
            bucket = sites.setdefault(contig, {})
            for pos, (ref_allele, alt_allele) in variants.items():
                bucket.setdefault(pos, (ref_allele, alt_allele))
    return sites


def haplotype_alleles(
    group: StrainGroup, sites: dict[str, dict[int, tuple[str, str]]]
) -> dict[str, dict[str, dict[int, str]]]:
    """``strain_id -> contig -> {pos: allele}`` over the union sites.

    A strain gets the ALT allele where its own variant matches the site's ALT,
    the REF allele where it has no variant there, and no call where it carries a
    *different* alternate — which would otherwise be scored as if it matched the
    reference.
    """
    out: dict[str, dict[str, dict[int, str]]] = {}
    for strain_id in group.strain_ids:
        own = group.variants.get(strain_id, {})
        per_contig: dict[str, dict[int, str]] = {}
        for contig, contig_sites in sites.items():
            alleles: dict[int, str] = {}
            own_contig = own.get(contig, {})
            for pos, (ref_allele, alt_allele) in contig_sites.items():
                mine = own_contig.get(pos)
                if mine is None:
                    alleles[pos] = ref_allele
                elif mine[1] == alt_allele and mine[0] == ref_allele:
                    alleles[pos] = alt_allele
                # else: a third allele at this site; leave uncalled
            per_contig[contig] = alleles
        out[strain_id] = per_contig
    return out
