"""Ground-truth simulator for longitudinal strain mixtures.

Produces a complete, self-contained dataset: a reference FASTA, one sorted and
indexed BAM per timepoint, one VCF per timepoint, and the truth tables defined
in :mod:`spbench.formats`.

Design notes that matter for the comparison being fair
------------------------------------------------------

*Strains are related by a tree, not drawn independently.* Independently drawn
haplotypes are pairwise equidistant, which is the easy case for every clustering
method. Real strain mixtures contain closely related pairs nested inside more
distant clades, and that nesting is what separates methods. Mutations are placed
along the branches of a random tree, so the resulting mixture contains both easy
and hard pairs.

*Base qualities are calibrated, not decorative.* Each base gets an error
probability drawn from ``Uniform(0, 2 * error_rate)``, its quality is
``-10 log10`` of that probability, and the base is then corrupted with exactly
that probability. So ``Q`` means what it says. This matters because several
methods here - strainphase among them - weight evidence by base quality, and a
simulator that emitted a flat ``Q40`` for every base would hand them an
advantage they would not have on real data.

*Variant calls are simulated, not handed over.* By default the VCF fed to the
tools is a *called* VCF derived from the simulated reads (a site is called only
if enough reads actually carry the alt allele), not the truth VCF. Handing every
tool the exact truth site list removes the low-abundance detection problem,
which is the problem this benchmark is most interested in. ``--vcf-mode truth``
is available for ablations where you want site selection held fixed.

*Sequencing error is substitution-only.* HiFi indel error is dominated by
homopolymer slippage, and its interaction with aligner placement cannot be
modelled honestly without running a real aligner. Simulating it badly would be
worse than not simulating it. Germline indel *variants* are simulated (see
``--indel-fraction``); indel *errors* are not. This is stated in the report.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from spbench.formats import (
    ABUNDANCE_COLUMNS,
    READ_ORIGINS_COLUMNS,
    SITES_COLUMNS,
    STRAINS_COLUMNS,
    encode_alleles,
    write_table,
)

logger = logging.getLogger(__name__)

BASES = ("A", "C", "G", "T")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class SimConfig:
    """Everything that determines a dataset. Two runs with equal configs and
    equal seeds produce byte-identical outputs."""

    name: str = "dataset"
    seed: int = 0

    # Reference
    reference_fasta: str | None = None  # None -> generate a random reference
    contig_length: int = 200_000
    n_contigs: int = 1

    # Strain mixture
    n_strains: int = 4
    n_mutations: int = 2_000  # total mutations placed across the tree
    indel_fraction: float = 0.0  # fraction of mutations that are indels
    max_indel_length: int = 12

    # Longitudinal structure
    n_timepoints: int = 4
    include_sweep: bool = True
    include_rescue_strain: bool = True
    #: Trough coverage of the rescue strain, in x. This is specified as an
    #: absolute depth rather than as an abundance fraction on purpose. A fixed
    #: 1% trough means 0.1x at 10x sequencing and 0.6x at 60x - in the first
    #: case no method can recover the strain because the reads do not exist, so
    #: the "rare strain" test degenerates into a test of nothing. Pinning the
    #: trough to a small but non-zero depth keeps the scenario meaningful at
    #: every coverage: enough reads to be rescued from a neighbouring timepoint,
    #: too few to be discovered de novo within one.
    rescue_trough_coverage: float = 1.5
    #: Explicit trough abundance. Overrides ``rescue_trough_coverage`` when set.
    rare_abundance: float | None = None
    bloom_abundance: float = 0.30  # peak abundance of the rescue strain

    # Sequencing
    coverage: float = 30.0
    mean_read_length: int = 15_000
    read_length_sd: int = 4_000
    min_read_length: int = 4_000
    max_read_length: int = 30_000
    error_rate: float = 0.001
    mapq: int = 60

    # Read realism. See spbench/hifi.py for what each path does and does not
    # model.
    #
    #   read_model "exact"  - substitution-only error, reads emitted at their
    #                         true coordinates with exact CIGARs. Fast, needs no
    #                         extra packages, deterministic. Right for CI and for
    #                         developing metrics. Not what HiFi data looks like.
    #   read_model "hifi"   - homopolymer-aware indel error plus substitutions,
    #                         reads reverse-complemented half the time and
    #                         aligned back to the reference with minimap2. This
    #                         is the path for numbers that go in a paper.
    #
    # "hifi" requires minimap2 alignment: the whole point of simulating
    # homopolymer slippage is that an aligner has to place it, and handing over
    # the true CIGAR would remove exactly the difficulty being tested. The two
    # settings are validated against each other below.
    read_model: str = "exact"
    aligner: str = "exact"  # "exact" | "minimap2"
    minimap2_preset: str = "map-hifi"
    hifi_substitution_fraction: float = 0.35
    hifi_homopolymer_min_length: int = 4
    hifi_homopolymer_exponent: float = 2.0
    hifi_quality_penalty: float = 4.0
    hifi_max_indel_error_length: int = 2

    # Variant calling simulation
    vcf_mode: str = "called"  # "called" | "truth"
    call_min_alt_reads: int = 3
    call_min_af: float = 0.02
    call_fp_rate: float = 2e-5  # false-positive sites per bp per sample

    def __post_init__(self) -> None:
        if self.read_model not in ("exact", "hifi"):
            raise ValueError(f"read_model must be 'exact' or 'hifi', got {self.read_model!r}")
        if self.aligner not in ("exact", "minimap2"):
            raise ValueError(f"aligner must be 'exact' or 'minimap2', got {self.aligner!r}")
        if self.read_model == "hifi" and self.aligner != "minimap2":
            raise ValueError(
                "read_model='hifi' requires aligner='minimap2'. Simulating "
                "homopolymer indel error and then handing over the true CIGAR "
                "would remove the placement ambiguity that is the entire reason "
                "to simulate it."
            )
        if self.vcf_mode not in ("called", "truth"):
            raise ValueError(f"vcf_mode must be 'called' or 'truth', got {self.vcf_mode!r}")

    @property
    def resolved_rare_abundance(self) -> float:
        """Trough abundance of the rescue strain, from depth unless overridden."""
        if self.rare_abundance is not None:
            return self.rare_abundance
        if self.coverage <= 0:
            return 0.01
        return min(0.10, max(1e-4, self.rescue_trough_coverage / self.coverage))

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Variant model
# --------------------------------------------------------------------------- #


@dataclass
class Variant:
    """One germline difference relative to the reference, in VCF convention.

    ``pos`` is the 1-based VCF anchor position. For SNVs ``ref``/``alt`` are
    single bases at that position. For indels the anchor base is included in
    both, following VCF left-anchoring, so a 3 bp deletion at 100 is
    ``pos=100, ref="ACGT", alt="A"`` and deletes reference bases 101-103.
    """

    pos: int
    kind: str  # "snv" | "del" | "ins"
    ref: str
    alt: str

    @property
    def deleted_span(self) -> tuple[int, int] | None:
        """Inclusive 1-based reference footprint removed by a deletion."""
        if self.kind != "del":
            return None
        return (self.pos + 1, self.pos + len(self.ref) - len(self.alt))


@dataclass
class Strain:
    """A simulated strain: its variants keyed by anchor position."""

    strain_id: str
    parent: str | None
    variants: dict[int, Variant] = field(default_factory=dict)
    deleted: set[int] = field(default_factory=set)  # reference positions removed

    def copy_as(self, strain_id: str) -> Strain:
        return Strain(
            strain_id=strain_id,
            parent=self.strain_id,
            variants=dict(self.variants),
            deleted=set(self.deleted),
        )


# --------------------------------------------------------------------------- #
# Reference handling
# --------------------------------------------------------------------------- #


def _random_reference(rng: np.random.Generator, length: int, n_contigs: int) -> dict[str, str]:
    """Generate a random reference. Used by the smoke tier so that the fastest
    benchmark tier requires no downloads at all."""
    contigs = {}
    for i in range(n_contigs):
        arr = rng.choice(np.array(BASES), size=length)
        contigs[f"sim_contig_{i + 1}"] = "".join(arr.tolist())
    return contigs


def _read_fasta(path: str | Path) -> dict[str, str]:
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


def _write_fasta(path: Path, contigs: dict[str, str], width: int = 60) -> None:
    with open(path, "w") as handle:
        for name, seq in contigs.items():
            handle.write(f">{name}\n")
            for i in range(0, len(seq), width):
                handle.write(seq[i : i + width] + "\n")


# --------------------------------------------------------------------------- #
# Strain tree
# --------------------------------------------------------------------------- #


def build_strain_tree(
    rng: np.random.Generator,
    config: SimConfig,
    contigs: dict[str, str],
) -> tuple[list[Strain], dict[str, dict[int, Variant]]]:
    """Place ``config.n_mutations`` mutations along a random tree of strains.

    Returns the strains (strain 0 is the reference itself) and, for
    bookkeeping, the per-contig union of all variants ever placed.
    """
    contig_names = list(contigs)
    strains: list[Strain] = [Strain(strain_id="strain_0", parent=None)]

    # Random branch lengths: an exponential draw per strain gives a spread of
    # close and distant relatives rather than a uniform star topology.
    branch_lengths = rng.exponential(scale=1.0, size=max(0, config.n_strains - 1))
    total_branch = float(branch_lengths.sum()) or 1.0
    mutations_per_branch = [
        max(1, int(round(config.n_mutations * bl / total_branch))) for bl in branch_lengths
    ]

    all_variants: dict[str, dict[int, Variant]] = {name: {} for name in contig_names}
    guard = config.max_indel_length + 2

    for idx in range(1, config.n_strains):
        parent = strains[int(rng.integers(0, len(strains)))]
        child = parent.copy_as(f"strain_{idx}")

        for _ in range(mutations_per_branch[idx - 1]):
            contig = contig_names[int(rng.integers(0, len(contig_names)))]
            seq = contigs[contig]
            is_indel = config.indel_fraction > 0 and rng.random() < config.indel_fraction

            # Anchor away from contig edges so indels always have room.
            pos = int(rng.integers(guard + 1, len(seq) - guard))

            # Never mutate inside a region this lineage has already deleted -
            # the variant would be unobservable and the truth table would lie.
            if pos in child.deleted:
                continue
            if is_indel and any(p in child.deleted for p in range(pos, pos + guard)):
                continue
            # Indels need a clear neighbourhood: overlapping germline indels are
            # a normalisation problem, not a phasing problem, and simulating
            # them here would only test VCF handling.
            if is_indel and any(
                p in child.variants for p in range(pos - guard, pos + guard + 1)
            ):
                continue

            variant = _make_variant(rng, config, child, contig, seq, pos, is_indel)
            if variant is None:
                continue

            child.variants[variant.pos] = variant
            span = variant.deleted_span
            if span is not None:
                child.deleted.update(range(span[0], span[1] + 1))
            all_variants[contig][variant.pos] = variant

        strains.append(child)

    return strains, all_variants


def _make_variant(
    rng: np.random.Generator,
    config: SimConfig,
    strain: Strain,
    contig: str,
    seq: str,
    pos: int,
    is_indel: bool,
) -> Variant | None:
    """Build one variant at ``pos`` (1-based) on ``seq``, or None if impossible."""
    ref_base = seq[pos - 1]
    if ref_base not in BASES:
        return None

    if not is_indel:
        # Mutate away from whatever this lineage currently carries, so a
        # back-mutation to the parent allele never silently no-ops.
        current = strain.variants.get(pos)
        current_base = current.alt if current is not None and current.kind == "snv" else ref_base
        options = [b for b in BASES if b != current_base]
        return Variant(pos=pos, kind="snv", ref=ref_base, alt=str(rng.choice(options)))

    length = int(rng.integers(1, config.max_indel_length + 1))
    if rng.random() < 0.5:
        deleted = seq[pos : pos + length]
        if any(b not in BASES for b in deleted):
            return None
        return Variant(pos=pos, kind="del", ref=ref_base + deleted, alt=ref_base)

    inserted = "".join(str(b) for b in rng.choice(np.array(BASES), size=length))
    return Variant(pos=pos, kind="ins", ref=ref_base, alt=ref_base + inserted)


# --------------------------------------------------------------------------- #
# Abundance trajectories
# --------------------------------------------------------------------------- #


def build_abundances(
    rng: np.random.Generator, config: SimConfig, strain_ids: list[str]
) -> dict[tuple[str, str], float]:
    """Abundance of every strain at every timepoint, normalised per timepoint.

    The trajectories are not decoration. Two of them encode the scenarios the
    longitudinal claim rests on:

    * a **sweep** - one strain rises while another falls, testing whether a
      method tracks identity through a large abundance change;
    * a **rescue strain** - present the whole time but sitting near or below the
      single-sample detection floor at most timepoints and blooming at one.
      Single-sample methods should miss it at its trough by construction. That
      is the point of including it, and the report says so.
    """
    samples = [f"T{i + 1}" for i in range(config.n_timepoints)]
    n = len(strain_ids)
    abundance: dict[tuple[str, str], float] = {}

    rescue_idx = -1
    if config.include_rescue_strain and n >= 3:
        rescue_idx = n - 1
    bloom_at = int(rng.integers(0, config.n_timepoints)) if rescue_idx >= 0 else -1

    # Baseline: a smooth log-space random walk per strain.
    log_levels = rng.normal(loc=0.0, scale=0.8, size=n)
    raw = np.zeros((n, config.n_timepoints))
    for t in range(config.n_timepoints):
        log_levels = 0.75 * log_levels + rng.normal(0.0, 0.35, size=n)
        raw[:, t] = np.exp(log_levels)

    if config.include_sweep and n >= 2:
        frac = np.linspace(0.0, 1.0, config.n_timepoints)
        raw[0, :] = np.exp(np.log(3.0) + (np.log(0.3) - np.log(3.0)) * frac)
        raw[1, :] = np.exp(np.log(0.3) + (np.log(3.0) - np.log(0.3)) * frac)

    for t, sample in enumerate(samples):
        column = raw[:, t].copy()
        if rescue_idx >= 0:
            column[rescue_idx] = 0.0
        total = column.sum()
        column = column / total if total > 0 else np.full(n, 1.0 / n)

        if rescue_idx >= 0:
            target = (
                config.bloom_abundance
                if t == bloom_at
                else config.resolved_rare_abundance
            )
            column *= 1.0 - target
            column[rescue_idx] = target

        for strain_id, value in zip(strain_ids, column, strict=True):
            abundance[(sample, strain_id)] = float(value)

    return abundance


# --------------------------------------------------------------------------- #
# Read simulation
# --------------------------------------------------------------------------- #


@dataclass
class SimulatedRead:
    read_id: str
    contig: str
    ref_start: int  # 0-based, BAM convention
    sequence: str
    qualities: list[int]
    cigar: list[tuple[int, int]]
    strain_id: str
    n_mismatches: int
    mapq: int = 60
    is_reverse: bool = False

    @property
    def ref_end(self) -> int:
        """0-based exclusive reference end. Soft clips consume no reference."""
        span = sum(length for op, length in self.cigar if op in (0, 2, 7, 8))
        return self.ref_start + span


def _build_read(
    rng: np.random.Generator,
    config: SimConfig,
    strain: Strain,
    contig: str,
    seq: str,
    ref_start: int,
    ref_end: int,
    read_id: str,
) -> SimulatedRead | None:
    """Walk the reference from ``ref_start`` to ``ref_end`` (0-based, half-open)
    applying the strain's variants, emitting sequence, qualities and CIGAR."""
    pieces: list[str] = []
    ops: list[tuple[int, int]] = []

    def push(op: int, length: int) -> None:
        if length <= 0:
            return
        if ops and ops[-1][0] == op:
            ops[-1] = (op, ops[-1][1] + length)
        else:
            ops.append((op, length))

    ref_pos = ref_start  # 0-based
    while ref_pos < ref_end:
        one_based = ref_pos + 1
        variant = strain.variants.get(one_based)

        if variant is None:
            pieces.append(seq[ref_pos])
            push(0, 1)
            ref_pos += 1
            continue

        if variant.kind == "snv":
            pieces.append(variant.alt)
            push(0, 1)
            ref_pos += 1
        elif variant.kind == "del":
            del_len = len(variant.ref) - len(variant.alt)
            if ref_pos + 1 + del_len > ref_end:
                # The deletion runs past the read's end; stop cleanly on the
                # anchor base rather than emitting a truncated D operation.
                pieces.append(variant.alt)
                push(0, 1)
                ref_pos += 1
                break
            pieces.append(variant.alt)
            push(0, 1)
            push(2, del_len)
            ref_pos += 1 + del_len
        else:  # insertion
            ins = variant.alt[len(variant.ref) :]
            pieces.append(variant.ref)
            push(0, len(variant.ref))
            pieces.append(ins)
            push(1, len(ins))
            ref_pos += len(variant.ref)

    sequence = "".join(pieces)
    if not sequence or not ops:
        return None

    # Calibrated quality + error model: draw the per-base error probability,
    # derive Q from it, then corrupt with exactly that probability.
    n = len(sequence)
    err_prob = rng.uniform(0.0, 2.0 * config.error_rate, size=n)
    err_prob = np.clip(err_prob, 1e-6, 0.5)
    quals = np.clip(np.round(-10.0 * np.log10(err_prob)), 2, 60).astype(int)
    corrupt = rng.random(n) < err_prob

    bases = list(sequence)
    n_mismatches = 0
    for i in np.nonzero(corrupt)[0]:
        original = bases[i]
        if original not in BASES:
            continue
        options = [b for b in BASES if b != original]
        bases[i] = str(rng.choice(options))
        n_mismatches += 1

    return SimulatedRead(
        read_id=read_id,
        contig=contig,
        ref_start=ref_start,
        sequence="".join(bases),
        qualities=quals.tolist(),
        cigar=ops,
        strain_id=strain.strain_id,
        n_mismatches=n_mismatches,
    )


def simulate_reads_for_sample(
    rng: np.random.Generator,
    config: SimConfig,
    sample: str,
    strains: list[Strain],
    contigs: dict[str, str],
    abundance: dict[tuple[str, str], float],
    aligner=None,
    strain_sequences: dict[tuple[str, str], str] | None = None,
) -> list[SimulatedRead]:
    """Simulate one timepoint's worth of reads across all contigs."""
    if config.read_model == "hifi":
        return _simulate_reads_hifi(
            rng, config, sample, strains, contigs, abundance, aligner, strain_sequences or {}
        )

    weights = np.array([abundance[(sample, s.strain_id)] for s in strains], dtype=float)
    weights = weights / weights.sum()

    reads: list[SimulatedRead] = []
    counter = 0
    for contig, seq in contigs.items():
        n_reads = int(round(config.coverage * len(seq) / config.mean_read_length))
        for _ in range(n_reads):
            length = int(
                np.clip(
                    rng.normal(config.mean_read_length, config.read_length_sd),
                    config.min_read_length,
                    config.max_read_length,
                )
            )
            length = min(length, len(seq))
            ref_start = int(rng.integers(0, max(1, len(seq) - length)))
            strain = strains[int(rng.choice(len(strains), p=weights))]
            counter += 1
            read = _build_read(
                rng,
                config,
                strain,
                contig,
                seq,
                ref_start,
                ref_start + length,
                f"{sample}_read_{counter:07d}",
            )
            if read is not None:
                reads.append(read)

    reads.sort(key=lambda r: (r.contig, r.ref_start))
    return reads


def _simulate_reads_hifi(
    rng: np.random.Generator,
    config: SimConfig,
    sample: str,
    strains: list[Strain],
    contigs: dict[str, str],
    abundance: dict[tuple[str, str], float],
    aligner,
    strain_sequences: dict[tuple[str, str], str],
) -> list[SimulatedRead]:
    """Realistic path: sample from strain genomes, corrupt, align back.

    Reads are drawn from the *strain's* coordinate system, given HiFi-shaped
    error, reverse-complemented half the time, and then aligned to the reference
    with minimap2. Their placement, CIGAR and mapping quality come from the
    aligner, not from the simulator - so indel placement ambiguity, soft
    clipping and mismapping are all present and are the same for every tool
    scored downstream.

    Reads that fail to align are dropped and counted. Losing a small percentage
    is what happens on real data.
    """
    from spbench.hifi import HiFiErrorModel, apply_hifi_errors, reverse_complement

    model = HiFiErrorModel(
        error_rate=config.error_rate,
        substitution_fraction=config.hifi_substitution_fraction,
        homopolymer_min_length=config.hifi_homopolymer_min_length,
        homopolymer_exponent=config.hifi_homopolymer_exponent,
        quality_penalty_in_homopolymer=config.hifi_quality_penalty,
        max_indel_error_length=config.hifi_max_indel_error_length,
    )

    weights = np.array([abundance[(sample, s.strain_id)] for s in strains], dtype=float)
    weights = weights / weights.sum()

    reads: list[SimulatedRead] = []
    counter = 0
    unaligned = 0

    for contig, ref_seq in contigs.items():
        n_reads = int(round(config.coverage * len(ref_seq) / config.mean_read_length))
        for _ in range(n_reads):
            strain = strains[int(rng.choice(len(strains), p=weights))]
            strain_seq = strain_sequences[(strain.strain_id, contig)]
            if len(strain_seq) < config.min_read_length:
                continue

            length = int(
                np.clip(
                    rng.normal(config.mean_read_length, config.read_length_sd),
                    config.min_read_length,
                    config.max_read_length,
                )
            )
            length = min(length, len(strain_seq))
            start = int(rng.integers(0, max(1, len(strain_seq) - length)))
            template = strain_seq[start : start + length]

            corrupted, quals = apply_hifi_errors(template, rng, model)
            if not corrupted:
                continue
            # Half the molecules are sequenced from the other strand. The
            # aligner recovers the orientation; nothing downstream is told it.
            if rng.random() < 0.5:
                corrupted = reverse_complement(corrupted)
                quals = list(reversed(quals))

            counter += 1
            read_id = f"{sample}_read_{counter:07d}"
            alignment = aligner.align(corrupted, quals)
            if alignment is None:
                unaligned += 1
                continue

            reads.append(
                SimulatedRead(
                    read_id=read_id,
                    contig=alignment.contig,
                    ref_start=alignment.ref_start,
                    sequence=alignment.sequence,
                    qualities=alignment.qualities,
                    cigar=alignment.cigar,
                    strain_id=strain.strain_id,
                    n_mismatches=alignment.nm,
                    mapq=alignment.mapq,
                    is_reverse=alignment.is_reverse,
                )
            )

    if unaligned:
        logger.info(
            "%s: %d/%d reads failed to align and were dropped (%.1f%%)",
            sample,
            unaligned,
            counter,
            100.0 * unaligned / max(1, counter),
        )

    reads.sort(key=lambda r: (r.contig, r.ref_start))
    return reads


# --------------------------------------------------------------------------- #
# BAM / VCF output
# --------------------------------------------------------------------------- #


def write_bam(path: Path, contigs: dict[str, str], reads: list[SimulatedRead]) -> None:
    import pysam

    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": len(seq)} for name, seq in contigs.items()],
        "RG": [{"ID": "sim", "SM": path.stem, "PL": "PACBIO"}],
    }
    order = {name: i for i, name in enumerate(contigs)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for read in reads:
            aln = pysam.AlignedSegment(out.header)
            aln.query_name = read.read_id
            aln.query_sequence = read.sequence
            aln.query_qualities = pysam.qualitystring_to_array(
                "".join(chr(q + 33) for q in read.qualities)
            )
            aln.flag = 16 if read.is_reverse else 0
            aln.reference_id = order[read.contig]
            aln.reference_start = read.ref_start
            aln.mapping_quality = read.mapq
            aln.cigartuples = read.cigar
            aln.next_reference_id = -1
            aln.next_reference_start = -1
            aln.template_length = 0
            aln.set_tag("NM", read.n_mismatches, value_type="i")
            aln.set_tag("RG", "sim", value_type="Z")
            out.write(aln)

    pysam.index(str(path))


def observe_site_support(
    reads: list[SimulatedRead],
    strains: list[Strain],
    contig: str,
    variant: Variant,
) -> tuple[int, int]:
    """Count (depth, alt-supporting reads) at a variant, from the simulated reads.

    Support is counted from the read's *strain of origin* rather than by
    re-genotyping the emitted sequence. Sequencing error moves individual bases
    but not a read's true haplotype, and the purpose here is to model which
    sites a caller would have the statistical power to call - which is driven by
    how many molecules carry the allele, not by per-base noise.
    """
    carriers = {
        s.strain_id
        for s in strains
        if (v := s.variants.get(variant.pos)) is not None
        and v.kind == variant.kind
        and v.alt == variant.alt
    }
    depth = 0
    alt = 0
    for read in reads:
        if read.contig != contig:
            continue
        if read.ref_start <= variant.pos - 1 < read.ref_end:
            depth += 1
            if read.strain_id in carriers:
                alt += 1
    return depth, alt


def write_vcf(
    path: Path,
    contigs: dict[str, str],
    sample: str,
    records: list[tuple[str, Variant, int, int]],
) -> None:
    """Write a bgzipped, tabix-indexed VCF. ``records`` are
    ``(contig, variant, depth, alt_count)`` tuples."""
    import pysam

    path.parent.mkdir(parents=True, exist_ok=True)
    plain = path.with_suffix("")  # strip .gz
    with open(plain, "w") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##source=spbench-simulate\n")
        for name, seq in contigs.items():
            handle.write(f"##contig=<ID={name},length={len(seq)}>\n")
        handle.write('##FILTER=<ID=PASS,Description="All filters passed">\n')
        handle.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">\n')
        handle.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n')
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n')
        handle.write(
            '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">\n'
        )
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sample + "\n"
        )
        order = {name: i for i, name in enumerate(contigs)}
        for contig, variant, depth, alt_count in sorted(
            records, key=lambda r: (order[r[0]], r[1].pos)
        ):
            af = (alt_count / depth) if depth else 0.0
            qual = min(99.0, 10.0 * math.log10(alt_count + 1) * 10.0)
            handle.write(
                f"{contig}\t{variant.pos}\t.\t{variant.ref}\t{variant.alt}\t"
                f"{qual:.1f}\tPASS\tDP={depth};AF={af:.4f}\tGT:DP:AD\t"
                f"0/1:{depth}:{max(0, depth - alt_count)},{alt_count}\n"
            )

    pysam.tabix_compress(str(plain), str(path), force=True)
    plain.unlink()
    pysam.tabix_index(str(path), preset="vcf", force=True)


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #


def simulate(config: SimConfig, outdir: str | Path) -> Path:
    """Generate a complete dataset under ``outdir``. Returns the dataset root."""
    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    rng = np.random.default_rng(config.seed)

    # 1. Reference
    if config.reference_fasta:
        contigs = _read_fasta(config.reference_fasta)
        if config.contig_length > 0:
            contigs = {
                name: seq[: config.contig_length]
                for name, seq in list(contigs.items())[: config.n_contigs]
            }
    else:
        contigs = _random_reference(rng, config.contig_length, config.n_contigs)
    _write_fasta(outdir / "reference.fasta", contigs)

    import pysam

    pysam.faidx(str(outdir / "reference.fasta"))

    # 2. Strains and abundances
    strains, all_variants = build_strain_tree(rng, config, contigs)
    strain_ids = [s.strain_id for s in strains]
    abundance = build_abundances(rng, config, strain_ids)
    samples = sorted({sample for sample, _ in abundance})

    # 3. Reads, BAMs, VCFs
    #
    # The realistic path needs each strain's genome materialised (to sample
    # reads from) and a minimap2 index over the reference (to align them back).
    # Both are built once and reused across timepoints.
    aligner = None
    strain_sequences: dict[tuple[str, str], str] = {}
    if config.read_model == "hifi" or config.aligner == "minimap2":
        from spbench.hifi import Minimap2Aligner, build_strain_sequence

        aligner = Minimap2Aligner.from_fasta(
            str(outdir / "reference.fasta"), preset=config.minimap2_preset
        )
        for strain in strains:
            for contig, ref_seq in contigs.items():
                on_contig = {
                    pos: variant
                    for pos, variant in strain.variants.items()
                    if pos in all_variants.get(contig, {})
                }
                strain_sequences[(strain.strain_id, contig)] = build_strain_sequence(
                    ref_seq, on_contig
                )

    truth_reads: list[dict] = []
    for sample in samples:
        reads = simulate_reads_for_sample(
            rng, config, sample, strains, contigs, abundance, aligner, strain_sequences
        )
        write_bam(outdir / "bam" / f"{sample}.bam", contigs, reads)

        records: list[tuple[str, Variant, int, int]] = []
        for contig, variants in all_variants.items():
            for variant in variants.values():
                depth, alt_count = observe_site_support(reads, strains, contig, variant)
                if config.vcf_mode == "called":
                    af = (alt_count / depth) if depth else 0.0
                    if alt_count < config.call_min_alt_reads or af < config.call_min_af:
                        continue
                elif depth == 0:
                    continue
                records.append((contig, variant, depth, alt_count))

        if config.vcf_mode == "called" and config.call_fp_rate > 0:
            records.extend(_false_positive_sites(rng, config, contigs, all_variants, reads))

        write_vcf(outdir / "variants" / f"{sample}.vcf.gz", contigs, sample, records)

        for read in reads:
            truth_reads.append(
                {
                    "sample": sample,
                    "read_id": read.read_id,
                    "contig": read.contig,
                    "strain_id": read.strain_id,
                    "start": read.ref_start,
                    "end": read.ref_end,
                }
            )
        logger.info("%s: %d reads, %d variant sites called", sample, len(reads), len(records))

    _write_union_vcf(outdir, contigs, samples)

    # 4. Truth tables
    truth_dir = outdir / "truth"
    site_rows = []
    for contig, variants in all_variants.items():
        for pos in sorted(variants):
            variant = variants[pos]
            site_rows.append(
                {
                    "contig": contig,
                    "pos": pos,
                    "ref": variant.ref,
                    "alt": variant.alt,
                }
            )
    write_table(truth_dir / "sites.tsv", SITES_COLUMNS, site_rows)

    strain_rows = []
    for strain in strains:
        for contig, variants in all_variants.items():
            alleles: dict[int, str] = {}
            for pos, variant in variants.items():
                own = strain.variants.get(pos)
                if own is not None and own.kind == variant.kind and own.alt == variant.alt:
                    alleles[pos] = variant.alt
                elif pos in strain.deleted:
                    continue  # site is inside a deletion this strain carries
                else:
                    alleles[pos] = variant.ref
            strain_rows.append(
                {
                    "strain_id": strain.strain_id,
                    "contig": contig,
                    "n_sites": len(alleles),
                    "alleles": encode_alleles(alleles),
                }
            )
    write_table(truth_dir / "strains.tsv", STRAINS_COLUMNS, strain_rows)

    write_table(
        truth_dir / "abundance.tsv",
        ABUNDANCE_COLUMNS,
        (
            {"sample": sample, "strain_id": strain_id, "abundance": f"{value:.6f}"}
            for (sample, strain_id), value in sorted(abundance.items())
        ),
    )
    write_table(truth_dir / "read_origins.tsv", READ_ORIGINS_COLUMNS, truth_reads)

    # 5. Manifest
    manifest = {
        "name": config.name,
        "spbench_version": __import__("spbench").__version__,
        "config": asdict(config),
        "config_fingerprint": config.fingerprint(),
        "samples": samples,
        "contigs": {name: len(seq) for name, seq in contigs.items()},
        "n_strains": len(strains),
        "n_variant_sites": sum(len(v) for v in all_variants.values()),
        "strain_tree": {s.strain_id: s.parent for s in strains},
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info(
        "Simulated %s: %d strains, %d sites, %d samples -> %s",
        config.name,
        len(strains),
        manifest["n_variant_sites"],
        len(samples),
        outdir,
    )
    return outdir


def _false_positive_sites(
    rng: np.random.Generator,
    config: SimConfig,
    contigs: dict[str, str],
    all_variants: dict[str, dict[int, Variant]],
    reads: list[SimulatedRead],
) -> list[tuple[str, Variant, int, int]]:
    """Spurious low-frequency calls, as a real caller would emit on this data."""
    out: list[tuple[str, Variant, int, int]] = []
    mean_depth = max(1, int(config.coverage))
    for contig, seq in contigs.items():
        n_fp = rng.poisson(config.call_fp_rate * len(seq))
        for _ in range(int(n_fp)):
            pos = int(rng.integers(2, len(seq) - 1))
            if pos in all_variants.get(contig, {}):
                continue
            ref_base = seq[pos - 1]
            if ref_base not in BASES:
                continue
            alt = str(rng.choice([b for b in BASES if b != ref_base]))
            depth = max(config.call_min_alt_reads + 1, int(rng.poisson(mean_depth)))
            alt_count = int(rng.integers(config.call_min_alt_reads, max(depth // 4, 4)))
            out.append((contig, Variant(pos, "snv", ref_base, alt), depth, min(alt_count, depth)))
    return out


def _write_union_vcf(outdir: Path, contigs: dict[str, str], samples: list[str]) -> None:
    """A single VCF holding every site called in any sample.

    Some tools (Strainy in particular) expect one variant file for a run rather
    than one per alignment. Giving them the union keeps their site list at least
    as good as the per-sample lists, so the comparison never penalises a tool
    for an interface mismatch.
    """
    import pysam

    seen: dict[tuple[str, int], tuple[str, str, int, int]] = {}
    for sample in samples:
        with pysam.VariantFile(str(outdir / "variants" / f"{sample}.vcf.gz")) as vcf:
            for rec in vcf.fetch():
                key = (rec.contig, rec.pos)
                depth = int(rec.info.get("DP", 0) or 0)
                if key not in seen or depth > seen[key][2]:
                    seen[key] = (rec.ref, rec.alts[0], depth, 0)

    order = {name: i for i, name in enumerate(contigs)}
    plain = outdir / "variants" / "union.vcf"
    with open(plain, "w") as handle:
        handle.write("##fileformat=VCFv4.2\n##source=spbench-simulate\n")
        for name, seq in contigs.items():
            handle.write(f"##contig=<ID={name},length={len(seq)}>\n")
        handle.write('##FILTER=<ID=PASS,Description="All filters passed">\n')
        handle.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">\n')
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tunion\n")
        for (contig, pos), (ref, alt, depth, _) in sorted(
            seen.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])
        ):
            handle.write(
                f"{contig}\t{pos}\t.\t{ref}\t{alt}\t50.0\tPASS\tDP={depth}\tGT\t0/1\n"
            )

    gz = outdir / "variants" / "union.vcf.gz"
    pysam.tabix_compress(str(plain), str(gz), force=True)
    plain.unlink()
    pysam.tabix_index(str(gz), preset="vcf", force=True)
