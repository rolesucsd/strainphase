"""A HiFi-shaped error model, and real alignment with minimap2.

Why this module exists
----------------------
The default simulator path emits reads at their true reference coordinates with
exact CIGARs and substitution-only error. That is fast, dependency-free and
deterministic, which makes it right for CI and for developing metrics - but it
is *not* what PacBio HiFi data looks like, in two ways that matter:

1. **HiFi error is dominated by indels in homopolymers**, not by substitutions.
   CCS consensus resolves most substitution error; what survives is
   predominantly insertion/deletion slippage in homopolymer runs and short
   tandem repeats, and the rate climbs steeply with run length. A
   substitution-only model therefore understates the error mode that actually
   causes trouble.

2. **Those errors matter mainly because an aligner has to place them.** A
   homopolymer indel is not hard because the base is wrong; it is hard because
   minimap2 can shift it, and every downstream tool then disagrees about where a
   variant sits. Emitting exact CIGARs removes precisely the difficulty the
   indel axis is supposed to test.

So this module provides the realistic path: a context-aware error model plus
genuine minimap2 alignment through the ``mappy`` bindings. Reads are generated
from a materialised strain genome, corrupted, reverse-complemented half the
time, and then *aligned back* to the reference like real data. Nothing tells the
aligner where a read came from.

Calibration
-----------
The defaults below are literature-shaped, not fitted: roughly 0.1% total error
with a minority of substitutions, indel error concentrated in homopolymers of
length >= 4 and rising with run length. They are approximations and every one of
them is configurable, because the right values depend on chemistry, and a run
sequenced on different chemistry than these defaults assume should set its own.
Treat them as a plausible HiFi-like regime, not as a model of your instrument.

If you need higher fidelity than this, generate reads with PBSIM3 (multi-pass
CLR followed by ``ccs``) per strain genome and feed the resulting FASTQs in;
the read-to-strain mapping is preserved because each strain is simulated
separately. That path is not wired up here - this module is the middle ground
between "exact and unrealistic" and "a full CCS simulation".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


# --------------------------------------------------------------------------- #
# Strain genomes
# --------------------------------------------------------------------------- #


def build_strain_sequence(reference: str, variants: dict) -> str:
    """Materialise a strain's genome by applying its variants to the reference.

    The realistic path needs the actual strain sequence to sample reads from,
    because after error corruption the read no longer has a derivable CIGAR - it
    has to be aligned. ``variants`` maps 1-based anchor position to a
    :class:`~spbench.simulate.Variant`.
    """
    pieces: list[str] = []
    pos = 0  # 0-based cursor into the reference
    for anchor in sorted(variants):
        variant = variants[anchor]
        start = anchor - 1
        if start < pos:
            continue  # overlapped by a preceding deletion; skip
        pieces.append(reference[pos:start])
        if variant.kind == "snv":
            pieces.append(variant.alt)
            pos = start + 1
        elif variant.kind == "del":
            pieces.append(variant.alt)  # the retained anchor base
            pos = start + len(variant.ref)
        else:  # insertion
            pieces.append(variant.alt)
            pos = start + len(variant.ref)
    pieces.append(reference[pos:])
    return "".join(pieces)


# --------------------------------------------------------------------------- #
# Error model
# --------------------------------------------------------------------------- #


@dataclass
class HiFiErrorModel:
    """Context-aware HiFi error model.

    Parameters
    ----------
    error_rate
        Total per-base error rate. 0.001 is roughly Q30, the usual HiFi median.
    substitution_fraction
        Share of total error that is substitution. The remainder is indel.
        HiFi is indel-dominated, hence a default well below 0.5.
    homopolymer_min_length
        Runs shorter than this are treated as ordinary sequence. Slippage in
        2-3 bp runs is rare enough to ignore next to everything else here.
    homopolymer_exponent
        How sharply indel probability rises with run length. Indel weight for a
        run of length L is ``(L - min_length + 1) ** exponent``, so 2 makes an
        8 bp run about 25x more error-prone than a 4 bp run.
    quality_penalty_in_homopolymer
        Factor applied to the per-base error probability - and hence reflected
        in the emitted Q - inside a long homopolymer. Real CCS quality values do
        drop in these contexts, and a model that emitted flat Q40 across a 10 bp
        run would give quality-weighted methods information they do not have.
    """

    error_rate: float = 0.001
    substitution_fraction: float = 0.35
    homopolymer_min_length: int = 4
    homopolymer_exponent: float = 2.0
    quality_penalty_in_homopolymer: float = 4.0
    max_indel_error_length: int = 2


def homopolymer_runs(seq: str, min_length: int) -> list[tuple[int, int]]:
    """``(start, length)`` for every homopolymer run of at least ``min_length``."""
    runs: list[tuple[int, int]] = []
    if not seq:
        return runs
    start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            length = i - start
            if length >= min_length:
                runs.append((start, length))
            start = i
    return runs


def apply_hifi_errors(
    seq: str,
    rng: np.random.Generator,
    model: HiFiErrorModel,
) -> tuple[str, list[int]]:
    """Corrupt a read and emit calibrated per-base qualities.

    Returns ``(sequence, qualities)``. The sequence length changes when indel
    errors are applied, and the qualities returned always match it.

    Two error processes run:

    * **Substitution**, per base, at ``error_rate * substitution_fraction``,
      modulated by homopolymer context so Q tracks the real probability.
    * **Indel**, allocated as a budget of expected events over the read and
      distributed across homopolymer runs in proportion to
      ``(L - min + 1) ** exponent``. Allocating a budget rather than rolling a
      per-run coin keeps the total error rate at the configured value however
      homopolymer-rich the underlying sequence happens to be, so two references
      with different composition stay comparable.
    """
    if not seq:
        return seq, []

    n = len(seq)
    runs = homopolymer_runs(seq, model.homopolymer_min_length)

    # Per-base context factor: 1.0 outside long homopolymers, higher inside.
    context = np.ones(n)
    for start, length in runs:
        weight = 1.0 + (model.quality_penalty_in_homopolymer - 1.0) * min(
            1.0, (length - model.homopolymer_min_length + 1) / 5.0
        )
        context[start : start + length] = weight

    sub_rate = model.error_rate * model.substitution_fraction
    per_base = np.clip(context * sub_rate * rng.uniform(0.0, 2.0, size=n), 1e-6, 0.5)
    quals = np.clip(np.round(-10.0 * np.log10(per_base)), 2, 60).astype(int).tolist()

    bases = list(seq)
    substitute = rng.random(n) < per_base
    for i in np.nonzero(substitute)[0]:
        current = bases[i]
        if current not in "ACGT":
            continue
        bases[i] = str(rng.choice([b for b in "ACGT" if b != current]))

    # Indel budget, spent on homopolymer runs.
    indel_budget = model.error_rate * (1.0 - model.substitution_fraction) * n
    edits: list[tuple[int, int, int]] = []  # (start, length, delta) delta>0 insert
    if runs and indel_budget > 0:
        weights = np.array(
            [
                (length - model.homopolymer_min_length + 1) ** model.homopolymer_exponent
                for _start, length in runs
            ],
            dtype=float,
        )
        weights /= weights.sum()
        n_events = int(rng.poisson(indel_budget))
        if n_events:
            chosen = rng.choice(len(runs), size=n_events, p=weights)
            for run_index in chosen:
                start, length = runs[int(run_index)]
                size = int(rng.integers(1, model.max_indel_error_length + 1))
                # Contraction is more common than expansion in CCS consensus,
                # and a contraction can never remove more than the run itself.
                if rng.random() < 0.6:
                    size = min(size, length - 1)
                    if size > 0:
                        edits.append((start, length, -size))
                else:
                    edits.append((start, length, size))

    if edits:
        bases, quals = _apply_edits(bases, quals, edits)

    return "".join(bases), quals


def _apply_edits(
    bases: list[str], quals: list[int], edits: list[tuple[int, int, int]]
) -> tuple[list[str], list[int]]:
    """Apply homopolymer expansions/contractions right to left.

    Right to left so that earlier edits' coordinates stay valid, and one edit
    per run so two events in the same run cannot interact.
    """
    seen_runs: set[int] = set()
    ordered = []
    for start, length, delta in sorted(edits, key=lambda e: -e[0]):
        if start in seen_runs:
            continue
        seen_runs.add(start)
        ordered.append((start, length, delta))

    for start, length, delta in ordered:
        if delta > 0:
            base = bases[start]
            bases[start:start] = [base] * delta
            quals[start:start] = [max(2, quals[start] - 10)] * delta
        else:
            drop = min(-delta, length - 1)
            if drop > 0:
                del bases[start : start + drop]
                del quals[start : start + drop]
    return bases, quals


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


@dataclass
class Alignment:
    """One primary alignment, in BAM terms."""

    contig: str
    ref_start: int  # 0-based
    is_reverse: bool
    mapq: int
    cigar: list[tuple[int, int]]  # (op, length), BAM op codes
    nm: int
    sequence: str  # already oriented as it should appear in the BAM
    qualities: list[int]


class Minimap2Aligner:
    """Thin wrapper over ``mappy`` that returns BAM-ready primary alignments.

    Using the real aligner is the point: it reintroduces the placement
    ambiguity, mapping-quality variation and soft-clipping that an exact
    simulator removes, and it does so identically for every tool downstream.
    """

    def __init__(self, contigs: dict[str, str], preset: str = "map-hifi", seed: int = 0):
        try:
            import mappy
        except ImportError as exc:  # pragma: no cover - guarded by config validation
            raise ImportError(
                "aligner='minimap2' needs the `mappy` package (pip install mappy)"
            ) from exc

        self._mappy = mappy
        if len(contigs) == 1:
            name, seq = next(iter(contigs.items()))
            self._single = name
            self._aligner = mappy.Aligner(seq=seq, preset=preset, n_threads=1)
        else:
            # mappy builds a single-sequence index from `seq=`; for several
            # contigs it needs a file, which the caller writes as the reference
            # FASTA anyway.
            raise ValueError("Minimap2Aligner.from_fasta is required for multi-contig references")
        if not self._aligner:
            raise RuntimeError("failed to build the minimap2 index")

    @classmethod
    def from_fasta(cls, path: str, preset: str = "map-hifi") -> Minimap2Aligner:
        import mappy

        obj = cls.__new__(cls)
        obj._mappy = mappy
        obj._single = None
        obj._aligner = mappy.Aligner(path, preset=preset, n_threads=1)
        if not obj._aligner:
            raise RuntimeError(f"failed to build a minimap2 index from {path}")
        return obj

    def align(self, sequence: str, qualities: list[int]) -> Alignment | None:
        """Best primary alignment, or ``None`` if the read does not map.

        Unmapped reads are dropped rather than forced into place. Real HiFi
        metagenomes lose a few percent of reads this way and every tool sees the
        same loss, so it belongs in the simulation.
        """
        best = None
        for hit in self._aligner.map(sequence):
            if not hit.is_primary:
                continue
            if best is None or hit.mlen > best.mlen:
                best = hit
        if best is None:
            return None

        is_reverse = best.strand == -1
        if is_reverse:
            bam_seq = reverse_complement(sequence)
            bam_quals = list(reversed(qualities))
            left_clip = len(sequence) - best.q_en
            right_clip = best.q_st
        else:
            bam_seq = sequence
            bam_quals = list(qualities)
            left_clip = best.q_st
            right_clip = len(sequence) - best.q_en

        # mappy cigar entries are (length, op) with op 0=M, 1=I, 2=D.
        cigar: list[tuple[int, int]] = []
        if left_clip:
            cigar.append((4, left_clip))
        cigar.extend((int(op), int(length)) for length, op in best.cigar)
        if right_clip:
            cigar.append((4, right_clip))

        return Alignment(
            contig=best.ctg,
            ref_start=best.r_st,
            is_reverse=is_reverse,
            mapq=int(best.mapq),
            cigar=cigar,
            nm=int(best.NM),
            sequence=bam_seq,
            qualities=bam_quals,
        )
