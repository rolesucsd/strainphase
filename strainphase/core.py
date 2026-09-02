#!/usr/bin/env python3
"""Hybrid graph-probabilistic haplotype reconstruction for long-read metagenomics.

1. Create overlapping windows (step = window_size / 2)
2. Per window: graph initialization + EM refinement
3. Link haplotypes across windows where consensus agrees on shared SNVs
4. Output tracks with merged consensus spanning multiple windows (results_to_dataframe:
   one row per TRACK, span_bp = span_end - span_start, n_windows = windows spanned)
"""

from __future__ import annotations

import bisect
import logging
import os
import warnings
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from multiprocessing import Pool

import community as community_louvain
import networkx as nx
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

# python-louvain installs as `community`; an unrelated PyPI package of the same
# name shadows it and fails later with an AttributeError. Check identity once, here.
if not hasattr(community_louvain, "best_partition"):
    raise ImportError(
        "The installed `community` module is not python-louvain: it has no "
        "best_partition(). Some other package of that name is shadowing it. "
        "Fix with:  pip uninstall -y community && pip install python-louvain  "
        "(conda: the package is `python-louvain`, NOT `louvain`, which is an "
        "unrelated igraph binding.)"
    )

try:
    import pysam

    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False
    logging.warning("pysam not installed; I/O functions will not work")


# =============================================================================
# WARNING THROTTLING
# =============================================================================


class WarningThrottler:
    """Throttle repeated warnings to avoid spam."""

    _warned: set[str] = set()

    @classmethod
    def warn_once(cls, key: str, message: str):
        if key not in cls._warned:
            warnings.warn(message, stacklevel=2)
            cls._warned.add(key)



# =============================================================================
# CONFIGURATION WITH VALIDATION
# =============================================================================


@dataclass
class HaplotyperConfig:
    """Configuration for the haplotyper.

    Every threshold and filter is a field here; ``__post_init__`` validates them
    on construction.
    """

    # =========== WINDOW PARAMETERS ===========
    window_size: int = 20000
    # Two deliberate depth floors. min_reads_per_window gates de-novo PHASING (separating
    # haplotypes at 50/50 needs ~10-20 reads; 3 manufactures the abundance==1.0 artifact);
    # min_reads_for_rescue gates whether a window is BUILT AT ALL, so it can still receive a
    # rescued anchor haplotype (which needs less evidence than de-novo separation). A window
    # with rescue <= n < phasing is created but not phased.
    # min_snvs_per_window: 1 skips only empty windows; higher filters sparse regions.
    min_snvs_per_window: int = 1
    min_reads_per_window: int = 10
    min_reads_for_rescue: int = 5

    # =========== READ FILTERING ===========
    min_mapq: int = 20
    min_base_quality: int = 20
    default_base_quality: int = 20
    # Cap per window; reads above this are uniformly subsampled with the config seed.
    max_reads_per_window: int = 500
    # A read must physically cover at least this many bp of a window to count in it, so a
    # 1 bp clip does not enter n_reads_examined / junk / the abundance denominator as a full span.
    min_read_window_overlap_bp: int = 500
    # Two reads must physically overlap by at least this much to be compared at all.
    min_read_read_overlap_bp: int = 500

    # Spill per-sample WindowResults to <output_dir>/tmp during the first pass rather than
    # holding every sample's reads in RAM. Cross-sample rescue reads only `.haplotypes` off
    # other samples, so the heavy fields can live on disk. See longitudinal.process_mag_longitudinal.
    spill_results_to_disk: bool = True
    # Windows are handed to the worker pool in batches of n_workers * this, so the input
    # list for a contig is never fully materialised at once.
    window_batch_factor: int = 4

    # =========== VARIANT FILTERING ===========
    min_depth_site: int = 3
    # Optional AF band. None (default) keeps all variants: a position fixed at AF=0/1 in one
    # timepoint is still informative across timepoints. Set e.g. (0.05, 0.95) for within-sample
    # polymorphic sites only.
    af_range: tuple[float, float] | None = None

    # =========== MUTATION HANDLING — INVARIANT ===========
    # No ``include_indels`` or ``require_biallelic`` flag exists. Every mutation
    # type (SNV, MNP->SNVs, insertion, deletion) is loaded and every multi-allelic
    # (>2 allele) site is kept. Hard invariant — see ``docs/MUTATION_HANDLING.md``.

    # =========== GRAPH CONSTRUCTION ===========
    min_shared_snvs_for_edge: int = 1
    min_reads_per_cluster: int = 3

    # Minimum physical overlap between two entities, below which the verdict is an
    # explicit NON-MERGE rather than "unknown" (Strainy's I = 1000).
    min_entity_overlap_bp: int = 500
    # SVs stay in the identity comparison (default False): an inversion is a marker like any
    # other, and its two orientations trade frequency over time as the flip trajectory. Knob
    # only to measure the effect.
    exclude_sv_from_identity: bool = False

    # =========== EM PARAMETERS ===========
    em_max_iter: int = 30
    em_tolerance: float = 1e-5
    dirichlet_alpha: float = 1.0
    min_hap_eff_weight: float = 3.0
    min_gamma_for_vote: float = 0.01

    # =========== JUNK MODEL ===========
    junk_divergence_rate: float = 0.10

    # HAPLOTYPE IDENTITY: identity_distance is the single rate for every "same entity?" test
    # (read-vs-read, hap-vs-hap, rescue); each comparison carries its own evidence floor:
    #   read vs read       min_shared_snvs_for_edge = 1
    #   haplotype vs hap   min_shared_markers       = 3
    #   read vs haplotype  min_shared_for_rescue    = 3
    #   cross-sample       track_merge_min_shared_markers = 1
    identity_distance: float = 0.02
    min_shared_markers: int = 3
    # Agreeing markers a track pair needs before the cross-sample merge joins them. 1 is a
    # permissive first pass: exact agreement cannot fuse two genotypes that disagree
    # anywhere both call, so this trades only evidence volume. Raising it splits hard.
    track_merge_min_shared_markers: int = 1
    # --- MARKER SET (computed once, used by the cross-sample merge) ----------------------
    # A position is a candidate marker when >1 allele is seen at it anywhere on the contig.
    # An allele is kept when it reaches marker_min_reads reads OR marker_min_frac of a
    # sample's reads at that position, in at least marker_min_samples samples; a position
    # survives when >= 2 alleles do. The two alleles need NOT co-occur in one sample: a
    # swept position is fixed within each sample, so requiring both in one would discard the
    # very events this tool finds. See supported_marker_positions.
    marker_min_frac: float = 0.10
    marker_min_reads: int = 3
    marker_min_samples: int = 2

    assign_confidence_threshold: float = 0.90

    binomial_alpha: float = 0.05

    # =========== LONGITUDINAL PARAMETERS ===========
    min_weight_for_anchor: float = 0.15
    min_shared_for_rescue: int = 3  # Min shared SNVs with actual calls for rescue matching
    rescued_min_weight: float = 0.02

    # =========== ABUNDANCE COHERENCE ===========
    # A genome cannot hold two frequencies at one locus at one time. Tested on raw counts
    # (never the derived, already-quantised `abundance`) with Fisher's exact test, so the
    # rule tightens as depth grows instead of using a fixed cutoff. Single timepoint only:
    # a window-merging check, never a cross-timepoint comparison.
    abundance_coherence_alpha: float = 0.01
    min_reads_for_coherence: int = 10

    # =========== LINKING DIAGNOSTICS ===========
    linking_debug: bool = False  # Record detailed linking diagnostics
    linking_debug_max_records: int = 5000  # Cap to avoid massive files

    # Window-level shared SNV POSITIONS (does the window pair even have common sites).
    min_shared_snvs_for_link: int = 3
    # How many windows ahead link_windows may reach. Overlapping neighbours are compared on
    # consensus; reach > 1 also admits NON-overlapping windows (disjoint positions, so linked
    # on shared reads alone), covering the marker-free window that would otherwise be an
    # unbridgeable hub. Default 2; set 1 to link only overlapping windows. Rationale: docs/design/core.md.
    link_window_reach: int = 2
    # Shared reads to link a NON-overlapping pair. Consensus cannot gate these (disjoint
    # positions), so this plus reciprocal best match is the whole evidence; do not lower casually.
    link_min_shared_reads: int = 2
    # Coverage-invariant companion to link_min_shared_reads: a non-overlapping link also needs
    # shared_reads >= frac * min(each hap's CONTINUING reads), so the bar rises with depth for
    # abundant strains and stays at the floor for rare ones. 0.0 = disabled (floor only).
    link_shared_read_frac: float = 0.0
    # Co-supported span of the two haplotypes inside the shared region, as a fraction of it.
    # Window geometry alone is useless (tiles overlap by exactly 50% or 0%). 25% of a 10 kb
    # overlap is 2500 bp, above min_entity_overlap_bp; 50% is too aggressive given under-merging risk.
    min_cosupported_span_frac: float = 0.25

    # =========== RUNTIME PARAMETERS ===========
    # Seeded by DEFAULT (42), not None. Louvain read clustering (GraphInitializer) is handed
    # this as random_state, and the per-window subsample is a stable function of it, so both
    # depend on the seed. None made reruns disagree in the ~8th decimal; set an integer to
    # vary the draw deliberately.
    random_seed: int | None = 42
    n_workers: int = 1  # Number of parallel workers for window processing (1=sequential)

    def __post_init__(self):
        """Validate configuration parameters."""

        # Junk divergence rate
        if not (0 < self.junk_divergence_rate < 0.75):
            raise ValueError(
                f"junk_divergence_rate must be in (0, 0.75), got {self.junk_divergence_rate}"
            )

        if self.track_merge_min_shared_markers < 1:
            raise ValueError(
                "track_merge_min_shared_markers must be >= 1, got "
                f"{self.track_merge_min_shared_markers}"
            )

        if self.marker_min_frac > 0.5:
            raise ValueError(
                f"marker_min_frac should be <= 0.5, got {self.marker_min_frac}"
            )

        # A reach below 1 drops the overlapping-neighbour link every track is built from.
        if self.link_window_reach < 1:
            raise ValueError(
                f"link_window_reach must be >= 1, got {self.link_window_reach}"
            )
        if self.link_min_shared_reads < 1:
            raise ValueError(
                f"link_min_shared_reads must be >= 1, got {self.link_min_shared_reads}"
            )
        if not (0.0 <= self.link_shared_read_frac < 1.0):
            raise ValueError(
                f"link_shared_read_frac must be in [0, 1), got {self.link_shared_read_frac}"
            )

        # Merge distance threshold
        if not (0 <= self.identity_distance <= 1):
            raise ValueError(
                f"identity_distance must be in [0, 1], got {self.identity_distance}"
            )

        # AF range (optional)
        if self.af_range is not None and not (
            0 <= self.af_range[0] < self.af_range[1] <= 1
        ):
            raise ValueError(
                f"af_range must be (low, high) with 0 <= low < high <= 1, got {self.af_range}"
            )

        # Confidence threshold
        if not (0 < self.assign_confidence_threshold <= 1):
            raise ValueError(
                f"assign_confidence_threshold must be in (0, 1], got {self.assign_confidence_threshold}"
            )

        # Window size
        if self.window_size < 100:
            raise ValueError(f"window_size too small: {self.window_size}")

        # Rescue floor must sit at or below the phasing floor (a window must be buildable
        # before it can be phased). Clamp rather than raise: lowering the phasing floor means
        # "be more permissive".
        if self.min_reads_for_rescue > self.min_reads_per_window:
            object.__setattr__(self, "min_reads_for_rescue", self.min_reads_per_window)


        if not (0 <= self.min_cosupported_span_frac <= 1):
            raise ValueError(
                f"min_cosupported_span_frac must be in [0, 1], got "
                f"{self.min_cosupported_span_frac}"
            )


        if not (0 < self.abundance_coherence_alpha < 1):
            raise ValueError(
                f"abundance_coherence_alpha must be in (0, 1), got "
                f"{self.abundance_coherence_alpha}"
            )

        # EM iterations
        if self.em_max_iter < 1:
            raise ValueError(f"em_max_iter must be >= 1, got {self.em_max_iter}")

    def get_rng(self) -> np.random.Generator:
        """Get a reproducible random number generator."""
        return np.random.default_rng(self.random_seed)


DEFAULT_CONFIG = HaplotyperConfig()


# =============================================================================
# DATA STRUCTURES
# =============================================================================


# config field -> the argparse attribute that sets it. A field whose flag a given parser
# does not define keeps its dataclass default, which lets one builder serve every subcommand.
_CONFIG_FROM_ARG: dict[str, str] = {
    "window_size": "window_size",
    "max_reads_per_window": "max_reads",
    "min_mapq": "min_mapq",
    "min_depth_site": "min_depth_site",
    "random_seed": "seed",
    "min_weight_for_anchor": "min_anchor_weight",
    "rescued_min_weight": "rescued_min_weight",
    "min_reads_per_window": "min_reads_per_window",
    "min_reads_for_rescue": "min_reads_for_rescue",
    "min_cosupported_span_frac": "min_cosupported_span_frac",
    "min_shared_snvs_for_link": "min_shared_snvs_for_link",
    "identity_distance": "identity_distance",
    "min_shared_markers": "min_shared_markers",
    "track_merge_min_shared_markers": "track_merge_min_shared_markers",
    "link_window_reach": "link_window_reach",
    "link_min_shared_reads": "link_min_shared_reads",
    "link_shared_read_frac": "link_shared_read_frac",
    "abundance_coherence_alpha": "abundance_coherence_alpha",
    "min_reads_for_coherence": "min_reads_for_coherence",
    "window_batch_factor": "window_batch_factor",
}


def config_from_args(args, **overrides) -> HaplotyperConfig:
    """Build one ``HaplotyperConfig`` from any entry point's parsed arguments.

    ``overrides`` win over the parsed values, for the few settings an entry point fixes
    rather than exposes. Flags that invert or derive their field (``--no-spill``,
    ``--workers``, ``--af-range``) are handled explicitly below, since a name-to-name
    table cannot express them.
    """
    values: dict = {}
    for cfg_field, attr in _CONFIG_FROM_ARG.items():
        if hasattr(args, attr):
            values[cfg_field] = getattr(args, attr)

    # One --min-overlap-bp governs all three physical-overlap floors (read-vs-window,
    # read-vs-read, entity-vs-entity); the fields stay separate so a library caller can differ them.
    if getattr(args, "min_overlap_bp", None) is not None:
        values["min_read_window_overlap_bp"] = args.min_overlap_bp
        values["min_read_read_overlap_bp"] = args.min_overlap_bp
        values["min_entity_overlap_bp"] = args.min_overlap_bp

    # argparse hands back a LIST for nargs=2, and af_range is part of the VCF-load
    # cache key, which a list cannot be - so convert here rather than at the use site.
    if getattr(args, "af_range", None):
        values["af_range"] = tuple(args.af_range)

    # Inverted and derived flags.
    if hasattr(args, "no_spill"):
        values["spill_results_to_disk"] = not args.no_spill
    if hasattr(args, "workers"):
        values["n_workers"] = max(1, args.workers)

    values.update(overrides)
    return HaplotyperConfig(**values)


@dataclass
class Read:
    """Lightweight container for read data. All positions are 1-based (VCF convention)."""

    id: str
    contig: str
    mapq: int
    alleles: dict[int, str] = field(default_factory=dict)
    quals: dict[int, int] = field(default_factory=dict)
    sample: str | None = None
    # Reference alignment span, 1-based [start, end), for the physical-overlap gates.
    # Default (0, 0) means "unknown" - the overlap gates are then skipped, not made to reject.
    ref_start: int = 0
    ref_end: int = 0
    # Aligned reference intervals, 1-based [start, end), one per alignment segment. Empty for
    # a single-alignment read (its outer span IS its coverage); a split-alignment molecule
    # carries one per segment so a covers() test does not answer yes inside its own gap.
    segments: list[tuple[int, int]] = field(default_factory=list)

    def covers(self, pos: int) -> bool:
        """Is *pos* inside an ALIGNED part of this read? Unknown span (0, 0) is False."""
        if self.segments:
            return any(s <= pos < e for s, e in self.segments)
        return self.ref_start <= pos < self.ref_end

    def overlap_bp(self, other: Read) -> int:
        """Physical reference overlap with *other*, in bp. -1 when either span is unknown."""
        if self.ref_end <= self.ref_start or other.ref_end <= other.ref_start:
            return -1
        return max(0, min(self.ref_end, other.ref_end) - max(self.ref_start, other.ref_start))


@dataclass
class Window:
    """A genomic window (contig interval) with its SNVs and reads.

    snv_pos/ref_alleles come from the VCF; reads are pulled from the BAM in make_windows_lazy.
    """

    contig: str
    start: int  # 1-based, inclusive
    end: int  # 1-based, exclusive
    snv_pos: list[int] = field(default_factory=list)  # SNV positions (from VCF)
    ref_alleles: dict[int, str] = field(default_factory=dict)  # REF base per SNV (from VCF)
    # Reads overlapping this window, in gamma-row order. After offload_heavy these are
    # id-only _ReadRef stand-ins: row order and ids survive, alleles do not, so anything
    # needing alleles must run before the offload.
    reads: list[Read] = field(default_factory=list)
    # Per-position site type ("snv"/"del"/"ins"/"sv"). Carried here because the identity
    # code needs to know which positions are SVs to (optionally) exclude them from the distance.
    site_type: dict[int, str] = field(default_factory=dict)
    sample: str | None = None  # Optional timepoint/sample label (redundant with Read.sample)
    window_idx: int = 0  # Position in contig's window sequence

    # Cached position sets for graph building (optimization)
    _pos_sets: list[set[int]] | None = field(default=None, repr=False)

    def get_read_position_sets(self) -> list[set[int]]:
        """Get precomputed position sets for each read (cached)."""
        if self._pos_sets is None:
            self._pos_sets = [
                {p for p in r.alleles if self.start <= p < self.end} for r in self.reads
            ]
        return self._pos_sets

    @property
    def n_snvs(self) -> int:
        return len(self.snv_pos)

    @property
    def n_reads(self) -> int:
        return len(self.reads)


@dataclass
class Haplotype:
    """A resolved haplotype within a window."""

    consensus: dict[int, str]
    weight: float = 0.0  # Mixture weight (pi) after EM / post-merge / rescue.
    supporting_reads: int = 0
    confidence: float = 0.0  # Mean gamma over confident reads for this haplotype.
    track_id: str | None = None  # Assigned after window linking

    def distance_to(
        self, other: Haplotype, positions: list[int], max_mismatches: int | None = None
    ) -> tuple[float, int, int]:
        """Normalized Hamming distance over *positions*, with optional early exit at
        *max_mismatches*. Returns (distance, n_mismatches, n_shared).

        INVARIANT: n_shared == 0 returns distance 1.0 (incomparable); callers must check
        n_shared before trusting the distance.
        """
        total = 0
        mismatches = 0
        for pos in positions:
            b1 = self.consensus.get(pos)
            b2 = other.consensus.get(pos)
            if b1 is None or b2 is None:
                continue
            total += 1
            if b1 != b2:
                mismatches += 1
                if max_mismatches is not None and mismatches > max_mismatches:
                    return 1.0, mismatches, total
        if total == 0:
            return 1.0, 0, 0
        return mismatches / total, mismatches, total

    def get_differing_positions(self, other: Haplotype, positions: list[int]) -> list[int]:
        """Return list of positions where haplotypes differ."""
        return [
            pos
            for pos in positions
            if (b1 := self.consensus.get(pos)) is not None
            and (b2 := other.consensus.get(pos)) is not None
            and b1 != b2
        ]


@dataclass
class WindowResult:
    """Complete results from processing a single window."""

    window: Window
    haplotypes: list[Haplotype]
    gamma: np.ndarray
    pi: np.ndarray
    log_likelihood: float
    assignments: list[dict]
    converged: bool
    iterations: int
    linking_debug: list[dict] = field(default_factory=list)
    # Step-1 comparisons that failed on a genuine allele disagreement (failed_mismatch only),
    # kept so the verdict survives to the output as an absolute negative. failed_no_evidence
    # is a measurement hole and is not recorded.
    link_mismatches: list[dict] = field(default_factory=list)
    # Step-1 abundance refusals (one sample, two adjacent windows), recorded so the verdict
    # carries over to the cross-sample merge instead of being recomputed pairwise.
    link_abundance_refusals: list[dict] = field(default_factory=list)
    n_reads_examined: int = 0
    reads_within_mismatch_per_hap: list[int] = field(default_factory=list)
    # Scalar summaries of `gamma`, recorded before the heavy fields are offloaded so the
    # output tables can still be built while reads/gamma live on disk. -1 = not recorded.
    n_reads_total: int = -1
    n_junk_reads: int = -1
    heavy_offloaded: bool = False

    # ---------------- Heavy-field offload ----------------
    # window.reads is ~99% of a WindowResult's footprint and is needed only while the
    # window's own sample is phased or rescued, so it is offloaded; gamma (~1000x smaller)
    # is still read when the output tables are built and stays resident. The WindowResult
    # object itself must stay resident: the rescue panel mutates its Haplotype weights in place.
    # Footprint measurement: docs/design/core.md.

    def offload_heavy(self) -> list:
        """Detach and return this window's reads, recording the read-count summaries."""
        if self.heavy_offloaded:
            return []
        self.n_reads_total, self.n_junk_reads = self.junk_read_counts()
        reads = self.window.reads
        self.window.reads = []
        self.window._pos_sets = None
        # `assignments` is deliberately NOT cleared: read-overlap threading reads it after
        # every sample is phased, and it is cheap (one small dict per read). See docs/design/core.md.
        self.heavy_offloaded = True
        return reads

    def restore_heavy(self, reads: list) -> None:
        """Re-attach reads detached by :meth:`offload_heavy`."""
        self.window.reads = reads
        self.window._pos_sets = None
        self.heavy_offloaded = False

    def junk_read_counts(self) -> tuple[int, int]:
        """``(n_reads_total, n_junk_reads)`` whether or not gamma is resident."""
        if self.gamma is not None and self.gamma.size:
            return (
                int(self.gamma.shape[0]),
                int((self.gamma[:, self.gamma.shape[1] - 1] >= 0.5).sum()),
            )
        if self.n_reads_total >= 0:
            return self.n_reads_total, self.n_junk_reads
        return 0, 0

    def validate(self) -> bool:
        """Validate internal consistency."""
        n_reads = len(self.window.reads)
        n_haps = len(self.haplotypes)
        k_eff = n_haps + 1  # haplotypes + 1 junk component

        assert self.gamma.shape == (
            n_reads,
            k_eff,
        ), f"gamma shape {self.gamma.shape} != expected ({n_reads}, {k_eff})"

        # Each row of gamma is a probability distribution (sums to ~1).
        row_sums = self.gamma.sum(axis=1)
        assert np.allclose(
            row_sums, 1.0, atol=1e-6
        ), f"gamma rows don't sum to 1: min={row_sums.min()}, max={row_sums.max()}"

        # pi is a probability distribution (sums to ~1) with k_eff entries.
        assert np.isclose(self.pi.sum(), 1.0, atol=1e-6), f"pi doesn't sum to 1: {self.pi.sum()}"

        assert len(self.pi) == k_eff, f"pi length {len(self.pi)} != k_eff {k_eff}"

        if self.reads_within_mismatch_per_hap:
            assert len(self.reads_within_mismatch_per_hap) == n_haps, (
                f"reads_within_mismatch_per_hap length {len(self.reads_within_mismatch_per_hap)} != n_haps {n_haps}"
            )

        return True


class _ReadRef:
    """A read's identity, kept after its alleles have been released.

    window.reads is emptied by offload_heavy, but a gamma row still has to say WHICH read it
    belongs to - that correspondence is the read partition. The id alone preserves it at
    ~1500x less memory than the whole Read; nothing else survives, so reaching for .alleles
    here raises an AttributeError naming this class rather than silently seeing none.
    Rationale: docs/design/core.md.
    """

    __slots__ = ("id",)

    def __init__(self, read_id: str) -> None:
        self.id = read_id

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_ReadRef({self.id!r})"


def _read_sort_hash(read_id: str, seed: int | None) -> int:
    """Stable per-read sort key for window-consistent subsampling.

    CRC32 over the seeded id (not hash(), which is salted per process) is stable across
    processes, runs and machines; the seed is mixed in so a different random_seed redraws.
    """
    return zlib.crc32(f"{seed}:{read_id}".encode())


_EMPTY_READS: frozenset = frozenset()


def _detach_reads(wr: WindowResult) -> list:
    """Offload a window's reads, leaving id-only stand-ins in gamma-row order.

    Returns the detached Read objects so the caller can spill them; ``restore_heavy``
    lays the real ones back over the stand-ins. Idempotent, and order-independent across
    the several WindowResults that can share one Window after rescue.
    """
    refs = [r if isinstance(r, _ReadRef) else _ReadRef(r.id) for r in wr.window.reads]
    reads = wr.offload_heavy()
    wr.window.reads = refs
    return reads


def _compute_read_mismatch_counts(
    window: Window,
    haplotypes: list[Haplotype],
    identity_distance: float,
) -> tuple[int, list[int]]:
    """Count reads within ``identity_distance`` of each haplotype consensus."""
    n_reads = len(window.reads)
    if not haplotypes or n_reads == 0:
        return n_reads, [0] * len(haplotypes)

    hap_positions = [set(h.consensus.keys()) for h in haplotypes]
    counts = [0] * len(haplotypes)
    read_pos_sets = window.get_read_position_sets()

    for read, read_pos in zip(window.reads, read_pos_sets):  # noqa: B905
        if not read_pos:
            continue
        for hi, hap in enumerate(haplotypes):
            shared_positions = read_pos & hap_positions[hi]
            n_shared = len(shared_positions)
            if n_shared == 0:
                continue
            mismatches = 0
            for pos in shared_positions:
                if read.alleles.get(pos) != hap.consensus.get(pos):
                    mismatches += 1
            if (mismatches / n_shared) < identity_distance:
                counts[hi] += 1

    return n_reads, counts


@dataclass
class RescueStatistic:
    """Statistics for a single haplotype rescue event."""

    sample: str  # Timepoint where rescue occurred
    rescued_timepoint: str  # Timepoint label for the rescued read/haplotype (usually same as sample)
    contig: str
    window_start: int
    track_id: str
    was_rescued: bool
    original_weight: float
    rescued_weight: float
    donor_timepoint: str  # Timepoint that provided the anchor
    anchor_distance: float
    n_shared_with_anchor: int
    n_mismatched_with_anchor: int
    reason: str = ""  # Debug reason for rescue outcome


@dataclass
class RescuedReadInfo:
    """Per-read information for rescue events."""

    read_name: str
    sample: str  # Timepoint where rescue occurred
    contig: str
    window_start: int
    window_end: int
    donor_timepoint: str  # Timepoint that provided the anchor haplotype
    n_snps_agree: int  # read vs rescued haplotype
    n_snps_disagree: int
    n_snps_total: int  # agree + disagree
    rescued_haplotype_weight: float


# =============================================================================
# LOG-PROBABILITY CACHE
# =============================================================================


class LogProbCache:
    """Cache of log-probability computations, avoiding redundant 10**(-Q/10) calls.

    Mismatch probability is spread uniformly over the n_alleles - 1 non-matching states. The
    alphabet is always {A,C,G,T,DEL,INS} (n_alleles=6) because indels are always processed
    (invariant, docs/MUTATION_HANDLING.md); SV pseudo-alleles reuse the same 6-state model.
    """

    def __init__(self, max_q: int = 60, n_alleles: int = 4):
        """Precompute log probabilities for all Q scores."""
        if n_alleles < 2:
            raise ValueError(f"n_alleles must be >= 2, got {n_alleles}")
        self._log_match = np.zeros(max_q + 1)
        self._log_mismatch = np.zeros(max_q + 1)
        self.n_alleles = n_alleles
        # Junk tables depend on junk_divergence_rate, which is config, not alphabet.
        self._junk_tables: dict[float, tuple[np.ndarray, np.ndarray]] = {}

        denom = float(n_alleles - 1)
        for q in range(max_q + 1):
            p_err = 10 ** (-q / 10.0)
            self._log_match[q] = np.log(1.0 - p_err + 1e-12)
            self._log_mismatch[q] = np.log(p_err / denom + 1e-12)

    def log_prob_base(self, hap_base: str, read_base: str, q: int) -> float:
        """Get log probability from cache."""
        q = min(q, len(self._log_match) - 1)
        if hap_base == read_base:
            return self._log_match[q]
        return self._log_mismatch[q]

    def log_mismatch(self, q: int) -> float:
        """``log P(obs | the true allele is something else)`` at quality *q*."""
        return float(self._log_mismatch[min(q, len(self._log_mismatch) - 1)])

    def log_odds(self, q: int) -> float:
        """``log P(obs | agree) - log P(obs | disagree)`` at quality *q*: the weight a
        consensus vote uses. Spans 6.20 nats at Q20 to 15.42 at Q60, so a confident base
        outvotes several unreliable ones.
        """
        q = min(q, len(self._log_match) - 1)
        return float(self._log_match[q] - self._log_mismatch[q])

    def junk_tables(self, p_div: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-quality (log P(obs == REF), log P(obs == a given non-REF)) under the junk
        model, cached per divergence rate.

        Junk is a read off a genome diverged from the reference at p_div, OBSERVED through the
        same per-base error channel the haplotype components use, so the two effects compose
        here rather than junk being scored at a flat rate. The boundary stays quality-dependent
        (junk_divergence_rate is the prior on divergence, not a decision threshold).
        Crossover measurements: docs/design/core.md.
        """
        cached = self._junk_tables.get(p_div)
        if cached is None:
            n = float(self.n_alleles)
            d = float(p_div)
            q = np.arange(len(self._log_match))
            e = 10.0 ** (-q / 10.0)
            p_ref = (1.0 - d) * (1.0 - e) + d * e / (n - 1.0)
            p_alt = (
                (1.0 - d) * e / (n - 1.0)
                + (d / (n - 1.0)) * (1.0 - e)
                + (n - 2.0) * (d / (n - 1.0)) * (e / (n - 1.0))
            )
            cached = (np.log(p_ref + 1e-12), np.log(p_alt + 1e-12))
            self._junk_tables[p_div] = cached
        return cached


# Single global cache. The alphabet is always {A,C,G,T,DEL,INS} (indels are
# always processed — invariant), so the 6-state error model is used everywhere.
_LOG_PROB_CACHE = LogProbCache(n_alleles=6)

# log P(read allele | a component with no allele at this position): marginalising the
# unknown allele over the alphabet gives exactly 1/n_alleles. Every mixture component must be
# scored over the SAME site set or the softmax that makes gamma is not a posterior (worked
# example of the footprint asymmetry this fixes: docs/design/core.md).
_LOG_MISSING_SITE = float(np.log(1.0 / _LOG_PROB_CACHE.n_alleles))

# Reference base an SV pseudo-site carries: an SV has no single reference allele, so the
# anchor gets a placeholder no read can equal, which is why the junk model steps over these sites.
_SV_PLACEHOLDER_REF = "N"


def _read_support(read: Read, snv_set: set[int]) -> list[int]:
    """The sites this read is scored at: its own calls that are window variant sites.

    One list per read, shared by every mixture component, which is what makes the E-step
    a comparison of like with like.
    """
    return [p for p in read.alleles if p in snv_set]


def _log_prob_read_hap(
    read: Read, consensus: dict[int, str], support: list[int], default_q: int
) -> float | None:
    """``log P(read | haplotype)`` over *support*.

    A support position the consensus does not reach is explicit missing data, not an
    absent term: it costs ``_LOG_MISSING_SITE``, the same constant every other component
    pays there. ``None`` means the haplotype says nothing about this read at all (no
    shared called site), which is the one case where it must not compete for it.
    """
    log_prob = 0.0
    overlap = 0
    for pos in support:
        hap_base = consensus.get(pos)
        if hap_base is None:
            log_prob += _LOG_MISSING_SITE
            continue
        q = read.quals.get(pos, default_q)
        log_prob += _LOG_PROB_CACHE.log_prob_base(hap_base, read.alleles[pos], q)
        overlap += 1
    return log_prob if overlap > 0 else None


def _log_prob_read_junk(
    read: Read,
    support: list[int],
    ref_alleles: dict[int, str],
    p_div: float,
    default_q: int,
) -> float:
    """``log P(read | junk)`` over the same *support* the haplotype components use.

    Junk is a genome diverged from the reference at p_div, so a site with no usable reference
    allele carries no junk evidence and takes the missing-data term: sites the loader never
    anchored, and SV pseudo-sites (anchor "N"). SV pseudo-sites must be stepped over or junk
    pays a phantom mismatch at every SV anchor. Split-read breakpoints are NOT excluded: their
    anchor holds CONTINUOUS, which a read crossing with an unbroken alignment genuinely matches.
    Measurement: docs/design/core.md.
    """
    log_ref, log_alt = _LOG_PROB_CACHE.junk_tables(p_div)
    max_q = len(log_ref) - 1
    log_prob = 0.0
    for pos in support:
        ref_base = ref_alleles.get(pos)
        if ref_base is None or ref_base == _SV_PLACEHOLDER_REF:
            log_prob += _LOG_MISSING_SITE
            continue
        q = min(read.quals.get(pos, default_q), max_q)
        log_prob += log_ref[q] if read.alleles[pos] == ref_base else log_alt[q]
    return float(log_prob)

def _em_read_tensors(reads, support, ref_alleles, p_div, default_q, cache):
    """Build the E-step tensors that are CONSTANT across EM iterations.

    Reads, support sites and qualities do not change during EM (only consensuses do), so the
    integer allele matrix, per-(read, site) log_match/log_mismatch, and junk log-likelihood
    are built ONCE here. Vectorised replacement for the per-(read, hap) _log_prob_read_hap /
    _log_prob_read_junk loop; numerically identical to summing those calls (float order aside).
    Returns tensors plus ``alleles``, the shared string->int encoder so read/hap/ref alleles
    compare as integers.
    """
    sites = sorted({p for sup in support for p in sup})
    site_idx = {p: s for s, p in enumerate(sites)}
    n_reads, n_sites = len(reads), len(sites)
    max_q = len(cache._log_match) - 1

    alleles: dict[str, int] = {}

    def code(a: str) -> int:
        c = alleles.get(a)
        if c is None:
            c = len(alleles)
            alleles[a] = c
        return c

    read_code = np.full((n_reads, n_sites), -1, dtype=np.int32)
    read_q = np.zeros((n_reads, n_sites), dtype=np.int32)
    for i, r in enumerate(reads):
        r_alleles, r_quals = r.alleles, r.quals
        for p in support[i]:
            s = site_idx[p]
            read_code[i, s] = code(r_alleles[p])
            read_q[i, s] = min(r_quals.get(p, default_q), max_q)

    sup_mask = read_code >= 0
    lm = cache._log_match[read_q]        # (n_reads, n_sites) log P(match) at each q
    lmm = cache._log_mismatch[read_q]    # (n_reads, n_sites) log P(mismatch)

    # Junk component: constant across iterations. A site with no reference allele,
    # or an SV placeholder, takes the missing-data term (see _log_prob_read_junk).
    log_ref, log_alt = cache.junk_tables(p_div)
    ref_code = np.full(n_sites, -1, dtype=np.int32)
    for p, base in ref_alleles.items():
        s = site_idx.get(p)
        if s is not None and base != _SV_PLACEHOLDER_REF:
            ref_code[s] = code(base)
    ref_present = ref_code >= 0
    jr = log_ref[read_q]
    ja = log_alt[read_q]
    junk_contrib = np.where(
        ref_present[None, :],
        np.where(read_code == ref_code[None, :], jr, ja),
        _LOG_MISSING_SITE,
    )
    logl_junk = np.where(sup_mask, junk_contrib, 0.0).sum(axis=1)

    return site_idx, alleles, code, read_code, sup_mask, lm, lmm, logl_junk


def _em_logl_hap(haplotypes, site_idx, code, read_code, sup_mask, lm, lmm):
    """Vectorised ``log P(read | hap)`` matrix, one column per haplotype.

    Identical to _log_prob_read_hap per (read, hap): a support site the consensus does not
    reach costs _LOG_MISSING_SITE; a covered site is log_match/log_mismatch at the read's
    quality; a read with zero shared covered sites gets -inf (must not compete), matching the
    None return. ``code`` extends the encoder with hap-only alleles (which can only mismatch).
    """
    n_reads, n_sites = read_code.shape
    n_haps = len(haplotypes)
    logl = np.full((n_reads, n_haps), -np.inf)
    for k, hap in enumerate(haplotypes):
        hc = np.full(n_sites, -1, dtype=np.int32)
        for p, base in hap.consensus.items():
            s = site_idx.get(p)
            if s is not None:
                hc[s] = code(base)
        present = hc >= 0
        contrib = np.where(
            present[None, :],
            np.where(read_code == hc[None, :], lm, lmm),
            _LOG_MISSING_SITE,
        )
        total = np.where(sup_mask, contrib, 0.0).sum(axis=1)
        overlap = (sup_mask & present[None, :]).sum(axis=1)
        logl[:, k] = np.where(overlap > 0, total, -np.inf)
    return logl


def _em_hap_consensus(gamma_k, read_code, sup_mask, lo, lmm, n_alleles,
                      sites, dec, snv_pos, min_gamma):
    """Vectorised weighted-vote consensus for one haplotype (M-step).

    Identical to the per-read voting loop: each read with ``gamma >= min_gamma``
    contributes, at each of its support sites, ``gamma * log_odds(q)`` toward the
    allele it calls and ``gamma * log_mismatch(q)`` to the position's mismatch
    floor; an allele is taken only where ``floor + its vote`` beats calling
    nothing (``cover * _LOG_MISSING_SITE``). Reuses the read x site tensors built
    once by :func:`_em_read_tensors`. Returns the consensus dict, or ``None`` if no
    read voted (the "if not allele_votes: continue" case).
    """
    w = np.where(gamma_k >= min_gamma, gamma_k, 0.0)
    ew = w[:, None] * sup_mask                # effective weight per (read, site)
    if not ew.any():
        return None
    cover = ew.sum(axis=0)                    # (n_sites,)
    floor = (ew * lmm).sum(axis=0)            # (n_sites,)
    contrib = ew * lo                         # (n_reads, n_sites) vote weight
    n_sites = read_code.shape[1]
    votes = np.full((n_sites, n_alleles), -np.inf)
    for a in range(n_alleles):
        m = read_code == a
        if not m.any():
            continue
        col = np.where(m, contrib, 0.0).sum(axis=0)
        votes[(m & sup_mask).any(axis=0), a] = col[(m & sup_mask).any(axis=0)]
    best = votes.argmax(axis=1)
    best_val = votes[np.arange(n_sites), best]
    keep = np.isfinite(best_val) & (floor + best_val > cover * _LOG_MISSING_SITE)
    return {
        sites[s]: dec[int(best[s])]
        for s in np.nonzero(keep)[0]
        if sites[s] in snv_pos
    }


# Padding (bp) around an SV breakpoint anchor when deciding whether a read's
# reference span brackets it. Absorbs Sniffles breakpoint imprecision
# (CIPOS/CIEND) and soft-clip edges. See strainphase.sv_encoding.
_SV_ANCHOR_PAD = 50


# =============================================================================
# I/O FUNCTIONS - LAZY LOADING
# =============================================================================

# Canonical single-base alleles. Anything else at a substitution offset (N,
# IUPAC codes, symbolic) is counted and logged rather than silently taken.
_ACGT = frozenset("ACGT")


def _atomize_allele(pos: int, ref: str, alt: str) -> list[tuple[int, str, str, str]]:
    """Decompose one REF/ALT pair into atomic variants ``(pos, ref, alt, type)``.

    The single uniform decomposition applied to every allele:

    * 1bp REF / 1bp ALT              -> one ``snv``
    * equal-length multi-base (MNP)  -> one ``snv`` per differing offset
    * REF longer  (net deletion)     -> one ``del`` (positions trusted as-is)
    * ALT longer  (net insertion)    -> one ``ins`` (positions trusted as-is)

    INVARIANT: indels and MNPs are ALWAYS decomposed and kept (docs/MUTATION_HANDLING.md).
    Non-ACGT bases at a substitution offset yield no primitive; the empty result is counted
    by the caller, never dropped silently.
    """
    ref = ref.upper()
    alt = alt.upper()
    if ref == alt:
        return []
    lr, la = len(ref), len(alt)
    if lr == la:
        # SNV (lr == 1) or MNP (lr > 1): one SNV per differing, canonical offset.
        return [
            (pos + i, ref[i], alt[i], "snv")
            for i in range(lr)
            if ref[i] != alt[i] and ref[i] in _ACGT and alt[i] in _ACGT
        ]
    # Length-changing: a single indel primitive (del if REF longer, else ins).
    return [(pos, ref, alt, "del" if lr > la else "ins")]


def _log_load_summary(vcf_path: str, contig_id: str | None, stats: dict[str, int]) -> None:
    """Log a full per-load accounting so no variant is ever dropped silently.

    Emits one INFO line with sites loaded per type and every skip reason with a
    non-zero count. If any records were skipped, also emits a one-line WARNING
    naming the reasons, so a reviewer scanning logs sees the loss immediately.
    """
    loaded = {
        k[len("loaded_") :]: v for k, v in stats.items() if k.startswith("loaded_")
    }
    n_sites = sum(loaded.values())
    skips = {
        k: v for k, v in stats.items() if k.startswith("skip_") or k in (
            "position_conflict",
            "duplicate_site",
            # These must be listed or the loss goes unreported.
            "allele_collapsed_same_key",
            "anchor_base_conflict",
        )
    }
    where = f" [{contig_id}]" if contig_id else ""
    loaded_str = " ".join(f"{k}={v}" for k, v in sorted(loaded.items())) or "none"
    logging.info(
        # "edits" not "sites": one position may hold several, so per-kind counts sum to more.
        "load_snvs%s %s: %d records seen -> %d edits at %d positions (%s); "
        "mnp_atomized=%d multiallelic_records=%d",
        where,
        os.path.basename(vcf_path),
        stats.get("records_seen", 0),
        n_sites,
        stats.get("n_positions", 0),
        loaded_str,
        stats.get("mnp_atomized", 0),
        stats.get("multiallelic_records", 0),
    )
    dropped = {k: v for k, v in skips.items() if v}
    if dropped:
        logging.warning(
            "load_snvs%s %s: %d record/allele rejections (NOT silent) -> %s",
            where,
            os.path.basename(vcf_path),
            sum(dropped.values()),
            " ".join(f"{k}={v}" for k, v in sorted(dropped.items())),
        )


# One parsed VCF per (path, contig, sample, gate settings). A longitudinal run calls
# process_contig once per sample, and under a cohort union VCF each re-parses the identical
# file; keyed on the settings that change what is kept, so a config change still re-parses.
# Bounded because a run touches one contig at a time. Measurement: docs/design/core.md.
_SNV_CACHE: dict[tuple, tuple] = {}
_SNV_CACHE_MAX = 8


def _copy_snv_tables(tables: tuple) -> tuple:
    """Shallow-copy every container in a parsed-VCF tuple.

    One copy per container suffices: mutating callers (process_contig's SV merge,
    iter_windows_lazy's breakpoint registration) append to the position list and assign into
    the top-level dicts; nobody reaches del_span/ins_len's inner sets, which stay shared.
    Written by type so a caller stubbing the loader with a shorter tuple still gets copies.
    """
    out = []
    for t in tables:
        if isinstance(t, list):
            out.append(list(t))
        elif isinstance(t, dict):
            out.append(dict(t))
        elif isinstance(t, set):
            out.append(set(t))
        else:
            out.append(t)
    return tuple(out)


def load_snvs(
    vcf_path: str,
    contig_id: str | None = None,
    sample_name: str | None = None,
    config: HaplotyperConfig = DEFAULT_CONFIG,
):
    """Cached wrapper (see _SNV_CACHE). Returns a FRESH SHALLOW COPY of the parsed tables on
    every call, hit or miss, so a caller may mutate what it gets back.

    It must: process_contig and iter_windows_lazy append SV/breakpoint pseudo-sites into the
    returned tables, so handing back the cached objects would leak one sample's SV anchors
    into the next under a shared union VCF. History: docs/design/core.md.
    """
    key = (vcf_path, contig_id, sample_name, config.min_depth_site, config.af_range,
           config.process_indels if hasattr(config, "process_indels") else None)
    hit = _SNV_CACHE.get(key)
    if hit is not None:
        return _copy_snv_tables(hit)
    out = _load_snvs_uncached(vcf_path, contig_id, sample_name, config)
    if len(_SNV_CACHE) >= _SNV_CACHE_MAX:
        _SNV_CACHE.pop(next(iter(_SNV_CACHE)))
    _SNV_CACHE[key] = out
    return _copy_snv_tables(out)



def _load_snvs_uncached(
    vcf_path: str,
    contig_id: str | None = None,
    sample_name: str | None = None,
    config: HaplotyperConfig = DEFAULT_CONFIG,
) -> tuple[
    list[int],
    dict[int, str],
    dict[int, int],
    dict[int, float | None],
    dict[int, str],
    dict[int, frozenset[str]],
    dict[int, set[tuple[int, int]]],
    dict[int, set[int]],
]:
    """Load variants from a VCF as atomic sites: SNVs (MNP blocks split per position) and
    indels, each site typed ``"snv"``/``"del"``/``"ins"``. Indels and multi-allelic sites are
    ALWAYS kept (invariant, docs/MUTATION_HANDLING.md).

    The caller is trusted: no realignment or fuzz-matching. Run ``bcftools norm -f REF``
    upstream so placement is canonical.

    Returns
    -------
    snv_pos       Sorted variant positions (1-based VCF anchor).
    ref_alleles   Single-base REFERENCE base at the anchor, NOT the record's full REF string
                  (reads carry one base or an indel token, so a multi-base REF could never
                  match and made every read a mismatch at every deletion site).
    depth, af     Per-position depth and alt AF (may be None). AF is Number=A, so a
                  multi-allelic position holds the FIRST allele's frequency; nothing
                  downstream reads it (the per-ALT values drive the af_range gate).
    site_type     Per-position type of the FIRST variant registered (kept a plain str so
                  ``site_type[p] == "sv"`` tests hold; ``"sv"`` is added later by the SV merge).
    site_kinds    Per-position set of ALL kinds registered, e.g. ``{"snv", "del"}``.
    del_span      Per deletion site: SET of inclusive 1-based ``(start, end)`` footprints;
                  each becomes its own ``DEL<len>`` allele.
    ins_len       Per insertion site: SET of inserted lengths; each its own ``INS<len>``.

    Nothing is dropped silently: every rejection (FILTER!=PASS, missing/low DP, optional AF
    band) is counted and logged. FILTER/DP are record-level; the AF band is per-ALLELE, so it
    rejects only the allele outside it, never a multi-allelic record's other alleles. A
    position may hold more than one variant, each its own allele downstream. One collapse
    remains, counted as ``allele_collapsed_same_key``: two insertions of the same length at one
    anchor differing only in inserted sequence (token is ``INS<len>``). Details: docs/design/core.md.
    """
    if not HAS_PYSAM:
        raise ImportError("pysam required for VCF parsing")

    snv_pos: list[int] = []
    seen_pos: set[int] = set()
    ref_alleles: dict[int, str] = {}
    depth: dict[int, int] = {}
    af: dict[int, float | None] = {}
    site_type: dict[int, str] = {}
    site_kinds: dict[int, set[str]] = {}
    del_span: dict[int, set[tuple[int, int]]] = {}
    ins_len: dict[int, set[int]] = {}

    # Full accounting so no variant is ever dropped silently.
    stats: dict[str, int] = defaultdict(int)

    def _register(apos: int, aref: str, aalt: str, stype: str, dp, site_af) -> bool:
        """Record one atomic variant; returns True if it added a new allele-defining edit.

        A position may hold more than one variant, each its own allele downstream.
        ``ref_alleles[apos]`` holds the single-base anchor base, not the record's full REF
        string (see load_snvs).
        """
        if apos <= 0:
            stats["skip_out_of_bounds"] += 1
            return False

        anchor = aref[0]
        if apos not in seen_pos:
            seen_pos.add(apos)
            snv_pos.append(apos)
            ref_alleles[apos] = anchor
            depth[apos] = dp
            af[apos] = site_af
            site_type[apos] = stype  # FIRST kind wins; see site_kinds for all of them
            site_kinds[apos] = set()
        elif ref_alleles[apos] != anchor:
            # Records disagreeing on a position's reference base = inconsistent input.
            stats["anchor_base_conflict"] += 1

        # After bcftools norm, indels are left-anchored: REF starts with the anchor base(s)
        # matching ALT, then deletes/inserts the trailing bases. Positions are trusted exactly.
        added = False
        if stype == "del":
            del_start = apos + len(aalt)
            del_end = apos + len(aref) - 1
            if del_end >= del_start:
                spans = del_span.setdefault(apos, set())
                added = (del_start, del_end) not in spans
                spans.add((del_start, del_end))
            else:
                stats["skip_degenerate_del"] += 1
        elif stype == "ins":
            lens = ins_len.setdefault(apos, set())
            ilen = len(aalt) - len(aref)
            added = ilen not in lens
            if not added:
                # Same anchor and inserted length, different sequence: INS<len> cannot tell
                # them apart. Counted, not silent.
                stats["allele_collapsed_same_key"] += 1
            lens.add(ilen)
        else:  # snv — multiple ALTs at one position need no extra bookkeeping,
            # the read supplies whichever base it carries.
            added = "snv" not in site_kinds[apos]
            if not added:
                stats["duplicate_site"] += 1

        if stype == "del" and not added:
            stats["duplicate_site"] += 1
        site_kinds[apos].add(stype)
        if added:
            stats[f"loaded_{stype}"] += 1
        return added

    vcf = pysam.VariantFile(vcf_path)

    n_samples = len(vcf.header.samples)
    if n_samples > 1 and sample_name is None:
        raise ValueError(
            f"VCF has {n_samples} samples but no sample_name specified. "
            f"Available: {list(vcf.header.samples)}"
        )

    # A contig the VCF never declares carries no variants (callers emit ##contig lines only
    # for contigs they called on). pysam raises "invalid contig" on fetch, so ask first.
    if contig_id and contig_id not in vcf.header.contigs:
        records = ()
        stats["contig_not_in_vcf"] = 1
    elif contig_id:
        records = vcf.fetch(contig=contig_id)
    else:
        records = vcf.fetch()

    for record in records:
        stats["records_seen"] += 1

        # Filter check (counted, not silent).
        if record.filter.keys() and "PASS" not in record.filter.keys():
            stats["skip_filter_not_pass"] += 1
            continue

        alts = record.alts
        if alts is None or len(alts) == 0:
            stats["skip_no_alt"] += 1
            continue

        ref = record.ref

        # Get sample (for DP/AD fallbacks on caller VCFs that carry them there).
        if sample_name is not None:
            sample = record.samples[sample_name]
        elif n_samples > 0:
            sample = record.samples[0]
        else:
            sample = None

        site_depth = None
        if "DP" in record.info:
            site_depth = record.info["DP"]
        elif sample is not None and "DP" in sample:
            site_depth = sample["DP"]

        if site_depth is None:
            stats["skip_no_depth"] += 1
            continue
        if site_depth < config.min_depth_site:
            stats["skip_low_depth"] += 1
            continue

        # Extract AF. INFO/AF is Number=A — one frequency PER ALT — so it is kept as a
        # per-ALT sequence rather than collapsed to element [0]. AD is Number=R (REF
        # first), so ALT i's frequency is ad[i + 1] over the total.
        alt_afs: tuple[float, ...] | None = None
        if "AF" in record.info:
            raw_af = record.info["AF"]
            alt_afs = tuple(raw_af) if isinstance(raw_af, tuple) else (raw_af,)
        elif sample is not None and "AD" in sample:
            ad = sample["AD"]
            if ad and len(ad) >= 2 and sum(ad) > 0:
                ad_total = sum(ad)
                alt_afs = tuple(a / ad_total for a in ad[1:])

        if len(alts) > 1:
            stats["multiallelic_records"] += 1

        # Decompose EVERY allele into atomic primitives via the one shared helper. All
        # mutation types and all (>2) alleles are ALWAYS kept; nothing here can turn that off.
        for alt_idx, alt in enumerate(alts):
            # The AF band is a PER-ALLELE gate: applied here, not once per record, so it
            # rejects only the allele outside the band, not the record's other alleles.
            alt_af = (
                alt_afs[alt_idx]
                if alt_afs is not None and alt_idx < len(alt_afs)
                else None
            )
            if config.af_range is not None and alt_af is not None:
                if not (config.af_range[0] <= alt_af <= config.af_range[1]):
                    stats["skip_af_band"] += 1
                    continue
            if alt is None or alt == "*" or alt.startswith("<"):
                # Spanning-deletion star / symbolic (gVCF <*>, <DEL> …): no
                # concrete allele to phase against.
                stats["skip_symbolic_allele"] += 1
                continue
            prims = _atomize_allele(record.pos, ref, alt)
            if not prims:
                # Identical (no-op) or non-ACGT ambiguity code — counted here.
                stats["skip_noncanonical"] += 1
                continue
            if len(ref) == len(alt) and len(ref) > 1:
                stats["mnp_atomized"] += 1
            for apos, aref, aalt, stype in prims:
                _register(apos, aref, aalt, stype, site_depth, alt_af)

    vcf.close()

    stats["n_positions"] = len(seen_pos)
    _log_load_summary(vcf_path, contig_id, stats)
    return (
        snv_pos, ref_alleles, depth, af, site_type,
        {p: frozenset(k) for p, k in site_kinds.items()},
        del_span, ins_len,
    )


def make_windows_lazy(
    bam_path: str,
    contig_id: str,
    contig_length: int,
    snv_positions: list[int],
    ref_alleles: dict[int, str],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    sample_id: str | None = None,
    site_type: dict[int, str] | None = None,
    site_kinds: dict[int, frozenset[str]] | None = None,
    del_span: dict[int, set[tuple[int, int]]] | None = None,
    ins_len: dict[int, set[int]] | None = None,
    sv_support: dict[int, dict[str, set[str]]] | None = None,
) -> list[Window]:
    """Eager wrapper around :func:`iter_windows_lazy` for callers wanting all of a contig's
    windows at once (tests, scripts). Prefer the iterator in production: materialising every
    window holds every window's reads at once, the single largest allocation on a dense contig.
    """
    return list(
        iter_windows_lazy(
            bam_path,
            contig_id,
            contig_length,
            snv_positions,
            ref_alleles,
            config,
            sample_id,
            site_type=site_type,
            site_kinds=site_kinds,
            del_span=del_span,
            ins_len=ins_len,
            sv_support=sv_support,
        )
    )


def iter_windows_lazy(
    bam_path: str,
    contig_id: str,
    contig_length: int,
    snv_positions: list[int],
    ref_alleles: dict[int, str],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    sample_id: str | None = None,
    site_type: dict[int, str] | None = None,
    site_kinds: dict[int, frozenset[str]] | None = None,
    del_span: dict[int, set[tuple[int, int]]] | None = None,
    ins_len: dict[int, set[int]] | None = None,
    sv_support: dict[int, dict[str, set[str]]] | None = None,
) -> Iterator[Window]:
    """Yield overlapping windows (50% overlap, step = window_size / 2) with lazy per-window
    read loading, so haplotypes can link across boundaries via shared SNVs.

    O(W * reads_per_window) time, O(window) memory: the caller processes and releases each
    window before the next one's reads are pulled from the BAM.
    """
    if not HAS_PYSAM:
        raise ImportError("pysam required for BAM parsing")

    snv_pos_sorted = sorted([p for p in snv_positions if 0 < p <= contig_length])
    if not snv_pos_sorted:
        return

    # No RNG here any more: the per-window read cap now selects by a stable hash of the
    # read id (see the subsample below), so window contents no longer depend on a draw.
    bam = pysam.AlignmentFile(bam_path, "rb")

    step_size = config.window_size // 2
    window_idx = 0

    for start in range(1, contig_length + 1, step_size):
        end = min(start + config.window_size, contig_length + 1)

        # No size-based window skipping: small/trailing windows are kept and filtered
        # downstream by the empty-window check / min_reads_per_window.

        # Bisect the [start, end) slice of sorted snv_pos_sorted instead of scanning all
        # variants per window (the linear scan dominated self-time; see docs/design/core.md).
        lo = bisect.bisect_left(snv_pos_sorted, start)
        hi = bisect.bisect_left(snv_pos_sorted, end)
        window_snvs = snv_pos_sorted[lo:hi]

        if len(window_snvs) < config.min_snvs_per_window:
            continue

        snv_set = set(window_snvs)
        reads = []

        # Partition window sites by type. Sites without a recorded type default to "snv".
        st = site_type or {}
        # Fall back to one kind per position (not an empty map) so a caller passing site_type
        # but not site_kinds keeps single-kind behaviour rather than losing indel parsing.
        sk = site_kinds or {p: frozenset({t}) for p, t in st.items()}
        ds = del_span or {}
        il = ins_len or {}
        sv_sup = sv_support or {}
        # Per-window indel index (O(1) lookups at use time):
        #   del_key_to_pos: (D-op start_1b, D-op length) -> indel site pos
        #   ins_key_to_pos: (I-op anchor_1b, inserted length) -> indel site pos
        # Keys are SIZE-specific, so a <len>-bp indel is its own DEL<len>/INS<len> allele.
        # Membership is by site_kinds, not site_type: a position may be both SNV and indel, so
        # it must reach the CIGAR walk; reads without the indel fall through to the anchor base.
        indel_site_set = {
            p for p in window_snvs
            if sk.get(p, frozenset()) & {"del", "ins"}
        }
        # SV pseudo-sites: the "present" allele is the UNIQUE event ID (not a generic INS/DEL
        # token), so reads cluster only on the identical event. Kept separate from
        # indel_site_set so the D/I exact-match logic never touches them.
        sv_site_set = {p for p in window_snvs if st.get(p, "snv") == "sv"}  # SV anchors never co-occur with a called variant (see process_contig)
        # Sites parsed outside the base-by-base SNV loop (indels + SVs).
        special_site_set = indel_site_set | sv_site_set
        del_key_to_pos: dict[tuple[int, int], int] = {}
        ins_key_to_pos: dict[tuple[int, int], int] = {}
        for p in indel_site_set:
            # ds[p]/il[p] are SETS: one anchor may carry several edits, each its own
            # DEL<len>/INS<len> allele. A position can appear in both loops.
            for d_start, d_end in ds.get(p, ()):
                del_key_to_pos[(d_start, d_end - d_start + 1)] = p
            for ilen in il.get(p, ()):
                # (anchor, inserted length): a read matches only a same-size insertion.
                ins_key_to_pos[(p, ilen)] = p

        # pysam fetch uses 0-based coordinates
        for aln in bam.fetch(contig_id, start - 1, end - 1):
            # SECONDARY stays out (an alternative placement of counted sequence); SUPPLEMENTARY
            # is kept as a different PART of the same molecule, merged into one Read below.
            if aln.is_secondary or aln.is_unmapped:
                continue
            if aln.mapping_quality < config.min_mapq:
                continue

            # Physical read-to-window overlap. pysam reference_start/_end are 0-based
            # half-open; the window is [start-1, end-1) in the same frame.
            a_start = aln.reference_start
            a_end = aln.reference_end
            if a_end is None:
                continue
            win_overlap = min(a_end, end - 1) - max(a_start, start - 1)
            if win_overlap < config.min_read_window_overlap_bp:
                # A read barely clipping the window edge carries almost no information yet
                # would count like a full-span read in n_reads_examined / the abundance denom.
                continue

            r = Read(
                id=aln.query_name,
                contig=contig_id,
                mapq=aln.mapping_quality,
                sample=sample_id,
                ref_start=a_start + 1,  # store 1-based to match allele positions
                ref_end=a_end + 1,
            )

            query_seq = aln.query_sequence
            query_qual = aln.query_qualities

            if query_seq is None:
                continue

            if query_qual is None:
                WarningThrottler.warn_once(
                    "no_qual",
                    f"Some reads lack quality scores. Using default Q{config.default_base_quality}.",
                )

            # Extract alleles at SNV positions via a TARGETED CIGAR walk: emit only at this
            # read's SNV positions, O(variants on read) not O(read length), byte-identical to
            # get_aligned_pairs. Indel/SV sites are excluded here (handled by the scan below).
            has_overlap = False
            # SNV-only targets, sorted, so the walk advances monotonically (break sites are
            # appended only after this loop).
            snv_targets = (
                window_snvs
                if not special_site_set
                else [p for p in window_snvs if p not in special_site_set]
            )
            n_targets = len(snv_targets)
            if n_targets:
                ti = 0
                ref_cur = aln.reference_start  # 0-based
                q_cur = 0
                for op, length in (aln.cigartuples or ()):
                    if ti >= n_targets:
                        break
                    if op in (0, 7, 8):  # M / = / X consume ref and query together
                        seg_end = ref_cur + length  # 0-based, exclusive
                        # Targets before this segment fell in a D/N gap: no query base, no call.
                        while ti < n_targets and snv_targets[ti] - 1 < ref_cur:
                            ti += 1
                        while ti < n_targets and snv_targets[ti] - 1 < seg_end:
                            pos = snv_targets[ti]
                            qp = q_cur + (pos - 1 - ref_cur)
                            qual = (
                                query_qual[qp] if query_qual else config.default_base_quality
                            )
                            if qual >= config.min_base_quality:
                                r.alleles[pos] = query_seq[qp]
                                r.quals[pos] = qual
                                has_overlap = True
                            ti += 1
                        ref_cur, q_cur = seg_end, q_cur + length
                    elif op == 1:  # I consumes query only
                        q_cur += length
                    elif op in (2, 3):  # D / N consume ref only
                        ref_cur += length
                    elif op == 4:  # S (soft clip) consumes query only
                        q_cur += length
                    # H (5) / P (6) consume neither

            # Extract alleles at indel sites. A read carries the variant iff its CIGAR has the
            # matching indel op (positions trusted exactly):
            #   DEL: a D op of exactly the deleted length -> "DEL<len>"
            #   INS: an I op of exactly the inserted length at the anchor -> "INS<len>"
            # Size is part of the allele, so a <len>-bp indel clusters only with same-size
            # reads. Else a matched base over the anchor is the read's "vote against"; else no call.
            if special_site_set and aln.cigartuples:
                # Quality for a DEL<len>/INS<len>/event-id token. These are not base calls, so
                # they take default_base_quality, NOT MAPQ (which the EM would misread as a
                # per-base Phred, over-weighting indels; see docs/design/core.md). MAPQ lives on Read.mapq.
                token_qual = config.default_base_quality
                # Single CIGAR walk: collect indel events and the matched-base
                # ref->query mapping for indel/SV-anchor positions only.
                ref_cursor = aln.reference_start  # 0-based
                query_cursor = 0
                # Calls at indel anchors; REF-base fallback resolved at the end.
                indel_calls: dict[int, tuple[str, int]] = {}
                # ref_pos_1b -> query_idx for indel/SV anchors covered by M/=/X.
                anchor_qpos: dict[int, int] = {}

                for op, length in aln.cigartuples:
                    if op in (0, 7, 8):  # M / = / X — consumes both
                        # Remember the query index of each indel/SV anchor for the matched-base fallback.
                        op_start_1b = ref_cursor + 1
                        op_end_1b_excl = ref_cursor + length + 1
                        for pos in special_site_set:
                            if op_start_1b <= pos < op_end_1b_excl:
                                anchor_qpos[pos] = query_cursor + (pos - op_start_1b)
                        ref_cursor += length
                        query_cursor += length
                    elif op == 2:  # D
                        key = (ref_cursor + 1, length)
                        pos = del_key_to_pos.get(key)
                        if pos is not None:
                            # per-edit token: a <len>-bp deletion is its own allele.
                            indel_calls[pos] = (f"DEL{length}", token_qual)
                        ref_cursor += length
                    elif op == 1:  # I
                        # Anchor = ref_cursor in 1-based terms; match on (anchor, inserted
                        # length) so a <len>-bp insertion is its own allele.
                        pos = ins_key_to_pos.get((ref_cursor, length))
                        if pos is not None:
                            indel_calls[pos] = (f"INS{length}", token_qual)
                        query_cursor += length
                    elif op == 3:  # N: consumes ref only
                        ref_cursor += length
                    elif op == 4:  # S: consumes query only
                        query_cursor += length
                    # op 5 (H) and 6 (P) consume neither.

                # Apply DEL/INS calls.
                for pos, (allele, mq) in indel_calls.items():
                    r.alleles[pos] = allele
                    r.quals[pos] = mq
                    has_overlap = True

                # Apply SV pseudo-site "present" calls: a read carries an event iff its name is
                # in THAT event's supporting-read set AND this alignment's span brackets the
                # anchor (small pad). The allele is the event ID (distinct events = multi-allelic);
                # the span check stops a read's OTHER split segment from crediting the event here.
                if sv_site_set and aln.reference_end is not None:
                    aln_start_1b = aln.reference_start + 1
                    aln_end_1b = aln.reference_end  # 1-based inclusive (None-guarded above)
                    for pos in sv_site_set:
                        events = sv_sup.get(pos)
                        if not events:
                            continue
                        if not (aln_start_1b - _SV_ANCHOR_PAD <= pos <= aln_end_1b + _SV_ANCHOR_PAD):
                            continue
                        for event_id, ev_reads in events.items():
                            if aln.query_name in ev_reads:
                                r.alleles[pos] = event_id
                                r.quals[pos] = token_qual
                                has_overlap = True
                                break

                # For indel/SV sites with no event call but a matched anchor
                # base, record the base as the read's "ref-like" (absent) vote.
                for pos, qpos in anchor_qpos.items():
                    if pos in r.alleles:  # already called (indel or SV present)
                        continue
                    base = query_seq[qpos]
                    qual = (
                        query_qual[qpos] if query_qual else config.default_base_quality
                    )
                    if qual >= config.min_base_quality:
                        r.alleles[pos] = base
                        r.quals[pos] = qual
                        has_overlap = True

            if has_overlap:
                reads.append(r)

        # Re-assemble split molecules into single reads, and register the breakpoints
        # they reveal as sites. Done before subsampling so the cap counts molecules.
        reads, break_sites = _merge_split_reads(reads, config, snv_set)
        for bp in break_sites:
            if bp not in snv_set and start <= bp < end:
                window_snvs.append(bp)
                snv_set.add(bp)
                ref_alleles[bp] = CONTINUOUS
                st[bp] = "sv"
        if break_sites:
            window_snvs.sort()

        # Subsample consistently across windows: keep the reads whose ids hash smallest, so two
        # overlapping windows pick the same reads from the molecules they share (an independent
        # per-window draw left ~2-3 shared where biology has hundreds; the linker depends on this).
        if config.max_reads_per_window and len(reads) > config.max_reads_per_window:
            reads = [
                r for _, _, r in sorted(
                    (_read_sort_hash(r.id, config.random_seed), i, r)
                    for i, r in enumerate(reads)
                )[: config.max_reads_per_window]
            ]

        # Window CREATION uses the lower rescue floor, so the [rescue, phasing) band still
        # exists and can receive a rescued anchor haplotype. De-novo PHASING is gated separately
        # in process_window() at the higher min_reads_per_window.
        if len(reads) < config.min_reads_for_rescue:
            continue

        w = Window(contig=contig_id, start=start, end=end, sample=sample_id, window_idx=window_idx)
        w.snv_pos = window_snvs
        # Guarded: a caller supplying positions without an anchor base must not take the
        # whole contig down here.
        w.ref_alleles = {p: ref_alleles[p] for p in window_snvs if p in ref_alleles}
        w.reads = reads
        # Carry site types through; the identity code needs them to exclude SV positions.
        w.site_type = {p: st[p] for p in window_snvs if p in st}
        yield w
        window_idx += 1

    bam.close()


# =============================================================================
# OPTIMIZED GRAPH INITIALIZER
# =============================================================================


class GraphInitializer:
    """Seed haplotypes by clustering the read-overlap graph into Louvain communities."""

    def __init__(self, config: HaplotyperConfig = DEFAULT_CONFIG):
        self.config = config

    def build_overlap_graph(self, window: Window) -> nx.Graph:
        """Build the read-overlap graph: an edge joins two reads within identity_distance."""
        graph = nx.Graph()
        reads = window.reads
        n_reads = len(reads)

        # one node per read (read indices)
        for i in range(n_reads):
            graph.add_node(i)

        # SNV positions each read covers (cached)
        pos_sets = window.get_read_position_sets()

        # Connect reads that share enough SNVs and agree closely.
        for i in range(n_reads):
            pos_i = pos_sets[i]
            if not pos_i:
                continue

            for j in range(i + 1, n_reads):
                pos_j = pos_sets[j]
                shared = pos_i & pos_j
                n_shared = len(shared)

                if n_shared < self.config.min_shared_snvs_for_edge:
                    continue

                r_i, r_j = reads[i], reads[j]

                # Physical read-read overlap. -1 = spans unknown (e.g. synthetic test reads),
                # so the gate is skipped rather than rejecting everything.
                ov = r_i.overlap_bp(r_j)
                if 0 <= ov < self.config.min_read_read_overlap_bp:
                    continue

                # Count mismatches with early exit. The gate is the rate:
                # floor(identity_distance * n_shared) (forces 0 mismatches below n_shared=50 at 2%).
                max_allowed = int(self.config.identity_distance * n_shared)
                mismatches = 0
                exceeded = False

                for p in shared:
                    if r_i.alleles[p] != r_j.alleles[p]:
                        mismatches += 1
                        if mismatches > max_allowed:
                            exceeded = True
                            break

                if not exceeded:
                    mismatch_frac = mismatches / n_shared
                    # Edge weight = #shared SNVs scaled by agreement (higher is better).
                    weight = (1.0 - mismatch_frac) * n_shared
                    graph.add_edge(i, j, weight=weight)

        return graph

    def derive_consensus(self, cluster_reads: list[Read], window: Window) -> dict[int, str]:
        """Derive consensus from cluster reads."""
        allele_counts = defaultdict(lambda: defaultdict(int))

        for r in cluster_reads:
            for pos, base in r.alleles.items():
                if window.start <= pos < window.end and pos in window.snv_pos:
                    allele_counts[pos][base] += 1

        # Consensus = most frequent allele at each SNV position.
        consensus = {}
        for pos in window.snv_pos:
            if pos in allele_counts:
                consensus[pos] = max(allele_counts[pos], key=allele_counts[pos].get)

        return consensus

    def get_initial_haplotypes(self, window: Window) -> tuple[list[Haplotype], list[int]]:
        """Initialize haplotypes using graph clustering."""
        graph = self.build_overlap_graph(window)

        if graph.number_of_edges() == 0:
            # No edges => no clustering signal; fall back to single consensus haplotype.
            consensus = self.derive_consensus(window.reads, window)
            if consensus:
                return [Haplotype(consensus=consensus, supporting_reads=len(window.reads))], [
                    len(window.reads)
                ]
            return [], []

        # Louvain community detection. random_state is passed explicitly: without it
        # python-louvain draws from the unseeded global `random`, making reruns disagree at ~1e-7.
        partition = community_louvain.best_partition(
            graph, weight="weight", random_state=self.config.random_seed
        )

        clusters = defaultdict(list)
        for node_idx, cluster_id in partition.items():
            clusters[cluster_id].append(window.reads[node_idx])

        # one consensus per cluster
        initial_haps = []
        cluster_sizes = []

        for _cluster_id, cluster_reads in clusters.items():
            if len(cluster_reads) < self.config.min_reads_per_cluster:
                continue

            consensus = self.derive_consensus(cluster_reads, window)
            if consensus:
                hap = Haplotype(consensus=consensus, supporting_reads=len(cluster_reads))
                initial_haps.append(hap)
                cluster_sizes.append(len(cluster_reads))

        return initial_haps, cluster_sizes


# =============================================================================
# OPTIMIZED EM ENGINE
# =============================================================================


class EMHaplotyper:
    """Quality-weighted EM refinement of window haplotypes, with a junk component."""

    def __init__(
        self,
        window: Window,
        initial_haplotypes: list[Haplotype],
        cluster_sizes: list[int] | None = None,
        config: HaplotyperConfig = DEFAULT_CONFIG,
    ):
        self.window = window
        self.haplotypes = initial_haplotypes
        self.cluster_sizes = cluster_sizes
        self.reads = window.reads
        self.config = config

        # Single 6-state alphabet {A,C,G,T,DEL,INS} — indels are always on.
        self._cache = _LOG_PROB_CACHE

        # The site set every component is scored over, fixed for the whole run: this is what
        # makes gamma a posterior (components compared on the same observations) and the
        # log-likelihood a function of the parameters alone.
        self._snv_set = set(window.snv_pos)
        self._support = [_read_support(r, self._snv_set) for r in self.reads]



    def run(self) -> tuple[list[Haplotype], np.ndarray, np.ndarray, float, bool, int]:
        """Run EM with cached log-probability computations."""
        haplotypes = self.haplotypes
        reads = self.reads
        n_reads = len(reads)
        n_haps = len(haplotypes)

        if n_haps == 0:
            # Degenerate case: only junk class.
            gamma = np.ones((n_reads, 1))
            pi = np.array([1.0])
            return [], gamma, pi, -np.inf, True, 0

        k_eff = n_haps + 1
        junk_idx = n_haps

        # Initialize mixture weights (pi): either from cluster sizes or uniform.
        if self.cluster_sizes:
            cluster_total = sum(self.cluster_sizes)
            junk_init = max(1, n_reads - cluster_total)
            pi = np.array(self.cluster_sizes + [junk_init], dtype=float)
            pi /= pi.sum()
        else:
            pi = np.ones(k_eff) / k_eff

        gamma = np.zeros((n_reads, k_eff))
        prev_log_like = -np.inf
        converged = False

        # E-step tensors constant across iterations (allele matrix, match/mismatch log-probs,
        # junk log-likelihood), built once; only logl_hap is recomputed per iteration.
        (
            _em_site_idx, _em_alleles, _em_code, _em_read_code,
            _em_sup_mask, _em_lm, _em_lmm, logl_junk,
        ) = _em_read_tensors(
            reads,
            self._support,
            self.window.ref_alleles,
            self.config.junk_divergence_rate,
            self.config.default_base_quality,
            _LOG_PROB_CACHE,
        )
        # Derived constants for the vectorised M-step vote: log_odds = log_match - log_mismatch,
        # a code->allele decoder, a site-index->position list, the read-allele alphabet size.
        _em_lo = _em_lm - _em_lmm
        _em_sites = sorted(_em_site_idx, key=_em_site_idx.get)
        _em_dec = {c: a for a, c in _em_alleles.items()}
        _em_n_alleles = (
            int(_em_read_code.max()) + 1
            if _em_read_code.size and _em_read_code.max() >= 0
            else 0
        )
        _em_snv_pos = set(self.window.snv_pos)

        for iteration in range(self.config.em_max_iter):
            # E-STEP prep: log P(read | haplotype) for the current consensuses.
            # (logl_junk is constant across iterations, computed once above.)
            logl_hap = _em_logl_hap(
                haplotypes, _em_site_idx, _em_code, _em_read_code,
                _em_sup_mask, _em_lm, _em_lmm,
            )

            # E-STEP: gamma[i,k] and the data log-likelihood from ONE batched logsumexp over
            # the (n_reads x k_eff) log-posterior.
            log_pi = np.log(pi + 1e-12)
            logp = np.full((n_reads, k_eff), -np.inf)
            logp[:, :n_haps] = log_pi[:n_haps][None, :] + logl_hap  # -inf where logl_hap -inf
            logp[:, junk_idx] = log_pi[junk_idx] + logl_junk
            log_sum = logsumexp(logp, axis=1)          # (n_reads,)
            gamma[:] = 0.0
            good = ~np.isneginf(log_sum)
            gamma[good] = np.exp(logp[good] - log_sum[good, None])
            gamma[~good, junk_idx] = 1.0
            # Junk is always finite, so log_sum is finite for every read; summing it reproduces
            # the per-read logsumexp total.
            log_like = float(log_sum.sum())

            # M-STEP: update mixture weights and haplotype consensuses.
            # nk = effective counts per component (with Dirichlet smoothing).
            nk = gamma.sum(axis=0) + (self.config.dirichlet_alpha - 1.0)
            pi = nk / nk.sum()

            # Rebuild haplotypes by weighted voting over reads.
            new_haps = []
            surviving_indices = []

            for k in range(n_haps):
                if nk[k] < self.config.min_hap_eff_weight:
                    continue

                # Rebuild this haplotype's consensus by gamma-weighted voting (see
                # _em_hap_consensus). "no vote" -> None and "all failed the floor" -> {}.
                new_consensus = _em_hap_consensus(
                    gamma[:, k], _em_read_code, _em_sup_mask, _em_lo, _em_lmm,
                    _em_n_alleles, _em_sites, _em_dec, _em_snv_pos,
                    self.config.min_gamma_for_vote,
                )
                if len(new_consensus) >= self.config.min_snvs_per_window:
                    new_haps.append(Haplotype(consensus=new_consensus))
                    surviving_indices.append(k)

            # Update structures after pruning low-weight haplotypes.
            haplotypes = new_haps
            n_haps = len(haplotypes)

            if n_haps == 0:
                pi = np.array([1.0])
                gamma = np.ones((n_reads, 1))
                return [], gamma, pi, log_like, True, iteration + 1

            # Rebuild pi and gamma to match the surviving haplotypes.
            junk_mass = nk[-1]
            nk_surv = nk[surviving_indices]
            nk_new = np.concatenate([nk_surv, [junk_mass]])
            pi = nk_new / nk_new.sum()

            k_eff = n_haps + 1
            junk_idx = n_haps

            gamma_new = np.zeros((n_reads, k_eff))
            for new_k, old_k in enumerate(surviving_indices):
                gamma_new[:, new_k] = gamma[:, old_k]
            gamma_new[:, junk_idx] = gamma[:, -1]
            gamma = gamma_new

            row_sums = gamma.sum(axis=1, keepdims=True)
            # A read whose whole responsibility sat on pruned haplotypes has nothing left; put
            # its mass on junk (what "no haplotype explains this read" means) so the row stays a
            # distribution. The next E-step usually overwrites it, except on a converging iteration.
            dead_rows = (row_sums[:, 0] == 0.0)
            if dead_rows.any():
                gamma[dead_rows, :] = 0.0
                gamma[dead_rows, junk_idx] = 1.0
                row_sums[dead_rows] = 1.0
            gamma /= row_sums

            # Convergence: relative change in log-likelihood (large negative numbers).
            if prev_log_like != -np.inf and abs(prev_log_like) > 1e-10:
                relative_change = abs(log_like - prev_log_like) / abs(prev_log_like)
                if relative_change < self.config.em_tolerance:
                    converged = True
                    break
            else:
                # For first iteration or very small log-likelihood, use absolute tolerance
                if abs(log_like - prev_log_like) < self.config.em_tolerance:
                    converged = True
                    break
            prev_log_like = log_like

        # Final metadata: weights, read support, and confidence per haplotype.
        for k, hap in enumerate(haplotypes):
            hap.weight = pi[k]
            hap.supporting_reads = int(
                (gamma[:, k] >= self.config.assign_confidence_threshold).sum()
            )
            confident_mask = gamma[:, k] >= self.config.assign_confidence_threshold
            if confident_mask.sum() > 0:
                hap.confidence = float(gamma[confident_mask, k].mean())

        return haplotypes, gamma, pi, log_like, converged, iteration + 1


# =============================================================================
# POST-PROCESSOR (with 1-SNP validation)
# =============================================================================


class PostProcessor:
    """Post-processing: within-window haplotype merging and the 1-SNV validator."""

    def __init__(self, config: HaplotyperConfig = DEFAULT_CONFIG):
        self.config = config

    def should_merge_1snp_pair(
        self,
        hap1: Haplotype,
        hap2: Haplotype,
        k1: int,
        k2: int,
        window: Window,
        gamma: np.ndarray,
        n_timepoints_seen: int = 1,
    ) -> bool:
        """Decide whether a pair differing at one SNV should be merged.

        Always applied: a single differing site is exactly where a real low-abundance
        strain and a sequencing artefact look alike. Read support leads, then the binomial
        test when few timepoints are seen, then the frequency floor as a backstop.
        """
        diff_positions = hap1.get_differing_positions(hap2, window.snv_pos)
        if len(diff_positions) != 1:
            return True

        diff_pos = diff_positions[0]

        if hap1.weight < hap2.weight:
            minor_hap, minor_k = hap1, k1
        else:
            minor_hap, minor_k = hap2, k2

        minor_supporting = int((gamma[:, minor_k] >= self.config.assign_confidence_threshold).sum())
        if minor_supporting < self.config.marker_min_reads:
            return True

        if n_timepoints_seen < self.config.marker_min_samples:
            minor_base = minor_hap.consensus.get(diff_pos)
            if minor_base is None:
                return True

            minor_count = 0
            total_at_pos = 0
            p_error_sum = 0.0
            for read in window.reads:
                if diff_pos in read.alleles:
                    total_at_pos += 1
                    q = read.quals.get(diff_pos, self.config.default_base_quality)
                    # /3: the null is that the minor calls are sequencing errors that
                    # happened to produce THIS base, one of the three alternatives.
                    p_error_sum += 10 ** (-q / 10.0) / 3.0
                    if read.alleles[diff_pos] == minor_base:
                        minor_count += 1

            if total_at_pos == 0:
                return True

            # The reads' own base qualities, not a fixed Q30: assuming Q30 asserts an accuracy
            # the data need not carry and errs towards SPLIT (phantom strains; measurement in
            # docs/design/core.md). The binomial wants one rate, so the mean over the reads stands in.
            p_error = p_error_sum / total_at_pos
            alpha_corrected = self.config.binomial_alpha / len(window.snv_pos)
            p_value = 1 - binom.cdf(minor_count - 1, total_at_pos, p_error)

            if p_value > alpha_corrected:
                return True

        # Frequency floor, as a backstop: a haplotype below marker_min_frac is still not
        # counted as a separate strain even when the evidence above looks real.
        if minor_hap.weight < self.config.marker_min_frac:
            return True

        return False

    def merge_similar_haplotypes(
        self,
        haplotypes: list[Haplotype],
        gamma: np.ndarray,
        pi: np.ndarray,
        window: Window,
        n_timepoints_seen: int = 1,
    ) -> tuple[list[Haplotype], np.ndarray, np.ndarray]:
        """Merge haplotypes that are one entity, comparing over informative positions."""
        n_haps = len(haplotypes)
        if n_haps <= 1:
            return haplotypes, gamma, pi

        # Compare over informative positions only, like link_windows: a rate over every called
        # site dilutes a true difference as depth grows. Comparison-time only; nothing is pruned
        # from a consensus, so wider-scope steps still see every position.
        markers = variable_marker_positions(
            (h.consensus for h in haplotypes), window.site_type, self.config
        )
        compare_positions = sorted(markers) if markers else list(window.snv_pos)
        max_mismatches = None

        used = set()
        new_haplotypes = []
        old_to_new = [-1] * n_haps

        # Greedy grouping: for each unused haplotype, merge any other haplotype
        # within the distance threshold (and passing the 1-SNP guard if needed).
        for i in range(n_haps):
            if i in used:
                continue

            group = [i]
            for j in range(i + 1, n_haps):
                if j in used:
                    continue

                # Two questions, two denominators. Co-support ("overlap enough to compare?") is
                # over every called position (what min_shared_markers was calibrated against);
                # identity ("same entity?") is over the informative positions only.
                _d, _nd, n_shared = haplotypes[i].distance_to(
                    haplotypes[j], window.snv_pos, None
                )
                if n_shared < self.config.min_shared_markers:
                    continue
                _d2, n_diff, _ns = haplotypes[i].distance_to(
                    haplotypes[j], compare_positions, max_mismatches
                )

                # No rate over informative positions, only a count: 0 diffs is one entity, 1 is
                # the should_merge_1snp_pair case, >=2 is a real difference at any depth. Same
                # rule track_merge applies across samples (byte-identity + a 1-SNV guard), so the
                # within-window and cross-sample merges agree.
                if n_diff == 0:
                    group.append(j)
                elif n_diff == 1:
                    if not self.should_merge_1snp_pair(
                        haplotypes[i], haplotypes[j], i, j, window, gamma,
                        n_timepoints_seen
                    ):
                        continue
                    group.append(j)

            used.update(group)

            # Merge consensus by weighted voting across the group.
            allele_votes = defaultdict(lambda: defaultdict(float))
            for g in group:
                weight = pi[g]
                for pos, base in haplotypes[g].consensus.items():
                    allele_votes[pos][base] += weight

            merged_consensus = {}
            for pos, counts in allele_votes.items():
                merged_consensus[pos] = max(counts, key=counts.get)

            new_idx = len(new_haplotypes)
            new_haplotypes.append(Haplotype(consensus=merged_consensus))
            for g in group:
                old_to_new[g] = new_idx

        # Rebuild pi and gamma for the merged haplotypes.
        new_k_count = len(new_haplotypes)
        new_pi = np.zeros(new_k_count + 1)

        for old_k, new_k in enumerate(old_to_new):
            if new_k >= 0:
                new_pi[new_k] += pi[old_k]
        new_pi[-1] = pi[-1]
        new_pi /= new_pi.sum()

        new_gamma = np.zeros((gamma.shape[0], new_k_count + 1))
        for old_k, new_k in enumerate(old_to_new):
            if new_k >= 0:
                new_gamma[:, new_k] += gamma[:, old_k]
        new_gamma[:, -1] = gamma[:, -1]

        row_sums = new_gamma.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        new_gamma /= row_sums

        # Update haplotype metadata after merging.
        for k, hap in enumerate(new_haplotypes):
            hap.weight = new_pi[k]
            hap.supporting_reads = int(
                (new_gamma[:, k] >= self.config.assign_confidence_threshold).sum()
            )
            confident_mask = new_gamma[:, k] >= self.config.assign_confidence_threshold
            if confident_mask.sum() > 0:
                hap.confidence = float(new_gamma[confident_mask, k].mean())

        return new_haplotypes, new_gamma, new_pi

    def assign_reads(self, reads: list[Read], gamma: np.ndarray) -> list[dict]:
        """Hard assignment of reads.

        Load-bearing, not a debugging aid: feeds the shared-read link path in link_windows and
        post-hoc read-overlap threading. Cheap (one small dict per read).
        """
        assignments = []
        n_reads, k_eff = gamma.shape
        junk_idx = k_eff - 1

        for i in range(n_reads):
            probs = gamma[i, :]
            best_k = int(np.argmax(probs))
            best_prob = float(probs[best_k])

            is_junk = best_k == junk_idx

            if is_junk:
                hap_id = None
                is_ambiguous = False
            elif best_prob >= self.config.assign_confidence_threshold:
                hap_id = best_k
                is_ambiguous = False
            else:
                hap_id = None
                is_ambiguous = True

            assignments.append(
                {
                    "read_id": reads[i].id,
                    "hap_id": hap_id,
                    # The argmax haplotype REGARDLESS of confidence (None only for junk). hap_id
                    # above is withheld below assign_confidence_threshold, right for calling a
                    # read's haplotype but wrong for LINKING: near-identical strains leave most
                    # reads at gamma 0.6-0.89, exactly where linking evidence matters most.
                    "best_hap": None if is_junk else best_k,
                    "prob": best_prob,
                    "is_junk": is_junk,
                    "is_ambiguous": is_ambiguous,
                }
            )

        return assignments


# =============================================================================
# OPTIMIZED LONGITUDINAL INTEGRATOR
# =============================================================================

# Weight the junk component keeps no matter what a rescue takes from it, for numerical
# stability. A rescue is funded ENTIRELY out of junk, so this floor is also the gate on
# whether a rescue can happen at all: below it there is nothing to fund one with.
_MIN_JUNK_WEIGHT = 0.01


class LongitudinalIntegrator:
    """Cross-timepoint integration: anchor panels and low-abundance rescue."""

    def __init__(self, config: HaplotyperConfig = DEFAULT_CONFIG):
        self.config = config
        self.rescue_statistics: list[RescueStatistic] = []
        self.rescued_reads: list[RescuedReadInfo] = []

    def build_anchor_panel_for_key(
        self,
        sample_results: dict[str, WindowResult],
        *,
        include_low_weight: bool = False,
        exclude_sample: str | None = None,
    ) -> tuple[list[Haplotype], list[str]]:
        """Build the anchor panel from a sample_results dict already filtered to one window key."""
        anchor_haps = []
        anchor_samples = []

        for sample_id, wr in sample_results.items():
            if exclude_sample and sample_id == exclude_sample:
                continue
            for hap in wr.haplotypes:
                if include_low_weight or hap.weight >= self.config.min_weight_for_anchor:
                    anchor_haps.append(hap)
                    anchor_samples.append(sample_id)

        return anchor_haps, anchor_samples

    def count_timepoints_for_haplotype(
        self, hap: Haplotype, sample_results: dict[str, WindowResult], positions: list[int]
    ) -> int:
        """Count timepoints where this haplotype appears."""
        count = 0
        for _sample_id, wr in sample_results.items():
            for other_hap in wr.haplotypes:
                dist, _, n_shared = hap.distance_to(other_hap, positions)
                if n_shared >= self.config.min_shared_for_rescue:
                    if dist <= self.config.identity_distance:
                        count += 1
                        break
        return count

    def rescue_window_result(
        self,
        window_result: WindowResult,
        anchor_haps: list[Haplotype],
        anchor_samples: list[str],
        sample_results: dict[str, WindowResult],
        current_sample: str,
    ) -> WindowResult:
        """Rescue missing haplotypes: junk-assigned reads that match an anchor from another
        timepoint become a new haplotype built from those reads.
        """
        if not anchor_haps:
            # Try to get anchors including low-weight ones
            anchor_haps, anchor_samples = self.build_anchor_panel_for_key(
                sample_results,
                include_low_weight=True,
                exclude_sample=current_sample,
            )
            if not anchor_haps:
                self.rescue_statistics.append(
                    RescueStatistic(
                        sample=current_sample,
                        rescued_timepoint=current_sample,
                        contig=window_result.window.contig,
                        window_start=window_result.window.start,
                        track_id="window",
                        was_rescued=False,
                        original_weight=0.0,
                        rescued_weight=0.0,
                        donor_timepoint="",
                        anchor_distance=-1.0,
                        n_shared_with_anchor=0,
                        n_mismatched_with_anchor=0,
                        reason="no_anchors",
                    )
                )
                return window_result

        window = window_result.window
        haplotypes = list(window_result.haplotypes)  # Make mutable copy
        gamma = window_result.gamma.copy()
        pi = window_result.pi.copy()
        reads = window.reads

        n_haps = len(haplotypes)
        junk_idx = n_haps  # Last column in gamma/pi is the junk component.
        junk_weight = pi[junk_idx] if len(pi) > junk_idx else 0.0

        junk_threshold = 0.5  # Read is "junk" if gamma[:, junk_idx] > this
        junk_read_mask = gamma[:, junk_idx] > junk_threshold
        n_junk_reads = junk_read_mask.sum()

        logging.debug(
            f"    Rescue check: {n_junk_reads}/{len(reads)} junk reads, "
            f"junk_weight={junk_weight:.3f}, {len(anchor_haps)} anchors"
        )

        # Even a single junk read matching an anchor from another timepoint is meaningful,
        # as long as the match is near-exact (controlled by identity_distance).
        if n_junk_reads < 1:
            # Not enough junk reads to rescue
            self.rescue_statistics.append(
                RescueStatistic(
                    sample=current_sample,
                    rescued_timepoint=current_sample,
                    contig=window.contig,
                    window_start=window.start,
                    track_id="window",
                    was_rescued=False,
                    original_weight=junk_weight,
                    rescued_weight=junk_weight,
                    donor_timepoint="",
                    anchor_distance=-1.0,
                    n_shared_with_anchor=0,
                    n_mismatched_with_anchor=0,
                    reason="no_junk_reads",
                )
            )
            return window_result

        # A rescue is funded entirely out of junk's weight, down to _MIN_JUNK_WEIGHT; below
        # that floor there is nothing to fund one with and proceeding is harmful (rescued
        # weights scale to 0, junk inflates). A junk read count above 0 does not imply the weight is there.
        if junk_weight - _MIN_JUNK_WEIGHT <= 0.0:
            self.rescue_statistics.append(
                RescueStatistic(
                    sample=current_sample,
                    rescued_timepoint=current_sample,
                    contig=window.contig,
                    window_start=window.start,
                    track_id="window",
                    was_rescued=False,
                    original_weight=junk_weight,
                    rescued_weight=junk_weight,
                    donor_timepoint="",
                    anchor_distance=-1.0,
                    n_shared_with_anchor=0,
                    n_mismatched_with_anchor=0,
                    reason="junk_below_floor",
                )
            )
            return window_result

        # Avoid duplicating an already-present haplotype in this window.
        existing_consensuses = [h.consensus for h in haplotypes]

        max_distance = self.config.identity_distance
        min_shared = self.config.min_shared_for_rescue
        min_shared_for_read = 2  # Lower threshold for individual reads

        # ---- Step 1: For each junk read, find all anchors it matches (distance).
        # read_matches[i] = list of (anchor_idx, distance) for read i (junk only).
        read_matches: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for i, read in enumerate(reads):
            if not junk_read_mask[i]:
                continue
            for anchor_idx, anchor in enumerate(anchor_haps):
                n_shared = 0
                n_match = 0
                for pos, allele in read.alleles.items():
                    if pos in anchor.consensus:
                        n_shared += 1
                        if anchor.consensus[pos] == allele:
                            n_match += 1
                if n_shared >= min_shared_for_read:
                    distance = 1.0 - (n_match / n_shared)
                    if distance <= max_distance:
                        read_matches[i].append((anchor_idx, distance))

        # ---- Step 2: Assign each junk read to exactly one anchor (best distance, then lowest
        # index), avoiding the double-counting that inflated total_rescued_weight.
        anchor_to_reads: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
        for i, matches in read_matches.items():
            if not matches:
                continue
            # Best = smallest distance, then smallest anchor_idx
            anchor_idx, dist = min(matches, key=lambda x: (x[1], x[0]))
            read = reads[i]
            n_shared = 0
            n_match = 0
            for pos, allele in read.alleles.items():
                if pos in anchor_haps[anchor_idx].consensus:
                    n_shared += 1
                    if anchor_haps[anchor_idx].consensus[pos] == allele:
                        n_match += 1
            # (read_idx, n_agree, n_disagree, n_total)
            anchor_to_reads[anchor_idx].append((i, n_match, n_shared - n_match, n_shared))

        # ---- Step 3: Build rescued haplotypes only for anchors that are not already present
        # and have at least one uniquely assigned read.
        rescued_any = False
        new_haplotypes = []
        # Index of each rescued haplotype's statistic, so its recorded weight can be corrected
        # to the post-scaling value the window actually carries.
        new_hap_stat_idx: list[int] = []

        for anchor_idx, anchor in enumerate(anchor_haps):
            anchor_already_present = False
            for existing in existing_consensuses:
                n_shared = 0
                n_match = 0
                for pos in window.snv_pos:
                    if pos in anchor.consensus and pos in existing:
                        n_shared += 1
                        if anchor.consensus[pos] == existing[pos]:
                            n_match += 1
                if n_shared >= min_shared:
                    distance = 1.0 - (n_match / n_shared) if n_shared > 0 else 1.0
                    if distance <= max_distance:
                        anchor_already_present = True
                        break

            if anchor_already_present:
                continue

            matching_read_info = anchor_to_reads.get(anchor_idx, [])
            n_matching_junk = len(matching_read_info)

            if n_matching_junk >= 1:
                new_consensus = {
                    pos: anchor.consensus[pos]
                    for pos in window.snv_pos
                    if pos in anchor.consensus
                }

                if new_consensus:
                    rescued_weight = max(
                        n_matching_junk / len(reads),
                        self.config.rescued_min_weight,
                    )

                    new_hap = Haplotype(
                        consensus=new_consensus,
                        weight=rescued_weight,
                        supporting_reads=n_matching_junk,
                        confidence=0.8,
                        track_id=None,
                    )
                    new_haplotypes.append(new_hap)

                    donor_timepoint = anchor_samples[anchor_idx]
                    new_hap_stat_idx.append(len(self.rescue_statistics))
                    self.rescue_statistics.append(
                        RescueStatistic(
                            sample=current_sample,
                            rescued_timepoint=current_sample,
                            contig=window.contig,
                            window_start=window.start,
                            track_id=f"rescued_from_{donor_timepoint}",
                            was_rescued=True,
                            original_weight=0.0,
                            rescued_weight=rescued_weight,
                            donor_timepoint=donor_timepoint,
                            anchor_distance=0.0,
                            n_shared_with_anchor=len(new_consensus),
                            n_mismatched_with_anchor=0,
                            reason=f"rescued_from_junk({n_matching_junk}_reads)",
                        )
                    )
                    rescued_any = True

                    for read_idx, n_agree, n_disagree, n_total in matching_read_info:
                        read = reads[read_idx]
                        read_name = getattr(read, "name", f"read_{read_idx}")
                        self.rescued_reads.append(
                            RescuedReadInfo(
                                read_name=read_name,
                                sample=current_sample,
                                contig=window.contig,
                                window_start=window.start,
                                window_end=window.end,
                                donor_timepoint=donor_timepoint,
                                n_snps_agree=n_agree,
                                n_snps_disagree=n_disagree,
                                n_snps_total=n_total,
                                rescued_haplotype_weight=rescued_weight,
                            )
                        )

                    logging.debug(
                        f"    Rescued haplotype from {donor_timepoint}: "
                        f"{n_matching_junk} junk reads, weight={rescued_weight:.3f}"
                    )

        if not rescued_any:
            self.rescue_statistics.append(
                RescueStatistic(
                    sample=current_sample,
                    rescued_timepoint=current_sample,
                    contig=window.contig,
                    window_start=window.start,
                    track_id="window",
                    was_rescued=False,
                    original_weight=junk_weight,
                    rescued_weight=junk_weight,
                    donor_timepoint="",
                    anchor_distance=-1.0,
                    n_shared_with_anchor=0,
                    n_mismatched_with_anchor=0,
                    reason="no_anchor_matches_junk",
                )
            )
            return window_result

        # Add rescued haplotypes and rebuild gamma/pi
        haplotypes.extend(new_haplotypes)
        n_haps_new = len(haplotypes)
        k_eff_new = n_haps_new + 1

        # Redistribute weight: take from junk only, never more than available, so original
        # haplotypes are not zeroed when total_rescued_weight would exceed old junk weight.
        old_junk_weight = pi[junk_idx]
        available_from_junk = max(0.0, old_junk_weight - _MIN_JUNK_WEIGHT)

        total_rescued_weight = sum(h.weight for h in new_haplotypes)
        if total_rescued_weight > available_from_junk and total_rescued_weight > 0:
            # Scale down rescued weights so they sum to available_from_junk (preserve ratios).
            scale_rescued = available_from_junk / total_rescued_weight
            for h in new_haplotypes:
                h.weight *= scale_rescued
            total_rescued_weight = available_from_junk
            # Report what the haplotype ends up with, not what it asked for.
            for h, stat_idx in zip(new_haplotypes, new_hap_stat_idx):
                self.rescue_statistics[stat_idx].rescued_weight = h.weight

        new_junk_weight = max(_MIN_JUNK_WEIGHT, old_junk_weight - total_rescued_weight)

        # Build new pi: original haplotypes scaled to fill (1 - total_rescued - new_junk).
        pi_new = np.zeros(k_eff_new)
        non_junk_before = 1.0 - old_junk_weight
        non_junk_after = 1.0 - total_rescued_weight - new_junk_weight
        scale = (non_junk_after / non_junk_before) if non_junk_before > 0 else 0.0
        scale = max(0.0, scale)  # avoid negative if floating point
        for k in range(n_haps):
            pi_new[k] = pi[k] * scale
        for k, new_hap in enumerate(new_haplotypes):
            pi_new[n_haps + k] = new_hap.weight
        pi_new[-1] = new_junk_weight
        pi_new = pi_new / pi_new.sum()

        # Update haplotype weights
        for k, hap in enumerate(haplotypes):
            hap.weight = pi_new[k]

        # Recompute gamma with new haplotypes
        gamma_new = self._recompute_gamma(window, haplotypes, pi_new)

        # Recompute assignments
        post = PostProcessor(self.config)
        assignments = post.assign_reads(reads, gamma_new)

        for k, hap in enumerate(haplotypes):
            hap.supporting_reads = int(
                (gamma_new[:, k] >= self.config.assign_confidence_threshold).sum()
            )

        n_reads_examined, reads_within_mismatch_per_hap = _compute_read_mismatch_counts(
            window, haplotypes, self.config.identity_distance
        )

        return WindowResult(
            window=window,
            haplotypes=haplotypes,
            gamma=gamma_new,
            pi=pi_new,
            log_likelihood=window_result.log_likelihood,
            assignments=assignments,
            converged=window_result.converged,
            iterations=window_result.iterations,
            n_reads_examined=n_reads_examined,
            reads_within_mismatch_per_hap=reads_within_mismatch_per_hap,
        )

    def _recompute_gamma(
        self, window: Window, haplotypes: list[Haplotype], pi: np.ndarray
    ) -> np.ndarray:
        """Recompute gamma with fixed pi (E-step only)."""
        reads = window.reads
        n_reads = len(reads)
        n_haps = len(haplotypes)
        k_eff = n_haps + 1
        junk_idx = n_haps

        gamma = np.zeros((n_reads, k_eff))

        # Scored through the same two helpers the EM uses. A rescued haplotype takes the DONOR's
        # footprint, so this is exactly where a footprint-asymmetric likelihood would do the most
        # damage - the rescued component is guaranteed a different site set from the resident ones.
        snv_set = set(window.snv_pos)
        p_div = self.config.junk_divergence_rate
        default_q = self.config.default_base_quality

        for i, read in enumerate(reads):
            logp_k = np.full(k_eff, -np.inf)
            support = _read_support(read, snv_set)

            for k in range(n_haps):
                log_prob = _log_prob_read_hap(
                    read, haplotypes[k].consensus, support, default_q
                )
                if log_prob is not None:
                    logp_k[k] = np.log(pi[k] + 1e-12) + log_prob

            logp_k[junk_idx] = np.log(pi[junk_idx] + 1e-12) + _log_prob_read_junk(
                read, support, window.ref_alleles, p_div, default_q
            )

            log_sum = logsumexp(logp_k)
            if np.isneginf(log_sum):
                gamma[i, junk_idx] = 1.0
            else:
                gamma[i, :] = np.exp(logp_k - log_sum)

        return gamma

    def rescue_low_abundance(
        self,
        results_by_timepoint: dict[str, list[WindowResult]],
        only_sample: str | None = None,
    ) -> dict[str, list[WindowResult]]:
        """Rescue low-abundance haplotypes across timepoints.

        ``only_sample`` restricts rescue to one sample; other samples still feed the anchor
        panel, which needs only their ``.haplotypes``, so their reads/gamma may be offloaded.
        Calling once per sample in iteration order equals the all-samples call (rescue mutates
        weights in place either way).
        """
        if len(results_by_timepoint) < 2:
            return results_by_timepoint

        # Group WindowResults by genomic window so we can compare across timepoints.
        windows_by_position: dict[tuple, dict[str, WindowResult]] = defaultdict(dict)

        for sample_id, window_results in results_by_timepoint.items():
            for wr in window_results:
                key = (wr.window.contig, wr.window.start, wr.window.end)
                windows_by_position[key][sample_id] = wr

        # Diagnostic summary: how many anchors and junk reads exist overall.
        n_windows_with_multiple_timepoints = 0
        total_junk_reads = 0
        total_reads = 0
        n_anchors = 0

        for _window_key, sample_results in windows_by_position.items():
            if len(sample_results) >= 2:
                n_windows_with_multiple_timepoints += 1
            for _sample_id, wr in sample_results.items():
                # Works whether or not gamma is resident (offloaded samples carry the
                # scalar summary instead).
                n_reads, junk_reads = wr.junk_read_counts()
                total_reads += n_reads
                total_junk_reads += junk_reads
                n_anchors += sum(1 for h in wr.haplotypes if h.weight >= self.config.min_weight_for_anchor)

        junk_pct = 100 * total_junk_reads / total_reads if total_reads > 0 else 0
        logging.info(
            f"    Rescue diagnostics: {len(windows_by_position)} window positions, "
            f"{n_windows_with_multiple_timepoints} shared across >=2 timepoints, "
            f"{n_anchors} anchors, {total_junk_reads}/{total_reads} junk reads ({junk_pct:.1f}%)"
        )

        # Rescue each window position independently.
        rescued_results: dict[str, list[WindowResult]] = defaultdict(list)

        for _window_key, sample_results in windows_by_position.items():
            for sample_id, wr in sample_results.items():
                if only_sample is not None and sample_id != only_sample:
                    continue
                # Build anchor panel excluding the current sample to avoid self-rescue.
                anchor_haps, anchor_samples = self.build_anchor_panel_for_key(
                    sample_results, exclude_sample=sample_id
                )
                rescued_wr = self.rescue_window_result(
                    wr, anchor_haps, anchor_samples, sample_results, sample_id
                )
                rescued_results[sample_id].append(rescued_wr)

        return dict(rescued_results)

    def write_rescue_statistics(self, output_path: str) -> str:
        """Write rescue_statistics.tsv with details of rescue events."""
        import csv

        n_rescued = sum(1 for s in self.rescue_statistics if s.was_rescued)
        logging.info(
            f"Writing rescue_statistics.tsv: {len(self.rescue_statistics)} records "
            f"({n_rescued} rescued)"
        )

        fieldnames = [
            "sample",
            "rescued_timepoint",
            "contig",
            "window_start",
            "track_id",
            "was_rescued",
            "original_weight",
            "rescued_weight",
            "donor_timepoint",
            "anchor_distance",
            "n_shared_with_anchor",
            "n_mismatched_with_anchor",
            "reason",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for stat in self.rescue_statistics:
                writer.writerow(
                    {
                        "sample": stat.sample,
                        "rescued_timepoint": stat.rescued_timepoint,
                        "contig": stat.contig,
                        "window_start": stat.window_start,
                        "track_id": stat.track_id,
                        "was_rescued": stat.was_rescued,
                        "original_weight": f"{stat.original_weight:.6f}",
                        "rescued_weight": f"{stat.rescued_weight:.6f}",
                        "donor_timepoint": stat.donor_timepoint,
                        "anchor_distance": (
                            f"{stat.anchor_distance:.6f}" if stat.anchor_distance >= 0 else "NA"
                        ),
                        "n_shared_with_anchor": stat.n_shared_with_anchor,
                        "n_mismatched_with_anchor": stat.n_mismatched_with_anchor,
                        "reason": stat.reason,
                    }
                )

        return output_path

    def write_rescued_reads(self, output_path: str) -> str:
        """Write rescued_reads.tsv with per-read details of rescue events."""
        import csv

        logging.info(
            f"Writing rescued_reads.tsv: {len(self.rescued_reads)} reads"
        )

        fieldnames = [
            "read_name",
            "sample",
            "contig",
            "window_start",
            "window_end",
            "donor_timepoint",
            "n_snps_agree",
            "n_snps_disagree",
            "n_snps_total",
            "agreement_rate",
            "rescued_haplotype_weight",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for read_info in self.rescued_reads:
                agreement_rate = (
                    read_info.n_snps_agree / read_info.n_snps_total
                    if read_info.n_snps_total > 0
                    else 0.0
                )
                writer.writerow(
                    {
                        "read_name": read_info.read_name,
                        "sample": read_info.sample,
                        "contig": read_info.contig,
                        "window_start": read_info.window_start,
                        "window_end": read_info.window_end,
                        "donor_timepoint": read_info.donor_timepoint,
                        "n_snps_agree": read_info.n_snps_agree,
                        "n_snps_disagree": read_info.n_snps_disagree,
                        "n_snps_total": read_info.n_snps_total,
                        "agreement_rate": f"{agreement_rate:.4f}",
                        "rescued_haplotype_weight": f"{read_info.rescued_haplotype_weight:.6f}",
                    }
                )

        return output_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def process_window(
    window: Window, config: HaplotyperConfig = DEFAULT_CONFIG, n_timepoints_seen: int = 1
) -> WindowResult:
    """Process a single window through the full pipeline."""
    post = PostProcessor(config)

    # 0) Resolving-depth floor. Windows are CREATED at min_reads_for_rescue, but de-novo
    #    phasing needs more reads to separate haplotypes; below the floor emit a junk-only
    #    result so the trajectory carries a gap, not a manufactured abundance of 1.0.
    if len(window.reads) < config.min_reads_per_window:
        n_reads = len(window.reads)
        gamma = np.ones((n_reads, 1))
        pi = np.array([1.0])
        n_reads_examined, reads_within_mismatch_per_hap = _compute_read_mismatch_counts(
            window, [], config.identity_distance
        )
        return WindowResult(
            window=window,
            haplotypes=[],
            gamma=gamma,
            pi=pi,
            log_likelihood=-np.inf,
            assignments=post.assign_reads(window.reads, gamma),
            converged=True,
            iterations=0,
            n_reads_examined=n_reads_examined,
            reads_within_mismatch_per_hap=reads_within_mismatch_per_hap,
        )

    # 1) Initialize haplotypes via read clustering on the overlap graph.
    initializer = GraphInitializer(config)
    initial_haps, cluster_sizes = initializer.get_initial_haplotypes(window)

    if not initial_haps:
        # No clustering signal -> junk-only result (gamma = ones).
        n_reads = len(window.reads)
        gamma = np.ones((n_reads, 1))
        pi = np.array([1.0])
        assignments = post.assign_reads(window.reads, gamma)

        n_reads_examined, reads_within_mismatch_per_hap = _compute_read_mismatch_counts(
            window, [], config.identity_distance
        )
        return WindowResult(
            window=window,
            haplotypes=[],
            gamma=gamma,
            pi=pi,
            log_likelihood=-np.inf,
            assignments=assignments,
            converged=True,
            iterations=0,
            n_reads_examined=n_reads_examined,
            reads_within_mismatch_per_hap=reads_within_mismatch_per_hap,
        )

    # 2) EM haplotyping: refine haplotype consensus and weights.
    em = EMHaplotyper(window, initial_haps, cluster_sizes, config)
    haplotypes, gamma, pi, log_lik, converged, iterations = em.run()

    if not haplotypes:
        # EM pruned all haplotypes; keep the (junk) assignments.
        assignments = post.assign_reads(window.reads, gamma)
        n_reads_examined, reads_within_mismatch_per_hap = _compute_read_mismatch_counts(
            window, [], config.identity_distance
        )
        return WindowResult(
            window=window,
            haplotypes=[],
            gamma=gamma,
            pi=pi,
            log_likelihood=log_lik,
            assignments=assignments,
            converged=converged,
            iterations=iterations,
            n_reads_examined=n_reads_examined,
            reads_within_mismatch_per_hap=reads_within_mismatch_per_hap,
        )

    # 3) Post-processing: merge near-duplicate haplotypes with 1-SNP guard.
    n_after_em = len(haplotypes)
    merged_haps, final_gamma, final_pi = post.merge_similar_haplotypes(
        haplotypes, gamma, pi, window, n_timepoints_seen
    )

    # STAGE COUNTS. K is set by Louvain, EM can only PRUNE it, this merge can only SHRINK it
    # further. Logging all three separates "Louvain found less structure" from "the merge
    # collapsed it", which the post-hoc tables cannot distinguish.
    logging.debug(
        "stage_counts\t%s\t%d\t%d\t%d\t%d\t%d",
        window.contig, window.start, len(window.reads),
        len(initial_haps), n_after_em, len(merged_haps),
    )

    # 3b) No invariant-site pruning here: construction keeps every position, pruning is a
    #     comparison-time concern. variable_marker_positions() computes the marker set at the
    #     widest scope available and the linking code filters against it, so nothing is
    #     destroyed before the scope is known.

    assignments = post.assign_reads(window.reads, final_gamma)

    n_reads_examined, reads_within_mismatch_per_hap = _compute_read_mismatch_counts(
        window, merged_haps, config.identity_distance
    )

    result = WindowResult(
        window=window,
        haplotypes=merged_haps,
        gamma=final_gamma,
        pi=final_pi,
        log_likelihood=log_lik,
        assignments=assignments,
        converged=converged,
        iterations=iterations,
        n_reads_examined=n_reads_examined,
        reads_within_mismatch_per_hap=reads_within_mismatch_per_hap,
    )

    return result



# =============================================================================
# SPLIT MOLECULES: re-assembly and the BREAK marker
# =============================================================================

# Allele values at a breakpoint site: a read crossing with a continuous alignment carries
# CONTINUOUS; one split there carries BREAK_PREFIX + the resume coordinate, so different
# events at one left coordinate stay distinct alleles. Same shape as INS<len>/DEL<len>.
CONTINUOUS = "CONT"
BREAK_PREFIX = "BRK"


def _merge_split_reads(
    reads: list[Read],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    snv_set: set[int] | None = None,
) -> tuple[list[Read], set[int]]:
    """Re-assemble split alignments into one Read each, and report the breakpoints.

    A molecule the aligner emits as a primary plus supplementary alignments (a divergent
    cassette, insertion, rearrangement) is ONE molecule, so its segments merge back into a
    single Read. Two consequences:

    1. The merged read carries alleles from both sides of the break, so the EM assigns the
       whole molecule to one haplotype instead of several fragments.
    2. The break becomes a positive identity marker (``BRK<resume_pos>`` vs ``CONT``), a
       discriminating allele like any other, which is why SVs are not excluded from identity.

    The breakpoint is anchored at the LAST aligned position of the preceding segment, or the
    nearest earlier position that is not a called variant site (pass *snv_set* to enable the
    check; the anchor is written unconditionally and would otherwise clobber a real allele).
    Returns the merged reads and the set of breakpoint positions. History: docs/design/core.md.
    """
    by_molecule: dict[str, list[Read]] = defaultdict(list)
    for r in reads:
        by_molecule[r.id].append(r)

    called = snv_set if snv_set is not None else frozenset()
    merged: list[Read] = []
    breaks: set[int] = set()
    n_anchors_dropped = 0
    qual = config.default_base_quality

    for segs in by_molecule.values():
        if len(segs) == 1:
            merged.append(segs[0])
            continue
        segs.sort(key=lambda r: (r.ref_start, r.ref_end))
        # Captured before `whole` (which IS segs[0]) has its own span widened below.
        spans = [(s.ref_start, s.ref_end) for s in segs]
        whole = segs[0]
        for seg in segs[1:]:
            whole.alleles.update(seg.alleles)
            whole.quals.update(seg.quals)
        # Set the break markers AFTER the union so they cannot be overwritten by a
        # later segment that happens to have a call at the same position.
        for prev, nxt in zip(segs, segs[1:]):  # noqa: B905
            if nxt.ref_start <= prev.ref_end:
                continue  # overlapping segments: no gap, nothing to mark
            # Walk back inside the preceding segment to an anchor no variant was called at.
            # Anywhere in that segment carries the same meaning, so shifting keeps the marker.
            anchor = prev.ref_end - 1
            while anchor in called and anchor > prev.ref_start:
                anchor -= 1
            if anchor in called:
                n_anchors_dropped += 1
                continue
            whole.alleles[anchor] = f"{BREAK_PREFIX}{nxt.ref_start}"
            whole.quals[anchor] = qual
            breaks.add(anchor)
        whole.ref_start = min(s for s, _ in spans)
        whole.ref_end = max(e for _, e in spans)
        # The per-segment spans, which the outer span no longer describes.
        whole.segments = spans
        merged.append(whole)

    # A read spanning a breakpoint with an unbroken alignment is evidence AGAINST the event,
    # not absence of evidence; without this only the broken strain carries a call and the
    # marker cannot discriminate. Read.covers, not the outer span, so a molecule's own
    # unaligned gap is not falsely marked "no break here". Detail: docs/design/core.md.
    for pos in breaks:
        for r in merged:
            if pos in r.alleles or not r.covers(pos):
                continue
            r.alleles[pos] = CONTINUOUS
            r.quals[pos] = qual

    if n_anchors_dropped:
        logging.debug(
            f"    _merge_split_reads: {n_anchors_dropped} breakpoint anchors dropped "
            f"(no free position in the preceding segment)"
        )

    return merged, breaks


# =============================================================================
# IDENTITY: marker set + the shared gate stack
# =============================================================================


def supported_marker_positions(
    observations,
    site_type: dict[int, str] | None = None,
    config: HaplotyperConfig = DEFAULT_CONFIG,
) -> frozenset[int]:
    """Identity markers that real, replicated reads support.

    ``observations`` yields ``(sample, consensus, reads)`` - one per window-haplotype. A
    position is a candidate when it holds >1 allele anywhere. An allele is kept if it reaches
    ``marker_min_reads`` reads OR ``marker_min_frac`` of a sample's reads at that position, in
    ``marker_min_samples`` samples - counted per allele, so the two need not co-occur in one
    sample (a swept position is fixed within each sample). A position survives when >= 2 alleles
    do. Returned as a frozenset (consumers only test membership or intersect).
    """
    # position -> sample -> allele -> reads
    seen: dict[int, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float)))
    for sample, consensus, reads in observations:
        for pos, base in consensus.items():
            seen[pos][sample][base] += reads

    keep: set[int] = set()
    for pos, per_sample in seen.items():
        qualifying: Counter = Counter()
        for alleles in per_sample.values():
            total = sum(alleles.values())
            if not total:
                continue
            for base, n in alleles.items():
                # Reads OR frequency, not both: requiring both denied a 2-5% strain any
                # discriminating marker. Read support is independent evidence an allele is real.
                if n >= config.marker_min_reads or n / total >= config.marker_min_frac:
                    qualifying[base] += 1
        if sum(1 for c in qualifying.values() if c >= config.marker_min_samples) >= 2:
            keep.add(pos)

    if config.exclude_sv_from_identity and site_type:
        keep -= {pos for pos in keep if site_type.get(pos) == "sv"}
    return frozenset(keep)


def variable_marker_positions(
    consensuses: Iterable[dict[int, str]],
    site_type: dict[int, str] | None = None,
    config: HaplotyperConfig = DEFAULT_CONFIG,
) -> set[int]:
    """Positions usable as identity markers, computed over *consensuses*: a position is a
    marker only if >= 2 distinct alleles appear across the whole collection (a position where
    every haplotype agrees carries no identity information yet inflates n_shared).

    Call at the widest scope available: window linking -> every haplotype in that sample/contig;
    the cross-sample merge -> every haplotype in every sample, that contig.

    SVs are identity markers (exclude_sv_from_identity defaults False): an invertible element's
    two orientations are two entities trading frequency, the flip trajectory. Flag measures only.
    """
    seen: dict[int, set[str]] = defaultdict(set)
    for consensus in consensuses:
        for pos, base in consensus.items():
            seen[pos].add(base)

    markers = {pos for pos, alleles in seen.items() if len(alleles) > 1}
    if config.exclude_sv_from_identity and site_type:
        markers -= {pos for pos in markers if site_type.get(pos) == "sv"}
    return markers


def consensus_footprint(
    consensus: dict[int, str], region: tuple[int, int] | None = None
) -> tuple[int, int]:
    """First and last position a consensus covers, optionally clipped to *region*.

    Returned as (lo, hi) with hi < lo when nothing falls inside the region, so callers test
    emptiness without a second pass. Avoids materialising a set (called once per haplotype per batch).
    """
    if region is None:
        return (min(consensus), max(consensus)) if consensus else (1, 0)
    lo_r, hi_r = region
    lo = hi = None
    for p in consensus:
        if lo_r <= p <= hi_r:
            if lo is None or p < lo:
                lo = p
            if hi is None or p > hi:
                hi = p
    return (lo, hi) if lo is not None else (1, 0)


@dataclass
class GateResult:
    """Outcome of one identity comparison. ``reason`` tells a measurement hole from a real
    genotypic wall:

      ``"linked"``              passed every gate
      ``"failed_no_evidence"``  too few shared markers, or too little physical overlap - a
                                DROPOUT. Nothing was shown to differ.
      ``"failed_mismatch"``     enough shared markers, but the alleles genuinely disagree - a
                                candidate recombination breakpoint.
    """

    passed: bool
    reason: str
    rate: float
    n_shared: int
    n_diff: int


def compare_consensus(
    a: dict[int, str],
    b: dict[int, str],
    markers: set[int],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    min_shared: int | None = None,
    region: tuple[int, int] | None = None,
    min_cospan_frac: float | None = None,
    max_rate: float | None = None,
    a_span: tuple[int, int] | None = None,
    b_span: tuple[int, int] | None = None,
) -> GateResult:
    """Apply the identity gate stack to two consensus dicts.

    Gates, in order: footprints overlap by >= ``min_entity_overlap_bp`` (and >= ``min_cospan_frac``
    of ``region``); shared markers >= ``min_shared``; mismatch rate over shared markers <=
    ``identity_distance``. The verdict rests only on discriminating markers; the overlap gate asks
    how much sequence both covered, not where the markers fall.

    ``a_span``/``b_span`` are precomputed footprints; pass them when comparing many pairs.
    """
    if min_shared is None:
        min_shared = config.min_shared_markers
    if min_cospan_frac is None:
        min_cospan_frac = config.min_cosupported_span_frac
    if max_rate is None:
        max_rate = config.identity_distance

    def _restrict(positions):
        if region is None:
            # already a set from the & below; copying it was pure overhead
            return positions if isinstance(positions, set) else set(positions)
        lo, hi = region
        return {p for p in positions if lo <= p <= hi}

    shared = _restrict(a.keys() & b.keys() & markers)
    n_shared = len(shared)

    # How much sequence both haplotypes actually cover: their footprints' intersection,
    # independent of where the markers sit (the marker-subset span would penalise loci where
    # variation clusters, which is backwards). Hoist the footprint out of a pairwise loop.
    lo_a, hi_a = a_span if a_span is not None else consensus_footprint(a, region)
    lo_b, hi_b = b_span if b_span is not None else consensus_footprint(b, region)
    overlap = (min(hi_a, hi_b) - max(lo_a, lo_b)) if (hi_a >= lo_a and hi_b >= lo_b) else 0
    if overlap < config.min_entity_overlap_bp:
        return GateResult(False, "failed_no_evidence", 1.0, n_shared, 0)
    if region is not None:
        lo, hi = region
        if hi > lo and overlap < min_cospan_frac * (hi - lo):
            return GateResult(False, "failed_no_evidence", 1.0, n_shared, 0)

    # ...and was any of it informative? Separate question, separate gate.
    if n_shared < min_shared:
        return GateResult(False, "failed_no_evidence", 1.0, n_shared, 0)

    n_diff = sum(1 for p in shared if a[p] != b[p])
    rate = n_diff / n_shared
    if rate > max_rate:
        return GateResult(False, "failed_mismatch", rate, n_shared, n_diff)
    return GateResult(True, "linked", rate, n_shared, n_diff)


def unique_best_matches(
    matches: dict[int, list[tuple[float, int]]],
) -> dict[int, int]:
    """Keep only unambiguous best matches; a tie contributes nothing.

    Uniqueness on both sides makes a connected component a path, not a hub, bounding an
    entity's size structurally rather than by tuning.
    """
    unique: dict[int, int] = {}
    for idx, options in matches.items():
        options.sort(key=lambda x: x[0])
        best_dist = options[0][0]
        if len([o for o in options if o[0] == best_dist]) == 1:
            unique[idx] = options[0][1]
    return unique


def link_windows(
    results: list[WindowResult], config: HaplotyperConfig = DEFAULT_CONFIG
) -> list[WindowResult]:
    """Link haplotypes across windows into tracks, on consensus and on shared reads.

    Overlapping windows share SNV positions and are linked (same track_id) when their consensus
    agrees on the shared SNVs and their within-window shares are compatible. With
    ``link_window_reach`` above 1, NON-overlapping windows within the reach are also linked:
    they call disjoint positions, so consensus agreement is undefined and the evidence is
    ``link_min_shared_reads`` shared reads plus reciprocal best match.

    The abundance check is an ELIMINATOR, never an indicator: two windows in one sample are the
    same timepoint, so a genuine share disagreement means they are not one entity. Agreement
    earns no credit and cannot rescue a failed identity gate. Run on RAW COUNTS. It applies to
    both paths and is the only veto on the read path.

    Modifies haplotypes in-place, setting their track_id. Reach horizon / quantisation detail:
    docs/design/core.md.
    """
    # Deferred: strainphase.coherence imports HaplotyperConfig from this module, so a
    # top-level import here would be circular. The cost is one lookup per call.
    from strainphase.coherence import abundance_coherent

    # This pass re-derives every mismatch it records, so it owns the list rather than
    # appending: the longitudinal driver calls link_windows again after rescue, and an
    # append-only list would duplicate an unrescued window's first-pass rows.
    for wr in results:
        wr.link_mismatches.clear()

    if len(results) < 2:
        # Single window: each haplotype is its own track.
        track_counter = 0
        for wr in results:
            for hap in wr.haplotypes:
                track_counter += 1
                hap.track_id = f"T{track_counter:04d}"
        return results

    # Sort by genomic coordinate so adjacent windows are compared in order.
    sorted_results = sorted(results, key=lambda wr: wr.window.start)

    # Marker set at the widest scope here: every haplotype in this sample/contig. Positions
    # where all haplotypes agree, and SV sites, are excluded. Construction no longer prunes.
    site_type_all: dict[int, str] = {}
    for wr in sorted_results:
        site_type_all.update(wr.window.site_type)
    markers = variable_marker_positions(
        (hap.consensus for wr in sorted_results for hap in wr.haplotypes),
        site_type_all,
        config,
    )

    # Build graph: nodes = (window_idx, hap_idx); edges = linkable haplotype pairs.
    graph = nx.Graph()  # Undirected for connected components

    for i, wr in enumerate(sorted_results):
        for j in range(len(wr.haplotypes)):
            graph.add_node((i, j))

    debug_records = 0

    def record_debug(wr: WindowResult, entry: dict):
        nonlocal debug_records
        if not config.linking_debug:
            return
        if debug_records >= config.linking_debug_max_records:
            return
        wr.linking_debug.append(entry)
        debug_records += 1

    # Connect haplotypes in overlapping windows.
    for i in range(len(sorted_results) - 1):
        curr_wr = sorted_results[i]
        curr_snvs = set(curr_wr.window.snv_pos)

        # Check next few windows. Overlapping ones are compared on consensus; beyond the
        # overlap, `link_window_reach` decides how far the READ path may still reach.
        horizon = max(2, config.link_window_reach)
        for k in range(i + 1, min(i + 1 + horizon, len(sorted_results))):
            next_wr = sorted_results[k]

            overlapping = next_wr.window.start < curr_wr.window.end
            if not overlapping:
                # Reads can bridge a disjoint pair only within the reach (reach 1 admits none);
                # k only increases, so nothing beyond this one can link either.
                if config.link_window_reach <= 1 or (k - i) > config.link_window_reach:
                    break

            if overlapping:
                next_snvs = set(next_wr.window.snv_pos)
                shared_snvs = list(curr_snvs & next_snvs)

                # Window-level overlap only; the per-haplotype check is below.
                if len(shared_snvs) < config.min_shared_snvs_for_link:
                    continue

                # The region shared by the two windows; the co-supported span gate is
                # measured as a fraction of it.
                region = (
                    max(curr_wr.window.start, next_wr.window.start),
                    min(curr_wr.window.end, next_wr.window.end) - 1,
                )
            else:
                # Disjoint windows: no shared region or position, so the pair loop below
                # takes the read-only path.
                region = None

            # Evaluate candidate pairings before linking. Full gate stack: shared markers,
            # co-supported span >= 25% of the region, rate <= identity_distance.
            candidates: list[tuple[int, int, float, int]] = []

            # Non-junk read count per window - the denominator the abundance eliminator
            # tests against. Same definition build_window_tables uses for `total_reads`.
            def _nonjunk(wr) -> int:
                if wr.gamma is None or wr.gamma.size == 0:
                    return 0
                junk = wr.gamma.shape[1] - 1
                return int(wr.gamma.shape[0] - (wr.gamma[:, junk] >= 0.5).sum())

            n_curr, n_next = _nonjunk(curr_wr), _nonjunk(next_wr)

            # Reads backing each haplotype, for the shared-read ranking. best_hap is the argmax
            # regardless of confidence (near-identical strains leave most reads below threshold).
            def _hap_reads(wr) -> dict[int, set]:
                out: dict[int, set] = {}
                for a in getattr(wr, "assignments", None) or []:
                    k = a.get("best_hap", a.get("hap_id"))
                    if k is None:
                        continue
                    out.setdefault(int(k), set()).add(a.get("read_id"))
                return out

            reads_i, reads_j = _hap_reads(curr_wr), _hap_reads(next_wr)
            # union of reads placed in each window, for the coverage-invariant link gate:
            # a hap's "continuing reads" = its reads that also appear in the other window.
            reads_i_all = set().union(*reads_i.values()) if reads_i else _EMPTY_READS
            reads_j_all = set().union(*reads_j.values()) if reads_j else _EMPTY_READS

            # per-haplotype footprints, clipped to the shared region, hoisted out of the
            # pairwise loop (see consensus_footprint)
            span_i = (
                [consensus_footprint(h.consensus, region) for h in curr_wr.haplotypes]
                if overlapping else []
            )
            span_j = (
                [consensus_footprint(h.consensus, region) for h in next_wr.haplotypes]
                if overlapping else []
            )
            for hi, hap_i in enumerate(curr_wr.haplotypes):
                for hj, hap_j in enumerate(next_wr.haplotypes):
                    # Shared reads are needed by BOTH paths (ranking score on the consensus
                    # path, sole evidence on the read path), so hoist the intersection here.
                    shared_reads = len(
                        reads_i.get(hi, _EMPTY_READS) & reads_j.get(hj, _EMPTY_READS)
                    )
                    if not overlapping:
                        # NON-OVERLAPPING PAIR: disjoint positions, so consensus agreement is
                        # undefined. A read carried by a haplotype in both windows is the only
                        # evidence they are one lineage, so it must stand on its own count.
                        need = config.link_min_shared_reads
                        if config.link_shared_read_frac > 0.0:
                            # Also require a fraction of each hap's OWN continuing reads, so the
                            # bar scales with depth for abundant strains, floor for rare ones.
                            cont_i = len(reads_i.get(hi, _EMPTY_READS) & reads_j_all)
                            cont_j = len(reads_j.get(hj, _EMPTY_READS) & reads_i_all)
                            need = max(need, int(np.ceil(config.link_shared_read_frac * min(cont_i, cont_j))))
                        if shared_reads < need:
                            continue
                        gate = None
                        score = -float(shared_reads)
                        n_shared_out = 0
                    else:
                        gate = compare_consensus(
                            hap_i.consensus,
                            hap_j.consensus,
                            markers,
                            config,
                            min_shared=config.min_shared_markers,
                            region=region,
                            max_rate=config.identity_distance,
                            a_span=span_i[hi],
                            b_span=span_j[hj],
                        )
                        if not gate.passed:
                            if gate.reason == "failed_mismatch":
                                curr_wr.link_mismatches.append(
                                    {
                                        "contig": curr_wr.window.contig,
                                        "window_a": curr_wr.window.start,
                                        "hap_a_idx": hi,
                                        "window_b": next_wr.window.start,
                                        "hap_b_idx": hj,
                                        "rate": round(gate.rate, 6),
                                        "n_shared": gate.n_shared,
                                        "n_diff": gate.n_diff,
                                    }
                                )
                            continue
                        # Rank by shared reads (lower is better for unique_best): consensus has
                        # already vetoed, and where two candidates are byte-identical shared reads
                        # discriminate. Falls back to the consensus rate when no assignments exist.
                        score = -float(shared_reads) if (reads_i or reads_j) else gate.rate
                        n_shared_out = gate.n_shared
                    # Abundance is an eliminator, never an indicator (see docstring): a genuine
                    # share disagreement in one sample means the pair is not one entity, however
                    # well the alleles match. Agreement never scores. Run on raw counts.
                    if not abundance_coherent(
                        [(hap_i.supporting_reads, n_curr),
                         (hap_j.supporting_reads, n_next)], config
                    ).coherent:
                        record_debug(curr_wr, {
                            "contig": curr_wr.window.contig,
                            "window_start": curr_wr.window.start,
                            "next_window_start": next_wr.window.start,
                            "hap_i": hi, "hap_j": hj,
                            "decision": "refused",
                            "reason": "incompatible_abundance",
                        })
                        # Propagate, don't just trace: recording the refusal lets the
                        # cross-sample merge inherit it instead of re-testing the pair.
                        curr_wr.link_abundance_refusals.append(
                            {
                                "contig": curr_wr.window.contig,
                                "window_a": curr_wr.window.start,
                                "hap_a_idx": hi,
                                "window_b": next_wr.window.start,
                                "hap_b_idx": hj,
                            }
                        )
                        continue
                    candidates.append((hi, hj, score, n_shared_out))

            if not candidates:
                if config.linking_debug:
                    record_debug(
                        curr_wr,
                        {
                            "contig": curr_wr.window.contig,
                            "window_start": curr_wr.window.start,
                            "window_end": curr_wr.window.end,
                            "next_window_start": next_wr.window.start,
                            "next_window_end": next_wr.window.end,
                            "decision": "no_candidates",
                            "reason": "no_pairs_within_distance_and_shared_snvs",
                        },
                    )
                continue

            # Track unique best matches for each haplotype on both sides.
            best_for_i: dict[int, list[tuple[float, int, int]]] = {}
            best_for_j: dict[int, list[tuple[float, int, int]]] = {}
            for hi, hj, dist, n_shared in candidates:
                best_for_i.setdefault(hi, []).append((dist, hj, n_shared))
                best_for_j.setdefault(hj, []).append((dist, hi, n_shared))

            def unique_best(
                matches: dict[int, list[tuple[float, int, int]]],
            ) -> dict[int, tuple[int, float, int]]:
                unique: dict[int, tuple[int, float, int]] = {}
                for idx, options in matches.items():
                    options.sort(key=lambda x: x[0])
                    best_dist, best_partner, best_shared = options[0]
                    bests = [opt for opt in options if opt[0] == best_dist]
                    # Skip ambiguous ties (including multiple perfect matches).
                    if len(bests) == 1:
                        unique[idx] = (best_partner, best_dist, best_shared)
                return unique

            unique_i = unique_best(best_for_i)
            unique_j = unique_best(best_for_j)

            if config.linking_debug:
                for hi, options in best_for_i.items():
                    options_sorted = sorted(options, key=lambda x: x[0])
                    best_dist, best_hj, best_shared = options_sorted[0]
                    second_dist = options_sorted[1][0] if len(options_sorted) > 1 else None
                    bests = [opt for opt in options_sorted if opt[0] == best_dist]
                    if len(bests) != 1:
                        record_debug(
                            curr_wr,
                            {
                                "contig": curr_wr.window.contig,
                                "window_start": curr_wr.window.start,
                                "window_end": curr_wr.window.end,
                                "next_window_start": next_wr.window.start,
                                "next_window_end": next_wr.window.end,
                                "hap_i": hi,
                                "best_hap_j": best_hj,
                                "best_dist": round(best_dist, 6),
                                "second_best_dist": round(second_dist, 6) if second_dist is not None else None,
                                "n_shared_best": best_shared,
                                "decision": "skip",
                                "reason": "ambiguous_tie",
                                "tie_count": len(bests),
                            },
                        )

            # Link only if the best match is mutual (unique on both sides).
            for hi, (hj, _dist, _n_shared) in unique_i.items():
                if hj in unique_j and unique_j[hj][0] == hi:
                    graph.add_edge((i, hi), (k, hj))
                    if config.linking_debug:
                        record_debug(
                            curr_wr,
                            {
                                "contig": curr_wr.window.contig,
                                "window_start": curr_wr.window.start,
                                "window_end": curr_wr.window.end,
                                "next_window_start": next_wr.window.start,
                                "next_window_end": next_wr.window.end,
                                "hap_i": hi,
                                "hap_j": hj,
                                "decision": "link",
                                "reason": "unique_best_mutual",
                            },
                        )
                elif config.linking_debug:
                    record_debug(
                        curr_wr,
                        {
                            "contig": curr_wr.window.contig,
                            "window_start": curr_wr.window.start,
                            "window_end": curr_wr.window.end,
                            "next_window_start": next_wr.window.start,
                            "next_window_end": next_wr.window.end,
                            "hap_i": hi,
                            "hap_j": hj,
                            "decision": "skip",
                            "reason": "not_reciprocal_best",
                        },
                    )

    # Connected components correspond to tracks across windows.
    components = list(nx.connected_components(graph))

    # Assign a track_id to each haplotype in a component.
    for track_idx, component in enumerate(components):
        track_id = f"T{track_idx + 1:04d}"
        for w_idx, h_idx in component:
            sorted_results[w_idx].haplotypes[h_idx].track_id = track_id

    logging.debug(
        f"Linked {sum(len(wr.haplotypes) for wr in sorted_results)} haplotypes "
        f"into {len(components)} tracks"
    )

    return sorted_results


def process_contig(
    bam_path: str,
    vcf_path: str,
    contig_id: str,
    contig_length: int,
    config: HaplotyperConfig = DEFAULT_CONFIG,
    sample_id: str | None = None,
    vcf_sample_name: str | None = None,
    pool: Pool | None = None,
    sv_sidecar_path: str | None = None,
    offload_reads: bool = False,
) -> list[WindowResult]:
    """Process all windows in a contig and link haplotypes across windows.

    If ``sv_sidecar_path`` is given (see :mod:`strainphase.sv_encoding`), structural variants
    are merged in as pseudo-sites and co-phased with SNVs; the "present" allele is the unique
    event ID (a locus with two events is multi-allelic). ``None`` reproduces SNV/indel-only behavior.
    """
    # 1) Load SNVs (and indels, if enabled) for this contig from the VCF.
    (
        snv_pos, ref_alleles, depth, af, site_type, site_kinds, del_span, ins_len
    ) = load_snvs(vcf_path, contig_id, vcf_sample_name, config)

    # 1b) Merge SV pseudo-sites from the sidecar (if provided). Anchors that
    # collide with a real variant are dropped to avoid clobbering (caveat 7).
    sv_support = None
    if sv_sidecar_path:
        from strainphase.sv_encoding import _RECONCILE_MAX_SPAN, load_sv_sidecar_for_contig

        # reconcile may move an anchor by up to _RECONCILE_MAX_SPAN, and a read is credited only
        # if its span brackets the anchor within _SV_ANCHOR_PAD; if the former exceeds the latter,
        # reconciling silently costs present-calls. The assert enforces the coupling.
        assert _RECONCILE_MAX_SPAN <= _SV_ANCHOR_PAD, (
            f"sv_encoding._RECONCILE_MAX_SPAN ({_RECONCILE_MAX_SPAN}) exceeds "
            f"core._SV_ANCHOR_PAD ({_SV_ANCHOR_PAD}); reconciled anchors can fall "
            "outside the span that credits a read with the event"
        )
        sv_pos, sv_ref, sv_stype, sv_support = load_sv_sidecar_for_contig(
            sv_sidecar_path, contig_id
        )
        n_added = n_collision = 0
        for p in sv_pos:
            if p in ref_alleles:
                n_collision += 1
                sv_support.pop(p, None)
                continue
            snv_pos.append(p)
            ref_alleles[p] = sv_ref[p]  # "N" placeholder; SV sites are scored by event id
            site_type[p] = sv_stype[p]
            site_kinds[p] = frozenset({sv_stype[p]})
            n_added += 1
        if n_added or n_collision:
            logging.info(
                f"Contig {contig_id}: merged {n_added} SV pseudo-sites "
                f"({n_collision} dropped for colliding with a called variant)"
            )

    if not snv_pos:
        logging.warning(f"No variants found for contig {contig_id}")
        return []

    # 2) Create overlapping windows with lazy read loading. An ITERATOR: windows are pulled
    #    from the BAM a batch at a time, so the whole contig's reads are never resident at once.
    window_iter = iter_windows_lazy(
        bam_path,
        contig_id,
        contig_length,
        snv_pos,
        ref_alleles,
        config,
        sample_id,
        site_type=site_type,
        site_kinds=site_kinds,
        del_span=del_span,
        ins_len=ins_len,
        sv_support=sv_support,
    )

    # 3) Process windows (parallel if a pool is supplied or n_workers > 1). Batching serves
    #    memory, not scheduling: pool.map over the full window list keeps both copies of every
    #    window/result alive in the parent; batches cap that transient at batch_size windows.
    n_workers = config.n_workers
    own_pool = None
    if pool is not None:
        active_pool, n_pool_workers = pool, getattr(pool, "_processes", 1)
    elif n_workers > 1:
        own_pool = active_pool = make_worker_pool(n_workers, config)
        n_pool_workers = n_workers
    else:
        active_pool, n_pool_workers = None, 1

    batch_size = max(1, n_pool_workers * max(1, config.window_batch_factor))
    results: list[WindowResult] = []
    n_windows = 0
    try:
        for batch in _batched(window_iter, batch_size):
            if n_windows == 0:
                # Logged on dispatch, not completion: shows the run is alive on a slow contig
                # (the total is not known up front - windows arrive from an iterator).
                logging.info(
                    f"Processing windows on {contig_id} with {n_pool_workers} shared "
                    f"workers in batches of {batch_size}"
                )
            n_windows += len(batch)
            if active_pool is not None and (len(batch) > 1 or n_windows > 1):
                chunksize = max(1, len(batch) // n_pool_workers)
                batch_results = active_pool.map(
                    _process_window_wrapper, batch, chunksize=chunksize
                )
            else:
                batch_results = [process_window(w, config) for w in batch]
            # Release each window's read payload (~97% of a WindowResult) as its EM finishes,
            # keeping id-only stand-ins in gamma-row order; neither link_windows nor a
            # read-partition consumer needs the alleles again here. OFF by default: the
            # longitudinal caller manages its own spill/rescue, which re-read alleles.
            if offload_reads:
                for wr in batch_results:
                    _detach_reads(wr)
            results.extend(batch_results)
            # Drop this batch's Window references before the next batch; the results still hold
            # their own window (rescue needs the reads), but the input list must not pin a copy.
            batch.clear()
    finally:
        if own_pool is not None:
            own_pool.close()
            own_pool.join()

    if not results:
        logging.warning(f"No valid windows for contig {contig_id}")
        return []

    logging.info(f"Processed {n_windows} windows on {contig_id}")

    # 4) Link haplotypes across overlapping windows into tracks.
    results = link_windows(results, config)

    return results


def _batched(iterable: Iterable, n: int) -> Iterator[list]:
    """Yield successive lists of up to *n* items. Local stand-in for
    itertools.batched (3.12+) so this runs on the cluster's interpreter."""
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


_WORKER_CONFIG: HaplotyperConfig | None = None


def _init_worker(config: HaplotyperConfig) -> None:
    """Pool initializer: stash config in a module global so workers can read it
    without paying pickle cost per task."""
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _process_window_wrapper(window: Window) -> WindowResult:
    """Wrapper for process_window. Reads config from the worker-local global
    set by _init_worker, avoiding per-task config pickling."""
    assert _WORKER_CONFIG is not None, "Worker pool was not initialized with a config"
    return process_window(window, _WORKER_CONFIG)


def make_worker_pool(n_workers: int, config: HaplotyperConfig) -> Pool:
    """Create a multiprocessing Pool whose workers are pre-initialized with
    `config`. Reuse this pool across many process_contig calls to avoid
    paying spawn/import overhead per contig."""
    return Pool(n_workers, initializer=_init_worker, initargs=(config,))


def process_mag_longitudinal(*args, **kwargs):
    """Backwards-compatible wrapper; canonical impl is
    strainphase.longitudinal.process_mag_longitudinal.
    """
    # Import lazily to avoid circular imports (`strainphase.longitudinal` imports `core`).
    from strainphase.longitudinal import process_mag_longitudinal as _impl

    # Legacy signature: process_mag_longitudinal(samples: {sid: (bam, vcf)}, mag_contigs, config).
    # Canonical: process_mag_longitudinal(mag_name, mag_contigs, samples, bam_paths, vcf_paths, config).
    if (
        len(args) >= 2
        and isinstance(args[0], dict)
        and isinstance(args[1], dict)
        and "samples" not in kwargs
    ):
        samples_dict: dict[str, tuple[str, str]] = args[0]
        mag_contigs: dict[str, int] = args[1]
        config: HaplotyperConfig = (
            args[2] if len(args) >= 3 else kwargs.get("config", DEFAULT_CONFIG)
        )

        sample_ids = list(samples_dict.keys())
        bam_paths = {sid: samples_dict[sid][0] for sid in sample_ids}
        vcf_paths = {sid: samples_dict[sid][1] for sid in sample_ids}
        return _impl(None, mag_contigs, sample_ids, bam_paths, vcf_paths, config)

    return _impl(*args, **kwargs)


# =============================================================================
# RESULTS EXPORT
# =============================================================================


def results_to_dataframe(results: dict[str, list[WindowResult]]) -> list[dict]:
    """Convert results to track-based records for a DataFrame: one row per track, grouping
    haplotypes by track_id and computing span_start/span_end across all its windows.
    """
    records = []

    for contig_id, window_results in results.items():
        tracks: dict[str, list[tuple[WindowResult, int, Haplotype]]] = defaultdict(list)

        for wr in window_results:
            for k, hap in enumerate(wr.haplotypes):
                track_id = hap.track_id or f"unlinked_{wr.window.start}_{k}"
                tracks[track_id].append((wr, k, hap))

        # Build one record per track
        for track_id, members in tracks.items():
            # Window span (fallback if no consensus SNVs)
            window_span_start = min(wr.window.start for wr, _, _ in members)
            window_span_end = max(wr.window.end for wr, _, _ in members)
            n_windows = len(members)

            # Merge consensus across all windows (weighted voting)
            position_votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            total_weight = 0.0
            total_reads = 0
            confidences = []

            window_abundances = []
            window_read_weights = []

            for _wr, _k, hap in members:
                total_weight += hap.weight
                total_reads += hap.supporting_reads
                confidences.append(hap.confidence)

                for pos, base in hap.consensus.items():
                    position_votes[pos][base] += hap.weight

                # Per-window abundance (conditioned on non-junk), or NO MEASUREMENT: a window
                # with no/short pi or all-junk did not measure this track at 0, it did not
                # measure it. Counting those as 0.0 dropped a track from 0.87 to 0.50.
                pi_vec = _wr.pi
                if pi_vec is None or len(pi_vec) <= _k:
                    continue
                denom = 1.0 - float(pi_vec[-1])
                if denom <= 0:
                    continue
                wa = max(0.0, min(1.0, float(pi_vec[_k]) / denom))
                junk_col = _wr.gamma.shape[1] - 1
                n_junk = int((_wr.gamma[:, junk_col] >= 0.5).sum())
                n_nonjunk = max(
                    getattr(_wr, "n_reads_examined", len(_wr.window.reads)) - n_junk, 0
                )
                window_abundances.append(wa)
                window_read_weights.append(n_nonjunk)

            merged_consensus = {}
            for pos, votes in position_votes.items():
                best_base = max(votes.keys(), key=lambda b: votes[b])
                merged_consensus[pos] = best_base

            # Span: first to last consensus SNV position (fallback to window span)
            if merged_consensus:
                span_start = min(merged_consensus.keys())
                span_end = max(merged_consensus.keys())
            else:
                span_start = window_span_start
                span_end = window_span_end

            # Get sample from first window (all should be same)
            sample = members[0][0].window.sample

            # Pooled over reads, not averaged over windows (a mean of per-window ratios weights
            # a 5-read window like a 40-read one - the estimator documented elsewhere as wrong).
            # NaN, not 0.0, when no window measured the track: a consumer can drop a NaN.
            total_window_reads = sum(window_read_weights)
            if total_window_reads > 0:
                mean_weight = (
                    sum(a * w for a, w in zip(window_abundances, window_read_weights))
                    / total_window_reads
                )
            elif window_abundances:
                # Measured, but no window has any non-junk read to weight by.
                mean_weight = sum(window_abundances) / len(window_abundances)
            else:
                mean_weight = float("nan")

            records.append(
                {
                    "contig": contig_id,
                    "sample": sample,
                    "track_id": track_id,
                    "span_start": span_start,
                    "span_end": span_end,
                    "span_bp": span_end - span_start,
                    "n_windows": n_windows,
                    "n_snvs": len(merged_consensus),
                    "mean_weight": mean_weight,
                    "total_supporting_reads": total_reads,
                    "mean_confidence": np.mean(confidences) if confidences else 0.0,
                    "consensus": "|".join(
                        f"{pos}:{base}" for pos, base in sorted(merged_consensus.items())
                    ),
                }
            )

    # Sort by contig, then by span_start
    records.sort(key=lambda r: (r["contig"], r["span_start"]))

    return records
