#!/usr/bin/env python3
"""
Hybrid Graph-Probabilistic Haplotype Reconstruction for Long-Read Metagenomics
Version 3.1 - With Window Linking for Contig-Spanning Haplotypes

Key Features:
- Overlapping windows (50% step) enable haplotype linking across windows
- Haplotypes are linked based on consensus similarity in shared SNV positions
- Output is TRACK-based: span_start/span_end reflect full linked extent
- Track length limited only by SNV density and haplotype consistency

Algorithm:
1. Create overlapping windows (step = window_size / 2)
2. For each window: graph initialization + EM refinement
3. Link haplotypes across windows if consensus agrees on shared SNVs
4. Output tracks with merged consensus spanning multiple windows

Output Format (results_to_dataframe):
- One row per TRACK (linked haplotype chain), not per window
- span_bp = span_end - span_start reflects true haplotype length
- n_windows shows how many windows the track spans

Date: 2025
"""

from __future__ import annotations

import bisect
import logging
import os
import warnings
import zlib
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from multiprocessing import Pool

import community as community_louvain
import networkx as nx
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

# `community` is the module python-louvain installs, and it is a generic enough
# name that PyPI also carries an unrelated package called `community`. Install
# that one - or conda-forge's unrelated igraph-based `louvain` - and the import
# above succeeds, then clustering dies much later with an AttributeError that
# reads like a bug in strainphase rather than a wrong package. Check identity
# once, here, where the message can say what to do about it.
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

    @classmethod
    def reset(cls):
        cls._warned.clear()


# =============================================================================
# CONFIGURATION WITH VALIDATION
# =============================================================================


@dataclass
class HaplotyperConfig:
    """
    Configuration parameters for the haplotyper.

    ALL thresholds and filtering parameters are explicitly defined here.
    Parameters are validated on construction via __post_init__.
    """

    # =========== WINDOW PARAMETERS ===========
    window_size: int = 20000
    min_snvs_per_window: int = 1
    # Depth policy (FIGURE4 diagnosis §6 #3). Two DIFFERENT floors:
    #   min_reads_per_window  - reads needed to PHASE a window de novo. Separating two
    #                           haplotypes at 50/50 needs ~10-20 reads; 3 cannot resolve
    #                           anything and manufactures the abundance==1.0 artifact.
    #   min_reads_for_rescue  - reads needed for a window to be BUILT AT ALL, so it can
    #                           still receive a rescued haplotype from the anchor panel.
    #                           Rescue matches an established anchor rather than doing de
    #                           novo separation, so it legitimately needs less evidence.
    # A window with min_reads_for_rescue <= n < min_reads_per_window is created but not phased.
    min_reads_per_window: int = 10
    min_reads_for_rescue: int = 5

    # =========== READ FILTERING ===========
    min_mapq: int = 20
    min_base_quality: int = 20
    default_base_quality: int = 20
    # Cap per window; reads above this are uniformly subsampled with the config seed.
    max_reads_per_window: int = 500
    # How the cap above picks WHICH reads to keep. False = an independent random draw
    # per window (historic). True = the reads whose ids hash smallest, so overlapping
    # windows keep the SAME reads out of the molecules they share - required for
    # read-overlap linking, which otherwise sees (cap/N)^2 of the shared reads and
    # reports a sampling artefact as a linking failure. Forced on by
    # link_by_read_overlap; harmless (and more reproducible) on its own.
    consistent_read_subsampling: bool = False
    # A read must physically cover at least this many bp of a window to be counted in it.
    # Without this a read overlapping by 1 bp entered n_reads_examined, the junk
    # classification and the abundance denominator identically to one spanning 20 kb.
    # (FIGURE4 diagnosis §6 #6b; costs ~8% of reads on 000089747_1.)
    min_read_window_overlap_bp: int = 1000
    # Two reads must physically overlap by at least this much to be compared at all
    # (FIGURE4 diagnosis §6 #8, LEVEL 1).
    min_read_read_overlap_bp: int = 1000

    # =========== MEMORY ===========
    # Per-read hard assignments (WindowResult.assignments) are a debugging aid: one dict
    # per read per window, never written to any output file and never read by any other
    # function in the package. On a 146-sample MAG that is tens of millions of dicts
    # retained for nothing, so they are off by default. Turn on to inspect them.
    keep_read_assignments: bool = False
    # Spill per-sample WindowResults to <output_dir>/tmp during the first pass instead of
    # holding every sample's reads in RAM until the rescue pass. Cross-sample rescue only
    # ever reads `.haplotypes` off OTHER samples (build_anchor_panel_for_key /
    # count_timepoints_for_haplotype), so the heavy fields can live on disk and be
    # reloaded one sample at a time. See longitudinal.process_mag_longitudinal.
    spill_results_to_disk: bool = True
    # Windows are handed to the worker pool in batches of n_workers * this, so the input
    # list for a contig is never fully materialised at once.
    window_batch_factor: int = 4

    # =========== VARIANT FILTERING ===========
    min_depth_site: int = 3
    # Optional AF range filter. ``None`` (default) keeps all variants regardless
    # of allele frequency, which is what you want for longitudinal phasing
    # because a position fixed at AF=0 or AF=1 in one timepoint can still be
    # informative across timepoints. Set to e.g. ``(0.05, 0.95)`` to restrict
    # to within-sample polymorphic sites.
    af_range: tuple[float, float] | None = None

    # =========== MUTATION HANDLING — INVARIANT ===========
    # There is deliberately NO ``include_indels`` and NO ``require_biallelic``
    # flag. strainphase ALWAYS loads every mutation type (SNV, MNP->SNVs,
    # insertion, deletion) and ALWAYS keeps multi-allelic (>2 allele) sites.
    # This is a hard invariant — see ``docs/MUTATION_HANDLING.md``. Do not add a
    # switch that lets a caller drop indels or collapse alleles.

    # =========== GRAPH CONSTRUCTION ===========
    min_shared_snvs_for_edge: int = 3
    max_mismatch_frac: float = 0.01
    min_reads_per_cluster: int = 3

    # =========== IDENTITY GATES (shared by all three linking levels) ===========
    # The rate gate is applied as int(max_mismatch_frac * n_shared) - a FLOOR - so it
    # already forces 0 mismatches below n_shared=100 and 1 below n_shared=200. The
    # absolute cap therefore does nothing at low n_shared and becomes the binding gate
    # at high n_shared, where a rate alone would tolerate 11 mismatches at n=1172.
    # The two guard opposite ends of the range; both are required.
    # (FIGURE4 diagnosis §6 #8.)
    max_num_diff: int = 1
    # A within-sample link mismatch vetoes a cross-window lineage continuation
    # only when at least this many timepoints INDEPENDENTLY flag the same join.
    # 1 = the old behaviour (a single timepoint's per-window EM miscall cuts the
    # link). The per-window EM miscalls ~0.03% of sites, so over a short overlap a
    # lone 1-SNV error trips the zero-tolerance link gate in exactly one timepoint
    # and severed strains the POOLED consensus agrees on over ~90 markers. Requiring
    # corroboration (2) drops those: a genuine strain difference shows in every
    # timepoint the two strains co-occur, a random EM miscall in only one.
    step1_veto_min_timepoints: int = 2
    # Minimum physical overlap between two entities, below which the verdict is an
    # explicit NON-MERGE rather than "unknown" (Strainy's I = 1000).
    min_entity_overlap_bp: int = 1000
    # AUTHOR'S DECISION: structural variants are NEVER excluded from identity. Capturing
    # the trajectory of a flip is a goal of the analysis, not noise to be filtered, so an
    # inversion is a first-class marker like any other.
    #
    # The consequence is deliberate and worth stating: when an invertible element flips,
    # the two orientations become genotypically distinct and are reported as two entities
    # whose frequencies trade off over time. That IS the flip trajectory - it is recorded
    # as a pair of anti-correlated entities rather than as one entity changing state.
    #
    # Left as a knob only so the effect can be measured; do not flip the default. The
    # earlier True default was set in code while FIGURE4 diagnosis §6 #16 was still an
    # open author decision.
    exclude_sv_from_identity: bool = False

    # =========== EM PARAMETERS ===========
    em_max_iter: int = 30
    em_tolerance: float = 1e-5
    dirichlet_alpha: float = 1.0
    min_hap_eff_weight: float = 3.0
    min_gamma_for_vote: float = 0.01
    use_cluster_pi_init: bool = True

    # =========== JUNK MODEL ===========
    junk_divergence_rate: float = 0.10

    # =========== POST-PROCESSING ===========
    merge_distance_threshold: float = 0.01
    min_shared_for_merge: int = 2  # Min shared SNVs with actual calls to consider merging
    assign_confidence_threshold: float = 0.90

    # =========== 1-SNP VALIDATION ===========
    validate_1snp_differences: bool = True
    min_minor_frequency_1snp: float = 0.10
    min_minor_supporting_reads_1snp: int = 3
    min_timepoints_for_1snp: int = 2
    use_binomial_test_1snp: bool = True
    binomial_alpha: float = 0.05

    # =========== LONGITUDINAL PARAMETERS ===========
    min_weight_for_anchor: float = 0.2
    rescue_match_distance: float = 0.01  # 1% divergence — matches unified distance threshold
    min_shared_for_rescue: int = 2  # Min shared SNVs with actual calls for rescue matching
    rescued_min_weight: float = 0.02

    # =========== CROSS-SAMPLE WINDOW GROUPING ===========
    # Groups haplotypes at ONE FIXED WINDOW across samples (the "vertical" axis).
    # Windows are fixed coordinate tiles, so every comparison here has an identical
    # footprint - no span gating, nothing to expand, no imputation gap.
    lineage_merge_distance: float = 0.01  # Max mismatch rate to group
    min_shared_for_lineage: int = 3  # Min shared markers (raised 2 -> 3 to match linking)
    # Fraction of testable samples allowed to disagree on abundance before the
    # abundance ELIMINATOR vetoes a cross-window continuation. 0.0 = zero
    # tolerance (any one incompatible sample vetoes), which on real divergent
    # data breaks clean (n_diff=0) same-strain links on noisy per-window
    # abundance estimates. Exposed so it can be relaxed; the veto still only
    # sees pairs that already PASSED the consensus gate, so relaxing it cannot
    # merge consensus-divergent strains.
    lineage_max_bad_frac: float = 0.0
    # Require a step-1 within-sample link vote to chain two window-groups into one
    # lineage (default). When False, the "votes are the only score" rule is
    # dropped: a vote-less join is allowed and ranked by consensus agreement
    # instead (fewest mismatches, then most shared markers). Exposed to test
    # whether the vote requirement is over-fragmenting consensus-identical joins.
    require_link_votes: bool = True
    # --- READ-OVERLAP THREADING (step 3 linker) ---------------------------------
    # Chain two window-groups when the SAME PHYSICAL READS sit in both, instead of
    # ranking candidates by consensus identity under a 1-to-1 reciprocal-best-match.
    #
    # Measured on div0050_k4 (per-window read->haplotype dump, w10k): read overlap
    # linked 920 adjacent-window pairs with 100% same-strain and ZERO cross-strain
    # errors, threading the dominant strain into ONE component spanning the whole
    # 4.87 Mb contig - where the reciprocal-best linker emitted 93 pieces. Half of
    # its cut points (76/152 chain terminations) were BYTE-IDENTICAL continuations
    # (n_diff=0) rejected as `failed_not_mutual`: 67 of them because the single
    # identical target preferred a rival that matched it equally well ("target
    # contested"), 8 because the source itself had two identical targets. A strain
    # that momentarily over-splits into two groups therefore loses one of them
    # permanently. Read overlap has no such constraint: two groups sharing reads
    # with one target simply merge, which is the correct answer when the split was
    # the artifact.
    #
    # OFF by default: it needs per-read assignments (forced on below), so the
    # existing consensus linker stays the shipped behaviour until this is scored.
    link_by_read_overlap: bool = False
    # Reads that must sit in BOTH groups before their join is made. A 15 kb read
    # spans ~3 windows at a 10 kb window / 5 kb step, and observed overlaps on real
    # joins were ~33 reads, so 3 is a floor against coincidence, not a real gate.
    min_shared_reads_for_link: int = 3
    # Identity shape. This decision is still OPEN (FIGURE4 diagnosis §6 #9); both are
    # implemented so they can be compared on identical inputs.
    #   "clique"     - complete linkage: a group is a clique, every member passes the
    #                  gates against every other member. No time axis, so immune to
    #                  irregular timepoint spacing and to sample-ordering mistakes.
    #   "reciprocal" - unique-best-on-both-sides + mutual between consecutive samples,
    #                  with a per-haplotype dropout skip. Requires a correct
    #                  chronological sample order to mean anything.
    # SPLIT MOLECULES (troubleshooting U1). A molecule the aligner had to split across a
    # divergent segment is re-assembled into one Read, and the breakpoint is registered as
    # a site carrying BRK<resume_pos> / CONT. Off only to measure the difference.
    merge_split_reads: bool = True

    # Step 3 runs inside the pipeline. Off only to skip it on a run that is producing the
    # window tables for something else.
    build_lineages: bool = True

    cross_sample_method: str = "clique"

    # =========== ABUNDANCE COHERENCE ===========
    # A genome cannot hold two frequencies at one locus at one time. Tested on RAW
    # counts (never the derived, already-quantised `abundance`) with Fisher's exact
    # test, so the rule self-tightens as depth grows instead of using a fixed cutoff.
    # SINGLE TIMEPOINT ONLY - this is a window-merging check, never a cross-timepoint
    # comparison. (FIGURE4 diagnosis §6 #14.)
    abundance_coherence_alpha: float = 0.01
    min_reads_for_coherence: int = 10

    # =========== LINKING DIAGNOSTICS ===========
    linking_debug: bool = False  # Record detailed linking diagnostics
    linking_debug_max_records: int = 5000  # Cap to avoid massive files
    max_span_gap_for_lineage: int = 10000  # Max gap between track spans to consider same locus

    # =========== WINDOW LINKING PARAMETERS ===========
    # Haplotypes in adjacent overlapping windows are linked if their
    # consensus agrees on shared SNVs (Hamming distance <= max_link_distance).
    # 0.02 (2%): the per-window EM miscalls ~0.03%/site, so over a short window
    # overlap (~50 markers) a lone 1-SNV error is ~2%. At the old 0.01 that lone
    # error was a hard mismatch and severed same-strain tracks; the within-sample
    # link check is a rate gate now (its absolute cap is off, see max_link_num_diff)
    # so it tolerates a couple of expected miscalls without merging cross-strain
    # pairs, which disagree at ~50% of markers.
    max_link_distance: float = 0.02  # Max mismatch fraction to link
    # The absolute mismatch cap for the WITHIN-SAMPLE link gate only. Effectively off
    # (rate-gate only) so a single EM-miscall SNV over a short overlap is not a hard
    # mismatch. The cross-strain lineage gate keeps its own cap (config.max_num_diff).
    max_link_num_diff: int = 1_000_000
    # Window-level shared SNV POSITIONS (does the window pair even have common sites).
    min_shared_snvs_for_link: int = 3
    # Haplotype-level shared ACTUAL CALLS. Previously the same knob as the line above,
    # which meant the two could not be set independently (FIGURE4 diagnosis §6 #8, LEVEL 2).
    min_shared_calls_for_link: int = 3
    # The two haplotypes' CO-SUPPORTED SPAN inside the shared region, as a fraction of
    # that region. Window geometry itself is not a useful gate (tiles overlap by exactly
    # 50% or exactly 0%, nothing between). Measured on 000089747_1: 25% rejects 16.0% of
    # adjacent-window pairs, 50% rejects 30.0% - too aggressive given under-merging is the
    # standing risk. 25% of a 10 kb overlap is 2500 bp, comfortably above
    # min_entity_overlap_bp so the two gates agree rather than one masking the other.
    min_cosupported_span_frac: float = 0.25

    # =========== RUNTIME PARAMETERS ===========
    # Seeded by DEFAULT, not None. Two things consume randomness and both change which
    # haplotypes get called:
    #   - make_windows_lazy subsamples to max_reads_per_window with config.get_rng(), so
    #     an unseeded run draws a DIFFERENT set of reads in every window above the cap;
    #   - Louvain read clustering (GraphInitializer) is handed this seed as random_state.
    # Leaving this None made reruns disagree - identical inputs gave abundances that
    # differed in the 8th decimal, and different subsampled reads above the cap. Set an
    # explicit integer to vary it deliberately (e.g. to check a result is not an artefact
    # of one draw); None restores the old unseeded behaviour.
    random_seed: int | None = 42
    validate_results: bool = False  # Set False for production runs
    n_workers: int = 1  # Number of parallel workers for window processing (1=sequential)

    def __post_init__(self):
        """Validate configuration parameters."""
        # Read-overlap threading reads WindowResult.assignments, which assign_reads
        # only populates under keep_read_assignments. Forcing it here means the
        # linker cannot be switched on into a silent no-op (every join would see
        # zero shared reads and fail), at the cost of retaining one small dict per
        # read - ~0.2% of a Read's footprint, which holds two position-keyed dicts.
        if self.link_by_read_overlap:
            self.keep_read_assignments = True
            # Without this the linker measures the SUBSAMPLE's overlap, not the reads'.
            self.consistent_read_subsampling = True

        # Junk divergence rate
        if not (0 < self.junk_divergence_rate < 0.75):
            raise ValueError(
                f"junk_divergence_rate must be in (0, 0.75), got {self.junk_divergence_rate}"
            )

        # Merge distance threshold
        if not (0 <= self.merge_distance_threshold <= 1):
            raise ValueError(
                f"merge_distance_threshold must be in [0, 1], got {self.merge_distance_threshold}"
            )

        # AF range (optional)
        if self.af_range is not None and not (
            0 <= self.af_range[0] < self.af_range[1] <= 1
        ):
            raise ValueError(
                f"af_range must be (low, high) with 0 <= low < high <= 1, got {self.af_range}"
            )

        # Minor frequency for 1-SNP
        if self.min_minor_frequency_1snp > 0.5:
            raise ValueError(
                f"min_minor_frequency_1snp should be <= 0.5, got {self.min_minor_frequency_1snp}"
            )

        # Confidence threshold
        if not (0 < self.assign_confidence_threshold <= 1):
            raise ValueError(
                f"assign_confidence_threshold must be in (0, 1], got {self.assign_confidence_threshold}"
            )

        # Window size
        if self.window_size < 100:
            raise ValueError(f"window_size too small: {self.window_size}")

        # Depth policy: a window must be buildable before it can be phased, so the
        # rescue floor has to sit at or below the phasing floor. Clamp rather than raise -
        # a caller lowering min_reads_per_window (tests, low-coverage runs) means "be more
        # permissive", and erroring on that would be hostile. Above the phasing floor the
        # rescue band would simply be empty, which is silent and confusing.
        if self.min_reads_for_rescue > self.min_reads_per_window:
            object.__setattr__(self, "min_reads_for_rescue", self.min_reads_per_window)

        if self.max_num_diff < 0:
            raise ValueError(f"max_num_diff must be >= 0, got {self.max_num_diff}")

        if not (0 <= self.min_cosupported_span_frac <= 1):
            raise ValueError(
                f"min_cosupported_span_frac must be in [0, 1], got "
                f"{self.min_cosupported_span_frac}"
            )

        if self.cross_sample_method not in ("clique", "reciprocal"):
            raise ValueError(
                f"cross_sample_method must be 'clique' or 'reciprocal', got "
                f"{self.cross_sample_method!r}"
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


@dataclass
class Read:
    """Lightweight container for read data. All positions are 1-based (VCF convention)."""

    id: str
    contig: str
    mapq: int
    alleles: dict[int, str] = field(default_factory=dict)
    quals: dict[int, int] = field(default_factory=dict)
    sample: str | None = None
    # Reference alignment span, 1-based inclusive-exclusive (from aln.reference_start /
    # aln.reference_end). Needed for the physical-overlap gates; without these the only
    # available proxy is the span between shared VARIANT sites, which under-counts badly
    # in variant-sparse regions. Default (0, 0) means "unknown" - overlap gates are then
    # skipped rather than silently rejecting everything.
    ref_start: int = 0
    ref_end: int = 0
    # Aligned reference intervals, 1-based [start, end), one per alignment segment. Empty
    # for a read from a single alignment, whose outer span IS its coverage. A molecule
    # re-assembled from split alignments carries one entry per segment, because there
    # ref_start/ref_end bracket the unaligned gap between the segments as well - asking
    # the outer span whether the read covers a position answers yes inside its own gap.
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
    """
    Represents a genomic window (contig interval) with associated SNVs and reads.

    Notes:
    - snv_pos and ref_alleles are populated from the VCF (see load_snvs).
    - reads are pulled from the BAM and filtered to this window in make_windows_lazy.
    - sample is optional metadata; reads may also carry their own sample tag.
    """

    contig: str
    start: int  # 1-based, inclusive
    end: int  # 1-based, exclusive
    snv_pos: list[int] = field(default_factory=list)  # SNV positions (from VCF)
    ref_alleles: dict[int, str] = field(default_factory=dict)  # REF base per SNV (from VCF)
    # Reads overlapping this window (from BAM), in gamma-row order. AFTER a
    # WindowResult offloads its heavy fields these are id-only stand-ins
    # (_ReadRef) rather than Reads: the payload is released but the row
    # order and the read ids survive, so a consumer can still say which read each gamma
    # row belongs to. Anything needing alleles must run before the offload - which
    # everything in this module does.
    reads: list[Read] = field(default_factory=list)
    # Per-position site type ("snv" / "del" / "ins" / "sv") for the positions in this
    # window. Carried on the Window because the identity code downstream needs to know
    # which positions are structural variants in order to exclude them from the distance
    # (an invertible promoter flips independently of strain background). Without this the
    # SV exclusion cannot fire at all.
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
    """
    A resolved haplotype within a window.

    Notes on fields:
    - weight: mixture weight for this haplotype in the window (pi[k] from EM),
      i.e., estimated fraction of reads assigned to this haplotype.
    - confidence: mean posterior assignment probability for reads confidently
      assigned to this haplotype (computed from gamma with assign_confidence_threshold).
    """

    consensus: dict[int, str]
    weight: float = 0.0  # Mixture weight (pi) after EM / post-merge / rescue.
    supporting_reads: int = 0
    confidence: float = 0.0  # Mean gamma over confident reads for this haplotype.
    track_id: str | None = None  # Assigned after window linking

    def distance_to(
        self, other: Haplotype, positions: list[int], max_mismatches: int | None = None
    ) -> tuple[float, int, int]:
        """
        Compute normalized Hamming distance with optional early exit.

        Args:
            max_mismatches: If set, stop counting after this many mismatches

        Returns:
            (distance, n_mismatches, n_shared_positions)

        IMPORTANT: If n_shared_positions == 0, distance is 1.0 (incomparable).
        Callers MUST check n_shared before trusting the distance value.
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
            # No shared positions - cannot compare, return max distance
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
    # Step-1 comparisons that failed on a GENUINE ALLELE DISAGREEMENT, recorded so the
    # verdict survives to the output. Only `failed_mismatch` is kept: it is a real
    # genotypic wall (a candidate recombination breakpoint) and is the one negative the
    # merge rules treat as absolute. `failed_no_evidence` is a measurement hole and is
    # deliberately NOT recorded - reporting absence of coverage would bury the signal.
    link_mismatches: list[dict] = field(default_factory=list)
    n_reads_examined: int = 0
    reads_within_mismatch_per_hap: list[int] = field(default_factory=list)
    # Scalar summaries of `gamma`, recorded before the heavy fields are offloaded so the
    # output tables can still be built while reads/gamma live on disk. -1 = not recorded.
    n_reads_total: int = -1
    n_junk_reads: int = -1
    heavy_offloaded: bool = False

    # ---------------- Heavy-field offload ----------------
    # `window.reads` is ~99% of a WindowResult's footprint: a Read holds two
    # position-keyed dicts, and on a contig with a variant every ~25 bp a single 15 kb
    # HiFi read costs ~90 KB. Reads are needed only while the window's OWN sample is
    # being phased or rescued - cross-sample rescue reads nothing but `.haplotypes` off
    # other samples. `gamma` is ~1000x smaller and IS still read when the output tables
    # are built, so it stays resident.
    #
    # The WindowResult object itself stays resident too, and that is deliberate: the
    # rescue panel holds references to these very Haplotype objects and mutates their
    # weights in place, so replacing the objects would change results.

    def offload_heavy(self) -> list:
        """Detach and return this window's reads, recording the read-count summaries."""
        if self.heavy_offloaded:
            return []
        self.n_reads_total, self.n_junk_reads = self.junk_read_counts()
        reads = self.window.reads
        self.window.reads = []
        self.window._pos_sets = None
        # `assignments` is deliberately NOT cleared. It is populated only under
        # keep_read_assignments, so dropping it here discarded the very thing that
        # flag exists to produce - the per-window read->haplotype table came back
        # covering only the windows that happened to escape a spill (16 of 975
        # windows in one sample, 888 in another). It is also cheap next to what is
        # being offloaded: one small dict per read against a Read's two
        # position-keyed dicts (~90 KB for a 15 kb HiFi read). Read-overlap
        # threading reads it after every sample is phased, so it must survive.
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
        k_eff = n_haps + 1
        # k_eff = number of haplotypes + 1 junk component.

        # gamma shape must match (n_reads x k_eff).
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

    ``window.reads`` is emptied as soon as a window's own sample is finished with it (see
    WindowResult.offload_heavy), but the WindowResults handed back to the caller still
    have to say WHICH read each gamma row belongs to. That correspondence IS the read
    partition, and it is the entire output for a caller scoring reads rather than
    haplotypes. Dropping it returned every window with zero reads against a gamma of
    50-odd rows, so a partition built from the return value came out empty and the
    pipeline looked like it had phased nothing.

    Holding the whole Read instead is not an option - two position-keyed dicts, ~90 KB on
    a variant-dense contig, times every sample resident at once - which is exactly what
    the offload exists to prevent. The id alone is ~1500x cheaper (48 bytes against the
    ~70 KB of a 600-marker read) and preserves the row correspondence exactly. Nothing
    else survives, on purpose: code that reaches for
    ``.alleles`` here is reading a released read, and an AttributeError naming this class
    is far better than silently seeing no alleles.
    """

    __slots__ = ("id",)

    def __init__(self, read_id: str) -> None:
        self.id = read_id

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_ReadRef({self.id!r})"


def _read_sort_hash(read_id: str, seed: int | None) -> int:
    """Stable per-read sort key for window-consistent subsampling.

    ``hash()`` is salted per process for str, so it would pick a different subset on
    every run and break reproducibility; CRC32 over the seeded id is stable across
    processes, runs and machines. The seed is mixed in so a different ``random_seed``
    still draws a different (but internally consistent) subset.
    """
    return zlib.crc32(f"{seed}:{read_id}".encode())


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
    max_mismatch_frac: float,
) -> tuple[int, list[int]]:
    """Count reads within max_mismatch_frac of each haplotype consensus."""
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
            if (mismatches / n_shared) < max_mismatch_frac:
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
    anchor_distance: float  # Distance to matching anchor
    n_shared_with_anchor: int  # Number of shared SNVs with anchor
    n_mismatched_with_anchor: int  # Number of mismatched SNVs with anchor
    reason: str = ""  # Debug reason for rescue outcome


@dataclass
class RescuedReadInfo:
    """Per-read information for rescue events."""

    read_name: str  # Read identifier
    sample: str  # Timepoint where rescue occurred
    contig: str
    window_start: int
    window_end: int
    donor_timepoint: str  # Timepoint that provided the anchor haplotype
    n_snps_agree: int  # Number of SNPs where read agrees with rescued haplotype
    n_snps_disagree: int  # Number of SNPs where read disagrees with rescued haplotype
    n_snps_total: int  # Total SNPs in the comparison (agree + disagree)
    rescued_haplotype_weight: float  # Weight of the rescued haplotype


# =============================================================================
# LOG-PROBABILITY CACHE
# =============================================================================


class LogProbCache:
    """
    Cache for log probability computations.

    Avoids redundant 10**(-Q/10) calculations. The mismatch probability is
    spread uniformly over the ``n_alleles - 1`` non-matching states.

    The alphabet is always {A, C, G, T, DEL, INS} → ``n_alleles=6``, because
    indels are always processed (invariant — see ``docs/MUTATION_HANDLING.md``).
    SV pseudo-alleles reuse the same 6-state error model.
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
        """``log P(obs | agree) - log P(obs | disagree)`` at quality *q*.

        The evidence one base of quality *q* carries for the allele it calls. This is the
        weight a consensus vote must use: it spans 6.20 nats at Q20 to 15.42 at Q60, so a
        confident base genuinely outvotes several unreliable ones.
        """
        q = min(q, len(self._log_match) - 1)
        return float(self._log_match[q] - self._log_mismatch[q])

    def junk_tables(self, p_div: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-quality (log P(obs == REF), log P(obs == a given non-REF)) under the junk
        model, cached per divergence rate.

        The junk component describes a read off a genome diverged from the reference at
        ``p_div``; what is OBSERVED is that genome seen through the same per-base error
        channel the haplotype components use, so the two effects compose here rather than
        junk being scored at a flat rate while every haplotype was quality-scaled. That
        asymmetry was the defect: the softmax was comparing a model that believes in
        sequencing error against one that does not.

        It does NOT make the junk/haplotype boundary independent of base quality, and
        should not: a likelihood ratio treats a Q60 mismatch as far stronger evidence than
        a Q20 one, so the crossover sits at 4.25% divergence at Q20 and 1.0% at Q60 on a
        400-site window. ``junk_divergence_rate`` is the prior on divergence, not a
        decision threshold.
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

# log P(read allele | a component that has no allele at this position). Marginalising the
# unknown allele over the alphabet gives exactly 1/n_alleles whatever the read carries.
#
# Every mixture component must be scored over the SAME set of sites or the softmax that
# turns those scores into gamma is not a posterior. It used to score each haplotype only
# where its own consensus reached and junk over every window site, so a narrower footprint
# collected free likelihood: two haplotypes with identical alleles but footprints of 400
# and 150 sites split a read 0.075 / 0.925, and at Q20 a 150-site haplotype that
# MISMATCHED a read beat a 400-site one that matched it perfectly.
_LOG_MISSING_SITE = float(np.log(1.0 / _LOG_PROB_CACHE.n_alleles))

# The reference base an SV pseudo-site carries (strainphase.sv_encoding writes it): a
# structural variant has no single reference allele, so the anchor gets a placeholder.
# No read can ever equal it, which is why the junk model has to step over these sites -
# see _log_prob_read_junk.
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

    Junk is "a genome diverged from the reference at ``p_div``", so a site with no usable
    reference allele carries no junk evidence and takes the missing-data term instead.
    Two kinds of site qualify: one the loader never assigned an anchor base, and an SV
    pseudo-site, whose anchor is the ``"N"`` placeholder. Charging the placeholder as a
    mismatch made junk pay ~3.9 nats at EVERY SV anchor whether or not the read carried
    the event, and adding a 40-anchor sidecar to an unchanged BAM collapsed pi_junk from
    0.178 to 0.0 - ten genuinely divergent reads absorbed by pseudo-sites no read carried.
    Split-read breakpoints are the other SV mechanism and are NOT excluded: their anchor
    holds CONTINUOUS, which a read crossing the position with an unbroken alignment
    genuinely matches (see iter_windows_lazy).
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

    The reads, their support sites and their qualities do not change during EM -
    only the haplotype consensuses do (M-step) - so encoding the reads to an
    integer allele matrix, precomputing the per-(read, site) log_match/log_mismatch
    at each read's quality, and computing the junk log-likelihood (which depends
    only on the reads and the reference) are all done ONCE here. This is the
    vectorised replacement for the per-(read, hap) Python loop over
    ``_log_prob_read_hap`` / ``_log_prob_read_junk`` that dominated EM runtime;
    it is numerically identical to summing those calls (float order aside).

    Returns ``(site_idx, alleles, read_code, sup_mask, lm, lmm, logl_junk)`` where
    ``alleles`` is the shared string->int encoder (a dict, extended in place when
    haplotype consensuses are encoded per iteration) so read/hap/ref alleles all
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

    Identical to calling ``_log_prob_read_hap`` for every (read, hap): a support
    site the consensus does not reach costs ``_LOG_MISSING_SITE``; a covered site
    is log_match/log_mismatch at the read's quality; a read with zero shared
    covered sites gets ``-inf`` (must not compete), matching the ``None`` return.
    ``code`` extends the shared encoder with any hap-only alleles (no read carries
    them, so they can only mismatch - the correct outcome).
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

    This is the SINGLE, uniform decomposition applied to every allele, so all
    mutation types are handled the same clean way:

    * 1bp REF / 1bp ALT              -> one ``snv``
    * equal-length multi-base (MNP)  -> one ``snv`` per differing offset
    * REF longer  (net deletion)     -> one ``del`` (positions trusted as-is)
    * ALT longer  (net insertion)    -> one ``ins`` (positions trusted as-is)

    INVARIANT: indels and multi-nucleotide substitutions are ALWAYS decomposed
    and kept — there is no option to disable either (see
    ``docs/MUTATION_HANDLING.md``). Non-ACGT bases at a substitution offset are
    ambiguity codes, not phaseable alleles, so they yield no primitive; the
    empty result is counted by the caller, never dropped silently.
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
            # A position may now hold several edits, so these are the only two
            # ways an allele can still be lost or an input can be inconsistent.
            # They MUST be listed here or the loss goes unreported.
            "allele_collapsed_same_key",
            "anchor_base_conflict",
        )
    }
    where = f" [{contig_id}]" if contig_id else ""
    loaded_str = " ".join(f"{k}={v}" for k, v in sorted(loaded.items())) or "none"
    logging.info(
        # "edits" not "sites": one position may hold several (an SNV and a
        # deletion, or two deletion lengths), so the per-kind counts sum to more
        # than the number of positions. Both numbers are reported.
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
# process_contig once PER SAMPLE, and under a cohort union VCF every one of those calls
# parses the identical file: on 000066952_0 that was the same 76,988 records re-read 146
# times. Keyed on the settings that change what is kept, so a config change still
# re-parses. Bounded because a run touches one contig at a time.
_SNV_CACHE: dict[tuple, tuple] = {}
_SNV_CACHE_MAX = 8


def _copy_snv_tables(tables: tuple) -> tuple:
    """Shallow-copy every container in a parsed-VCF tuple.

    One copy per container is enough: the callers that mutate (``process_contig``'s SV
    merge, ``iter_windows_lazy``'s breakpoint registration) append to the position list
    and assign into the top-level dicts. Nothing reaches into ``del_span`` / ``ins_len``'s
    inner sets, which stay shared. Written by type rather than by position so a caller
    stubbing out the loader with a shorter tuple still gets copies.
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
    """Cached wrapper - see _SNV_CACHE. Returns a FRESH SHALLOW COPY of the parsed tables
    on every call, hit or miss, so a caller is free to mutate what it gets back.

    It has to. ``process_contig`` appends SV pseudo-sites to ``snv_pos`` and writes their
    anchor bases and site types, and ``iter_windows_lazy`` registers split-read
    breakpoints the same way. Handing back the cached objects meant sample 1's SV anchors
    were already present when sample 2 ran under the same union VCF, so sample 2 saw them
    as collisions with a called variant and dropped its own SV sites - one timepoint kept
    the event and every other timepoint lost it.
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
    """Load variants from a VCF.

    Returns every mutation as atomic sites: SNVs (incl. MNP blocks split into
    per-position SNVs) and indels. The site type per position is one of
    ``"snv"``, ``"del"``, or ``"ins"``. Indels and multi-allelic sites are
    always kept (invariant — see ``docs/MUTATION_HANDLING.md``).

    The caller is trusted: this loader does not realign, fuzz-match, or attempt
    to reconcile alignment differences. Each VCF record's position and alleles
    are taken as-is. Run ``bcftools norm -f REF`` upstream so the placement is
    canonical.

    Returns
    -------
    snv_pos
        Sorted list of variant positions (1-based VCF anchor position).
    ref_alleles
        Per-position single-base REFERENCE base at the anchor. NOT the record's
        full REF string: reads only ever carry one base or an indel token, so a
        multi-base REF could never equal a read's allele and made every read a
        mismatch at every deletion site.
    depth, af
        Per-position site depth and alt-allele frequency (AF may be ``None``).
        AF is ``Number=A``, so a multi-allelic position holds the frequency of
        the FIRST allele registered there — the value is a per-ALLELE quantity
        and a position is not one. Nothing downstream reads it; the per-ALT
        values are used where they belong, in the ``af_range`` gate.
    site_type
        Per-position type of the FIRST variant registered there: ``"snv"`` /
        ``"del"`` / ``"ins"`` (``"sv"`` is added later by the SV sidecar merge,
        which drops anchors colliding with a called variant, so an ``"sv"``
        position never holds another kind). Kept a plain ``str`` so
        ``site_type[p] == "sv"`` tests stay correct; use ``site_kinds`` to ask
        what a position holds.
    site_kinds
        Per-position set of ALL variant kinds registered there. A position may
        be e.g. ``{"snv", "del"}``.
    del_span
        For deletion sites: the SET of inclusive 1-based deleted-base footprints
        ``(start, end)`` anchored here. More than one length may share an anchor;
        each becomes its own ``DEL<len>`` allele.
    ins_len
        For insertion sites: the SET of inserted lengths anchored here. Each
        becomes its own ``INS<len>`` allele.

    Notes
    -----
    **Nothing is dropped silently.** Every input record is decomposed into atomic
    variants so no mutation is lost, and a full accounting (loaded per type +
    every skip reason) is logged at INFO before returning:

    * SNVs (1bp REF / 1bp ALT) are loaded as-is.
    * **MNPs** (equal-length multi-base substitutions, e.g. ``TACG>CACC``) are
      **atomized** into one SNV per offset where the bases differ.
    * **Multi-allelic** records are decomposed allele-by-allele; each ALT is
      classified independently. ALWAYS kept — there is no biallelic filter.
    * **Indels** (length-changing REF/ALT) are ALWAYS loaded as-is. Positions
      are trusted exactly — run ``bcftools norm -f REF`` upstream so placement
      is canonical.

    All four are handled by the one ``_atomize_allele`` helper; there is no flag
    to disable indels or collapse alleles (invariant — see the module doc and
    ``docs/MUTATION_HANDLING.md``).

    Quality gates (FILTER!=PASS, missing/low DP, optional AF band) still apply,
    but each rejection is COUNTED and logged, never silent. FILTER and DP are
    record-level and reject the whole record; the AF band is per-ALLELE and
    rejects only the allele that falls outside it, so it cannot take a
    multi-allelic record's other alleles down with it.

    **A position may hold more than one variant.** Two deletions of different
    length anchored on the same base, or an SNV and an indel at one position,
    are all registered; each distinct edit becomes its own allele downstream
    (``DEL<len>`` / ``INS<len>`` / a base), so they separate haplotypes instead
    of one silently displacing the others.

    One collapse remains and is counted as ``allele_collapsed_same_key``: two
    insertions of the SAME length at the same anchor differing only in inserted
    SEQUENCE. The allele token is ``INS<len>`` and the read-match key is
    ``(anchor, length)``, so sequence-level insertion identity is not
    representable. Making it representable means putting the sequence in the
    token, which is a separate decision.
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
        """Record one atomic variant.

        A position may hold MORE THAN ONE variant. Each distinct edit at a
        position becomes its own allele downstream (``DEL<len>`` / ``INS<len>``
        / a base), so a 1bp and a 2bp deletion sharing an anchor are two
        alleles rather than one surviving the other. Returns True if this
        variant added a new allele-defining edit.

        ``ref_alleles[apos]`` holds the single-base REFERENCE base at the
        anchor, NOT the record's full REF string. Reads carry either an indel
        token or one base, so a multi-base REF ("TG") could never equal
        anything a read holds and every read scored as a mismatch at every
        deletion site. The anchor base is what makes reference-likeness
        answerable at a position that is both an SNV and an indel.
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
            # The reference base at a position is a property of the reference,
            # so records disagreeing about it means inconsistent input.
            stats["anchor_base_conflict"] += 1

        # After ``bcftools norm``, indels are left-anchored: REF starts with the
        # anchor base(s) matching ALT, then deletes (REF longer) / inserts (ALT
        # longer) the trailing bases. Positions are trusted exactly.
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
                # Same anchor AND same inserted length, different sequence. The
                # allele token is INS<len> and the match key is (anchor, length),
                # so these genuinely cannot be told apart. Counted, not silent.
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

    # Handle multi-sample VCFs
    n_samples = len(vcf.header.samples)
    if n_samples > 1 and sample_name is None:
        raise ValueError(
            f"VCF has {n_samples} samples but no sample_name specified. "
            f"Available: {list(vcf.header.samples)}"
        )

    # A contig the VCF never declares carries no variants, which is a fact about the
    # data, not an error: callers routinely emit ##contig lines only for the contigs they
    # called something on, and a MAG assembly is mostly such contigs. pysam raises
    # "invalid contig" on fetch instead, which aborted the whole run on the first
    # variant-free contig. Ask, and take the empty answer.
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

        # Extract depth
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

        # Decompose EVERY allele into atomic primitives via the one shared
        # helper. All mutation types — SNV, MNP, insertion, deletion — and all
        # (>2) alleles are ALWAYS kept; nothing here can turn that off.
        for alt_idx, alt in enumerate(alts):
            # The AF band is a PER-ALLELE gate, applied here rather than once per record.
            # Evaluating it on ALT-0's frequency and `continue`ing the record threw away
            # every other allele on the record: AF=0.01,0.45 under af_range=(0.05, 0.95)
            # lost the 45% allele on the strength of the 1% one, which contradicts the
            # invariant two paragraphs down in this function's docstring.
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
    """Eager wrapper around :func:`iter_windows_lazy`, kept for callers that want the
    whole contig's windows at once (tests, ad-hoc scripts).

    Prefer the iterator in production: materialising every window for a contig means
    holding every window's reads simultaneously, which on a variant-dense contig is the
    single largest allocation in the run.
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
    """
    Yield overlapping windows with lazy per-window read loading.

    Windows overlap by 50% (step = window_size / 2) to enable linking
    of haplotypes across window boundaries via shared SNVs.

    This is O(W * reads_per_window) instead of O(W * total_reads),
    and uses O(window) memory instead of O(contig).

    Yielding rather than returning a list means the caller can process (and release) each
    window before the next one's reads are pulled from the BAM.
    """
    if not HAS_PYSAM:
        raise ImportError("pysam required for BAM parsing")

    snv_pos_sorted = sorted([p for p in snv_positions if 0 < p <= contig_length])
    if not snv_pos_sorted:
        return

    rng = config.get_rng()

    bam = pysam.AlignmentFile(bam_path, "rb")

    # 50% overlap: step = window_size / 2
    step_size = config.window_size // 2
    window_idx = 0

    for start in range(1, contig_length + 1, step_size):
        end = min(start + config.window_size, contig_length + 1)

        # Note: no size-based window skipping. Small windows (including
        # trailing windows and contigs shorter than window_size) are kept
        # and filtered downstream by min_snvs_per_window / min_reads_per_window.

        # Collect SNVs in this window. snv_pos_sorted is sorted, so bisect the
        # [start, end) slice instead of scanning all ~N variants per window - the
        # linear scan was O(windows * total_variants) (~12M comparisons on a 5 Mb
        # contig with 24k sites) and dominated this function's self-time.
        lo = bisect.bisect_left(snv_pos_sorted, start)
        hi = bisect.bisect_left(snv_pos_sorted, end)
        window_snvs = snv_pos_sorted[lo:hi]

        if len(window_snvs) < config.min_snvs_per_window:
            continue

        # Lazy load reads for this window only using pysam.fetch
        snv_set = set(window_snvs)
        reads = []

        # Partition window sites by type for fast lookup during extraction.
        # Sites without a recorded type default to "snv" (back-compat).
        st = site_type or {}
        # Fall back to one kind per position rather than an empty map: a caller
        # that passes site_type but not site_kinds must get the old single-kind
        # behaviour, NOT silently lose all indel parsing.
        sk = site_kinds or {p: frozenset({t}) for p, t in st.items()}
        ds = del_span or {}
        il = ins_len or {}
        sv_sup = sv_support or {}
        # Per-window indel index. Decisions are dict/set lookups (O(1)) at
        # use time:
        #   del_key_to_pos: (D-op start_1b, D-op length) -> indel site pos
        #   ins_key_to_pos: (I-op anchor_1b, inserted length) -> indel site pos
        # Both keys are SIZE-specific, so a <len>-bp indel is its own allele
        # (DEL<len> / INS<len>) rather than collapsing every indel here to a
        # generic "DEL"/"INS" — mirrors the per-event SV encoding.
        # Membership is decided by site_kinds, not site_type: a position may be
        # both an SNV and an indel, and it must reach the CIGAR walk to have its
        # indel alleles read. Reads without the indel fall through to the anchor
        # base, which is how the co-located SNV allele is still captured.
        indel_site_set = {
            p for p in window_snvs
            if sk.get(p, frozenset()) & {"del", "ins"}
        }
        # SV pseudo-sites: the "present" allele is the UNIQUE event ID (from
        # Sniffles' supporting-read set; see strainphase.sv_encoding), NOT a
        # generic INS/DEL token — distinct events stay distinct alleles so reads
        # cluster only when they carry the identical event. Kept separate from
        # indel_site_set so the D/I exact-match logic never touches them.
        sv_site_set = {p for p in window_snvs if st.get(p, "snv") == "sv"}  # SV anchors never co-occur with a called variant (see process_contig)
        # Sites parsed outside the base-by-base SNV loop (indels + SVs).
        special_site_set = indel_site_set | sv_site_set
        del_key_to_pos: dict[tuple[int, int], int] = {}
        ins_key_to_pos: dict[tuple[int, int], int] = {}
        for p in indel_site_set:
            # ds[p] / il[p] are SETS: one anchor may carry several edits, and
            # each gets its own key so it becomes its own DEL<len>/INS<len>
            # allele. A position can appear in both loops.
            for d_start, d_end in ds.get(p, ()):
                del_key_to_pos[(d_start, d_end - d_start + 1)] = p
            for ilen in il.get(p, ()):
                # (anchor, inserted length): a read only matches an insertion of
                # the SAME size, so different-size insertions stay distinct.
                ins_key_to_pos[(p, ilen)] = p

        # pysam fetch uses 0-based coordinates
        for aln in bam.fetch(contig_id, start - 1, end - 1):
            # SECONDARY stays out: an alternative placement of sequence already counted.
            # SUPPLEMENTARY is kept - it is a different PART of the same molecule, and
            # the segments are merged back into ONE Read below. See _merge_split_reads.
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
                # A read barely clipping the window edge carries almost no information
                # about it, yet would otherwise count identically to one spanning the
                # whole window in n_reads_examined and the abundance denominator.
                continue

            # Parse alleles at SNV sites
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

            # Handle missing quality (warn once)
            if query_qual is None:
                WarningThrottler.warn_once(
                    "no_qual",
                    f"Some reads lack quality scores. Using default Q{config.default_base_quality}.",
                )

            # Extract alleles at SNV positions (matched bases only) via a TARGETED
            # CIGAR walk. get_aligned_pairs() materialises the whole read-length
            # alignment (~15k tuples on a HiFi read) just to find the handful of SNV
            # columns in this window; walking the CIGAR once and emitting only at
            # this read's SNV positions is the IDENTICAL result at O(variants on
            # read) instead of O(read length) - the dominant cost of iter_windows_lazy.
            # Verified byte-identical against the get_aligned_pairs form on real data.
            # Indel/SV sites are excluded here and handled by the CIGAR scan below.
            has_overlap = False
            # SNV-only targets in this window. window_snvs is sorted (a bisect slice
            # of snv_pos_sorted; break sites are appended only after this loop), so
            # the walk can advance through it monotonically.
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
                        # Targets before this segment fell in a D/N gap (or before the
                        # read): no query base, no call - exactly what get_aligned_pairs
                        # yields as (None, ref) for a deleted reference position.
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

            # Extract alleles at indel sites.
            #
            # We trust the variant caller's positions exactly (run bcftools
            # norm upstream). For each VCF indel site, the read carries the
            # variant iff its CIGAR contains the matching indel op:
            #
            #   DEL: a D op of exactly the deleted length, at the footprint's
            #        first base -> allele "DEL<len>"
            #   INS: an I op of exactly the inserted length, at the VCF anchor
            #        -> allele "INS<len>"
            # The size is part of the allele, so a <len>-bp indel clusters only
            # with reads carrying the same-size edit (not every indel here).
            # Otherwise, if a matched base (M/=/X) covers the anchor, record
            # it as the read's "vote against" the indel. Otherwise, no call.
            if special_site_set and aln.cigartuples:
                # Quality attached to a DEL<len> / INS<len> / event-id token. These are
                # not base calls, so there is no per-base Phred score for them; MAPQ used
                # to be stored here and was then read back as one by the EM, pricing a
                # MAPQ-60 indel disagreement at 15.42 nats against 8.52 for a Q30 SNV -
                # one indel outweighing 1.8 SNVs, on the least reliable marker class.
                # MAPQ is a property of the alignment and is kept where it belongs, on
                # Read.mapq; min_mapq has already gated on it above.
                token_qual = config.default_base_quality
                # Single CIGAR walk: collect indel events and the matched-base
                # ref->query mapping for indel/SV-anchor positions only.
                ref_cursor = aln.reference_start  # 0-based
                query_cursor = 0
                # Calls produced by this read at indel anchors. We assemble
                # them, then resolve REF-base fallback at the end.
                indel_calls: dict[int, tuple[str, int]] = {}
                # ref_pos_1b -> query_idx for indel/SV anchors covered by M/=/X.
                anchor_qpos: dict[int, int] = {}

                for op, length in aln.cigartuples:
                    if op in (0, 7, 8):  # M / = / X — consumes both
                        # For each indel/SV anchor inside this M op, remember
                        # the query index so we can record the matched base
                        # if no DEL/INS/SV call is made.
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
                        # Anchor is the 1-based position immediately before the
                        # inserted bases (== ref_cursor in 1-based terms). Match on
                        # (anchor, inserted length) so a <len>-bp insertion is its
                        # own allele rather than collapsing to a generic "INS".
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

                # Apply SV pseudo-site "present" calls. A read carries a specific
                # event iff its name is in THAT event's supporting-read set AND
                # this alignment's reference span brackets the breakpoint anchor
                # (small pad for imprecision / soft-clip). The allele is the
                # event ID, so two distinct events at one anchor are a genuinely
                # multi-allelic site. The span check prevents a read whose *other*
                # split segment supports the SV elsewhere from being called here.
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
        if config.merge_split_reads:
            reads, break_sites = _merge_split_reads(reads, config, snv_set)
            for bp in break_sites:
                if bp not in snv_set and start <= bp < end:
                    window_snvs.append(bp)
                    snv_set.add(bp)
                    ref_alleles[bp] = CONTINUOUS
                    st[bp] = "sv"
            if break_sites:
                window_snvs.sort()

        # Subsample if needed (reproducible)
        if config.max_reads_per_window and len(reads) > config.max_reads_per_window:
            if config.consistent_read_subsampling:
                # CONSISTENT across windows: keep the reads whose id hashes smallest,
                # so two overlapping windows pick THE SAME reads out of the molecules
                # they share. An independent draw per window (the branch below) keeps
                # a shared read in both only with probability (cap/N)^2 - at 20 kb
                # windows and a 200-read cap over a few thousand reads that leaves
                # ~2-3 shared reads where the biology has hundreds, which reads as a
                # linking failure when it is really a sampling artefact. Selecting on
                # a stable function of the read ID makes the loss LINEAR instead:
                # a read kept in one window is kept in its neighbour too.
                reads = [
                    r for _, _, r in sorted(
                        (_read_sort_hash(r.id, config.random_seed), i, r)
                        for i, r in enumerate(reads)
                    )[: config.max_reads_per_window]
                ]
            else:
                indices = rng.permutation(len(reads))[: config.max_reads_per_window]
                reads = [reads[i] for i in indices]

        # Window CREATION uses the lower rescue floor, so windows in the
        # [min_reads_for_rescue, min_reads_per_window) band still exist and can receive a
        # rescued haplotype from the anchor panel. De-novo PHASING is gated separately in
        # process_window() at the higher min_reads_per_window. If creation used the
        # phasing floor instead, the rescue floor would be unreachable.
        if len(reads) < config.min_reads_for_rescue:
            continue

        w = Window(contig=contig_id, start=start, end=end, sample=sample_id, window_idx=window_idx)
        w.snv_pos = window_snvs
        # Guarded: every registered position gets an anchor base, and SV
        # pseudo-sites get one at merge time, but a caller supplying positions
        # without one must not take the whole contig down here.
        w.ref_alleles = {p: ref_alleles[p] for p in window_snvs if p in ref_alleles}
        w.reads = reads
        # Carry the site types through. The identity code needs them to exclude SV
        # positions from the distance; if they stop here the exclusion silently no-ops.
        w.site_type = {p: st[p] for p in window_snvs if p in st}
        yield w
        window_idx += 1

    bam.close()


# =============================================================================
# OPTIMIZED GRAPH INITIALIZER
# =============================================================================


class GraphInitializer:
    """
    Graph-based initialization with performance optimizations:
    - Precomputed position sets
    - Early exit on mismatch threshold
    """

    def __init__(self, config: HaplotyperConfig = DEFAULT_CONFIG):
        self.config = config

    def build_overlap_graph(self, window: Window) -> nx.Graph:
        """Build overlap graph with optimized edge computation."""
        graph = nx.Graph()
        reads = window.reads
        n_reads = len(reads)

        # Add one node per read (nodes are read indices).
        for i in range(n_reads):
            graph.add_node(i)

        # Precompute the set of SNV positions each read covers
        # (window.get_read_position_sets caches these).
        pos_sets = window.get_read_position_sets()

        # Compare read pairs to decide if they should be connected.
        # We only connect reads that share enough SNVs and agree closely.
        for i in range(n_reads):
            pos_i = pos_sets[i]
            if not pos_i:
                continue

            for j in range(i + 1, n_reads):
                pos_j = pos_sets[j]
                # Shared SNV positions between the two reads.
                shared = pos_i & pos_j
                n_shared = len(shared)

                # Require a minimum amount of overlap to reduce noise.
                if n_shared < self.config.min_shared_snvs_for_edge:
                    continue

                r_i, r_j = reads[i], reads[j]

                # Physical read-read overlap. -1 means the alignment spans are unknown
                # (e.g. synthetic reads in tests), in which case the gate is skipped
                # rather than rejecting everything.
                ov = r_i.overlap_bp(r_j)
                if 0 <= ov < self.config.min_read_read_overlap_bp:
                    continue

                # Count mismatches with early exit. Two independent gates: the RATE
                # (floor(max_mismatch_frac * n_shared), which forces 0 mismatches below
                # n_shared=100) and the ABSOLUTE cap, which is what actually binds once
                # n_shared is large enough for the rate to go permissive.
                max_allowed = min(
                    int(self.config.max_mismatch_frac * n_shared), self.config.max_num_diff
                )
                mismatches = 0
                exceeded = False

                for p in shared:
                    if r_i.alleles[p] != r_j.alleles[p]:
                        mismatches += 1
                        if mismatches > max_allowed:
                            exceeded = True
                            break

                # Add an edge if reads are sufficiently similar.
                if not exceeded:
                    mismatch_frac = mismatches / n_shared
                    # Edge weight = #shared SNVs scaled by agreement (higher is better).
                    weight = (1.0 - mismatch_frac) * n_shared
                    graph.add_edge(i, j, weight=weight)

        return graph

    def derive_consensus(self, cluster_reads: list[Read], window: Window) -> dict[int, str]:
        """Derive consensus from cluster reads."""
        allele_counts = defaultdict(lambda: defaultdict(int))

        # Count alleles at each SNV position across reads in this cluster.
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
        # Build read overlap graph where edges connect reads that agree on SNVs.
        graph = self.build_overlap_graph(window)

        if graph.number_of_edges() == 0:
            # No edges => no clustering signal; fall back to single consensus haplotype.
            consensus = self.derive_consensus(window.reads, window)
            if consensus:
                return [Haplotype(consensus=consensus, supporting_reads=len(window.reads))], [
                    len(window.reads)
                ]
            return [], []

        # Partition reads into clusters.
        # Louvain community detection for read clustering. random_state is passed
        # explicitly: without it python-louvain draws from the unseeded global `random`,
        # which made two runs of an identical config disagree on abundance at ~1e-7.
        partition = community_louvain.best_partition(
            graph, weight="weight", random_state=self.config.random_seed
        )

        # Group by cluster
        clusters = defaultdict(list)
        for node_idx, cluster_id in partition.items():
            clusters[cluster_id].append(window.reads[node_idx])

        # Build initial haplotypes: one consensus per cluster.
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
    """
    EM engine with cached log-probability computations.

    Avoids double computation of log-probs in E-step and log-likelihood.
    """

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

        # The site set every component is scored over, fixed for the whole run. Fixed is
        # the point: it is what makes gamma a posterior (components compared on the same
        # observations) and what makes the log-likelihood a function of the parameters
        # alone, so a decreasing objective is now a real signal rather than the
        # bookkeeping artefact of a consensus footprint that grew between iterations.
        self._snv_set = set(window.snv_pos)
        self._support = [_read_support(r, self._snv_set) for r in self.reads]

    def _compute_log_prob_read_hap(self, read: Read, haplotype: Haplotype) -> float | None:
        """Compute log P(read | haplotype) over this read's fixed support."""
        return _log_prob_read_hap(
            read,
            haplotype.consensus,
            _read_support(read, self._snv_set),
            self.config.default_base_quality,
        )

    def _compute_log_prob_read_junk(self, read: Read) -> float:
        """Compute log P(read | junk) using the divergent reference model."""
        return _log_prob_read_junk(
            read,
            _read_support(read, self._snv_set),
            self.window.ref_alleles,
            self.config.junk_divergence_rate,
            self.config.default_base_quality,
        )

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
        if self.config.use_cluster_pi_init and self.cluster_sizes:
            cluster_total = sum(self.cluster_sizes)
            junk_init = max(1, n_reads - cluster_total)
            pi = np.array(self.cluster_sizes + [junk_init], dtype=float)
            pi /= pi.sum()
        else:
            pi = np.ones(k_eff) / k_eff

        gamma = np.zeros((n_reads, k_eff))
        prev_log_like = -np.inf
        converged = False

        # E-STEP tensors that do not change across iterations: the integer read
        # allele matrix, per-(read, site) match/mismatch log-probs, and the junk
        # log-likelihood (which depends only on the reads and the reference). Built
        # once; only logl_hap is recomputed per iteration as consensuses update.
        # Vectorised replacement for the per-(read, hap) _log_prob_read_hap /
        # _log_prob_read_junk loop that dominated EM runtime.
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
        # Derived, also constant across iterations, for the vectorised M-step vote:
        # per-(read, site) log_odds = log_match - log_mismatch; a code->allele
        # decoder and a site-index->position list; the read-allele alphabet size.
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

            # E-STEP: responsibilities gamma[i,k] and the data log-likelihood, both
            # from ONE batched logsumexp over the (n_reads x k_eff) log-posterior.
            # (Was two per-read Python loops each calling scipy logsumexp per read.)
            log_pi = np.log(pi + 1e-12)
            logp = np.full((n_reads, k_eff), -np.inf)
            logp[:, :n_haps] = log_pi[:n_haps][None, :] + logl_hap  # -inf where logl_hap -inf
            logp[:, junk_idx] = log_pi[junk_idx] + logl_junk
            log_sum = logsumexp(logp, axis=1)          # (n_reads,)
            gamma[:] = 0.0
            good = ~np.isneginf(log_sum)
            gamma[good] = np.exp(logp[good] - log_sum[good, None])
            gamma[~good, junk_idx] = 1.0
            # Junk is always finite, so log_sum is finite for every read; summing all
            # of it reproduces the per-read logsumexp total (a -inf read would sum to
            # -inf here too, matching the old loop).
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

                # Rebuild this haplotype's consensus by gamma-weighted voting. The
                # objective, quality weighting (gamma * log_odds vote, gamma *
                # log_mismatch floor) and the call-nothing test are unchanged; the
                # per-read Python triple loop is replaced by numpy over the read x
                # site tensors (see _em_hap_consensus). "no vote" -> None, "all
                # positions failed the floor" -> {} both fall through to `if
                # new_consensus:` below, exactly as `continue` did.
                new_consensus = _em_hap_consensus(
                    gamma[:, k], _em_read_code, _em_sup_mask, _em_lo, _em_lmm,
                    _em_n_alleles, _em_sites, _em_dec, _em_snv_pos,
                    self.config.min_gamma_for_vote,
                )
                if new_consensus:
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
            # A read whose whole responsibility sat on haplotypes this iteration pruned
            # has nothing left. Rescaling by 1.0 left the row all zeros, which is not a
            # distribution: validate() rejects it, and junk_read_counts scores the read as
            # neither resolved nor junk. Junk is exactly what "no haplotype explains this
            # read" means, so that is where the mass goes. Usually the next E-step
            # overwrites the row anyway - unless the prune lands on the iteration that
            # also breaks for convergence, which is when this used to escape.
            dead_rows = (row_sums[:, 0] == 0.0)
            if dead_rows.any():
                gamma[dead_rows, :] = 0.0
                gamma[dead_rows, junk_idx] = 1.0
                row_sums[dead_rows] = 1.0
            gamma /= row_sums

            # Convergence check using relative change in log-likelihood.
            # Use relative tolerance for log-likelihood (since log-likelihoods can be large negative numbers)
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
    """Post-processing with optimized merging and 1-SNP validation."""

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
        """Determine if 1-SNP pair should be merged."""
        if not self.config.validate_1snp_differences:
            return True

        diff_positions = hap1.get_differing_positions(hap2, window.snv_pos)
        if len(diff_positions) != 1:
            return True

        diff_pos = diff_positions[0]

        # Identify minor haplotype
        if hap1.weight < hap2.weight:
            minor_hap, minor_k = hap1, k1
        else:
            minor_hap, minor_k = hap2, k2

        # Check frequency
        if minor_hap.weight < self.config.min_minor_frequency_1snp:
            return True

        # Check supporting reads
        minor_supporting = int((gamma[:, minor_k] >= self.config.assign_confidence_threshold).sum())
        if minor_supporting < self.config.min_minor_supporting_reads_1snp:
            return True

        # Check timepoints
        if n_timepoints_seen < self.config.min_timepoints_for_1snp:
            if self.config.use_binomial_test_1snp:
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

                # The reads' own base qualities, not a fixed Q30. The bases at this
                # position are in read.quals and min_base_quality — the gate that decided
                # which of them got here at all — defaults to 20, so assuming Q30 asserted
                # an accuracy the data does not have to carry. It errs towards SPLIT: at
                # 3 minor reads of 40 in a 250-site window, Q30 gives p=3.6e-07 (split)
                # where the reads' real Q20 gives 3.3e-04 (merge), i.e. a phantom strain
                # above the 10% abundance floor. The binomial wants one rate, so the mean
                # over the reads at the position stands in for the Poisson-binomial.
                p_error = p_error_sum / total_at_pos
                alpha_corrected = self.config.binomial_alpha / len(window.snv_pos)
                p_value = 1 - binom.cdf(minor_count - 1, total_at_pos, p_error)

                if p_value > alpha_corrected:
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
        """Merge similar haplotypes with optimized distance computation."""
        n_haps = len(haplotypes)
        if n_haps <= 1:
            return haplotypes, gamma, pi

        # Precompute max allowed mismatches for early exit when comparing haplotypes.
        max_mismatches = int(self.config.merge_distance_threshold * len(window.snv_pos)) + 1

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

                # OPTIMIZATION: Use early exit distance
                dist, n_diff, n_shared = haplotypes[i].distance_to(
                    haplotypes[j], window.snv_pos, max_mismatches
                )

                # Require minimum shared positions to consider merging
                if n_shared < self.config.min_shared_for_merge:
                    continue

                if dist <= self.config.merge_distance_threshold:
                    if n_diff == 1:
                        should_merge = self.should_merge_1snp_pair(
                            haplotypes[i], haplotypes[j], i, j, window, gamma, n_timepoints_seen
                        )
                        if not should_merge:
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

    def assign_reads(self, reads: list[Read], gamma: np.ndarray, pi: np.ndarray) -> list[dict]:
        """Hard assignment of reads.

        Skipped entirely unless ``config.keep_read_assignments`` is set - see that flag.
        Returns ``[]`` in that case, which is what every downstream consumer already
        tolerates (nothing reads this field).
        """
        if not self.config.keep_read_assignments:
            return []

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
    """Cross-timepoint integration with optimized anchor panel construction."""

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
        """
        Build anchor panel directly from sample_results dict.

        OPTIMIZATION: Operates on pre-filtered results for this window key,
        not the full results dictionary.
        """
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
                # Require sufficient shared positions for meaningful comparison
                if n_shared >= self.config.min_shared_for_rescue:
                    if dist <= self.config.rescue_match_distance:
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
        """
        Rescue missing haplotypes by checking if junk reads match anchors from other timepoints.

        This looks at reads currently assigned to the junk model and checks if they
        match a haplotype that was detected in another timepoint. If so, it creates
        a new haplotype from those reads.
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

        # Local variables for readability.
        window = window_result.window
        haplotypes = list(window_result.haplotypes)  # Make mutable copy
        gamma = window_result.gamma.copy()
        pi = window_result.pi.copy()
        reads = window.reads

        n_haps = len(haplotypes)
        junk_idx = n_haps  # Last column in gamma/pi is the junk component.
        junk_weight = pi[junk_idx] if len(pi) > junk_idx else 0.0

        # Identify reads assigned to junk (by posterior probability).
        junk_threshold = 0.5  # Read is "junk" if gamma[:, junk_idx] > this
        junk_read_mask = gamma[:, junk_idx] > junk_threshold
        n_junk_reads = junk_read_mask.sum()

        logging.debug(
            f"    Rescue check: {n_junk_reads}/{len(reads)} junk reads, "
            f"junk_weight={junk_weight:.3f}, {len(anchor_haps)} anchors"
        )

        # Even a single junk read matching an anchor from another timepoint is meaningful,
        # as long as the match is near-exact (controlled by rescue_match_distance).
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

        # A rescue is funded entirely out of junk's weight, down to _MIN_JUNK_WEIGHT.
        # Below that floor there is nothing to fund one with, and proceeding anyway is
        # strictly harmful: every rescued weight is scaled to 0.0, junk is INFLATED up to
        # the floor, and every original haplotype in the window is scaled down to make
        # room for a haplotype contributing nothing - while the statistic recorded
        # was_rescued=True. The posterior read count (n_junk_reads) does not imply the
        # weight is there: a handful of reads can sit above gamma 0.5 on a pi_junk of
        # 0.002.
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

        # Matching thresholds for anchor comparisons.
        max_distance = self.config.rescue_match_distance
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

        # ---- Step 2: Assign each junk read to exactly one anchor (best distance, then lowest index).
        # This avoids double-counting: the same read was previously counted for every matching
        # anchor, inflating total_rescued_weight and biasing original haplotype weights.
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
        # Where each rescued haplotype's statistic landed, so the weight recorded there
        # can be corrected to the one the window actually ends up carrying. The scaling
        # below can only shrink it, and a statistic reporting the pre-scaling figure
        # overstates every rescue that had to compete for a limited junk budget.
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

                if len(new_consensus) >= self.config.min_snvs_per_window:
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

        # Redistribute weight: take from junk only; never take more than available.
        # This avoids zeroing out original haplotypes when rescued_min_weight or
        # (previously) double-counted reads made total_rescued_weight > old_junk_weight.
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
        assignments = post.assign_reads(reads, gamma_new, pi_new)

        for k, hap in enumerate(haplotypes):
            hap.supporting_reads = int(
                (gamma_new[:, k] >= self.config.assign_confidence_threshold).sum()
            )

        n_reads_examined, reads_within_mismatch_per_hap = _compute_read_mismatch_counts(
            window, haplotypes, self.config.max_mismatch_frac
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

        # Scored through the same two helpers the EM uses. A rescued haplotype takes the
        # DONOR's footprint (see the new_consensus build above), so this E-step is exactly
        # where a footprint-asymmetric likelihood would do the most damage - the rescued
        # component is guaranteed to have a different site set from the resident ones.
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

        ``only_sample`` restricts the rescue to that one sample, returning just its
        rescued windows. Every other sample still contributes to the anchor panel, which
        needs nothing but their ``.haplotypes`` - so their reads and gamma may be
        offloaded (see WindowResult.offload_heavy). Calling this once per sample in the
        same iteration order is equivalent to the all-samples call: the panel is indexed
        from the same WindowResult objects and rescue mutates haplotype weights in place
        either way.
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

    # 0) Resolving-depth floor. Windows are CREATED at min_reads_for_rescue so the
    #    anchor panel can still populate them, but de-novo phasing needs enough reads to
    #    actually separate haplotypes. Below the floor we emit a junk-only result: no
    #    haplotype is invented, so the trajectory carries a gap instead of a
    #    manufactured abundance of 1.0.
    if len(window.reads) < config.min_reads_per_window:
        n_reads = len(window.reads)
        gamma = np.ones((n_reads, 1))
        pi = np.array([1.0])
        n_reads_examined, reads_within_mismatch_per_hap = _compute_read_mismatch_counts(
            window, [], config.max_mismatch_frac
        )
        return WindowResult(
            window=window,
            haplotypes=[],
            gamma=gamma,
            pi=pi,
            log_likelihood=-np.inf,
            assignments=post.assign_reads(window.reads, gamma, pi),
            converged=True,
            iterations=0,
            n_reads_examined=n_reads_examined,
            reads_within_mismatch_per_hap=reads_within_mismatch_per_hap,
        )

    # 1) Initialize haplotypes via read clustering on the overlap graph.
    initializer = GraphInitializer(config)
    initial_haps, cluster_sizes = initializer.get_initial_haplotypes(window)

    if not initial_haps:
        # No clustering signal -> return junk-only result.
        # FIX: Return proper junk-only result (gamma = ones, not zeros)
        n_reads = len(window.reads)
        gamma = np.ones((n_reads, 1))
        pi = np.array([1.0])
        assignments = post.assign_reads(window.reads, gamma, pi)

        n_reads_examined, reads_within_mismatch_per_hap = _compute_read_mismatch_counts(
            window, [], config.max_mismatch_frac
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
        assignments = post.assign_reads(window.reads, gamma, pi)
        n_reads_examined, reads_within_mismatch_per_hap = _compute_read_mismatch_counts(
            window, [], config.max_mismatch_frac
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
    merged_haps, final_gamma, final_pi = post.merge_similar_haplotypes(
        haplotypes, gamma, pi, window, n_timepoints_seen
    )

    # 3b) NO invariant-site pruning here.
    #
    #     This step used to delete positions where the haplotypes in THIS window agreed,
    #     gated on `if len(merged_haps) >= 2`. It was wrong in both directions:
    #
    #       * skipped entirely for single-haplotype windows (85% of windows on
    #         000089747_1), so those kept every position they covered - a median of 743
    #         positions of which 42.6% are invariant across the entire MAG - while
    #         2-haplotype windows kept a median of 3. Distances were therefore computed
    #         on wildly asymmetric marker sets.
    #       * window-LOCAL, so a genuinely polymorphic position that happened to be
    #         monomorphic among this window's haplotypes was destroyed for good.
    #
    #     Pruning is now a comparison-time concern, not a construction-time one:
    #     `variable_marker_positions()` computes the marker set at the widest scope
    #     available (all haplotypes in the sample for window linking; all haplotypes in
    #     all samples for cross-sample grouping) and the linking code filters against it.
    #     Construction keeps everything; nothing is destroyed before the scope is known.
    #     (FIGURE4 diagnosis §2.6a.)

    assignments = post.assign_reads(window.reads, final_gamma, final_pi)

    n_reads_examined, reads_within_mismatch_per_hap = _compute_read_mismatch_counts(
        window, merged_haps, config.max_mismatch_frac
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

    # 4) Optional validation checks on the WindowResult structure.
    if config.validate_results:
        result.validate()

    return result



# =============================================================================
# SPLIT MOLECULES: re-assembly and the BREAK marker
# =============================================================================

# Allele values at a breakpoint site. A read that crosses the position with a continuous
# alignment carries CONTINUOUS; one whose alignment is split there carries
# BREAK_PREFIX + the coordinate it resumes at, so two different events at the same left
# coordinate stay distinct alleles. Same shape as the INS<len> / DEL<len> encoding.
CONTINUOUS = "CONT"
BREAK_PREFIX = "BRK"


def _merge_split_reads(
    reads: list[Read],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    snv_set: set[int] | None = None,
) -> tuple[list[Read], set[int]]:
    """Re-assemble split alignments into one Read each, and report the breakpoints.

    When a strain carries a segment the aligner cannot place - a divergent cassette, an
    insertion, a rearrangement - the molecule is emitted as a primary alignment plus one
    or more supplementary ones, with the unplaceable part clipped. Those segments are ONE
    molecule and therefore one strain, so they are merged back into a single Read.

    Two things follow, and both matter:

    1. The merged read carries alleles from BOTH sides of the break, so it links them in
       the read graph and the EM assigns the whole molecule to one haplotype. The
       fragments never form. Previously the segments were discarded outright and one
       strain was emitted as several haplotypes, each handed a share of the window's
       mixture weight (troubleshooting U1).
    2. The break itself becomes a POSITIVE identity marker. A strain carrying the cassette
       reads ``BRK<resume_pos>``; a strain without it reads ``CONT``. That is a
       discriminating allele like any other, so the structural variant is tracked rather
       than being a hole in the data - which is the point of not excluding SVs from
       identity.

    The breakpoint is anchored at the LAST aligned reference position of the preceding
    segment, or at the nearest earlier position inside that segment that is not already a
    called variant site. Shifting matters: the anchor is written unconditionally, so on a
    position the VCF also calls, the BRK token replaced the read's real allele there and
    ``site_type`` still said ``"snv"`` - the same clobbering the SV sidecar merge refuses
    outright (caveat 7). Pass *snv_set* to enable the check; without it there is nothing
    to check against and the anchor is taken as-is.

    Returns the merged reads and the set of breakpoint positions.
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
            # Walk back inside the preceding segment until the anchor is a position no
            # variant was called at. Anywhere in that segment carries the same meaning -
            # every read continuous across the gap is aligned there too - so shifting
            # keeps the marker instead of choosing between it and a real allele.
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

    # A read that spans a breakpoint with an unbroken alignment is evidence AGAINST the
    # event, not absence of evidence. Without this the marker cannot discriminate: only
    # the broken strain would carry a call and compare_consensus scores solely positions
    # where both sides have one.
    #
    # Read.covers, not the outer span: a molecule merged from segments [1000,5000) and
    # [8000,15000) spans 1000-15000, so an outer-span test wrote "no break here" at 5001,
    # inside that read's OWN unaligned gap. Two molecules carrying the same event with
    # breakpoints ragged by a couple of bases then disagreed at the fabricated site and
    # build_overlap_graph dropped the edge between them - the U1 fragmentation this
    # function exists to prevent.
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


def variable_marker_positions(
    consensuses: Iterable[dict[int, str]],
    site_type: dict[int, str] | None = None,
    config: HaplotyperConfig = DEFAULT_CONFIG,
) -> set[int]:
    """Positions usable as identity markers, computed over *consensuses*.

    A position is a marker only if at least two distinct alleles are observed across the
    whole collection. A position where every haplotype agrees carries zero identity
    information, yet still inflates ``n_shared`` and dilutes the mismatch rate - on
    ``000089747_1`` 42.6% of emitted positions were invariant MAG-wide, accounting for
    38.5% of all consensus entries.

    Call this at the WIDEST scope available for the comparison being made:
      * window linking within a sample -> every haplotype in that sample/contig
      * grouping across samples        -> every haplotype in every sample, that contig

    Structural variants ARE identity markers (``exclude_sv_from_identity`` defaults to
    False, author's decision): capturing the trajectory of a flip is a goal of the
    analysis. When an invertible element flips, the two orientations are reported as two
    entities trading frequency over time, which is the trajectory. The flag exists only
    so that effect can be measured.
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

    Returned as (lo, hi) with hi < lo when nothing falls inside the region, so callers can
    test emptiness without a second pass. Deliberately avoids materialising a set: this is
    called once per haplotype per comparison batch, not once per pair.
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
    """Outcome of one identity comparison.

    ``reason`` distinguishes the two ways a comparison can fail, which is the
    information the (paused) lineage layer needs in order to tell a measurement hole
    from a real genotypic wall:

      ``"linked"``              passed every gate
      ``"failed_no_evidence"``  too few shared markers, or too little physical overlap -
                                a DROPOUT. Nothing was shown to differ.
      ``"failed_mismatch"``     enough shared markers, but the alleles genuinely disagree
                                beyond the gates - a candidate recombination breakpoint.
    """

    passed: bool
    reason: str
    rate: float
    n_shared: int
    n_diff: int
    # True when no discriminating markers were available and the comparison fell back to
    # all co-covered positions. See compare_consensus for why that fallback exists.
    used_fallback: bool = False


def compare_consensus(
    a: dict[int, str],
    b: dict[int, str],
    markers: set[int],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    min_shared: int | None = None,
    region: tuple[int, int] | None = None,
    min_cospan_frac: float | None = None,
    max_rate: float | None = None,
    max_num_diff: int | None = None,
    allow_fallback: bool = True,
    a_span: tuple[int, int] | None = None,
    b_span: tuple[int, int] | None = None,
) -> GateResult:
    """Apply the full identity gate stack to two consensus dicts.

    Gates, in order: the two footprints must overlap by >= ``min_entity_overlap_bp``
    (and >= ``min_cospan_frac`` of ``region``); shared markers >= ``min_shared``;
    absolute mismatches <= ``max_num_diff``; mismatch rate <= ``lineage_merge_distance``.

    The overlap gate asks only "how much sequence did both haplotypes cover", never
    where the markers within it happen to fall.

    ``a_span``/``b_span`` are precomputed ``consensus_footprint`` results. Pass them when
    comparing many pairs - the footprint does not depend on the partner, and recomputing it
    per pair dominates the cost.

    ``allow_fallback=False`` forbids the clonal fallback, so the verdict rests only on
    genuinely discriminating markers. Use it whenever a NEGATIVE verdict will be treated
    as absolute: the fallback pads the comparison with positions that are invariant in
    scope and therefore cannot disagree, which is harmless for a merge but would let a
    mismatch be declared off evidence that was never discriminating.

    The absolute cap and the rate guard opposite ends of the range: the rate is applied
    as a floor, so it already forces zero mismatches below n_shared=100, while at
    n_shared=1172 it would tolerate 11 - which is where the absolute cap binds.
    """
    if min_shared is None:
        min_shared = config.min_shared_for_lineage
    if min_cospan_frac is None:
        min_cospan_frac = config.min_cosupported_span_frac
    if max_rate is None:
        max_rate = config.lineage_merge_distance
    if max_num_diff is None:
        max_num_diff = config.max_num_diff

    def _restrict(positions):
        if region is None:
            # already a set from the & below; copying it was pure overhead
            return positions if isinstance(positions, set) else set(positions)
        lo, hi = region
        return {p for p in positions if lo <= p <= hi}

    shared = _restrict(a.keys() & b.keys() & markers)
    used_fallback = False

    if len(shared) < min_shared and allow_fallback:
        # No DISCRIMINATING markers between these two. Absence of discriminating
        # evidence is not evidence of difference: a clonal locus legitimately has no
        # variable positions at all, and a clonal SAMPLE can have almost none - 85% of
        # windows on 000089747_1 hold a single haplotype. Restricting to markers would
        # then make every comparison impossible and split a real lineage into singletons.
        # Fall back to all co-covered positions, and record that we did so.
        fallback = _restrict(a.keys() & b.keys())
        if len(fallback) >= min_shared:
            shared, used_fallback = fallback, True

    n_shared = len(shared)

    # HOW MUCH SEQUENCE DID WE ACTUALLY COMPARE? The stretch over which BOTH haplotypes
    # have calls - their footprints' intersection. This is deliberately independent of
    # WHERE the markers sit.
    #
    # It used to be `max(shared) - min(shared)`, the span of the marker subset, which
    # measures how spread out the informative positions happen to be rather than how
    # much sequence was jointly observed. That penalises exactly the loci where variation
    # clusters (recombination tracts, hypervariable and phase-variable regions) and is
    # backwards on evidence: 50 markers packed into 300 bp failed, while 2 markers
    # 1,100 bp apart passed. Measured on 000066952_0: at window 1,880,001 every one of
    # the 703 pairs overlapped by 6,126 bp yet had its 2 markers only 190 bp apart, so
    # all 703 were rejected and a 2-genotype locus was emitted as 38 singleton groups.
    # A footprint is a property of ONE haplotype, so it must not be recomputed per pair.
    # Building the two position sets here cost 77% of this function's runtime and scaled
    # as O(n^2 * positions) per window; `consensus_footprint` is O(positions) and callers
    # that compare many pairs should hoist it out of the loop entirely (see a_span/b_span).
    lo_a, hi_a = a_span if a_span is not None else consensus_footprint(a, region)
    lo_b, hi_b = b_span if b_span is not None else consensus_footprint(b, region)
    overlap = (min(hi_a, hi_b) - max(lo_a, lo_b)) if (hi_a >= lo_a and hi_b >= lo_b) else 0
    if overlap < config.min_entity_overlap_bp:
        return GateResult(False, "failed_no_evidence", 1.0, n_shared, 0, used_fallback)
    if region is not None:
        lo, hi = region
        if hi > lo and overlap < min_cospan_frac * (hi - lo):
            return GateResult(False, "failed_no_evidence", 1.0, n_shared, 0, used_fallback)

    # ...AND WAS ANY OF IT INFORMATIVE? Separate question, separate gate. These two were
    # previously tangled into one number, which is also why min_shared ended up doubling
    # as the trigger for the clonal fallback above.
    if n_shared < min_shared:
        return GateResult(False, "failed_no_evidence", 1.0, n_shared, 0, used_fallback)

    n_diff = sum(1 for p in shared if a[p] != b[p])
    rate = n_diff / n_shared
    if n_diff > max_num_diff or rate > max_rate:
        return GateResult(False, "failed_mismatch", rate, n_shared, n_diff, used_fallback)
    return GateResult(True, "linked", rate, n_shared, n_diff, used_fallback)


def unique_best_matches(
    matches: dict[int, list[tuple[float, int]]],
) -> dict[int, int]:
    """Keep only unambiguous best matches; a tie contributes NOTHING.

    Shared by window linking and cross-sample grouping. Uniqueness on both sides means
    every node has at most one partner in each direction, so a connected component is a
    PATH rather than a hub - which is what bounds an entity's size by construction
    instead of by tuning.
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
    """
    Link haplotypes across overlapping windows based on consensus similarity.

    Since windows overlap by 50%, adjacent windows share SNV positions.
    Haplotypes are linked (assigned the same track_id) if their consensus agrees on the
    shared SNVs AND their within-window shares are compatible.

    The abundance check is an ELIMINATOR, never an indicator: two adjacent windows in one
    sample are the same timepoint, so a genome cannot sit at two frequencies across them,
    and a genuine disagreement means they are not one entity. Agreement earns no credit
    and cannot rescue a failed identity gate. Tested on RAW COUNTS - the derived abundance
    is already quantised onto unit fractions by a median denominator of 9 non-junk reads.

    This modifies haplotypes in-place by setting their track_id field.
    """
    # Deferred: strainphase.coherence imports HaplotyperConfig from this module, so a
    # top-level import here would be circular. The cost is one lookup per call.
    from strainphase.coherence import abundance_coherent

    # This pass re-derives every mismatch it is about to record, so it owns the list
    # rather than appending to whatever was there. The longitudinal driver calls this a
    # second time after cross-timepoint rescue, and rescue hands back the ORIGINAL object
    # on every early-return path - so an append-only list gave an unrescued window its
    # first-pass rows plus a byte-identical second copy, inflating both
    # mismatches_within_sample.tsv and the QC count logged from it.
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

    # Marker set at the widest scope available here: every haplotype in this sample and
    # contig. Positions where every haplotype agrees carry no identity information and
    # are excluded, as are SV sites. Construction no longer prunes, so this is where the
    # non-informative positions are removed.
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

    # Add all nodes
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

        # Check next few windows for overlap
        for k in range(i + 1, min(i + 3, len(sorted_results))):
            next_wr = sorted_results[k]

            # Check if windows overlap
            if next_wr.window.start >= curr_wr.window.end:
                break  # No more overlapping windows

            next_snvs = set(next_wr.window.snv_pos)
            shared_snvs = list(curr_snvs & next_snvs)

            # Note: This checks window-level overlap, but the real check is below
            # where we verify haplotypes actually have calls at shared positions
            if len(shared_snvs) < config.min_shared_snvs_for_link:
                continue

            # The region shared by the two windows; the co-supported span gate is
            # measured as a fraction of it.
            region = (
                max(curr_wr.window.start, next_wr.window.start),
                min(curr_wr.window.end, next_wr.window.end) - 1,
            )

            # Evaluate candidate pairings before linking (avoid cross-links).
            # Full gate stack: shared markers >= min_shared_calls_for_link, co-supported
            # span >= 25% of the shared region, num_diff <= 1, rate <= max_link_distance.
            candidates: list[tuple[int, int, float, int]] = []

            # Non-junk read count per window - the denominator the abundance eliminator
            # tests against. Same definition build_window_tables uses for `total_reads`.
            def _nonjunk(wr) -> int:
                if wr.gamma is None or wr.gamma.size == 0:
                    return 0
                junk = wr.gamma.shape[1] - 1
                return int(wr.gamma.shape[0] - (wr.gamma[:, junk] >= 0.5).sum())

            n_curr, n_next = _nonjunk(curr_wr), _nonjunk(next_wr)

            # per-haplotype footprints, clipped to the shared region, hoisted out of the
            # pairwise loop (see consensus_footprint)
            span_i = [consensus_footprint(h.consensus, region) for h in curr_wr.haplotypes]
            span_j = [consensus_footprint(h.consensus, region) for h in next_wr.haplotypes]
            for hi, hap_i in enumerate(curr_wr.haplotypes):
                for hj, hap_j in enumerate(next_wr.haplotypes):
                    gate = compare_consensus(
                        hap_i.consensus,
                        hap_j.consensus,
                        markers,
                        config,
                        min_shared=config.min_shared_calls_for_link,
                        region=region,
                        max_rate=config.max_link_distance,
                        max_num_diff=config.max_link_num_diff,
                        a_span=span_i[hi],
                        b_span=span_j[hj],
                    )
                    if gate.passed:
                        # ABUNDANCE AS AN ELIMINATOR (author's rule, 2026-07-28).
                        #
                        # This is a SINGLE-TIMEPOINT comparison - one sample, two adjacent
                        # windows - so it is exactly the case the coherence test is for: a
                        # genome cannot sit at two different frequencies at one moment.
                        # Two window-haplotypes whose shares genuinely disagree are not the
                        # same entity, however well their alleles match.
                        #
                        # ELIMINATOR ONLY. Agreement never scores and never rescues a
                        # failed identity gate; it can only refuse. And the test is run on
                        # RAW COUNTS, never the derived abundance, which is already
                        # quantised onto unit fractions by a median denominator of 9.
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
                            continue
                        candidates.append((hi, hj, gate.rate, gate.n_shared))
                    elif gate.reason == "failed_mismatch":
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
                                "used_fallback": gate.used_fallback,
                            }
                        )

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
    """
    Process all windows in a contig and link haplotypes across windows.

    Windows overlap by 50% to enable linking haplotypes based on
    consensus similarity in shared SNV positions.

    If ``sv_sidecar_path`` is given (see :mod:`strainphase.sv_encoding`),
    structural variants are merged in as pseudo-sites and co-phased with SNVs.
    The "present" allele is the unique event ID, so a locus carrying two distinct
    events is a multi-allelic site. Backward compatible: ``None`` reproduces
    SNV/indel-only behavior.
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

        # reconcile may move an event's anchor by up to _RECONCILE_MAX_SPAN, and a read
        # is only credited with the event if its span brackets the anchor to within
        # _SV_ANCHOR_PAD. If the former ever exceeds the latter, reconciling a sidecar
        # silently costs present-calls on reads that genuinely carry the event. The
        # coupling was stated in prose at both ends and enforced at neither.
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

    # 2) Create overlapping windows with lazy read loading.
    #    This is an ITERATOR: windows are pulled from the BAM a batch at a time so the
    #    whole contig's reads are never resident at once (see iter_windows_lazy).
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

    # 3) Process windows (parallel if a pool is supplied or n_workers > 1).
    #    Batching serves memory, not scheduling: pool.map over the full window list
    #    pickles every window out and every result back with both copies alive in the
    #    parent. Feeding batches caps that transient at batch_size windows.
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
                # Logged on dispatch, not on completion: a variant-dense contig can take
                # many minutes and this is the line that shows the run is alive. The
                # total is not known up front - windows arrive from an iterator.
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
            # Release each window's read payload (the position-keyed allele/qual dicts,
            # ~97% of a WindowResult's footprint) as soon as its EM is done, keeping
            # id-only stand-ins in gamma-row order. Neither link_windows (haplotypes +
            # gamma only) nor a read-partition consumer needs the alleles again on this
            # path, so holding every window's reads to the end of the contig is dead
            # weight. OFF by default: the longitudinal caller manages its own spill and
            # rescue, which DO re-read alleles, so it must keep them. See _detach_reads.
            if offload_reads:
                for wr in batch_results:
                    _detach_reads(wr)
            results.extend(batch_results)
            # Drop this batch's Window references before the next one is pulled. The
            # results still hold their own window (rescue needs the reads), but the
            # input list must not pin a second copy.
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
    """
    Backwards-compatible wrapper.

    Canonical implementation lives in `strainphase.longitudinal.process_mag_longitudinal`.
    This wrapper exists so older code that imported `process_mag_longitudinal` from
    `strainphase.core` / `strainphase` keeps working.
    """
    # Import lazily to avoid circular imports (`strainphase.longitudinal` imports `core`).
    from strainphase.longitudinal import process_mag_longitudinal as _impl

    # Legacy signature:
    #   process_mag_longitudinal(samples: Dict[str, Tuple[bam, vcf]],
    #                            mag_contigs: Dict[str, int],
    #                            config: HaplotyperConfig = DEFAULT_CONFIG)
    #
    # New canonical signature:
    #   process_mag_longitudinal(mag_name: Optional[str],
    #                            mag_contigs: Dict[str, int],
    #                            samples: List[str],
    #                            bam_paths: Dict[str, str],
    #                            vcf_paths: Dict[str, str],
    #                            config: HaplotyperConfig)
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
    """
    Convert results to track-based records for DataFrame.

    Groups haplotypes by track_id and computes span across all windows
    in each track. This produces one row per track, with span_start and
    span_end reflecting the full linked haplotype extent.
    """
    records = []

    for contig_id, window_results in results.items():
        # Group haplotypes by track_id
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

                # Per-window abundance (conditioned on non-junk), or NO MEASUREMENT.
                # A window with no pi vector, a short one, or one that came out entirely
                # junk did not measure this track at zero — it did not measure it. Same
                # rule as longitudinal._window_conditional_abundance, and for the same
                # reason: counting those windows as 0.0 dropped a track from 0.87 to 0.50
                # on the strength of a window that made no measurement at all.
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

            # Build merged consensus from votes
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

            # Pooled over reads, not averaged over windows. A mean of per-window ratios
            # weights a 5-read window like a 40-read one — the estimator this codebase
            # documents at length as the wrong one (longitudinal._pooled_abundance,
            # lineages.PooledAbundance: it is what produced the sawtooth). NaN, not 0.0,
            # when no window measured the track: absent evidence is not evidence of
            # absence, and a consumer can drop a NaN but cannot recover a fabricated zero.
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


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Haplotype reconstruction for PacBio HiFi metagenomics"
    )
    parser.add_argument("--bam", required=True)
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--contig", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--sample", help="Sample ID")
    parser.add_argument("--vcf-sample", help="Sample name in VCF")
    parser.add_argument("--output", default="haplotypes.tsv")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Seeded by default - see HaplotyperConfig.random_seed.",
    )
    parser.add_argument("--window-size", type=int, default=20000)
    parser.add_argument("--max-reads", type=int, default=10000)
    parser.add_argument("--no-validate", action="store_true", help="Disable result validation")

    args = parser.parse_args()

    config = HaplotyperConfig(
        window_size=args.window_size,
        max_reads_per_window=args.max_reads,
        random_seed=args.seed,
        validate_results=not args.no_validate,
    )

    logging.basicConfig(level=logging.INFO)

    results = process_contig(
        args.bam, args.vcf, args.contig, args.length, config, args.sample, args.vcf_sample
    )

    records = results_to_dataframe({args.contig: results})

    if records:
        import csv

        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(records)
        print(f"Wrote {len(records)} haplotypes to {args.output}")
    else:
        print("No haplotypes found")
