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

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool

import community as community_louvain
import networkx as nx
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

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
    window_size: int = 10000
    min_snvs_per_window: int = 3
    min_reads_per_window: int = 5

    # =========== READ FILTERING ===========
    min_mapq: int = 20
    min_base_quality: int = 20
    default_base_quality: int = 20
    max_reads_per_window: int = 1000

    # =========== SNV FILTERING (Clair3) ===========
    min_depth_site: int = 3
    af_range: tuple[float, float] = (0.05, 0.95)
    require_biallelic: bool = True
    skip_af_filter_if_missing: bool = True

    # =========== GRAPH CONSTRUCTION ===========
    min_shared_snvs_for_edge: int = 1
    max_mismatch_frac: float = 0.005
    min_reads_per_cluster: int = 3

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
    merge_distance_threshold: float = 0.02
    min_shared_for_merge: int = 3  # Min shared SNVs with actual calls to consider merging
    assign_confidence_threshold: float = 0.80

    # =========== 1-SNP VALIDATION ===========
    validate_1snp_differences: bool = True
    min_minor_frequency_1snp: float = 0.10
    min_minor_supporting_reads_1snp: int = 3
    min_timepoints_for_1snp: int = 2
    use_binomial_test_1snp: bool = True
    binomial_alpha: float = 0.05

    # =========== LONGITUDINAL PARAMETERS ===========
    min_weight_for_anchor: float = 0.2
    rescue_match_distance: float = 0.01  # 0.1% error rate — near-exact match required
    min_shared_for_rescue: int = 3  # Min shared SNVs with actual calls for rescue matching
    rescued_min_weight: float = 0.02

    # =========== LINEAGE CLUSTERING PARAMETERS ===========
    # Controls how tracks are clustered into lineages across samples
    lineage_merge_distance: float = 0.02  # Max distance to merge tracks into same lineage
    min_shared_for_lineage: int = 3  # Min shared SNVs to consider merging into lineage

    # =========== LINKING DIAGNOSTICS ===========
    linking_debug: bool = False  # Record detailed linking diagnostics
    linking_debug_max_records: int = 5000  # Cap to avoid massive files
    max_span_gap_for_lineage: int = 10000  # Max gap between track spans to consider same locus

    # =========== WINDOW LINKING PARAMETERS ===========
    # Haplotypes in adjacent overlapping windows are linked if their
    # consensus agrees on shared SNVs (Hamming distance <= max_link_distance)
    max_link_distance: float = 0.005  # Max mismatch fraction to link
    min_shared_snvs_for_link: int = (
        3  # Min shared SNVs with ACTUAL CALLS to link (not just window overlap)
    )

    # =========== RUNTIME PARAMETERS ===========
    random_seed: int | None = None
    validate_results: bool = False  # Set False for production runs
    n_workers: int = 1  # Number of parallel workers for window processing (1=sequential)

    def __post_init__(self):
        """Validate configuration parameters."""
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

        # AF range
        if not (0 <= self.af_range[0] < self.af_range[1] <= 1):
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


@dataclass
class Window:
    """
    Represents a genomic window (contig interval) with associated SNVs and reads.

    Notes:
    - snv_pos and ref_alleles are populated from the VCF (see load_snvs_from_clair3).
    - reads are pulled from the BAM and filtered to this window in make_windows_lazy.
    - sample is optional metadata; reads may also carry their own sample tag.
    """

    contig: str
    start: int  # 1-based, inclusive
    end: int  # 1-based, exclusive
    snv_pos: list[int] = field(default_factory=list)  # SNV positions (from VCF)
    ref_alleles: dict[int, str] = field(default_factory=dict)  # REF base per SNV (from VCF)
    reads: list[Read] = field(default_factory=list)  # Reads overlapping this window (from BAM)
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
        self, other: "Haplotype", positions: list[int], max_mismatches: int | None = None
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

    def get_differing_positions(self, other: "Haplotype", positions: list[int]) -> list[int]:
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

        return True


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

    Avoids redundant 10**(-Q/10) calculations.
    """

    def __init__(self, max_q: int = 60):
        """Precompute log probabilities for all Q scores."""
        self._log_match = np.zeros(max_q + 1)
        self._log_mismatch = np.zeros(max_q + 1)

        for q in range(max_q + 1):
            p_err = 10 ** (-q / 10.0)
            self._log_match[q] = np.log(1.0 - p_err + 1e-12)
            self._log_mismatch[q] = np.log(p_err / 3.0 + 1e-12)

    def log_prob_base(self, hap_base: str, read_base: str, q: int) -> float:
        """Get log probability from cache."""
        q = min(q, len(self._log_match) - 1)
        if hap_base == read_base:
            return self._log_match[q]
        return self._log_mismatch[q]


# Global cache instance
_LOG_PROB_CACHE = LogProbCache()


# =============================================================================
# I/O FUNCTIONS - LAZY LOADING
# =============================================================================


def load_snvs_from_clair3(
    vcf_path: str,
    contig_id: str | None = None,
    sample_name: str | None = None,
    config: HaplotyperConfig = DEFAULT_CONFIG,
) -> tuple[list[int], dict[int, str], dict[int, int], dict[int, float | None]]:
    """Load SNVs from Clair3 VCF."""
    if not HAS_PYSAM:
        raise ImportError("pysam required for VCF parsing")

    snv_pos = []
    ref_alleles = {}
    depth = {}
    af = {}

    vcf = pysam.VariantFile(vcf_path)

    # Handle multi-sample VCFs
    n_samples = len(vcf.header.samples)
    if n_samples > 1 and sample_name is None:
        raise ValueError(
            f"VCF has {n_samples} samples but no sample_name specified. "
            f"Available: {list(vcf.header.samples)}"
        )

    for record in vcf.fetch(contig=contig_id) if contig_id else vcf.fetch():
        # Filter check
        if record.filter.keys() and "PASS" not in record.filter.keys():
            continue

        # SNP only
        if len(record.ref) != 1:
            continue

        alts = record.alts
        if alts is None or len(alts) == 0:
            continue

        # Biallelic check
        if config.require_biallelic and len(alts) > 1:
            continue

        alt = alts[0]
        if len(alt) != 1:
            continue

        # Get sample
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

        if site_depth is None or site_depth < config.min_depth_site:
            continue

        # Extract AF
        site_af = None
        if "AF" in record.info:
            site_af = record.info["AF"]
            if isinstance(site_af, tuple):
                site_af = site_af[0]
        elif sample is not None and "AD" in sample:
            ad = sample["AD"]
            if ad and len(ad) >= 2 and sum(ad) > 0:
                site_af = ad[1] / sum(ad)

        # AF filter
        if site_af is not None:
            if not (config.af_range[0] <= site_af <= config.af_range[1]):
                continue
        elif not config.skip_af_filter_if_missing:
            continue

        pos = record.pos
        snv_pos.append(pos)
        ref_alleles[pos] = record.ref
        depth[pos] = site_depth
        af[pos] = site_af

    vcf.close()
    return snv_pos, ref_alleles, depth, af


def make_windows_lazy(
    bam_path: str,
    contig_id: str,
    contig_length: int,
    snv_positions: list[int],
    ref_alleles: dict[int, str],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    sample_id: str | None = None,
) -> list[Window]:
    """
    Create overlapping windows with lazy per-window read loading.

    Windows overlap by 50% (step = window_size / 2) to enable linking
    of haplotypes across window boundaries via shared SNVs.

    This is O(W * reads_per_window) instead of O(W * total_reads),
    and uses O(window) memory instead of O(contig).
    """
    if not HAS_PYSAM:
        raise ImportError("pysam required for BAM parsing")

    snv_pos_sorted = sorted([p for p in snv_positions if 0 < p <= contig_length])
    if not snv_pos_sorted:
        return []

    windows = []
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

        # Collect SNVs in this window
        window_snvs = [p for p in snv_pos_sorted if start <= p < end]

        if len(window_snvs) < config.min_snvs_per_window:
            continue

        # Lazy load reads for this window only using pysam.fetch
        snv_set = set(window_snvs)
        reads = []

        # pysam fetch uses 0-based coordinates
        for aln in bam.fetch(contig_id, start - 1, end - 1):
            if aln.is_secondary or aln.is_supplementary or aln.is_unmapped:
                continue
            if aln.mapping_quality < config.min_mapq:
                continue

            # Parse alleles at SNV sites
            r = Read(
                id=aln.query_name, contig=contig_id, mapq=aln.mapping_quality, sample=sample_id
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

            # Extract alleles at SNV positions
            has_overlap = False
            for query_pos, ref_pos in aln.get_aligned_pairs(with_seq=False):
                if query_pos is None or ref_pos is None:
                    continue

                ref_pos_1based = ref_pos + 1
                if ref_pos_1based not in snv_set:
                    continue

                base = query_seq[query_pos]
                qual = query_qual[query_pos] if query_qual else config.default_base_quality

                if qual >= config.min_base_quality:
                    r.alleles[ref_pos_1based] = base
                    r.quals[ref_pos_1based] = qual
                    has_overlap = True

            if has_overlap:
                reads.append(r)

        # Subsample if needed (reproducible)
        if config.max_reads_per_window and len(reads) > config.max_reads_per_window:
            indices = rng.permutation(len(reads))[: config.max_reads_per_window]
            reads = [reads[i] for i in indices]

        if len(reads) < config.min_reads_per_window:
            continue

        w = Window(contig=contig_id, start=start, end=end, sample=sample_id, window_idx=window_idx)
        w.snv_pos = window_snvs
        w.ref_alleles = {p: ref_alleles[p] for p in window_snvs}
        w.reads = reads
        windows.append(w)
        window_idx += 1

    bam.close()
    return windows


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

                # Count mismatches with early exit.
                # We stop once mismatches exceed the allowed fraction.
                max_allowed = int(self.config.max_mismatch_frac * n_shared)
                mismatches = 0
                exceeded = False

                r_i, r_j = reads[i], reads[j]
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
        # Louvain community detection for read clustering.
        partition = community_louvain.best_partition(graph, weight="weight")

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

        # Use global log probability cache
        self._cache = _LOG_PROB_CACHE

    def _compute_log_prob_read_hap(self, read: Read, haplotype: Haplotype) -> float | None:
        """Compute log P(read | haplotype) using cached base probs."""
        log_prob = 0.0
        overlap = 0

        for pos, read_base in read.alleles.items():
            if pos in haplotype.consensus:
                q = read.quals.get(pos, self.config.default_base_quality)
                log_prob += self._cache.log_prob_base(haplotype.consensus[pos], read_base, q)
                overlap += 1

        return log_prob if overlap > 0 else None

    def _compute_log_prob_read_junk(self, read: Read) -> float:
        """Compute log P(read | junk) using divergent reference model."""
        p_div = self.config.junk_divergence_rate
        log_match = np.log(1.0 - p_div + 1e-12)
        log_miss = np.log(p_div / 3.0 + 1e-12)

        log_prob = 0.0
        for pos, read_base in read.alleles.items():
            if pos not in self.window.snv_pos:
                continue
            ref_base = self.window.ref_alleles.get(pos)
            if ref_base is None:
                continue
            if read_base == ref_base:
                log_prob += log_match
            else:
                log_prob += log_miss

        return log_prob

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

        for iteration in range(self.config.em_max_iter):
            # E-STEP prep: cache log P(read | haplotype) and log P(read | junk)
            # so we do not recompute them in multiple places.
            logl_hap = np.full((n_reads, n_haps), -np.inf)
            logl_junk = np.zeros(n_reads)

            for i, read in enumerate(reads):
                for k in range(n_haps):
                    lp = self._compute_log_prob_read_hap(read, haplotypes[k])
                    if lp is not None:
                        logl_hap[i, k] = lp
                logl_junk[i] = self._compute_log_prob_read_junk(read)

            # E-STEP: compute responsibilities gamma[i, k] = P(haplotype k | read i).
            for i in range(n_reads):
                logp_k = np.full(k_eff, -np.inf)

                for k in range(n_haps):
                    if logl_hap[i, k] > -np.inf:
                        logp_k[k] = np.log(pi[k] + 1e-12) + logl_hap[i, k]

                logp_k[junk_idx] = np.log(pi[junk_idx] + 1e-12) + logl_junk[i]

                log_sum = logsumexp(logp_k)
                if np.isneginf(log_sum):
                    gamma[i, :] = 0.0
                    gamma[i, junk_idx] = 1.0
                else:
                    gamma[i, :] = np.exp(logp_k - log_sum)

            # Log-likelihood: sum over reads of log(sum_k pi_k * P(read | k)).
            log_like = 0.0
            for i in range(n_reads):
                terms = []
                for k in range(n_haps):
                    if logl_hap[i, k] > -np.inf:
                        terms.append(np.log(pi[k] + 1e-12) + logl_hap[i, k])
                terms.append(np.log(pi[junk_idx] + 1e-12) + logl_junk[i])
                if terms:
                    log_like += logsumexp(np.array(terms))

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

                allele_votes = defaultdict(lambda: defaultdict(float))

                for i, read in enumerate(reads):
                    w = gamma[i, k]
                    if w < self.config.min_gamma_for_vote:
                        continue

                    for pos, base in read.alleles.items():
                        if pos not in self.window.snv_pos:
                            continue
                        q = read.quals.get(pos, self.config.default_base_quality)
                        q_weight = 1.0 - 10 ** (-q / 10.0)
                        allele_votes[pos][base] += w * q_weight

                if not allele_votes:
                    continue

                new_consensus = {}
                for pos in self.window.snv_pos:
                    if pos in allele_votes:
                        new_consensus[pos] = max(allele_votes[pos], key=allele_votes[pos].get)

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
            row_sums[row_sums == 0] = 1.0
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
                for read in window.reads:
                    if diff_pos in read.alleles:
                        total_at_pos += 1
                        if read.alleles[diff_pos] == minor_base:
                            minor_count += 1

                if total_at_pos == 0:
                    return True

                p_error = 10 ** (-30 / 10.0) / 3.0
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
        """Hard assignment of reads."""
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
                    reason=f"no_junk_reads",
                )
            )
            return window_result

        # Avoid duplicating an already-present haplotype in this window.
        existing_consensuses = [h.consensus for h in haplotypes]

        rescued_any = False
        new_haplotypes = []

        # Matching thresholds for anchor comparisons.
        max_distance = self.config.rescue_match_distance  # default 0.0005 = 99.95% match required
        min_shared = self.config.min_shared_for_rescue

        for anchor_idx, anchor in enumerate(anchor_haps):
            # Check if this anchor is already present as a haplotype
            anchor_already_present = False
            for existing in existing_consensuses:
                # Compare anchor vs existing consensus on shared SNV positions.
                n_shared = 0
                n_match = 0
                for pos in window.snv_pos:
                    if pos in anchor.consensus and pos in existing:
                        n_shared += 1
                        if anchor.consensus[pos] == existing[pos]:
                            n_match += 1
                if n_shared >= min_shared:
                    distance = 1.0 - (n_match / n_shared) if n_shared > 0 else 1.0
                    if distance <= max_distance:  # Already have this haplotype
                        anchor_already_present = True
                        break

            if anchor_already_present:
                continue

            # Count junk reads that are consistent with this anchor.
            n_matching_junk = 0
            matching_read_indices = []
            matching_read_info = []  # Store (read_idx, n_agree, n_disagree, n_total) for rescued reads
            min_shared_for_read = 2  # Lower threshold for individual reads

            for i, read in enumerate(reads):
                if not junk_read_mask[i]:
                    continue

                # Check if this read matches the anchor within error tolerance.
                n_shared = 0
                n_match = 0
                for pos, allele in read.alleles.items():
                    if pos in anchor.consensus:
                        n_shared += 1
                        if anchor.consensus[pos] == allele:
                            n_match += 1

                # Require sufficient shared positions and distance within error tolerance
                if n_shared >= min_shared_for_read:
                    distance = 1.0 - (n_match / n_shared)
                    if distance <= max_distance:  # Within sequencing error tolerance
                        n_matching_junk += 1
                        matching_read_indices.append(i)
                        # Store SNP agreement info: (read_idx, n_agree, n_disagree, n_total)
                        matching_read_info.append((i, n_match, n_shared - n_match, n_shared))

            # If any junk reads match this anchor near-exactly, create a rescued haplotype.
            # The strict rescue_match_distance (0.05%) ensures only truly matching reads pass.
            if n_matching_junk >= 1:
                # Create new haplotype from anchor consensus (restricted to this window's SNVs).
                new_consensus = {
                    pos: anchor.consensus[pos]
                    for pos in window.snv_pos
                    if pos in anchor.consensus
                }

                if len(new_consensus) >= self.config.min_snvs_per_window:
                    # Estimate weight from junk reads; enforce rescued_min_weight.
                    rescued_weight = max(
                        n_matching_junk / len(reads),
                        self.config.rescued_min_weight
                    )

                    new_hap = Haplotype(
                        consensus=new_consensus,
                        weight=rescued_weight,
                        supporting_reads=n_matching_junk,
                        confidence=0.8,  # Mark as rescued
                        track_id=None,  # Will be assigned during linking
                    )
                    new_haplotypes.append(new_hap)

                    donor_timepoint = anchor_samples[anchor_idx]
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

                    # Track each rescued read with SNP agreement/disagreement info
                    for read_idx, n_agree, n_disagree, n_total in matching_read_info:
                        read = reads[read_idx]
                        read_name = getattr(read, 'name', f"read_{read_idx}")
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

        # Redistribute weight: take from junk, give to rescued
        total_rescued_weight = sum(h.weight for h in new_haplotypes)
        old_junk_weight = pi[junk_idx]
        new_junk_weight = max(0.01, old_junk_weight - total_rescued_weight)

        # Build new pi
        pi_new = np.zeros(k_eff_new)
        # Scale down existing haplotype weights proportionally
        scale = (1.0 - total_rescued_weight - new_junk_weight) / (1.0 - old_junk_weight) if old_junk_weight < 1.0 else 1.0
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

        return WindowResult(
            window=window,
            haplotypes=haplotypes,
            gamma=gamma_new,
            pi=pi_new,
            log_likelihood=window_result.log_likelihood,
            assignments=assignments,
            converged=window_result.converged,
            iterations=window_result.iterations,
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
        cache = _LOG_PROB_CACHE

        # Junk model constants
        p_div = self.config.junk_divergence_rate
        log_junk_match = np.log(1.0 - p_div + 1e-12)
        log_junk_miss = np.log(p_div / 3.0 + 1e-12)

        for i, read in enumerate(reads):
            logp_k = np.full(k_eff, -np.inf)

            # Haplotype likelihoods
            for k in range(n_haps):
                log_prob = 0.0
                overlap = 0
                for pos, read_base in read.alleles.items():
                    if pos in haplotypes[k].consensus:
                        q = read.quals.get(pos, self.config.default_base_quality)
                        log_prob += cache.log_prob_base(haplotypes[k].consensus[pos], read_base, q)
                        overlap += 1
                if overlap > 0:
                    logp_k[k] = np.log(pi[k] + 1e-12) + log_prob

            # Junk likelihood
            log_junk = 0.0
            for pos, read_base in read.alleles.items():
                if pos in window.snv_pos:
                    ref_base = window.ref_alleles.get(pos)
                    if ref_base:
                        if read_base == ref_base:
                            log_junk += log_junk_match
                        else:
                            log_junk += log_junk_miss

            logp_k[junk_idx] = np.log(pi[junk_idx] + 1e-12) + log_junk

            log_sum = logsumexp(logp_k)
            if np.isneginf(log_sum):
                gamma[i, junk_idx] = 1.0
            else:
                gamma[i, :] = np.exp(logp_k - log_sum)

        return gamma

    def rescue_low_abundance(
        self, results_by_timepoint: dict[str, list[WindowResult]]
    ) -> dict[str, list[WindowResult]]:
        """Rescue low-abundance haplotypes across timepoints."""
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

        for window_key, sample_results in windows_by_position.items():
            if len(sample_results) >= 2:
                n_windows_with_multiple_timepoints += 1
            for sample_id, wr in sample_results.items():
                n_reads = wr.gamma.shape[0]
                junk_idx = wr.gamma.shape[1] - 1
                junk_reads = (wr.gamma[:, junk_idx] > 0.5).sum()
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

        return WindowResult(
            window=window,
            haplotypes=[],
            gamma=gamma,
            pi=pi,
            log_likelihood=-np.inf,
            assignments=assignments,
            converged=True,
            iterations=0,
        )

    # 2) EM haplotyping: refine haplotype consensus and weights.
    em = EMHaplotyper(window, initial_haps, cluster_sizes, config)
    haplotypes, gamma, pi, log_lik, converged, iterations = em.run()

    if not haplotypes:
        # EM pruned all haplotypes; keep the (junk) assignments.
        assignments = post.assign_reads(window.reads, gamma, pi)
        return WindowResult(
            window=window,
            haplotypes=[],
            gamma=gamma,
            pi=pi,
            log_likelihood=log_lik,
            assignments=assignments,
            converged=converged,
            iterations=iterations,
        )

    # 3) Post-processing: merge near-duplicate haplotypes with 1-SNP guard.
    merged_haps, final_gamma, final_pi = post.merge_similar_haplotypes(
        haplotypes, gamma, pi, window, n_timepoints_seen
    )
    assignments = post.assign_reads(window.reads, final_gamma, final_pi)

    result = WindowResult(
        window=window,
        haplotypes=merged_haps,
        gamma=final_gamma,
        pi=final_pi,
        log_likelihood=log_lik,
        assignments=assignments,
        converged=converged,
        iterations=iterations,
    )

    # 4) Optional validation checks on the WindowResult structure.
    if config.validate_results:
        result.validate()

    return result


def link_windows(
    results: list[WindowResult], config: HaplotyperConfig = DEFAULT_CONFIG
) -> list[WindowResult]:
    """
    Link haplotypes across overlapping windows based on consensus similarity.

    Since windows overlap by 50%, adjacent windows share SNV positions.
    Haplotypes are linked (assigned the same track_id) if their consensus
    agrees on the shared SNVs.

    This modifies haplotypes in-place by setting their track_id field.
    """
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

            # Evaluate candidate pairings before linking (avoid cross-links).
            candidates: list[tuple[int, int, float, int]] = []
            for hi, hap_i in enumerate(curr_wr.haplotypes):
                for hj, hap_j in enumerate(next_wr.haplotypes):
                    dist, _, n_shared = hap_i.distance_to(hap_j, shared_snvs)
                    # Only consider pairs with real shared calls
                    if n_shared < config.min_shared_snvs_for_link:
                        continue
                    if dist <= config.max_link_distance:
                        candidates.append((hi, hj, dist, n_shared))

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
) -> list[WindowResult]:
    """
    Process all windows in a contig and link haplotypes across windows.

    Windows overlap by 50% to enable linking haplotypes based on
    consensus similarity in shared SNV positions.
    """
    # 1) Load SNVs for this contig from the VCF.
    snv_pos, ref_alleles, depth, af = load_snvs_from_clair3(
        vcf_path, contig_id, vcf_sample_name, config
    )

    if not snv_pos:
        logging.warning(f"No SNVs found for contig {contig_id}")
        return []

    # 2) Create overlapping windows with lazy read loading.
    windows = make_windows_lazy(
        bam_path, contig_id, contig_length, snv_pos, ref_alleles, config, sample_id
    )

    if not windows:
        logging.warning(f"No valid windows for contig {contig_id}")
        return []

    # 3) Process windows (parallel if n_workers > 1).
    n_workers = config.n_workers
    if n_workers > 1 and len(windows) > 1:
        # Parallel processing using multiprocessing Pool
        n_workers = min(n_workers, len(windows))
        logging.info(f"Processing {len(windows)} windows with {n_workers} workers")

        # Use partial to bind config to process_window
        process_func = partial(_process_window_wrapper, config=config)

        with Pool(n_workers) as pool:
            results = pool.map(process_func, windows)
    else:
        # Sequential processing
        results = []
        for window in windows:
            result = process_window(window, config)
            results.append(result)

    # 4) Link haplotypes across overlapping windows into tracks.
    results = link_windows(results, config)

    return results


def _process_window_wrapper(window: Window, config: HaplotyperConfig) -> WindowResult:
    """Wrapper for process_window that can be pickled for multiprocessing."""
    return process_window(window, config)


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
            # Compute span and aggregate stats
            span_start = min(wr.window.start for wr, _, _ in members)
            span_end = max(wr.window.end for wr, _, _ in members)
            n_windows = len(members)

            # Merge consensus across all windows (weighted voting)
            position_votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            total_weight = 0.0
            total_reads = 0
            confidences = []

            for _wr, _k, hap in members:
                total_weight += hap.weight
                total_reads += hap.supporting_reads
                confidences.append(hap.confidence)

                for pos, base in hap.consensus.items():
                    position_votes[pos][base] += hap.weight

            # Build merged consensus from votes
            merged_consensus = {}
            for pos, votes in position_votes.items():
                best_base = max(votes.keys(), key=lambda b: votes[b])
                merged_consensus[pos] = best_base

            # Get sample from first window (all should be same)
            sample = members[0][0].window.sample

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
                    "mean_weight": total_weight / n_windows if n_windows > 0 else 0.0,
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
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--window-size", type=int, default=3000)
    parser.add_argument("--max-reads", type=int, default=300)
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
