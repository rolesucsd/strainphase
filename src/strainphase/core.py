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

import numpy as np
import networkx as nx
from scipy.special import logsumexp
from scipy.stats import binom
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Union, Any
import logging
import warnings
from functools import lru_cache

# Optional imports
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    logging.warning("python-louvain not installed; falling back to connected components")

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
    _warned: Set[str] = set()
    
    @classmethod
    def warn_once(cls, key: str, message: str):
        if key not in cls._warned:
            warnings.warn(message)
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
    min_reads_per_window: int = 10
    
    # =========== READ FILTERING ===========
    min_mapq: int = 20
    min_base_quality: int = 10
    default_base_quality: int = 20
    max_reads_per_window: int = 100
    
    # =========== SNV FILTERING (Clair3) ===========
    min_depth_site: int = 10
    af_range: Tuple[float, float] = (0.05, 0.95)
    require_biallelic: bool = True
    skip_af_filter_if_missing: bool = True
    
    # =========== GRAPH CONSTRUCTION ===========
    min_shared_snvs_for_edge: int = 3
    max_mismatch_frac: float = 0.02
    min_reads_per_cluster: int = 3
    
    # =========== EM PARAMETERS ===========
    em_max_iter: int = 20
    em_tolerance: float = 1e-4
    dirichlet_alpha: float = 1.0
    min_hap_eff_weight: float = 3.0
    min_gamma_for_vote: float = 0.01
    use_cluster_pi_init: bool = True
    
    # =========== JUNK MODEL ===========
    junk_divergence_rate: float = 0.10
    
    # =========== POST-PROCESSING ===========
    merge_distance_threshold: float = 0.01
    min_shared_for_merge: int = 3  # Min shared SNVs with actual calls to consider merging
    assign_confidence_threshold: float = 0.90
    
    # =========== 1-SNP VALIDATION ===========
    validate_1snp_differences: bool = True
    min_minor_frequency_1snp: float = 0.10
    min_minor_supporting_reads_1snp: int = 3
    min_timepoints_for_1snp: int = 2
    use_binomial_test_1snp: bool = True
    binomial_alpha: float = 0.05
    
    # =========== LONGITUDINAL PARAMETERS ===========
    min_weight_for_anchor: float = 0.20
    rescue_match_distance: float = 0.01
    min_shared_for_rescue: int = 3  # Min shared SNVs with actual calls for rescue matching
    rescued_min_weight: float = 0.02
    
    # =========== LINEAGE CLUSTERING PARAMETERS ===========
    # Controls how tracks are clustered into lineages across samples
    lineage_merge_distance: float = 0.02  # Max distance to merge tracks into same lineage
    min_shared_for_lineage: int = 3  # Min shared SNVs to consider merging into lineage
    max_span_gap_for_lineage: int = 10000  # Max gap between track spans to consider same locus
    
    # =========== WINDOW LINKING PARAMETERS ===========
    # Haplotypes in adjacent overlapping windows are linked if their
    # consensus agrees on shared SNVs (Hamming distance <= max_link_distance)
    max_link_distance: float = 0.02  # Max mismatch fraction to link
    min_shared_snvs_for_link: int = 3  # Min shared SNVs with ACTUAL CALLS to link (not just window overlap)
    
    # =========== RUNTIME PARAMETERS ===========
    random_seed: Optional[int] = None
    validate_results: bool = False  # Set False for production runs
    
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
    alleles: Dict[int, str] = field(default_factory=dict)
    quals: Dict[int, int] = field(default_factory=dict)
    sample: Optional[str] = None


@dataclass 
class Window:
    """Represents a genomic window with associated SNVs and reads."""
    contig: str
    start: int  # 1-based, inclusive
    end: int    # 1-based, exclusive
    snv_pos: List[int] = field(default_factory=list)
    ref_alleles: Dict[int, str] = field(default_factory=dict)
    reads: List[Read] = field(default_factory=list)
    sample: Optional[str] = None
    window_idx: int = 0  # Position in contig's window sequence
    
    # Cached position sets for graph building (optimization)
    _pos_sets: Optional[List[Set[int]]] = field(default=None, repr=False)
    
    def get_read_position_sets(self) -> List[Set[int]]:
        """Get precomputed position sets for each read (cached)."""
        if self._pos_sets is None:
            self._pos_sets = [
                {p for p in r.alleles if self.start <= p < self.end}
                for r in self.reads
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
    consensus: Dict[int, str]
    weight: float = 0.0
    supporting_reads: int = 0
    confidence: float = 0.0
    track_id: Optional[str] = None  # Assigned after window linking
    
    def distance_to(self, other: 'Haplotype', positions: List[int], 
                    max_mismatches: Optional[int] = None) -> Tuple[float, int, int]:
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
    
    def get_differing_positions(self, other: 'Haplotype', positions: List[int]) -> List[int]:
        """Return list of positions where haplotypes differ."""
        return [
            pos for pos in positions
            if (b1 := self.consensus.get(pos)) is not None
            and (b2 := other.consensus.get(pos)) is not None
            and b1 != b2
        ]


@dataclass
class WindowResult:
    """Complete results from processing a single window."""
    window: Window
    haplotypes: List[Haplotype]
    gamma: np.ndarray
    pi: np.ndarray
    log_likelihood: float
    assignments: List[Dict]
    converged: bool
    iterations: int
    
    def validate(self) -> bool:
        """Validate internal consistency."""
        N = len(self.window.reads)
        K = len(self.haplotypes)
        K_eff = K + 1
        
        assert self.gamma.shape == (N, K_eff), \
            f"gamma shape {self.gamma.shape} != expected ({N}, {K_eff})"
        
        row_sums = self.gamma.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), \
            f"gamma rows don't sum to 1: min={row_sums.min()}, max={row_sums.max()}"
        
        assert np.isclose(self.pi.sum(), 1.0, atol=1e-6), \
            f"pi doesn't sum to 1: {self.pi.sum()}"
        
        assert len(self.pi) == K_eff, \
            f"pi length {len(self.pi)} != K_eff {K_eff}"
        
        return True


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
    
    def log_prob_base(self, hap_base: str, read_base: str, Q: int) -> float:
        """Get log probability from cache."""
        Q = min(Q, len(self._log_match) - 1)
        if hap_base == read_base:
            return self._log_match[Q]
        return self._log_mismatch[Q]


# Global cache instance
_LOG_PROB_CACHE = LogProbCache()


# =============================================================================
# I/O FUNCTIONS - LAZY LOADING
# =============================================================================

def load_snvs_from_clair3(
    vcf_path: str,
    contig_id: Optional[str] = None,
    sample_name: Optional[str] = None,
    config: HaplotyperConfig = DEFAULT_CONFIG
) -> Tuple[List[int], Dict[int, str], Dict[int, int], Dict[int, Optional[float]]]:
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
        if record.filter.keys() and 'PASS' not in record.filter.keys():
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
        if 'DP' in record.info:
            site_depth = record.info['DP']
        elif sample is not None and 'DP' in sample:
            site_depth = sample['DP']
        
        if site_depth is None or site_depth < config.min_depth_site:
            continue
        
        # Extract AF
        site_af = None
        if 'AF' in record.info:
            site_af = record.info['AF']
            if isinstance(site_af, tuple):
                site_af = site_af[0]
        elif sample is not None and 'AD' in sample:
            ad = sample['AD']
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
    snv_positions: List[int],
    ref_alleles: Dict[int, str],
    config: HaplotyperConfig = DEFAULT_CONFIG,
    sample_id: Optional[str] = None
) -> List[Window]:
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
        
        # Skip very short final windows
        if end - start < step_size:
            continue
        
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
                id=aln.query_name,
                contig=contig_id,
                mapq=aln.mapping_quality,
                sample=sample_id
            )
            
            query_seq = aln.query_sequence
            query_qual = aln.query_qualities
            
            if query_seq is None:
                continue
            
            # Handle missing quality (warn once)
            if query_qual is None:
                WarningThrottler.warn_once(
                    "no_qual",
                    f"Some reads lack quality scores. Using default Q{config.default_base_quality}."
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
            indices = rng.permutation(len(reads))[:config.max_reads_per_window]
            reads = [reads[i] for i in indices]
        
        if len(reads) < config.min_reads_per_window:
            continue
        
        w = Window(
            contig=contig_id,
            start=start,
            end=end,
            sample=sample_id,
            window_idx=window_idx
        )
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
        G = nx.Graph()
        reads = window.reads
        n_reads = len(reads)
        
        for i in range(n_reads):
            G.add_node(i)
        
        # OPTIMIZATION: Use precomputed position sets
        pos_sets = window.get_read_position_sets()
        
        # Precompute max allowed mismatches for early exit
        for i in range(n_reads):
            S_i = pos_sets[i]
            if not S_i:
                continue
            
            for j in range(i + 1, n_reads):
                S_j = pos_sets[j]
                shared = S_i & S_j
                n_shared = len(shared)
                
                if n_shared < self.config.min_shared_snvs_for_edge:
                    continue
                
                # OPTIMIZATION: Early exit mismatch counting
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
                
                if not exceeded:
                    mismatch_frac = mismatches / n_shared
                    weight = (1.0 - mismatch_frac) * n_shared
                    G.add_edge(i, j, weight=weight)
        
        return G
    
    def derive_consensus(self, cluster_reads: List[Read], window: Window) -> Dict[int, str]:
        """Derive consensus from cluster reads."""
        allele_counts = defaultdict(lambda: defaultdict(int))
        
        for r in cluster_reads:
            for pos, base in r.alleles.items():
                if window.start <= pos < window.end and pos in window.snv_pos:
                    allele_counts[pos][base] += 1
        
        consensus = {}
        for pos in window.snv_pos:
            if pos in allele_counts:
                consensus[pos] = max(allele_counts[pos], key=allele_counts[pos].get)
        
        return consensus
    
    def get_initial_haplotypes(self, window: Window) -> Tuple[List[Haplotype], List[int]]:
        """Initialize haplotypes using graph clustering."""
        G = self.build_overlap_graph(window)
        
        if G.number_of_edges() == 0:
            consensus = self.derive_consensus(window.reads, window)
            if consensus:
                return [Haplotype(consensus=consensus, supporting_reads=len(window.reads))], [len(window.reads)]
            return [], []
        
        # Partition
        if HAS_LOUVAIN:
            partition = community_louvain.best_partition(G, weight='weight')
        else:
            partition = {}
            for idx, component in enumerate(nx.connected_components(G)):
                for node in component:
                    partition[node] = idx
        
        # Group by cluster
        clusters = defaultdict(list)
        for node_idx, cluster_id in partition.items():
            clusters[cluster_id].append(window.reads[node_idx])
        
        # Build haplotypes
        initial_haps = []
        cluster_sizes = []
        
        for cluster_id, cluster_reads in clusters.items():
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
        initial_haplotypes: List[Haplotype],
        cluster_sizes: Optional[List[int]] = None,
        config: HaplotyperConfig = DEFAULT_CONFIG
    ):
        self.window = window
        self.haplotypes = initial_haplotypes
        self.cluster_sizes = cluster_sizes
        self.reads = window.reads
        self.config = config
        
        # Use global log probability cache
        self._cache = _LOG_PROB_CACHE
    
    def _compute_log_prob_read_hap(self, read: Read, haplotype: Haplotype) -> Optional[float]:
        """Compute log P(read | haplotype) using cached base probs."""
        log_prob = 0.0
        overlap = 0
        
        for pos, read_base in read.alleles.items():
            if pos in haplotype.consensus:
                Q = read.quals.get(pos, self.config.default_base_quality)
                log_prob += self._cache.log_prob_base(haplotype.consensus[pos], read_base, Q)
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
    
    def run(self) -> Tuple[List[Haplotype], np.ndarray, np.ndarray, float, bool, int]:
        """Run EM with cached log-probability computations."""
        H = self.haplotypes
        reads = self.reads
        N = len(reads)
        K = len(H)
        
        if K == 0:
            gamma = np.ones((N, 1))
            pi = np.array([1.0])
            return [], gamma, pi, -np.inf, True, 0
        
        K_eff = K + 1
        junk_idx = K
        
        # Initialize pi
        if self.config.use_cluster_pi_init and self.cluster_sizes:
            cluster_total = sum(self.cluster_sizes)
            junk_init = max(1, N - cluster_total)
            pi = np.array(self.cluster_sizes + [junk_init], dtype=float)
            pi /= pi.sum()
        else:
            pi = np.ones(K_eff) / K_eff
        
        gamma = np.zeros((N, K_eff))
        prev_log_like = -np.inf
        converged = False
        
        for iteration in range(self.config.em_max_iter):
            # OPTIMIZATION: Cache all log-likelihoods once per iteration
            logL_hap = np.full((N, K), -np.inf)
            logL_junk = np.zeros(N)
            
            for i, read in enumerate(reads):
                for k in range(K):
                    lp = self._compute_log_prob_read_hap(read, H[k])
                    if lp is not None:
                        logL_hap[i, k] = lp
                logL_junk[i] = self._compute_log_prob_read_junk(read)
            
            # E-STEP using cached values
            for i in range(N):
                logp_k = np.full(K_eff, -np.inf)
                
                for k in range(K):
                    if logL_hap[i, k] > -np.inf:
                        logp_k[k] = np.log(pi[k] + 1e-12) + logL_hap[i, k]
                
                logp_k[junk_idx] = np.log(pi[junk_idx] + 1e-12) + logL_junk[i]
                
                log_sum = logsumexp(logp_k)
                if np.isneginf(log_sum):
                    gamma[i, :] = 0.0
                    gamma[i, junk_idx] = 1.0
                else:
                    gamma[i, :] = np.exp(logp_k - log_sum)
            
            # LOG-LIKELIHOOD using cached values (no recomputation!)
            log_like = 0.0
            for i in range(N):
                terms = []
                for k in range(K):
                    if logL_hap[i, k] > -np.inf:
                        terms.append(np.log(pi[k] + 1e-12) + logL_hap[i, k])
                terms.append(np.log(pi[junk_idx] + 1e-12) + logL_junk[i])
                if terms:
                    log_like += logsumexp(np.array(terms))
            
            # M-STEP
            Nk = gamma.sum(axis=0) + (self.config.dirichlet_alpha - 1.0)
            pi = Nk / Nk.sum()
            
            # Update haplotypes
            new_H = []
            surviving_indices = []
            
            for k in range(K):
                if Nk[k] < self.config.min_hap_eff_weight:
                    continue
                
                allele_votes = defaultdict(lambda: defaultdict(float))
                
                for i, read in enumerate(reads):
                    w = gamma[i, k]
                    if w < self.config.min_gamma_for_vote:
                        continue
                    
                    for pos, base in read.alleles.items():
                        if pos not in self.window.snv_pos:
                            continue
                        Q = read.quals.get(pos, self.config.default_base_quality)
                        q_weight = 1.0 - 10 ** (-Q / 10.0)
                        allele_votes[pos][base] += w * q_weight
                
                if not allele_votes:
                    continue
                
                new_consensus = {}
                for pos in self.window.snv_pos:
                    if pos in allele_votes:
                        new_consensus[pos] = max(allele_votes[pos], key=allele_votes[pos].get)
                
                if new_consensus:
                    new_H.append(Haplotype(consensus=new_consensus))
                    surviving_indices.append(k)
            
            # Update structures
            H = new_H
            K = len(H)
            
            if K == 0:
                pi = np.array([1.0])
                gamma = np.ones((N, 1))
                return [], gamma, pi, log_like, True, iteration + 1
            
            # Rebuild pi and gamma
            junk_mass = Nk[-1]
            Nk_surv = Nk[surviving_indices]
            Nk_new = np.concatenate([Nk_surv, [junk_mass]])
            pi = Nk_new / Nk_new.sum()
            
            K_eff = K + 1
            junk_idx = K
            
            gamma_new = np.zeros((N, K_eff))
            for new_k, old_k in enumerate(surviving_indices):
                gamma_new[:, new_k] = gamma[:, old_k]
            gamma_new[:, junk_idx] = gamma[:, -1]
            gamma = gamma_new
            
            row_sums = gamma.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            gamma /= row_sums
            
            # Convergence check
            if abs(log_like - prev_log_like) < self.config.em_tolerance:
                converged = True
                break
            prev_log_like = log_like
        
        # Update haplotype metadata
        for k, hap in enumerate(H):
            hap.weight = pi[k]
            hap.supporting_reads = int(
                (gamma[:, k] >= self.config.assign_confidence_threshold).sum()
            )
            confident_mask = gamma[:, k] >= self.config.assign_confidence_threshold
            if confident_mask.sum() > 0:
                hap.confidence = float(gamma[confident_mask, k].mean())
        
        return H, gamma, pi, log_like, converged, iteration + 1


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
        n_timepoints_seen: int = 1
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
        minor_supporting = int(
            (gamma[:, minor_k] >= self.config.assign_confidence_threshold).sum()
        )
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
        haplotypes: List[Haplotype],
        gamma: np.ndarray,
        pi: np.ndarray,
        window: Window,
        n_timepoints_seen: int = 1
    ) -> Tuple[List[Haplotype], np.ndarray, np.ndarray]:
        """Merge similar haplotypes with optimized distance computation."""
        K = len(haplotypes)
        if K <= 1:
            return haplotypes, gamma, pi
        
        # Precompute max allowed mismatches for early exit
        max_mismatches = int(self.config.merge_distance_threshold * len(window.snv_pos)) + 1
        
        used = set()
        new_haplotypes = []
        old_to_new = [-1] * K
        
        for i in range(K):
            if i in used:
                continue
            
            group = [i]
            for j in range(i + 1, K):
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
                            haplotypes[i], haplotypes[j],
                            i, j, window, gamma, n_timepoints_seen
                        )
                        if not should_merge:
                            continue
                    group.append(j)
            
            used.update(group)
            
            # Merge consensus
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
        
        # Rebuild pi and gamma
        new_K = len(new_haplotypes)
        new_pi = np.zeros(new_K + 1)
        
        for old_k, new_k in enumerate(old_to_new):
            if new_k >= 0:
                new_pi[new_k] += pi[old_k]
        new_pi[-1] = pi[-1]
        new_pi /= new_pi.sum()
        
        new_gamma = np.zeros((gamma.shape[0], new_K + 1))
        for old_k, new_k in enumerate(old_to_new):
            if new_k >= 0:
                new_gamma[:, new_k] += gamma[:, old_k]
        new_gamma[:, -1] = gamma[:, -1]
        
        row_sums = new_gamma.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        new_gamma /= row_sums
        
        # Update haplotype metadata (FIX: recompute after merge)
        for k, hap in enumerate(new_haplotypes):
            hap.weight = new_pi[k]
            hap.supporting_reads = int(
                (new_gamma[:, k] >= self.config.assign_confidence_threshold).sum()
            )
            confident_mask = new_gamma[:, k] >= self.config.assign_confidence_threshold
            if confident_mask.sum() > 0:
                hap.confidence = float(new_gamma[confident_mask, k].mean())
        
        return new_haplotypes, new_gamma, new_pi
    
    def assign_reads(
        self,
        reads: List[Read],
        gamma: np.ndarray,
        pi: np.ndarray
    ) -> List[Dict]:
        """Hard assignment of reads."""
        assignments = []
        N, K_eff = gamma.shape
        junk_idx = K_eff - 1
        
        for i in range(N):
            probs = gamma[i, :]
            best_k = int(np.argmax(probs))
            best_prob = float(probs[best_k])
            
            is_junk = (best_k == junk_idx)
            
            if is_junk:
                hap_id = None
                is_ambiguous = False
            elif best_prob >= self.config.assign_confidence_threshold:
                hap_id = best_k
                is_ambiguous = False
            else:
                hap_id = None
                is_ambiguous = True
            
            assignments.append({
                'read_id': reads[i].id,
                'hap_id': hap_id,
                'prob': best_prob,
                'is_junk': is_junk,
                'is_ambiguous': is_ambiguous
            })
        
        return assignments


# =============================================================================
# OPTIMIZED LONGITUDINAL INTEGRATOR
# =============================================================================

class LongitudinalIntegrator:
    """Cross-timepoint integration with optimized anchor panel construction."""
    
    def __init__(self, config: HaplotyperConfig = DEFAULT_CONFIG):
        self.config = config
    
    def build_anchor_panel_for_key(
        self,
        sample_results: Dict[str, WindowResult]
    ) -> Tuple[List[Haplotype], List[str]]:
        """
        Build anchor panel directly from sample_results dict.
        
        OPTIMIZATION: Operates on pre-filtered results for this window key,
        not the full results dictionary.
        """
        anchor_haps = []
        anchor_samples = []
        
        for sample_id, wr in sample_results.items():
            for hap in wr.haplotypes:
                if hap.weight >= self.config.min_weight_for_anchor:
                    anchor_haps.append(hap)
                    anchor_samples.append(sample_id)
        
        return anchor_haps, anchor_samples
    
    def count_timepoints_for_haplotype(
        self,
        hap: Haplotype,
        sample_results: Dict[str, WindowResult],
        positions: List[int]
    ) -> int:
        """Count timepoints where this haplotype appears."""
        count = 0
        for sample_id, wr in sample_results.items():
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
        anchor_haps: List[Haplotype],
        sample_results: Dict[str, WindowResult]
    ) -> WindowResult:
        """Rescue low-confidence haplotypes using anchors."""
        if not anchor_haps:
            return window_result
        
        window = window_result.window
        haplotypes = window_result.haplotypes
        gamma = window_result.gamma.copy()
        pi = window_result.pi.copy()
        
        rescued_any = False
        
        for k, hap in enumerate(haplotypes):
            if hap.weight >= self.config.min_weight_for_anchor:
                continue
            
            # Check for anchor match - only consider anchors with sufficient shared positions
            best_dist = float('inf')
            best_n_shared = 0
            for anchor in anchor_haps:
                dist, _, n_shared = hap.distance_to(anchor, window.snv_pos)
                if n_shared >= self.config.min_shared_for_rescue and dist < best_dist:
                    best_dist = dist
                    best_n_shared = n_shared
            
            if best_n_shared >= self.config.min_shared_for_rescue and best_dist <= self.config.rescue_match_distance:
                old_weight = pi[k]
                new_weight = max(old_weight, self.config.rescued_min_weight)
                
                if new_weight > old_weight:
                    excess = new_weight - old_weight
                    pi[k] = new_weight
                    
                    junk_idx = len(haplotypes)
                    if pi[junk_idx] > excess:
                        pi[junk_idx] -= excess
                    
                    pi = pi / pi.sum()
                    hap.confidence = 1.0
                    rescued_any = True
        
        if rescued_any:
            pi = pi / pi.sum()
            
            for k, hap in enumerate(haplotypes):
                hap.weight = pi[k]
            
            # Recompute gamma with fixed pi
            gamma = self._recompute_gamma(window, haplotypes, pi)
            
            post = PostProcessor(self.config)
            assignments = post.assign_reads(window.reads, gamma, pi)
            
            for k, hap in enumerate(haplotypes):
                hap.supporting_reads = int(
                    (gamma[:, k] >= self.config.assign_confidence_threshold).sum()
                )
            
            return WindowResult(
                window=window,
                haplotypes=haplotypes,
                gamma=gamma,
                pi=pi,
                log_likelihood=window_result.log_likelihood,
                assignments=assignments,
                converged=window_result.converged,
                iterations=window_result.iterations
            )
        
        return window_result
    
    def _recompute_gamma(
        self,
        window: Window,
        haplotypes: List[Haplotype],
        pi: np.ndarray
    ) -> np.ndarray:
        """Recompute gamma with fixed pi (E-step only)."""
        reads = window.reads
        N = len(reads)
        K = len(haplotypes)
        K_eff = K + 1
        junk_idx = K
        
        gamma = np.zeros((N, K_eff))
        cache = _LOG_PROB_CACHE
        
        # Junk model constants
        p_div = self.config.junk_divergence_rate
        log_junk_match = np.log(1.0 - p_div + 1e-12)
        log_junk_miss = np.log(p_div / 3.0 + 1e-12)
        
        for i, read in enumerate(reads):
            logp_k = np.full(K_eff, -np.inf)
            
            # Haplotype likelihoods
            for k in range(K):
                log_prob = 0.0
                overlap = 0
                for pos, read_base in read.alleles.items():
                    if pos in haplotypes[k].consensus:
                        Q = read.quals.get(pos, self.config.default_base_quality)
                        log_prob += cache.log_prob_base(haplotypes[k].consensus[pos], read_base, Q)
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
        self,
        results_by_timepoint: Dict[str, List[WindowResult]]
    ) -> Dict[str, List[WindowResult]]:
        """Rescue low-abundance haplotypes across timepoints."""
        if len(results_by_timepoint) < 2:
            return results_by_timepoint
        
        # Group by window position
        windows_by_position: Dict[Tuple, Dict[str, WindowResult]] = defaultdict(dict)
        
        for sample_id, window_results in results_by_timepoint.items():
            for wr in window_results:
                key = (wr.window.contig, wr.window.start, wr.window.end)
                windows_by_position[key][sample_id] = wr
        
        # Process each position
        rescued_results: Dict[str, List[WindowResult]] = defaultdict(list)
        
        for window_key, sample_results in windows_by_position.items():
            # OPTIMIZATION: Build anchor panel from sample_results directly
            anchor_haps, anchor_samples = self.build_anchor_panel_for_key(sample_results)
            
            for sample_id, wr in sample_results.items():
                rescued_wr = self.rescue_window_result(wr, anchor_haps, sample_results)
                rescued_results[sample_id].append(rescued_wr)
        
        return dict(rescued_results)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_window(
    window: Window,
    config: HaplotyperConfig = DEFAULT_CONFIG,
    n_timepoints_seen: int = 1
) -> WindowResult:
    """Process a single window through the full pipeline."""
    post = PostProcessor(config)
    
    # Initialize
    initializer = GraphInitializer(config)
    initial_haps, cluster_sizes = initializer.get_initial_haplotypes(window)
    
    if not initial_haps:
        # FIX: Return proper junk-only result (gamma = ones, not zeros)
        N = len(window.reads)
        gamma = np.ones((N, 1))
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
            iterations=0
        )
    
    # EM
    em = EMHaplotyper(window, initial_haps, cluster_sizes, config)
    haplotypes, gamma, pi, log_lik, converged, iterations = em.run()
    
    if not haplotypes:
        assignments = post.assign_reads(window.reads, gamma, pi)
        return WindowResult(
            window=window,
            haplotypes=[],
            gamma=gamma,
            pi=pi,
            log_likelihood=log_lik,
            assignments=assignments,
            converged=converged,
            iterations=iterations
        )
    
    # Post-processing with 1-SNP validation
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
        iterations=iterations
    )
    
    # Optional validation
    if config.validate_results:
        result.validate()
    
    return result


def link_windows(
    results: List[WindowResult],
    config: HaplotyperConfig = DEFAULT_CONFIG
) -> List[WindowResult]:
    """
    Link haplotypes across overlapping windows based on consensus similarity.
    
    Since windows overlap by 50%, adjacent windows share SNV positions.
    Haplotypes are linked (assigned the same track_id) if their consensus
    agrees on the shared SNVs.
    
    This modifies haplotypes in-place by setting their track_id field.
    """
    if len(results) < 2:
        # Single window - each haplotype gets its own track
        track_counter = 0
        for wr in results:
            for hap in wr.haplotypes:
                track_counter += 1
                hap.track_id = f"T{track_counter:04d}"
        return results
    
    # Sort by window start position
    sorted_results = sorted(results, key=lambda wr: wr.window.start)
    
    # Build graph: nodes = (window_idx, hap_idx), edges = compatible haplotypes
    G = nx.Graph()  # Undirected for connected components
    
    # Add all nodes
    for i, wr in enumerate(sorted_results):
        for j in range(len(wr.haplotypes)):
            G.add_node((i, j))
    
    # Connect haplotypes in overlapping windows
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
            
            # Try to link each pair of haplotypes
            for hi, hap_i in enumerate(curr_wr.haplotypes):
                for hj, hap_j in enumerate(next_wr.haplotypes):
                    # Compute distance on shared positions
                    # n_shared is the count of positions where BOTH haplotypes have calls
                    dist, _, n_shared = hap_i.distance_to(hap_j, shared_snvs)
                    
                    # CRITICAL: Only link if haplotypes actually share called positions
                    # This prevents chaining unrelated haplotypes that happen to lack
                    # calls in the window overlap region
                    if n_shared >= config.min_shared_snvs_for_link and dist <= config.max_link_distance:
                        G.add_edge((i, hi), (k, hj))
    
    # Find connected components - each is a track
    components = list(nx.connected_components(G))
    
    # Assign track_ids
    for track_idx, component in enumerate(components):
        track_id = f"T{track_idx + 1:04d}"
        for (w_idx, h_idx) in component:
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
    sample_id: Optional[str] = None,
    vcf_sample_name: Optional[str] = None
) -> List[WindowResult]:
    """
    Process all windows in a contig and link haplotypes across windows.
    
    Windows overlap by 50% to enable linking haplotypes based on
    consensus similarity in shared SNV positions.
    """
    # Load SNVs
    snv_pos, ref_alleles, depth, af = load_snvs_from_clair3(
        vcf_path, contig_id, vcf_sample_name, config
    )
    
    if not snv_pos:
        logging.warning(f"No SNVs found for contig {contig_id}")
        return []
    
    # Create overlapping windows with lazy read loading
    windows = make_windows_lazy(
        bam_path, contig_id, contig_length, snv_pos, ref_alleles, config, sample_id
    )
    
    if not windows:
        logging.warning(f"No valid windows for contig {contig_id}")
        return []
    
    # Process each window
    results = []
    for window in windows:
        result = process_window(window, config)
        results.append(result)
    
    # Link haplotypes across overlapping windows
    results = link_windows(results, config)
    
    return results


def process_mag_longitudinal(
    samples: Dict[str, Tuple[str, str]],
    mag_contigs: Dict[str, int],
    config: HaplotyperConfig = DEFAULT_CONFIG
) -> Dict[str, Dict[str, List[WindowResult]]]:
    """
    Process MAG across timepoints with longitudinal rescue.
    
    Haplotypes are linked across windows after initial processing,
    after rescue, and after 1-SNP validation merge.
    """
    # First pass: process all samples (includes initial linking)
    all_results: Dict[str, Dict[str, List[WindowResult]]] = {}
    
    for sample_id, (bam_path, vcf_path) in samples.items():
        logging.info(f"Processing sample {sample_id}")
        all_results[sample_id] = {}
        
        for contig_id, contig_length in mag_contigs.items():
            results = process_contig(
                bam_path, vcf_path, contig_id, contig_length, config, sample_id
            )
            all_results[sample_id][contig_id] = results
    
    # Second pass: longitudinal rescue with proper timepoint counting
    integrator = LongitudinalIntegrator(config)
    
    for contig_id in mag_contigs.keys():
        # Collect results for this contig
        results_by_timepoint: Dict[str, List[WindowResult]] = {}
        for sample_id in samples.keys():
            if contig_id in all_results[sample_id]:
                results_by_timepoint[sample_id] = all_results[sample_id][contig_id]
        
        if len(results_by_timepoint) >= 2:
            # Apply rescue
            rescued = integrator.rescue_low_abundance(results_by_timepoint)
            
            for sample_id, window_results in rescued.items():
                # Re-link after rescue (may have new haplotypes)
                window_results = link_windows(window_results, config)
                all_results[sample_id][contig_id] = window_results
    
    # Third pass: re-run merge with correct n_timepoints for 1-SNP validation
    if config.validate_1snp_differences and len(samples) >= config.min_timepoints_for_1snp:
        for contig_id in mag_contigs.keys():
            # Group windows by position
            windows_by_pos: Dict[Tuple, Dict[str, WindowResult]] = defaultdict(dict)
            
            for sample_id in samples.keys():
                for wr in all_results[sample_id].get(contig_id, []):
                    key = (wr.window.start, wr.window.end)
                    windows_by_pos[key][sample_id] = wr
            
            # Count timepoints for each haplotype and re-merge if needed
            post = PostProcessor(config)
            
            for window_key, sample_wrs in windows_by_pos.items():
                n_timepoints = len(sample_wrs)
                
                for sample_id, wr in sample_wrs.items():
                    if len(wr.haplotypes) > 1:
                        # Re-run merge with correct timepoint count
                        merged_haps, final_gamma, final_pi = post.merge_similar_haplotypes(
                            wr.haplotypes, wr.gamma, wr.pi, wr.window, n_timepoints
                        )
                        
                        if len(merged_haps) != len(wr.haplotypes):
                            # Update result
                            assignments = post.assign_reads(wr.window.reads, final_gamma, final_pi)
                            new_wr = WindowResult(
                                window=wr.window,
                                haplotypes=merged_haps,
                                gamma=final_gamma,
                                pi=final_pi,
                                log_likelihood=wr.log_likelihood,
                                assignments=assignments,
                                converged=wr.converged,
                                iterations=wr.iterations
                            )
                            
                            # Find and replace in results
                            for i, old_wr in enumerate(all_results[sample_id][contig_id]):
                                if old_wr.window.start == wr.window.start:
                                    all_results[sample_id][contig_id][i] = new_wr
                                    break
        
        # Re-link after merge (haplotypes may have been merged)
        for sample_id in samples.keys():
            for contig_id in mag_contigs.keys():
                if contig_id in all_results[sample_id]:
                    all_results[sample_id][contig_id] = link_windows(
                        all_results[sample_id][contig_id], config
                    )
    
    return all_results


# =============================================================================
# RESULTS EXPORT
# =============================================================================

def results_to_dataframe(results: Dict[str, List[WindowResult]]) -> List[Dict]:
    """
    Convert results to track-based records for DataFrame.
    
    Groups haplotypes by track_id and computes span across all windows
    in each track. This produces one row per track, with span_start and
    span_end reflecting the full linked haplotype extent.
    """
    records = []
    
    for contig_id, window_results in results.items():
        # Group haplotypes by track_id
        tracks: Dict[str, List[Tuple[WindowResult, int, Haplotype]]] = defaultdict(list)
        
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
            position_votes: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
            total_weight = 0.0
            total_reads = 0
            confidences = []
            
            for wr, k, hap in members:
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
            
            records.append({
                'contig': contig_id,
                'sample': sample,
                'track_id': track_id,
                'span_start': span_start,
                'span_end': span_end,
                'span_bp': span_end - span_start,
                'n_windows': n_windows,
                'n_snvs': len(merged_consensus),
                'mean_weight': total_weight / n_windows if n_windows > 0 else 0.0,
                'total_supporting_reads': total_reads,
                'mean_confidence': np.mean(confidences) if confidences else 0.0,
                'consensus': '|'.join(
                    f"{pos}:{base}" 
                    for pos, base in sorted(merged_consensus.items())
                ),
            })
    
    # Sort by contig, then by span_start
    records.sort(key=lambda r: (r['contig'], r['span_start']))
    
    return records


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Haplotype reconstruction for PacBio HiFi metagenomics")
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
        validate_results=not args.no_validate
    )
    
    logging.basicConfig(level=logging.INFO)
    
    results = process_contig(
        args.bam, args.vcf, args.contig, args.length,
        config, args.sample, args.vcf_sample
    )
    
    records = results_to_dataframe({args.contig: results})
    
    if records:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(records)
        print(f"Wrote {len(records)} haplotypes to {args.output}")
    else:
        print("No haplotypes found")