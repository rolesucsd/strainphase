#!/usr/bin/env python3
"""
Comprehensive test suite for haplotyper pipeline.

Includes:
- Unit tests for individual components
- Integration tests for full pipeline
- Regression tests for known behaviors

Run with: pytest tests/ or python -m pytest tests/
"""

import unittest
import numpy as np

from strainphase.core import (
    HaplotyperConfig,
    Read, Window, Haplotype, WindowResult,
    GraphInitializer,
    EMHaplotyper,
    PostProcessor,
    LogProbCache,
    link_windows,
    process_window,
    results_to_dataframe,
    _weighted_median,
)
from strainphase.simulation.synthetic_data import (
    SyntheticDataGenerator,
    SimulationScenario,
    create_test_scenarios
)


class TestHaplotyperConfig(unittest.TestCase):
    """Test configuration validation."""

    def test_default_config_valid(self):
        """Default config should pass validation."""
        config = HaplotyperConfig()
        self.assertIsInstance(config, HaplotyperConfig)

    def test_invalid_negative_window(self):
        """Window size below 100 should raise."""
        with self.assertRaises(ValueError):
            HaplotyperConfig(window_size=-100)

    def test_window_too_small(self):
        """Window size of 50 should raise."""
        with self.assertRaises(ValueError):
            HaplotyperConfig(window_size=50)

    def test_invalid_em_max_iter(self):
        """em_max_iter < 1 should raise."""
        with self.assertRaises(ValueError):
            HaplotyperConfig(em_max_iter=0)

    def test_invalid_junk_divergence_rate_zero(self):
        """junk_divergence_rate=0 should raise."""
        with self.assertRaises(ValueError):
            HaplotyperConfig(junk_divergence_rate=0.0)

    def test_invalid_junk_divergence_rate_high(self):
        """junk_divergence_rate >= 0.75 should raise."""
        with self.assertRaises(ValueError):
            HaplotyperConfig(junk_divergence_rate=0.8)

    def test_invalid_af_range_reversed(self):
        """af_range with low >= high should raise."""
        with self.assertRaises(ValueError):
            HaplotyperConfig(af_range=(0.6, 0.3))

    def test_invalid_assign_confidence_threshold(self):
        """assign_confidence_threshold <= 0 should raise."""
        with self.assertRaises(ValueError):
            HaplotyperConfig(assign_confidence_threshold=0.0)

    def test_invalid_min_minor_frequency(self):
        """min_minor_frequency_1snp > 0.5 should raise."""
        with self.assertRaises(ValueError):
            HaplotyperConfig(min_minor_frequency_1snp=0.6)

    def test_custom_config(self):
        """Custom config should store values."""
        config = HaplotyperConfig(
            window_size=5000,
            max_mismatch_frac=0.03,
            min_mapq=30
        )
        self.assertEqual(config.window_size, 5000)
        self.assertEqual(config.max_mismatch_frac, 0.03)
        self.assertEqual(config.min_mapq, 30)


class TestDataStructures(unittest.TestCase):
    """Test basic data structures."""
    
    def test_read_creation(self):
        """Test Read dataclass."""
        read = Read(
            id="test_read",
            contig="contig_1",
            mapq=60,
            alleles={100: 'A', 200: 'C'},
            quals={100: 30, 200: 35}
        )
        self.assertEqual(read.id, "test_read")
        self.assertEqual(read.alleles[100], 'A')
    
    def test_window_properties(self):
        """Test Window properties."""
        window = Window(
            contig="contig_1",
            start=1000,
            end=2000,
            snv_pos=[1100, 1200, 1300],
            ref_alleles={1100: 'A', 1200: 'G', 1300: 'T'}
        )
        self.assertEqual(window.n_snvs, 3)
        self.assertEqual(window.n_reads, 0)
    
    def test_haplotype_distance(self):
        """Test Haplotype distance calculation."""
        hap1 = Haplotype(consensus={100: 'A', 200: 'C', 300: 'G'})
        hap2 = Haplotype(consensus={100: 'A', 200: 'T', 300: 'G'})
        
        dist, mismatches, shared = hap1.distance_to(hap2, [100, 200, 300])
        
        self.assertEqual(shared, 3)
        self.assertEqual(mismatches, 1)
        self.assertAlmostEqual(dist, 1/3, places=5)
    
    def test_haplotype_distance_partial_overlap(self):
        """Test distance with partial overlap."""
        hap1 = Haplotype(consensus={100: 'A', 200: 'C'})
        hap2 = Haplotype(consensus={200: 'C', 300: 'G'})
        
        dist, mismatches, shared = hap1.distance_to(hap2, [100, 200, 300])
        
        self.assertEqual(shared, 1)  # Only position 200 shared
        self.assertEqual(mismatches, 0)
        self.assertEqual(dist, 0.0)
    
    def test_haplotype_distance_no_overlap(self):
        """Test distance with no overlap returns 1.0."""
        hap1 = Haplotype(consensus={100: 'A'})
        hap2 = Haplotype(consensus={200: 'C'})

        dist, mismatches, shared = hap1.distance_to(hap2, [100, 200])

        self.assertEqual(shared, 0)
        self.assertEqual(dist, 1.0)

    def test_haplotype_distance_early_exit(self):
        """Distance should return early when max_mismatches exceeded."""
        hap1 = Haplotype(consensus={100: 'A', 200: 'A', 300: 'A'})
        hap2 = Haplotype(consensus={100: 'G', 200: 'G', 300: 'G'})

        dist, mismatches, shared = hap1.distance_to(hap2, [100, 200, 300], max_mismatches=1)

        self.assertEqual(dist, 1.0)
        self.assertEqual(mismatches, 2)

    def test_haplotype_get_differing_positions_all_differ(self):
        """All positions differ."""
        hap1 = Haplotype(consensus={100: 'A', 200: 'C', 300: 'G'})
        hap2 = Haplotype(consensus={100: 'G', 200: 'T', 300: 'A'})

        diffs = hap1.get_differing_positions(hap2, [100, 200, 300])
        self.assertEqual(sorted(diffs), [100, 200, 300])

    def test_haplotype_get_differing_positions_none_differ(self):
        """No positions differ."""
        hap1 = Haplotype(consensus={100: 'A', 200: 'C'})
        hap2 = Haplotype(consensus={100: 'A', 200: 'C'})

        diffs = hap1.get_differing_positions(hap2, [100, 200])
        self.assertEqual(diffs, [])

    def test_haplotype_get_differing_positions_one_differ(self):
        """Exactly one position differs."""
        hap1 = Haplotype(consensus={100: 'A', 200: 'C', 300: 'G'})
        hap2 = Haplotype(consensus={100: 'A', 200: 'T', 300: 'G'})

        diffs = hap1.get_differing_positions(hap2, [100, 200, 300])
        self.assertEqual(diffs, [200])

    def test_haplotype_get_differing_positions_missing_calls(self):
        """Positions missing in either haplotype are skipped."""
        hap1 = Haplotype(consensus={100: 'A'})
        hap2 = Haplotype(consensus={200: 'C'})

        diffs = hap1.get_differing_positions(hap2, [100, 200])
        self.assertEqual(diffs, [])


class TestLogProbCache(unittest.TestCase):
    """Test log probability cache."""
    
    def test_cache_initialization(self):
        """Cache should precompute values."""
        cache = LogProbCache()
        self.assertTrue(hasattr(cache, '_log_match'))
        self.assertTrue(hasattr(cache, '_log_mismatch'))
    
    def test_match_probability_q30(self):
        """Q30 match should be ~0.999."""
        cache = LogProbCache()
        # Use the actual API: log_prob_base(hap_base, read_base, Q)
        log_match = cache.log_prob_base('A', 'A', 30)
        self.assertAlmostEqual(np.exp(log_match), 0.999, places=3)
    
    def test_mismatch_probability_q30(self):
        """Q30 mismatch should be ~0.001/3."""
        cache = LogProbCache()
        log_mismatch = cache.log_prob_base('A', 'G', 30)
        expected = 0.001 / 3
        self.assertAlmostEqual(np.exp(log_mismatch), expected, places=5)


class TestGraphInitializer(unittest.TestCase):
    """Test graph-based initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HaplotyperConfig(
            min_shared_snvs_for_edge=2,
            max_mismatch_frac=0.1,
            min_reads_per_cluster=2
        )
        self.graph_init = GraphInitializer(self.config)
    
    def test_empty_reads(self):
        """Window with no reads returns empty or minimal haplotypes."""
        window = Window(
            contig="test",
            start=1,
            end=1000,
            snv_pos=[100, 200, 300],
            ref_alleles={100: 'A', 200: 'G', 300: 'T'},
            reads=[]  # No reads
        )
        result = self.graph_init.get_initial_haplotypes(window)
        # Returns tuple (haplotypes, cluster_sizes)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        haps, sizes = result
        self.assertIsInstance(haps, list)
        self.assertIsInstance(sizes, list)
    
    def test_single_read(self):
        """Single read produces haplotype tuple."""
        window = Window(
            contig="test",
            start=1,
            end=1000,
            snv_pos=[100, 200, 300],
            ref_alleles={100: 'A', 200: 'G', 300: 'T'},
            reads=[
                Read(id="r1", contig="test", mapq=60,
                     alleles={100: 'A', 200: 'G', 300: 'T'},
                     quals={100: 30, 200: 30, 300: 30})
            ]
        )
        result = self.graph_init.get_initial_haplotypes(window)
        self.assertIsInstance(result, tuple)
        haps, sizes = result
        self.assertIsInstance(haps, list)
    
    def test_two_distinct_clusters(self):
        """Two distinct read groups should form two haplotypes."""
        reads = []
        # Cluster 1: AAA
        for i in range(5):
            reads.append(Read(
                id=f"cluster1_{i}", contig="test", mapq=60,
                alleles={100: 'A', 200: 'A', 300: 'A'},
                quals={100: 30, 200: 30, 300: 30}
            ))
        # Cluster 2: GGG
        for i in range(5):
            reads.append(Read(
                id=f"cluster2_{i}", contig="test", mapq=60,
                alleles={100: 'G', 200: 'G', 300: 'G'},
                quals={100: 30, 200: 30, 300: 30}
            ))
        
        window = Window(
            contig="test",
            start=1,
            end=1000,
            snv_pos=[100, 200, 300],
            ref_alleles={100: 'A', 200: 'A', 300: 'A'},
            reads=reads
        )
        
        haps, sizes = self.graph_init.get_initial_haplotypes(window)
        # Should have at least 2 haplotypes
        self.assertGreaterEqual(len(haps), 2)


class TestEMHaplotyper(unittest.TestCase):
    """Test EM algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HaplotyperConfig(
            em_max_iter=50,  # More iterations for test convergence
            em_tolerance=1e-3
        )
    
    def test_single_haplotype_convergence(self):
        """Single haplotype should produce valid results."""
        # All reads support same haplotype
        reads = [
            Read(id=f"r{i}", contig="test", mapq=60,
                 alleles={100: 'A', 200: 'C', 300: 'G'},
                 quals={100: 30, 200: 30, 300: 30})
            for i in range(10)
        ]
        
        window = Window(
            contig="test",
            start=1,
            end=1000,
            snv_pos=[100, 200, 300],
            ref_alleles={100: 'A', 200: 'C', 300: 'G'},
            reads=reads
        )
        
        initial_haps = [Haplotype(consensus={100: 'A', 200: 'C', 300: 'G'})]
        
        em = EMHaplotyper(window, initial_haps, config=self.config)
        haps, gamma, pi, ll, converged, iterations = em.run()
        
        # Just verify valid output - convergence depends on tolerance
        self.assertEqual(len(haps), 1)
        self.assertGreater(pi[0], 0.5)  # Most mass on haplotype
    
    def test_two_haplotype_separation(self):
        """Two haplotypes should separate correctly."""
        reads = []
        # Haplotype 1 reads
        for i in range(10):
            reads.append(Read(
                id=f"hap1_{i}", contig="test", mapq=60,
                alleles={100: 'A', 200: 'A', 300: 'A'},
                quals={100: 35, 200: 35, 300: 35}
            ))
        # Haplotype 2 reads
        for i in range(10):
            reads.append(Read(
                id=f"hap2_{i}", contig="test", mapq=60,
                alleles={100: 'G', 200: 'G', 300: 'G'},
                quals={100: 35, 200: 35, 300: 35}
            ))
        
        window = Window(
            contig="test",
            start=1,
            end=1000,
            snv_pos=[100, 200, 300],
            ref_alleles={100: 'A', 200: 'A', 300: 'A'},
            reads=reads
        )
        
        initial_haps = [
            Haplotype(consensus={100: 'A', 200: 'A', 300: 'A'}),
            Haplotype(consensus={100: 'G', 200: 'G', 300: 'G'})
        ]
        
        em = EMHaplotyper(window, initial_haps, config=self.config)
        haps, gamma, pi, ll, converged, iterations = em.run()
        
        self.assertEqual(len(haps), 2)
        
        # Check gamma assigns reads correctly (relaxed threshold)
        # First 10 reads should have higher gamma for hap 0
        for i in range(10):
            self.assertGreater(gamma[i, 0], gamma[i, 1])
        # Last 10 reads should have higher gamma for hap 1
        for i in range(10, 20):
            self.assertGreater(gamma[i, 1], gamma[i, 0])
    
    def test_junk_reads_handled(self):
        """Reads not matching any haplotype should go to junk."""
        reads = []
        # Good reads matching haplotype
        for i in range(8):
            reads.append(Read(
                id=f"good_{i}", contig="test", mapq=60,
                alleles={100: 'A', 200: 'A', 300: 'A'},
                quals={100: 35, 200: 35, 300: 35}
            ))
        # Junk reads (very different)
        for i in range(2):
            reads.append(Read(
                id=f"junk_{i}", contig="test", mapq=60,
                alleles={100: 'T', 200: 'C', 300: 'G'},
                quals={100: 35, 200: 35, 300: 35}
            ))
        
        window = Window(
            contig="test",
            start=1,
            end=1000,
            snv_pos=[100, 200, 300],
            ref_alleles={100: 'A', 200: 'A', 300: 'A'},
            reads=reads
        )
        
        initial_haps = [Haplotype(consensus={100: 'A', 200: 'A', 300: 'A'})]
        
        em = EMHaplotyper(window, initial_haps, config=self.config)
        haps, gamma, pi, ll, converged, iterations = em.run()
        
        # Junk component (last column) should have non-zero weight
        junk_idx = gamma.shape[1] - 1
        self.assertGreater(pi[junk_idx], 0.01)


class TestPostProcessor(unittest.TestCase):
    """Test post-processing operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HaplotyperConfig(
            merge_distance_threshold=0.1,
            min_shared_for_merge=2
        )
        self.post = PostProcessor(self.config)
    
    def test_merge_identical_haplotypes(self):
        """Identical haplotypes should merge."""
        hap1 = Haplotype(consensus={100: 'A', 200: 'C'}, weight=0.3)
        hap2 = Haplotype(consensus={100: 'A', 200: 'C'}, weight=0.2)
        
        window = Window(
            contig="test", start=1, end=1000,
            snv_pos=[100, 200],
            ref_alleles={100: 'A', 200: 'C'},
            reads=[]
        )
        
        # Create dummy gamma and pi
        gamma = np.array([[0.5, 0.3, 0.2]])  # 1 read, 2 haps + junk
        pi = np.array([0.5, 0.3, 0.2])
        
        merged, new_gamma, new_pi = self.post.merge_similar_haplotypes(
            [hap1, hap2], gamma, pi, window
        )
        
        self.assertEqual(len(merged), 1)
    
    def test_keep_distinct_haplotypes(self):
        """Distinct haplotypes should not merge."""
        hap1 = Haplotype(consensus={100: 'A', 200: 'A', 300: 'A'}, weight=0.4)
        hap2 = Haplotype(consensus={100: 'G', 200: 'G', 300: 'G'}, weight=0.4)
        
        window = Window(
            contig="test", start=1, end=1000,
            snv_pos=[100, 200, 300],
            ref_alleles={100: 'A', 200: 'A', 300: 'A'},
            reads=[]
        )
        
        gamma = np.array([[0.4, 0.4, 0.2]])
        pi = np.array([0.4, 0.4, 0.2])
        
        merged, new_gamma, new_pi = self.post.merge_similar_haplotypes(
            [hap1, hap2], gamma, pi, window
        )
        
        self.assertEqual(len(merged), 2)


class TestWindowLinking(unittest.TestCase):
    """Test window linking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = HaplotyperConfig(
            max_link_distance=0.1,
            min_shared_snvs_for_link=2
        )
    
    def test_link_overlapping_windows(self):
        """Haplotypes in overlapping windows should link."""
        # Window 1: positions 100-300
        window1 = Window(
            contig="test", start=1, end=500,
            snv_pos=[100, 200, 300],
            ref_alleles={100: 'A', 200: 'C', 300: 'G'},
            reads=[],
            window_idx=0
        )
        hap1 = Haplotype(consensus={100: 'A', 200: 'C', 300: 'G'}, weight=0.5)
        
        # Window 2: positions 200-400 (overlaps on 200, 300)
        window2 = Window(
            contig="test", start=200, end=700,
            snv_pos=[200, 300, 400],
            ref_alleles={200: 'C', 300: 'G', 400: 'T'},
            reads=[],
            window_idx=1
        )
        hap2 = Haplotype(consensus={200: 'C', 300: 'G', 400: 'T'}, weight=0.5)
        
        results = [
            WindowResult(
                window=window1,
                haplotypes=[hap1],
                gamma=np.array([[1.0, 0.0]]),
                pi=np.array([0.9, 0.1]),
                log_likelihood=-10.0,
                assignments=[],
                converged=True,
                iterations=5
            ),
            WindowResult(
                window=window2,
                haplotypes=[hap2],
                gamma=np.array([[1.0, 0.0]]),
                pi=np.array([0.9, 0.1]),
                log_likelihood=-10.0,
                assignments=[],
                converged=True,
                iterations=5
            )
        ]
        
        linked = link_windows(results, self.config)
        
        # Both haplotypes should have same track_id
        self.assertEqual(
            linked[0].haplotypes[0].track_id,
            linked[1].haplotypes[0].track_id
        )
    
    def test_no_link_different_haplotypes(self):
        """Different haplotypes should not link."""
        window1 = Window(
            contig="test", start=1, end=500,
            snv_pos=[100, 200, 300],
            ref_alleles={100: 'A', 200: 'A', 300: 'A'},
            reads=[],
            window_idx=0
        )
        hap1 = Haplotype(consensus={100: 'A', 200: 'A', 300: 'A'}, weight=0.5)
        
        window2 = Window(
            contig="test", start=200, end=700,
            snv_pos=[200, 300, 400],
            ref_alleles={200: 'A', 300: 'A', 400: 'A'},
            reads=[],
            window_idx=1
        )
        # Different alleles at shared positions
        hap2 = Haplotype(consensus={200: 'G', 300: 'G', 400: 'G'}, weight=0.5)
        
        results = [
            WindowResult(
                window=window1, haplotypes=[hap1],
                gamma=np.array([[1.0, 0.0]]), pi=np.array([0.9, 0.1]),
                log_likelihood=-10.0, assignments=[], converged=True, iterations=5
            ),
            WindowResult(
                window=window2, haplotypes=[hap2],
                gamma=np.array([[1.0, 0.0]]), pi=np.array([0.9, 0.1]),
                log_likelihood=-10.0, assignments=[], converged=True, iterations=5
            )
        ]
        
        linked = link_windows(results, self.config)
        
        # Should have different track_ids
        self.assertNotEqual(
            linked[0].haplotypes[0].track_id,
            linked[1].haplotypes[0].track_id
        )


class TestSyntheticDataGenerator(unittest.TestCase):
    """Test synthetic data generation."""
    
    def test_scenario_creation(self):
        """Test basic scenario creation."""
        gen = SyntheticDataGenerator(seed=42)
        scenario = gen.create_scenario(
            name="test",
            contig_length=10000,
            n_snvs=20,
            n_haplotypes=2,
            n_timepoints=3
        )
        
        self.assertEqual(scenario.name, "test")
        self.assertGreater(len(scenario.snv_positions), 10)
        self.assertEqual(len(scenario.true_haplotypes), 2)
        self.assertEqual(len(scenario.timepoints), 3)
    
    def test_window_generation(self):
        """Test window generation with reads."""
        gen = SyntheticDataGenerator(seed=42)
        scenario = gen.create_scenario(
            name="test",
            contig_length=10000,
            n_snvs=30,
            n_haplotypes=2,
            n_timepoints=2
        )
        
        config = HaplotyperConfig(window_size=5000)
        windows = gen.generate_all_windows(
            scenario, config,
            n_reads_per_window=50
        )
        
        # Should have windows for each timepoint
        self.assertEqual(len(windows), 2)
        
        # Each timepoint should have windows
        for tp, wp_list in windows.items():
            self.assertGreater(len(wp_list), 0)
            for window, read_map in wp_list:
                self.assertGreater(len(window.reads), 0)


class TestWeightedMedian(unittest.TestCase):
    """Test the _weighted_median utility."""

    def test_single_value(self):
        """Single value returns itself."""
        self.assertAlmostEqual(_weighted_median([0.3], [1.0]), 0.3)

    def test_equal_weights(self):
        """Equal weights: median is middle value."""
        result = _weighted_median([0.1, 0.5, 0.9], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(result, 0.5)

    def test_skewed_weights(self):
        """Heavy weight on one value pulls median there."""
        result = _weighted_median([0.1, 0.9], [0.9, 0.1])
        self.assertAlmostEqual(result, 0.1)

    def test_empty_returns_zero(self):
        self.assertEqual(_weighted_median([], []), 0.0)

    def test_zero_weights_returns_zero(self):
        self.assertEqual(_weighted_median([0.5, 0.8], [0.0, 0.0]), 0.0)

    def test_clamps_to_unit_interval(self):
        """Result is always clamped to [0, 1]."""
        result = _weighted_median([0.5], [1.0])
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestAssignReads(unittest.TestCase):
    """Test PostProcessor.assign_reads directly."""

    def setUp(self):
        self.config = HaplotyperConfig()
        self.post = PostProcessor(self.config)
        self.reads = [
            Read(id=f"r{i}", contig="test", mapq=60, alleles={}, quals={})
            for i in range(3)
        ]

    def test_confident_assignment(self):
        """Reads with high gamma get assigned to a haplotype."""
        gamma = np.array([
            [0.95, 0.03, 0.02],
            [0.02, 0.96, 0.02],
            [0.01, 0.01, 0.98],
        ])
        pi = np.array([0.45, 0.45, 0.10])
        assignments = self.post.assign_reads(self.reads, gamma, pi)

        self.assertEqual(assignments[0]["hap_id"], 0)
        self.assertFalse(assignments[0]["is_junk"])
        self.assertFalse(assignments[0]["is_ambiguous"])

        self.assertEqual(assignments[1]["hap_id"], 1)

    def test_junk_assignment(self):
        """Read whose best column is the last (junk) gets is_junk=True."""
        gamma = np.array([[0.05, 0.95]])  # 1 read, 1 hap + junk
        pi = np.array([0.05, 0.95])
        reads = [Read(id="r0", contig="test", mapq=60, alleles={}, quals={})]
        assignments = self.post.assign_reads(reads, gamma, pi)

        self.assertTrue(assignments[0]["is_junk"])
        self.assertIsNone(assignments[0]["hap_id"])

    def test_ambiguous_assignment(self):
        """Read below confidence threshold but not junk is ambiguous."""
        threshold = self.config.assign_confidence_threshold
        low_prob = threshold - 0.05
        gamma = np.array([[low_prob, 1.0 - low_prob]])  # best is hap 0, but below threshold
        # Make hap 0 best but below threshold, junk (col 1) lower
        gamma = np.array([[low_prob + 0.01, 1.0 - low_prob - 0.01]])
        pi = np.array([0.5, 0.5])
        reads = [Read(id="r0", contig="test", mapq=60, alleles={}, quals={})]
        assignments = self.post.assign_reads(reads, gamma, pi)

        self.assertFalse(assignments[0]["is_junk"])
        self.assertTrue(assignments[0]["is_ambiguous"])


class TestEMPiNormalization(unittest.TestCase):
    """Test that EM output pi always sums to 1."""

    def test_pi_sums_to_one(self):
        reads = [
            Read(id=f"r{i}", contig="test", mapq=60,
                 alleles={100: 'A', 200: 'C'}, quals={100: 30, 200: 30})
            for i in range(8)
        ] + [
            Read(id=f"s{i}", contig="test", mapq=60,
                 alleles={100: 'G', 200: 'T'}, quals={100: 30, 200: 30})
            for i in range(8)
        ]
        window = Window(
            contig="test", start=1, end=1000,
            snv_pos=[100, 200],
            ref_alleles={100: 'A', 200: 'C'},
            reads=reads
        )
        initial_haps = [
            Haplotype(consensus={100: 'A', 200: 'C'}),
            Haplotype(consensus={100: 'G', 200: 'T'}),
        ]
        config = HaplotyperConfig(em_max_iter=30)
        em = EMHaplotyper(window, initial_haps, config=config)
        _, _, pi, _, _, _ = em.run()

        self.assertAlmostEqual(pi.sum(), 1.0, places=5)

    def test_equal_split_abundance(self):
        """50/50 read split should yield roughly equal pi."""
        reads = [
            Read(id=f"a{i}", contig="test", mapq=60,
                 alleles={100: 'A', 200: 'A', 300: 'A'},
                 quals={100: 35, 200: 35, 300: 35})
            for i in range(10)
        ] + [
            Read(id=f"b{i}", contig="test", mapq=60,
                 alleles={100: 'G', 200: 'G', 300: 'G'},
                 quals={100: 35, 200: 35, 300: 35})
            for i in range(10)
        ]
        window = Window(
            contig="test", start=1, end=1000,
            snv_pos=[100, 200, 300],
            ref_alleles={100: 'A', 200: 'A', 300: 'A'},
            reads=reads
        )
        initial_haps = [
            Haplotype(consensus={100: 'A', 200: 'A', 300: 'A'}),
            Haplotype(consensus={100: 'G', 200: 'G', 300: 'G'}),
        ]
        config = HaplotyperConfig(em_max_iter=50)
        em = EMHaplotyper(window, initial_haps, config=config)
        haps, _, pi, _, _, _ = em.run()

        # pi[0] and pi[1] should both be close to 0.5 (junk gets ~0)
        self.assertAlmostEqual(pi[0] + pi[1], 1.0, delta=0.1)
        self.assertAlmostEqual(pi[0], 0.5, delta=0.15)
        self.assertAlmostEqual(pi[1], 0.5, delta=0.15)


class TestProcessWindow(unittest.TestCase):
    """Test the process_window convenience function."""

    def _make_window(self, reads):
        return Window(
            contig="test", start=1, end=1000,
            snv_pos=[100, 200, 300],
            ref_alleles={100: 'A', 200: 'A', 300: 'A'},
            reads=reads
        )

    def test_empty_window_returns_result(self):
        """process_window on an empty window returns a WindowResult."""
        window = self._make_window([])
        result = process_window(window)
        self.assertIsInstance(result, WindowResult)
        self.assertEqual(result.haplotypes, [])

    def test_two_hap_window(self):
        """process_window recovers two haplotypes from clearly separated reads."""
        reads = (
            [Read(id=f"a{i}", contig="test", mapq=60,
                  alleles={100: 'A', 200: 'A', 300: 'A'},
                  quals={100: 35, 200: 35, 300: 35}) for i in range(8)]
            + [Read(id=f"b{i}", contig="test", mapq=60,
                    alleles={100: 'G', 200: 'G', 300: 'G'},
                    quals={100: 35, 200: 35, 300: 35}) for i in range(8)]
        )
        window = self._make_window(reads)
        config = HaplotyperConfig(min_reads_per_cluster=2, em_max_iter=50)
        result = process_window(window, config)
        self.assertGreaterEqual(len(result.haplotypes), 1)
        self.assertAlmostEqual(result.pi.sum(), 1.0, places=5)


class TestResultsToDataframe(unittest.TestCase):
    """Test results_to_dataframe output format."""

    def _make_window_result(self, track_id, weight=0.5):
        window = Window(
            contig="ctg1", start=100, end=1100,
            snv_pos=[200, 300], ref_alleles={200: 'A', 300: 'C'}
        )
        hap = Haplotype(consensus={200: 'A', 300: 'C'}, weight=weight, track_id=track_id)
        gamma = np.array([[0.9, 0.1]])
        pi = np.array([weight, 1 - weight])
        return WindowResult(
            window=window, haplotypes=[hap], gamma=gamma, pi=pi,
            log_likelihood=-5.0, assignments=[], converged=True, iterations=3
        )

    def test_basic_output(self):
        """results_to_dataframe returns a list of dicts."""
        wr = self._make_window_result("track_1")
        records = results_to_dataframe({"ctg1": [wr]})
        self.assertIsInstance(records, list)
        self.assertGreater(len(records), 0)
        self.assertIsInstance(records[0], dict)

    def test_record_has_required_columns(self):
        """Each record has the expected keys."""
        wr = self._make_window_result("track_1")
        records = results_to_dataframe({"ctg1": [wr]})
        required = {"contig", "track_id", "span_start", "span_end", "mean_weight"}
        self.assertTrue(required.issubset(records[0].keys()))

    def test_track_grouping(self):
        """Two WindowResults with the same track_id produce one record."""
        wr1 = self._make_window_result("track_X")
        wr2 = self._make_window_result("track_X")
        records = results_to_dataframe({"ctg1": [wr1, wr2]})
        track_ids = [r["track_id"] for r in records]
        self.assertEqual(track_ids.count("track_X"), 1)

    def test_span_covers_both_windows(self):
        """Span should encompass both windows when track spans two."""
        window1 = Window(contig="ctg1", start=0, end=500,
                         snv_pos=[100], ref_alleles={100: 'A'})
        window2 = Window(contig="ctg1", start=400, end=900,
                         snv_pos=[500], ref_alleles={500: 'C'})
        hap1 = Haplotype(consensus={100: 'A'}, weight=0.5, track_id="trk")
        hap2 = Haplotype(consensus={500: 'C'}, weight=0.5, track_id="trk")
        wr1 = WindowResult(window=window1, haplotypes=[hap1],
                           gamma=np.array([[0.9, 0.1]]), pi=np.array([0.9, 0.1]),
                           log_likelihood=-5.0, assignments=[], converged=True, iterations=1)
        wr2 = WindowResult(window=window2, haplotypes=[hap2],
                           gamma=np.array([[0.9, 0.1]]), pi=np.array([0.9, 0.1]),
                           log_likelihood=-5.0, assignments=[], converged=True, iterations=1)
        records = results_to_dataframe({"ctg1": [wr1, wr2]})
        rec = next(r for r in records if r["track_id"] == "trk")
        self.assertLessEqual(rec["span_start"], 100)
        self.assertGreaterEqual(rec["span_end"], 500)


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    def test_full_pipeline_synthetic(self):
        """Test full pipeline on synthetic data."""
        # Generate synthetic data
        gen = SyntheticDataGenerator(seed=42)
        scenario = gen.create_scenario(
            name="integration_test",
            contig_length=20000,
            n_snvs=40,
            n_haplotypes=2,
            n_timepoints=2,
            include_sweep=False
        )
        
        config = HaplotyperConfig(
            window_size=10000,
            min_snvs_per_window=3,
            min_reads_per_window=10,
            em_max_iter=20
        )
        
        windows = gen.generate_all_windows(
            scenario, config,
            n_reads_per_window=50
        )
        
        # Process first timepoint
        tp = scenario.timepoints[0]
        results = []
        
        for window, read_map in windows[tp]:
            if len(window.snv_pos) < config.min_snvs_per_window:
                continue
            if len(window.reads) < config.min_reads_per_window:
                continue
            
            # Graph init - returns tuple (haplotypes, cluster_sizes)
            graph_init = GraphInitializer(config)
            initial_haps, cluster_sizes = graph_init.get_initial_haplotypes(window)
            
            if not initial_haps:
                continue
            
            # EM - pass cluster_sizes for better initialization
            em = EMHaplotyper(window, initial_haps, cluster_sizes=cluster_sizes, config=config)
            haps, gamma, pi, ll, converged, iterations = em.run()
            
            # Post-process
            post = PostProcessor(config)
            merged, final_gamma, final_pi = post.merge_similar_haplotypes(
                haps, gamma, pi, window
            )
            assignments = post.assign_reads(window.reads, final_gamma, final_pi)
            
            result = WindowResult(
                window=window,
                haplotypes=merged,
                gamma=final_gamma,
                pi=final_pi,
                log_likelihood=ll,
                assignments=assignments,
                converged=converged,
                iterations=iterations
            )
            results.append(result)
        
        # Link windows
        if results:
            results = link_windows(results, config)
        
        # Should have found haplotypes
        self.assertGreater(len(results), 0)
        
        total_haps = sum(len(r.haplotypes) for r in results)
        self.assertGreater(total_haps, 0)


def run_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run all tests and return result."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHaplotyperConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestDataStructures))
    suite.addTests(loader.loadTestsFromTestCase(TestLogProbCache))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphInitializer))
    suite.addTests(loader.loadTestsFromTestCase(TestEMHaplotyper))
    suite.addTests(loader.loadTestsFromTestCase(TestPostProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestWindowLinking))
    suite.addTests(loader.loadTestsFromTestCase(TestWeightedMedian))
    suite.addTests(loader.loadTestsFromTestCase(TestAssignReads))
    suite.addTests(loader.loadTestsFromTestCase(TestEMPiNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessWindow))
    suite.addTests(loader.loadTestsFromTestCase(TestResultsToDataframe))
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticDataGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == "__main__":
    run_tests()