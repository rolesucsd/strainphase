#!/usr/bin/env python3
"""
Tests for strainphase.longitudinal module.

Covers:
- _weighted_median
- LongitudinalIntegrator.build_anchor_panel_for_key
- LongitudinalIntegrator.count_timepoints_for_haplotype
- LongitudinalIntegrator.rescue_window_result (basic)
"""

import unittest
import numpy as np

from strainphase.core import (
    Haplotype,
    HaplotyperConfig,
    Window,
    WindowResult,
    LongitudinalIntegrator,
)
from strainphase.longitudinal import _weighted_median


def _make_window_result(haplotypes, snv_pos=None, start=0, end=1000):
    """Helper to build a minimal WindowResult."""
    snv_pos = snv_pos or [100, 200, 300]
    ref_alleles = {p: 'A' for p in snv_pos}
    window = Window(contig="ctg1", start=start, end=end,
                    snv_pos=snv_pos, ref_alleles=ref_alleles, reads=[])
    n_haps = len(haplotypes)
    gamma = np.ones((1, n_haps + 1)) / (n_haps + 1)
    pi = np.array([h.weight for h in haplotypes] + [0.0])
    if pi.sum() > 0:
        pi /= pi.sum()
    return WindowResult(
        window=window, haplotypes=haplotypes, gamma=gamma, pi=pi,
        log_likelihood=-10.0, assignments=[], converged=True, iterations=5,
    )


class TestLongitudinalWeightedMedian(unittest.TestCase):
    """Test the _weighted_median helper in longitudinal.py."""

    def test_single_value(self):
        self.assertAlmostEqual(_weighted_median([0.4], [1.0]), 0.4)

    def test_equal_weights_three_values(self):
        result = _weighted_median([0.1, 0.5, 0.9], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(result, 0.5)

    def test_heavy_weight_on_low_value(self):
        result = _weighted_median([0.1, 0.9], [0.9, 0.1])
        self.assertAlmostEqual(result, 0.1)

    def test_empty_returns_zero(self):
        self.assertEqual(_weighted_median([], []), 0.0)

    def test_zero_weights_returns_zero(self):
        self.assertEqual(_weighted_median([0.3, 0.7], [0.0, 0.0]), 0.0)

    def test_result_clamped_to_unit_interval(self):
        result = _weighted_median([0.5], [1.0])
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestBuildAnchorPanel(unittest.TestCase):
    """Test LongitudinalIntegrator.build_anchor_panel_for_key."""

    def setUp(self):
        self.config = HaplotyperConfig(min_weight_for_anchor=0.2)
        self.integrator = LongitudinalIntegrator(self.config)

    def test_high_weight_haps_included(self):
        """Haplotypes above min_weight_for_anchor are added to the panel."""
        hap = Haplotype(consensus={100: 'A'}, weight=0.5)
        wr = _make_window_result([hap])
        sample_results = {"T1": wr}

        anchors, samples = self.integrator.build_anchor_panel_for_key(sample_results)

        self.assertEqual(len(anchors), 1)
        self.assertIn("T1", samples)

    def test_low_weight_haps_excluded(self):
        """Haplotypes below min_weight_for_anchor are excluded by default."""
        hap = Haplotype(consensus={100: 'A'}, weight=0.05)
        wr = _make_window_result([hap])
        sample_results = {"T1": wr}

        anchors, _ = self.integrator.build_anchor_panel_for_key(sample_results)

        self.assertEqual(len(anchors), 0)

    def test_include_low_weight_flag(self):
        """include_low_weight=True includes below-threshold haplotypes."""
        hap = Haplotype(consensus={100: 'A'}, weight=0.05)
        wr = _make_window_result([hap])
        sample_results = {"T1": wr}

        anchors, _ = self.integrator.build_anchor_panel_for_key(
            sample_results, include_low_weight=True
        )

        self.assertEqual(len(anchors), 1)

    def test_exclude_sample(self):
        """exclude_sample removes that timepoint's haplotypes."""
        hap1 = Haplotype(consensus={100: 'A'}, weight=0.5)
        hap2 = Haplotype(consensus={100: 'G'}, weight=0.5)
        wr1 = _make_window_result([hap1])
        wr2 = _make_window_result([hap2])
        sample_results = {"T1": wr1, "T2": wr2}

        anchors, samples = self.integrator.build_anchor_panel_for_key(
            sample_results, exclude_sample="T1"
        )

        self.assertNotIn("T1", samples)
        self.assertIn("T2", samples)

    def test_multiple_timepoints_pooled(self):
        """Anchors from multiple timepoints are all collected."""
        hap1 = Haplotype(consensus={100: 'A'}, weight=0.5)
        hap2 = Haplotype(consensus={100: 'G'}, weight=0.5)
        sample_results = {
            "T1": _make_window_result([hap1]),
            "T2": _make_window_result([hap2]),
        }

        anchors, _ = self.integrator.build_anchor_panel_for_key(sample_results)

        self.assertEqual(len(anchors), 2)


class TestCountTimepointsForHaplotype(unittest.TestCase):
    """Test LongitudinalIntegrator.count_timepoints_for_haplotype."""

    def setUp(self):
        self.config = HaplotyperConfig(
            rescue_match_distance=0.02,
            min_shared_for_rescue=2,
        )
        self.integrator = LongitudinalIntegrator(self.config)

    def test_matching_hap_counts_timepoint(self):
        """An identical haplotype in another timepoint is counted."""
        hap = Haplotype(consensus={100: 'A', 200: 'C', 300: 'G'}, weight=0.5)
        same_hap = Haplotype(consensus={100: 'A', 200: 'C', 300: 'G'}, weight=0.5)
        wr = _make_window_result([same_hap], snv_pos=[100, 200, 300])
        sample_results = {"T1": wr}

        count = self.integrator.count_timepoints_for_haplotype(
            hap, sample_results, [100, 200, 300]
        )
        self.assertEqual(count, 1)

    def test_different_hap_not_counted(self):
        """A divergent haplotype in another timepoint is not counted."""
        hap = Haplotype(consensus={100: 'A', 200: 'A', 300: 'A'}, weight=0.5)
        other = Haplotype(consensus={100: 'G', 200: 'G', 300: 'G'}, weight=0.5)
        wr = _make_window_result([other], snv_pos=[100, 200, 300])
        sample_results = {"T1": wr}

        count = self.integrator.count_timepoints_for_haplotype(
            hap, sample_results, [100, 200, 300]
        )
        self.assertEqual(count, 0)

    def test_counted_across_multiple_timepoints(self):
        """Same haplotype present in two timepoints is counted twice."""
        hap = Haplotype(consensus={100: 'A', 200: 'C', 300: 'G'}, weight=0.5)
        same = Haplotype(consensus={100: 'A', 200: 'C', 300: 'G'}, weight=0.5)
        sample_results = {
            "T1": _make_window_result([same], snv_pos=[100, 200, 300]),
            "T2": _make_window_result([same], snv_pos=[100, 200, 300]),
        }

        count = self.integrator.count_timepoints_for_haplotype(
            hap, sample_results, [100, 200, 300]
        )
        self.assertEqual(count, 2)

    def test_insufficient_shared_snvs_not_counted(self):
        """Match requires min_shared_for_rescue shared positions."""
        config = HaplotyperConfig(min_shared_for_rescue=3)
        integrator = LongitudinalIntegrator(config)

        hap = Haplotype(consensus={100: 'A', 200: 'C'}, weight=0.5)
        same = Haplotype(consensus={100: 'A', 200: 'C'}, weight=0.5)
        wr = _make_window_result([same], snv_pos=[100, 200])
        sample_results = {"T1": wr}

        count = integrator.count_timepoints_for_haplotype(
            hap, sample_results, [100, 200]
        )
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
