#!/usr/bin/env python3
"""
Tests for strainphase.longitudinal module.

Covers:
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


class TestRescueBelowTheJunkFloor(unittest.TestCase):
    """REGRESSION (R1-16): a rescue is funded ENTIRELY out of junk's weight.

    Below `_MIN_JUNK_WEIGHT` there is nothing to fund one with, and proceeding anyway is
    strictly harmful: the rescued weight is scaled to 0.0, junk is INFLATED up to the
    floor, and every original haplotype is scaled down to make room for a haplotype
    contributing nothing - while the statistic said `was_rescued=True` with a weight it
    never moved. The posterior read count does not imply the weight is there: a handful
    of reads can sit above gamma 0.5 on a pi_junk of 0.002.
    """

    def _window(self, pi_junk):
        from strainphase.core import Read

        snv_pos = [100, 200, 300]
        reads = []
        for i in range(3):
            r = Read(id=f"r{i}", contig="ctg1", mapq=60, ref_start=1, ref_end=1000)
            r.alleles = {100: "G", 200: "G", 300: "G"}
            r.quals = dict.fromkeys(r.alleles, 30)
            reads.append(r)
        window = Window(contig="ctg1", start=1, end=1001, snv_pos=snv_pos,
                        ref_alleles={p: "A" for p in snv_pos}, reads=reads)
        haps = [
            Haplotype(consensus={100: "A", 200: "A", 300: "A"}, weight=0.6),
            Haplotype(consensus={100: "C", 200: "C", 300: "C"}, weight=0.4 - pi_junk),
        ]
        # Every read sits on junk by posterior, however little weight junk holds.
        gamma = np.zeros((len(reads), 3))
        gamma[:, 2] = 1.0
        pi = np.array([0.6, 0.4 - pi_junk, pi_junk])
        return WindowResult(window=window, haplotypes=haps, gamma=gamma, pi=pi,
                            log_likelihood=-10.0, assignments=[], converged=True,
                            iterations=5)

    def _rescue(self, wr):
        config = HaplotyperConfig(rescue_match_distance=0.05, min_shared_for_rescue=2)
        integrator = LongitudinalIntegrator(config)
        donor = Haplotype(consensus={100: "G", 200: "G", 300: "G"}, weight=0.5)
        out = integrator.rescue_window_result(wr, [donor], ["T2"], {}, "T1")
        return out, integrator.rescue_statistics

    def test_a_window_below_the_floor_is_left_exactly_as_it_was(self):
        wr = self._window(pi_junk=0.002)
        before = wr.pi.copy()
        out, stats = self._rescue(wr)

        self.assertEqual([s.reason for s in stats], ["junk_below_floor"])
        self.assertFalse(any(s.was_rescued for s in stats))
        np.testing.assert_array_equal(out.pi, before)
        self.assertEqual(len(out.haplotypes), 2, "no phantom haplotype may be added")

    def test_a_window_with_weight_to_spare_still_rescues(self):
        """Guards the guard: the floor must not make every rescue unreachable."""
        wr = self._window(pi_junk=0.2)
        out, stats = self._rescue(wr)

        self.assertTrue(any(s.was_rescued for s in stats), [s.reason for s in stats])
        self.assertEqual(len(out.haplotypes), 3)
        self.assertAlmostEqual(float(out.pi.sum()), 1.0, places=9)


class TestPublishedAbundancesSumToOne(unittest.TestCase):
    """REGRESSION (R1-3): every haplotype holding pi weight gets a row.

    `abundance` is pi_k / (1 - pi_junk), which sums to 1 over all k by construction. A
    haplotype with no CONFIDENT read (supporting_reads counts gamma >= 0.90, and nothing
    prunes on read support) was skipped, so the published abundances summed to less than
    1 with no residual column and no log line - a reader could not tell mass was missing.
    Two strains agreeing on all but a couple of markers at ~10x depth do exactly that.
    """

    def test_a_haplotype_with_no_confident_read_still_gets_a_row(self):
        from strainphase.longitudinal import build_window_tables

        snv_pos = [12000, 15000, 18000]
        window = Window(contig="c1", start=1, end=20001, snv_pos=snv_pos,
                        ref_alleles={p: "A" for p in snv_pos})
        haps = [
            Haplotype(consensus={12000: "A", 15000: "C", 18000: "G"}, supporting_reads=18),
            Haplotype(consensus={12000: "T", 15000: "C", 18000: "G"}, supporting_reads=0),
        ]
        # Twenty reads, none of them confident about the second haplotype: every row
        # sits at 0.6/0.4 between the two, well under the 0.90 assignment threshold.
        gamma = np.zeros((20, 3))
        gamma[:, 0] = 0.6
        gamma[:, 1] = 0.4
        wr = WindowResult(window=window, haplotypes=haps, gamma=gamma,
                          pi=np.array([0.55, 0.35, 0.10]), log_likelihood=0.0,
                          assignments=[], converged=True, iterations=1)

        config = HaplotyperConfig(window_size=20000, min_entity_overlap_bp=0,
                                  min_cosupported_span_frac=0.0)
        rows = build_window_tables("/tmp/unused", {"MAG1": {"t0": {"c1": [wr]}}},
                                   config, sample_order=["t0"])[0]

        self.assertEqual(len(rows), 2, "a haplotype with pi weight must be published")
        self.assertAlmostEqual(sum(r["abundance"] for r in rows), 1.0, places=9)
        self.assertEqual(min(r["reads"] for r in rows), 0)


class TestRelinkDoesNotDuplicateMismatchRows(unittest.TestCase):
    """REGRESSION (R1-20): re-linking a sample's windows re-derives its mismatch rows.

    The rescue pass calls `link_windows` a second time on the SAME WindowResult objects.
    Appending to `link_mismatches` instead of resetting it doubled every row, byte for
    byte - and those rows are also step 3's veto set, so the duplication is not confined
    to a diagnostics file.
    """

    def test_link_mismatches_are_reset_not_appended(self):
        from strainphase.core import link_windows

        shared = {12000: "A", 14000: "C", 16000: "G", 18000: "T"}
        flipped = {p: {"A": "T", "C": "G", "G": "C", "T": "A"}[b] for p, b in shared.items()}

        def wr(start, consensus):
            w = Window(contig="c1", start=start, end=start + 20000,
                       snv_pos=sorted(consensus))
            hap = Haplotype(consensus=dict(consensus), supporting_reads=20)
            return WindowResult(window=w, haplotypes=[hap], gamma=np.zeros((1, 2)),
                                pi=np.zeros(2), log_likelihood=0.0, assignments=[],
                                converged=True, iterations=1)

        config = HaplotyperConfig(min_entity_overlap_bp=0, min_cosupported_span_frac=0.0)
        results = [wr(1, shared), wr(10001, flipped)]

        first = link_windows(results, config)
        once = [dict(m) for r in first for m in r.link_mismatches]
        self.assertTrue(once, "a genuine disagreement must reach the output")

        second = link_windows(first, config)
        twice = [dict(m) for r in second for m in r.link_mismatches]
        self.assertEqual(twice, once, "re-linking duplicated the mismatch rows")


if __name__ == "__main__":
    unittest.main()
