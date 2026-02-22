# Parameter Choice Justification

## Overview

StrainPhase uses two families of thresholds that recur at multiple pipeline stages:

1. **Distance thresholds** (normalized Hamming distance on shared SNV positions) -- control when two sequences are considered "same strain"
2. **Minimum shared SNV thresholds** -- control how much overlap is required before computing a distance

We unify these to consistent values across all pipeline stages, supported by parameter sweep evidence showing performance is insensitive to the specific value within a broad range.

---

## Distance Threshold: 0.02 (2% divergence)

### Parameters unified at 0.02

| Parameter | Pipeline Stage | Compares | Previous Default |
|-----------|---------------|----------|-----------------|
| `max_mismatch_frac` | Graph construction | read vs read | 0.005 |
| `merge_distance_threshold` | Within-window haplotype merging | consensus vs consensus | 0.02 |
| `max_link_distance` | Cross-window haplotype linking | consensus vs consensus | 0.005 |
| `rescue_match_distance` | Cross-timepoint rescue | read vs consensus | 0.01 |
| `lineage_merge_distance` | Cross-sample lineage clustering | consensus vs consensus | 0.02 |

### Rationale

All five parameters answer the same question: "is the sequence divergence low enough that these represent the same strain?" A uniform threshold of 2% is justified on three grounds:

1. **Sequencing error tolerance**: Long-read sequencing (ONT/PacBio) has per-read error rates of ~1-2%. When comparing two reads from the same haplotype, independent errors in each read produce an expected pairwise mismatch rate of ~2%. A threshold of 0.02 accommodates this while remaining well below the divergence between distinct strains at variable regions (typically >5%).

2. **Sweep insensitivity**: Parameter grid optimization across 5x, 10x, 20x, and 50x coverage shows that all distance parameters produce equivalent performance (haplotype F1, abundance MAE, SNV F1) across the range 0.005-0.1. The table below shows hap_f1 at 50x for each parameter across tested values:

   | Value | max_mismatch_frac | merge_distance | max_link_distance |
   |-------|-------------------|----------------|-------------------|
   | 0.005 | 0.850 | 0.855 | 0.850 |
   | 0.01  | 0.853 | 0.850 | 0.855 |
   | **0.02** | **0.850** | **0.850** | **0.847** |
   | 0.05  | 0.861 | 0.855 | 0.850 |
   | 0.1   | 0.850 | 0.845 | 0.851 |

   Performance varies by <1.5% across the entire 20-fold range (0.005 to 0.1), confirming the specific value has negligible impact.

3. **Parsimony**: Using a single value across all stages eliminates the need to justify five different thresholds serving the same conceptual role.

### Exception: `junk_divergence_rate` = 0.10

This parameter is *not* a "same strain" threshold. It models the expected mismatch rate of reads that do not belong to any true haplotype (chimeric reads, mapping artifacts, highly erroneous reads). It parameterizes the background noise model in the EM algorithm and lives in a fundamentally different range (~10%) than the strain-identity thresholds (~2%). The sweep shows optimal performance at 0.10-0.20.

---

## Minimum Shared SNV Threshold: 3 (with one exception)

### Parameters at 3

| Parameter | Pipeline Stage | Default |
|-----------|---------------|---------|
| `min_shared_for_merge` | Within-window haplotype merging | 3 |
| `min_shared_snvs_for_link` | Cross-window haplotype linking | 3 |
| `min_shared_for_rescue` | Cross-timepoint rescue | 3 |
| `min_shared_for_lineage` | Cross-sample lineage clustering | 3 |

### Rationale

Three shared informative positions is the minimum for a reliable distance estimate. With only 1-2 shared SNVs, a single sequencing error can swing the normalized Hamming distance from 0.0 to 0.5 or 1.0, making the comparison unreliable. Requiring 3 positions ensures the distance estimate has meaningful resolution.

The sweep confirms values 1-3 produce equivalent results for all four parameters, while values >= 4 cause progressive degradation (up to 3% hap_f1 loss) by preventing valid comparisons in regions with sparse SNV coverage.

### Exception: `min_shared_snvs_for_edge` = 1

The graph construction step (`min_shared_snvs_for_edge`) retains a minimum of 1 shared SNV rather than 3. This is the most sensitivity-critical parameter in the sweep:

| min_shared_snvs_for_edge | hap_f1 (50x, ws=10000) | hap_f1 (50x, ws=500) |
|--------------------------|------------------------|----------------------|
| 1 | 0.850 | 0.894 |
| 2 | 0.791 | 0.894 |
| 3 | 0.797 | 0.894 |
| 4 | 0.762 | 0.860 |

At the recommended window size (500bp), values 1-3 are equivalent. However, at larger window sizes (5000-10000bp), requiring 3 shared SNVs causes a 3.7-5.3% drop in haplotype F1 because reads spanning sparse SNV regions may share only 1-2 informative positions. Since graph edges are between raw reads (not error-corrected consensus sequences), the distance threshold of 0.02 already guards against spurious connections from single-SNV overlaps. Keeping `min_shared_snvs_for_edge=1` maintains robustness across all window sizes without sacrificing accuracy.

---

## Parameter Summary

| Parameter | Value | Group |
|-----------|-------|-------|
| `max_mismatch_frac` | 0.02 | Unified distance |
| `merge_distance_threshold` | 0.02 | Unified distance |
| `max_link_distance` | 0.02 | Unified distance |
| `rescue_match_distance` | 0.02 | Unified distance |
| `lineage_merge_distance` | 0.02 | Unified distance |
| `junk_divergence_rate` | 0.10 | Exception: noise model, not strain identity |
| `min_shared_snvs_for_edge` | 1 | Exception: read-level robustness at all window sizes |
| `min_shared_for_merge` | 3 | Unified min overlap |
| `min_shared_snvs_for_link` | 3 | Unified min overlap |
| `min_shared_for_rescue` | 3 | Unified min overlap |
| `min_shared_for_lineage` | 3 | Unified min overlap |

## Sweep Data Source

Parameter sensitivity was evaluated using coordinate-descent optimization across the following grid, at 5x/10x/20x/50x coverage on real isolate benchmarks (4 strains, 4 timepoints, sweeping abundance profiles):

- Distance values tested: 0.005, 0.01, 0.02, 0.05, 0.1
- Min shared SNV values tested: 1, 2, 3, 4, 5, 6
- Window sizes tested: 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000
- Junk divergence rates tested: 0.05, 0.10, 0.20

Results are in `strainphase_tests/results/test_real_strains_{depth}x/sweep_results/parameter_grid_summary.tsv`.
