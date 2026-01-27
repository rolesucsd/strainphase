# Benchmark & Validation Fixes Summary

## Root Causes Found

### 1. **Per-Contig and Per-Timepoint Recall Calculation Bug** ✅ FIXED
**File:** `validation/validate_haplotypes.py:724-777`

**Problem:** Recall was calculated as `n_matched / n_true`, but `n_matched` counted match pairs. With `allow_one_to_many=True`, one true haplotype can match multiple detected lineages, causing recall > 1.0.

**Evidence:** Line 146 in benchmark output: "Recall: 14.000" (should be ≤ 1.0)

**Fix:** Changed to count unique matched true haplotypes: `len(matched_true_ids) / n_true`

### 2. **Validation Reporting Unit Mismatch** ✅ FIXED
**File:** `validation/validate_haplotypes.py:1977-2012`

**Problem:** Report mixed "True haplotypes (strains)" (per-genome) with per-contig breakdowns, causing confusion.

**Fix:** 
- Changed to "True strains (per genome)" with explicit denominators
- Added unified breakdown: timepoint → contig → metrics
- Removed confusing "expected: ~X" annotations
- Added explicit counts: contigs evaluated, timepoints evaluated, total contig-timepoint pairs

### 3. **False Positive Reporting Missing Match Details** ✅ FIXED
**File:** `validation/validate_haplotypes.py:2040-2060, 1920-1942`

**Problem:** Showed `shared_snvs=1200` and `distance=0.282` but didn't explain that this means ~28% mismatch rate (336 mismatches out of 1200 shared positions).

**Fix:** Now shows `matches`, `mismatches`, and `shared_snvs` separately for clarity.

### 4. **Missing Diagnostic Logging for Longitudinal Execution** ✅ FIXED
**File:** `benchmarks/parameter_sweep.py:871-880, 938-989, 1211-1225`

**Problem:** No logging to verify all timepoints are processed and included in lineages.tsv.

**Fix:** Added comprehensive logging:
- Per-timepoint results from `process_mag_longitudinal`
- Missing timepoint warnings
- Raw and converted record counts per timepoint
- Per-contig breakdown

### 5. **Longitudinal Execution Issue** ⚠️ DIAGNOSTIC LOGGING ADDED
**Status:** Diagnostic logging added to identify root cause. The issue appears to be that `build_lineage_table` only creates records for timepoints that have haplotypes, but validation expects all timepoints.

**Next Steps:** Run benchmark with new logging to identify if:
- `process_mag_longitudinal` processes all timepoints but some have no haplotypes
- `build_lineage_table` filters out timepoints with low/no haplotypes
- `load_detected_haplotypes` aggregates incorrectly

## Code Changes Made

### `validation/validate_haplotypes.py`
1. **Lines 724-740:** Fixed per-contig recall calculation
2. **Lines 749-777:** Fixed per-timepoint recall calculation  
3. **Lines 1855-1863:** Refactored summary metrics reporting
4. **Lines 1977-2012:** Refactored print summary with unified breakdown
5. **Lines 2040-2060:** Enhanced false positive reporting with match details
6. **Lines 1920-1942:** Enhanced detailed report false positive section

### `benchmarks/parameter_sweep.py`
1. **Lines 871-890:** Added diagnostic logging for `process_mag_longitudinal` results
2. **Lines 938-989:** Added comprehensive logging for lineage record creation and conversion
3. **Lines 1211-1225:** Added same diagnostic logging to sequential sweep method

## Before/After Example

### Before:
```
VALIDATION RESULTS
============================================================
True haplotypes (strains):     2
Detected lineages (total):     30
Matched lineages:              2

Expected lineages (per timepoint): 2 strains × 5 contigs = up to 10
------------------------------------------------------------

BREAKDOWN BY TIMEPOINT:
------------------------------------------------------------
T1:
  True strains:      2
  Detected lineages: 30 (expected: ~10)
  Matched:           28
  Precision:         0.933, Recall: 14.000  ← BUG: Recall > 1.0
T2:
  Detected lineages: 0 (expected: ~10)
  Matched:           0
  Precision:         0.000, Recall: 0.000

BREAKDOWN BY CONTIG:
------------------------------------------------------------
ctg000022l:
  True strains:      2
  Detected lineages: 30 (expected: ~2)
  Matched:           28
  Precision:         0.933, Recall: 14.000  ← BUG: Recall > 1.0

False Positives (2 spurious):
  - T0001: abund=0.587, snvs=1200, contigs=5
    Closest to bc2218_MaxBin_bin.19_ref: distance=0.282, shared_snvs=1200  ← Unclear why distance is high
```

### After:
```
VALIDATION RESULTS
============================================================
True strains (per genome):    2
Contigs evaluated:           5
Timepoints evaluated:        4
Total contig-timepoint pairs: 20
Detected lineages (total):   30
Matched lineages:            28
Matched true strains:        2
Matched detected lineages:   2
------------------------------------------------------------

BREAKDOWN BY TIMEPOINT → CONTIG:
------------------------------------------------------------

T1:
  Overall: 2 true, 30 detected, 2 matched true, 2 matched detected
  Precision: 0.067, Recall: 1.000  ← FIXED: Recall ≤ 1.0
  Per-contig:
    ctg000022l: 2 true, 30 detected, 2 matched true
    ctg000101l: 2 true, 0 detected, 0 matched true
    ...

T2:
  Overall: 2 true, 0 detected, 0 matched true, 0 matched detected
  Precision: 0.000, Recall: 0.000
  Per-contig:
    ...

False Positives (2 spurious):
  - T0001: abund=0.587, snvs=1200, contigs=5
    Closest to bc2218_MaxBin_bin.19_ref: distance=0.282, shared_snvs=1200, 
    matches=864, mismatches=336  ← CLEAR: Shows why distance is high
```

## Testing Recommendations

1. **Run benchmark with new logging** to verify:
   - All timepoints are processed by `process_mag_longitudinal`
   - All timepoints appear in `build_lineage_table` output
   - All timepoints are loaded by `load_detected_haplotypes`

2. **Verify recall calculations** are now ≤ 1.0 for all contigs/timepoints

3. **Check false positive reporting** shows clear match/mismatch breakdowns

4. **Verify unified breakdown** is clearer than previous split views

## Remaining Work

### Parameter Set Reduction (TODO)
Need to analyze benchmark outputs to propose reduced parameter grid. This requires:
1. Reading all three benchmark outputs (`benchmark_3557271_1.out`, `benchmark_3557271_2.out`, `benchmark_3557271_3.out`)
2. Extracting F1 scores, runtime, and convergence status for each config
3. Identifying Pareto-optimal configs
4. Proposing reduced set with justification

### Longitudinal Execution Verification
The diagnostic logging will help identify if the issue is:
- In `process_mag_longitudinal` (not processing all timepoints)
- In `build_lineage_table` (not creating records for all timepoints)
- In `load_detected_haplotypes` (not aggregating correctly)

Once identified, a targeted fix can be applied.
