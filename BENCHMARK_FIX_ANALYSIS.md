# Benchmark & Validation Fix Analysis

## Root Causes Identified

### 1. Longitudinal Execution Failure

**Evidence from `benchmark_3557271_1.out`:**
- Line 67: "Longitudinal mode: 4 timepoints (T1, T2, T3, T4)" - Mode correctly detected
- Line 86: "Running validation (single-timepoint mode)..." - But validation runs in single-timepoint mode
- Lines 114-128: T2, T3, T4 all show "Detected lineages: 0" - Only T1 has results

**Root Cause:**
The logging for "Sample {sample_id}: initial contig processing" (line 171 in `longitudinal.py`) is NOT appearing in the output, suggesting `process_mag_longitudinal` may not be processing all timepoints, OR the logging level filters it out. However, the real issue is that `build_lineage_table` should create records for all timepoints, but the validation shows only T1.

**Likely Issue:**
- `process_mag_longitudinal` processes all timepoints correctly (line 170: `for sample_id in samples`)
- BUT `build_lineage_table` may only be creating records for timepoints that have haplotypes
- OR `load_detected_haplotypes` is not correctly aggregating abundances from multiple timepoint records

**File/Line:** `benchmarks/parameter_sweep.py:930-970` (lineage record conversion), `validation/validate_haplotypes.py:323-388` (loading detected haplotypes)

### 2. Validation Reporting Unit Mismatch

**Evidence:**
- Line 100: "True haplotypes (strains): 2" - This is per-genome
- Line 104: "Expected lineages (per timepoint): 2 strains × 5 contigs = up to 10" - Mixing units
- Lines 130-156: "BREAKDOWN BY CONTIG" shows per-contig metrics
- Lines 107-128: "BREAKDOWN BY TIMEPOINT" shows per-timepoint metrics

**Root Cause:**
The report mixes "strains" (per-genome) with "lineages" (per-contig). The "True haplotypes (strains)" label is misleading because:
- Strainphase splits lineages per-contig
- So 2 strains × 5 contigs = up to 10 true haplotypes (per contig)
- But the report says "2" which is per-genome

**File/Line:** `validation/validate_haplotypes.py:1980-2012` (print_validation_summary), `validation/validate_haplotypes.py:1855-1863` (detailed_report.txt)

### 3. Overall Metrics Inconsistency

**Evidence:**
- Line 160: "Precision: 0.933, Recall: 1.000" - Overall metrics look good
- But lines 136, 141, 150, 155: Many contigs show "Precision: 0.000, Recall: 0.000"
- Line 146: One contig (ctg000022l) shows "Precision: 0.933, Recall: 14.000" (recall > 1.0 is wrong!)

**Root Cause:**
- Overall metrics are computed at the strain level (lines 599-603 in `validate_haplotypes.py`)
- Per-contig metrics are computed separately (lines 724-740)
- The aggregation doesn't account for the fact that one strain can appear in multiple contigs
- Recall > 1.0 indicates a bug in per-contig recall calculation

**File/Line:** `validation/validate_haplotypes.py:572-798` (compute_validation_metrics)

### 4. False Positive Distance/Shared_SNVs Confusion

**Evidence:**
- Line 182: "distance=0.282, shared_snvs=1200"
- If `shared_snvs=1200` equals total SNVs in true haplotype, this means detected has ALL true SNVs
- But `distance=0.282` means ~28% mismatch rate
- This is actually CORRECT: distance = 1 - (n_matches / n_shared)
- But the reporting is confusing because it doesn't show `n_matches` vs `n_mismatches`

**Root Cause:**
The reporting shows `shared_snvs` but not `n_matches` or `n_mismatches`, making it unclear why distance is high when shared_snvs equals total.

**File/Line:** `validation/validate_haplotypes.py:2040-2060` (false positive reporting)

### 5. Missing Diagnostic Logging

**Evidence:**
- The log doesn't show "Lineage records include timepoints: [...]" 
- The log doesn't show "Records per timepoint: {...}"
- The log doesn't show per-timepoint processing results

**Root Cause:**
The diagnostic logging I added isn't appearing, suggesting either:
- The code wasn't deployed when this benchmark ran
- OR the logging is being filtered out
- OR there's an exception being caught silently

**File/Line:** `benchmarks/parameter_sweep.py:941-970` (should have logging but doesn't appear)

## Fixes Required

### Fix 1: Ensure All Timepoints Are Processed and Included

1. Add explicit logging to verify all timepoints are processed
2. Verify `build_lineage_table` creates records for all timepoints
3. Verify `load_detected_haplotypes` correctly aggregates abundances from multiple timepoint records
4. Fix the validation mode detection (currently says "single-timepoint" even when longitudinal)

### Fix 2: Refactor Validation Reporting

1. Change "True haplotypes (strains)" to "True haplotypes (per contig)" or "True strains (per genome)"
2. Make all metrics consistent with a single unit (contig)
3. Remove confusing "expected: ~X" annotations
4. Create a unified breakdown: timepoint → contig → metrics (or vice versa)

### Fix 3: Fix Metric Aggregation

1. Fix per-contig recall calculation (should not exceed 1.0)
2. Make overall metrics aggregate correctly from per-contig metrics
3. Add explicit denominator counts in the report

### Fix 4: Improve False Positive Reporting

1. Show `n_matches`, `n_mismatches`, and `n_shared` separately
2. Add a sanity check comment explaining the relationship

### Fix 5: Add Diagnostic Logging

1. Log which timepoints have lineage records
2. Log per-timepoint processing results
3. Log per-timepoint record counts
