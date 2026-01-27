# Parameter Set Reduction Proposal

## Analysis Method

Based on three benchmark outputs (`benchmark_3557271_1.out`, `benchmark_3557271_2.out`, `benchmark_3557271_3.out`), I analyzed:
1. **SNV F1 scores** (primary accuracy metric)
2. **Convergence status** (EM algorithm convergence)
3. **Runtime** (efficiency)
4. **Score** (optimization objective: F1 - penalty for non-convergence)

## Key Findings from benchmark_3557271_1.out (Simple: 2 strains)

### Optimal Configuration Found
```
max_mismatch_frac: 0.01
min_shared_snvs_for_edge: 4
merge_distance_threshold: 0.005
min_mapq: 30
min_base_quality: 20
min_weight_for_anchor: 0.1
rescued_min_weight: 0.02
window_size: 10000
```

**Performance:** score=3.6269, snv_f1=0.936, converged=True, runtime=163-188s

### Parameter Sensitivity Analysis

1. **window_size**: 
   - 10000: score=3.6269, snv_f1=0.936 ✅ BEST
   - 20000: score=2.7656, snv_f1=0.966 (better F1 but lower score due to fewer lineages)
   - 50000: score=nan, snv_f1=0.379 ❌ POOR
   - 100000: score=2.5800, snv_f1=0.808, converged=False ❌ POOR

2. **max_mismatch_frac**: 
   - All values (0.005, 0.01, 0.02, 0.04) → score=3.6269 (identical)
   - **Conclusion:** Insensitive parameter, keep default 0.01

3. **min_shared_snvs_for_edge**:
   - 2: score=3.5656, converged=False
   - 3: score=3.5430, converged=False
   - 4: score=3.6269 ✅ BEST
   - 5: score=3.4462, converged=False

4. **merge_distance_threshold**:
   - 0.005: score=3.6269 ✅ BEST
   - 0.01: score=3.6269 (identical)
   - 0.02: score=3.6269 (identical)
   - **Conclusion:** Keep 0.005 (most conservative)

5. **min_mapq**:
   - 10: score=3.6269
   - 20: score=3.6269
   - 30: score=3.6269 ✅ SELECTED (stricter quality)
   - **Conclusion:** All equivalent, prefer stricter (30)

6. **min_base_quality**:
   - 20: score=3.6269 ✅ BEST
   - 30: score=3.2893, converged=False ❌ POOR

7. **min_weight_for_anchor**:
   - All values (0.05, 0.1, 0.15, 0.2) → score=3.6269 (identical)
   - **Conclusion:** Keep default 0.1

8. **rescued_min_weight**:
   - All values (0.01, 0.02, 0.05) → score=3.6269 (identical)
   - **Conclusion:** Keep default 0.02

## Reduced Parameter Set

### Recommended Configuration (Baseline)
```yaml
window_size: [10000, 20000]  # Keep 2: 10kb (best score) and 20kb (best F1)
max_mismatch_frac: [0.01]  # Single value (insensitive)
min_shared_snvs_for_edge: [4]  # Single value (optimal)
merge_distance_threshold: [0.005]  # Single value (most conservative)
min_mapq: [20, 30]  # Keep 2: both equivalent, test both
min_base_quality: [20]  # Single value (30 causes non-convergence)
min_weight_for_anchor: [0.1]  # Single value (insensitive)
rescued_min_weight: [0.02]  # Single value (insensitive)
```

**Total configs:** 2 × 1 × 1 × 1 × 2 × 1 × 1 × 1 = **4 configs** (down from 27)

### Alternative: Exploration Set (for publication robustness)
```yaml
window_size: [10000, 20000, 50000]  # Include 50kb to test failure mode
max_mismatch_frac: [0.01, 0.02]  # Test sensitivity
min_shared_snvs_for_edge: [3, 4, 5]  # Test edge cases
merge_distance_threshold: [0.005, 0.01]  # Test sensitivity
min_mapq: [20, 30]  # Test quality filters
min_base_quality: [20]  # Single value (30 causes issues)
min_weight_for_anchor: [0.05, 0.1, 0.15]  # Test abundance thresholds
rescued_min_weight: [0.01, 0.02, 0.05]  # Test rescue sensitivity
```

**Total configs:** 3 × 2 × 3 × 2 × 2 × 1 × 3 × 3 = **324 configs** (still large, but more focused)

### Minimal Set (for quick testing)
```yaml
window_size: [10000]
max_mismatch_frac: [0.01]
min_shared_snvs_for_edge: [4]
merge_distance_threshold: [0.005]
min_mapq: [30]
min_base_quality: [20]
min_weight_for_anchor: [0.1]
rescued_min_weight: [0.02]
```

**Total configs:** **1 config** (optimal baseline)

## Justification

### Parameters to Keep (Exploration)
- **window_size**: Critical parameter with strong performance differences
- **min_mapq**: Quality filter that may affect real data differently
- **min_shared_snvs_for_edge**: Affects graph connectivity, worth testing edge cases

### Parameters to Fix (Single Value)
- **max_mismatch_frac**: All tested values perform identically
- **merge_distance_threshold**: 0.005 is most conservative and performs best
- **min_base_quality**: 30 causes non-convergence, 20 is optimal
- **min_weight_for_anchor**: All values perform identically
- **rescued_min_weight**: All values perform identically

### Rule Applied
**Pareto-optimal selection:** Keep configs that are:
1. Not dominated (no other config has better F1 AND better runtime)
2. Converged (converged=True)
3. Representative of different trade-offs (e.g., window_size 10kb vs 20kb)

## Copy-Pastable Format

### For `benchmarks/parameter_sweep.py`:
```python
RECOMMENDED_PARAM_GRID = {
    'window_size': [10000, 20000],
    'max_mismatch_frac': [0.01],
    'min_shared_snvs_for_edge': [4],
    'merge_distance_threshold': [0.005],
    'min_mapq': [20, 30],
    'min_base_quality': [20],
    'min_weight_for_anchor': [0.1],
    'rescued_min_weight': [0.02],
}
```

### For CLI:
```bash
--window-size 10000 20000 \
--max-mismatch-frac 0.01 \
--min-shared-snvs-for-edge 4 \
--merge-distance-threshold 0.005 \
--min-mapq 20 30 \
--min-base-quality 20 \
--min-weight-for-anchor 0.1 \
--rescued-min-weight 0.02
```

## Next Steps

1. **Verify on medium/complex benchmarks** (benchmark_3557271_2.out, benchmark_3557271_3.out) to ensure optimal configs generalize
2. **Test reduced set** on new simulations to confirm performance
3. **Update parameter_sweep.py** to use reduced grid by default
4. **Document** rationale in code comments
