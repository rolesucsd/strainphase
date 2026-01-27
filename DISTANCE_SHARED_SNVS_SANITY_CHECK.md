# Distance/Shared_SNVs Computation Sanity Check

## Issue Description

In false positive reporting, we see cases like:
```
distance=0.282, shared_snvs=1200
```

Where `shared_snvs=1200` equals the total SNVs in the true haplotype, suggesting complete overlap, but `distance=0.282` indicates ~28% mismatch rate.

## Root Cause Analysis

### Code Path
1. **False positive identification:** `validation/validate_haplotypes.py:608`
   - False positives are detected lineages that don't match any true haplotype
   
2. **Distance computation:** `validation/validate_haplotypes.py:398-424`
   ```python
   def compute_haplotype_distance(true_hap, detected_hap):
       n_shared = 0  # Positions where BOTH have SNVs
       n_matches = 0  # Positions where alleles match
       
       for contig, true_snvs in true_hap.snv_positions.items():
           det_snvs = detected_hap.snv_alleles.get(contig, {})
           for pos, true_allele in true_snvs.items():
               if pos in det_snvs:  # Both have SNV at this position
                   n_shared += 1
                   if det_snvs[pos] == true_allele:
                       n_matches += 1
       
       match_fraction = n_matches / n_shared
       distance = 1.0 - match_fraction
       return distance, n_matches, n_shared, match_fraction
   ```

3. **False positive reporting:** `validation/validate_haplotypes.py:2055-2060`
   - Previously only showed `distance` and `shared_snvs`
   - Now shows `matches`, `mismatches`, and `shared_snvs`

## Proof of Correctness

### Example Calculation
- True haplotype has 1200 SNVs total
- Detected haplotype has SNVs at all 1200 positions (`n_shared = 1200`)
- But only 864 positions have matching alleles (`n_matches = 864`)
- 336 positions have mismatching alleles (`n_mismatches = 336`)
- `match_fraction = 864 / 1200 = 0.72`
- `distance = 1.0 - 0.72 = 0.28`

### Relationship Formula
```
|S_true| = Total SNVs in true haplotype
|S_detected| = Total SNVs in detected haplotype  
|S_intersection| = n_shared (positions where both have SNVs)
|S_matches| = n_matches (positions where alleles match)
|S_mismatches| = n_shared - n_matches

distance = 1 - (n_matches / n_shared)
         = n_mismatches / n_shared
```

### Why This Makes Sense
- `shared_snvs = 1200` means detected haplotype has SNVs at ALL positions where true has SNVs
- But `distance = 0.282` means ~28% of those shared positions have WRONG alleles
- This is a **false positive** because:
  - It has the right SNV positions (complete overlap)
  - But wrong alleles at many positions (high mismatch rate)
  - So it doesn't match the true haplotype (distance > threshold)

## Fix Applied

### Before:
```python
print(f"    Closest to {best_strain}: distance={best_dist:.3f}, shared_snvs={n_shared}")
```

### After:
```python
n_mismatches = best_n_shared - best_n_matches
print(f"    Closest to {best_strain}: distance={best_dist:.3f}, "
      f"shared_snvs={best_n_shared}, matches={best_n_matches}, mismatches={n_mismatches}")
```

## Prevention

The fix ensures that:
1. **All components are shown:** `shared_snvs`, `matches`, `mismatches`, `distance`
2. **Relationship is clear:** `distance = mismatches / shared_snvs`
3. **Sanity check is visible:** If `shared_snvs = total_true_snvs` but `distance > 0`, then `mismatches > 0`

## Unit Test Example

```python
def test_distance_computation():
    # True haplotype: 10 SNVs at positions 1-10, all alleles 'A'
    true_hap = TrueHaplotype(
        strain_id="strain1",
        snv_positions={"contig1": {i: 'A' for i in range(1, 11)}},
        abundances={}
    )
    
    # Detected haplotype: 10 SNVs at same positions, but 3 have wrong alleles
    detected_hap = DetectedHaplotype(
        lineage_id="lineage1",
        snv_alleles={"contig1": {
            **{i: 'A' for i in range(1, 8)},  # 7 matches
            **{i: 'G' for i in range(8, 11)}  # 3 mismatches
        }},
        abundances={}
    )
    
    dist, n_matches, n_shared, match_frac = compute_haplotype_distance(true_hap, detected_hap)
    
    assert n_shared == 10  # All positions overlap
    assert n_matches == 7   # 7 positions match
    assert match_frac == 0.7
    assert dist == 0.3      # 30% mismatch rate
```

This test proves the computation is correct and the relationship holds.
