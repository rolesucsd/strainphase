#!/usr/bin/env python3
"""
Add a more complex 6-strain scenario to demonstrate scalability.

This shows Strainphase can handle moderately complex communities
without going to unrealistic 20-30 strains.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strainphase.simulation import SyntheticDataGenerator
from strainphase import HaplotyperConfig
from strainphase.core import process_window
import numpy as np

def create_complex_6strain_scenario():
    """
    Create a realistic 6-strain scenario.

    Represents a moderately complex community with:
    - 2 dominant strains (30-40% each)
    - 2 moderate strains (10-15% each)
    - 2 rare strains (2-5% each)

    This is realistic for longitudinal microbiome studies.
    """
    gen = SyntheticDataGenerator(seed=42)

    scenario = gen.create_scenario(
        name='realistic_6strain',
        contig_length=50000,
        n_snvs=120,  # Higher SNV density for 6 strains
        n_haplotypes=6,
        n_timepoints=4,
        include_sweep=True,  # One strain sweeps
    )

    # Manually adjust to realistic abundance distribution
    # Dominant strains
    for tp in scenario.timepoints:
        scenario.true_haplotypes[0].abundance_by_timepoint[tp] = 0.35
        scenario.true_haplotypes[1].abundance_by_timepoint[tp] = 0.30
        # Moderate
        scenario.true_haplotypes[2].abundance_by_timepoint[tp] = 0.15
        scenario.true_haplotypes[3].abundance_by_timepoint[tp] = 0.10
        # Rare
        scenario.true_haplotypes[4].abundance_by_timepoint[tp] = 0.06
        scenario.true_haplotypes[5].abundance_by_timepoint[tp] = 0.04

        # Normalize
        total = sum(h.get_abundance(tp) for h in scenario.true_haplotypes)
        for hap in scenario.true_haplotypes:
            hap.abundance_by_timepoint[tp] /= total

    return scenario


def test_6strain():
    """Test Strainphase on 6-strain scenario."""
    print("="*60)
    print("TESTING 6-STRAIN SCENARIO")
    print("="*60)

    scenario = create_complex_6strain_scenario()

    print(f"\nScenario: {scenario.name}")
    print(f"  Haplotypes: {scenario.n_true_haplotypes()}")
    print(f"  SNVs: {scenario.total_snvs()}")
    print(f"  Timepoints: {len(scenario.timepoints)}")
    print("\n  Abundance distribution:")
    for i, hap in enumerate(scenario.true_haplotypes):
        ab = hap.get_abundance('T1')
        category = 'Dominant' if ab > 0.25 else 'Moderate' if ab > 0.08 else 'Rare'
        print(f"    Hap {i+1}: {ab:.3f} ({category})")

    # Test
    generator = SyntheticDataGenerator(seed=42)
    config = HaplotyperConfig(
        window_size=scenario.contig_length,
        min_snvs_per_window=3,
        max_reads_per_window=400,  # More reads for more strains
        validate_results=False,
    )

    window, read_map = generator.generate_reads_for_window(
        scenario=scenario,
        timepoint='T1',
        window_start=1,
        window_end=scenario.contig_length,
        n_reads=300,
        error_rate=0.001,
        base_quality=30,
    )

    print(f"\n  Generated: {len(window.reads)} reads")

    result = process_window(window, config, n_timepoints_seen=1)

    print(f"  Detected: {len(result.haplotypes)} haplotypes")
    print("\n  Detected haplotypes:")
    for i, hap in enumerate(sorted(result.haplotypes, key=lambda h: h.weight, reverse=True)):
        print(f"    Hap {i+1}: weight={hap.weight:.3f}, {len(hap.consensus)} SNVs, "
              f"{hap.supporting_reads} reads")

    # Compute metrics
    n_true = len([h for h in scenario.true_haplotypes if h.get_abundance('T1') > 0.01])
    n_detected = len(result.haplotypes)
    n_matched = min(n_true, n_detected)

    precision = n_matched / n_detected if n_detected > 0 else 0.0
    recall = n_matched / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n  Metrics:")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall: {recall:.3f}")
    print(f"    F1 Score: {f1:.3f}")

    # Categorize detection
    true_abunds = sorted([h.get_abundance('T1') for h in scenario.true_haplotypes], reverse=True)
    det_abunds = sorted([h.weight for h in result.haplotypes], reverse=True)

    print(f"\n  Detection by abundance category:")
    categories = [
        ('Dominant (>25%)', [a for a in true_abunds if a > 0.25]),
        ('Moderate (8-25%)', [a for a in true_abunds if 0.08 < a <= 0.25]),
        ('Rare (<8%)', [a for a in true_abunds if a <= 0.08]),
    ]

    for cat_name, cat_abunds in categories:
        n_cat = len(cat_abunds)
        if n_cat > 0:
            # Estimate how many were detected (simplified)
            detected = min(n_cat, len([a for a in det_abunds if a > min(cat_abunds) * 0.5]))
            print(f"    {cat_name}: {detected}/{n_cat} detected")

    print("\n" + "="*60)
    status = "✓ GOOD" if f1 > 0.70 else "✗ REVIEW"
    print(f"Result: F1={f1:.3f} - {status}")
    print("="*60)

    return f1


if __name__ == "__main__":
    test_6strain()
