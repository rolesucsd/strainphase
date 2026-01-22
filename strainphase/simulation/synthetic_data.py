#!/usr/bin/env python3
"""
Synthetic data generator for haplotyper pipeline testing.

Creates realistic synthetic metagenomic data with:
- Multiple true haplotypes with defined consensus sequences
- Reads sampled from haplotypes with configurable error rates
- Temporal dynamics (abundance changes over timepoints)
- Option for selective sweep events

This allows testing pipeline behavior without real BAM/VCF files.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strainphase.core import Read, Window, HaplotyperConfig


@dataclass
class TrueHaplotype:
    """Ground truth haplotype for simulation."""
    id: str
    consensus: Dict[int, str]  # pos -> base
    abundance_by_timepoint: Dict[str, float] = field(default_factory=dict)
    
    def get_abundance(self, timepoint: str) -> float:
        return self.abundance_by_timepoint.get(timepoint, 0.0)


@dataclass
class SimulationScenario:
    """Complete simulation scenario with ground truth."""
    name: str
    contig_id: str
    contig_length: int
    snv_positions: List[int]
    ref_alleles: Dict[int, str]
    true_haplotypes: List[TrueHaplotype]
    timepoints: List[str]
    
    # For tracking sweeps
    sweep_events: List[Dict] = field(default_factory=list)
    
    def total_snvs(self) -> int:
        return len(self.snv_positions)
    
    def n_true_haplotypes(self) -> int:
        return len(self.true_haplotypes)


class SyntheticDataGenerator:
    """
    Generator for synthetic metagenomic haplotype data.
    
    Creates Windows with reads that can be fed directly to the pipeline,
    bypassing file I/O.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.bases = ['A', 'C', 'G', 'T']
    
    def create_scenario(
        self,
        name: str = "test_scenario",
        contig_length: int = 50000,
        n_snvs: int = 100,
        n_haplotypes: int = 3,
        n_timepoints: int = 4,
        include_sweep: bool = True,
        snv_density_per_kb: float = 2.0
    ) -> SimulationScenario:
        """
        Create a complete simulation scenario with ground truth haplotypes.
        
        Args:
            name: Scenario identifier
            contig_length: Length of simulated contig
            n_snvs: Number of SNV positions
            n_haplotypes: Number of true haplotypes
            n_timepoints: Number of timepoints to simulate
            include_sweep: Whether to include a selective sweep event
            snv_density_per_kb: Target SNV density (may adjust n_snvs)
        """
        contig_id = f"{name}_contig_1"
        
        # Generate SNV positions (evenly distributed with some noise)
        snv_positions = self._generate_snv_positions(contig_length, n_snvs)
        
        # Generate reference alleles
        ref_alleles = {pos: self.rng.choice(self.bases) for pos in snv_positions}
        
        # Generate true haplotypes
        timepoints = [f"T{i+1}" for i in range(n_timepoints)]
        true_haplotypes = self._generate_haplotypes(
            snv_positions, ref_alleles, n_haplotypes, timepoints, include_sweep
        )
        
        # Record sweep events if any
        sweep_events = []
        if include_sweep and n_haplotypes >= 2:
            sweep_events.append({
                'type': 'selective_sweep',
                'winner': true_haplotypes[0].id,
                'loser': true_haplotypes[1].id,
                'start_timepoint': timepoints[1],
                'end_timepoint': timepoints[-1],
            })
        
        return SimulationScenario(
            name=name,
            contig_id=contig_id,
            contig_length=contig_length,
            snv_positions=snv_positions,
            ref_alleles=ref_alleles,
            true_haplotypes=true_haplotypes,
            timepoints=timepoints,
            sweep_events=sweep_events
        )
    
    def _generate_snv_positions(self, contig_length: int, n_snvs: int) -> List[int]:
        """Generate SNV positions with some clustering (realistic)."""
        # Start with even spacing
        base_spacing = contig_length / (n_snvs + 1)
        positions = []
        
        for i in range(n_snvs):
            base_pos = int((i + 1) * base_spacing)
            # Add noise (±20% of spacing)
            noise = int(self.rng.normal(0, base_spacing * 0.2))
            pos = max(1, min(contig_length - 1, base_pos + noise))
            positions.append(pos)
        
        # Ensure unique and sorted
        positions = sorted(set(positions))
        return positions
    
    def _generate_haplotypes(
        self,
        snv_positions: List[int],
        ref_alleles: Dict[int, str],
        n_haplotypes: int,
        timepoints: List[str],
        include_sweep: bool
    ) -> List[TrueHaplotype]:
        """
        Generate true haplotypes with defined relationships and temporal dynamics.
        """
        haplotypes = []
        
        for h_idx in range(n_haplotypes):
            hap_id = f"TRUE_HAP_{h_idx}"
            
            # Generate consensus (divergent from reference)
            consensus = {}
            for pos in snv_positions:
                ref_base = ref_alleles[pos]
                
                if h_idx == 0:
                    # First haplotype: ~30% divergent from ref
                    if self.rng.random() < 0.3:
                        alt_bases = [b for b in self.bases if b != ref_base]
                        consensus[pos] = self.rng.choice(alt_bases)
                    else:
                        consensus[pos] = ref_base
                else:
                    # Subsequent haplotypes: derived from first with mutations
                    base_consensus = haplotypes[0].consensus[pos]
                    
                    # ~5-10% different from dominant haplotype
                    if self.rng.random() < 0.05 + 0.05 * h_idx:
                        alt_bases = [b for b in self.bases if b != base_consensus]
                        consensus[pos] = self.rng.choice(alt_bases)
                    else:
                        consensus[pos] = base_consensus
            
            # Generate temporal abundance dynamics
            abundance_by_timepoint = self._generate_abundance_trajectory(
                h_idx, n_haplotypes, timepoints, include_sweep
            )
            
            haplotypes.append(TrueHaplotype(
                id=hap_id,
                consensus=consensus,
                abundance_by_timepoint=abundance_by_timepoint
            ))
        
        # Normalize abundances so they sum to 1.0 for each timepoint
        for tp in timepoints:
            total = sum(h.get_abundance(tp) for h in haplotypes)
            if total > 0:
                for hap in haplotypes:
                    hap.abundance_by_timepoint[tp] = hap.get_abundance(tp) / total
        
        return haplotypes
    
    def _generate_abundance_trajectory(
        self,
        hap_idx: int,
        n_haplotypes: int,
        timepoints: List[str],
        include_sweep: bool
    ) -> Dict[str, float]:
        """
        Generate realistic abundance trajectory for a haplotype.
        
        If include_sweep is True, haplotype 0 will sweep to dominance
        while haplotype 1 decreases.
        """
        abundances = {}
        n_tp = len(timepoints)
        
        if include_sweep:
            if hap_idx == 0:
                # Dominant strain: starts moderate, sweeps to high
                start_abund = 0.40
                end_abund = 0.75
            elif hap_idx == 1:
                # Swept strain: starts moderate, declines
                start_abund = 0.35
                end_abund = 0.05
            else:
                # Other strains: relatively stable low abundance
                start_abund = (1.0 - 0.75 - 0.35) / max(1, n_haplotypes - 2)
                end_abund = (1.0 - 0.80 - 0.05) / max(1, n_haplotypes - 2)
        else:
            # No sweep: roughly stable abundances
            base_abund = 1.0 / n_haplotypes
            start_abund = base_abund * (1.2 if hap_idx == 0 else 0.9)
            end_abund = start_abund
        
        for i, tp in enumerate(timepoints):
            # Linear interpolation with noise
            frac = i / max(1, n_tp - 1)
            base = start_abund + (end_abund - start_abund) * frac
            noise = self.rng.normal(0, 0.03)
            abundances[tp] = max(0.01, min(0.95, base + noise))
        
        return abundances
    
    def generate_reads_for_window(
        self,
        scenario: SimulationScenario,
        timepoint: str,
        window_start: int,
        window_end: int,
        n_reads: int = 100,
        error_rate: float = 0.001,
        base_quality: int = 30,
        mapq: int = 60
    ) -> Tuple[Window, Dict[str, str]]:
        """
        Generate synthetic reads for a window.
        
        Returns:
            (Window object with reads, dict mapping read_id to true haplotype)
        """
        # Get SNVs in this window
        window_snvs = [p for p in scenario.snv_positions 
                       if window_start <= p < window_end]
        
        if not window_snvs:
            return None, {}
        
        # Build reference alleles for window
        ref_alleles = {p: scenario.ref_alleles[p] for p in window_snvs}
        
        reads = []
        read_true_haplotypes = {}
        
        # Sample reads from haplotypes according to abundance
        abundances = [h.get_abundance(timepoint) for h in scenario.true_haplotypes]
        total_abund = sum(abundances)
        if total_abund <= 0:
            return None, {}
        
        probs = [a / total_abund for a in abundances]
        
        for read_idx in range(n_reads):
            # Pick source haplotype
            source_idx = self.rng.choice(len(scenario.true_haplotypes), p=probs)
            source_hap = scenario.true_haplotypes[source_idx]
            
            read_id = f"read_{timepoint}_{window_start}_{read_idx}"
            read_true_haplotypes[read_id] = source_hap.id
            
            # Generate read alleles (with errors)
            alleles = {}
            quals = {}
            
            # Read covers ~60-90% of SNVs in window
            coverage_frac = self.rng.uniform(0.6, 0.9)
            covered_snvs = self.rng.choice(
                window_snvs,
                size=max(1, int(len(window_snvs) * coverage_frac)),
                replace=False
            )
            
            for pos in covered_snvs:
                true_base = source_hap.consensus.get(pos, scenario.ref_alleles[pos])
                
                # Apply sequencing error
                if self.rng.random() < error_rate:
                    alt_bases = [b for b in self.bases if b != true_base]
                    alleles[pos] = self.rng.choice(alt_bases)
                    # Lower quality for errors (simulates real behavior)
                    quals[pos] = max(5, base_quality - 15)
                else:
                    alleles[pos] = true_base
                    quals[pos] = base_quality + self.rng.integers(-3, 4)
            
            reads.append(Read(
                id=read_id,
                contig=scenario.contig_id,
                mapq=mapq,
                alleles=alleles,
                quals=quals,
                sample=timepoint
            ))
        
        window = Window(
            contig=scenario.contig_id,
            start=window_start,
            end=window_end,
            snv_pos=window_snvs,
            ref_alleles=ref_alleles,
            reads=reads,
            sample=timepoint
        )
        
        return window, read_true_haplotypes
    
    def generate_all_windows(
        self,
        scenario: SimulationScenario,
        config: HaplotyperConfig,
        n_reads_per_window: int = 100,
        error_rate: float = 0.001
    ) -> Dict[str, List[Tuple[Window, Dict[str, str]]]]:
        """
        Generate all windows for all timepoints.
        
        Returns:
            {timepoint: [(Window, read_to_haplotype_mapping), ...]}
        """
        results = {}
        
        window_size = config.window_size
        step = window_size // 2  # 50% overlap
        
        for timepoint in scenario.timepoints:
            windows = []
            
            start = 1
            window_idx = 0
            while start < scenario.contig_length:
                end = min(start + window_size, scenario.contig_length)
                
                window, read_map = self.generate_reads_for_window(
                    scenario=scenario,
                    timepoint=timepoint,
                    window_start=start,
                    window_end=end,
                    n_reads=n_reads_per_window,
                    error_rate=error_rate
                )
                
                if window is not None and len(window.snv_pos) >= config.min_snvs_per_window:
                    window.window_idx = window_idx
                    windows.append((window, read_map))
                    window_idx += 1
                
                start += step
            
            results[timepoint] = windows
        
        return results


def create_test_scenarios() -> Dict[str, SimulationScenario]:
    """
    Create a set of standard test scenarios.
    
    Returns dict of scenario_name -> SimulationScenario
    """
    gen = SyntheticDataGenerator(seed=42)
    
    scenarios = {}
    
    # Simple: 2 haplotypes, clear separation
    scenarios['simple_2hap'] = gen.create_scenario(
        name='simple_2hap',
        contig_length=30000,
        n_snvs=60,
        n_haplotypes=2,
        n_timepoints=3,
        include_sweep=False
    )
    
    # Sweep: 2 haplotypes with selective sweep
    scenarios['sweep_2hap'] = gen.create_scenario(
        name='sweep_2hap',
        contig_length=30000,
        n_snvs=60,
        n_haplotypes=2,
        n_timepoints=4,
        include_sweep=True
    )
    
    # Complex: 4 haplotypes, some closely related
    scenarios['complex_4hap'] = gen.create_scenario(
        name='complex_4hap',
        contig_length=50000,
        n_snvs=100,
        n_haplotypes=4,
        n_timepoints=5,
        include_sweep=True
    )
    
    # Low abundance: includes a rare strain
    scenarios['low_abundance'] = gen.create_scenario(
        name='low_abundance',
        contig_length=30000,
        n_snvs=50,
        n_haplotypes=3,
        n_timepoints=4,
        include_sweep=False
    )
    # Manually adjust abundance for one haplotype to be very low
    # Then renormalize so abundances sum to 1.0
    for tp in scenarios['low_abundance'].timepoints:
        scenarios['low_abundance'].true_haplotypes[2].abundance_by_timepoint[tp] = 0.03
        # Renormalize
        total = sum(h.get_abundance(tp) for h in scenarios['low_abundance'].true_haplotypes)
        if total > 0:
            for hap in scenarios['low_abundance'].true_haplotypes:
                hap.abundance_by_timepoint[tp] = hap.get_abundance(tp) / total
    
    return scenarios


if __name__ == "__main__":
    # Test the generator
    scenarios = create_test_scenarios()
    
    for name, scenario in scenarios.items():
        print(f"\n=== Scenario: {name} ===")
        print(f"  Contig: {scenario.contig_id} ({scenario.contig_length} bp)")
        print(f"  SNVs: {scenario.total_snvs()}")
        print(f"  True haplotypes: {scenario.n_true_haplotypes()}")
        print(f"  Timepoints: {scenario.timepoints}")
        
        if scenario.sweep_events:
            print(f"  Sweep events: {scenario.sweep_events}")
        
        print("  Abundance trajectories:")
        for hap in scenario.true_haplotypes:
            abunds = [f"{tp}={hap.get_abundance(tp):.2f}" 
                      for tp in scenario.timepoints]
            print(f"    {hap.id}: {', '.join(abunds)}")
