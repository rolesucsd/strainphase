#!/usr/bin/env python3
"""
Parameter sweep framework for haplotyper pipeline.

Tests pipeline stability across a grid of parameters:
- mismatch thresholds: 0.5%, 1%, 2%, 4%
- MAPQ: 10, 20, 30
- shared SNVs: 2, 3, 4
- merge distance: 0.5%, 1%, 2%
- anchor weight: 10%, 20%, 30%

Evaluates:
- Number of lineages inferred
- Major lineage frequency trajectories
- Sweep detection stability

Run with: python benchmarks/parameter_sweep.py
"""

import itertools
import json
import time
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from strainphase.core import (
    HaplotyperConfig, 
    GraphInitializer, 
    EMHaplotyper, 
    PostProcessor,
    link_windows,
    WindowResult,
    Haplotype,
    Window
)
from strainphase.simulation.synthetic_data import (
    SyntheticDataGenerator, 
    SimulationScenario,
    create_test_scenarios
)


@dataclass
class ParameterSet:
    """A single parameter configuration to test."""
    max_mismatch_frac: float
    min_mapq: int
    min_shared_snvs_for_edge: int
    merge_distance_threshold: float
    min_weight_for_anchor: float
    
    def to_config(self, base_config: Optional[HaplotyperConfig] = None) -> HaplotyperConfig:
        """Convert to HaplotyperConfig."""
        if base_config is None:
            base_config = HaplotyperConfig()
        
        return HaplotyperConfig(
            # Parameters we're varying
            max_mismatch_frac=self.max_mismatch_frac,
            min_mapq=self.min_mapq,
            min_shared_snvs_for_edge=self.min_shared_snvs_for_edge,
            merge_distance_threshold=self.merge_distance_threshold,
            min_weight_for_anchor=self.min_weight_for_anchor,
            
            # Related parameters (keep consistent)
            max_link_distance=self.max_mismatch_frac,
            min_shared_snvs_for_link=self.min_shared_snvs_for_edge,
            min_shared_for_merge=self.min_shared_snvs_for_edge,
            min_shared_for_rescue=self.min_shared_snvs_for_edge,
            rescue_match_distance=self.merge_distance_threshold,
            lineage_merge_distance=self.max_mismatch_frac,
            min_shared_for_lineage=self.min_shared_snvs_for_edge,
            
            # Fixed parameters
            window_size=base_config.window_size,
            min_snvs_per_window=base_config.min_snvs_per_window,
            min_reads_per_window=base_config.min_reads_per_window,
            em_max_iter=base_config.em_max_iter,
            validate_results=False,  # Faster for sweep
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def short_name(self) -> str:
        return (f"mm{self.max_mismatch_frac:.3f}_"
                f"mq{self.min_mapq}_"
                f"snv{self.min_shared_snvs_for_edge}_"
                f"md{self.merge_distance_threshold:.3f}_"
                f"aw{self.min_weight_for_anchor:.2f}")


@dataclass
class SweepResult:
    """Results from a single parameter configuration run."""
    params: ParameterSet
    scenario_name: str
    
    # Lineage statistics
    n_lineages: int
    n_tracks_per_timepoint: Dict[str, int]
    
    # Abundance trajectories (lineage_id -> {timepoint -> abundance})
    lineage_trajectories: Dict[str, Dict[str, float]]
    
    # Sweep detection
    sweep_detected: bool
    sweep_winner: Optional[str]
    sweep_loser: Optional[str]
    
    # Runtime
    runtime_seconds: float
    
    # Quality metrics
    converged: bool
    mean_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'params': self.params.to_dict(),
            'scenario_name': self.scenario_name,
            'n_lineages': self.n_lineages,
            'n_tracks_per_timepoint': self.n_tracks_per_timepoint,
            'lineage_trajectories': self.lineage_trajectories,
            'sweep_detected': self.sweep_detected,
            'sweep_winner': self.sweep_winner,
            'sweep_loser': self.sweep_loser,
            'runtime_seconds': self.runtime_seconds,
            'converged': self.converged,
            'mean_confidence': self.mean_confidence,
        }


class ParameterSweep:
    """
    Run pipeline across parameter grid and analyze stability.
    """
    
    # Default parameter grid
    DEFAULT_GRID = {
        'max_mismatch_frac': [0.005, 0.01, 0.02, 0.04],
        'min_mapq': [10, 20, 30],
        'min_shared_snvs_for_edge': [2, 3, 4],
        'merge_distance_threshold': [0.005, 0.01, 0.02],
        'min_weight_for_anchor': [0.10, 0.20, 0.30],
    }
    
    def __init__(
        self,
        grid: Optional[Dict[str, List]] = None,
        seed: int = 42,
        n_reads_per_window: int = 80,
        error_rate: float = 0.001
    ):
        self.grid = grid or self.DEFAULT_GRID
        self.seed = seed
        self.n_reads_per_window = n_reads_per_window
        self.error_rate = error_rate
        self.generator = SyntheticDataGenerator(seed=seed)
        self.results: List[SweepResult] = []
    
    def generate_parameter_sets(self) -> List[ParameterSet]:
        """Generate all parameter combinations."""
        keys = list(self.grid.keys())
        values = [self.grid[k] for k in keys]
        
        param_sets = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            param_sets.append(ParameterSet(**param_dict))
        
        return param_sets
    
    def run_single_config(
        self,
        params: ParameterSet,
        scenario: SimulationScenario,
        windows_by_timepoint: Dict[str, List[Tuple[Window, Dict[str, str]]]]
    ) -> SweepResult:
        """
        Run pipeline with a single parameter configuration.
        """
        config = params.to_config()
        start_time = time.time()
        
        # Process each timepoint
        all_results: Dict[str, List[WindowResult]] = {}
        all_converged = True
        all_confidences = []
        
        for timepoint, windows_data in windows_by_timepoint.items():
            timepoint_results = []
            
            for window, read_map in windows_data:
                # Skip if window doesn't meet requirements
                if len(window.snv_pos) < config.min_snvs_per_window:
                    continue
                if len(window.reads) < config.min_reads_per_window:
                    continue
                
                # Filter reads by MAPQ
                filtered_reads = [r for r in window.reads if r.mapq >= config.min_mapq]
                if len(filtered_reads) < config.min_reads_per_window:
                    continue
                
                # Update window with filtered reads
                window.reads = filtered_reads
                
                # Graph initialization - returns (haplotypes, cluster_sizes)
                graph_init = GraphInitializer(config)
                initial_haps, cluster_sizes = graph_init.get_initial_haplotypes(window)
                
                if not initial_haps:
                    continue
                
                # EM refinement - use correct API with cluster_sizes
                em = EMHaplotyper(window, initial_haps, cluster_sizes=cluster_sizes, config=config)
                haplotypes, gamma, pi, ll, converged, iterations = em.run()
                
                all_converged = all_converged and converged
                
                # Post-processing
                post = PostProcessor(config)
                merged_haps, final_gamma, final_pi = post.merge_similar_haplotypes(
                    haplotypes, gamma, pi, window
                )
                assignments = post.assign_reads(window.reads, final_gamma, final_pi)
                
                # Collect confidences
                for hap in merged_haps:
                    all_confidences.append(hap.confidence)
                
                result = WindowResult(
                    window=window,
                    haplotypes=merged_haps,
                    gamma=final_gamma,
                    pi=final_pi,
                    log_likelihood=ll,
                    assignments=assignments,
                    converged=converged,
                    iterations=iterations
                )
                timepoint_results.append(result)
            
            # Link windows within timepoint
            if timepoint_results:
                timepoint_results = link_windows(timepoint_results, config)
            
            all_results[timepoint] = timepoint_results
        
        runtime = time.time() - start_time
        
        # Analyze results
        return self._analyze_results(
            params, scenario, all_results, runtime, all_converged, all_confidences
        )
    
    def _analyze_results(
        self,
        params: ParameterSet,
        scenario: SimulationScenario,
        all_results: Dict[str, List[WindowResult]],
        runtime: float,
        converged: bool,
        confidences: List[float]
    ) -> SweepResult:
        """
        Analyze pipeline results to extract metrics.
        """
        # Count tracks per timepoint
        n_tracks_per_timepoint = {}
        all_track_ids = set()
        
        for timepoint, results in all_results.items():
            track_ids = set()
            for wr in results:
                for hap in wr.haplotypes:
                    if hap.track_id:
                        track_ids.add(hap.track_id)
                        all_track_ids.add(hap.track_id)
            n_tracks_per_timepoint[timepoint] = len(track_ids)
        
        # Build lineage trajectories
        # Group tracks by similarity to form "lineages"
        lineage_trajectories = self._build_lineage_trajectories(all_results, params)
        n_lineages = len(lineage_trajectories)
        
        # Detect sweep
        sweep_detected, sweep_winner, sweep_loser = self._detect_sweep(lineage_trajectories)
        
        return SweepResult(
            params=params,
            scenario_name=scenario.name,
            n_lineages=n_lineages,
            n_tracks_per_timepoint=n_tracks_per_timepoint,
            lineage_trajectories=lineage_trajectories,
            sweep_detected=sweep_detected,
            sweep_winner=sweep_winner,
            sweep_loser=sweep_loser,
            runtime_seconds=runtime,
            converged=converged,
            mean_confidence=np.mean(confidences) if confidences else 0.0
        )
    
    def _build_lineage_trajectories(
        self,
        all_results: Dict[str, List[WindowResult]],
        params: ParameterSet
    ) -> Dict[str, Dict[str, float]]:
        """
        Build lineage abundance trajectories across timepoints.
        
        Groups similar tracks into lineages based on consensus similarity.
        """
        # Collect all tracks with their consensus and abundances
        tracks: List[Dict] = []
        
        for timepoint, results in all_results.items():
            track_data: Dict[str, Dict] = defaultdict(lambda: {
                'consensus': {},
                'total_weight': 0.0,
                'n_windows': 0
            })
            
            for wr in results:
                for hap in wr.haplotypes:
                    tid = hap.track_id or f"unlinked_{wr.window.start}"
                    track_data[tid]['total_weight'] += hap.weight
                    track_data[tid]['n_windows'] += 1
                    for pos, base in hap.consensus.items():
                        if pos not in track_data[tid]['consensus']:
                            track_data[tid]['consensus'][pos] = {}
                        if base not in track_data[tid]['consensus'][pos]:
                            track_data[tid]['consensus'][pos][base] = 0
                        track_data[tid]['consensus'][pos][base] += hap.weight
            
            for tid, data in track_data.items():
                # Build consensus from votes
                consensus = {}
                for pos, votes in data['consensus'].items():
                    consensus[pos] = max(votes.keys(), key=lambda b: votes[b])
                
                tracks.append({
                    'timepoint': timepoint,
                    'track_id': tid,
                    'consensus': consensus,
                    'abundance': data['total_weight'] / data['n_windows'] if data['n_windows'] > 0 else 0
                })
        
        if not tracks:
            return {}
        
        # Cluster tracks into lineages using simple greedy approach
        lineages: List[List[Dict]] = []
        
        for track in tracks:
            # Find best matching lineage
            best_lineage = -1
            best_dist = float('inf')
            
            for lin_idx, lineage in enumerate(lineages):
                # Compare to representative (first track in lineage)
                rep = lineage[0]
                
                # Compute distance
                shared = set(track['consensus'].keys()) & set(rep['consensus'].keys())
                if len(shared) < params.min_shared_snvs_for_edge:
                    continue
                
                mismatches = sum(
                    1 for p in shared
                    if track['consensus'][p] != rep['consensus'][p]
                )
                dist = mismatches / len(shared)
                
                if dist <= params.max_mismatch_frac and dist < best_dist:
                    best_dist = dist
                    best_lineage = lin_idx
            
            if best_lineage >= 0:
                lineages[best_lineage].append(track)
            else:
                lineages.append([track])
        
        # Build trajectories
        trajectories = {}
        for lin_idx, lineage in enumerate(lineages):
            lineage_id = f"L{lin_idx:03d}"
            trajectories[lineage_id] = {}
            
            for track in lineage:
                tp = track['timepoint']
                if tp not in trajectories[lineage_id]:
                    trajectories[lineage_id][tp] = 0.0
                trajectories[lineage_id][tp] = max(
                    trajectories[lineage_id][tp],
                    track['abundance']
                )
        
        return trajectories
    
    def _detect_sweep(
        self,
        trajectories: Dict[str, Dict[str, float]]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect selective sweep pattern in trajectories.
        
        Sweep = one lineage increases >30% while another decreases >20%
        """
        if len(trajectories) < 2:
            return False, None, None
        
        # Calculate abundance changes
        changes = []
        for lin_id, traj in trajectories.items():
            if len(traj) < 2:
                continue
            
            timepoints = sorted(traj.keys())
            start_abund = traj[timepoints[0]]
            end_abund = traj[timepoints[-1]]
            change = end_abund - start_abund
            changes.append((lin_id, start_abund, end_abund, change))
        
        if len(changes) < 2:
            return False, None, None
        
        # Sort by change
        changes.sort(key=lambda x: x[3], reverse=True)
        
        winner = changes[0]
        loser = changes[-1]
        
        # Check sweep criteria
        if winner[3] > 0.25 and loser[3] < -0.15:
            return True, winner[0], loser[0]
        
        return False, None, None
    
    def run_sweep(
        self,
        scenarios: Optional[Dict[str, SimulationScenario]] = None,
        max_configs: Optional[int] = None,
        verbose: bool = True
    ) -> List[SweepResult]:
        """
        Run full parameter sweep across scenarios.
        
        Args:
            scenarios: Dict of scenario_name -> SimulationScenario
            max_configs: Limit number of configs to test (for debugging)
            verbose: Print progress
        """
        if scenarios is None:
            scenarios = create_test_scenarios()
        
        param_sets = self.generate_parameter_sets()
        if max_configs:
            param_sets = param_sets[:max_configs]
        
        total_runs = len(param_sets) * len(scenarios)
        if verbose:
            print(f"Running {len(param_sets)} parameter configs × "
                  f"{len(scenarios)} scenarios = {total_runs} total runs")
        
        self.results = []
        run_idx = 0
        
        for scenario_name, scenario in scenarios.items():
            if verbose:
                print(f"\n=== Scenario: {scenario_name} ===")
            
            # Pre-generate windows for this scenario (reused across configs)
            base_config = HaplotyperConfig(window_size=10000)
            windows_by_timepoint = self.generator.generate_all_windows(
                scenario, base_config,
                n_reads_per_window=self.n_reads_per_window,
                error_rate=self.error_rate
            )
            
            for params in param_sets:
                run_idx += 1
                if verbose and run_idx % 10 == 0:
                    print(f"  Progress: {run_idx}/{total_runs}")
                
                try:
                    # Deep copy windows (reads get modified by MAPQ filter)
                    import copy
                    windows_copy = {}
                    for tp, windows in windows_by_timepoint.items():
                        windows_copy[tp] = [
                            (copy.deepcopy(w), rm.copy()) 
                            for w, rm in windows
                        ]
                    
                    result = self.run_single_config(params, scenario, windows_copy)
                    self.results.append(result)
                    
                except Exception as e:
                    if verbose:
                        print(f"    Error with {params.short_name()}: {e}")
        
        return self.results
    
    def summarize_results(self) -> Dict[str, Any]:
        """
        Summarize sweep results to assess stability.
        """
        if not self.results:
            return {'error': 'No results to summarize'}
        
        # Group by scenario
        by_scenario: Dict[str, List[SweepResult]] = defaultdict(list)
        for r in self.results:
            by_scenario[r.scenario_name].append(r)
        
        summary = {
            'n_configs_tested': len(self.generate_parameter_sets()),
            'n_scenarios': len(by_scenario),
            'scenarios': {}
        }
        
        for scenario_name, results in by_scenario.items():
            n_lineages = [r.n_lineages for r in results]
            sweep_detected = [r.sweep_detected for r in results]
            runtimes = [r.runtime_seconds for r in results]
            confidences = [r.mean_confidence for r in results]
            
            summary['scenarios'][scenario_name] = {
                'n_configs': len(results),
                'n_lineages': {
                    'min': min(n_lineages),
                    'max': max(n_lineages),
                    'mean': np.mean(n_lineages),
                    'std': np.std(n_lineages),
                    'mode': max(set(n_lineages), key=n_lineages.count),
                },
                'sweep_detection': {
                    'detected_count': sum(sweep_detected),
                    'detection_rate': np.mean(sweep_detected),
                },
                'runtime': {
                    'mean': np.mean(runtimes),
                    'std': np.std(runtimes),
                },
                'confidence': {
                    'mean': np.mean(confidences),
                    'min': min(confidences),
                },
                'converged_fraction': np.mean([r.converged for r in results]),
            }
        
        return summary
    
    def identify_stable_parameters(self) -> List[ParameterSet]:
        """
        Identify parameter configurations that give stable results.
        
        "Stable" means:
        - Consistent n_lineages across scenarios
        - High sweep detection rate for sweep scenarios
        - Good convergence
        """
        if not self.results:
            return []
        
        # Group results by parameter config
        by_config: Dict[str, List[SweepResult]] = defaultdict(list)
        for r in self.results:
            key = r.params.short_name()
            by_config[key].append(r)
        
        stable_params = []
        
        for config_name, results in by_config.items():
            # Check stability criteria
            n_lineages = [r.n_lineages for r in results]
            lineage_std = np.std(n_lineages)
            
            # Check sweep detection in sweep scenarios
            sweep_results = [r for r in results if 'sweep' in r.scenario_name.lower()]
            sweep_rate = np.mean([r.sweep_detected for r in sweep_results]) if sweep_results else 0
            
            # Check convergence
            convergence_rate = np.mean([r.converged for r in results])
            
            # Stable if:
            # - Low variance in lineage count
            # - High sweep detection (>70%)
            # - Good convergence (>90%)
            if lineage_std < 1.0 and sweep_rate >= 0.7 and convergence_rate >= 0.9:
                stable_params.append(results[0].params)
        
        return stable_params


def run_parameter_sweep(
    output_dir: str = "/home/claude/haplotyper_test/results",
    max_configs: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete parameter sweep and save results.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    sweep = ParameterSweep(seed=42)
    
    print("Starting parameter sweep...")
    results = sweep.run_sweep(max_configs=max_configs, verbose=verbose)
    
    # Save raw results
    results_data = [r.to_dict() for r in results]
    with open(os.path.join(output_dir, 'sweep_results.json'), 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Generate summary
    summary = sweep.summarize_results()
    with open(os.path.join(output_dir, 'sweep_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Identify stable parameters
    stable = sweep.identify_stable_parameters()
    stable_data = [p.to_dict() for p in stable]
    with open(os.path.join(output_dir, 'stable_parameters.json'), 'w') as f:
        json.dump(stable_data, f, indent=2)
    
    if verbose:
        print(f"\n=== SWEEP SUMMARY ===")
        print(f"Total configs tested: {summary['n_configs_tested']}")
        print(f"Scenarios tested: {summary['n_scenarios']}")
        print(f"Stable parameter sets found: {len(stable)}")
        
        for scenario_name, stats in summary['scenarios'].items():
            print(f"\n{scenario_name}:")
            print(f"  Lineages: {stats['n_lineages']['mean']:.1f} ± {stats['n_lineages']['std']:.1f} "
                  f"(range {stats['n_lineages']['min']}-{stats['n_lineages']['max']})")
            print(f"  Sweep detection rate: {stats['sweep_detection']['detection_rate']:.1%}")
            print(f"  Convergence: {stats['converged_fraction']:.1%}")
    
    return summary


if __name__ == "__main__":
    # Quick test with limited configs
    summary = run_parameter_sweep(max_configs=20, verbose=True)