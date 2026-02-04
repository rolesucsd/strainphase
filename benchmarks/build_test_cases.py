#!/usr/bin/env python3
"""
build_test_cases.py

Build 84 test case directories from dereplicated MAG collection.

Inputs:
    --taxonomy      taxonomy.tsv (derep_mag_id -> species)
    --ani           ani_to_derep.tsv (isolate -> nearest_derep -> ani -> species)
    --fasta-dir     Directory containing all FASTA files
    --species-list  Text file with 20 target species (one per line)
    --output-dir    Output directory (default: Strains/)
    --seed          Random seed for reproducibility

Outputs:
    Strains/{condition}_rep{NN}/ directories, each containing:
        - reference.fasta (one strain designated as reference)
        - strain_XX.fasta (remaining strains)
"""

import argparse
import random
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Condition definitions
# =============================================================================

CONDITIONS = {
    # Axis A: Similarity gradient (4 strains, single species)
    "A1": {"n_strains": 4, "n_species": 1, "ani_min": 97.0, "ani_max": 98.5, "n_reps": 10},
    "A2": {"n_strains": 4, "n_species": 1, "ani_min": 99.0, "ani_max": 99.4, "n_reps": 10},
    "A3": {"n_strains": 4, "n_species": 1, "ani_min": 99.5, "ani_max": 100.0, "n_reps": 20},
    
    # Axis B: Complexity gradient (single species, ≥99.5% ANI)
    "B1": {"n_strains": 2, "n_species": 1, "ani_min": 99.5, "ani_max": 100.0, "n_reps": 16},
    "B3": {"n_strains": 6, "n_species": 1, "ani_min": 99.5, "ani_max": 100.0, "n_reps": 16},
    
    # Axis C: Multi-species (3 species × 2 strains, within-species ≥99.5%, between ≤95%)
    "C1": {
        "n_strains": 6, 
        "n_species": 3, 
        "strains_per_species": 2,
        "ani_within_min": 99.5, 
        "ani_between_max": 95.0, 
        "n_reps": 12
    },
}


# =============================================================================
# Data loading
# =============================================================================

def load_taxonomy(path: str) -> pd.DataFrame:
    """Load taxonomy assignments: derep_mag_id -> species."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.lower().strip() for c in df.columns]
    required = {"derep_mag_id", "species"}
    if not required.issubset(df.columns):
        raise ValueError(f"taxonomy.tsv must have columns: {required}")
    return df


def load_ani(path: str) -> pd.DataFrame:
    """Load ANI table: isolate -> nearest_derep -> ani -> species."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.lower().strip() for c in df.columns]
    required = {"isolate_id", "nearest_derep_mag", "ani_to_derep", "species"}
    if not required.issubset(df.columns):
        raise ValueError(f"ani_to_derep.tsv must have columns: {required}")
    return df


def load_species_list(path: str) -> set:
    """Load target species list."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


# =============================================================================
# Strain selection logic
# =============================================================================

def compute_pairwise_ani_proxy(isolates: list, ani_df: pd.DataFrame) -> dict:
    """
    Compute approximate pairwise ANI between isolates.
    
    Since we only have ANI-to-derep, we estimate pairwise ANI as:
        ani(A, B) ≈ min(ani(A, derep), ani(B, derep)) 
                    if same derep cluster
        ani(A, B) ≈ max of their ANI-to-derep minus some penalty
                    if different derep clusters (conservative: assume low)
    
    For isolates in the SAME derep cluster (same nearest_derep_mag):
        Their pairwise ANI is bounded below by the min of their ANIs to the derep.
        
    Returns: dict of (isolate_a, isolate_b) -> estimated_ani
    """
    # Build lookup: isolate -> (derep, ani_to_derep)
    lookup = {}
    for _, row in ani_df.iterrows():
        lookup[row["isolate_id"]] = (row["nearest_derep_mag"], row["ani_to_derep"])
    
    pairwise = {}
    for i, iso_a in enumerate(isolates):
        for iso_b in isolates[i+1:]:
            if iso_a not in lookup or iso_b not in lookup:
                continue
            derep_a, ani_a = lookup[iso_a]
            derep_b, ani_b = lookup[iso_b]
            
            if derep_a == derep_b:
                # Same cluster: estimate as min of both ANIs to derep
                # (conservative lower bound on their true pairwise ANI)
                est_ani = min(ani_a, ani_b)
            else:
                # Different clusters: assume they're more distant
                # Use a conservative estimate (lower than both)
                est_ani = min(ani_a, ani_b) - 2.0  # penalty for different clusters
            
            pairwise[(iso_a, iso_b)] = est_ani
            pairwise[(iso_b, iso_a)] = est_ani
    
    return pairwise


def select_strain_set_single_species(
    species: str,
    ani_df: pd.DataFrame,
    n_strains: int,
    ani_min: float,
    ani_max: float,
    used_sets: set,
    rng: random.Random,
    max_attempts: int = 100,
) -> list | None:
    """
    Select n_strains from a single species meeting ANI constraints.
    
    All pairwise ANIs must be in [ani_min, ani_max].
    Returns None if no valid set found.
    """
    # Filter to target species
    species_df = ani_df[ani_df["species"] == species].copy()
    
    if len(species_df) < n_strains:
        logger.warning(f"Species {species} has only {len(species_df)} isolates, need {n_strains}")
        return None
    
    isolates = species_df["isolate_id"].tolist()
    
    # Pre-filter: only keep isolates whose ANI-to-derep is in range
    # (if ani_to_derep < ani_min, they can't form valid pairs)
    if ani_min > 97.0:  # Only filter for high-ANI conditions
        isolates = [
            iso for iso in isolates 
            if species_df[species_df["isolate_id"] == iso]["ani_to_derep"].values[0] >= ani_min - 1.0
        ]
    
    if len(isolates) < n_strains:
        logger.warning(f"Species {species} has only {len(isolates)} isolates after ANI pre-filter")
        return None
    
    # Compute pairwise ANI estimates
    pairwise_ani = compute_pairwise_ani_proxy(isolates, species_df)
    
    # Greedy selection with restarts
    for attempt in range(max_attempts):
        # Start with a random isolate
        seed_isolate = rng.choice(isolates)
        selected = [seed_isolate]
        candidates = [iso for iso in isolates if iso != seed_isolate]
        rng.shuffle(candidates)
        
        for candidate in candidates:
            if len(selected) >= n_strains:
                break
            
            # Check if candidate is compatible with all selected
            compatible = True
            for sel in selected:
                pair_ani = pairwise_ani.get((candidate, sel), 0.0)
                if not (ani_min <= pair_ani <= ani_max):
                    compatible = False
                    break
            
            if compatible:
                selected.append(candidate)
        
        if len(selected) == n_strains:
            # Check if this exact set was already used
            set_key = tuple(sorted(selected))
            if set_key not in used_sets:
                used_sets.add(set_key)
                return selected
    
    return None


def select_strain_set_multi_species(
    species_list: list,
    ani_df: pd.DataFrame,
    n_species: int,
    strains_per_species: int,
    ani_within_min: float,
    used_sets: set,
    rng: random.Random,
    max_attempts: int = 50,
) -> list | None:
    """
    Select strains from multiple species.
    
    Within-species pairs must have ANI >= ani_within_min.
    Returns flat list of isolate IDs.
    """
    for attempt in range(max_attempts):
        # Pick n_species random species
        selected_species = rng.sample(species_list, n_species)
        
        all_strains = []
        success = True
        
        for sp in selected_species:
            # Select strains_per_species from this species
            sp_set = select_strain_set_single_species(
                species=sp,
                ani_df=ani_df,
                n_strains=strains_per_species,
                ani_min=ani_within_min,
                ani_max=100.0,
                used_sets=set(),  # Don't track per-species, only full set
                rng=rng,
                max_attempts=20,
            )
            if sp_set is None:
                success = False
                break
            all_strains.extend(sp_set)
        
        if success and len(all_strains) == n_species * strains_per_species:
            set_key = tuple(sorted(all_strains))
            if set_key not in used_sets:
                used_sets.add(set_key)
                return all_strains
    
    return None


# =============================================================================
# Directory creation
# =============================================================================

def create_test_case_dir(
    output_dir: Path,
    condition: str,
    rep: int,
    strain_ids: list,
    fasta_dir: Path,
    rng: random.Random,
):
    """
    Create a test case directory with FASTA files.
    
    - One strain is randomly designated as reference.fasta
    - Others are named strain_XX.fasta
    """
    case_dir = output_dir / f"{condition}_rep{rep:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and designate first as reference
    strains = list(strain_ids)
    rng.shuffle(strains)
    
    for i, strain_id in enumerate(strains):
        # Find FASTA file (try common extensions)
        src = None
        for ext in [".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz"]:
            candidate = fasta_dir / f"{strain_id}{ext}"
            if candidate.exists():
                src = candidate
                break
        
        if src is None:
            raise FileNotFoundError(f"FASTA not found for {strain_id} in {fasta_dir}")
        
        # Destination name
        if i == 0:
            dst_name = "reference.fasta"
        else:
            dst_name = f"strain_{i:02d}.fasta"
        
        dst = case_dir / dst_name
        
        # Copy (decompress if needed)
        if str(src).endswith(".gz"):
            import gzip
            with gzip.open(src, "rt") as f_in, open(dst, "w") as f_out:
                f_out.write(f_in.read())
        else:
            shutil.copy(src, dst)
    
    # Write manifest
    manifest_path = case_dir / "manifest.tsv"
    with open(manifest_path, "w") as f:
        f.write("role\toriginal_id\tfilename\n")
        for i, strain_id in enumerate(strains):
            role = "reference" if i == 0 else f"strain_{i:02d}"
            fname = "reference.fasta" if i == 0 else f"strain_{i:02d}.fasta"
            f.write(f"{role}\t{strain_id}\t{fname}\n")
    
    logger.info(f"Created {case_dir.name} with {len(strains)} strains")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build test case directories for Strainphase benchmarking")
    parser.add_argument("--taxonomy", required=True, help="taxonomy.tsv file")
    parser.add_argument("--ani", required=True, help="ani_to_derep.tsv file")
    parser.add_argument("--fasta-dir", required=True, help="Directory with FASTA files")
    parser.add_argument("--species-list", required=True, help="Text file with target species")
    parser.add_argument("--output-dir", default="Strains", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Initialize RNG
    rng = random.Random(args.seed)
    
    # Load data
    logger.info("Loading input files...")
    taxonomy_df = load_taxonomy(args.taxonomy)
    ani_df = load_ani(args.ani)
    target_species = load_species_list(args.species_list)
    fasta_dir = Path(args.fasta_dir)
    output_dir = Path(args.output_dir)
    
    logger.info(f"Loaded {len(taxonomy_df)} taxonomy entries")
    logger.info(f"Loaded {len(ani_df)} ANI entries")
    logger.info(f"Target species: {len(target_species)}")
    
    # Filter ANI table to target species
    ani_df = ani_df[ani_df["species"].isin(target_species)]
    logger.info(f"After species filter: {len(ani_df)} isolates")
    
    # Track which strain sets have been used (avoid duplicates)
    used_sets = set()
    
    # Process each condition
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort species by isolate count (prefer species with more isolates)
    species_counts = ani_df.groupby("species").size().to_dict()
    available_species = sorted(
        [sp for sp in target_species if sp in species_counts],
        key=lambda x: species_counts.get(x, 0),
        reverse=True
    )
    logger.info(f"Species with isolates: {len(available_species)}")
    
    total_created = 0
    failed_conditions = []
    
    for cond_name, cond_spec in CONDITIONS.items():
        logger.info(f"\n=== Condition {cond_name} ===")
        
        n_reps = cond_spec["n_reps"]
        created = 0
        
        for rep in range(1, n_reps + 1):
            # Select strain set based on condition type
            if cond_spec["n_species"] == 1:
                # Single-species condition: try each species until success
                strain_set = None
                for species in available_species:
                    strain_set = select_strain_set_single_species(
                        species=species,
                        ani_df=ani_df,
                        n_strains=cond_spec["n_strains"],
                        ani_min=cond_spec["ani_min"],
                        ani_max=cond_spec["ani_max"],
                        used_sets=used_sets,
                        rng=rng,
                    )
                    if strain_set is not None:
                        break
            else:
                # Multi-species condition
                strain_set = select_strain_set_multi_species(
                    species_list=available_species,
                    ani_df=ani_df,
                    n_species=cond_spec["n_species"],
                    strains_per_species=cond_spec["strains_per_species"],
                    ani_within_min=cond_spec["ani_within_min"],
                    used_sets=used_sets,
                    rng=rng,
                )
            
            if strain_set is None:
                logger.warning(f"Failed to create {cond_name}_rep{rep:02d}")
                continue
            
            # Create directory
            try:
                create_test_case_dir(
                    output_dir=output_dir,
                    condition=cond_name,
                    rep=rep,
                    strain_ids=strain_set,
                    fasta_dir=fasta_dir,
                    rng=rng,
                )
                created += 1
                total_created += 1
            except Exception as e:
                logger.error(f"Error creating {cond_name}_rep{rep:02d}: {e}")
        
        logger.info(f"Condition {cond_name}: created {created}/{n_reps} replicates")
        if created < n_reps:
            failed_conditions.append((cond_name, n_reps - created))
    
    # Summary
    logger.info(f"\n=== SUMMARY ===")
    logger.info(f"Total test cases created: {total_created}/84")
    if failed_conditions:
        logger.warning(f"Failed conditions: {failed_conditions}")
    
    # Write summary manifest
    summary_path = output_dir / "test_cases_summary.tsv"
    with open(summary_path, "w") as f:
        f.write("condition\treplicate\tpath\tn_strains\n")
        for case_dir in sorted(output_dir.iterdir()):
            if case_dir.is_dir() and case_dir.name != "__pycache__":
                parts = case_dir.name.rsplit("_rep", 1)
                if len(parts) == 2:
                    cond = parts[0]
                    rep = int(parts[1])
                    n_strains = len(list(case_dir.glob("*.fasta")))
                    f.write(f"{cond}\t{rep}\t{case_dir}\t{n_strains}\n")
    
    logger.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
