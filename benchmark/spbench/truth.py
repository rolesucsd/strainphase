"""Assemblies in; reference, ground truth, and a read-simulation plan out.

This is the first Snakemake rule's payload. It is deterministic given the seed,
and it writes everything downstream needs:

``reference.fasta``      one of the assemblies, drawn by seed
``truth/sites.tsv``      every position polymorphic across the group
``truth/strains.tsv``    each strain's true haplotype over those sites
``truth/abundance.tsv``  strain x timepoint, sums to 1 per timepoint
``plan.tsv``             sample, strain, assembly path, coverage — what to
                         simulate. Separating the plan from the simulation is
                         what lets Snakemake see the read step as one rule per
                         sample instead of needing a checkpoint.
``manifest.json``        the config, the reference choice, the archetype per
                         strain, and a fingerprint
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from spbench.abundance import AbundanceConfig, build_trajectories
from spbench.formats import (
    ABUNDANCE_COLUMNS,
    SITES_COLUMNS,
    STRAINS_COLUMNS,
    encode_alleles,
    write_table,
)
from spbench.strains import (
    build_group,
    discover_assemblies,
    haplotype_alleles,
    read_fasta,
    union_sites,
    write_fasta,
)

logger = logging.getLogger(__name__)

PLAN_COLUMNS = ["sample", "strain_id", "assembly", "coverage"]


def build(
    assemblies: str,
    outdir: str | Path,
    seed: int = 0,
    n_timepoints: int = 6,
    coverage: float = 60.0,
    asm_preset: str = "asm5",
    abundance_overrides: dict | None = None,
) -> Path:
    """Build reference, ground truth and the simulation plan for one group."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    group = build_group(outdir.name, discover_assemblies(assemblies), rng, preset=asm_preset)
    reference_contigs = read_fasta(group.reference_path)
    write_fasta(outdir / "reference.fasta", reference_contigs)

    sites = union_sites(group)
    alleles = haplotype_alleles(group, sites)

    abundance_config = AbundanceConfig(n_timepoints=n_timepoints, **(abundance_overrides or {}))
    abundance, archetypes = build_trajectories(group.strain_ids, rng, abundance_config)
    samples = [f"T{i + 1}" for i in range(n_timepoints)]

    truth_dir = outdir / "truth"
    write_table(
        truth_dir / "sites.tsv",
        SITES_COLUMNS,
        (
            {"contig": contig, "pos": pos, "ref": ref, "alt": alt}
            for contig, contig_sites in sites.items()
            for pos, (ref, alt) in sorted(contig_sites.items())
        ),
    )
    write_table(
        truth_dir / "strains.tsv",
        STRAINS_COLUMNS,
        (
            {
                "strain_id": strain_id,
                "contig": contig,
                "n_sites": len(per_contig),
                "alleles": encode_alleles(per_contig),
            }
            for strain_id, contigs in alleles.items()
            for contig, per_contig in contigs.items()
        ),
    )
    write_table(
        truth_dir / "abundance.tsv",
        ABUNDANCE_COLUMNS,
        (
            {"sample": sample, "strain_id": strain_id, "abundance": f"{value:.6f}"}
            for (sample, strain_id), value in sorted(abundance.items())
        ),
    )

    # The plan carries absolute per-strain depth, not fraction, because that is
    # what a read simulator is asked for. A strain at zero abundance is omitted
    # entirely rather than listed at 0x, so an absent strain contributes no
    # reads at all and stays a false-positive test.
    plan_rows = [
        {
            "sample": sample,
            "strain_id": strain_id,
            "assembly": str(group.assemblies[strain_id]),
            "coverage": f"{abundance[(sample, strain_id)] * coverage:.4f}",
        }
        for sample in samples
        for strain_id in group.strain_ids
        if abundance[(sample, strain_id)] * coverage > 0.01
    ]
    write_table(outdir / "plan.tsv", PLAN_COLUMNS, plan_rows)

    settings = {
        "assemblies": assemblies,
        "seed": seed,
        "n_timepoints": n_timepoints,
        "coverage": coverage,
        "asm_preset": asm_preset,
        "abundance": abundance_overrides or {},
    }
    manifest = {
        "name": outdir.name,
        "spbench_version": __import__("spbench").__version__,
        "settings": settings,
        "fingerprint": hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()[:16],
        "samples": samples,
        "contigs": {name: len(seq) for name, seq in reference_contigs.items()},
        "reference_strain": group.reference_id,
        "strains": group.strain_ids,
        "archetypes": archetypes,
        "n_variant_sites": sum(len(v) for v in sites.values()),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info(
        "%s: %d strains (reference %s), %d sites, %d timepoints, %d read jobs",
        outdir.name,
        len(group.strain_ids),
        group.reference_id,
        manifest["n_variant_sites"],
        len(samples),
        len(plan_rows),
    )
    return outdir
