"""Build a longitudinal dataset from real strain assemblies.

Nothing about the genomes is invented. The reference is one of the assemblies,
the strains are the others, and their differences are whatever the organisms
actually differ by. Three things are simulated, because they have to be:

1. **Abundance over time** — see :mod:`spbench.abundance`.
2. **Reads** — Badread, at each strain's abundance times the target coverage.
   Badread is what Strainy's own HiFi benchmark used; running the comparator's
   simulator removes an obvious objection, and it is better calibrated than
   anything written for this repository would be.
3. **Nothing else.** Alignment and variant calling run the same commands the
   real analysis runs, declared in the config, so the BAMs and VCFs the tools
   receive are produced by the pipeline being written about rather than by a
   simulator's idea of one.

Read provenance is exact despite the real aligner: Badread is invoked once per
strain, and its reads are renamed with that strain's id before the per-sample
FASTQs are concatenated. Every read therefore carries its true origin in its
name, and ``truth/read_origins.tsv`` is a fact rather than an inference.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from spbench.abundance import AbundanceConfig, build_trajectories
from spbench.formats import (
    ABUNDANCE_COLUMNS,
    READ_ORIGINS_COLUMNS,
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

#: Defaults chosen to match common practice for PacBio HiFi metagenomes.
#: Override them in the config with the exact commands your own workflow runs —
#: the point of this being a template is that the benchmark's BAMs and VCFs
#: should be produced by your pipeline, not by an approximation of it.
DEFAULT_ALIGN_CMD = (
    "minimap2 -ax map-hifi -t {threads} --secondary=no {reference} {fastq} "
    "| samtools sort -@ {threads} -o {bam} - && samtools index {bam}"
)
DEFAULT_CALL_CMD = (
    "run_clair3.sh --bam_fn={bam} --ref_fn={reference} --threads={threads} "
    "--platform=hifi --model_path=${{CLAIR3_MODEL_PATH}} --output={vcf_dir} "
    "--include_all_ctgs --no_phasing_for_fa --sample_name={sample}"
)
DEFAULT_READS_CMD = (
    "badread simulate --reference {assembly} --quantity {quantity}x "
    "--error_model pacbio2021 --qscore_model pacbio2021 "
    "--identity 30,3 --length {mean_length},{length_sd} --seed {seed}"
)


@dataclass
class SimConfig:
    """Everything that determines a dataset."""

    name: str = "dataset"
    seed: int = 0

    #: Directory or glob of assemblies for this strain group. Required.
    assemblies: str = ""

    # Sequencing
    coverage: float = 60.0  # total across all strains
    mean_read_length: int = 15_000
    read_length_sd: int = 4_000

    # Timecourse
    n_timepoints: int = 6
    abundance: dict = field(default_factory=dict)  # AbundanceConfig overrides

    # Commands. Placeholders are filled by name; unknown placeholders are an
    # error rather than a silently empty string.
    reads_cmd: str = DEFAULT_READS_CMD
    align_cmd: str = DEFAULT_ALIGN_CMD
    call_cmd: str = DEFAULT_CALL_CMD
    #: Path, relative to the per-sample variant directory, of the VCF the caller
    #: produces. Clair3 writes merge_output.vcf.gz; set this to match yours.
    call_output: str = "merge_output.vcf.gz"
    #: Optional cohort step run ONCE across all per-sample VCFs. When set, its
    #: output becomes the VCF handed to every tool for every sample.
    #:
    #: This exists because the production pipeline works that way: sites
    #: polymorphic in any timepoint are genotyped in all of them, so a strain
    #: that sweeps to fixation or drops out is still reported. It also matters
    #: for fairness - a union site list is a multi-sample advantage, and it has
    #: to be given to every tool or to none. Placeholders: {vcfs} {vcf}
    union_cmd: str = ""

    minimap2_asm_preset: str = "asm5"

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _run(command: str, log_path: Path, env: dict | None = None) -> None:
    """Run a shell command, teeing output to a log, failing loudly."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("$ %s", command)
    with open(log_path, "w") as handle:
        handle.write(f"$ {command}\n\n")
        handle.flush()
        proc = subprocess.run(
            command, shell=True, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False
        )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(f"command failed (exit {proc.returncode}):\n{command}\n{tail}")


def _format(template: str, **values) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        raise KeyError(
            f"command template uses unknown placeholder {exc}. Available: "
            f"{sorted(values)}"
        ) from exc


#: Not worth reporting as "missing": either always present or not a program.
_SHELL_NOISE = frozenset({"bash", "sh", "cd", "set", "export", "echo", "true", "cat"})


def required_binaries(config: SimConfig) -> list[str]:
    """Every program the configured commands invoke.

    Each command is split on the shell operators that start a new program
    (``|``, ``&&``, ``;``) and the leading token of each segment is taken. This
    is a best-effort read of a shell string, not a parse — it is here so a typo
    or an unactivated environment surfaces from ``spbench check-env`` rather
    than from a failed cluster job, and it errs toward reporting too much.
    """
    binaries: set[str] = set()
    for template in (config.reads_cmd, config.align_cmd, config.call_cmd, config.union_cmd):
        if not template:
            continue
        for separator in ("&&", "||", ";", "|"):
            template = template.replace(separator, "\n")
        for segment in template.split("\n"):
            try:
                tokens = shlex.split(segment)
            except ValueError:
                continue
            if not tokens:
                continue
            name = tokens[0]
            # Skip paths, variable expansions and shell plumbing.
            if name.startswith("-") or name in _SHELL_NOISE:
                continue
            if "/" in name or "{" in name or "$" in name:
                continue
            binaries.add(name)
    return sorted(binaries)


def check_environment(config: SimConfig) -> list[str]:
    """Return the configured binaries that are missing from PATH."""
    return [b for b in required_binaries(config) if shutil.which(b) is None]


def simulate_reads(
    config: SimConfig,
    sample: str,
    group,
    abundance: dict[tuple[str, str], float],
    outdir: Path,
    rng: np.random.Generator,
) -> tuple[Path, list[dict]]:
    """One FASTQ per timepoint, plus exact read origins.

    Badread runs once per strain at ``abundance x coverage``. A strain at zero
    abundance is skipped entirely, so an absent strain contributes no reads at
    all — which is what makes the colonisation and clearance archetypes a real
    test rather than a low-abundance one.
    """
    fastq_dir = outdir / "fastq"
    fastq_dir.mkdir(parents=True, exist_ok=True)
    combined = fastq_dir / f"{sample}.fastq.gz"
    origins: list[dict] = []

    if combined.exists():
        combined.unlink()

    with gzip.open(combined, "wt") as out:
        for strain_id in group.strain_ids:
            fraction = abundance.get((sample, strain_id), 0.0)
            quantity = fraction * config.coverage
            if quantity <= 0.01:
                continue

            per_strain = fastq_dir / f"{sample}.{strain_id}.fastq"
            command = _format(
                config.reads_cmd,
                assembly=group.assemblies[strain_id],
                quantity=f"{quantity:.3f}",
                mean_length=config.mean_read_length,
                length_sd=config.read_length_sd,
                seed=int(rng.integers(1, 2**31 - 1)),
            )
            _run(f"{command} > {per_strain}", outdir / "logs" / f"reads.{sample}.{strain_id}.log")

            n = 0
            with open(per_strain) as handle:
                for i, line in enumerate(handle):
                    if i % 4 == 0:
                        n += 1
                        read_id = f"{sample}|{strain_id}|{n:07d}"
                        out.write(f"@{read_id}\n")
                        origins.append(
                            {
                                "sample": sample,
                                "read_id": read_id,
                                "contig": "",
                                "strain_id": strain_id,
                                "start": "",
                                "end": "",
                            }
                        )
                    else:
                        out.write(line)
            per_strain.unlink()
            logger.info("  %s / %s: %d reads at %.2fx", sample, strain_id, n, quantity)

    return combined, origins


def simulate(config: SimConfig, outdir: str | Path, threads: int = 4) -> Path:
    """Build one complete dataset. Returns the dataset root."""
    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)

    missing = check_environment(config)
    if missing:
        raise RuntimeError(
            f"missing required binaries: {missing}. Install the benchmark "
            f"environment (benchmark/envs/spbench.yml) or point the *_cmd "
            f"settings at what your workflow uses."
        )

    rng = np.random.default_rng(config.seed)

    # 1. Real strains, one of them the reference.
    assemblies = discover_assemblies(config.assemblies)
    group = build_group(config.name, assemblies, rng, preset=config.minimap2_asm_preset)
    reference_contigs = read_fasta(group.reference_path)
    write_fasta(outdir / "reference.fasta", reference_contigs)

    sites = union_sites(group)
    alleles = haplotype_alleles(group, sites)

    # 2. Abundance over time. Only strains other than the reference vary? No -
    #    the reference strain is a member of the community like any other, and
    #    excluding it would make the reference-allele haplotype untestable.
    abundance_config = AbundanceConfig(n_timepoints=config.n_timepoints, **config.abundance)
    abundance, archetypes = build_trajectories(group.strain_ids, rng, abundance_config)
    samples = [f"T{i + 1}" for i in range(config.n_timepoints)]

    # 3. Reads, then the real alignment and calling commands.
    all_origins: list[dict] = []
    per_sample_vcfs: list[Path] = []
    for sample in samples:
        fastq, origins = simulate_reads(config, sample, group, abundance, outdir, rng)
        all_origins.extend(origins)

        bam = outdir / "bam" / f"{sample}.bam"
        bam.parent.mkdir(parents=True, exist_ok=True)
        _run(
            _format(
                config.align_cmd,
                reference=outdir / "reference.fasta",
                fastq=fastq,
                bam=bam,
                threads=threads,
                sample=sample,
            ),
            outdir / "logs" / f"align.{sample}.log",
        )

        vcf_dir = outdir / "variants" / sample
        vcf_dir.mkdir(parents=True, exist_ok=True)
        _run(
            _format(
                config.call_cmd,
                reference=outdir / "reference.fasta",
                bam=bam,
                vcf_dir=vcf_dir,
                vcf=vcf_dir / "calls.vcf.gz",
                threads=threads,
                sample=sample,
            ),
            outdir / "logs" / f"call.{sample}.log",
        )

        produced = vcf_dir / config.call_output
        target = outdir / "variants" / f"{sample}.vcf.gz"
        if not produced.exists():
            raise FileNotFoundError(
                f"variant caller did not produce {produced}. Set `call_output` "
                f"to the file your caller writes."
            )
        shutil.copy(produced, target)
        _index_vcf(target)
        per_sample_vcfs.append(target)

    if config.union_cmd:
        union = outdir / "variants" / "union_sites.vcf.gz"
        _run(
            _format(
                config.union_cmd,
                vcfs=" ".join(str(v) for v in per_sample_vcfs),
                vcf=union,
                reference=outdir / "reference.fasta",
                threads=threads,
            ),
            outdir / "logs" / "union.log",
        )
        if not union.exists():
            raise FileNotFoundError(f"union_cmd did not produce {union}")
        # Every sample is handed the same site list, exactly as in production -
        # and identically for every tool, so the union is not an advantage one
        # method gets and the others do not.
        for sample in samples:
            target = outdir / "variants" / f"{sample}.vcf.gz"
            target.unlink(missing_ok=True)
            Path(str(target) + ".tbi").unlink(missing_ok=True)
            shutil.copy(union, target)
            _index_vcf(target)

    # 4. Truth tables.
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
    write_table(truth_dir / "read_origins.tsv", READ_ORIGINS_COLUMNS, all_origins)

    manifest = {
        "name": config.name,
        "spbench_version": __import__("spbench").__version__,
        "config": asdict(config),
        "config_fingerprint": config.fingerprint(),
        "samples": samples,
        "contigs": {name: len(seq) for name, seq in reference_contigs.items()},
        "reference_strain": group.reference_id,
        "strains": group.strain_ids,
        "archetypes": archetypes,
        "n_variant_sites": sum(len(v) for v in sites.values()),
        "assemblies": {k: str(v) for k, v in group.assemblies.items()},
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info(
        "%s: %d strains (ref %s), %d sites, %d timepoints -> %s",
        config.name,
        len(group.strain_ids),
        group.reference_id,
        manifest["n_variant_sites"],
        len(samples),
        outdir,
    )
    return outdir


def _index_vcf(path: Path) -> None:
    import pysam

    try:
        pysam.tabix_index(str(path), preset="vcf", force=True)
    except Exception:  # noqa: BLE001 - already indexed is not an error
        if not Path(str(path) + ".tbi").exists():
            raise
