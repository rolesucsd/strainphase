"""Adapters for third-party tools invoked as external binaries.

Verification status
-------------------
These adapters shell out to tools that cannot be installed in this repository's
CI (they are Rust/C++ binaries distributed through bioconda). The parsers below
were written against each tool's documented output format, and the Floria parser
additionally matches the format this project's earlier private converter parsed
successfully.

**Run ``spbench check-tools`` after installing them.** It executes each adapter
on the smoke dataset and reports whether the output parsed, so a format drift in
a newer release surfaces as a clear error before you spend cluster time on a
full run. Any tool whose output does not parse produces a ``failed`` row with
the command log attached - never a silent zero.

Every tool is run at its published defaults for its read technology. Extra
arguments can be supplied per tool from the benchmark config, and whatever was
actually run is recorded verbatim in the results so the parameters behind any
number are recoverable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from spbench.adapters.base import Adapter, ToolInfo, run_command
from spbench.dataset import Dataset
from spbench.formats import UNASSIGNED

logger = logging.getLogger(__name__)


def _bam_to_fastq(bam_path: Path, fastq_path: Path) -> Path:
    """Write a FASTQ from a BAM using pysam, so no samtools binary is needed."""
    import pysam

    if fastq_path.exists():
        return fastq_path
    fastq_path.parent.mkdir(parents=True, exist_ok=True)
    with pysam.AlignmentFile(str(bam_path), "rb") as bam, open(fastq_path, "w") as out:
        for aln in bam.fetch(until_eof=True):
            if aln.is_secondary or aln.is_supplementary or aln.query_sequence is None:
                continue
            qual = aln.qual or ("I" * len(aln.query_sequence))
            out.write(f"@{aln.query_name}\n{aln.query_sequence}\n+\n{qual}\n")
    return fastq_path


def _tagged_bam_partition(bam_path: Path, sample: str, tags=("HP", "YC")) -> dict[tuple[str, str], str]:
    """Read a haplotagged BAM into a read partition.

    Tries each tag in order. ``HP:i`` is the SAM convention used by whatshap and
    devider's ``haplotag_bam``; Strainy documents ``YC``, which carries a colour
    string that is constant within a cluster and therefore serves as a cluster
    label.
    """
    import pysam

    assignments: dict[tuple[str, str], str] = {}
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for aln in bam.fetch(until_eof=True):
            if aln.is_secondary or aln.is_supplementary:
                continue
            label = UNASSIGNED
            for tag in tags:
                if aln.has_tag(tag):
                    label = f"{tag}_{aln.get_tag(tag)}"
                    break
            assignments[(sample, aln.query_name)] = label
    return assignments


class FloriaAdapter(Adapter):
    info = ToolInfo(
        name="floria",
        designed_for=(
            "Single-sample strain haplotyping of metagenomes from short or long "
            "reads, via minimum-error-correction read clustering and a "
            "strain-preserving network flow model. Emits vartigs and the reads "
            "supporting each."
        ),
        citation="Shaw, Boucher, Yu, Noyes & Li, Bioinformatics 40(Suppl 1), 2024",
        supports_cross_sample_ids=False,
        multi_sample=False,
        requires=["floria"],
    )

    def __init__(self, extra_args: list[str] | None = None) -> None:
        self.extra_args = list(extra_args or [])

    def partition(
        self, dataset: Dataset, workdir: Path, threads: int
    ) -> dict[tuple[str, str], str]:
        assignments: dict[tuple[str, str], str] = {}
        for sample in dataset.samples:
            outdir = workdir / sample
            run_command(
                [
                    "floria",
                    "-b", str(dataset.bams[sample]),
                    "-v", str(dataset.vcfs[sample]),
                    "-r", str(dataset.reference),
                    "-o", str(outdir),
                    "-t", str(threads),
                    "--overwrite",
                    *self.extra_args,
                ],
                workdir / f"{sample}.log",
            )
            assignments.update(self._parse(outdir, sample))
        return assignments

    @staticmethod
    def _parse(outdir: Path, sample: str) -> dict[tuple[str, str], str]:
        """Read every ``*.haplosets`` file: ``>HAP<n>...`` header lines followed
        by one read name per line."""
        assignments: dict[tuple[str, str], str] = {}
        files = sorted(outdir.rglob("*.haplosets"))
        if not files:
            raise RuntimeError(
                f"floria produced no *.haplosets under {outdir}; see the run log"
            )
        for path in files:
            contig = path.stem
            current: str | None = None
            for line in path.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    header = line[1:].split("\t")[0]
                    current = f"{contig}:{header.split('.')[0]}"
                elif current is not None:
                    assignments[(sample, line.split()[0])] = current
        return assignments


class StrainyAdapter(Adapter):
    info = ToolInfo(
        name="strainy",
        designed_for=(
            "Single-sample phasing and assembly of strain haplotypes from "
            "long-read metagenomes, operating on an assembly graph or a linear "
            "reference and producing strain unitigs."
        ),
        citation="Kazantseva, Donmez, Frolova, Pop & Kolmogorov, Nature Methods, 2024",
        supports_cross_sample_ids=False,
        multi_sample=False,
        requires=["strainy"],
    )

    def __init__(self, mode: str = "hifi", extra_args: list[str] | None = None) -> None:
        self.mode = mode
        self.extra_args = list(extra_args or [])

    def partition(
        self, dataset: Dataset, workdir: Path, threads: int
    ) -> dict[tuple[str, str], str]:
        assignments: dict[tuple[str, str], str] = {}
        for sample in dataset.samples:
            outdir = workdir / sample
            fastq = _bam_to_fastq(dataset.bams[sample], workdir / "fastq" / f"{sample}.fastq")
            run_command(
                [
                    "strainy",
                    "--fasta_ref", str(dataset.reference),
                    "--fastq", str(fastq),
                    "--bam", str(dataset.bams[sample]),
                    "--snp", str(dataset.vcfs[sample]),
                    "--mode", self.mode,
                    "--stage", "phase",
                    "--threads", str(threads),
                    "--output", str(outdir),
                    *self.extra_args,
                ],
                workdir / f"{sample}.log",
            )
            phased = outdir / "alignment_phased.bam"
            if not phased.exists():
                candidates = sorted(outdir.rglob("*phased*.bam"))
                if not candidates:
                    raise RuntimeError(
                        f"strainy produced no phased BAM under {outdir}; see the run log"
                    )
                phased = candidates[0]
            assignments.update(_tagged_bam_partition(phased, sample, tags=("HP", "YC")))
        return assignments


class DeviderAdapter(Adapter):
    info = ToolInfo(
        name="devider",
        designed_for=(
            "Haplotyping small sequences - viruses, plasmids, single genes - "
            "from long reads, using a positional de Bruijn graph over informative "
            "alleles. Included because it is the current state of the art at high "
            "coverage over short spans, not because it targets whole MAGs."
        ),
        citation="Shaw & Yu, 2025",
        supports_cross_sample_ids=False,
        multi_sample=False,
        requires=["devider"],
    )

    def __init__(self, extra_args: list[str] | None = None) -> None:
        self.extra_args = list(extra_args or [])

    def partition(
        self, dataset: Dataset, workdir: Path, threads: int
    ) -> dict[tuple[str, str], str]:
        assignments: dict[tuple[str, str], str] = {}
        for sample in dataset.samples:
            outdir = workdir / sample
            run_command(
                [
                    "devider",
                    "-b", str(dataset.bams[sample]),
                    "-v", str(dataset.vcfs[sample]),
                    "-r", str(dataset.reference),
                    "-o", str(outdir),
                    "-t", str(threads),
                    *self.extra_args,
                ],
                workdir / f"{sample}.log",
            )
            assignments.update(self._parse(outdir, sample))
        return assignments

    @staticmethod
    def _parse(outdir: Path, sample: str) -> dict[tuple[str, str], str]:
        """Parse ``ids.txt``: tab-separated ``contig, range, haplotype id``
        followed by the read identifiers belonging to that haplotype."""
        assignments: dict[tuple[str, str], str] = {}
        files = sorted(outdir.rglob("ids.txt"))
        if not files:
            raise RuntimeError(f"devider produced no ids.txt under {outdir}; see the run log")
        for path in files:
            for line in path.read_text(errors="replace").splitlines():
                fields = [f for f in line.rstrip("\n").split("\t") if f.strip()]
                if len(fields) < 4 or fields[0].startswith("#"):
                    continue
                contig, _range, hap_id, *read_ids = fields
                cluster = f"{contig}:{hap_id}"
                for read_id in read_ids:
                    assignments[(sample, read_id.strip())] = cluster
        return assignments


class WhatshapAdapter(Adapter):
    info = ToolInfo(
        name="whatshap-diploid",
        designed_for=(
            "Diploid read-backed phasing of a single individual. It is not a "
            "metagenomic strain method and cannot represent more than two "
            "haplotypes. Included because reviewers reasonably ask why a "
            "standard phaser will not do, and the honest answer is a measured "
            "one: it should be competitive at two strains and degrade sharply "
            "above that."
        ),
        citation="Martin et al., 2016; Patterson et al., J Comput Biol 22, 2015",
        supports_cross_sample_ids=False,
        multi_sample=False,
        requires=["whatshap"],
    )

    def partition(
        self, dataset: Dataset, workdir: Path, threads: int
    ) -> dict[tuple[str, str], str]:
        assignments: dict[tuple[str, str], str] = {}
        for sample in dataset.samples:
            phased_vcf = workdir / f"{sample}.phased.vcf.gz"
            tagged_bam = workdir / f"{sample}.tagged.bam"
            run_command(
                [
                    "whatshap", "phase",
                    "-o", str(phased_vcf),
                    "--reference", str(dataset.reference),
                    "--ignore-read-groups",
                    str(dataset.vcfs[sample]),
                    str(dataset.bams[sample]),
                ],
                workdir / f"{sample}.phase.log",
            )
            run_command(["tabix", "-f", "-p", "vcf", str(phased_vcf)], workdir / f"{sample}.tabix.log")
            run_command(
                [
                    "whatshap", "haplotag",
                    "-o", str(tagged_bam),
                    "--reference", str(dataset.reference),
                    "--ignore-read-groups",
                    str(phased_vcf),
                    str(dataset.bams[sample]),
                ],
                workdir / f"{sample}.haplotag.log",
            )
            assignments.update(_tagged_bam_partition(tagged_bam, sample, tags=("HP",)))
        return assignments

    def available(self) -> tuple[bool, str]:
        ok, reason = super().available()
        if not ok:
            return ok, reason
        import shutil

        if shutil.which("tabix") is None:
            return False, "'tabix' not found on PATH (needed to index whatshap output)"
        return True, ""
