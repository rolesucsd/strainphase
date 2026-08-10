"""The adapter contract.

An adapter's one obligation is to produce a read partition:
``(sample, read_id) -> cluster_id``. Everything else - consensus haplotypes,
abundances, all the metrics - is derived from that by shared code, so adding a
tool means writing one function and no metric changes.

Adapters also carry metadata that the report uses to avoid unfair comparisons:
what the tool was designed to do, whether it carries identity across samples,
and whether it sees all timepoints at once. A reviewer reading the report can
see, per tool, exactly which columns are meaningful for it.
"""

from __future__ import annotations

import logging
import resource
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from spbench.dataset import Dataset
from spbench.formats import Prediction
from spbench.reads import consensus_from_partition, load_sites, read_alleles

logger = logging.getLogger(__name__)


@dataclass
class ToolInfo:
    """What the report needs to know to describe a tool honestly."""

    name: str
    #: One line on what the tool was built for. Printed in the report so the
    #: reader can judge whether a column is a fair test of it.
    designed_for: str
    citation: str = ""
    #: True if the tool assigns stable identifiers across samples. False means
    #: the cross-timepoint identity columns are reported as n/a, not as 0.
    supports_cross_sample_ids: bool = False
    #: True if the tool sees every timepoint in one invocation. Used to label
    #: which comparisons are like-for-like.
    multi_sample: bool = False
    #: External executables that must be on PATH.
    requires: list[str] = field(default_factory=list)


class Adapter:
    """Base class. Subclasses implement :meth:`partition`."""

    info: ToolInfo

    def available(self) -> tuple[bool, str]:
        """Can this tool run here? Never raises - a missing tool is a reported
        ``skipped`` row, not a crashed benchmark run."""
        for executable in self.info.requires:
            if shutil.which(executable) is None:
                return False, f"{executable!r} not found on PATH"
        return True, ""

    def partition(
        self, dataset: Dataset, workdir: Path, threads: int
    ) -> dict[tuple[str, str], str]:
        """Return ``(sample, read_id) -> cluster_id``."""
        raise NotImplementedError

    def native_haplotypes(self, dataset: Dataset, workdir: Path) -> list:
        """Optional tool-native haplotype output, scored as a separate row."""
        return []

    # ---------------------------------------------------------------- driver

    def run(self, dataset: Dataset, workdir: Path, threads: int = 1) -> Prediction:
        """Run the tool, time it, and derive the common-format prediction."""
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        prediction = Prediction(tool=self.info.name)

        ok, reason = self.available()
        if not ok:
            prediction.status = "skipped"
            prediction.message = reason
            logger.warning("%s: skipped (%s)", self.info.name, reason)
            return prediction

        start = time.monotonic()
        rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        try:
            assignments = self.partition(dataset, workdir, threads)
        except Exception as exc:  # noqa: BLE001 - one tool failing must not stop the run
            prediction.status = "failed"
            prediction.message = f"{type(exc).__name__}: {exc}"
            prediction.wall_seconds = time.monotonic() - start
            logger.exception("%s: failed", self.info.name)
            return prediction

        prediction.wall_seconds = time.monotonic() - start
        rss_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        prediction.peak_rss_mb = max(rss_after - rss_before, self_rss) / 1024.0

        prediction.read_assignments = dict(assignments)
        prediction.haplotypes = self._derive_haplotypes(dataset, assignments)
        logger.info(
            "%s: %d reads assigned, %d haplotypes, %.1fs",
            self.info.name,
            len(assignments),
            len(prediction.haplotypes),
            prediction.wall_seconds,
        )
        return prediction

    def _derive_haplotypes(
        self, dataset: Dataset, assignments: dict[tuple[str, str], str]
    ) -> list:
        """Consensus haplotypes from the partition - identical code for every tool."""
        haplotypes = []
        for sample in dataset.samples:
            per_sample = {
                read_id: cluster
                for (samp, read_id), cluster in assignments.items()
                if samp == sample
            }
            if not per_sample:
                continue
            for contig in dataset.contigs:
                sites = load_sites(str(dataset.vcfs[sample]), contig)
                if not sites:
                    continue
                alleles = read_alleles(str(dataset.bams[sample]), contig, sites)
                haplotypes.extend(
                    consensus_from_partition(
                        per_sample, alleles, sample, contig, self.info.name
                    )
                )
        return haplotypes


def run_command(cmd: list[str], log_path: Path, cwd: Path | None = None) -> None:
    """Run an external tool, teeing stdout+stderr to ``log_path``.

    Failures raise with the tail of the log attached, so a broken external tool
    produces a diagnosable ``failed`` row rather than an empty one.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("$ %s", " ".join(cmd))
    with open(log_path, "w") as handle:
        proc = subprocess.run(
            cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=cwd, check=False
        )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {' '.join(cmd)}\n{tail}"
        )
