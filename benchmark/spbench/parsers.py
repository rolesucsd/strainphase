"""Each tool's native output -> a read partition.

This is the only tool-specific code left in the package. Everything downstream —
consensus, metrics, report — sees one format, which is what lets the report say
all tools were scored identically.

A partition is written as ``read_assignments.tsv``: ``sample, read_id, hap_id``.
Reads a tool saw but declined to place are written with ``hap_id = UNASSIGNED``
rather than dropped, so ``assigned_fraction`` reflects reality.
"""

from __future__ import annotations

import logging
from pathlib import Path

from spbench.formats import UNASSIGNED, write_table

logger = logging.getLogger(__name__)

READ_ASSIGNMENT_COLUMNS = ["sample", "read_id", "hap_id", "confidence"]


def write_partition(
    path: str | Path,
    assignments: dict[tuple[str, str], str],
    confidence: dict[tuple[str, str], float] | None = None,
) -> int:
    confidence = confidence or {}
    return write_table(
        path,
        READ_ASSIGNMENT_COLUMNS,
        (
            {
                "sample": sample,
                "read_id": read_id,
                "hap_id": hap_id,
                "confidence": confidence.get((sample, read_id), ""),
            }
            for (sample, read_id), hap_id in sorted(assignments.items())
        ),
    )


# --------------------------------------------------------------------------- #
# Floria
# --------------------------------------------------------------------------- #


def parse_floria(outdir: str | Path, sample: str) -> dict[tuple[str, str], str]:
    """Read every ``*.haplosets``: ``>HAP<n>...`` headers, then one read per line."""
    outdir = Path(outdir)
    assignments: dict[tuple[str, str], str] = {}
    files = sorted(outdir.rglob("*.haplosets"))
    if not files:
        raise RuntimeError(f"floria produced no *.haplosets under {outdir}; see its log")

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


# --------------------------------------------------------------------------- #
# Strainy
# --------------------------------------------------------------------------- #


def parse_strainy(outdir: str | Path, sample: str) -> dict[tuple[str, str], str]:
    """Read Strainy's phased BAM. ``HP`` if present, else the documented ``YC``.

    ``YC`` carries a colour string that is constant within a cluster, so it
    serves as a cluster label even though it was meant for visualisation.
    """
    import pysam

    outdir = Path(outdir)
    phased = outdir / "alignment_phased.bam"
    if not phased.exists():
        candidates = sorted(outdir.rglob("*phased*.bam"))
        if not candidates:
            raise RuntimeError(f"strainy produced no phased BAM under {outdir}; see its log")
        phased = candidates[0]

    assignments: dict[tuple[str, str], str] = {}
    with pysam.AlignmentFile(str(phased), "rb") as bam:
        for aln in bam.fetch(until_eof=True):
            if aln.is_secondary or aln.is_supplementary:
                continue
            label = UNASSIGNED
            for tag in ("HP", "YC"):
                if aln.has_tag(tag):
                    label = f"{tag}_{aln.get_tag(tag)}"
                    break
            assignments[(sample, aln.query_name)] = label

    if all(v == UNASSIGNED for v in assignments.values()):
        raise RuntimeError(
            f"no HP or YC tags found in {phased}; strainy's output format may "
            f"have changed. Nothing downstream can score an empty partition, so "
            f"this fails here rather than reporting a zero."
        )
    return assignments


PARSERS = {"floria": parse_floria, "strainy": parse_strainy}
