"""Data contracts shared by the simulator, the tool adapters, and the metrics.

Two families of tables are defined here.

**Truth** (written by :mod:`spbench.simulate`, read by :mod:`spbench.evaluate`):

``truth/sites.tsv``
    ``contig  pos  ref  alt`` - every position that is polymorphic across the
    simulated strains. ``pos`` is 1-based, matching VCF convention. ``alt`` is a
    comma-separated list of the non-reference alleles observed at that site.

``truth/strains.tsv``
    ``strain_id  contig  n_sites  alleles`` - the true haplotype of each strain,
    encoded as ``pos:allele`` pairs over the sites above.

``truth/abundance.tsv``
    ``sample  strain_id  abundance`` - relative abundance per timepoint. Columns
    sum to 1.0 within a sample.

``truth/read_origins.tsv``
    ``sample  read_id  contig  strain_id  start  end`` - which strain each
    simulated read actually came from. This is the table that makes
    cross-tool comparison possible at all (see below).

**Predictions** (written by every adapter in :mod:`spbench.adapters`):

``haplotypes.tsv``
    ``sample  contig  hap_id  start  end  abundance  n_sites  alleles``

``read_assignments.tsv``
    ``sample  contig  read_id  hap_id  confidence``

Why two prediction tables
-------------------------
Tools in this space disagree about what a "result" is. Floria emits vartigs plus
read sets; Strainy emits phased assembly graph paths; strainphase emits
window-linked tracks with abundances. The one thing they all commit to is a
partition of the input reads. ``read_assignments.tsv`` is therefore the
universal comparison surface, and read-partition metrics (ARI, V-measure) are
the ones every tool can be scored on without any charitable reinterpretation.

``haplotypes.tsv`` is the richer surface - it supports allele-level accuracy and
abundance error - but not every tool populates ``abundance``. Adapters leave it
empty rather than inventing a value, and the metrics report abundance scores
only for tools that supply it.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Column definitions. Adapters and metrics both import these so a schema change
# is a one-line change in one file.
# --------------------------------------------------------------------------- #

SITES_COLUMNS = ["contig", "pos", "ref", "alt"]
STRAINS_COLUMNS = ["strain_id", "contig", "n_sites", "alleles"]
ABUNDANCE_COLUMNS = ["sample", "strain_id", "abundance"]
READ_ORIGINS_COLUMNS = ["sample", "read_id", "contig", "strain_id", "start", "end"]

HAPLOTYPE_COLUMNS = [
    "sample",
    "contig",
    "hap_id",
    "start",
    "end",
    "abundance",
    "n_sites",
    "alleles",
]
READ_ASSIGNMENT_COLUMNS = ["sample", "contig", "read_id", "hap_id", "confidence"]

#: Sentinel used in ``read_assignments.tsv`` for reads a tool saw but refused to
#: place. Distinguishing "not placed" from "not seen" matters: a tool that
#: assigns 10% of reads perfectly should not outscore one that assigns 100% of
#: reads well, and the metrics report assigned-fraction alongside every score.
UNASSIGNED = "UNASSIGNED"


def _open(path: str | Path, mode: str = "rt"):
    """Open plain or gzipped text transparently."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return open(path, mode, newline="" if "w" in mode else None)


def write_table(path: str | Path, columns: list[str], rows: Iterable[dict]) -> int:
    """Write a TSV with a fixed header. Returns the number of data rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "wt") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return n


def read_table(path: str | Path) -> Iterator[dict]:
    """Stream a TSV written by :func:`write_table`."""
    with _open(path, "rt") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


# --------------------------------------------------------------------------- #
# Allele encoding
# --------------------------------------------------------------------------- #
#
# A haplotype is encoded as a comma-separated list of ``pos:allele`` pairs, with
# ``pos`` 1-based and ``allele`` the literal allele *sequence* at that VCF site
# ("A" for a SNV, "ACGT" for an insertion allele, "" is never used).
#
# Encoding the sequence rather than an allele index means adapters do the
# index -> sequence lookup against the VCF once, in tool-specific code, instead
# of the metrics having to know each tool's indexing convention. Floria's
# 0-based-position/allele-index scheme and Strainy's cluster-consensus scheme
# then meet in the same space.


def encode_alleles(alleles: dict[int, str]) -> str:
    """``{101: "A", 250: "GT"}`` -> ``"101:A,250:GT"`` (sorted by position)."""
    return ",".join(f"{pos}:{alleles[pos]}" for pos in sorted(alleles))


def decode_alleles(encoded: str) -> dict[int, str]:
    """Inverse of :func:`encode_alleles`. Tolerates empty / whitespace input."""
    out: dict[int, str] = {}
    if not encoded:
        return out
    for item in encoded.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        pos_str, _, allele = item.partition(":")
        try:
            out[int(pos_str)] = allele
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------- #
# In-memory views
# --------------------------------------------------------------------------- #


@dataclass
class Haplotype:
    """One reconstructed (or true) haplotype in one sample."""

    hap_id: str
    sample: str
    contig: str
    alleles: dict[int, str]
    start: int = 0
    end: int = 0
    abundance: float | None = None

    @property
    def n_sites(self) -> int:
        return len(self.alleles)

    def to_row(self) -> dict:
        return {
            "sample": self.sample,
            "contig": self.contig,
            "hap_id": self.hap_id,
            "start": self.start,
            "end": self.end,
            "abundance": "" if self.abundance is None else f"{self.abundance:.6f}",
            "n_sites": self.n_sites,
            "alleles": encode_alleles(self.alleles),
        }


@dataclass
class Prediction:
    """Everything one tool produced on one dataset, plus how much it cost."""

    tool: str
    haplotypes: list[Haplotype] = field(default_factory=list)
    #: ``(sample, read_id) -> hap_id``. Missing keys mean "tool never saw it".
    read_assignments: dict[tuple[str, str], str] = field(default_factory=dict)
    #: ``(sample, read_id) -> confidence`` in [0, 1]; optional.
    read_confidence: dict[tuple[str, str], float] = field(default_factory=dict)
    read_contigs: dict[tuple[str, str], str] = field(default_factory=dict)
    wall_seconds: float | None = None
    peak_rss_mb: float | None = None
    status: str = "ok"
    message: str = ""

    def write(self, outdir: str | Path) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        write_table(
            outdir / "haplotypes.tsv",
            HAPLOTYPE_COLUMNS,
            (hap.to_row() for hap in self.haplotypes),
        )
        write_table(
            outdir / "read_assignments.tsv",
            READ_ASSIGNMENT_COLUMNS,
            (
                {
                    "sample": sample,
                    "contig": self.read_contigs.get((sample, read_id), ""),
                    "read_id": read_id,
                    "hap_id": hap_id,
                    "confidence": self.read_confidence.get((sample, read_id), ""),
                }
                for (sample, read_id), hap_id in sorted(self.read_assignments.items())
            ),
        )

    @classmethod
    def read(cls, outdir: str | Path, tool: str) -> Prediction:
        outdir = Path(outdir)
        pred = cls(tool=tool)
        hap_path = outdir / "haplotypes.tsv"
        if hap_path.exists():
            for row in read_table(hap_path):
                abundance = row.get("abundance", "").strip()
                pred.haplotypes.append(
                    Haplotype(
                        hap_id=row["hap_id"],
                        sample=row["sample"],
                        contig=row["contig"],
                        alleles=decode_alleles(row.get("alleles", "")),
                        start=int(row.get("start") or 0),
                        end=int(row.get("end") or 0),
                        abundance=float(abundance) if abundance else None,
                    )
                )
        read_path = outdir / "read_assignments.tsv"
        if read_path.exists():
            for row in read_table(read_path):
                key = (row["sample"], row["read_id"])
                pred.read_assignments[key] = row["hap_id"]
                pred.read_contigs[key] = row.get("contig", "")
                conf = row.get("confidence", "").strip()
                if conf:
                    pred.read_confidence[key] = float(conf)
        return pred


@dataclass
class Truth:
    """The simulated ground truth for one dataset."""

    #: ``contig -> {pos -> (ref, [alt, ...])}``
    sites: dict[str, dict[int, tuple[str, list[str]]]] = field(default_factory=dict)
    #: ``strain_id -> Haplotype`` (``sample`` is unused, ``contig`` is set)
    strains: dict[str, Haplotype] = field(default_factory=dict)
    #: ``(sample, strain_id) -> abundance``
    abundance: dict[tuple[str, str], float] = field(default_factory=dict)
    #: ``(sample, read_id) -> strain_id``
    read_origins: dict[tuple[str, str], str] = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)

    @classmethod
    def read(cls, truth_dir: str | Path) -> Truth:
        truth_dir = Path(truth_dir)
        truth = cls()

        for row in read_table(truth_dir / "sites.tsv"):
            contig = row["contig"]
            alt = [a for a in row["alt"].split(",") if a]
            truth.sites.setdefault(contig, {})[int(row["pos"])] = (row["ref"], alt)

        for row in read_table(truth_dir / "strains.tsv"):
            truth.strains[row["strain_id"]] = Haplotype(
                hap_id=row["strain_id"],
                sample="",
                contig=row["contig"],
                alleles=decode_alleles(row["alleles"]),
            )

        seen_samples: list[str] = []
        for row in read_table(truth_dir / "abundance.tsv"):
            truth.abundance[(row["sample"], row["strain_id"])] = float(row["abundance"])
            if row["sample"] not in seen_samples:
                seen_samples.append(row["sample"])
        truth.samples = seen_samples

        origins_path = truth_dir / "read_origins.tsv"
        if origins_path.exists():
            for row in read_table(origins_path):
                truth.read_origins[(row["sample"], row["read_id"])] = row["strain_id"]

        return truth

    def strains_present(self, sample: str, min_abundance: float = 0.0) -> list[str]:
        """Strain IDs with abundance strictly above ``min_abundance`` in ``sample``."""
        return sorted(
            strain
            for (samp, strain), value in self.abundance.items()
            if samp == sample and value > min_abundance
        )
