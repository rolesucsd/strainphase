"""Loading a simulated dataset from disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dataset:
    """A simulated dataset: everything a tool adapter is allowed to look at.

    Adapters receive one of these and nothing else. In particular they are not
    handed the truth directory, so a tool cannot accidentally be run with
    knowledge it would not have on real data.
    """

    root: Path
    name: str
    reference: Path
    bams: dict[str, Path]
    vcfs: dict[str, Path]
    samples: list[str]
    contigs: dict[str, int]
    manifest: dict

    @property
    def reference_strain(self) -> str:
        """Which assembly was designated the reference for this dataset."""
        return self.manifest.get("reference_strain", "")

    @classmethod
    def load(cls, root: str | Path) -> Dataset:
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        samples = list(manifest["samples"])
        return cls(
            root=root,
            name=manifest.get("name", root.name),
            reference=root / "reference.fasta",
            bams={s: root / "bam" / f"{s}.bam" for s in samples},
            vcfs={s: root / "variants" / f"{s}.vcf.gz" for s in samples},
            samples=samples,
            contigs=dict(manifest["contigs"]),
            manifest=manifest,
        )

    @property
    def truth_dir(self) -> Path:
        """Only the evaluator touches this - never an adapter."""
        return self.root / "truth"
