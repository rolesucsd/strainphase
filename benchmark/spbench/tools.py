"""What each tool claims to do.

The report uses this to avoid unfair comparisons: a tool is never scored on a
column it does not claim. Keeping it as data rather than as adapter classes
means the Snakemake rules own *running* the tools and this module owns only
describing them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolInfo:
    name: str
    #: One line on what the tool was built for. Printed in the report so a
    #: reader can judge whether a column is a fair test of it.
    designed_for: str
    citation: str = ""
    #: Does it see every timepoint in one invocation?
    multi_sample: bool = False
    #: Does it claim the same identifier for the same organism across samples?
    #: False means the cross-timepoint columns read n/a, never 0.
    supports_cross_sample_ids: bool = False


TOOLS: dict[str, ToolInfo] = {
    "strainphase-single": ToolInfo(
        name="strainphase-single",
        designed_for=(
            "Ablation of strainphase with cross-timepoint rescue disabled: "
            "graph initialisation plus quality-weighted EM within one sample. "
            "This is the row that belongs beside Floria and Strainy."
        ),
    ),
    "strainphase-longitudinal": ToolInfo(
        name="strainphase-longitudinal",
        designed_for=(
            "Longitudinal strain reconstruction: all timepoints jointly, with "
            "cross-timepoint rescue of strains below the single-sample "
            "detection floor, and stable lineage identity across samples."
        ),
        multi_sample=True,
        supports_cross_sample_ids=True,
    ),
    "floria": ToolInfo(
        name="floria",
        designed_for=(
            "Single-sample strain haplotyping of metagenomes via "
            "minimum-error-correction read clustering and a strain-preserving "
            "network flow model."
        ),
        citation="Shaw, Boucher, Yu, Noyes & Li, Bioinformatics 40(Suppl 1), 2024",
    ),
    "strainy": ToolInfo(
        name="strainy",
        designed_for=(
            "Single-sample phasing and assembly of strain haplotypes from "
            "long-read metagenomes, producing strain unitigs."
        ),
        citation="Kazantseva, Donmez, Frolova, Pop & Kolmogorov, Nature Methods, 2024",
    ),
}
