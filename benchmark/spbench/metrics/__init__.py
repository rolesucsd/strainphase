"""Metric implementations.

Every metric here reads only the common intermediate format from
:mod:`spbench.formats`. None of them import any tool under evaluation. That is
what lets the report claim that all tools were scored identically.

The three modules mirror the three questions the benchmark asks:

``partition``
    Did the tool put the right reads together? Answerable for every tool, and
    the primary comparison surface.
``haplotype``
    Did the tool reconstruct the right allele sequences, at the right
    abundances, in the right number?
``longitudinal``
    Did the tool find strains at abundances where a single sample does not
    contain enough evidence, and did it keep their identity across timepoints?
    Only the last part is strainphase-specific; the first part is answerable for
    every tool and is where the longitudinal claim has to be won.
"""

from spbench.metrics.haplotype import haplotype_metrics, match_haplotypes
from spbench.metrics.longitudinal import longitudinal_metrics
from spbench.metrics.partition import partition_metrics

__all__ = [
    "partition_metrics",
    "haplotype_metrics",
    "match_haplotypes",
    "longitudinal_metrics",
]
