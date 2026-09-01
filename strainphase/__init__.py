"""
Strainphase: Hybrid graph-probabilistic haplotype reconstruction for PacBio HiFi metagenomic data.

This package reconstructs distinct bacterial haplotypes (strain-specific SNV patterns)
from mixed metagenomic reads using a hybrid approach combining graph-based initialization
with probabilistic EM refinement.

Example usage:
    >>> from strainphase import HaplotyperConfig, process_contig
    >>> config = HaplotyperConfig(window_size=20000, identity_distance=0.02)
    >>> results = process_contig(bam, vcf, contig_id, contig_length, config)

CLI usage:
    $ strainphase run --bam sample.bam --vcf variants.vcf --contig ctg1 --length 50000
    $ strainphase longitudinal --samples T1,T2,T3 --bams mapping/{sample}.bam ...
    $ strainphase sv reconcile ...
    $ strainphase test
"""

# Single source of truth is pyproject.toml. Hard-coding it here meant the two
# could drift - and they had: pyproject said 0.1.0rc1 while this said 0.1.0, so
# `strainphase version` would have misreported a release. The fallback covers a
# source tree that was never installed.
try:  # pragma: no cover - trivial, and the except path needs an uninstalled tree
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("strainphase")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"
__author__ = "Renee Oles"
__email__ = "roles@ucsd.edu"

from strainphase.core import (
    DEFAULT_CONFIG,
    EMHaplotyper,
    GraphInitializer,
    Haplotype,
    HaplotyperConfig,
    LogProbCache,
    LongitudinalIntegrator,
    PostProcessor,
    Read,
    Window,
    WindowResult,
    link_windows,
    process_contig,
    results_to_dataframe,
)
from strainphase.longitudinal import process_mag_longitudinal

# Core algorithms and pipeline entry points.

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Configuration
    "HaplotyperConfig",
    "DEFAULT_CONFIG",
    # Data structures
    "Read",
    "Window",
    "Haplotype",
    "WindowResult",
    # Core algorithms
    "GraphInitializer",
    "EMHaplotyper",
    "PostProcessor",
    "LongitudinalIntegrator",
    "LogProbCache",
    # Main functions
    "process_contig",
    "process_mag_longitudinal",
    "link_windows",
    "results_to_dataframe",
]
