"""
Strainphase: Hybrid graph-probabilistic haplotype reconstruction for PacBio HiFi metagenomic data.

This package reconstructs distinct bacterial haplotypes (strain-specific SNV patterns)
from mixed metagenomic reads using a hybrid approach combining graph-based initialization
with probabilistic EM refinement.

Example usage:
    >>> from strainphase import HaplotyperConfig, process_contig
    >>> config = HaplotyperConfig(window_size=3000, max_mismatch_frac=0.02)
    >>> results = process_contig(bam, vcf, contig_id, contig_length, config)

CLI usage:
    $ strainphase run --bam sample.bam --vcf variants.vcf --contig ctg1 --length 50000
    $ strainphase longitudinal --samples T1,T2,T3 --bams mapping/{sample}.bam ...
    $ strainphase test --quick
"""

__version__ = "0.1.0"
__author__ = "Renee Oles"
__email__ = "roles@ucsd.edu"

from strainphase.core import (
    # Configuration
    HaplotyperConfig,
    DEFAULT_CONFIG,
    # Data structures
    Read,
    Window,
    Haplotype,
    WindowResult,
    # Core algorithms
    GraphInitializer,
    EMHaplotyper,
    PostProcessor,
    LongitudinalIntegrator,
    LogProbCache,
    # Main functions
    process_contig,
    process_mag_longitudinal,
    link_windows,
    results_to_dataframe,
)

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
