"""spbench: the public benchmark harness for strainphase.

The harness is deliberately separate from the ``strainphase`` package. It knows
nothing about strainphase internals: every tool under evaluation is run through
an *adapter* that emits the same common intermediate format (see
:mod:`spbench.formats`), and every metric is computed from that format alone.

That separation is the point. A reviewer can check that strainphase and Floria
are scored by literally the same code path, and can add a new tool by writing
one adapter without touching the metrics.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
