"""Adapter registry.

Adding a tool is one entry here plus one :class:`~spbench.adapters.base.Adapter`
subclass implementing ``partition()``. No metric, report, or config change is
needed, which is the property that makes it reasonable to ask a reviewer to
believe all tools were scored the same way.
"""

from __future__ import annotations

from collections.abc import Callable

from spbench.adapters.base import Adapter, ToolInfo
from spbench.adapters.external import (
    DeviderAdapter,
    FloriaAdapter,
    StrainyAdapter,
    WhatshapAdapter,
)
from spbench.adapters.naive import NaiveAdapter
from spbench.adapters.strainphase_adapter import (
    StrainphaseLongitudinalAdapter,
    StrainphaseSingleAdapter,
)

#: Name -> factory. Names are what appear in config files and in the results.
REGISTRY: dict[str, Callable[..., Adapter]] = {
    "naive-greedy": NaiveAdapter,
    "strainphase-single": StrainphaseSingleAdapter,
    "strainphase-longitudinal": StrainphaseLongitudinalAdapter,
    "floria": FloriaAdapter,
    "strainy": StrainyAdapter,
    "devider": DeviderAdapter,
    "whatshap-diploid": WhatshapAdapter,
}


def build(name: str, **kwargs) -> Adapter:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown tool {name!r}; known tools: {known}")
    return REGISTRY[name](**kwargs)


__all__ = ["REGISTRY", "build", "Adapter", "ToolInfo"]
