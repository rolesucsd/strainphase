"""Benchmark configuration.

A config file names the datasets to simulate, the seeds to replicate them
across, and the tools to run. Everything that affects a number is in one file
that ships with the repository, so "which settings produced Table 2" has a
one-word answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from spbench.simulate import SimConfig


@dataclass
class ToolSpec:
    name: str
    #: Keyword arguments passed to the adapter constructor. Anything here is
    #: recorded in the results, so a tuned run is never mistaken for a
    #: default-parameter run.
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    name: str
    tools: list[ToolSpec]
    datasets: list[dict]
    seeds: list[int] = field(default_factory=lambda: [0])
    threads: int = 1
    match_threshold: float = 0.99
    min_shared_sites: int = 10
    description: str = ""

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkConfig:
        raw = yaml.safe_load(Path(path).read_text())
        tools = [
            ToolSpec(name=t, options={}) if isinstance(t, str)
            else ToolSpec(name=t["name"], options=dict(t.get("options", {})))
            for t in raw["tools"]
        ]
        return cls(
            name=raw.get("name", Path(path).stem),
            description=raw.get("description", ""),
            tools=tools,
            datasets=list(raw["datasets"]),
            seeds=list(raw.get("seeds", [0])),
            threads=int(raw.get("threads", 1)),
            match_threshold=float(raw.get("match_threshold", 0.99)),
            min_shared_sites=int(raw.get("min_shared_sites", 10)),
        )

    def expand(self) -> list[SimConfig]:
        """One :class:`SimConfig` per (dataset, seed) pair.

        Replicating across seeds is not optional decoration. A single simulated
        mixture can favour any method by luck; without replicates a difference
        between two tools cannot be distinguished from a difference between two
        random draws, and the report's confidence intervals come from here.
        """
        configs: list[SimConfig] = []
        valid = set(SimConfig.__dataclass_fields__)
        for spec in self.datasets:
            unknown = set(spec) - valid - {"name"}
            if unknown:
                raise ValueError(
                    f"dataset {spec.get('name', '?')!r} has unknown keys: "
                    f"{sorted(unknown)}; valid keys are {sorted(valid)}"
                )
            base = {k: v for k, v in spec.items() if k in valid}
            label = spec.get("name", "dataset")
            for seed in self.seeds:
                configs.append(
                    SimConfig(**{**base, "name": f"{label}.seed{seed}", "seed": seed})
                )
        return configs
