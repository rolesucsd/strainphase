"""Abundance trajectories for the simulated timecourse.

The strains are real and their differences are real; the only thing invented
here is how much of each one is present at each timepoint. That is the right
place to invent something, because it is the one quantity a longitudinal
benchmark has to control.

Trajectories are drawn from named archetypes rather than from noise. A random
walk produces trajectories no microbiologist would recognise and, worse, gives
no way to say "the method missed the colonisation events" — because there are no
colonisation events, only wiggle. Each strain is assigned an archetype, so every
dataset contains a known mixture of behaviours and the report can be read
per-behaviour.

    stable        persistent resident, small fluctuations about a level
    bloom         transient expansion and return — a disturbance response
    colonisation  absent at first (exactly 0), then logistic growth to a plateau
    decline       resident that is progressively lost, possibly to 0
    sweep_winner  rises as its partner falls
    sweep_loser   the partner

`colonisation` and `decline` can hit exactly zero, which matters: a strain that
is genuinely absent is the only way to test whether a method invents haplotypes,
and a strain arriving from zero is the case cross-timepoint methods should
handle well and single-sample methods cannot see coming.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARCHETYPES = (
    "stable",
    "bloom",
    "colonisation",
    "decline",
    "sweep_winner",
    "sweep_loser",
)


@dataclass
class AbundanceConfig:
    """Shape parameters for the trajectory archetypes.

    Levels are in *relative* units before normalisation; only their ratios
    matter. Defaults give a community with one clear dominant, a couple of
    mid-abundance residents and at least one strain near the detection floor.
    """

    n_timepoints: int = 6
    #: Log10 abundance spread of the resident strains. Real gut communities are
    #: heavily skewed, so a flat mixture would be the easy case.
    resident_log_spread: float = 0.9
    #: Fractional noise applied to every point, on top of the archetype shape.
    noise: float = 0.15
    #: Peak of a bloom relative to its baseline.
    bloom_fold: float = 12.0
    #: Plateau a coloniser reaches, relative to a typical resident.
    colonisation_plateau: float = 1.2
    #: Steepness of the logistic colonisation curve.
    colonisation_steepness: float = 1.6
    #: Fold change across a sweep.
    sweep_fold: float = 20.0


def assign_archetypes(n_strains: int, rng: np.random.Generator) -> list[str]:
    """Give each strain a behaviour, guaranteeing the interesting ones appear.

    With enough strains the assignment always contains at least one coloniser
    and one bloom, because those are the cases the benchmark exists to measure.
    A purely random assignment would sometimes produce a dataset of six stable
    residents, which tests nothing about longitudinal information.
    """
    if n_strains <= 0:
        return []
    if n_strains == 1:
        return ["stable"]
    if n_strains == 2:
        return ["sweep_winner", "sweep_loser"]

    assigned = ["colonisation", "bloom", "stable"]
    if n_strains >= 5:
        assigned += ["sweep_winner", "sweep_loser"]
    if n_strains >= 4 and "sweep_winner" not in assigned:
        assigned.append("decline")

    remaining = n_strains - len(assigned)
    if remaining > 0:
        pool = ["stable", "stable", "decline", "bloom"]
        assigned += [str(rng.choice(pool)) for _ in range(remaining)]

    assigned = assigned[:n_strains]
    rng.shuffle(assigned)
    return assigned


def _shape(archetype: str, t: np.ndarray, config: AbundanceConfig, rng) -> np.ndarray:
    """Unnormalised trajectory on [0, 1] time for one archetype."""
    if archetype == "stable":
        return np.ones_like(t)

    if archetype == "bloom":
        # Gaussian pulse. Centre kept off the endpoints so the rise and the
        # return are both observed rather than truncated by the sampling window.
        centre = float(rng.uniform(0.3, 0.7))
        width = float(rng.uniform(0.10, 0.18))
        return 1.0 + (config.bloom_fold - 1.0) * np.exp(-(((t - centre) / width) ** 2))

    if archetype == "colonisation":
        # Exactly zero before arrival, then logistic growth. Logistic because
        # that is what colonisation of an open niche looks like, and because a
        # linear ramp would make the low-abundance timepoints arbitrarily brief.
        arrival = float(rng.uniform(0.15, 0.45))
        out = np.zeros_like(t)
        after = t >= arrival
        scaled = (t[after] - arrival) / max(1e-6, 1.0 - arrival)
        out[after] = config.colonisation_plateau / (
            1.0 + np.exp(-config.colonisation_steepness * 6.0 * (scaled - 0.35))
        )
        return out

    if archetype == "decline":
        # Exponential loss; clears to exactly zero in the last third about half
        # the time, so "strain genuinely gone" is also represented.
        rate = float(rng.uniform(2.0, 4.5))
        out = np.exp(-rate * t)
        if rng.random() < 0.5:
            out[t > float(rng.uniform(0.6, 0.85))] = 0.0
        return out

    # A sweep has to end in a reversal, not a tie: the winner starts a factor of
    # `sweep_fold` below the loser and ends the same factor above it. Ramping
    # both to a common endpoint would leave them level at the last timepoint,
    # which is a convergence, not a sweep, and would make the archetype
    # unfalsifiable.
    if archetype == "sweep_winner":
        return config.sweep_fold ** (2.0 * t - 1.0)

    if archetype == "sweep_loser":
        return config.sweep_fold ** (1.0 - 2.0 * t)

    raise ValueError(f"unknown archetype {archetype!r}")


def build_trajectories(
    strain_ids: list[str],
    rng: np.random.Generator,
    config: AbundanceConfig,
    archetypes: list[str] | None = None,
) -> tuple[dict[tuple[str, str], float], dict[str, str]]:
    """Abundance of every strain at every timepoint, normalised per timepoint.

    Returns ``({(sample, strain_id): abundance}, {strain_id: archetype})``.
    Samples are named ``T1 … Tn``.
    """
    n = len(strain_ids)
    if n == 0:
        return {}, {}

    archetypes = archetypes or assign_archetypes(n, rng)
    samples = [f"T{i + 1}" for i in range(config.n_timepoints)]
    t = (
        np.linspace(0.0, 1.0, config.n_timepoints)
        if config.n_timepoints > 1
        else np.array([0.0])
    )

    # Baseline levels: log-uniform, so the community is skewed like a real one
    # rather than evenly mixed.
    levels = 10.0 ** rng.uniform(-config.resident_log_spread, 0.0, size=n)

    raw = np.zeros((n, config.n_timepoints))
    for i, archetype in enumerate(archetypes):
        shape = _shape(archetype, t, config, rng)
        jitter = 1.0 + rng.normal(0.0, config.noise, size=config.n_timepoints)
        series = levels[i] * shape * np.clip(jitter, 0.4, 1.6)
        # Jitter must never resurrect a strain the archetype set to zero.
        series[shape == 0.0] = 0.0
        raw[i, :] = np.maximum(series, 0.0)

    abundance: dict[tuple[str, str], float] = {}
    for j, sample in enumerate(samples):
        column = raw[:, j]
        total = column.sum()
        if total <= 0:
            column = np.full(n, 1.0 / n)
        else:
            column = column / total
        for strain_id, value in zip(strain_ids, column, strict=True):
            abundance[(sample, strain_id)] = float(value)

    return abundance, dict(zip(strain_ids, archetypes, strict=True))
