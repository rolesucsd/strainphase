"""Tests for the abundance archetypes.

The archetypes are the only invented part of a dataset, so their properties are
what the longitudinal claims are read against. Each test pins the property that
makes an archetype worth having: a coloniser must actually start at zero, a
bloom must actually return, a sweep must actually cross.
"""

from __future__ import annotations

import numpy as np
import pytest

from spbench.abundance import (
    ARCHETYPES,
    AbundanceConfig,
    assign_archetypes,
    build_trajectories,
)


def _series(strain_id: str, abundance: dict, n_timepoints: int) -> list[float]:
    return [abundance[(f"T{i + 1}", strain_id)] for i in range(n_timepoints)]


def _build(archetype: str, n_timepoints: int = 8, seed: int = 0, n_other: int = 2):
    """One strain with the archetype under test, plus stable filler."""
    strain_ids = ["focus"] + [f"other{i}" for i in range(n_other)]
    archetypes = [archetype] + ["stable"] * n_other
    rng = np.random.default_rng(seed)
    config = AbundanceConfig(n_timepoints=n_timepoints, noise=0.0)
    abundance, _ = build_trajectories(strain_ids, rng, config, archetypes=archetypes)
    return _series("focus", abundance, n_timepoints)


def test_abundances_sum_to_one_per_timepoint():
    rng = np.random.default_rng(3)
    config = AbundanceConfig(n_timepoints=6)
    abundance, _ = build_trajectories([f"s{i}" for i in range(5)], rng, config)
    for i in range(6):
        total = sum(v for (sample, _), v in abundance.items() if sample == f"T{i + 1}")
        assert total == pytest.approx(1.0, abs=1e-9)


def test_colonisation_starts_at_exactly_zero_and_rises():
    """The case that motivates cross-timepoint methods: a strain that is not
    there at all, then is."""
    series = _build("colonisation", n_timepoints=10, seed=1)
    assert series[0] == 0.0
    assert series[-1] > 0.0
    # Monotone increase after arrival, allowing for renormalisation against the
    # other strains.
    arrival = next(i for i, v in enumerate(series) if v > 0)
    rising = series[arrival:]
    assert rising[-1] > rising[0]


def test_bloom_rises_and_returns():
    series = _build("bloom", n_timepoints=12, seed=2)
    peak = max(range(len(series)), key=lambda i: series[i])
    assert 0 < peak < len(series) - 1, "peak should be interior, not at an endpoint"
    assert series[peak] > 3 * series[0]
    assert series[-1] < series[peak] / 2


def test_decline_falls():
    series = _build("decline", n_timepoints=10, seed=4)
    assert series[-1] < series[0]


def test_sweep_partners_cross():
    rng = np.random.default_rng(5)
    config = AbundanceConfig(n_timepoints=8, noise=0.0)
    abundance, _ = build_trajectories(
        ["up", "down"], rng, config, archetypes=["sweep_winner", "sweep_loser"]
    )
    up = _series("up", abundance, 8)
    down = _series("down", abundance, 8)
    assert up[0] < down[0]
    assert up[-1] > down[-1]


def test_stable_stays_within_an_order_of_magnitude():
    series = _build("stable", n_timepoints=10, seed=6)
    assert max(series) / max(1e-12, min(series)) < 10


def test_assignment_guarantees_the_interesting_archetypes():
    """A dataset of six stable residents would test nothing longitudinal."""
    rng = np.random.default_rng(7)
    for n in (3, 4, 5, 6, 8):
        assigned = assign_archetypes(n, rng)
        assert len(assigned) == n
        assert set(assigned) <= set(ARCHETYPES)
        assert "colonisation" in assigned
        assert "bloom" in assigned


def test_two_strains_become_a_sweep():
    rng = np.random.default_rng(8)
    assert sorted(assign_archetypes(2, rng)) == ["sweep_loser", "sweep_winner"]


def test_noise_never_resurrects_an_absent_strain():
    """Jitter is multiplicative on the archetype shape, so a zero must stay zero
    — otherwise 'absent' silently becomes 'very rare' and the false-positive
    test disappears."""
    rng = np.random.default_rng(9)
    config = AbundanceConfig(n_timepoints=10, noise=0.5)
    abundance, _ = build_trajectories(
        ["focus", "filler"], rng, config, archetypes=["colonisation", "stable"]
    )
    series = _series("focus", abundance, 10)
    assert series[0] == 0.0


def test_trajectories_are_deterministic():
    def once():
        rng = np.random.default_rng(11)
        return build_trajectories(
            ["a", "b", "c", "d"], rng, AbundanceConfig(n_timepoints=6)
        )[0]

    assert once() == once()
