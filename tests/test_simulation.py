from __future__ import annotations

import numpy as np

from voronoi_life.rules import RuleConfig
from voronoi_life.simulation import SimulationConfig, VoronoiLife


def test_initial_state_is_seed_reproducible() -> None:
    config = SimulationConfig(cells=40, seed=99, initial_alive_ratio=0.25)
    first = VoronoiLife(config)
    second = VoronoiLife(config)

    assert np.array_equal(first.space.points, second.space.points)
    assert np.array_equal(first.initial_alive, second.initial_alive)


def test_simulation_runs_more_than_100_steps() -> None:
    simulation = VoronoiLife(SimulationConfig(cells=80, seed=3))
    states = simulation.run(120)

    assert len(states) == 121
    assert simulation.step_index == 120
    assert states[-1].shape == (80,)


def test_continuous_simulation_is_seed_reproducible() -> None:
    config = SimulationConfig(
        cells=40,
        seed=11,
        rule=RuleConfig(
            rule_type="continuous",
            continuous_init="random_density",
            coupling="edge",
            diffusion_rate=0.005,
        ),
    )

    first = VoronoiLife(config)
    second = VoronoiLife(config)

    assert np.array_equal(first.space.points, second.space.points)
    assert np.allclose(first.initial_life_amount, second.initial_life_amount)
    assert first.density.shape == (40,)
