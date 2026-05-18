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
    simulation = VoronoiLife(
        SimulationConfig(cells=80, seed=3, rule=RuleConfig(rule_type="probabilistic"))
    )
    states = simulation.run(120)

    assert len(states) == 121
    assert simulation.step_index == 120
    assert states[-1].shape == (80,)
    assert simulation.stability_status.kind == "not_tracked"


def test_simulation_stops_on_steady_state() -> None:
    simulation = VoronoiLife(SimulationConfig(cells=20, seed=1, initial_alive_ratio=0.0))

    simulation.step()
    status = simulation.stability_status

    assert status.kind == "steady"
    assert status.stopped is True
    assert status.detected_step == 1

    simulation.step()
    assert simulation.step_index == 1


def test_simulation_stops_on_oscillation() -> None:
    config = SimulationConfig(
        cells=20,
        seed=1,
        initial_alive_ratio=0.0,
        rule=RuleConfig(birth_count=0, survive_counts=()),
    )
    simulation = VoronoiLife(config)

    states = simulation.run(10)
    status = simulation.stability_status

    assert status.kind == "oscillating"
    assert status.period == 2
    assert status.detected_step == 2
    assert simulation.step_index == 2
    assert len(states) == 3


def test_oscillation_cycles_display_after_detection() -> None:
    config = SimulationConfig(
        cells=20,
        seed=1,
        initial_alive_ratio=0.0,
        rule=RuleConfig(birth_count=0, survive_counts=()),
    )
    simulation = VoronoiLife(config)

    states = simulation.run(10)
    initial_state = states[0]
    next_state = states[1]

    simulation.step()
    assert simulation.step_index == 3
    assert simulation.stability_status.detected_step == 2
    assert np.array_equal(simulation.state, next_state)

    simulation.step()
    assert simulation.step_index == 4
    assert simulation.stability_status.detected_step == 2
    assert np.array_equal(simulation.state, initial_state)


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
