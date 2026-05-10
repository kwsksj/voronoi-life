from __future__ import annotations

import numpy as np

from voronoi_life.rules import (
    RuleConfig,
    density_from_life_amount,
    initialize_life_amount,
    step_life_amount,
    step_state,
)


ADJACENCY = (
    frozenset({1, 2, 3}),
    frozenset({0, 2, 3}),
    frozenset({0, 1, 3}),
    frozenset({0, 1, 2}),
)


def test_absolute_rule_birth_and_survival() -> None:
    alive = np.array([False, True, True, True])
    next_alive = step_state(alive, ADJACENCY, RuleConfig(rule_type="absolute"))
    assert next_alive.tolist() == [True, True, True, True]


def test_absolute_rule_underpopulation_death() -> None:
    alive = np.array([True, False, False, False])
    next_alive = step_state(alive, ADJACENCY, RuleConfig(rule_type="absolute"))
    assert next_alive.tolist() == [False, False, False, False]


def test_density_rule_thresholds() -> None:
    alive = np.array([False, True, False, False])
    rule = RuleConfig(
        rule_type="density",
        birth_min=0.30,
        birth_max=0.40,
        survive_min=0.20,
        survive_max=0.45,
    )
    next_alive = step_state(alive, ADJACENCY, rule)
    assert next_alive.tolist() == [True, False, True, True]


def test_continuous_graph_diffusion_preserves_total_life_amount() -> None:
    life_amount = np.array([0.0, 2.0], dtype=float)
    cell_areas = np.array([1.0, 1.0], dtype=float)
    rule = RuleConfig(
        rule_type="continuous",
        coupling="graph",
        diffusion_rate=0.25,
        reaction="none",
    )

    next_life = step_life_amount(
        life_amount,
        cell_areas,
        ((0, 1),),
        np.array([1.0], dtype=float),
        np.array([1.0], dtype=float),
        rule,
    )

    assert np.allclose(next_life, np.array([0.5, 1.5]))
    assert np.isclose(float(np.sum(next_life)), float(np.sum(life_amount)))


def test_continuous_edge_distance_coupling_uses_length_over_distance() -> None:
    life_amount = np.array([0.0, 2.0], dtype=float)
    cell_areas = np.array([1.0, 1.0], dtype=float)
    rule = RuleConfig(
        rule_type="continuous",
        coupling="edge_distance",
        diffusion_rate=0.25,
        reaction="none",
    )

    next_life = step_life_amount(
        life_amount,
        cell_areas,
        ((0, 1),),
        np.array([2.0], dtype=float),
        np.array([4.0], dtype=float),
        rule,
    )

    assert np.allclose(next_life, np.array([0.25, 1.75]))


def test_continuous_initialization_returns_life_amount_from_density() -> None:
    rng = np.random.default_rng(123)
    cell_areas = np.array([0.25, 0.75], dtype=float)
    points = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=float)
    rule = RuleConfig(
        rule_type="continuous",
        continuous_init="binary_density",
        alive_density=2.0,
    )

    life_amount = initialize_life_amount(
        cell_areas,
        points,
        rule,
        rng,
        initial_alive_ratio=1.0,
    )

    assert np.allclose(density_from_life_amount(life_amount, cell_areas), [2.0, 2.0])
