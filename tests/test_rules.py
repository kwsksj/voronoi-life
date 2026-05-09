from __future__ import annotations

import numpy as np

from voronoi_life.rules import RuleConfig, step_state


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
