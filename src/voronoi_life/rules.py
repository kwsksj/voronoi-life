from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

RuleType = Literal["absolute", "density", "probabilistic"]


@dataclass(frozen=True)
class RuleConfig:
    rule_type: RuleType = "absolute"
    birth_count: int = 3
    survive_counts: tuple[int, ...] = (2, 3)
    birth_min: float = 0.30
    birth_max: float = 0.45
    survive_min: float = 0.20
    survive_max: float = 0.45
    birth_threshold: float = 0.25
    optimal_density: float = 0.35
    birth_strength: float = 2.5
    death_strength: float = 2.0

    def to_json_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["survive_counts"] = list(self.survive_counts)
        return values


def step_state(
    alive: np.ndarray,
    adjacency: tuple[frozenset[int], ...],
    rule: RuleConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    alive = np.asarray(alive, dtype=bool)
    counts = alive_neighbor_counts(alive, adjacency)

    if rule.rule_type == "absolute":
        survives = np.isin(counts, np.asarray(rule.survive_counts, dtype=int))
        births = counts == rule.birth_count
        return np.where(alive, survives, births)

    density = alive_neighbor_density(alive, adjacency, counts=counts)
    if rule.rule_type == "density":
        births = (density >= rule.birth_min) & (density <= rule.birth_max)
        survives = (density >= rule.survive_min) & (density <= rule.survive_max)
        return np.where(alive, survives, births)

    if rule.rule_type == "probabilistic":
        if rng is None:
            rng = np.random.default_rng()
        birth_probability = np.clip(
            rule.birth_strength * (density - rule.birth_threshold),
            0.0,
            1.0,
        )
        death_probability = np.clip(
            rule.death_strength * np.abs(density - rule.optimal_density),
            0.0,
            1.0,
        )
        random_values = rng.random(len(alive))
        births = random_values < birth_probability
        survives = random_values >= death_probability
        return np.where(alive, survives, births)

    raise ValueError(f"unknown rule type: {rule.rule_type}")


def alive_neighbor_counts(
    alive: np.ndarray,
    adjacency: tuple[frozenset[int], ...],
) -> np.ndarray:
    alive = np.asarray(alive, dtype=bool)
    counts = np.zeros(len(adjacency), dtype=int)
    for index, neighbors in enumerate(adjacency):
        if neighbors:
            counts[index] = int(np.count_nonzero(alive[list(neighbors)]))
    return counts


def alive_neighbor_density(
    alive: np.ndarray,
    adjacency: tuple[frozenset[int], ...],
    counts: np.ndarray | None = None,
) -> np.ndarray:
    if counts is None:
        counts = alive_neighbor_counts(alive, adjacency)
    degrees = np.array([len(neighbors) for neighbors in adjacency], dtype=float)
    return np.divide(
        counts,
        degrees,
        out=np.zeros_like(degrees, dtype=float),
        where=degrees > 0,
    )
