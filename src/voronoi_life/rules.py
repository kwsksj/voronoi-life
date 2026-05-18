from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

RuleType = Literal["absolute", "density", "probabilistic", "continuous"]
ContinuousInit = Literal["random_density", "binary_density", "gaussian_blob"]
Coupling = Literal["graph", "edge", "edge_distance"]
Reaction = Literal["none", "logistic", "bell"]
DensityScale = Literal["auto", "fixed"]


@dataclass(frozen=True)
class RuleConfig:
    rule_type: RuleType = "absolute"
    birth_min_count: int = 3
    birth_max_count: int = 3
    survive_min_count: int = 2
    survive_max_count: int = 3
    birth_count: int | None = None
    survive_counts: tuple[int, ...] | None = None
    birth_min: float = 0.30
    birth_max: float = 0.45
    survive_min: float = 0.20
    survive_max: float = 0.45
    birth_threshold: float = 0.25
    optimal_density: float = 0.35
    birth_strength: float = 2.5
    death_strength: float = 2.0
    continuous_init: ContinuousInit = "random_density"
    initial_density_max: float = 1.0
    alive_density: float = 1.0
    coupling: Coupling = "edge"
    diffusion_rate: float = 0.01
    reaction: Reaction = "none"
    growth_rate: float = 0.02
    death_rate: float = 0.01
    carrying_capacity: float = 1.0
    sigma: float = 0.08
    rho_max: float | None = None
    density_scale: DensityScale = "auto"

    def __post_init__(self) -> None:
        if self.birth_count is not None:
            object.__setattr__(self, "birth_min_count", self.birth_count)
            object.__setattr__(self, "birth_max_count", self.birth_count)
        if self.survive_counts is not None and self.survive_counts:
            object.__setattr__(self, "survive_min_count", min(self.survive_counts))
            object.__setattr__(self, "survive_max_count", max(self.survive_counts))

    def to_json_dict(self) -> dict[str, object]:
        values = asdict(self)
        if self.survive_counts is None:
            values.pop("survive_counts")
        else:
            values["survive_counts"] = list(self.survive_counts)
        if self.birth_count is None:
            values.pop("birth_count")
        if self.rule_type != "continuous":
            for key in (
                "continuous_init",
                "initial_density_max",
                "alive_density",
                "coupling",
                "diffusion_rate",
                "reaction",
                "growth_rate",
                "death_rate",
                "carrying_capacity",
                "sigma",
                "rho_max",
                "density_scale",
            ):
                values.pop(key)
        return values

    def continuous_json_dict(self) -> dict[str, object]:
        return {
            "continuous_init": self.continuous_init,
            "coupling": self.coupling,
            "diffusion_rate": self.diffusion_rate,
            "reaction": self.reaction,
            "growth_rate": self.growth_rate,
            "death_rate": self.death_rate,
            "carrying_capacity": self.carrying_capacity,
            "optimal_density": self.optimal_density,
            "sigma": self.sigma,
            "rho_max": self.rho_max,
            "density_scale": self.density_scale,
            "initial_density_max": self.initial_density_max,
            "alive_density": self.alive_density,
        }


def step_state(
    alive: np.ndarray,
    adjacency: tuple[frozenset[int], ...],
    rule: RuleConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    alive = np.asarray(alive, dtype=bool)
    counts = alive_neighbor_counts(alive, adjacency)

    if rule.rule_type == "absolute":
        if rule.birth_count is None:
            births = (counts >= rule.birth_min_count) & (counts <= rule.birth_max_count)
        else:
            births = counts == rule.birth_count
        if rule.survive_counts is None:
            survives = (counts >= rule.survive_min_count) & (
                counts <= rule.survive_max_count
            )
        else:
            survives = np.isin(counts, np.asarray(rule.survive_counts, dtype=int))
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


def initialize_life_amount(
    cell_areas: np.ndarray,
    points: np.ndarray,
    rule: RuleConfig,
    rng: np.random.Generator,
    initial_alive_ratio: float,
    periodic: bool = False,
) -> np.ndarray:
    areas = np.asarray(cell_areas, dtype=float)
    points = np.asarray(points, dtype=float)

    if rule.continuous_init == "random_density":
        density = rng.uniform(0.0, rule.initial_density_max, size=len(areas))
    elif rule.continuous_init == "binary_density":
        density = np.where(
            rng.random(len(areas)) < initial_alive_ratio,
            rule.alive_density,
            0.0,
        )
    elif rule.continuous_init == "gaussian_blob":
        center = np.array([0.5, 0.5], dtype=float)
        delta = points - center
        if periodic:
            delta = delta - np.round(delta)
        distance_squared = np.sum(delta * delta, axis=1)
        width = max(rule.sigma, 1e-12)
        density = rule.initial_density_max * np.exp(-(distance_squared / (width * width)))
    else:
        raise ValueError(f"unknown continuous init: {rule.continuous_init}")

    return np.maximum(density, 0.0) * areas


def step_life_amount(
    life_amount: np.ndarray,
    cell_areas: np.ndarray,
    edge_pairs: tuple[tuple[int, int], ...],
    edge_lengths: np.ndarray,
    center_distances: np.ndarray,
    rule: RuleConfig,
) -> np.ndarray:
    life = np.asarray(life_amount, dtype=float)
    areas = np.asarray(cell_areas, dtype=float)
    next_life = life + _diffusion_delta(
        life,
        areas,
        edge_pairs,
        edge_lengths,
        center_distances,
        rule,
    )

    if rule.reaction != "none":
        density = density_from_life_amount(next_life, areas)
        next_life = next_life + _reaction_delta(density, areas, rule)

    next_life = np.maximum(next_life, 0.0)
    if rule.rho_max is not None:
        next_life = np.minimum(next_life, rule.rho_max * areas)
    return next_life


def density_from_life_amount(life_amount: np.ndarray, cell_areas: np.ndarray) -> np.ndarray:
    life = np.asarray(life_amount, dtype=float)
    areas = np.asarray(cell_areas, dtype=float)
    return np.divide(
        life,
        areas,
        out=np.zeros_like(life, dtype=float),
        where=areas > 0.0,
    )


def _diffusion_delta(
    life_amount: np.ndarray,
    cell_areas: np.ndarray,
    edge_pairs: tuple[tuple[int, int], ...],
    edge_lengths: np.ndarray,
    center_distances: np.ndarray,
    rule: RuleConfig,
) -> np.ndarray:
    delta = np.zeros_like(life_amount, dtype=float)
    if not edge_pairs or rule.diffusion_rate == 0.0:
        return delta
    if rule.diffusion_rate < 0.0:
        raise ValueError("diffusion_rate must be non-negative")

    pairs = np.asarray(edge_pairs, dtype=int)
    left = pairs[:, 0]
    right = pairs[:, 1]
    density = density_from_life_amount(life_amount, cell_areas)
    weights = _coupling_weights(edge_lengths, center_distances, rule.coupling)
    flows = rule.diffusion_rate * weights * (density[right] - density[left])
    np.add.at(delta, left, flows)
    np.add.at(delta, right, -flows)
    return delta


def _coupling_weights(
    edge_lengths: np.ndarray,
    center_distances: np.ndarray,
    coupling: Coupling,
) -> np.ndarray:
    if coupling == "graph":
        return np.ones_like(edge_lengths, dtype=float)
    if coupling == "edge":
        return np.asarray(edge_lengths, dtype=float)
    if coupling == "edge_distance":
        distances = np.asarray(center_distances, dtype=float)
        return np.divide(
            edge_lengths,
            distances,
            out=np.zeros_like(edge_lengths, dtype=float),
            where=distances > 0.0,
        )
    raise ValueError(f"unknown coupling: {coupling}")


def _reaction_delta(
    density: np.ndarray,
    cell_areas: np.ndarray,
    rule: RuleConfig,
) -> np.ndarray:
    if rule.reaction == "logistic":
        if rule.carrying_capacity <= 0.0:
            raise ValueError("carrying_capacity must be positive")
        return (
            rule.growth_rate
            * density
            * (1.0 - density / rule.carrying_capacity)
            * cell_areas
        )

    if rule.reaction == "bell":
        if rule.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        growth = rule.growth_rate * np.exp(
            -((density - rule.optimal_density) ** 2) / (2.0 * rule.sigma * rule.sigma)
        )
        death = rule.death_rate * density
        return (growth - death) * cell_areas

    raise ValueError(f"unknown reaction: {rule.reaction}")


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
