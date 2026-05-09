from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import VoronoiSpace, build_space
from .points import PointMethod, generate_points
from .rules import RuleConfig, step_state


@dataclass(frozen=True)
class SimulationConfig:
    cells: int = 300
    seed: int = 1
    initial_alive_ratio: float = 0.28
    point_method: PointMethod = "random"
    periodic: bool = False
    rule: RuleConfig = RuleConfig()


class VoronoiLife:
    def __init__(self, config: SimulationConfig):
        if not 0.0 <= config.initial_alive_ratio <= 1.0:
            raise ValueError("initial_alive_ratio must be between 0 and 1")
        self.config = config
        self._generation = 0
        self._load_generation(config.seed)

    @property
    def space(self) -> VoronoiSpace:
        return self._space

    @property
    def alive(self) -> np.ndarray:
        return self._alive

    @property
    def initial_alive(self) -> np.ndarray:
        return self._initial_alive

    @property
    def step_index(self) -> int:
        return self._step_index

    def step(self) -> np.ndarray:
        self._alive = step_state(
            self._alive,
            self.space.adjacency,
            self.config.rule,
            rng=self._rule_rng,
        )
        self._step_index += 1
        return self._alive

    def reset_state(self) -> None:
        self._alive = self._initial_alive.copy()
        self._step_index = 0
        self._rule_rng = np.random.default_rng(self._current_seed + 2)

    def randomize_state(self) -> None:
        self._generation += 1
        init_rng = np.random.default_rng(self.config.seed + self._generation * 1009 + 1)
        self._initial_alive = (
            init_rng.random(self.config.cells) < self.config.initial_alive_ratio
        )
        self.reset_state()

    def regenerate_points(self) -> None:
        self._generation += 1
        self._load_generation(self.config.seed + self._generation * 1009)

    def run(self, steps: int, include_initial: bool = True) -> list[np.ndarray]:
        states = [self.alive.copy()] if include_initial else []
        for _ in range(steps):
            states.append(self.step().copy())
        return states

    def _load_generation(self, seed: int) -> None:
        self._current_seed = seed
        points = generate_points(self.config.cells, self.config.point_method, seed=seed)
        self._space = build_space(points, periodic=self.config.periodic)
        init_rng = np.random.default_rng(seed + 1)
        self._initial_alive = init_rng.random(self.config.cells) < self.config.initial_alive_ratio
        self._alive = self._initial_alive.copy()
        self._rule_rng = np.random.default_rng(seed + 2)
        self._step_index = 0
