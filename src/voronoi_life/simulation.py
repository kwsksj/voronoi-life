from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .geometry import VoronoiSpace, build_space
from .points import PointMethod, generate_points
from .rules import (
    RuleConfig,
    density_from_life_amount,
    initialize_life_amount,
    step_life_amount,
    step_state,
)


@dataclass(frozen=True)
class SimulationConfig:
    cells: int = 300
    seed: int = 1
    initial_alive_ratio: float = 0.28
    point_method: PointMethod = "random"
    periodic: bool = False
    rule: RuleConfig = RuleConfig()
    include_edge_metrics: bool = False


StabilityKind = Literal["running", "steady", "oscillating", "not_tracked"]


@dataclass(frozen=True)
class StabilityStatus:
    kind: StabilityKind = "running"
    detected_step: int | None = None
    first_seen_step: int | None = None
    period: int | None = None
    reason: str = ""

    @property
    def stopped(self) -> bool:
        return self.kind in {"steady", "oscillating"}

    def to_json_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind,
            "stopped": self.stopped,
        }
        if self.detected_step is not None:
            payload["detected_step"] = self.detected_step
        if self.first_seen_step is not None:
            payload["first_seen_step"] = self.first_seen_step
        if self.period is not None:
            payload["period"] = self.period
        if self.reason:
            payload["reason"] = self.reason
        return payload


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
        if self.config.rule.rule_type == "continuous":
            return self.density > 0.0
        return self._alive

    @property
    def initial_alive(self) -> np.ndarray:
        return self._initial_alive

    @property
    def life_amount(self) -> np.ndarray:
        return self._life_amount

    @property
    def initial_life_amount(self) -> np.ndarray:
        return self._initial_life_amount

    @property
    def density(self) -> np.ndarray:
        return density_from_life_amount(self._life_amount, self.space.cell_areas)

    @property
    def state(self) -> np.ndarray:
        if self.config.rule.rule_type == "continuous":
            return self.density
        return self._alive

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def stability_status(self) -> StabilityStatus:
        return self._stability_status

    @property
    def is_stopped(self) -> bool:
        return self._stability_status.stopped

    def step(self) -> np.ndarray:
        if self.is_stopped:
            return self.state

        if self.config.rule.rule_type == "continuous":
            if self.space.edge_lengths is None or self.space.center_distances is None:
                raise ValueError("continuous rule requires edge metrics")
            self._life_amount = step_life_amount(
                self._life_amount,
                self.space.cell_areas,
                self.space.edge_pairs,
                self.space.edge_lengths,
                self.space.center_distances,
                self.config.rule,
            )
            self._alive = self.density > 0.0
            self._step_index += 1
            self._record_stability()
            return self.density

        self._alive = step_state(
            self._alive,
            self.space.adjacency,
            self.config.rule,
            rng=self._rule_rng,
        )
        self._step_index += 1
        self._record_stability()
        return self._alive

    def reset_state(self) -> None:
        self._alive = self._initial_alive.copy()
        self._life_amount = self._initial_life_amount.copy()
        self._step_index = 0
        self._rule_rng = np.random.default_rng(self._current_seed + 2)
        self._reset_stability_tracking()

    def randomize_state(self) -> None:
        self._generation += 1
        seed = self.config.seed + self._generation * 1009
        self._initialize_state(seed)
        self.reset_state()

    def regenerate_points(self) -> None:
        self._generation += 1
        self._load_generation(self.config.seed + self._generation * 1009)

    def run(self, steps: int, include_initial: bool = True) -> list[np.ndarray]:
        states = [self.state.copy()] if include_initial else []
        for _ in range(steps):
            if self.is_stopped:
                break
            states.append(self.step().copy())
        return states

    def _load_generation(self, seed: int) -> None:
        self._current_seed = seed
        points = generate_points(self.config.cells, self.config.point_method, seed=seed)
        self._space = build_space(
            points,
            periodic=self.config.periodic,
            include_edge_metrics=(
                self.config.include_edge_metrics
                or self.config.rule.rule_type == "continuous"
            ),
        )
        self._initialize_state(seed)
        self._rule_rng = np.random.default_rng(seed + 2)
        self._step_index = 0
        self._reset_stability_tracking()

    def _initialize_state(self, seed: int) -> None:
        init_rng = np.random.default_rng(seed + 1)
        self._initial_alive = init_rng.random(self.config.cells) < self.config.initial_alive_ratio
        self._alive = self._initial_alive.copy()
        if self.config.rule.rule_type == "continuous":
            self._initial_life_amount = initialize_life_amount(
                self.space.cell_areas,
                self.space.points,
                self.config.rule,
                init_rng,
                self.config.initial_alive_ratio,
                periodic=self.config.periodic,
            )
        else:
            self._initial_life_amount = self._initial_alive.astype(float) * self.space.cell_areas
        self._life_amount = self._initial_life_amount.copy()

    def _reset_stability_tracking(self) -> None:
        if self.config.rule.rule_type == "probabilistic":
            self._seen_state_steps: dict[bytes, int] = {}
            self._stability_status = StabilityStatus(
                kind="not_tracked",
                reason="probabilistic rule uses random choices",
            )
            return

        self._seen_state_steps = {self._state_fingerprint(): self._step_index}
        self._stability_status = StabilityStatus()

    def _record_stability(self) -> None:
        if self._stability_status.kind == "not_tracked":
            return

        fingerprint = self._state_fingerprint()
        first_seen_step = self._seen_state_steps.get(fingerprint)
        if first_seen_step is None:
            self._seen_state_steps[fingerprint] = self._step_index
            return

        period = self._step_index - first_seen_step
        self._stability_status = StabilityStatus(
            kind="steady" if period == 1 else "oscillating",
            detected_step=self._step_index,
            first_seen_step=first_seen_step,
            period=period,
        )

    def _state_fingerprint(self) -> bytes:
        if self.config.rule.rule_type == "continuous":
            state = np.round(np.asarray(self.state, dtype=float), decimals=12)
            return np.ascontiguousarray(state, dtype=np.float64).tobytes()

        packed = np.packbits(np.asarray(self._alive, dtype=np.uint8))
        return packed.tobytes()
