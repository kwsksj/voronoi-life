from __future__ import annotations

from typing import Literal

import numpy as np

PointMethod = Literal["random", "jittered-grid", "density-gradient"]


def generate_points(
    count: int,
    method: PointMethod = "random",
    seed: int | None = None,
) -> np.ndarray:
    if count < 4:
        raise ValueError("count must be at least 4 for Delaunay/Voronoi generation")

    rng = np.random.default_rng(seed)
    if method == "random":
        return rng.random((count, 2))
    if method == "jittered-grid":
        return _jittered_grid(count, rng)
    if method == "density-gradient":
        return _density_gradient(count, rng)
    raise ValueError(f"unknown point generation method: {method}")


def _jittered_grid(count: int, rng: np.random.Generator) -> np.ndarray:
    side = int(np.ceil(np.sqrt(count)))
    coords = (np.arange(side) + 0.5) / side
    xx, yy = np.meshgrid(coords, coords)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    rng.shuffle(points)
    points = points[:count]

    jitter = 0.35 / side
    points = points + rng.uniform(-jitter, jitter, size=points.shape)
    return np.clip(points, 1e-4, 1.0 - 1e-4)


def _density_gradient(count: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.beta(1.15, 2.7, size=count)
    y = rng.random(count)
    return np.column_stack([x, y])
