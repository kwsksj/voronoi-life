from __future__ import annotations

import numpy as np

from voronoi_life.points import generate_points


def test_point_generation_is_seed_reproducible() -> None:
    first = generate_points(30, "density-gradient", seed=123)
    second = generate_points(30, "density-gradient", seed=123)
    assert np.array_equal(first, second)


def test_all_point_methods_stay_in_unit_square() -> None:
    for method in ("random", "jittered-grid", "density-gradient"):
        points = generate_points(50, method, seed=7)
        assert points.shape == (50, 2)
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)
