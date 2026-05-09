from __future__ import annotations

import numpy as np

from voronoi_life.geometry import build_space
from voronoi_life.points import generate_points


def test_adjacency_has_no_self_edges_and_is_symmetric() -> None:
    points = generate_points(60, "random", seed=5)
    space = build_space(points, periodic=False)

    for index, neighbors in enumerate(space.adjacency):
        assert index not in neighbors
        for neighbor in neighbors:
            assert index in space.adjacency[neighbor]


def test_periodic_boundary_links_opposite_edges() -> None:
    points = np.array(
        [
            [0.02, 0.50],
            [0.98, 0.50],
            [0.50, 0.15],
            [0.50, 0.85],
            [0.50, 0.50],
            [0.25, 0.25],
            [0.75, 0.75],
        ],
        dtype=float,
    )
    space = build_space(points, periodic=True)
    assert 1 in space.adjacency[0]


def test_periodic_polygons_cover_every_cell() -> None:
    points = generate_points(40, "random", seed=12)
    space = build_space(points, periodic=True)
    assert all(len(polygons) >= 1 for polygons in space.cell_polygons)
