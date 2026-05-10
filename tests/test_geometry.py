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


def test_geometry_metrics_cover_cells_and_edges() -> None:
    points = generate_points(50, "random", seed=7)
    space = build_space(points, periodic=False, include_edge_metrics=True)

    assert space.cell_areas.shape == (50,)
    assert np.all(space.cell_areas >= 0.0)
    assert np.isclose(float(np.sum(space.cell_areas)), 1.0)
    assert space.edge_lengths is not None
    assert space.center_distances is not None
    assert len(space.edge_pairs) == len(space.edge_lengths)
    assert len(space.edge_pairs) == len(space.center_distances)
    assert np.all(space.center_distances > 0.0)


def test_edge_metrics_are_opt_in() -> None:
    points = generate_points(50, "random", seed=7)
    space = build_space(points, periodic=False)

    assert space.edge_pairs == ()
    assert space.edge_lengths is None
    assert space.center_distances is None


def test_periodic_center_distance_uses_shortest_wrap() -> None:
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
    space = build_space(points, periodic=True, include_edge_metrics=True)
    assert space.center_distances is not None
    pair_distances = {
        pair: distance
        for pair, distance in zip(space.edge_pairs, space.center_distances, strict=True)
    }

    assert np.isclose(pair_distances[(0, 1)], 0.04)


def test_periodic_edge_lengths_include_shifted_ridges_inside_domain() -> None:
    points = generate_points(50, "random", seed=7)
    space = build_space(points, periodic=True, include_edge_metrics=True)
    assert space.edge_lengths is not None
    pair_lengths = {
        pair: length
        for pair, length in zip(space.edge_pairs, space.edge_lengths, strict=True)
    }

    assert np.isclose(pair_lengths[(8, 29)], 0.11326748519805499)
