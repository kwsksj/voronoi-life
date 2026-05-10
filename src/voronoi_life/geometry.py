from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.spatial import Delaunay, Voronoi


@dataclass(frozen=True)
class VoronoiSpace:
    points: np.ndarray
    adjacency: tuple[frozenset[int], ...]
    cell_polygons: tuple[tuple[np.ndarray, ...], ...]
    periodic: bool
    cell_areas: np.ndarray
    edge_pairs: tuple[tuple[int, int], ...] = ()
    edge_lengths: np.ndarray | None = None
    center_distances: np.ndarray | None = None

    @property
    def degrees(self) -> np.ndarray:
        return np.array([len(neighbors) for neighbors in self.adjacency], dtype=int)

    @property
    def edge_length_sums(self) -> np.ndarray:
        if self.edge_lengths is None:
            raise ValueError("edge length metrics are not available")
        sums = np.zeros(len(self.points), dtype=float)
        for (left, right), length in zip(self.edge_pairs, self.edge_lengths, strict=True):
            sums[left] += length
            sums[right] += length
        return sums


def build_space(
    points: np.ndarray,
    periodic: bool = False,
    include_edge_metrics: bool = False,
) -> VoronoiSpace:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if len(points) < 4:
        raise ValueError("at least 4 points are required")

    if periodic:
        adjacency = _periodic_adjacency(points)
        polygons = _periodic_polygons(points)
    else:
        adjacency = _open_adjacency(points)
        polygons = _open_polygons(points)

    edge_pairs: tuple[tuple[int, int], ...] = ()
    edge_lengths: np.ndarray | None = None
    center_distances: np.ndarray | None = None
    if include_edge_metrics:
        edge_lengths_by_pair = (
            _periodic_edge_lengths(points) if periodic else _open_edge_lengths(points)
        )
        edge_pairs, edge_lengths, center_distances = _edge_metrics_from_adjacency(
            points,
            adjacency,
            edge_lengths_by_pair,
            periodic=periodic,
        )

    return VoronoiSpace(
        points=points,
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        cell_polygons=tuple(tuple(piece for piece in pieces) for pieces in polygons),
        periodic=periodic,
        cell_areas=_cell_areas(polygons),
        edge_pairs=edge_pairs,
        edge_lengths=edge_lengths,
        center_distances=center_distances,
    )


def _open_adjacency(points: np.ndarray) -> list[set[int]]:
    triangulation = Delaunay(points)
    adjacency = [set() for _ in range(len(points))]
    for simplex in triangulation.simplices:
        for a, b in combinations(simplex, 2):
            adjacency[a].add(b)
            adjacency[b].add(a)
    return adjacency


def _periodic_adjacency(points: np.ndarray) -> list[set[int]]:
    tiled_points, base_indices, shifts = tile_points(points)
    central = np.all(shifts == np.array([0.0, 0.0]), axis=1)
    triangulation = Delaunay(tiled_points)

    adjacency = [set() for _ in range(len(points))]
    for simplex in triangulation.simplices:
        for a, b in combinations(simplex, 2):
            if not (central[a] or central[b]):
                continue
            base_a = int(base_indices[a])
            base_b = int(base_indices[b])
            if base_a == base_b:
                continue
            adjacency[base_a].add(base_b)
            adjacency[base_b].add(base_a)
    return adjacency


def _open_polygons(points: np.ndarray) -> list[list[np.ndarray]]:
    voronoi = Voronoi(points)
    regions, vertices = finite_voronoi_regions(voronoi)
    polygons: list[list[np.ndarray]] = [[] for _ in range(len(points))]
    for point_index, region in enumerate(regions):
        clipped = clip_polygon(vertices[region])
        if polygon_area(clipped) > 1e-10:
            polygons[point_index].append(clipped)
    return polygons


def _periodic_polygons(points: np.ndarray) -> list[list[np.ndarray]]:
    tiled_points, base_indices, _shifts = tile_points(points)
    voronoi = Voronoi(tiled_points)
    regions, vertices = finite_voronoi_regions(voronoi, radius=4.0)

    polygons: list[list[np.ndarray]] = [[] for _ in range(len(points))]
    for tiled_index, region in enumerate(regions):
        clipped = clip_polygon(vertices[region])
        if polygon_area(clipped) > 1e-10:
            polygons[int(base_indices[tiled_index])].append(clipped)
    return polygons


def _cell_areas(polygons: list[list[np.ndarray]]) -> np.ndarray:
    return np.asarray(
        [sum(polygon_area(piece) for piece in pieces) for pieces in polygons],
        dtype=float,
    )


def _open_edge_lengths(points: np.ndarray) -> dict[tuple[int, int], float]:
    voronoi = Voronoi(points)
    return _ridge_edge_lengths(voronoi, np.arange(len(points), dtype=int))


def _periodic_edge_lengths(points: np.ndarray) -> dict[tuple[int, int], float]:
    tiled_points, base_indices, _shifts = tile_points(points)
    voronoi = Voronoi(tiled_points)
    return _ridge_edge_lengths(voronoi, base_indices)


def _ridge_edge_lengths(
    voronoi: Voronoi,
    base_indices: np.ndarray,
    central: np.ndarray | None = None,
) -> dict[tuple[int, int], float]:
    radius = float(np.ptp(voronoi.points, axis=0).max() * 2)
    center = voronoi.points.mean(axis=0)
    lengths: dict[tuple[int, int], float] = {}

    for (point_a, point_b), (vertex_a, vertex_b) in zip(
        voronoi.ridge_points,
        voronoi.ridge_vertices,
        strict=True,
    ):
        if central is not None and not (central[point_a] or central[point_b]):
            continue

        base_a = int(base_indices[point_a])
        base_b = int(base_indices[point_b])
        if base_a == base_b:
            continue

        segment = _ridge_segment(
            voronoi,
            int(point_a),
            int(point_b),
            int(vertex_a),
            int(vertex_b),
            center=center,
            radius=radius,
        )
        if segment is None:
            continue

        clipped = clip_segment(segment[0], segment[1])
        if clipped is None:
            continue

        length = float(np.linalg.norm(clipped[1] - clipped[0]))
        if length <= 1e-12:
            continue

        key = tuple(sorted((base_a, base_b)))
        lengths[key] = lengths.get(key, 0.0) + length

    return lengths


def _ridge_segment(
    voronoi: Voronoi,
    point_a: int,
    point_b: int,
    vertex_a: int,
    vertex_b: int,
    center: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if vertex_a >= 0 and vertex_b >= 0:
        return voronoi.vertices[vertex_a], voronoi.vertices[vertex_b]
    if vertex_a < 0 and vertex_b < 0:
        return None

    if vertex_b < 0:
        vertex_a, vertex_b = vertex_b, vertex_a

    tangent = voronoi.points[point_b] - voronoi.points[point_a]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        return None
    tangent /= norm
    normal = np.array([-tangent[1], tangent[0]])
    midpoint = voronoi.points[[point_a, point_b]].mean(axis=0)
    direction = np.sign(np.dot(midpoint - center, normal)) * normal
    if np.allclose(direction, 0.0):
        direction = normal
    finite = voronoi.vertices[vertex_b]
    return finite, finite + direction * radius


def _edge_metrics_from_adjacency(
    points: np.ndarray,
    adjacency: list[set[int]],
    edge_lengths_by_pair: dict[tuple[int, int], float],
    periodic: bool,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
    pairs: list[tuple[int, int]] = []
    lengths: list[float] = []
    distances: list[float] = []

    for left, neighbors in enumerate(adjacency):
        for right in sorted(neighbors):
            if left >= right:
                continue
            pair = (left, right)
            pairs.append(pair)
            lengths.append(float(edge_lengths_by_pair.get(pair, 0.0)))
            distances.append(_center_distance(points[left], points[right], periodic))

    return tuple(pairs), np.asarray(lengths, dtype=float), np.asarray(distances, dtype=float)


def _center_distance(left: np.ndarray, right: np.ndarray, periodic: bool) -> float:
    delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    if periodic:
        delta = delta - np.round(delta)
    return float(np.linalg.norm(delta))


def tile_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shifts = np.array(
        [(dx, dy) for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0)],
        dtype=float,
    )
    tiled_points = []
    base_indices = []
    tiled_shifts = []
    for shift in shifts:
        tiled_points.append(points + shift)
        base_indices.append(np.arange(len(points), dtype=int))
        tiled_shifts.append(np.repeat(shift[None, :], len(points), axis=0))
    return (
        np.vstack(tiled_points),
        np.concatenate(base_indices),
        np.vstack(tiled_shifts),
    )


def finite_voronoi_regions(
    voronoi: Voronoi,
    radius: float | None = None,
) -> tuple[list[list[int]], np.ndarray]:
    if voronoi.points.shape[1] != 2:
        raise ValueError("only 2D Voronoi diagrams are supported")

    if radius is None:
        radius = float(np.ptp(voronoi.points, axis=0).max() * 2)

    new_regions: list[list[int]] = []
    new_vertices = voronoi.vertices.tolist()
    center = voronoi.points.mean(axis=0)

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (point_a, point_b), (vertex_a, vertex_b) in zip(
        voronoi.ridge_points,
        voronoi.ridge_vertices,
        strict=True,
    ):
        all_ridges.setdefault(point_a, []).append((point_b, vertex_a, vertex_b))
        all_ridges.setdefault(point_b, []).append((point_a, vertex_a, vertex_b))

    for point_index, region_index in enumerate(voronoi.point_region):
        vertices = voronoi.regions[region_index]
        if vertices and all(vertex >= 0 for vertex in vertices):
            new_regions.append(list(vertices))
            continue

        ridges = all_ridges[point_index]
        new_region = [vertex for vertex in vertices if vertex >= 0]

        for other_point, vertex_a, vertex_b in ridges:
            if vertex_b < 0:
                vertex_a, vertex_b = vertex_b, vertex_a
            if vertex_a >= 0:
                continue

            tangent = voronoi.points[other_point] - voronoi.points[point_index]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = voronoi.points[[point_index, other_point]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = voronoi.vertices[vertex_b] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        region_vertices = np.asarray([new_vertices[v] for v in new_region])
        centroid = region_vertices.mean(axis=0)
        angles = np.arctan2(
            region_vertices[:, 1] - centroid[1],
            region_vertices[:, 0] - centroid[0],
        )
        new_region = [vertex for _, vertex in sorted(zip(angles, new_region, strict=True))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def clip_polygon(
    polygon: np.ndarray,
    min_x: float = 0.0,
    max_x: float = 1.0,
    min_y: float = 0.0,
    max_y: float = 1.0,
) -> np.ndarray:
    clipped = np.asarray(polygon, dtype=float)
    for inside, intersect in (
        (lambda point: point[0] >= min_x, lambda a, b: _intersect_x(a, b, min_x)),
        (lambda point: point[0] <= max_x, lambda a, b: _intersect_x(a, b, max_x)),
        (lambda point: point[1] >= min_y, lambda a, b: _intersect_y(a, b, min_y)),
        (lambda point: point[1] <= max_y, lambda a, b: _intersect_y(a, b, max_y)),
    ):
        clipped = _clip_edge(clipped, inside, intersect)
        if len(clipped) == 0:
            return clipped
    return clipped


def clip_segment(
    a: np.ndarray,
    b: np.ndarray,
    min_x: float = 0.0,
    max_x: float = 1.0,
    min_y: float = 0.0,
    max_y: float = 1.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    start = np.asarray(a, dtype=float)
    end = np.asarray(b, dtype=float)
    delta = end - start
    lower = 0.0
    upper = 1.0

    for p, q in (
        (-delta[0], start[0] - min_x),
        (delta[0], max_x - start[0]),
        (-delta[1], start[1] - min_y),
        (delta[1], max_y - start[1]),
    ):
        if np.isclose(p, 0.0):
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None

    return start + lower * delta, start + upper * delta


def _clip_edge(polygon, inside, intersect) -> np.ndarray:
    if len(polygon) == 0:
        return np.empty((0, 2), dtype=float)

    output = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous = current
        previous_inside = current_inside
    if not output:
        return np.empty((0, 2), dtype=float)
    return np.asarray(output, dtype=float)


def _intersect_x(a: np.ndarray, b: np.ndarray, x: float) -> np.ndarray:
    if np.isclose(a[0], b[0]):
        return np.array([x, a[1]], dtype=float)
    t = (x - a[0]) / (b[0] - a[0])
    return a + t * (b - a)


def _intersect_y(a: np.ndarray, b: np.ndarray, y: float) -> np.ndarray:
    if np.isclose(a[1], b[1]):
        return np.array([a[0], y], dtype=float)
    t = (y - a[1]) / (b[1] - a[1])
    return a + t * (b - a)


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
