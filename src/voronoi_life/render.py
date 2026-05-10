from __future__ import annotations

from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection

from .geometry import VoronoiSpace
from .rules import alive_neighbor_counts, alive_neighbor_density

Overlay = Literal["none", "degree", "alive-count", "alive-density", "area", "edge-length"]
DensityScale = Literal["auto", "fixed"]


def draw_state(
    ax,
    space: VoronoiSpace,
    alive: np.ndarray,
    overlay: Overlay = "none",
    step_index: int = 0,
    rule_label: str = "",
    add_colorbar: bool = False,
    density_scale: DensityScale = "auto",
    rho_max: float | None = None,
) -> None:
    ax.clear()
    collection = _collection_for_state(space, alive, overlay, density_scale, rho_max)
    ax.add_collection(collection)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    boundary = "periodic" if space.periodic else "open"
    ax.set_title(f"{rule_label} step={step_index} boundary={boundary}".strip())
    if add_colorbar and (overlay != "none" or not _is_binary_state(alive)):
        ax.figure.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)


def save_png(
    path: Path,
    space: VoronoiSpace,
    alive: np.ndarray,
    overlay: Overlay,
    step_index: int,
    rule_label: str,
    density_scale: DensityScale = "auto",
    rho_max: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_state(
        ax,
        space,
        alive,
        overlay,
        step_index,
        rule_label,
        add_colorbar=True,
        density_scale=density_scale,
        rho_max=rho_max,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_gif(
    path: Path,
    space: VoronoiSpace,
    states: list[np.ndarray],
    overlay: Overlay,
    rule_label: str,
    fps: int = 12,
    density_scale: DensityScale = "auto",
    rho_max: float | None = None,
) -> None:
    duration = 1.0 / max(fps, 1)
    frames = [
        _state_to_rgb(space, state, overlay, index, rule_label, density_scale, rho_max)
        for index, state in enumerate(states)
    ]
    imageio.mimsave(path, frames, duration=duration)


def _state_to_rgb(
    space: VoronoiSpace,
    alive: np.ndarray,
    overlay: Overlay,
    step_index: int,
    rule_label: str,
    density_scale: DensityScale,
    rho_max: float | None,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    FigureCanvasAgg(fig)
    draw_state(
        ax,
        space,
        alive,
        overlay,
        step_index,
        rule_label,
        add_colorbar=False,
        density_scale=density_scale,
        rho_max=rho_max,
    )
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def _collection_for_state(
    space: VoronoiSpace,
    state: np.ndarray,
    overlay: Overlay,
    density_scale: DensityScale,
    rho_max: float | None,
) -> PolyCollection:
    polygons, cell_indices = _flatten_polygons(space)
    state = np.asarray(state)
    if overlay == "none":
        if _is_binary_state(state):
            colors = ["#151515" if state[index] else "#fbfbf7" for index in cell_indices]
            return PolyCollection(
                polygons,
                facecolors=colors,
                edgecolors="#9a9a92",
                linewidths=0.35,
            )
        polygon_values = np.asarray([state[index] for index in cell_indices], dtype=float)
        collection = PolyCollection(
            polygons,
            array=polygon_values,
            cmap="Greys",
            edgecolors="#8a8a82",
            linewidths=0.25,
        )
        collection.set_clim(*_density_clim(state.astype(float), density_scale, rho_max))
        return collection

    values = _overlay_values(space, state, overlay)
    polygon_values = np.asarray([values[index] for index in cell_indices], dtype=float)
    collection = PolyCollection(
        polygons,
        array=polygon_values,
        cmap="viridis",
        edgecolors="#3a3a36",
        linewidths=0.25,
    )
    if len(polygon_values) > 0:
        collection.set_clim(float(np.min(values)), float(np.max(values) or 1.0))
    return collection


def _is_binary_state(state: np.ndarray) -> bool:
    return np.asarray(state).dtype == np.dtype(bool)


def _density_clim(
    values: np.ndarray,
    density_scale: DensityScale,
    rho_max: float | None,
) -> tuple[float, float]:
    if density_scale == "fixed":
        return 0.0, float(rho_max if rho_max is not None else max(np.max(values), 1.0))
    if len(values) == 0:
        return 0.0, 1.0
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(minimum, maximum):
        maximum = minimum + 1.0
    return minimum, maximum


def _flatten_polygons(space: VoronoiSpace) -> tuple[list[np.ndarray], list[int]]:
    polygons: list[np.ndarray] = []
    cell_indices: list[int] = []
    for cell_index, pieces in enumerate(space.cell_polygons):
        for polygon in pieces:
            if len(polygon) >= 3:
                polygons.append(polygon)
                cell_indices.append(cell_index)
    return polygons, cell_indices


def _overlay_values(space: VoronoiSpace, state: np.ndarray, overlay: Overlay) -> np.ndarray:
    if overlay == "degree":
        return space.degrees.astype(float)
    if overlay == "area":
        return space.cell_areas.astype(float)
    if overlay == "edge-length":
        return space.edge_length_sums.astype(float)
    if overlay == "alive-count":
        alive = np.asarray(state, dtype=bool)
        return alive_neighbor_counts(alive, space.adjacency).astype(float)
    if overlay == "alive-density":
        alive = np.asarray(state, dtype=bool)
        return alive_neighbor_density(alive, space.adjacency)
    raise ValueError(f"unknown overlay: {overlay}")
