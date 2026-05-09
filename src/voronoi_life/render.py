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

Overlay = Literal["none", "degree", "alive-count", "alive-density"]


def draw_state(
    ax,
    space: VoronoiSpace,
    alive: np.ndarray,
    overlay: Overlay = "none",
    step_index: int = 0,
    rule_label: str = "",
    add_colorbar: bool = False,
) -> None:
    ax.clear()
    collection = _collection_for_state(space, alive, overlay)
    ax.add_collection(collection)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    boundary = "periodic" if space.periodic else "open"
    ax.set_title(f"{rule_label} step={step_index} boundary={boundary}".strip())
    if add_colorbar and overlay != "none":
        ax.figure.colorbar(collection, ax=ax, fraction=0.046, pad=0.04)


def save_png(
    path: Path,
    space: VoronoiSpace,
    alive: np.ndarray,
    overlay: Overlay,
    step_index: int,
    rule_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_state(ax, space, alive, overlay, step_index, rule_label, add_colorbar=True)
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
) -> None:
    duration = 1.0 / max(fps, 1)
    frames = [
        _state_to_rgb(space, state, overlay, index, rule_label)
        for index, state in enumerate(states)
    ]
    imageio.mimsave(path, frames, duration=duration)


def _state_to_rgb(
    space: VoronoiSpace,
    alive: np.ndarray,
    overlay: Overlay,
    step_index: int,
    rule_label: str,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    FigureCanvasAgg(fig)
    draw_state(ax, space, alive, overlay, step_index, rule_label, add_colorbar=False)
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def _collection_for_state(
    space: VoronoiSpace,
    alive: np.ndarray,
    overlay: Overlay,
) -> PolyCollection:
    polygons, cell_indices = _flatten_polygons(space)
    if overlay == "none":
        colors = ["#151515" if alive[index] else "#fbfbf7" for index in cell_indices]
        return PolyCollection(
            polygons,
            facecolors=colors,
            edgecolors="#9a9a92",
            linewidths=0.35,
        )

    values = _overlay_values(space, alive, overlay)
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


def _flatten_polygons(space: VoronoiSpace) -> tuple[list[np.ndarray], list[int]]:
    polygons: list[np.ndarray] = []
    cell_indices: list[int] = []
    for cell_index, pieces in enumerate(space.cell_polygons):
        for polygon in pieces:
            if len(polygon) >= 3:
                polygons.append(polygon)
                cell_indices.append(cell_index)
    return polygons, cell_indices


def _overlay_values(space: VoronoiSpace, alive: np.ndarray, overlay: Overlay) -> np.ndarray:
    if overlay == "degree":
        return space.degrees.astype(float)
    if overlay == "alive-count":
        return alive_neighbor_counts(alive, space.adjacency).astype(float)
    if overlay == "alive-density":
        return alive_neighbor_density(alive, space.adjacency)
    raise ValueError(f"unknown overlay: {overlay}")
