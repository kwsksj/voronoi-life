from __future__ import annotations

import base64
import json
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .output import default_output_dir, write_experiment_json
from .render import DensityScale, Overlay, draw_state, save_png
from .rules import RuleConfig
from .simulation import SimulationConfig, VoronoiLife

POINT_METHODS = {"random", "jittered-grid", "density-gradient"}
RULE_TYPES = {"absolute", "density", "probabilistic", "continuous"}
OVERLAYS = {"none", "degree", "alive-count", "alive-density", "area", "edge-length"}
CONTINUOUS_INITS = {"random_density", "binary_density", "gaussian_blob"}
COUPLINGS = {"graph", "edge", "edge_distance"}
REACTIONS = {"none", "logistic", "bell"}
DENSITY_SCALES = {"auto", "fixed"}


@dataclass(frozen=True)
class ViewConfig:
    overlay: Overlay = "none"
    density_scale: DensityScale = "auto"
    rho_max: float | None = None


class GuiSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._output_dir: Path | None = None
        self.config = build_simulation_config({})
        self.view = build_view_config({})
        self.simulation = VoronoiLife(self.config)

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            return create_state_payload(self.simulation, self.view)

    def reset(self, raw: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.config = build_simulation_config(raw)
            self.view = build_view_config(raw, self.config.rule)
            self.simulation = VoronoiLife(self.config)
            return create_state_payload(self.simulation, self.view, "設定を適用しました。")

    def update_view(self, raw: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.view = build_view_config(raw, self.config.rule)
            return create_state_payload(self.simulation, self.view)

    def step(self, raw: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.view = build_view_config(raw, self.config.rule)
            steps = _bounded_int(raw, "steps", 1, minimum=1, maximum=500)
            for _ in range(steps):
                if self.simulation.is_stopped:
                    break
                self.simulation.step()
            return create_state_payload(self.simulation, self.view)

    def reset_state(self, raw: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.view = build_view_config(raw, self.config.rule)
            self.simulation.reset_state()
            return create_state_payload(self.simulation, self.view, "初期状態に戻しました。")

    def randomize_state(self, raw: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.view = build_view_config(raw, self.config.rule)
            self.simulation.randomize_state()
            return create_state_payload(self.simulation, self.view, "初期状態を作り直しました。")

    def regenerate_points(self, raw: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.view = build_view_config(raw, self.config.rule)
            self.simulation.regenerate_points()
            return create_state_payload(self.simulation, self.view, "点群を作り直しました。")

    def save_png(self, raw: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.view = build_view_config(raw, self.config.rule)
            if self._output_dir is None:
                self._output_dir = default_output_dir()
            self._output_dir.mkdir(parents=True, exist_ok=True)

            png_path = self._output_dir / f"gui-step-{self.simulation.step_index:05d}.png"
            save_png(
                png_path,
                self.simulation.space,
                self.simulation.state,
                self.view.overlay,
                self.simulation.step_index,
                _rule_label(self.config.rule),
                density_scale=self.view.density_scale,
                rho_max=self.view.rho_max,
            )
            outputs = {"png": str(png_path)}
            json_path = self._output_dir / "experiment.json"
            outputs["json"] = str(json_path)
            write_experiment_json(
                json_path,
                self.config,
                self.simulation.step_index,
                self.view.overlay,
                outputs,
                self.simulation.stability_status.to_json_dict(),
            )
            payload = create_state_payload(
                self.simulation,
                self.view,
                f"PNGを保存しました: {png_path}",
            )
            payload["saved"] = outputs
            return payload


def build_simulation_config(raw: dict[str, Any]) -> SimulationConfig:
    rule_type = _choice(raw, "ruleType", "absolute", RULE_TYPES)
    rule = RuleConfig(
        rule_type=rule_type,  # type: ignore[arg-type]
        birth_count=_bounded_int(raw, "birthCount", 3, minimum=0, maximum=24),
        survive_counts=_parse_counts(raw.get("surviveCounts", "2,3")),
        birth_min=_bounded_float(raw, "birthMin", 0.30, minimum=0.0, maximum=1.0),
        birth_max=_bounded_float(raw, "birthMax", 0.45, minimum=0.0, maximum=1.0),
        survive_min=_bounded_float(raw, "surviveMin", 0.20, minimum=0.0, maximum=1.0),
        survive_max=_bounded_float(raw, "surviveMax", 0.45, minimum=0.0, maximum=1.0),
        birth_threshold=_bounded_float(raw, "birthThreshold", 0.25, minimum=0.0, maximum=1.0),
        optimal_density=_bounded_float(raw, "optimalDensity", 0.35, minimum=0.0, maximum=5.0),
        birth_strength=_bounded_float(raw, "birthStrength", 2.5, minimum=0.0, maximum=20.0),
        death_strength=_bounded_float(raw, "deathStrength", 2.0, minimum=0.0, maximum=20.0),
        continuous_init=_choice(
            raw,
            "continuousInit",
            "random_density",
            CONTINUOUS_INITS,
        ),  # type: ignore[arg-type]
        initial_density_max=_bounded_float(
            raw,
            "initialDensityMax",
            1.0,
            minimum=0.0,
            maximum=10.0,
        ),
        alive_density=_bounded_float(raw, "aliveDensity", 1.0, minimum=0.0, maximum=10.0),
        coupling=_choice(raw, "coupling", "edge", COUPLINGS),  # type: ignore[arg-type]
        diffusion_rate=_bounded_float(raw, "diffusionRate", 0.01, minimum=0.0, maximum=10.0),
        reaction=_choice(raw, "reaction", "none", REACTIONS),  # type: ignore[arg-type]
        growth_rate=_bounded_float(raw, "growthRate", 0.02, minimum=0.0, maximum=10.0),
        death_rate=_bounded_float(raw, "deathRate", 0.01, minimum=0.0, maximum=10.0),
        carrying_capacity=_bounded_float(
            raw,
            "carryingCapacity",
            1.0,
            minimum=0.000001,
            maximum=10.0,
        ),
        sigma=_bounded_float(raw, "sigma", 0.08, minimum=0.000001, maximum=2.0),
        rho_max=_optional_float(raw.get("rhoMax")),
        density_scale=_choice(raw, "densityScale", "auto", DENSITY_SCALES),  # type: ignore[arg-type]
    )
    return SimulationConfig(
        cells=_bounded_int(raw, "cells", 300, minimum=4, maximum=5000),
        seed=_bounded_int(raw, "seed", 1, minimum=-2_147_483_648, maximum=2_147_483_647),
        initial_alive_ratio=_bounded_float(
            raw,
            "initialAliveRatio",
            0.28,
            minimum=0.0,
            maximum=1.0,
        ),
        point_method=_choice(raw, "pointMethod", "random", POINT_METHODS),  # type: ignore[arg-type]
        periodic=bool(raw.get("periodic", False)),
        rule=rule,
        include_edge_metrics=True,
    )


def build_view_config(raw: dict[str, Any], rule: RuleConfig | None = None) -> ViewConfig:
    overlay = _choice(raw, "overlay", "none", OVERLAYS)
    if rule is not None and rule.rule_type == "continuous" and overlay in {
        "alive-count",
        "alive-density",
    }:
        overlay = "none"
    return ViewConfig(
        overlay=overlay,  # type: ignore[arg-type]
        density_scale=_choice(raw, "densityScale", "auto", DENSITY_SCALES),  # type: ignore[arg-type]
        rho_max=_optional_float(raw.get("rhoMax")),
    )


def create_state_payload(
    simulation: VoronoiLife,
    view: ViewConfig,
    message: str = "",
) -> dict[str, Any]:
    stability = simulation.stability_status.to_json_dict()
    return {
        "ok": True,
        "image": render_state_data_url(simulation, view),
        "stats": _stats(simulation),
        "stability": stability,
        "view": {
            "overlay": view.overlay,
            "densityScale": view.density_scale,
            "rhoMax": view.rho_max,
        },
        "message": message or _stability_message(stability),
    }


def render_state_data_url(simulation: VoronoiLife, view: ViewConfig) -> str:
    figure = Figure(figsize=(7.2, 7.2), dpi=130)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    draw_state(
        axis,
        simulation.space,
        simulation.state,
        view.overlay,
        simulation.step_index,
        _rule_label(simulation.config.rule),
        add_colorbar=False,
        density_scale=view.density_scale,
        rho_max=view.rho_max,
    )
    figure.tight_layout(pad=0.4)
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def serve_gui(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    server = _make_server(host, port)
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    print(f"Voronoi Life GUI: {url}")
    print("終了するには、このターミナルで Ctrl+C を押してください。")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGUI server stopped.")
    finally:
        server.server_close()


def _make_server(host: str, port: int) -> "_GuiServer":
    last_error: OSError | None = None
    for candidate in _candidate_ports(port):
        try:
            return _GuiServer((host, candidate), _GuiRequestHandler)
        except OSError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _candidate_ports(port: int) -> range:
    if port == 0:
        return range(0, 1)
    return range(port, port + 20)


def _stats(simulation: VoronoiLife) -> dict[str, Any]:
    alive = np.asarray(simulation.alive, dtype=bool)
    density = np.asarray(simulation.density, dtype=float)
    stability = simulation.stability_status.to_json_dict()
    return {
        "step": simulation.step_index,
        "cells": simulation.config.cells,
        "aliveCount": int(np.count_nonzero(alive)),
        "aliveRatio": float(np.mean(alive)) if len(alive) else 0.0,
        "meanDensity": float(np.mean(density)) if len(density) else 0.0,
        "maxDensity": float(np.max(density)) if len(density) else 0.0,
        "totalLife": float(np.sum(simulation.life_amount)),
        "meanDegree": float(np.mean(simulation.space.degrees)),
        "boundary": "periodic" if simulation.space.periodic else "open",
        "rule": simulation.config.rule.rule_type,
        "points": simulation.config.point_method,
        "stabilityKind": stability["kind"],
        "stabilityStopped": stability["stopped"],
        "stabilityPeriod": stability.get("period"),
    }


def _stability_message(stability: dict[str, object]) -> str:
    if stability.get("kind") == "steady":
        return f"定常状態を検出したため停止しました（step {stability.get('detected_step')}）。"
    if stability.get("kind") == "oscillating":
        return (
            "振動状態を検出したため停止しました"
            f"（周期 {stability.get('period')}、step {stability.get('detected_step')}）。"
        )
    return ""


def _rule_label(rule: RuleConfig) -> str:
    if rule.rule_type == "continuous":
        return f"continuous {rule.coupling}/{rule.reaction}"
    return rule.rule_type


def _choice(raw: dict[str, Any], key: str, default: str, choices: set[str]) -> str:
    value = raw.get(key, default)
    if isinstance(value, str) and value in choices:
        return value
    return default


def _bounded_int(
    raw: dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(raw.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _bounded_float(
    raw: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(raw.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not np.isfinite(value):
        value = default
    return max(minimum, min(maximum, value))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "off", "unlimited"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _parse_counts(value: Any) -> tuple[int, ...]:
    try:
        if isinstance(value, (list, tuple)):
            counts = [int(item) for item in value if str(item).strip()]
        else:
            counts = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    except (TypeError, ValueError):
        return (2, 3)
    counts = sorted({count for count in counts if 0 <= count <= 24})
    return tuple(counts) or (2, 3)


class _GuiServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler]):
        super().__init__(server_address, handler_class)
        self.session = GuiSession()


class _GuiRequestHandler(BaseHTTPRequestHandler):
    server: _GuiServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(INDEX_HTML)
            return
        if path == "/api/state":
            self._send_json(self.server.session.state_payload())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        raw = self._read_json()
        try:
            if path == "/api/reset":
                payload = self.server.session.reset(raw)
            elif path == "/api/view":
                payload = self.server.session.update_view(raw)
            elif path == "/api/step":
                payload = self.server.session.step(raw)
            elif path == "/api/reset-state":
                payload = self.server.session.reset_state(raw)
            elif path == "/api/randomize-state":
                payload = self.server.session.randomize_state(raw)
            elif path == "/api/regenerate-points":
                payload = self.server.session.regenerate_points(raw)
            elif path == "/api/save-png":
                payload = self.server.session.save_png(raw)
            else:
                self.send_error(404)
                return
            self._send_json(payload)
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self._send_json({"ok": False, "error": str(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


INDEX_HTML = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Voronoi Life GUI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f5f2;
      --surface: #ffffff;
      --ink: #171713;
      --muted: #66675f;
      --line: #d7d7ce;
      --accent: #0f766e;
      --accent-dark: #0a5f59;
      --play: #15803d;
      --play-dark: #166534;
      --step: #2563eb;
      --step-dark: #1d4ed8;
      --stop: #c2410c;
      --stop-dark: #9a3412;
      --purple: #7c3aed;
      --purple-dark: #6d28d9;
      --warn: #a16207;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.4;
    }

    .app {
      display: grid;
      grid-template-columns: minmax(580px, 1fr) 390px;
      min-height: 100vh;
    }

    .stage {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
      padding: 18px;
      gap: 14px;
    }

    .topbar,
    .statusbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 38px;
    }

    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 720;
      letter-spacing: 0;
      white-space: nowrap;
    }

    .control-panel {
      display: flex;
      flex: 1 1 auto;
      flex-wrap: wrap;
      align-items: stretch;
      justify-content: flex-end;
      gap: 8px;
      min-width: 260px;
    }

    .control-group {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      padding: 5px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.68);
    }

    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }

    button {
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--surface);
      color: var(--ink);
      padding: 7px 10px;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }

    button.control-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      white-space: nowrap;
    }

    .button-icon {
      font-size: 12px;
      line-height: 1;
    }

    button:hover {
      border-color: #b7b8ad;
      background: #fbfbf8;
    }

    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #ffffff;
      font-weight: 650;
    }

    button.primary:hover {
      background: var(--accent-dark);
    }

    button.play-action {
      border-color: var(--play);
      background: var(--play);
      color: #ffffff;
      font-weight: 680;
    }

    button.play-action:hover {
      background: var(--play-dark);
    }

    button.play-action.is-stopping {
      border-color: var(--stop);
      background: var(--stop);
    }

    button.play-action.is-stopping:hover {
      background: var(--stop-dark);
    }

    button.step-action {
      border-color: var(--step);
      background: #eff6ff;
      color: #1e3a8a;
      font-weight: 650;
    }

    button.step-action:hover {
      border-color: var(--step-dark);
      background: #dbeafe;
    }

    button.utility-action {
      background: #f8fafc;
      color: #334155;
    }

    button.save-action {
      border-color: #c4b5fd;
      background: #f5f3ff;
      color: #4c1d95;
      font-weight: 650;
    }

    button.save-action:hover {
      border-color: var(--purple-dark);
      background: #ede9fe;
    }

    .playback-settings {
      min-width: 264px;
    }

    .compact-field {
      display: grid;
      grid-template-columns: auto minmax(70px, 92px);
      align-items: center;
      gap: 5px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }

    .compact-field input {
      min-height: 30px;
      padding: 5px 7px;
      font-size: 12px;
    }

    .visual {
      align-self: stretch;
      display: grid;
      place-items: center;
      min-height: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      overflow: hidden;
    }

    .visual img {
      display: block;
      width: min(100%, calc(100vh - 150px));
      max-height: calc(100vh - 150px);
      aspect-ratio: 1 / 1;
      object-fit: contain;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(6, minmax(86px, 1fr));
      gap: 8px;
      width: 100%;
    }

    .stat {
      position: relative;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--surface);
      padding: 8px 9px;
      min-width: 0;
      cursor: help;
      outline: none;
    }

    .stat:focus-visible {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(78, 124, 115, 0.18);
    }

    .stat span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.2;
    }

    .stat strong {
      display: block;
      margin-top: 3px;
      font-size: 15px;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 100%;
      min-height: 24px;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid transparent;
      font-size: 13px;
      font-weight: 760;
      line-height: 1.2;
    }

    .status-running {
      border-color: #bfdbfe;
      background: #dbeafe;
      color: #1e40af;
    }

    .status-steady {
      border-color: #bbf7d0;
      background: #dcfce7;
      color: #166534;
    }

    .status-oscillating {
      border-color: #ddd6fe;
      background: #ede9fe;
      color: #5b21b6;
    }

    .status-not-tracked {
      border-color: #e2e8f0;
      background: #f1f5f9;
      color: #475569;
    }

    aside {
      height: 100vh;
      overflow: auto;
      border-left: 1px solid var(--line);
      background: #fbfbf8;
      padding: 16px;
    }

    .settings-action-bar {
      display: grid;
      margin-bottom: 14px;
    }

    .settings-action-bar button {
      justify-self: stretch;
    }

    form {
      display: grid;
      gap: 14px;
    }

    fieldset {
      margin: 0;
      padding: 13px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
    }

    legend {
      padding: 0 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    label {
      position: relative;
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 620;
      cursor: help;
    }

    input,
    select {
      width: 100%;
      min-height: 32px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #ffffff;
      color: var(--ink);
      padding: 6px 8px;
      font: inherit;
      font-size: 13px;
    }

    input[type="checkbox"] {
      width: 18px;
      min-height: 18px;
      margin: 0;
    }

    input[type="range"] {
      padding: 0;
      accent-color: var(--accent);
    }

    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 32px;
      color: var(--ink);
    }

    .check-text {
      display: grid;
      gap: 2px;
      color: var(--ink);
    }

    .hint {
      position: absolute;
      z-index: 30;
      top: calc(100% + 6px);
      left: 0;
      width: min(280px, calc(100vw - 40px));
      padding: 8px 10px;
      border: 1px solid #d6d3c8;
      border-radius: 7px;
      background: #1f2937;
      color: #ffffff;
      box-shadow: 0 10px 24px rgba(31, 41, 55, 0.18);
      font-size: 11px;
      font-weight: 500;
      line-height: 1.35;
      opacity: 0;
      pointer-events: none;
      transform: translateY(-2px);
      transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
      visibility: hidden;
      white-space: normal;
    }

    .hint::before {
      position: absolute;
      top: -5px;
      left: 12px;
      width: 8px;
      height: 8px;
      border-top: 1px solid #d6d3c8;
      border-left: 1px solid #d6d3c8;
      background: #1f2937;
      content: "";
      transform: rotate(45deg);
    }

    .grid label:nth-child(2n) .hint {
      right: 0;
      left: auto;
    }

    .grid label:nth-child(2n) .hint::before {
      right: 12px;
      left: auto;
    }

    label:hover .hint,
    label:focus-within .hint,
    label.is-hint-open .hint,
    .stat:hover .hint,
    .stat:focus .hint,
    .stat:focus-within .hint,
    .stat.is-hint-open .hint {
      opacity: 1;
      transform: translateY(0);
      visibility: visible;
    }

    .stat .hint {
      top: auto;
      bottom: calc(100% + 6px);
      color: #ffffff;
    }

    .stat .hint::before {
      top: auto;
      bottom: -5px;
      border: 0;
      border-right: 1px solid #d6d3c8;
      border-bottom: 1px solid #d6d3c8;
    }

    .stats .stat:nth-child(n + 4) .hint {
      right: 0;
      left: auto;
    }

    .stats .stat:nth-child(n + 4) .hint::before {
      right: 12px;
      left: auto;
    }

    .hints-off .hint {
      display: none;
    }

    .hints-off label,
    .hints-off .stat {
      cursor: default;
    }

    .range-row {
      display: grid;
      grid-template-columns: 1fr 54px;
      align-items: center;
      gap: 8px;
    }

    output {
      font-variant-numeric: tabular-nums;
      color: var(--ink);
      text-align: right;
    }

    .formula-help {
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 12px;
    }

    .formula-body {
      display: grid;
      gap: 7px;
    }

    .formula-line {
      display: grid;
      gap: 2px;
      border-left: 3px solid #c7ddd9;
      padding-left: 8px;
    }

    .formula-line strong {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }

    .formula-line code {
      display: block;
      overflow-wrap: anywhere;
      color: var(--ink);
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: normal;
    }

    .rule-group[hidden] {
      display: none;
    }

    .message {
      color: var(--muted);
      font-size: 13px;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .message.warning {
      color: var(--warn);
    }

    @media (max-width: 980px) {
      .app {
        grid-template-columns: 1fr;
      }

      aside {
        height: auto;
        border-left: 0;
        border-top: 1px solid var(--line);
      }

      .stats {
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }

      .stats .stat:nth-child(n) .hint,
      .grid label:nth-child(n) .hint {
        right: auto;
        left: 0;
      }

      .stats .stat:nth-child(2n) .hint,
      .grid label:nth-child(2n) .hint {
        right: 0;
        left: auto;
      }

      .stats .stat:nth-child(n) .hint::before,
      .grid label:nth-child(n) .hint::before {
        right: auto;
        left: 12px;
      }

      .stats .stat:nth-child(2n) .hint::before,
      .grid label:nth-child(2n) .hint::before {
        right: 12px;
        left: auto;
      }

      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }

      .control-panel,
      .control-group,
      .playback-settings {
        width: 100%;
      }

      .control-group {
        justify-content: flex-start;
      }

      .compact-field {
        grid-template-columns: minmax(58px, auto) minmax(80px, 1fr);
        flex: 1 1 130px;
      }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="stage">
      <div class="topbar">
        <h1>Voronoi Life</h1>
        <div class="control-panel" aria-label="再生操作">
          <div class="control-group">
            <button id="play" class="control-button play-action" type="button" aria-pressed="false">
              <span class="button-icon" aria-hidden="true">▶</span><span class="button-text">再生</span>
            </button>
            <button id="resetState" class="control-button utility-action" type="button">
              <span class="button-icon" aria-hidden="true">↺</span>初期状態へ
            </button>
            <button id="step" class="control-button step-action" type="button">
              <span class="button-icon" aria-hidden="true">›</span>1ステップ
            </button>
            <button id="step10" class="control-button step-action" type="button">
              <span class="button-icon" aria-hidden="true">»</span>10ステップ
            </button>
          </div>
          <div class="control-group playback-settings" aria-label="再生速度">
            <label class="compact-field" for="playSteps">進める数
              <input id="playSteps" name="playSteps" form="settings" type="number" min="1" max="500" step="1" value="1">
            </label>
            <label class="compact-field" for="playInterval">間隔 ms
              <input id="playInterval" name="playInterval" form="settings" type="number" min="40" max="3000" step="10" value="160">
            </label>
          </div>
          <div class="control-group">
            <button id="savePng" class="control-button save-action" type="button">
              <span class="button-icon" aria-hidden="true">⬇</span>PNG保存
            </button>
          </div>
        </div>
      </div>

      <div class="visual">
        <img id="preview" alt="Voronoi Life simulation">
      </div>

      <div class="statusbar">
        <div class="stats" aria-live="polite">
          <div class="stat" tabindex="0" aria-describedby="hintStep">
            <span>step</span><strong id="statStep">0</strong>
            <span id="hintStep" class="hint">現在までに進んだ世代数です。1ステップごとに全セルの次の状態を計算します。</span>
          </div>
          <div class="stat" tabindex="0" aria-describedby="hintAlive">
            <span>alive</span><strong id="statAlive">0</strong>
            <span id="hintAlive" class="hint">alive のセル数と全体に占める割合です。binary 系ルールで主に使います。</span>
          </div>
          <div class="stat" tabindex="0" aria-describedby="hintDensity">
            <span>mean density</span><strong id="statDensity">0</strong>
            <span id="hintDensity" class="hint">各セルの密度の平均です。continuous ルールでは、画面全体の濃さの目安になります。</span>
          </div>
          <div class="stat" tabindex="0" aria-describedby="hintLife">
            <span>total life</span><strong id="statLife">0</strong>
            <span id="hintLife" class="hint">全セルが持っている life amount の合計です。continuous ルールでは総量の増減を確認できます。</span>
          </div>
          <div class="stat" tabindex="0" aria-describedby="hintDegree">
            <span>mean degree</span><strong id="statDegree">0</strong>
            <span id="hintDegree" class="hint">1つのセルが平均で何個の隣接セルを持つかです。点群の粗密や形の影響を受けます。</span>
          </div>
          <div class="stat" tabindex="0" aria-describedby="hintStatus">
            <span>state</span><strong id="statStatus" class="status-pill status-running">進行中</strong>
            <span id="hintStatus" class="hint">現在の判定状態です。定常は変化停止、振動は同じ形の周期的な繰り返しを示します。</span>
          </div>
        </div>
      </div>
      <div id="message" class="message">起動中...</div>
    </section>

    <aside>
      <div class="settings-action-bar">
        <button id="apply" class="primary control-button" type="button">
          <span class="button-icon" aria-hidden="true">✓</span>設定を適用
        </button>
      </div>
      <form id="settings">
        <fieldset>
          <legend>基本</legend>
          <div class="grid">
            <label>セル数
              <input name="cells" type="number" min="4" max="5000" step="1" value="300">
              <span class="hint">ボロノイセルの数です。多いほど細かく見えますが、計算は重くなります。</span>
            </label>
            <label>seed
              <input name="seed" type="number" step="1" value="1">
              <span class="hint">乱数の出発点です。同じ seed なら同じ点群と初期状態を再現できます。</span>
            </label>
            <label>点群
              <select name="pointMethod">
                <option value="random">random</option>
                <option value="jittered-grid">jittered-grid</option>
                <option value="density-gradient">density-gradient</option>
              </select>
              <span class="hint">セルの元になる点の並べ方です。空間の不均一さが変わります。</span>
            </label>
            <label>ルール
              <select name="ruleType" id="ruleType">
                <option value="absolute">absolute</option>
                <option value="density">density</option>
                <option value="probabilistic">probabilistic</option>
                <option value="continuous">continuous</option>
              </select>
              <span class="hint">次の状態を決める計算方法です。選ぶと下の設定欄と数式が切り替わります。</span>
            </label>
            <label>表示
              <select name="overlay" id="overlay">
                <option value="none">none</option>
                <option value="degree">degree</option>
                <option value="alive-count">alive-count</option>
                <option value="alive-density">alive-density</option>
                <option value="area">area</option>
                <option value="edge-length">edge-length</option>
              </select>
              <span class="hint">色で何を見るかを切り替えます。none は通常の生死または密度表示です。</span>
            </label>
            <label class="check">
              <input name="periodic" type="checkbox">
              <span class="check-text">
                <span>周期境界</span>
                <span class="hint">右端と左端、上端と下端がつながっているものとして計算します。</span>
              </span>
            </label>
            <label class="check">
              <input name="showHints" type="checkbox" checked>
              <span class="check-text">
                <span>ヒントを表示</span>
                <span class="hint">各パラメーターの意味を画面内に表示します。慣れたらオフにできます。</span>
              </span>
            </label>
          </div>
        </fieldset>

        <fieldset>
          <legend>初期状態</legend>
          <label>初期 alive 比率
            <span class="range-row">
              <input name="initialAliveRatio" type="range" min="0" max="1" step="0.01" value="0.28" data-output="initialAliveRatioOut">
              <output id="initialAliveRatioOut">0.28</output>
            </span>
            <span class="hint">binary 系の初期状態で、最初に生きているセルの割合です。</span>
          </label>
          <div class="grid">
            <label>continuous 初期値
              <select name="continuousInit">
                <option value="random_density">random_density</option>
                <option value="binary_density">binary_density</option>
                <option value="gaussian_blob">gaussian_blob</option>
              </select>
              <span class="hint">連続量ルールで、最初の密度をどう配るかを決めます。</span>
            </label>
            <label>初期密度上限
              <input name="initialDensityMax" type="number" min="0" max="10" step="0.01" value="1.0">
              <span class="hint">random_density と gaussian_blob の濃さの上限です。</span>
            </label>
            <label>alive 密度
              <input name="aliveDensity" type="number" min="0" max="10" step="0.01" value="1.0">
              <span class="hint">binary_density で alive になったセルに入れる密度です。</span>
            </label>
            <label>blob 幅
              <input name="sigma" type="number" min="0.000001" max="2" step="0.01" value="0.08">
              <span class="hint">gaussian_blob の広がりです。大きいほど中心の塊が広がります。</span>
            </label>
          </div>
          <div class="toolbar">
            <button id="randomizeState" type="button">初期状態を作り直す</button>
            <button id="regeneratePoints" type="button">点群を作り直す</button>
          </div>
        </fieldset>

        <fieldset>
          <legend>数式</legend>
          <p class="formula-help">現在の入力値が、次の状態の計算にどう入るかを表示します。</p>
          <div id="formulaBody" class="formula-body" aria-live="polite"></div>
        </fieldset>

        <fieldset class="rule-group" data-rule="absolute">
          <legend>absolute</legend>
          <div class="grid">
            <label>birth count
              <input name="birthCount" type="number" min="0" max="24" step="1" value="3">
              <span class="hint">死んでいるセルが alive になるために必要な alive 隣接セル数です。</span>
            </label>
            <label>survive counts
              <input name="surviveCounts" type="text" value="2,3">
              <span class="hint">alive のまま残る隣接セル数です。カンマ区切りで指定します。</span>
            </label>
          </div>
        </fieldset>

        <fieldset class="rule-group" data-rule="density">
          <legend>density</legend>
          <div class="grid">
            <label>birth min
              <input name="birthMin" type="number" min="0" max="1" step="0.01" value="0.30">
              <span class="hint">誕生に必要な alive 隣接割合の下限です。</span>
            </label>
            <label>birth max
              <input name="birthMax" type="number" min="0" max="1" step="0.01" value="0.45">
              <span class="hint">誕生に必要な alive 隣接割合の上限です。</span>
            </label>
            <label>survive min
              <input name="surviveMin" type="number" min="0" max="1" step="0.01" value="0.20">
              <span class="hint">生存に必要な alive 隣接割合の下限です。</span>
            </label>
            <label>survive max
              <input name="surviveMax" type="number" min="0" max="1" step="0.01" value="0.45">
              <span class="hint">生存に必要な alive 隣接割合の上限です。</span>
            </label>
          </div>
        </fieldset>

        <fieldset class="rule-group" data-rule="probabilistic">
          <legend>probabilistic</legend>
          <div class="grid">
            <label>birth threshold
              <input name="birthThreshold" type="number" min="0" max="1" step="0.01" value="0.25">
              <span class="hint">この密度を超えると、誕生確率が上がり始めます。</span>
            </label>
            <label>optimal density
              <input name="optimalDensity" type="number" min="0" max="5" step="0.01" value="0.35">
              <span class="hint">alive セルにとって望ましい密度です。離れるほど死亡確率が上がります。</span>
            </label>
            <label>birth strength
              <input name="birthStrength" type="number" min="0" max="20" step="0.1" value="2.5">
              <span class="hint">誕生確率の上がりやすさです。大きいほど増えやすくなります。</span>
            </label>
            <label>death strength
              <input name="deathStrength" type="number" min="0" max="20" step="0.1" value="2.0">
              <span class="hint">死亡確率の上がりやすさです。大きいほど環境から外れたセルが消えやすくなります。</span>
            </label>
          </div>
        </fieldset>

        <fieldset class="rule-group" data-rule="continuous">
          <legend>continuous</legend>
          <div class="grid">
            <label>coupling
              <select name="coupling">
                <option value="graph">graph</option>
                <option value="edge" selected>edge</option>
                <option value="edge_distance">edge_distance</option>
              </select>
              <span class="hint">隣接セル間で量が混ざる強さの決め方です。</span>
            </label>
            <label>reaction
              <select name="reaction">
                <option value="none">none</option>
                <option value="logistic">logistic</option>
                <option value="bell">bell</option>
              </select>
              <span class="hint">各セル内で量が自然に増減する効果です。</span>
            </label>
            <label>diffusion rate
              <input name="diffusionRate" type="number" min="0" max="10" step="0.001" value="0.01">
              <span class="hint">隣のセルへ密度がならされる速さです。大きいほど早く混ざります。</span>
            </label>
            <label>growth rate
              <input name="growthRate" type="number" min="0" max="10" step="0.001" value="0.02">
              <span class="hint">logistic / bell 反応で量が増える速さです。</span>
            </label>
            <label>death rate
              <input name="deathRate" type="number" min="0" max="10" step="0.001" value="0.01">
              <span class="hint">bell 反応で密度に応じて量が減る速さです。</span>
            </label>
            <label>capacity
              <input name="carryingCapacity" type="number" min="0.000001" max="10" step="0.01" value="1.0">
              <span class="hint">logistic 反応で、増加が止まりやすくなる密度の目安です。</span>
            </label>
            <label>rho max
              <input name="rhoMax" type="text" value="">
              <span class="hint">密度の上限です。空欄または none なら上限なしとして扱います。</span>
            </label>
            <label>濃淡範囲
              <select name="densityScale" id="densityScale">
                <option value="auto">auto</option>
                <option value="fixed">fixed</option>
              </select>
              <span class="hint">auto は毎回見やすく調整、fixed は rho max などで同じ濃淡範囲に固定します。</span>
            </label>
          </div>
        </fieldset>
      </form>
    </aside>
  </main>

  <script>
    const form = document.querySelector("#settings");
    const preview = document.querySelector("#preview");
    const message = document.querySelector("#message");
    const playButton = document.querySelector("#play");
    const playButtonIcon = playButton.querySelector(".button-icon");
    const playButtonText = playButton.querySelector(".button-text");
    const statusElement = document.querySelector("#statStatus");
    let playing = false;
    let playTimer = null;

    function formData(extra = {}) {
      const data = Object.fromEntries(new FormData(form).entries());
      data.periodic = form.elements.periodic.checked;
      return {...data, ...extra};
    }

    function numberValue(name, fallback) {
      const value = Number(form.elements[name].value);
      return Number.isFinite(value) ? value : fallback;
    }

    function fieldValue(name) {
      return form.elements[name]?.value ?? "";
    }

    function formatInput(name) {
      const value = Number(fieldValue(name));
      if (!Number.isFinite(value)) return fieldValue(name) || "none";
      if (Math.abs(value) >= 10) return value.toFixed(2).replace(/\\.00$/, "");
      return value.toFixed(3).replace(/0+$/, "").replace(/\\.$/, "");
    }

    async function callApi(path, body = {}) {
      message.textContent = "更新中...";
      message.classList.remove("warning");
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!payload.ok) {
        message.textContent = payload.error || "更新に失敗しました。";
        message.classList.add("warning");
        return payload;
      }
      renderPayload(payload);
      return payload;
    }

    function renderPayload(payload) {
      preview.src = payload.image;
      const stats = payload.stats;
      document.querySelector("#statStep").textContent = stats.step;
      document.querySelector("#statAlive").textContent = `${stats.aliveCount} (${formatPct(stats.aliveRatio)})`;
      document.querySelector("#statDensity").textContent = formatNumber(stats.meanDensity);
      document.querySelector("#statLife").textContent = formatNumber(stats.totalLife);
      document.querySelector("#statDegree").textContent = formatNumber(stats.meanDegree);
      statusElement.textContent = statusLabel(payload.stability);
      statusElement.className = `status-pill ${statusClass(payload.stability)}`;
      message.textContent = payload.message || `${stats.rule} / ${stats.points} / ${stats.boundary}`;
      message.classList.toggle("warning", Boolean(payload.stability?.stopped));
      if (payload.stability?.stopped && playing) {
        setPlaying(false);
      }
      if (payload.view && payload.view.overlay !== form.elements.overlay.value) {
        form.elements.overlay.value = payload.view.overlay;
      }
    }

    function statusLabel(stability) {
      if (!stability) return "進行中";
      if (stability.kind === "steady") return "定常";
      if (stability.kind === "oscillating") return `振動 ${stability.period}`;
      if (stability.kind === "not_tracked") return "判定対象外";
      return "進行中";
    }

    function statusClass(stability) {
      if (!stability) return "status-running";
      if (stability.kind === "steady") return "status-steady";
      if (stability.kind === "oscillating") return "status-oscillating";
      if (stability.kind === "not_tracked") return "status-not-tracked";
      return "status-running";
    }

    function formatPct(value) {
      return `${Math.round(value * 1000) / 10}%`;
    }

    function formatNumber(value) {
      if (Math.abs(value) >= 100) return value.toFixed(1);
      if (Math.abs(value) >= 1) return value.toFixed(3);
      return value.toPrecision(3);
    }

    function updateRuleGroups() {
      const rule = form.elements.ruleType.value;
      document.querySelectorAll(".rule-group").forEach(group => {
        group.hidden = group.dataset.rule !== rule;
      });
      const continuous = rule === "continuous";
      for (const option of form.elements.overlay.options) {
        option.disabled = continuous && (option.value === "alive-count" || option.value === "alive-density");
      }
      if (continuous && (form.elements.overlay.value === "alive-count" || form.elements.overlay.value === "alive-density")) {
        form.elements.overlay.value = "none";
      }
      updateFormula();
    }

    function updateOutputs() {
      document.querySelectorAll("input[type='range'][data-output]").forEach(input => {
        document.getElementById(input.dataset.output).textContent = Number(input.value).toFixed(2);
      });
      updateFormula();
      updateHintVisibility();
    }

    function updateHintVisibility() {
      document.body.classList.toggle("hints-off", !form.elements.showHints.checked);
    }

    function updateFormula() {
      const rule = form.elements.ruleType.value;
      const formulaBody = document.querySelector("#formulaBody");
      const lines = [
        ["記号", "n_i = cell i の生きている隣接セル数; deg_i = cell i の隣接セル数"],
      ];

      if (rule === "absolute") {
        lines.push(initialAliveFormulaLine());
        lines.push(
          ["誕生", `dead cell becomes alive when n_i == birth_count = ${formatInput("birthCount")}`],
          ["生存", `alive cell stays alive when n_i is in {${fieldValue("surviveCounts") || "2,3"}}`],
        );
      } else if (rule === "density") {
        lines.push(initialAliveFormulaLine());
        lines.push(
          ["密度", "d_i = n_i / deg_i"],
          ["誕生", `dead cell becomes alive when ${formatInput("birthMin")} <= d_i <= ${formatInput("birthMax")}`],
          ["生存", `alive cell stays alive when ${formatInput("surviveMin")} <= d_i <= ${formatInput("surviveMax")}`],
        );
      } else if (rule === "probabilistic") {
        lines.push(initialAliveFormulaLine());
        lines.push(
          ["密度", "d_i = n_i / deg_i"],
          ["誕生確率", `P_birth = clamp(${formatInput("birthStrength")} * (d_i - ${formatInput("birthThreshold")}), 0, 1)`],
          ["死亡確率", `P_death = clamp(${formatInput("deathStrength")} * abs(d_i - ${formatInput("optimalDensity")}), 0, 1)`],
          ["判定", "dead cell is born when random < P_birth; alive cell dies when random < P_death"],
        );
      } else {
        const coupling = fieldValue("coupling");
        const reaction = fieldValue("reaction");
        lines.push(
          ["初期密度", continuousInitFormula()],
          ["密度", "rho_i = life_amount_i / area_i"],
          ["拡散", `flow(i <- j) = ${formatInput("diffusionRate")} * weight(i,j) * (rho_j - rho_i)`],
          ["weight", couplingFormula(coupling)],
          ["反応", reactionFormula(reaction)],
          ["上限", rhoMaxFormula()],
        );
      }

      formulaBody.replaceChildren(...lines.map(([label, formula]) => formulaLine(label, formula)));
    }

    function initialAliveFormulaLine() {
      return ["初期 alive", `alive_i(0) is true with probability initial_alive_ratio = ${formatInput("initialAliveRatio")}`];
    }

    function continuousInitFormula() {
      const init = fieldValue("continuousInit");
      if (init === "binary_density") {
        return `rho_i(0) = ${formatInput("aliveDensity")} with probability ${formatInput("initialAliveRatio")}, otherwise 0`;
      }
      if (init === "gaussian_blob") {
        return `rho_i(0) = ${formatInput("initialDensityMax")} * exp(-distance_to_center_i^2 / ${formatInput("sigma")}^2)`;
      }
      return `rho_i(0) is sampled from Uniform(0, ${formatInput("initialDensityMax")})`;
    }

    function couplingFormula(coupling) {
      if (coupling === "graph") return "weight(i,j) = 1";
      if (coupling === "edge_distance") return "weight(i,j) = shared_edge_length(i,j) / center_distance(i,j)";
      return "weight(i,j) = shared_edge_length(i,j)";
    }

    function reactionFormula(reaction) {
      if (reaction === "logistic") {
        return `reaction_delta_i = ${formatInput("growthRate")} * rho_i * (1 - rho_i / ${formatInput("carryingCapacity")}) * area_i`;
      }
      if (reaction === "bell") {
        return `reaction_delta_i = (${formatInput("growthRate")} * exp(-((rho_i - ${formatInput("optimalDensity")})^2) / (2 * ${formatInput("sigma")}^2)) - ${formatInput("deathRate")} * rho_i) * area_i`;
      }
      return "reaction_delta_i = 0";
    }

    function rhoMaxFormula() {
      const value = fieldValue("rhoMax").trim();
      if (!value || value.toLowerCase() === "none") return "rho_max is off; life_amount is only clipped at 0";
      return `life_amount_i is capped at rho_max * area_i = ${value} * area_i`;
    }

    function formulaLine(label, formula) {
      const row = document.createElement("div");
      row.className = "formula-line";
      const title = document.createElement("strong");
      title.textContent = label;
      const code = document.createElement("code");
      code.textContent = formula;
      row.append(title, code);
      return row;
    }

    async function playLoop() {
      if (!playing) return;
      const steps = numberValue("playSteps", 1);
      await callApi("/api/step", formData({steps}));
      if (!playing) return;
      const interval = Math.max(40, numberValue("playInterval", 160));
      playTimer = window.setTimeout(playLoop, interval);
    }

    function setPlaying(next) {
      playing = next;
      playButtonText.textContent = playing ? "停止" : "再生";
      playButtonIcon.textContent = playing ? "■" : "▶";
      playButton.classList.toggle("is-stopping", playing);
      playButton.setAttribute("aria-pressed", String(playing));
      if (playTimer) {
        window.clearTimeout(playTimer);
        playTimer = null;
      }
      if (playing) playLoop();
    }

    document.querySelector("#apply").addEventListener("click", () => {
      updateRuleGroups();
      callApi("/api/reset", formData());
    });
    document.querySelector("#step").addEventListener("click", () => callApi("/api/step", formData({steps: 1})));
    document.querySelector("#step10").addEventListener("click", () => callApi("/api/step", formData({steps: 10})));
    document.querySelector("#resetState").addEventListener("click", () => callApi("/api/reset-state", formData()));
    document.querySelector("#randomizeState").addEventListener("click", () => callApi("/api/randomize-state", formData()));
    document.querySelector("#regeneratePoints").addEventListener("click", () => callApi("/api/regenerate-points", formData()));
    document.querySelector("#savePng").addEventListener("click", () => callApi("/api/save-png", formData()));
    playButton.addEventListener("click", () => setPlaying(!playing));

    form.elements.ruleType.addEventListener("change", updateRuleGroups);
    form.elements.overlay.addEventListener("change", () => callApi("/api/view", formData()));
    form.elements.densityScale.addEventListener("change", () => callApi("/api/view", formData()));
    form.elements.rhoMax.addEventListener("change", () => callApi("/api/view", formData()));
    form.elements.showHints.addEventListener("change", updateHintVisibility);
    form.addEventListener("input", updateOutputs);

    updateRuleGroups();
    updateOutputs();
    callApi("/api/reset", formData());
  </script>
</body>
</html>
"""
