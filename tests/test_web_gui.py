from __future__ import annotations

from voronoi_life.web_gui import (
    INDEX_HTML,
    ViewConfig,
    build_simulation_config,
    build_view_config,
    create_state_payload,
)
from voronoi_life.simulation import VoronoiLife


def test_gui_config_maps_form_values_to_simulation_config() -> None:
    config = build_simulation_config(
        {
            "cells": "120",
            "seed": "42",
            "initialAliveRatio": "0.4",
            "pointMethod": "jittered-grid",
            "periodic": True,
            "ruleType": "density",
            "birthMin": "0.2",
            "birthMax": "0.5",
            "surviveMin": "0.1",
            "surviveMax": "0.6",
        }
    )

    assert config.cells == 120
    assert config.seed == 42
    assert config.initial_alive_ratio == 0.4
    assert config.point_method == "jittered-grid"
    assert config.periodic is True
    assert config.rule.rule_type == "density"
    assert config.rule.birth_min == 0.2
    assert config.rule.birth_max == 0.5
    assert config.rule.survive_min == 0.1
    assert config.rule.survive_max == 0.6
    assert config.include_edge_metrics is True


def test_gui_view_avoids_binary_overlays_for_continuous_rule() -> None:
    config = build_simulation_config({"ruleType": "continuous"})
    view = build_view_config({"overlay": "alive-count"}, config.rule)

    assert view.overlay == "none"


def test_gui_state_payload_contains_rendered_image_and_stats() -> None:
    config = build_simulation_config({"cells": "20", "seed": "3"})
    simulation = VoronoiLife(config)
    payload = create_state_payload(simulation, ViewConfig())

    assert payload["ok"] is True
    assert payload["image"].startswith("data:image/png;base64,")
    assert payload["stats"]["cells"] == 20
    assert payload["stats"]["step"] == 0


def test_gui_html_contains_formula_and_hint_controls() -> None:
    assert "数式" in INDEX_HTML
    assert "formulaBody" in INDEX_HTML
    assert "ヒントを表示" in INDEX_HTML
    assert "hints-off" in INDEX_HTML
