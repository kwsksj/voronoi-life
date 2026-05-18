from __future__ import annotations

from voronoi_life.web_gui import (
    GuiSession,
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


def test_gui_config_maps_absolute_count_ranges() -> None:
    config = build_simulation_config(
        {
            "ruleType": "absolute",
            "birthMinCount": "2",
            "birthMaxCount": "4",
            "surviveMinCount": "1",
            "surviveMaxCount": "5",
        }
    )

    assert config.rule.rule_type == "absolute"
    assert config.rule.birth_min_count == 2
    assert config.rule.birth_max_count == 4
    assert config.rule.survive_min_count == 1
    assert config.rule.survive_max_count == 5


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
    assert payload["stability"]["kind"] == "running"


def test_gui_step_replays_oscillation_after_detection() -> None:
    session = GuiSession()
    raw = {
        "cells": "20",
        "seed": "1",
        "initialAliveRatio": "0",
        "ruleType": "absolute",
        "birthCount": "0",
        "surviveCounts": "0",
    }

    session.reset(raw)
    payload = session.step({**raw, "steps": "2"})

    assert payload["stability"]["kind"] == "oscillating"
    assert payload["stability"]["period"] == 2
    assert payload["stability"]["detected_step"] == 2
    assert payload["stats"]["step"] == 2

    payload = session.step({**raw, "steps": "1"})

    assert payload["stability"]["kind"] == "oscillating"
    assert payload["stability"]["detected_step"] == 2
    assert payload["stats"]["step"] == 3
    assert payload["message"].startswith("振動状態を繰り返し表示しています")


def test_gui_html_contains_formula_and_hint_controls() -> None:
    assert "数式" in INDEX_HTML
    assert "formulaBody" in INDEX_HTML
    assert 'name="birthMinCount"' in INDEX_HTML
    assert 'name="birthMaxCount"' in INDEX_HTML
    assert 'name="surviveMinCount"' in INDEX_HTML
    assert 'name="surviveMaxCount"' in INDEX_HTML
    assert "ヒントを表示" in INDEX_HTML
    assert "hints-off" in INDEX_HTML
    assert "label:hover .hint" in INDEX_HTML
    assert "label:focus-within .hint" in INDEX_HTML
    assert ".hints-off label" in INDEX_HTML
    assert "statStatus" in INDEX_HTML


def test_gui_html_contains_status_hints() -> None:
    assert 'tabindex="0" aria-describedby="hintStep"' in INDEX_HTML
    assert 'tabindex="0" aria-describedby="hintAlive" data-visible-when="binary"' in INDEX_HTML
    assert 'tabindex="0" aria-describedby="hintDensity" data-visible-when="continuous"' in INDEX_HTML
    assert 'tabindex="0" aria-describedby="hintMaxDensity" data-visible-when="continuous"' in INDEX_HTML
    assert 'tabindex="0" aria-describedby="hintLife" data-visible-when="continuous"' in INDEX_HTML
    assert 'tabindex="0" aria-describedby="hintDegree"' in INDEX_HTML
    assert 'tabindex="0" aria-describedby="hintStatus"' in INDEX_HTML
    assert "現在までに進んだ世代数です" in INDEX_HTML
    assert 'id="statMaxDensity"' in INDEX_HTML
    assert ".stat:hover .hint" in INDEX_HTML
    assert ".stat:focus .hint" in INDEX_HTML
    assert ".stat:focus-within .hint" in INDEX_HTML
    assert "color: #ffffff;" in INDEX_HTML


def test_gui_html_contains_mode_visibility_controls() -> None:
    assert 'data-visible-when="binary init:binary_density"' in INDEX_HTML
    assert 'data-visible-when="init:random_density init:gaussian_blob"' in INDEX_HTML
    assert 'data-visible-when="init:gaussian_blob reaction:bell"' in INDEX_HTML
    assert 'data-visible-when="reaction:logistic reaction:bell"' in INDEX_HTML
    assert 'data-visible-when="reaction:bell"' in INDEX_HTML
    assert "function updateConditionalVisibility()" in INDEX_HTML
    assert "function visibilityTokenMatches(token, rule, init, reaction)" in INDEX_HTML


def test_gui_html_hides_binary_overlays_for_continuous_mode() -> None:
    assert '<option value="alive-count" data-overlay-mode="binary">alive-count</option>' in INDEX_HTML
    assert '<option value="alive-density" data-overlay-mode="binary">alive-density</option>' in INDEX_HTML
    assert 'option.hidden = continuous && binaryOnly' in INDEX_HTML
    assert 'option.disabled = continuous && binaryOnly' in INDEX_HTML
    assert 'form.elements.overlay.value = "none"' in INDEX_HTML
    assert "function overlayForCurrentRule(overlay)" in INDEX_HTML
    assert "function isBinaryOverlay(overlay)" in INDEX_HTML


def test_gui_html_shares_optimal_density_between_probabilistic_and_continuous_bell() -> None:
    assert 'data-visible-when="probabilistic reaction:bell"' in INDEX_HTML
    assert 'name="optimalDensity"' in INDEX_HTML
    assert "probabilistic では alive セルにとって望ましい隣接密度" in INDEX_HTML


def test_gui_html_places_playback_controls_in_topbar() -> None:
    assert 'class="control-group playback-settings"' in INDEX_HTML
    assert 'name="playSteps" form="settings"' in INDEX_HTML
    assert 'name="playInterval" form="settings"' in INDEX_HTML
    assert 'legend>再生</legend>' not in INDEX_HTML


def test_gui_html_places_apply_and_reset_near_related_controls() -> None:
    aside_index = INDEX_HTML.index("<aside>")
    apply_index = INDEX_HTML.index('id="apply"')
    form_index = INDEX_HTML.index('<form id="settings">')
    play_index = INDEX_HTML.index('id="play"')
    reset_index = INDEX_HTML.index('id="resetState"')
    step_index = INDEX_HTML.index('id="step"')

    assert aside_index < apply_index < form_index
    assert play_index < reset_index < step_index


def test_gui_html_contains_status_pill_and_play_button_states() -> None:
    assert "status-pill status-running" in INDEX_HTML
    assert "status-steady" in INDEX_HTML
    assert "status-oscillating" in INDEX_HTML
    assert "status-not-tracked" in INDEX_HTML
    assert 'aria-pressed="false"' in INDEX_HTML
    assert 'playButton.setAttribute("aria-pressed", String(playing))' in INDEX_HTML
    assert 'payload.stability?.stopped && !isOscillating && playing' in INDEX_HTML
