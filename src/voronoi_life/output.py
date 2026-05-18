from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .simulation import SimulationConfig


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / stamp


def write_experiment_json(
    path: Path,
    config: SimulationConfig,
    steps: int,
    overlay: str,
    outputs: dict[str, str],
    stability_status: dict[str, object] | None = None,
) -> None:
    payload = {
        "cell_count": config.cells,
        "seed": config.seed,
        "point_generation_method": config.point_method,
        "boundary_condition": "periodic" if config.periodic else "open",
        "rule_type": config.rule.rule_type,
        "rule_parameters": config.rule.to_json_dict(),
        "initial_alive_ratio": config.initial_alive_ratio,
        "number_of_steps": steps,
        "overlay": overlay,
        "outputs": outputs,
    }
    if stability_status is not None:
        payload["stability_status"] = stability_status
    if config.rule.rule_type == "continuous":
        payload["continuous_parameters"] = config.rule.continuous_json_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
