from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from .output import default_output_dir, write_experiment_json
from .render import Overlay, draw_state, save_gif, save_png
from .rules import RuleConfig
from .simulation import SimulationConfig, VoronoiLife


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_command(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voronoi-life")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="run a Voronoi Life experiment")
    run.add_argument("--cells", type=int, default=300)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--steps", type=int, default=200)
    run.add_argument("--initial-alive-ratio", type=float, default=0.28)
    run.add_argument(
        "--points",
        choices=["random", "jittered-grid", "density-gradient"],
        default="random",
    )
    run.add_argument(
        "--rule",
        choices=["absolute", "density", "probabilistic"],
        default="absolute",
    )
    run.add_argument("--periodic", action=argparse.BooleanOptionalAction, default=False)
    run.add_argument(
        "--overlay",
        choices=["none", "degree", "alive-count", "alive-density"],
        default="none",
    )
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--no-output", action="store_true")
    run.add_argument("--gif", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--fps", type=int, default=12)
    run.add_argument("--show", action=argparse.BooleanOptionalAction, default=None)

    run.add_argument("--birth-count", type=int, default=3)
    run.add_argument("--survive-counts", default="2,3")
    run.add_argument("--birth-min", type=float, default=0.30)
    run.add_argument("--birth-max", type=float, default=0.45)
    run.add_argument("--survive-min", type=float, default=0.20)
    run.add_argument("--survive-max", type=float, default=0.45)
    run.add_argument("--birth-threshold", type=float, default=0.25)
    run.add_argument("--optimal-density", type=float, default=0.35)
    run.add_argument("--birth-strength", type=float, default=2.5)
    run.add_argument("--death-strength", type=float, default=2.0)
    return parser


def run_command(args: argparse.Namespace) -> int:
    if args.cells < 4:
        raise SystemExit("--cells must be at least 4")
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")

    rule = RuleConfig(
        rule_type=args.rule,
        birth_count=args.birth_count,
        survive_counts=parse_counts(args.survive_counts),
        birth_min=args.birth_min,
        birth_max=args.birth_max,
        survive_min=args.survive_min,
        survive_max=args.survive_max,
        birth_threshold=args.birth_threshold,
        optimal_density=args.optimal_density,
        birth_strength=args.birth_strength,
        death_strength=args.death_strength,
    )
    config = SimulationConfig(
        cells=args.cells,
        seed=args.seed,
        initial_alive_ratio=args.initial_alive_ratio,
        point_method=args.points,
        periodic=args.periodic,
        rule=rule,
    )

    simulation = VoronoiLife(config)
    states = simulation.run(args.steps, include_initial=True)
    overlay = args.overlay
    rule_label = args.rule

    outputs: dict[str, str] = {}
    output_dir = None
    if not args.no_output:
        output_dir = args.output or default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / "final.png"
        save_png(
            png_path,
            simulation.space,
            simulation.alive,
            overlay,
            simulation.step_index,
            rule_label,
        )
        outputs["png"] = str(png_path)

        if args.gif:
            gif_path = output_dir / "animation.gif"
            save_gif(gif_path, simulation.space, states, overlay, rule_label, fps=args.fps)
            outputs["gif"] = str(gif_path)

        json_path = output_dir / "experiment.json"
        outputs["json"] = str(json_path)
        write_experiment_json(json_path, config, args.steps, overlay, outputs)

    print_summary(config, args.steps, outputs)

    show = sys.stdout.isatty() if args.show is None else args.show
    if show:
        run_interactive(simulation, overlay, output_dir)

    return 0


def parse_counts(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("counts must be comma-separated integers") from exc


def print_summary(
    config: SimulationConfig,
    steps: int,
    outputs: dict[str, str],
) -> None:
    print(
        f"completed: cells={config.cells} steps={steps} seed={config.seed} "
        f"points={config.point_method} rule={config.rule.rule_type} "
        f"boundary={'periodic' if config.periodic else 'open'}"
    )
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


def run_interactive(
    simulation: VoronoiLife,
    overlay: Overlay,
    output_dir: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    running = {"value": False}

    def redraw() -> None:
        draw_state(
            ax,
            simulation.space,
            simulation.alive,
            overlay,
            simulation.step_index,
            simulation.config.rule.rule_type,
            add_colorbar=False,
        )
        fig.canvas.draw_idle()

    def tick() -> bool:
        if running["value"]:
            simulation.step()
            redraw()
        return True

    def save_current_png() -> None:
        target_dir = output_dir or Path("runs") / "interactive"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"step-{simulation.step_index:05d}.png"
        save_png(
            path,
            simulation.space,
            simulation.alive,
            overlay,
            simulation.step_index,
            simulation.config.rule.rule_type,
        )
        print(f"saved: {path}")

    def on_key(event) -> None:
        if event.key == " ":
            running["value"] = not running["value"]
        elif event.key == "n":
            simulation.step()
            redraw()
        elif event.key == "r":
            simulation.reset_state()
            redraw()
        elif event.key == "g":
            simulation.regenerate_points()
            redraw()
        elif event.key == "s":
            save_current_png()
        elif event.key == "q":
            plt.close(fig)

    redraw()
    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(tick)
    timer.start()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
