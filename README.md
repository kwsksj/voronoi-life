# voronoi-life

Experimental cellular automata on Voronoi tessellations.

## Setup

```bash
uv sync
```

## Run

```bash
uv run voronoi-life run --cells 300 --rule absolute --steps 200 --seed 1
```

Outputs are written under `runs/` by default:

- `final.png`
- `animation.gif`
- `experiment.json`

## Examples

Compare the absolute-count rule and density rule with the same seed:

```bash
uv run voronoi-life run --cells 200 --steps 50 --seed 1 --rule absolute
uv run voronoi-life run --cells 200 --steps 50 --seed 1 --rule density
```

Try different point fields:

```bash
uv run voronoi-life run --points random
uv run voronoi-life run --points jittered-grid
uv run voronoi-life run --points density-gradient
```

Enable periodic boundaries:

```bash
uv run voronoi-life run --periodic
```

## Tests

```bash
uv run pytest
```
