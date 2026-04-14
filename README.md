# pocket-vep

Pocket variant effect predictor.

## Layout

- `env/` — environment spec (conda or pip)
- `data/` — raw and processed data (gitignored)
- `notebooks/` — exploratory notebooks, one per person
- `src/` — importable shared code
  - `features/` — feature extraction
  - `models/` — model definitions
  - `eval/` — evaluation utilities
- `scripts/` — runnable entry points
- `results/` — outputs (gitignored except final figures)

## Setup

```
conda env create -f env/environment.yml
conda activate pocket-vep
```
