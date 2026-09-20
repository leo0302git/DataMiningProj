# DataMiningProj

Course project for interpretable data mining (2026 Fall).

## Scope

- DIA: binary classification using RDKit numerical descriptors; SMILES is retained only as an identifier and is not a model feature.
- Abalone: regression with `Rings` as the target.
- Models: RRL, a linear baseline, a tree-ensemble baseline, and a manually implemented generalized linear model.

## Repository layout

- `data/`: local UCI source files and processed data; not committed.
- `models/rrl/`: a fixed local copy of upstream RRL. See `UPSTREAM.md` in that directory.
- `models/linear/`, `models/tree_ensemble/`, `models/manual_glm/`: comparison-model implementations.
- `prepare_data.py`, `evaluate.py`, `run_experiments.py`: project pipeline entry points (to be implemented incrementally).

## Setup

```bash
uv sync
```

The data files are downloaded locally during repository bootstrap and intentionally excluded from Git.
