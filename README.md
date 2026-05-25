# DL Final Project

This repository contains the code, scripts, documentation, and representative results for the DL final project.

## Repository Contents

- `framework/`
  - Training and inference source code.
  - Conda environment file: `framework/environment.yml`.
- `scripts/`
  - Dataset download and batch inference helper scripts.
- `DOC/`
  - Proposal, paper notes, implementation plan, experiment log, code-change log, and packaging guide.
- `results/bbox_corner_compare/`
  - A/B/C comparison figures for bbox controllability.
- `results/bbox_spoon_compare/`
  - Spoon test 0 comparison figures.
- `logs/*_metrics.csv`
  - Training metrics CSV files.

Large local artifacts are intentionally not committed:

- `data/`
- `checkpoints/`
- full `results/`
- runtime `logs/*.log`

See `DOC/github_packaging.md` for details.

## Environment Setup

```bash
conda env create -n DL_Final -f framework/environment.yml
conda activate DL_Final
```

## Dataset

The PIPE dataset is large and should be downloaded locally instead of committed to GitHub.

```bash
python scripts/download_pipe_data.py
```

Expected local cache location:

```text
data/hf_cache/
```

## Checkpoints

Checkpoints are not included in the Git repository.

Place separately shared checkpoints under:

```text
checkpoints/strategy_a/
checkpoints/strategy_b/
checkpoints/strategy_c/
```

Important final checkpoints:

```text
checkpoints/strategy_a/strategy_a_final_epoch_0010_step_002560_lora/
checkpoints/strategy_b/strategy_b_final_epoch_0010_step_002560_lora/
checkpoints/strategy_c/strategy_c_final_epoch_0010_step_002560_controlnet/
```

## Representative Results

Useful comparison figures:

```text
results/bbox_corner_compare/
results/bbox_spoon_compare/
```

Experiment details and commands are documented in:

```text
DOC/experiment_log.md
DOC/codex_log.md
DOC/github_packaging.md
```

