# GitHub Packaging Guide

## What to Commit

Commit these files to GitHub:

- `framework/`
  - Source code and `framework/environment.yml`.
- `scripts/`
  - Dataset download, inference batch, and result stacking scripts.
- `DOC/`
  - Proposal, paper PDF if allowed, rule notes, logs, implementation plan, and experiment notes.
- `logs/*_metrics.csv`
  - Small metrics CSV files are useful for reproducibility.
- `.gitignore`
  - Prevents large local artifacts from being committed.
- Selected representative figures:
  - `results/bbox_corner_compare/*.png`
  - `results/bbox_spoon_compare/*.png`

## What Not to Commit

Do not commit these directories to normal GitHub:

- `data/`
  - About 157 GB. This contains Hugging Face PIPE dataset/model cache.
- `checkpoints/`
  - About 18 GB. Strategy C checkpoints are about 1.4 GB each.
- `download/`
  - Raw downloads or temporary setup artifacts.
- Full `results/`
  - About 330 MB. Only selected comparison PNGs are unignored.
- `logs/*.log`
  - Runtime logs can be long and noisy. Keep important summaries in `DOC/codex_log.md`.
- `__pycache__/`, `*.pyc`
  - Python generated cache files.

## How to Share Large Files

Use an external storage location for large artifacts:

- Dataset: ask teammates to run `scripts/download_pipe_data.py`.
- Strategy A/B final LoRA checkpoints:
  - Small enough to share separately if needed, but still better as a release artifact.
- Strategy C final ControlNet checkpoint:
  - `checkpoints/strategy_c/strategy_c_final_epoch_0010_step_002560_controlnet/`
  - About 1.4 GB, so do not commit it to git.
  - Share via Google Drive, Hugging Face Hub, or GitHub Release asset.

Recommended external package:

```text
DL_Final_large_artifacts/
├── checkpoints/
│   ├── strategy_a/strategy_a_final_epoch_0010_step_002560_lora/
│   ├── strategy_b/strategy_b_final_epoch_0010_step_002560_lora/
│   └── strategy_c/strategy_c_final_epoch_0010_step_002560_controlnet/
└── README_large_artifacts.txt
```

## Suggested Git Commands

From the project root:

```bash
git init
git status --short
git add .gitignore DOC framework scripts logs/*.csv results/bbox_corner_compare results/bbox_spoon_compare
git status --short
git commit -m "Prepare final project code and representative results"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Before pushing, check that no huge files are staged:

```bash
git diff --cached --name-only
git diff --cached --stat
```

If a large file appears in staging, unstage it:

```bash
git restore --staged <path>
```

## Teammate Setup

After cloning:

```bash
conda env create -n DL_Final -f framework/environment.yml
conda activate DL_Final
python scripts/download_pipe_data.py
```

If large checkpoints are shared separately, place them back under:

```text
checkpoints/strategy_a/
checkpoints/strategy_b/
checkpoints/strategy_c/
```

Then run inference scripts as documented in `DOC/experiment_log.md`.

