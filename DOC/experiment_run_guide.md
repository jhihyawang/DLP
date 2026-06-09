# Experiment Run Guide

This guide explains how to run training, inference, ablation, bbox-variation experiments, and evaluation for the bbox-guided object-addition project.

## 1. Environment

Use the project conda environment:

```bash
conda activate DL_Final
cd ~/NYCU_DL_Final_Project
```

If the environment must be rebuilt:

```bash
conda env create -n DL_Final -f framework/environment.yml
```

If the environment already exists and needs new packages:

```bash
conda env update -n DL_Final -f framework/environment.yml
```

Important runtime defaults:

```text
base model: paint-by-inpaint/add-base
dataset: paint-by-inpaint/PIPE
masks: paint-by-inpaint/PIPE_Masks
image size: 512
train samples: 256
epochs: 10
batch size: 1
learning rate: 1e-5
seed: 1234
inference steps: 50
guidance scale: 7.0
image guidance scale: 1.5
```

The scripts automatically choose cache/checkpoint roots. If the external data disk is writable, checkpoints are stored under:

```text
/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def/checkpoints
```

Otherwise, they fall back to project-local paths.

## 2. Main A/B/C/C+ Training and Inference

The main experiment compares:

```text
Strategy A: input-level bbox conditioning
Strategy B: LoRA with bbox-weighted loss and inference-time constraint
Strategy C: single-channel ControlNet-style bbox conditioning
Strategy C+: inner/outer two-channel ControlNet conditioning
```

Run all four strategies, including training and inference:

```bash
conda run -n DL_Final bash scripts/run_all_strategies_train_infer.sh
```

Default outputs:

```text
checkpoints: /media/.../checkpoints/all_strategies_matched
results: results/all_strategies_matched
logs: logs/all_strategies_matched
```

Run only selected strategies:

```bash
STRATEGIES="c cplus" \
conda run -n DL_Final bash scripts/run_all_strategies_train_infer.sh
```

Run training only:

```bash
RUN_INFER=0 \
conda run -n DL_Final bash scripts/run_all_strategies_train_infer.sh
```

Run inference only from existing checkpoints:

```bash
RUN_TRAIN=0 \
conda run -n DL_Final bash scripts/run_all_strategies_train_infer.sh
```

Use different test indices:

```bash
RUN_TRAIN=0 INDICES="0 1 2 3 4" \
conda run -n DL_Final bash scripts/run_all_strategies_train_infer.sh
```

Run the full PIPE test split for fair quantitative evaluation:

```bash
RUN_TRAIN=0 INDICES=all MAX_EVAL_SAMPLES=0 \
conda run -n DL_Final bash scripts/run_all_strategies_train_infer.sh
```

Run a quick PIPE pilot on the first 100 test samples:

```bash
RUN_TRAIN=0 INDICES=all MAX_EVAL_SAMPLES=100 \
conda run -n DL_Final bash scripts/run_all_strategies_train_infer.sh
```

Enable bbox sweep during the main inference script:

```bash
RUN_TRAIN=0 RUN_BBOX_SWEEP=1 BBOX_SWEEP_MODE=corners \
conda run -n DL_Final bash scripts/run_all_strategies_train_infer.sh
```

Supported main sweep modes:

```text
corners
small-corners
vertical
```

## 3. Final Demo BBox Variation Inference

This experiment uses existing A/B/C/C+ checkpoints and generates bbox sweeps for final demo figures.

Run bbox variation inference:

```bash
conda run -n DL_Final bash scripts/run_bbox_variation_grid_infer.sh
```

Default outputs:

```text
results: results/final_demo_bbox_variation
logs: logs/final_demo_bbox_variation
```

By default, this script keeps only grid images:

```text
KEEP_ONLY_GRID=1
```

This is good for demo figures but not ideal for automatic evaluation. To keep `source`, `target`, `info`, overlays, and individual output images:

```bash
KEEP_ONLY_GRID=0 \
conda run -n DL_Final bash scripts/run_bbox_variation_grid_infer.sh
```

Run only specific strategies:

```bash
STRATEGIES="b c cplus" \
conda run -n DL_Final bash scripts/run_bbox_variation_grid_infer.sh
```

Run only one sweep mode:

```bash
SWEEP_MODES="corners" \
conda run -n DL_Final bash scripts/run_bbox_variation_grid_infer.sh
```

Run both default sweep modes:

```bash
SWEEP_MODES="corners small-corners" \
conda run -n DL_Final bash scripts/run_bbox_variation_grid_infer.sh
```

## 4. Fixed-Location Ablation

The fixed-location ablation tests whether bbox control is truly learned. Training objects are relocated to a fixed placement such as top-left, then inference tests several bbox locations:

```text
zero
top-left
center
bottom-right
original
```

Run the default ablation. By default, `MODE=cplus`:

```bash
conda run -n DL_Final bash scripts/run_fixed_location_ablation.sh
```

Run both C and C+ ablations:

```bash
MODE=both \
conda run -n DL_Final bash scripts/run_fixed_location_ablation.sh
```

Run only Strategy C:

```bash
MODE=c \
conda run -n DL_Final bash scripts/run_fixed_location_ablation.sh
```

Run only inference from existing ablation checkpoints:

```bash
RUN_TRAIN=0 MODE=both \
conda run -n DL_Final bash scripts/run_fixed_location_ablation.sh
```

Run a different fixed training placement:

```bash
BBOX_PLACEMENT=bottom-right MODE=both \
conda run -n DL_Final bash scripts/run_fixed_location_ablation.sh
```

Default outputs:

```text
checkpoints: /media/.../checkpoints/fixed_location_ablation
results: results/fixed_location_ablation
logs: logs/fixed_location_ablation
```

## 5. MagicBrush Generalization Test

This test evaluates cross-dataset generalization. The model is still trained on PIPE; no MagicBrush fine-tuning is performed.

Dataset:

```text
osunlp/MagicBrush
```

MagicBrush provides:

```text
source_img
target_img
instruction
mask_img
img_id
turn_index
```

The project converts `mask_img` into an object/edit bbox and then reuses the same A/B/C/C+ inference pipelines.

Run the default MagicBrush generalization test:

```bash
conda run -n DL_Final bash scripts/run_magicbrush_generalization.sh
```

Default outputs:

```text
results: results/magicbrush_generalization
logs: logs/magicbrush_generalization
```

Default setting:

```text
checkpoint source: all_strategies_matched
MagicBrush split: dev
indices: 0 1 2
max_samples: 0
strategies: a b c cplus
```

For quantitative generalization evaluation, run inference on every sample in the MagicBrush dev split. The evaluator can only score samples that already have generated `source` / `target` / `output` / `info` files.

Full dev split:

```bash
INDICES=all MAX_SAMPLES=0 NO_BASELINE=1 \
conda run -n DL_Final bash scripts/run_magicbrush_generalization.sh
```

Quick quantitative pilot on the first 100 dev samples:

```bash
INDICES=all MAX_SAMPLES=100 NO_BASELINE=1 \
conda run -n DL_Final bash scripts/run_magicbrush_generalization.sh
```

Run only a few MagicBrush examples for visual comparison:

```bash
INDICES="0 1 2 3 4 5 6 7 8 9" \
conda run -n DL_Final bash scripts/run_magicbrush_generalization.sh
```

Recommended usage:

```text
quantitative table: INDICES=all, MAX_SAMPLES=0
quick sanity check: INDICES=all, MAX_SAMPLES=50 or 100
visualization/demo: manually selected INDICES, for example "0 7 15"
```

Run only C/C+:

```bash
STRATEGIES="c cplus" \
conda run -n DL_Final bash scripts/run_magicbrush_generalization.sh
```

Run bbox sweep on MagicBrush:

```bash
RUN_BBOX_SWEEP=1 BBOX_SWEEP_MODE=corners \
conda run -n DL_Final bash scripts/run_magicbrush_generalization.sh
```

If MagicBrush is not cached and Hugging Face network access is unavailable, download it on another machine:

```bash
huggingface-cli download osunlp/MagicBrush \
  --repo-type dataset \
  --local-dir data/magicbrush_snapshot
```

Then either copy it into the Hugging Face cache, or point the runner to the local snapshot:

```bash
MAGICBRUSH_DATASET=data/magicbrush_snapshot \
conda run -n DL_Final bash scripts/run_magicbrush_generalization.sh
```

## 6. Evaluation

The evaluator reads generated `source`, `target`, `info`, and strategy output images and computes:

```text
inside-bbox output-target L1 / MSE
outside-bbox output-source L1 / MSE
changed-region and bbox IoU
changed-region inside-bbox ratio
changed-region center distance to bbox center
optional CLIP text-image similarity
optional LPIPS perceptual distance
```

Fair evaluation protocol:

```text
1. Use the same evaluation split for all methods.
2. Use the same sample indices for all strategies.
3. Use the same inference seed, diffusion steps, guidance scale, image size, and checkpoint selection rule.
4. Report quantitative metrics over the full split when feasible.
5. If a subset is used because of compute limits, choose it before looking at results and report the exact indices or selection rule.
6. Use only a small selected subset for qualitative visualization; do not use it as the main quantitative result.
```

### 6.1 Core Metrics

Main A/B/C/C+ comparison:

```bash
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/all_strategies_matched \
  --output-csv results/evaluation/all_strategies_per_image_metrics.csv \
  --summary-csv results/evaluation/all_strategies_summary_metrics.csv
```

Fixed-location ablation:

```bash
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/fixed_location_ablation \
  --output-csv results/evaluation/fixed_location_per_image_metrics.csv \
  --summary-csv results/evaluation/fixed_location_summary_metrics.csv
```

Final demo bbox variation:

```bash
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/final_demo_bbox_variation \
  --output-csv results/evaluation/final_demo_bbox_variation_per_image_metrics.csv \
  --summary-csv results/evaluation/final_demo_bbox_variation_summary_metrics.csv
```

MagicBrush generalization:

```bash
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/magicbrush_generalization \
  --output-csv results/evaluation/magicbrush_generalization_per_image_metrics.csv \
  --summary-csv results/evaluation/magicbrush_generalization_summary_metrics.csv
```

Note: final demo evaluation requires the individual output files. If only grid images exist, rerun final-demo inference with:

```bash
KEEP_ONLY_GRID=0 \
conda run -n DL_Final bash scripts/run_bbox_variation_grid_infer.sh
```

### 6.2 CLIP Text-Image Similarity

Main comparison:

```bash
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/all_strategies_matched \
  --output-csv results/evaluation/all_strategies_per_image_clip_metrics.csv \
  --summary-csv results/evaluation/all_strategies_clip_summary_metrics.csv \
  --enable-clip
```

Fixed-location ablation:

```bash
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/fixed_location_ablation \
  --output-csv results/evaluation/fixed_location_per_image_clip_metrics.csv \
  --summary-csv results/evaluation/fixed_location_clip_summary_metrics.csv \
  --enable-clip
```

MagicBrush generalization:

```bash
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/magicbrush_generalization \
  --output-csv results/evaluation/magicbrush_generalization_per_image_clip_metrics.csv \
  --summary-csv results/evaluation/magicbrush_generalization_clip_summary_metrics.csv \
  --enable-clip
```

If Hugging Face network access is unavailable, download CLIP on another machine:

```bash
huggingface-cli download openai/clip-vit-base-patch32 \
  --local-dir models/openai_clip-vit-base-patch32
```

Then evaluate with a local path:

```bash
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/all_strategies_matched \
  --output-csv results/evaluation/all_strategies_per_image_clip_metrics.csv \
  --summary-csv results/evaluation/all_strategies_clip_summary_metrics.csv \
  --enable-clip \
  --clip-model models/openai_clip-vit-base-patch32
```

### 6.3 LPIPS Perceptual Distance

Install LPIPS if needed:

```bash
conda run -n DL_Final python -m pip install lpips
```

Standard LPIPS requires torchvision AlexNet weights. Use a writable torch cache:

```bash
mkdir -p data/torch_cache
```

Main comparison:

```bash
TORCH_HOME=$PWD/data/torch_cache \
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/all_strategies_matched \
  --output-csv results/evaluation/all_strategies_per_image_lpips_metrics.csv \
  --summary-csv results/evaluation/all_strategies_lpips_summary_metrics.csv \
  --enable-lpips
```

Fixed-location ablation:

```bash
TORCH_HOME=$PWD/data/torch_cache \
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/fixed_location_ablation \
  --output-csv results/evaluation/fixed_location_per_image_lpips_metrics.csv \
  --summary-csv results/evaluation/fixed_location_lpips_summary_metrics.csv \
  --enable-lpips
```

MagicBrush generalization:

```bash
TORCH_HOME=$PWD/data/torch_cache \
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/magicbrush_generalization \
  --output-csv results/evaluation/magicbrush_generalization_per_image_lpips_metrics.csv \
  --summary-csv results/evaluation/magicbrush_generalization_lpips_summary_metrics.csv \
  --enable-lpips
```

Offline fallback:

```bash
TORCH_HOME=$PWD/data/torch_cache \
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/all_strategies_matched \
  --output-csv results/evaluation/all_strategies_per_image_lpips_random_metrics.csv \
  --summary-csv results/evaluation/all_strategies_lpips_random_summary_metrics.csv \
  --enable-lpips \
  --lpips-random-backbone
```

Important: `--lpips-random-backbone` is an offline fallback. It can show trends, but standard LPIPS with pretrained backbone is preferred for final reporting.

## 7. Training Loss Summary

Training metrics are stored in:

```text
logs/all_strategies_matched/*metrics.csv
logs/fixed_location_ablation/*metrics.csv
```

Generate the training summary table:

```bash
python - <<'PY'
import csv
from pathlib import Path

files = sorted(Path('logs/all_strategies_matched').glob('*metrics.csv'))
files += sorted(Path('logs/fixed_location_ablation').glob('*metrics.csv'))
rows = []

for path in files:
    data = list(csv.DictReader(path.open()))
    if not data:
        continue
    losses = [float(r['loss']) for r in data if r.get('loss')]
    last = data[-1]
    rows.append({
        'run': path.stem.removesuffix('_metrics'),
        'metrics_csv': str(path),
        'steps': int(last['step']),
        'epochs': int(last['epoch']),
        'final_loss': float(last['loss']),
        'mean_loss': sum(losses) / len(losses),
        'min_loss': min(losses),
        'max_loss': max(losses),
        'learning_rate': last.get('learning_rate', ''),
    })

out = Path('results/evaluation/training_summary.csv')
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f'wrote {out}')
PY
```

## 8. Evaluation Summary

The final human-readable summary is:

```text
results/evaluation/evaluation_summary.md
```

It combines:

```text
all_strategies_summary_metrics.csv
all_strategies_clip_summary_metrics.csv
all_strategies_lpips_random_summary_metrics.csv
fixed_location_summary_metrics.csv
fixed_location_clip_summary_metrics.csv
fixed_location_lpips_random_summary_metrics.csv
final_demo_bbox_variation_summary_metrics.csv
training_summary.csv
```

If any of these CSV files is regenerated, rerun the summary-generation helper used in the project logs or update the markdown table manually.

## 9. Recommended Full Run Order

For a clean run from checkpoints/results:

```bash
# 1. Main matched strategies: train + infer
conda run -n DL_Final bash scripts/run_all_strategies_train_infer.sh

# 2. Fixed-location ablation: C and C+
MODE=both conda run -n DL_Final bash scripts/run_fixed_location_ablation.sh

# 3. Final demo bbox variation, keep files for evaluation
KEEP_ONLY_GRID=0 conda run -n DL_Final bash scripts/run_bbox_variation_grid_infer.sh

# 4. MagicBrush generalization
conda run -n DL_Final bash scripts/run_magicbrush_generalization.sh

# 5. Core evaluation
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/all_strategies_matched \
  --output-csv results/evaluation/all_strategies_per_image_metrics.csv \
  --summary-csv results/evaluation/all_strategies_summary_metrics.csv

conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/fixed_location_ablation \
  --output-csv results/evaluation/fixed_location_per_image_metrics.csv \
  --summary-csv results/evaluation/fixed_location_summary_metrics.csv

conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/final_demo_bbox_variation \
  --output-csv results/evaluation/final_demo_bbox_variation_per_image_metrics.csv \
  --summary-csv results/evaluation/final_demo_bbox_variation_summary_metrics.csv

conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/magicbrush_generalization \
  --output-csv results/evaluation/magicbrush_generalization_per_image_metrics.csv \
  --summary-csv results/evaluation/magicbrush_generalization_summary_metrics.csv

# 6. CLIP evaluation
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/all_strategies_matched \
  --output-csv results/evaluation/all_strategies_per_image_clip_metrics.csv \
  --summary-csv results/evaluation/all_strategies_clip_summary_metrics.csv \
  --enable-clip

conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/fixed_location_ablation \
  --output-csv results/evaluation/fixed_location_per_image_clip_metrics.csv \
  --summary-csv results/evaluation/fixed_location_clip_summary_metrics.csv \
  --enable-clip

# 7. LPIPS fallback if standard LPIPS cannot download backbone weights
TORCH_HOME=$PWD/data/torch_cache \
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/all_strategies_matched \
  --output-csv results/evaluation/all_strategies_per_image_lpips_random_metrics.csv \
  --summary-csv results/evaluation/all_strategies_lpips_random_summary_metrics.csv \
  --enable-lpips \
  --lpips-random-backbone

TORCH_HOME=$PWD/data/torch_cache \
conda run -n DL_Final python framework/sourcecode/evaluate.py \
  --results-root results/fixed_location_ablation \
  --output-csv results/evaluation/fixed_location_per_image_lpips_random_metrics.csv \
  --summary-csv results/evaluation/fixed_location_lpips_random_summary_metrics.csv \
  --enable-lpips \
  --lpips-random-backbone
```

## 10. Common Issues

### CLIP Processor / Tokenizer Missing

Symptom:

```text
Can't load processor for openai/clip-vit-base-patch32
Can't load image processor
Temporary failure in name resolution
```

Fix:

```bash
huggingface-cli download openai/clip-vit-base-patch32 \
  --local-dir models/openai_clip-vit-base-patch32
```

Then pass:

```bash
--clip-model models/openai_clip-vit-base-patch32
```

### LPIPS Tries to Write to Read-Only Torch Cache

Symptom:

```text
Read-only file system: /home/zia/.cache/torch/hub/checkpoints/...
```

Fix:

```bash
mkdir -p data/torch_cache
TORCH_HOME=$PWD/data/torch_cache ...
```

### LPIPS Backbone Download Fails

Symptom:

```text
Downloading alexnet-owt-7be5be79.pth
Temporary failure in name resolution
```

Preferred fix: run again when network works, or manually place the AlexNet weight in:

```text
data/torch_cache/hub/checkpoints/alexnet-owt-7be5be79.pth
```

Offline fallback:

```bash
--lpips-random-backbone
```

### Final Demo Has Only Grid Images

Symptom:

```text
results/final_demo_bbox_variation/... only contains *_grid.png
```

Fix:

```bash
KEEP_ONLY_GRID=0 \
conda run -n DL_Final bash scripts/run_bbox_variation_grid_infer.sh
```

### Strategy A Has No Metrics CSV

The main script logs Strategy A training to:

```text
logs/all_strategies_matched/strategy_a_train.log
```

Strategy B/C/C+ have metrics CSV files. Strategy A can still be evaluated through generated image metrics.
