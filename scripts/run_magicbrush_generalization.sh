#!/usr/bin/env bash
set -euo pipefail

# MagicBrush generalization test:
# - uses existing A/B/C/C+ checkpoints trained on PIPE
# - runs inference on osunlp/MagicBrush without additional training
# - writes source/target/info/output files so evaluate.py can compute metrics

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DISK="${DATA_DISK:-/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def}"
if [[ -d "$DATA_DISK" && -w "$DATA_DISK" ]]; then
  DEFAULT_HF_CACHE_DIR="$DATA_DISK/hf_cache"
  DEFAULT_MODEL_CACHE_DIR="$DATA_DISK/huggingface"
  DEFAULT_TMPDIR="$DATA_DISK/tmp"
  DEFAULT_CHECKPOINT_ROOT="$DATA_DISK/checkpoints"
else
  DEFAULT_HF_CACHE_DIR="$PROJECT_ROOT/data/hf_cache"
  DEFAULT_MODEL_CACHE_DIR="$PROJECT_ROOT/data/model_cache"
  DEFAULT_TMPDIR="${TMPDIR:-/tmp}"
  DEFAULT_CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"
fi

export PIPE_HF_CACHE_DIR="${PIPE_HF_CACHE_DIR:-$DEFAULT_HF_CACHE_DIR}"
export HF_HOME="${HF_HOME:-$PIPE_HF_CACHE_DIR/home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PIPE_HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PIPE_HF_CACHE_DIR/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$PIPE_HF_CACHE_DIR/xet}"
export PIPE_MODEL_CACHE_DIR="${PIPE_MODEL_CACHE_DIR:-$DEFAULT_MODEL_CACHE_DIR}"
export TMPDIR="${TMPDIR:-$DEFAULT_TMPDIR}"
export PYTHONPATH="${PYTHONPATH:-$PROJECT_ROOT/framework}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_EXPERIMENT_NAME="${SOURCE_EXPERIMENT_NAME:-all_strategies_matched}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$DEFAULT_CHECKPOINT_ROOT/$SOURCE_EXPERIMENT_NAME}"
RESULT_ROOT="${RESULT_ROOT:-results/magicbrush_generalization}"
LOG_ROOT="${LOG_ROOT:-logs/magicbrush_generalization}"

MODEL_NAME="${MODEL_NAME:-paint-by-inpaint/add-base}"
MAGICBRUSH_DATASET="${MAGICBRUSH_DATASET:-osunlp/MagicBrush}"
SPLIT="${SPLIT:-dev}"
INDICES="${INDICES:-0 1 2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
INFER_STEPS="${INFER_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.5}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
USE_FP16="${USE_FP16:-0}"
STRATEGIES="${STRATEGIES:-a b c cplus}"

RUN_BBOX_SWEEP="${RUN_BBOX_SWEEP:-0}"
BBOX_SWEEP_MODE="${BBOX_SWEEP_MODE:-corners}"
CONTROLNET_CONDITIONING_SCALE="${CONTROLNET_CONDITIONING_SCALE:-1.0}"
OUTER_BBOX_PADDING="${OUTER_BBOX_PADDING:-24}"
B_STRATEGY_BBOX_PADDING="${B_STRATEGY_BBOX_PADDING:-0}"
NO_BASELINE="${NO_BASELINE:-0}"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT" "$PIPE_HF_CACHE_DIR" "$PIPE_MODEL_CACHE_DIR" "$TMPDIR"

export MAGICBRUSH_DATASET SPLIT IMAGE_SIZE MAX_SAMPLES

resolve_indices() {
  if [[ "$INDICES" != "all" ]]; then
    echo "$INDICES"
    return 0
  fi

  "$PYTHON_BIN" - <<'PY'
import os
from sourcecode.pipe_bbox_dataset import create_bbox_dataset

dataset = create_bbox_dataset(
    dataset_name="magicbrush",
    split=os.environ["SPLIT"],
    image_size=int(os.environ["IMAGE_SIZE"]),
    magicbrush_dataset_name=os.environ["MAGICBRUSH_DATASET"],
)
max_samples = int(os.environ.get("MAX_SAMPLES", "0"))
n = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
print(" ".join(str(i) for i in range(n)))
PY
}

RESOLVED_INDICES="$(resolve_indices)"
INDEX_COUNT="$(wc -w <<< "$RESOLVED_INDICES")"

echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "MAGICBRUSH_DATASET=$MAGICBRUSH_DATASET split=$SPLIT indices=$INDICES resolved_count=$INDEX_COUNT max_samples=$MAX_SAMPLES"
echo "STRATEGIES=$STRATEGIES"

if [[ "$DEVICE" == "cuda" ]]; then
  if ! "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA is not available. Fix the NVIDIA driver or run with DEVICE=cpu for a very slow CPU run." >&2
    exit 1
  fi
fi

FP32_INFER_FLAG=()
if [[ "$USE_FP16" != "1" ]]; then
  FP32_INFER_FLAG=(--fp32)
fi

BASELINE_FLAG=()
if [[ "$NO_BASELINE" == "1" ]]; then
  BASELINE_FLAG=(--no-baseline)
fi

SWEEP_FLAG=()
if [[ "$RUN_BBOX_SWEEP" == "1" ]]; then
  SWEEP_FLAG=(--bbox-sweep --bbox-sweep-mode "$BBOX_SWEEP_MODE")
fi

common_infer_args=(
  --model-name "$MODEL_NAME"
  --dataset-name magicbrush
  --magicbrush-dataset "$MAGICBRUSH_DATASET"
  --split "$SPLIT"
  --image-size "$IMAGE_SIZE"
  --steps "$INFER_STEPS"
  --guidance-scale "$GUIDANCE_SCALE"
  --image-guidance-scale "$IMAGE_GUIDANCE_SCALE"
  --seed "$SEED"
  --device "$DEVICE"
  "${FP32_INFER_FLAG[@]}"
  "${BASELINE_FLAG[@]}"
  "${SWEEP_FLAG[@]}"
)

has_strategy() {
  local wanted="$1"
  for item in $STRATEGIES; do
    [[ "$item" == "$wanted" ]] && return 0
  done
  return 1
}

find_final_tag_from_dir() {
  local checkpoint_dir="$1"
  local pattern="$2"
  local prefix="$3"
  local suffix="$4"
  local final_path
  final_path="$(find "$checkpoint_dir" -maxdepth 1 -type d -name "$pattern" | sort | tail -n 1)"
  if [[ -z "$final_path" ]]; then
    echo "No final checkpoint matching $pattern under $checkpoint_dir" >&2
    exit 1
  fi
  local base
  base="$(basename "$final_path")"
  base="${base#$prefix}"
  base="${base%$suffix}"
  echo "$base"
}

find_final_tag_strategy_a() {
  local checkpoint_dir="$1"
  local final_path
  final_path="$(find "$checkpoint_dir" -maxdepth 1 -type d -name 'strategy_a_final_epoch_*_lora' | sort | tail -n 1)"
  if [[ -z "$final_path" ]]; then
    echo "No final Strategy A LoRA checkpoint under $checkpoint_dir" >&2
    exit 1
  fi
  local base
  base="$(basename "$final_path")"
  base="${base#strategy_a_}"
  base="${base%_lora}"
  echo "$base"
}

run_strategy_a() {
  local ckpt="$CHECKPOINT_ROOT/strategy_a"
  local tag
  tag="$(find_final_tag_strategy_a "$ckpt")"
  local out="$RESULT_ROOT/strategy_a"
  mkdir -p "$out"
  echo "== MagicBrush Strategy A tag=$tag =="
  for index in $RESOLVED_INDICES; do
    "$PYTHON_BIN" framework/sourcecode/infer_strategy_a.py \
      "${common_infer_args[@]}" \
      --checkpoint-dir "$ckpt" \
      --checkpoint-tag "$tag" \
      --index "$index" \
      --output-dir "$out" \
      2>&1 | tee -a "$LOG_ROOT/strategy_a.log"
  done
}

run_strategy_b() {
  local ckpt="$CHECKPOINT_ROOT/strategy_b"
  local tag
  tag="$(find_final_tag_from_dir "$ckpt" 'strategy_b_final_epoch_*_lora' 'strategy_b_' '_lora')"
  local out="$RESULT_ROOT/strategy_b"
  mkdir -p "$out"
  echo "== MagicBrush Strategy B tag=$tag =="
  for index in $RESOLVED_INDICES; do
    "$PYTHON_BIN" framework/sourcecode/infer_strategy_b.py \
      "${common_infer_args[@]}" \
      --checkpoint-dir "$ckpt" \
      --checkpoint-tag "$tag" \
      --index "$index" \
      --bbox-padding "$B_STRATEGY_BBOX_PADDING" \
      --output-dir "$out" \
      2>&1 | tee -a "$LOG_ROOT/strategy_b.log"
  done
}

run_strategy_c_common() {
  local strategy_name="$1"
  local control_mode="$2"
  local ckpt="$CHECKPOINT_ROOT/$strategy_name"
  local tag
  tag="$(find_final_tag_from_dir "$ckpt" 'strategy_c_final_epoch_*_controlnet' 'strategy_c_' '_controlnet')"
  local out="$RESULT_ROOT/$strategy_name"
  mkdir -p "$out"
  echo "== MagicBrush ${strategy_name} tag=$tag =="
  for index in $RESOLVED_INDICES; do
    "$PYTHON_BIN" framework/sourcecode/infer_strategy_c.py \
      "${common_infer_args[@]}" \
      --checkpoint-dir "$ckpt" \
      --checkpoint-tag "$tag" \
      --index "$index" \
      --output-dir "$out" \
      --control-conditioning-mode "$control_mode" \
      --outer-bbox-padding "$OUTER_BBOX_PADDING" \
      --controlnet-conditioning-scale "$CONTROLNET_CONDITIONING_SCALE" \
      2>&1 | tee -a "$LOG_ROOT/${strategy_name}.log"
  done
}

has_strategy a && run_strategy_a
has_strategy b && run_strategy_b
has_strategy c && run_strategy_c_common strategy_c bbox
has_strategy cplus && run_strategy_c_common strategy_cplus inner-outer

echo "MagicBrush generalization inference completed."
echo "Results: $RESULT_ROOT"
echo "Logs: $LOG_ROOT"
