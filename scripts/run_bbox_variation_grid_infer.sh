#!/usr/bin/env bash
set -euo pipefail

# Final-demo inference helper:
# - uses existing matched checkpoints for A/B/C/C+
# - generates bbox sweeps at different locations and smaller bbox sizes
# - keeps only *_grid.png files in the output directories

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
RESULT_ROOT="${RESULT_ROOT:-results/final_demo_bbox_variation}"
LOG_ROOT="${LOG_ROOT:-logs/final_demo_bbox_variation}"

MODEL_NAME="${MODEL_NAME:-paint-by-inpaint/add-base}"
SPLIT="${SPLIT:-test}"
INDICES="${INDICES:-0 1 2}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
INFER_STEPS="${INFER_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.5}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
USE_FP16="${USE_FP16:-0}"
STRATEGIES="${STRATEGIES:-a b c cplus}"

# Common sweep modes:
#   corners       = same bbox size, different locations
#   small-corners = smaller bbox, different locations
#   vertical      = same bbox size, top/middle/bottom
SWEEP_MODES="${SWEEP_MODES:-corners small-corners}"

CONTROLNET_CONDITIONING_SCALE="${CONTROLNET_CONDITIONING_SCALE:-1.0}"
OUTER_BBOX_PADDING="${OUTER_BBOX_PADDING:-24}"
B_STRATEGY_BBOX_PADDING="${B_STRATEGY_BBOX_PADDING:-0}"
KEEP_ONLY_GRID="${KEEP_ONLY_GRID:-1}"
NO_BASELINE="${NO_BASELINE:-0}"

mkdir -p "$RESULT_ROOT" "$LOG_ROOT" "$PIPE_HF_CACHE_DIR" "$PIPE_MODEL_CACHE_DIR" "$TMPDIR"

echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "STRATEGIES=$STRATEGIES"
echo "INDICES=$INDICES"
echo "SWEEP_MODES=$SWEEP_MODES"
echo "KEEP_ONLY_GRID=$KEEP_ONLY_GRID"

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

common_infer_args=(
  --model-name "$MODEL_NAME"
  --split "$SPLIT"
  --image-size "$IMAGE_SIZE"
  --steps "$INFER_STEPS"
  --guidance-scale "$GUIDANCE_SCALE"
  --image-guidance-scale "$IMAGE_GUIDANCE_SCALE"
  --seed "$SEED"
  --device "$DEVICE"
  "${FP32_INFER_FLAG[@]}"
  "${BASELINE_FLAG[@]}"
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

cleanup_keep_only_grid() {
  local output_dir="$1"
  if [[ "$KEEP_ONLY_GRID" != "1" ]]; then
    return
  fi
  find "$output_dir" -maxdepth 1 -type f ! -name '*_grid.png' -delete
}

run_strategy_a() {
  local ckpt="$CHECKPOINT_ROOT/strategy_a"
  local tag
  tag="$(find_final_tag_strategy_a "$ckpt")"

  for mode in $SWEEP_MODES; do
    local out="$RESULT_ROOT/strategy_a_${mode}"
    mkdir -p "$out"
    echo "== Strategy A sweep=$mode tag=$tag =="
    for index in $INDICES; do
      "$PYTHON_BIN" framework/sourcecode/infer_strategy_a.py \
        "${common_infer_args[@]}" \
        --checkpoint-dir "$ckpt" \
        --checkpoint-tag "$tag" \
        --index "$index" \
        --output-dir "$out" \
        --bbox-sweep \
        --bbox-sweep-mode "$mode" \
        2>&1 | tee -a "$LOG_ROOT/strategy_a_${mode}.log"
    done
    cleanup_keep_only_grid "$out"
  done
}

run_strategy_b() {
  local ckpt="$CHECKPOINT_ROOT/strategy_b"
  local tag
  tag="$(find_final_tag_from_dir "$ckpt" 'strategy_b_final_epoch_*_lora' 'strategy_b_' '_lora')"

  for mode in $SWEEP_MODES; do
    local out="$RESULT_ROOT/strategy_b_${mode}"
    mkdir -p "$out"
    echo "== Strategy B sweep=$mode tag=$tag =="
    for index in $INDICES; do
      "$PYTHON_BIN" framework/sourcecode/infer_strategy_b.py \
        "${common_infer_args[@]}" \
        --checkpoint-dir "$ckpt" \
        --checkpoint-tag "$tag" \
        --index "$index" \
        --output-dir "$out" \
        --bbox-padding "$B_STRATEGY_BBOX_PADDING" \
        --bbox-sweep \
        --bbox-sweep-mode "$mode" \
        2>&1 | tee -a "$LOG_ROOT/strategy_b_${mode}.log"
    done
    cleanup_keep_only_grid "$out"
  done
}

run_strategy_c_common() {
  local strategy_name="$1"
  local control_mode="$2"
  local ckpt="$CHECKPOINT_ROOT/$strategy_name"
  local tag
  tag="$(find_final_tag_from_dir "$ckpt" 'strategy_c_final_epoch_*_controlnet' 'strategy_c_' '_controlnet')"

  for mode in $SWEEP_MODES; do
    local out="$RESULT_ROOT/${strategy_name}_${mode}"
    mkdir -p "$out"
    echo "== ${strategy_name} sweep=$mode tag=$tag =="
    for index in $INDICES; do
      "$PYTHON_BIN" framework/sourcecode/infer_strategy_c.py \
        "${common_infer_args[@]}" \
        --checkpoint-dir "$ckpt" \
        --checkpoint-tag "$tag" \
        --index "$index" \
        --output-dir "$out" \
        --control-conditioning-mode "$control_mode" \
        --outer-bbox-padding "$OUTER_BBOX_PADDING" \
        --controlnet-conditioning-scale "$CONTROLNET_CONDITIONING_SCALE" \
        --bbox-sweep \
        --bbox-sweep-mode "$mode" \
        2>&1 | tee -a "$LOG_ROOT/${strategy_name}_${mode}.log"
    done
    cleanup_keep_only_grid "$out"
  done
}

has_strategy a && run_strategy_a
has_strategy b && run_strategy_b
has_strategy c && run_strategy_c_common strategy_c bbox
has_strategy cplus && run_strategy_c_common strategy_cplus inner-outer

echo "BBox variation grid inference completed."
echo "Only grid files are kept when KEEP_ONLY_GRID=1."
echo "Results: $RESULT_ROOT"
