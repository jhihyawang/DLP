#!/usr/bin/env bash
set -euo pipefail

# Fixed-location bbox ablation:
# train with all pseudo-target objects relocated to one fixed placement
# and infer with zero/top-left/center/bottom-right/original bboxes.

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

# MODE:
#   c      = Strategy C, single-channel bbox ControlNet
#   cplus  = Strategy C+, inner/outer two-channel ControlNet
#   both   = run c and cplus sequentially
MODE="${MODE:-cplus}"

MODEL_NAME="${MODEL_NAME:-paint-by-inpaint/add-base}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-256}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
LR="${LR:-1e-5}"
BBOX_LOSS_WEIGHT="${BBOX_LOSS_WEIGHT:-8.0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
CONTROLNET_CONDITIONING_SCALE="${CONTROLNET_CONDITIONING_SCALE:-1.0}"
OUTER_BBOX_PADDING="${OUTER_BBOX_PADDING:-24}"

# The actual ablation setting.
BBOX_SHIFT_PROB="${BBOX_SHIFT_PROB:-1.0}"
BBOX_PLACEMENT="${BBOX_PLACEMENT:-top-left}"
BBOX_PLACEMENT_MARGIN="${BBOX_PLACEMENT_MARGIN:-16}"

SAVE_EVERY="${SAVE_EVERY:-0}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
USE_FP16="${USE_FP16:-0}"

SPLIT="${SPLIT:-test}"
INDICES="${INDICES:-0 1 2}"
INFER_STEPS="${INFER_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.5}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
NO_BASELINE="${NO_BASELINE:-0}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-fixed_location_ablation}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$DEFAULT_CHECKPOINT_ROOT/$EXPERIMENT_NAME}"
RESULT_ROOT="${RESULT_ROOT:-results/$EXPERIMENT_NAME}"
LOG_ROOT="${LOG_ROOT:-logs/$EXPERIMENT_NAME}"

mkdir -p "$PIPE_HF_CACHE_DIR" "$PIPE_MODEL_CACHE_DIR" "$TMPDIR"
mkdir -p "$CHECKPOINT_ROOT" "$RESULT_ROOT" "$LOG_ROOT"

echo "MODE=$MODE"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "PIPE_HF_CACHE_DIR=$PIPE_HF_CACHE_DIR"
echo "PIPE_MODEL_CACHE_DIR=$PIPE_MODEL_CACHE_DIR"
echo "TMPDIR=$TMPDIR"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "ablation: placement=$BBOX_PLACEMENT shift_prob=$BBOX_SHIFT_PROB indices=$INDICES"
echo "training: samples=$MAX_TRAIN_SAMPLES epochs=$EPOCHS image=$IMAGE_SIZE lr=$LR seed=$SEED"

if [[ "$DEVICE" == "cuda" ]]; then
  if ! "$PYTHON_BIN" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA is not available. Fix the NVIDIA driver or run with DEVICE=cpu for a very slow smoke test." >&2
    exit 1
  fi
fi

FP16_FLAG=()
FP32_INFER_FLAG=()
if [[ "$USE_FP16" == "1" ]]; then
  FP16_FLAG=(--fp16)
else
  FP32_INFER_FLAG=(--fp32)
fi

BASELINE_FLAG=()
if [[ "$NO_BASELINE" == "1" ]]; then
  BASELINE_FLAG=(--no-baseline)
fi

common_train_args=(
  --model-name "$MODEL_NAME"
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --image-size "$IMAGE_SIZE"
  --lr "$LR"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --bbox-loss-weight "$BBOX_LOSS_WEIGHT"
  --bbox-shift-prob "$BBOX_SHIFT_PROB"
  --bbox-placement "$BBOX_PLACEMENT"
  --bbox-placement-margin "$BBOX_PLACEMENT_MARGIN"
  --controlnet-conditioning-scale "$CONTROLNET_CONDITIONING_SCALE"
  --save-every "$SAVE_EVERY"
  --seed "$SEED"
  --device "$DEVICE"
  --max-grad-norm "$MAX_GRAD_NORM"
  "${FP16_FLAG[@]}"
)

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

strategies_to_run() {
  if [[ "$MODE" == "both" ]]; then
    echo "c cplus"
  elif [[ "$MODE" == "c" || "$MODE" == "cplus" ]]; then
    echo "$MODE"
  else
    echo "MODE must be c, cplus, or both. Got: $MODE" >&2
    exit 1
  fi
}

control_mode_for_strategy() {
  local strategy="$1"
  if [[ "$strategy" == "cplus" ]]; then
    echo "inner-outer"
  else
    echo "bbox"
  fi
}

run_name_for_strategy() {
  local strategy="$1"
  if [[ "$strategy" == "cplus" ]]; then
    echo "strategy_cplus_${BBOX_PLACEMENT}_ablation"
  else
    echo "strategy_c_${BBOX_PLACEMENT}_ablation"
  fi
}

find_final_tag() {
  local checkpoint_dir="$1"
  local final_path
  final_path="$(find "$checkpoint_dir" -maxdepth 1 -type d -name 'strategy_c_final_epoch_*_controlnet' | sort | tail -n 1)"
  if [[ -z "$final_path" ]]; then
    echo "No final Strategy C checkpoint was found under $checkpoint_dir" >&2
    exit 1
  fi

  local base
  base="$(basename "$final_path")"
  base="${base#strategy_c_}"
  base="${base%_controlnet}"
  echo "$base"
}

run_train() {
  local strategy="$1"
  local run_name control_mode checkpoint_dir
  run_name="$(run_name_for_strategy "$strategy")"
  control_mode="$(control_mode_for_strategy "$strategy")"
  checkpoint_dir="$CHECKPOINT_ROOT/$run_name"

  echo "== Train $run_name =="
  "$PYTHON_BIN" framework/sourcecode/train_strategy_c.py \
    "${common_train_args[@]}" \
    --output-dir "$checkpoint_dir" \
    --metrics-csv "$LOG_ROOT/${run_name}_metrics.csv" \
    --control-conditioning-mode "$control_mode" \
    --outer-bbox-padding "$OUTER_BBOX_PADDING" \
    2>&1 | tee "$LOG_ROOT/${run_name}_train.log"
}

run_infer() {
  local strategy="$1"
  local run_name control_mode checkpoint_dir checkpoint_tag result_dir
  run_name="$(run_name_for_strategy "$strategy")"
  control_mode="$(control_mode_for_strategy "$strategy")"
  checkpoint_dir="$CHECKPOINT_ROOT/$run_name"
  checkpoint_tag="$(find_final_tag "$checkpoint_dir")"
  result_dir="$RESULT_ROOT/$run_name"

  mkdir -p "$result_dir"
  echo "== Infer $run_name tag=$checkpoint_tag =="

  for index in $INDICES; do
    "$PYTHON_BIN" framework/sourcecode/infer_strategy_c.py \
      "${common_infer_args[@]}" \
      --checkpoint-dir "$checkpoint_dir" \
      --checkpoint-tag "$checkpoint_tag" \
      --index "$index" \
      --output-dir "$result_dir" \
      --control-conditioning-mode "$control_mode" \
      --outer-bbox-padding "$OUTER_BBOX_PADDING" \
      --controlnet-conditioning-scale "$CONTROLNET_CONDITIONING_SCALE" \
      --bbox-sweep \
      --bbox-sweep-mode ablation \
      2>&1 | tee -a "$LOG_ROOT/${run_name}_infer.log"
  done
}

for strategy in $(strategies_to_run); do
  if [[ "$RUN_TRAIN" == "1" ]]; then
    run_train "$strategy"
  fi
  if [[ "$RUN_INFER" == "1" ]]; then
    run_infer "$strategy"
  fi
done

echo "Fixed-location ablation completed."
echo "Checkpoints: $CHECKPOINT_ROOT"
echo "Results: $RESULT_ROOT"
echo "Logs: $LOG_ROOT"
