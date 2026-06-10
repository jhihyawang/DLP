#!/usr/bin/env bash
set -euo pipefail

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

EXPERIMENT_NAME="${EXPERIMENT_NAME:-all_strategies_matched}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$DEFAULT_CHECKPOINT_ROOT/$EXPERIMENT_NAME}"
RESULT_ROOT="${RESULT_ROOT:-results/$EXPERIMENT_NAME}"
LOG_ROOT="${LOG_ROOT:-logs/$EXPERIMENT_NAME}"

MODEL_NAME="${MODEL_NAME:-paint-by-inpaint/add-base}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-256}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
LR="${LR:-1e-5}"
BBOX_LOSS_WEIGHT="${BBOX_LOSS_WEIGHT:-8.0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
CONDITIONING_DROPOUT_PROB="${CONDITIONING_DROPOUT_PROB:-0.05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-4}"
CONTROLNET_CONDITIONING_SCALE="${CONTROLNET_CONDITIONING_SCALE:-1.0}"
OUTER_BBOX_PADDING="${OUTER_BBOX_PADDING:-24}"
SAVE_EVERY="${SAVE_EVERY:-0}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"
USE_FP16="${USE_FP16:-0}"

SPLIT="${SPLIT:-test}"
INDICES="${INDICES:-0 1 2}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
INFER_STEPS="${INFER_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
IMAGE_GUIDANCE_SCALE="${IMAGE_GUIDANCE_SCALE:-1.5}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_BBOX_SWEEP="${RUN_BBOX_SWEEP:-0}"
BBOX_SWEEP_MODE="${BBOX_SWEEP_MODE:-corners}"
STRATEGIES="${STRATEGIES:-a b c cplus}"
NO_BASELINE="${NO_BASELINE:-0}"

mkdir -p "$CHECKPOINT_ROOT" "$RESULT_ROOT" "$LOG_ROOT"
mkdir -p "$PIPE_HF_CACHE_DIR" "$PIPE_MODEL_CACHE_DIR" "$TMPDIR"

export SPLIT IMAGE_SIZE MAX_EVAL_SAMPLES

resolve_indices() {
  if [[ "$INDICES" != "all" ]]; then
    echo "$INDICES"
    return 0
  fi

  "$PYTHON_BIN" - <<'PY'
import os
from sourcecode.pipe_bbox_dataset import create_bbox_dataset

dataset = create_bbox_dataset(
    dataset_name="pipe",
    split=os.environ["SPLIT"],
    image_size=int(os.environ["IMAGE_SIZE"]),
)
max_samples = int(os.environ.get("MAX_EVAL_SAMPLES", "0"))
n = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
print(" ".join(str(i) for i in range(n)))
PY
}

RESOLVED_INDICES="$(resolve_indices)"
INDEX_COUNT="$(wc -w <<< "$RESOLVED_INDICES")"

echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "PIPE_HF_CACHE_DIR=$PIPE_HF_CACHE_DIR"
echo "PIPE_MODEL_CACHE_DIR=$PIPE_MODEL_CACHE_DIR"
echo "TMPDIR=$TMPDIR"
echo "STRATEGIES=$STRATEGIES"
echo "shared: samples=$MAX_TRAIN_SAMPLES epochs=$EPOCHS batch=$BATCH_SIZE image=$IMAGE_SIZE lr=$LR bbox_loss=$BBOX_LOSS_WEIGHT seed=$SEED"
echo "infer: split=$SPLIT indices=$INDICES resolved_count=$INDEX_COUNT max_eval_samples=$MAX_EVAL_SAMPLES"

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
  --save-every "$SAVE_EVERY"
  --seed "$SEED"
  --device "$DEVICE"
  --max-grad-norm "$MAX_GRAD_NORM"
  "${FP16_FLAG[@]}"
)

common_lora_args=(
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --conditioning-dropout-prob "$CONDITIONING_DROPOUT_PROB"
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

run_strategy_a_train() {
  local ckpt="$CHECKPOINT_ROOT/strategy_a"
  echo "== Train Strategy A =="
  "$PYTHON_BIN" framework/sourcecode/train.py \
    "${common_train_args[@]}" \
    "${common_lora_args[@]}" \
    --output-dir "$ckpt" \
    2>&1 | tee "$LOG_ROOT/strategy_a_train.log"
}

run_strategy_b_train() {
  local ckpt="$CHECKPOINT_ROOT/strategy_b"
  echo "== Train Strategy B =="
  "$PYTHON_BIN" framework/sourcecode/train_strategy_b.py \
    "${common_train_args[@]}" \
    "${common_lora_args[@]}" \
    --output-dir "$ckpt" \
    --metrics-csv "$LOG_ROOT/strategy_b_metrics.csv" \
    2>&1 | tee "$LOG_ROOT/strategy_b_train.log"
}

run_strategy_c_train() {
  local ckpt="$CHECKPOINT_ROOT/strategy_c"
  echo "== Train Strategy C =="
  "$PYTHON_BIN" framework/sourcecode/train_strategy_c.py \
    "${common_train_args[@]}" \
    --output-dir "$ckpt" \
    --metrics-csv "$LOG_ROOT/strategy_c_metrics.csv" \
    --bbox-shift-prob 0.5 \
    --bbox-placement random \
    --control-conditioning-mode bbox \
    --controlnet-conditioning-scale "$CONTROLNET_CONDITIONING_SCALE" \
    2>&1 | tee "$LOG_ROOT/strategy_c_train.log"
}

run_strategy_cplus_train() {
  local ckpt="$CHECKPOINT_ROOT/strategy_cplus"
  echo "== Train Strategy C+ =="
  "$PYTHON_BIN" framework/sourcecode/train_strategy_c.py \
    "${common_train_args[@]}" \
    --output-dir "$ckpt" \
    --metrics-csv "$LOG_ROOT/strategy_cplus_metrics.csv" \
    --bbox-shift-prob 0.5 \
    --bbox-placement random \
    --control-conditioning-mode inner-outer \
    --outer-bbox-padding "$OUTER_BBOX_PADDING" \
    --controlnet-conditioning-scale "$CONTROLNET_CONDITIONING_SCALE" \
    2>&1 | tee "$LOG_ROOT/strategy_cplus_train.log"
}

run_strategy_a_infer() {
  local ckpt="$CHECKPOINT_ROOT/strategy_a"
  local tag
  tag="$(find_final_tag_strategy_a "$ckpt")"
  echo "== Infer Strategy A tag=$tag =="
  "$PYTHON_BIN" framework/sourcecode/infer_strategy_a.py \
    "${common_infer_args[@]}" \
    --checkpoint-dir "$ckpt" \
    --checkpoint-tag "$tag" \
    --indices "$RESOLVED_INDICES" \
    --output-dir "$RESULT_ROOT/strategy_a" \
    2>&1 | tee -a "$LOG_ROOT/strategy_a_infer.log"
  if [[ "$RUN_BBOX_SWEEP" == "1" ]]; then
    "$PYTHON_BIN" framework/sourcecode/infer_strategy_a.py \
      "${common_infer_args[@]}" \
      --checkpoint-dir "$ckpt" \
      --checkpoint-tag "$tag" \
      --indices "$RESOLVED_INDICES" \
      --output-dir "$RESULT_ROOT/strategy_a_bbox_sweep" \
      --bbox-sweep \
      --bbox-sweep-mode "$BBOX_SWEEP_MODE" \
      2>&1 | tee -a "$LOG_ROOT/strategy_a_infer.log"
  fi
}

run_strategy_b_infer() {
  local ckpt="$CHECKPOINT_ROOT/strategy_b"
  local tag
  tag="$(find_final_tag_from_dir "$ckpt" 'strategy_b_final_epoch_*_lora' 'strategy_b_' '_lora')"
  echo "== Infer Strategy B tag=$tag =="
  "$PYTHON_BIN" framework/sourcecode/infer_strategy_b.py \
    "${common_infer_args[@]}" \
    --checkpoint-dir "$ckpt" \
    --checkpoint-tag "$tag" \
    --indices "$RESOLVED_INDICES" \
    --output-dir "$RESULT_ROOT/strategy_b" \
    2>&1 | tee -a "$LOG_ROOT/strategy_b_infer.log"
  if [[ "$RUN_BBOX_SWEEP" == "1" ]]; then
    "$PYTHON_BIN" framework/sourcecode/infer_strategy_b.py \
      "${common_infer_args[@]}" \
      --checkpoint-dir "$ckpt" \
      --checkpoint-tag "$tag" \
      --indices "$RESOLVED_INDICES" \
      --output-dir "$RESULT_ROOT/strategy_b_bbox_sweep" \
      --bbox-sweep \
      --bbox-sweep-mode "$BBOX_SWEEP_MODE" \
      2>&1 | tee -a "$LOG_ROOT/strategy_b_infer.log"
  fi
}

run_strategy_c_infer() {
  local ckpt="$CHECKPOINT_ROOT/strategy_c"
  local tag
  tag="$(find_final_tag_from_dir "$ckpt" 'strategy_c_final_epoch_*_controlnet' 'strategy_c_' '_controlnet')"
  echo "== Infer Strategy C tag=$tag =="
  "$PYTHON_BIN" framework/sourcecode/infer_strategy_c.py \
    "${common_infer_args[@]}" \
    --checkpoint-dir "$ckpt" \
    --checkpoint-tag "$tag" \
    --indices "$RESOLVED_INDICES" \
    --output-dir "$RESULT_ROOT/strategy_c" \
    --control-conditioning-mode bbox \
    --controlnet-conditioning-scale "$CONTROLNET_CONDITIONING_SCALE" \
    2>&1 | tee -a "$LOG_ROOT/strategy_c_infer.log"
  if [[ "$RUN_BBOX_SWEEP" == "1" ]]; then
    "$PYTHON_BIN" framework/sourcecode/infer_strategy_c.py \
      "${common_infer_args[@]}" \
      --checkpoint-dir "$ckpt" \
      --checkpoint-tag "$tag" \
      --indices "$RESOLVED_INDICES" \
      --output-dir "$RESULT_ROOT/strategy_c_bbox_sweep" \
      --control-conditioning-mode bbox \
      --controlnet-conditioning-scale "$CONTROLNET_CONDITIONING_SCALE" \
      --bbox-sweep \
      --bbox-sweep-mode "$BBOX_SWEEP_MODE" \
      2>&1 | tee -a "$LOG_ROOT/strategy_c_infer.log"
  fi
}

run_strategy_cplus_infer() {
  local ckpt="$CHECKPOINT_ROOT/strategy_cplus"
  local tag
  tag="$(find_final_tag_from_dir "$ckpt" 'strategy_c_final_epoch_*_controlnet' 'strategy_c_' '_controlnet')"
  echo "== Infer Strategy C+ tag=$tag =="
  "$PYTHON_BIN" framework/sourcecode/infer_strategy_c.py \
    "${common_infer_args[@]}" \
    --checkpoint-dir "$ckpt" \
    --checkpoint-tag "$tag" \
    --indices "$RESOLVED_INDICES" \
    --output-dir "$RESULT_ROOT/strategy_cplus" \
    --control-conditioning-mode inner-outer \
    --outer-bbox-padding "$OUTER_BBOX_PADDING" \
    --controlnet-conditioning-scale "$CONTROLNET_CONDITIONING_SCALE" \
    2>&1 | tee -a "$LOG_ROOT/strategy_cplus_infer.log"
  if [[ "$RUN_BBOX_SWEEP" == "1" ]]; then
    "$PYTHON_BIN" framework/sourcecode/infer_strategy_c.py \
      "${common_infer_args[@]}" \
      --checkpoint-dir "$ckpt" \
      --checkpoint-tag "$tag" \
      --indices "$RESOLVED_INDICES" \
      --output-dir "$RESULT_ROOT/strategy_cplus_bbox_sweep" \
      --control-conditioning-mode inner-outer \
      --outer-bbox-padding "$OUTER_BBOX_PADDING" \
      --controlnet-conditioning-scale "$CONTROLNET_CONDITIONING_SCALE" \
      --bbox-sweep \
      --bbox-sweep-mode "$BBOX_SWEEP_MODE" \
      2>&1 | tee -a "$LOG_ROOT/strategy_cplus_infer.log"
  fi
}

if [[ "$RUN_TRAIN" == "1" ]]; then
  has_strategy a && run_strategy_a_train
  has_strategy b && run_strategy_b_train
  has_strategy c && run_strategy_c_train
  has_strategy cplus && run_strategy_cplus_train
fi

if [[ "$RUN_INFER" == "1" ]]; then
  has_strategy a && run_strategy_a_infer
  has_strategy b && run_strategy_b_infer
  has_strategy c && run_strategy_c_infer
  has_strategy cplus && run_strategy_cplus_infer
fi

echo "All requested strategies completed."
echo "Checkpoints: $CHECKPOINT_ROOT"
echo "Results: $RESULT_ROOT"
echo "Logs: $LOG_ROOT"
