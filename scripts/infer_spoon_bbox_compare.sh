#!/usr/bin/env bash
set -euo pipefail

checkpoint_tag="final_epoch_0010_step_002560"
shared_args=(
  --split test
  --index 0
  --steps 20
  --seed 1234
  --bbox-sweep
)

run_sweep() {
  local mode="$1"
  local suffix="$2"
  local output_prefix="$3"

  python framework/sourcecode/infer_strategy_a.py \
    --checkpoint-dir checkpoints/strategy_a \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir "results/strategy_a_${suffix}" \
    --bbox-sweep-mode "$mode" \
    "${shared_args[@]}"

  python framework/sourcecode/infer_strategy_b.py \
    --checkpoint-dir checkpoints/strategy_b \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir "results/strategy_b_${suffix}" \
    --bbox-sweep-mode "$mode" \
    "${shared_args[@]}"

  python framework/sourcecode/infer_strategy_c.py \
    --checkpoint-dir checkpoints/strategy_c \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir "results/strategy_c_${suffix}" \
    --bbox-sweep-mode "$mode" \
    "${shared_args[@]}"

  python scripts/stack_bbox_sweep_grids.py \
    --checkpoint-tag "$checkpoint_tag" \
    --index 0 \
    --seed 1234 \
    --strategy-suffix "$suffix" \
    --output-dir results/bbox_spoon_compare \
    --output-prefix "$output_prefix"
}

run_sweep small-corners spoon_small_bbox_sweep spoon_small_bbox_compare
run_sweep vertical spoon_original_vertical_sweep spoon_original_vertical_compare
