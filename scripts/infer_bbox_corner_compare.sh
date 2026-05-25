#!/usr/bin/env bash
set -euo pipefail

checkpoint_tag="final_epoch_0010_step_002560"
if [[ "$#" -gt 0 ]]; then
  indices=("$@")
else
  indices=(3 4 5)
fi
shared_args=(
  --split test
  --steps 20
  --seed 1234
  --bbox-sweep
)

for index in "${indices[@]}"; do
  python framework/sourcecode/infer_strategy_a.py \
    --checkpoint-dir checkpoints/strategy_a \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir results/strategy_a_bbox_corner_sweep \
    --index "$index" \
    "${shared_args[@]}"

  python framework/sourcecode/infer_strategy_b.py \
    --checkpoint-dir checkpoints/strategy_b \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir results/strategy_b_bbox_corner_sweep \
    --index "$index" \
    "${shared_args[@]}"

  python framework/sourcecode/infer_strategy_c.py \
    --checkpoint-dir checkpoints/strategy_c \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir results/strategy_c_bbox_corner_sweep \
    --index "$index" \
    "${shared_args[@]}"

  python scripts/stack_bbox_sweep_grids.py \
    --checkpoint-tag "$checkpoint_tag" \
    --index "$index" \
    --seed 1234
done
