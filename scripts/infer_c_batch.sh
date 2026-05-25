#!/usr/bin/env bash
set -euo pipefail

checkpoint_tag="final_epoch_0010_step_002560"
shared_args=(
  --split test
  --steps 20
  --seed 1234
)

for index in 1 2 3 4; do
  python framework/sourcecode/infer_strategy_c.py \
    --checkpoint-dir checkpoints/strategy_c \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir results/strategy_c_multi_test \
    --index "$index" \
    "${shared_args[@]}"
done

for index in 0 1 2; do
  python framework/sourcecode/infer_strategy_c.py \
    --checkpoint-dir checkpoints/strategy_c \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir results/strategy_c_bbox_sweep \
    --index "$index" \
    --bbox-sweep \
    "${shared_args[@]}"
done
