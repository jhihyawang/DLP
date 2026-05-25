#!/usr/bin/env bash
set -euo pipefail

checkpoint_tag="final_epoch_0010_step_002560"
shared_args=(
  --split test
  --steps 20
  --seed 1234
)

for index in 1 2 3 4; do
  python framework/sourcecode/infer_strategy_a.py \
    --checkpoint-dir checkpoints/strategy_a \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir results/strategy_a_multi_test \
    --index "$index" \
    "${shared_args[@]}"

  python framework/sourcecode/infer_strategy_b.py \
    --checkpoint-dir checkpoints/strategy_b \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir results/strategy_b_multi_test \
    --index "$index" \
    "${shared_args[@]}"
done

for index in 0 1 2; do
  python framework/sourcecode/infer_strategy_a.py \
    --checkpoint-dir checkpoints/strategy_a \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir results/strategy_a_bbox_sweep \
    --index "$index" \
    --bbox-sweep \
    "${shared_args[@]}"

  python framework/sourcecode/infer_strategy_b.py \
    --checkpoint-dir checkpoints/strategy_b \
    --checkpoint-tag "$checkpoint_tag" \
    --output-dir results/strategy_b_bbox_sweep \
    --index "$index" \
    --bbox-sweep \
    "${shared_args[@]}"
done
