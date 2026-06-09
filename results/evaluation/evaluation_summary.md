# Evaluation Summary

This file summarizes the latest automatic evaluation results.

Notes:
- The current automatic tables are pilot/demo evaluations, not full test-set results. Main comparison uses 3 PIPE test indices (`0, 1, 2`), fixed-location ablation uses the same 3 indices across 5 bbox settings, and final-demo bbox variation uses the same 3 indices across 6 bbox positions.
- For fair final reporting, rerun inference on the full evaluation split with the same indices, seed, inference steps, and checkpoint policy for every strategy; use a small manually selected subset only for qualitative visualization.
- CLIP is available for both `all_strategies_matched` and `fixed_location_ablation`.
- Main and fixed-location LPIPS values use `--lpips-random-backbone` because the standard AlexNet backbone weight download was unavailable during those runs. MagicBrush LPIPS values are from `magicbrush_generalization_lpips_summary_metrics.csv`.
- `results/final_demo_bbox_variation` is included through the available `final_demo_bbox_variation_summary_metrics.csv`; the full per-bbox values are in the CSV.

## Metric Meaning

| Metric | Meaning | Better |
| --- | --- | --- |
| Inside L1 | Output-target difference inside requested bbox | Lower |
| Outside L1 | Output-source difference outside requested bbox | Lower |
| CLIP | CLIP similarity between output and prompt | Higher |
| LPIPS fallback | Perceptual output-target distance using random-backbone LPIPS fallback | Lower |
| Changed IoU | IoU between changed region and requested bbox | Higher |
| Changed Inside Ratio | Fraction of changed pixels inside bbox | Higher |
| Center Dist Norm | Normalized distance from changed-region center to bbox center | Lower |

## Main Strategy Comparison

| Strategy | Inside L1 ↓ | Outside L1 ↓ | CLIP ↑ | LPIPS Fallback ↓ | Changed IoU ↑ | Changed Inside Ratio ↑ | Center Dist Norm ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strategy_a | 0.1819 | 0.0363 | 0.2489 | 0.0480 | 0.3582 | 0.6825 | 0.0640 |
| strategy_b | 0.1871 | 0.0071 | 0.2566 | 0.0461 | 0.3393 | 0.8983 | 0.0426 |
| strategy_c | 0.1857 | 0.0326 | 0.2552 | 0.0457 | 0.3796 | 0.7145 | 0.0506 |
| strategy_cplus | 0.1761 | 0.0327 | 0.2608 | 0.0429 | 0.3913 | 0.7228 | 0.0523 |

## Fixed-Location Ablation

| Run | BBox | Inside L1 ↓ | Outside L1 ↓ | CLIP ↑ | LPIPS Fallback ↓ | Changed IoU ↑ | Changed Inside Ratio ↑ | Center Dist Norm ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strategy_c_top-left_ablation | bottom_right | 0.1330 | 0.0356 | 0.2493 | 0.0807 | 0.2585 | 0.5815 | 0.1788 |
| strategy_c_top-left_ablation | center | 0.2058 | 0.0391 | 0.2534 | 0.0642 | 0.4391 | 0.7082 | 0.0324 |
| strategy_c_top-left_ablation | original | 0.2120 | 0.0350 | 0.2467 | 0.0594 | 0.4798 | 0.7464 | 0.0355 |
| strategy_c_top-left_ablation | top_left | 0.1165 | 0.0295 | 0.2429 | 0.0578 | 0.2844 | 0.6970 | 0.1350 |
| strategy_c_top-left_ablation | zero | N/A | 0.0569 | 0.2356 | 0.0620 | 0.0000 | 0.0000 | 0.4169 |
| strategy_cplus_top-left_ablation | bottom_right | 0.1948 | 0.0307 | 0.2417 | 0.0756 | 0.2817 | 0.6416 | 0.1512 |
| strategy_cplus_top-left_ablation | center | 0.1831 | 0.0355 | 0.2566 | 0.0572 | 0.3529 | 0.6690 | 0.0419 |
| strategy_cplus_top-left_ablation | original | 0.2104 | 0.0350 | 0.2556 | 0.0573 | 0.3762 | 0.6949 | 0.0444 |
| strategy_cplus_top-left_ablation | top_left | 0.1427 | 0.0290 | 0.2448 | 0.0715 | 0.2746 | 0.6925 | 0.1152 |
| strategy_cplus_top-left_ablation | zero | N/A | 0.0376 | 0.2408 | 0.0570 | 0.0000 | 0.0000 | 0.4550 |

## Final Demo BBox Variation

This table averages all bbox positions within each final-demo sweep directory. Full per-bbox values are in `final_demo_bbox_variation_summary_metrics.csv`.

| Run | Num BBoxes | Inside L1 ↓ | Outside L1 ↓ | Changed IoU ↑ | Changed Inside Ratio ↑ | Center Dist Norm ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strategy_b_corners | 6 | 0.0896 | 0.0039 | 0.1690 | 0.6989 | 0.1261 |
| strategy_b_small-corners | 6 | 0.0654 | 0.0000 | 0.0591 | 1.0000 | 0.0844 |
| strategy_c_corners | 6 | 0.1266 | 0.0364 | 0.2057 | 0.4628 | 0.1830 |
| strategy_c_small-corners | 6 | 0.1279 | 0.0296 | 0.1305 | 0.3797 | 0.1951 |
| strategy_cplus_corners | 6 | 0.1098 | 0.0346 | 0.2118 | 0.4735 | 0.1731 |
| strategy_cplus_small-corners | 6 | 0.1264 | 0.0297 | 0.1412 | 0.4059 | 0.1883 |

## MagicBrush Generalization

This evaluates cross-dataset generalization on the full MagicBrush dev split. All A/B/C/C+ strategies have 528 evaluated outputs, covering indices 0-527.

| Strategy | Num Outputs | Inside L1 ↓ | Outside L1 ↓ | CLIP ↑ | LPIPS ↓ | Changed IoU ↑ | Changed Inside Ratio ↑ | Center Dist Norm ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strategy_a | 528 | 0.3539 | 0.4038 | 0.2471 | 0.7342 | 0.8637 | 0.9980 | 0.0171 |
| strategy_b | 528 | 0.3641 | 0.0000 | 0.2467 | 0.7345 | 0.8602 | 1.0000 | 0.0175 |
| strategy_c | 528 | 0.0970 | 0.1081 | 0.2429 | 0.2454 | 0.1476 | 0.9989 | 0.1178 |
| strategy_cplus | 528 | 0.0919 | 0.1858 | 0.2384 | 0.2366 | 0.1532 | 0.9986 | 0.1032 |

## Training Summary

| Run | Steps | Epochs | Final Loss ↓ | Mean Loss ↓ |
| --- | ---: | ---: | ---: | ---: |
| strategy_b | 2560 | 10 | 0.0847 | 0.3221 |
| strategy_c | 2560 | 10 | 0.0252 | 0.2697 |
| strategy_cplus | 2560 | 10 | 0.0175 | 0.2723 |
| strategy_c_top-left_ablation | 2560 | 10 | 0.0304 | 0.2725 |
| strategy_cplus_top-left_ablation | 2560 | 10 | 0.0515 | 0.2722 |

## Current Takeaways

- Strategy C+ has the best main-comparison inside-bbox target similarity, CLIP score, LPIPS fallback, and changed-region IoU among A/B/C/C+.
- Strategy B has the lowest outside-bbox error, which is expected because its inference constraint strongly preserves regions outside the bbox.
- Strategy C/C+ provide stronger feature-level spatial control than A/B, while C+ better separates object extent from edit region.
- Fixed-location ablation shows that bbox-conditioned control can generalize to unseen bbox positions; bottom-right placement remains harder than center/original placements, but C+ improves bottom-right center distance over C.
- MagicBrush full-dev generalization shows that C/C+ transfer much better than A/B on target reconstruction and LPIPS. C+ is best on inside-bbox L1 and LPIPS, while B trivially preserves outside-bbox pixels because of its hard outside-bbox copy constraint.
- MagicBrush changed-region IoU should be interpreted carefully: A/B have very high IoU because their changed region covers almost the whole large MagicBrush mask, whereas C/C+ make more localized changes and therefore score lower on bbox coverage.
- LPIPS fallback should be treated as an auxiliary trend, not as final standard LPIPS, until pretrained AlexNet backbone weights are available.
