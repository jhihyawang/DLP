# Evaluation Summary

This file summarizes the latest automatic evaluation results.

Notes:
- Main Strategy Comparison is now based on the full PIPE test inference outputs `pipe_test_0` to `pipe_test_751`, for 752 samples per strategy.
- The result folder also contains old pilot outputs `test_0`, `test_1`, and `test_2`; those are excluded from the clean main comparison below.
- MagicBrush Generalization uses the corrected mask-difference bbox extraction, with 528 dev samples per strategy.
- LPIPS values below use the currently generated evaluation CSVs. The MagicBrush LPIPS file used here is the random-backbone fallback file, so LPIPS should be treated as an auxiliary trend unless standard pretrained LPIPS weights are available.
- Fixed-location and final-demo bbox variation tables are still diagnostic/demo experiments and should not be interpreted as full benchmark comparisons.

## Metric Meaning

| Metric | Meaning | Better |
| --- | --- | --- |
| Inside L1 | Output-target difference inside requested bbox | Lower |
| Outside L1 | Output-source difference outside requested bbox | Lower |
| CLIP | CLIP similarity between output and prompt | Higher |
| LPIPS Whole | Perceptual output-target distance on the whole image | Lower |
| Changed IoU | IoU between changed region and requested bbox | Higher |
| Changed Inside Ratio | Fraction of changed pixels inside bbox | Higher |
| Center Dist Norm | Normalized distance from changed-region center to bbox center | Lower |

## Main Strategy Comparison

Clean full PIPE test result. This table uses only `pipe_test_*` outputs, excluding old pilot `test_0/1/2` files.

| Strategy | Num Outputs | Inside L1 ↓ | Outside L1 ↓ | CLIP ↑ | LPIPS Whole ↓ | Changed IoU ↑ | Changed Inside Ratio ↑ | Center Dist Norm ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strategy_a | 752 | 0.1638 | 0.0407 | 0.2488 | 0.1530 | 0.2001 | 0.3309 | 0.1540 |
| strategy_b | 752 | 0.1669 | 0.0040 | 0.2478 | 0.0823 | 0.3621 | 0.7566 | 0.0344 |
| strategy_c | 752 | 0.1834 | 0.0385 | 0.2520 | 0.1502 | 0.2647 | 0.4097 | 0.1383 |
| strategy_cplus | 752 | 0.1872 | 0.0350 | 0.2503 | 0.1422 | 0.2492 | 0.4136 | 0.1350 |

### Main Comparison Takeaways

- Strategy B has the strongest spatial localization and background preservation: lowest Outside L1, highest Changed IoU, highest Changed Inside Ratio, and lowest Center Dist Norm.
- Strategy C achieves the highest CLIP score, indicating slightly better text-image alignment on the full PIPE test set.
- Strategy C+ achieves the lowest whole-image LPIPS among A/C/C+, but it does not dominate every metric in the full-test setting.
- Strategy A has the lowest Inside L1, but its changed-region localization is weaker than B/C/C+.

## MagicBrush Generalization

Corrected full MagicBrush dev result. This table uses `results/magicbrush_generalization_maskdiff`, where bbox is derived from the source-mask difference, with source-target difference as fallback.

| Strategy | Num Outputs | Inside L1 ↓ | Outside L1 ↓ | CLIP ↑ | LPIPS Whole ↓ | Changed IoU ↑ | Changed Inside Ratio ↑ | Center Dist Norm ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strategy_a | 528 | 0.3595 | 0.3514 | 0.2471 | 0.2111 | 0.5242 | 0.5745 | 0.1203 |
| strategy_b | 528 | 0.2933 | 0.0000 | 0.2422 | 0.1175 | 0.7422 | 1.0000 | 0.0196 |
| strategy_c | 528 | 0.1573 | 0.0500 | 0.2445 | 0.0504 | 0.2151 | 0.7599 | 0.1275 |
| strategy_cplus | 528 | 0.1605 | 0.0445 | 0.2412 | 0.0497 | 0.2200 | 0.7748 | 0.1119 |

### MagicBrush Takeaways

- Strategy B remains the most spatially conservative method: Outside L1 is 0, Changed Inside Ratio is 1.0, and Center Dist Norm is the lowest.
- Strategy C and C+ better match the target edited image on MagicBrush: they have much lower Inside L1 and LPIPS Whole than A/B.
- Strategy C+ slightly improves over C in Outside L1, LPIPS Whole, Changed IoU, Changed Inside Ratio, and Center Dist Norm, while C has a slightly better Inside L1 and CLIP score.
- Strategy A generalizes poorly on MagicBrush, with high inside and outside image differences.

## Fixed-Location Ablation

This is a diagnostic ablation, not a full benchmark table.

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

## Training Summary

| Run | Steps | Epochs | Final Loss ↓ | Mean Loss ↓ |
| --- | ---: | ---: | ---: | ---: |
| strategy_b | 2560 | 10 | 0.0847 | 0.3221 |
| strategy_c | 2560 | 10 | 0.0252 | 0.2697 |
| strategy_cplus | 2560 | 10 | 0.0175 | 0.2723 |
| strategy_c_top-left_ablation | 2560 | 10 | 0.0304 | 0.2725 |
| strategy_cplus_top-left_ablation | 2560 | 10 | 0.0515 | 0.2722 |

## Current Takeaways

- On full PIPE test, Strategy B is the strongest method for spatial localization and background preservation.
- On full PIPE test, Strategy C gives the best CLIP text-image similarity, while C+ gives the best whole-image LPIPS among the learned ControlNet-style variants.
- On MagicBrush, Strategy C/C+ generalize better in target-image similarity, while Strategy B remains the most conservative spatially constrained method.
- The old MagicBrush direct-mask result should no longer be used; use the corrected `magicbrush_generalization_maskdiff` result instead.
- Fixed-location and final-demo bbox variation should be presented as ablation/qualitative evidence rather than full benchmark conclusions.
