# Experiment Summary

## Goal

This project studies bounding-box-guided object addition. Given a source image, text instruction, and a target bounding box, the model should add the requested object near the specified region while preserving the surrounding background.

## Compared Strategies

| Strategy | Main idea | Role |
| --- | --- | --- |
| Strategy A | Input-level conditioning by concatenating bbox mask to the U-Net input | Direct bbox-conditioning baseline |
| Strategy B | LoRA baseline with bbox-weighted loss and inference-time latent/pixel constraint | Low-cost constraint baseline |
| Strategy C | ControlNet-style feature-level conditioning | Main spatial-control method |
| Strategy C+ | Inner/outer two-channel ControlNet conditioning | Improved Strategy C variant |

## Training Summary

| Run | Steps | Epochs | Final loss | Last-100 mean loss | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Strategy B | 2560 | 10 | 0.0825 | 0.3691 | LoRA-only bbox-constrained baseline |
| Strategy C | 2560 | 10 | 0.0171 | 0.2746 | Single-channel ControlNet-style conditioning |
| Strategy C fixed-location ablation | 2560 | 10 | 0.0202 | 0.2523 | All pseudo-target objects relocated to top-left |
| Strategy C+ inner/outer ablation | 2560 | 10 | 0.0536 | 0.2948 | Two-channel inner bbox + outer edit region |

Training loss is used mainly as a sanity check. The core evaluation is qualitative spatial controllability and visual fidelity.

## Main Observations

### Strategy A

Strategy A injects bbox information directly into the U-Net input. It is conceptually simple and useful as an input-level conditioning baseline. However, changing the pretrained input channel structure makes training less stable. In earlier checks, direct FP16 training for Strategy A produced non-finite values, while FP32 smoke training was stable.

### Strategy B

Strategy B is the lightest baseline. It keeps the pretrained U-Net input unchanged and relies on LoRA training plus bbox-weighted loss and inference-time blending/compositing. This helps preserve background outside the bbox, but bbox controllability is less intrinsic because the model itself does not receive explicit spatial features during denoising.

### Strategy C

Strategy C uses a ControlNet-style branch to inject bbox-conditioned residual features into the frozen editing U-Net. It shows the clearest spatial-control behavior among the original strategies. In bbox sweep results, objects generally follow the specified location better than the lighter Strategy B baseline.

However, single-channel conditioning has a limitation: the bbox can be interpreted simultaneously as object extent, object scale, and edit region. This creates a tradeoff between tight placement and boundary quality.

### Strategy C+

Strategy C+ separates object extent and edit region with two condition channels:

```text
channel 1: inner bbox, representing object size and placement
channel 2: outer padded bbox, representing the surrounding edit/blending region
```

Qualitatively, this improves the single-channel Strategy C failure mode where larger padded bboxes make the generated object grow. C+ preserves the idea that the object belongs in the inner box while giving the model extra surrounding space for boundary details and background compatibility.

## Fixed-Location Ablation

The fixed-location ablation tests whether bbox control generalizes beyond the training location distribution.

Training setup:

```text
All training pseudo-target objects are relocated to the top-left corner.
The corresponding training bbox condition is also placed at the top-left corner.
```

Inference setup:

```text
The same source image, prompt, seed, and checkpoint are used.
Only the inference-time bbox is changed:
zero / top-left / center / bottom-right / original
```

Key result:

```text
Even though training only uses top-left placements, bottom-right inference bboxes can still move the generated object to the bottom-right region.
```

Interpretation:

```text
The model learns a transferable bbox-conditioned spatial signal rather than merely memorizing the fixed top-left training location.
```

Limitations:

```text
Spatial placement can succeed while visual quality remains imperfect.
Fine structures such as hair may be truncated.
Lighting, contact, shadow, and local background compatibility are not always natural.
```

## Padding Sweep Follow-up

Inference-time hard padding was tested to reduce boundary artifacts:

```text
padding = 0 / 8 / 16 / 24
```

Observation:

```text
Small hard padding can improve boundary artifacts slightly.
Large hard padding causes object scale drift: the object tends to grow with the enlarged bbox.
```

Inference-only soft padding was also tested, but it made generated objects look strange. This is likely because the model was not trained with soft grayscale bbox conditions, making soft masks out-of-distribution at inference time.

Conclusion:

```text
Inference-only padding is useful as a diagnostic, but not a complete fix.
The stronger fix is to train the model with a representation that separates object extent from edit region.
```

## Final Conclusion

The experiments suggest that feature-level conditioning is the most effective direction for bbox-guided object addition. Strategy C provides clear spatial control, and the fixed-location ablation shows that this control can generalize to unseen inference locations. However, single-channel bbox conditioning entangles object scale with edit region, which can produce boundary artifacts or scale drift.

The inner/outer two-channel variant, Strategy C+, addresses this issue by explicitly separating the expected object extent from the surrounding blending region. Qualitative results indicate improved boundary/context behavior while maintaining spatial controllability.

## Report-Ready Wording

```text
We compare four bbox-conditioning strategies for object addition. Strategy A injects the bbox mask at the U-Net input level, Strategy B uses LoRA with bbox-weighted loss and inference-time constraints, Strategy C introduces a ControlNet-style feature-level bbox branch, and Strategy C+ extends Strategy C with inner/outer two-channel conditioning. Among the original strategies, Strategy C provides the strongest spatial controllability. A fixed-location ablation further shows that Strategy C can generalize beyond the training location distribution: although all training objects are relocated to the top-left corner, inference-time bboxes at unseen locations such as the bottom-right can still move the generated object accordingly.

However, qualitative results reveal that spatial control alone is insufficient for high-quality object insertion. Single-channel bbox conditioning entangles object extent and edit region, causing boundary artifacts and scale drift when the bbox is enlarged. To address this, Strategy C+ separates the inner object bbox from an outer padded edit region. This design better preserves object scale while providing additional context for boundary and background blending.
```
