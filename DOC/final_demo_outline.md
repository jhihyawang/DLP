# Final Demo Outline

This document organizes the stable parts of the final presentation/report/poster. Result analysis and final conclusions should be completed after all four strategies finish matched training and inference.

## 1. Introduction

### Background

Image editing diffusion models can add objects to an existing image according to a text instruction. A typical object-addition model takes:

```text
source image + text instruction -> edited image
```

However, text-only object addition does not provide precise spatial control. The model may decide where to place the new object based on scene priors, training distribution, or artifacts in the source image.

### Problem

This project studies bounding-box-guided object addition:

```text
source image + text instruction + bounding box -> edited image
```

The goal is to generate the requested object near the specified bounding box while preserving the background outside the editing region.

### Motivation

Spatial control is important because users often want to specify not only what object to add, but also where it should appear. Without explicit position conditioning, object addition may be difficult to control, especially when:

```text
the source image contains strong location priors
the removed object leaves artifacts
the requested object can plausibly appear in multiple places
the desired placement differs from the original object location
```

### Research Question

The central question is:

```text
Can a diffusion-based object addition model learn bbox-conditioned spatial control?
```

A stronger version of the question is tested through fixed-location ablation:

```text
If all training targets place the object at the top-left corner,
can the model still place the object at a bottom-right bbox during inference?
```

This separates true bbox-conditioned control from simple memorization of training-location bias.

### Contributions

This project compares four strategies for bbox-guided object addition:

```text
Strategy A: input-level bbox conditioning
Strategy B: latent/pixel-level bbox constraint
Strategy C: ControlNet-style feature-level conditioning
Strategy C+: inner/outer two-channel ControlNet conditioning
```

The project also includes ablation studies for spatial generalization, bbox padding, and the separation of object extent from edit region.

## 2. Proposed Method

### Base Task Formulation

Given a training sample from PIPE:

```text
source image: image with an object removed
target image: image containing the object
instruction: object addition text prompt
object mask: binary mask of the added object
bbox: bounding box derived from the object mask
```

The model is trained to denoise target-image latents conditioned on:

```text
source image latent
text embedding
bbox-related spatial condition
```

The bbox condition is derived from the PIPE object mask. Depending on the strategy, it is injected at the input level, used in loss/inference constraints, or provided to a ControlNet-style branch.

### Strategy A: Input-Level Conditioning

Strategy A directly concatenates the bbox condition to the U-Net input.

Conceptually:

```text
U-Net input = noisy target latent + source image latent + bbox latent
```

Implementation idea:

```text
expand U-Net conv_in to accept extra bbox channel
freeze VAE and text encoder
train U-Net LoRA and modified input convolution
use bbox-weighted denoising loss
```

Strength:

```text
The bbox signal enters the denoising network directly.
```

Limitation:

```text
Changing the pretrained U-Net input distribution can make training less stable.
```

### Strategy B: Latent/Pixel-Level Constraint

Strategy B keeps the pretrained U-Net input unchanged and uses bbox mainly as a training/inference constraint.

Training:

```text
train U-Net LoRA
use bbox-weighted denoising loss
```

Inference:

```text
blend latents outside bbox toward source image latents
optionally composite generated image with source image outside bbox
```

Strength:

```text
Low-cost baseline with minimal changes to the pretrained model.
```

Limitation:

```text
Spatial control is mostly imposed by loss and inference constraints, not by explicit feature-level conditioning.
```

### Strategy C: Feature-Level ControlNet Conditioning

Strategy C uses a ControlNet-style branch to inject bbox-conditioned residual features into the frozen editing U-Net.

Architecture:

```text
frozen InstructPix2Pix / Paint-by-Inpaint U-Net
trainable ControlNet branch
single-channel spatial condition
multi-level residual injection into the U-Net
```

Training:

```text
freeze VAE, text encoder, and base U-Net
train only the ControlNet branch
use bbox/object-mask weighted denoising loss
```

Strength:

```text
The bbox condition influences denoising through multi-level spatial features.
This is the main proposed spatial-control strategy.
```

Limitation:

```text
Single-channel conditioning can entangle object extent, object scale, and editable region.
```

### Strategy C+: Inner/Outer Two-Channel Conditioning

Strategy C+ is an improved variant motivated by the observed scale-drift issue in Strategy C.

Problem in single-channel conditioning:

```text
Increasing bbox padding can improve boundary artifacts,
but the generated object may also become larger.
```

This suggests that the model interprets a single bbox channel as both:

```text
object extent
editable / blending region
```

Strategy C+ separates these roles:

```text
channel 1: inner bbox
  controls expected object size and placement

channel 2: outer padded bbox
  controls surrounding edit/blending region
```

Expected benefit:

```text
preserve object scale using inner bbox
provide extra context for boundary details, shadows, and background compatibility
reduce scale drift caused by padded single-channel bbox
```

### Fixed-Location Pseudo-Target Construction

For spatial generalization ablation, each training object can be relocated to a fixed position such as the top-left corner.

For each sample:

```text
1. crop the object from the target image using the object mask
2. paste the object crop into the source image at the fixed location
3. create a new mask/bbox condition at the pasted location
4. train with the pseudo target and matching spatial condition
```

Important rule:

```text
The target image and bbox condition must be spatially consistent.
```

This avoids contradictory supervision where the target object remains at the original location but the bbox condition is moved elsewhere.

## 3. Experiments

### Dataset

The experiments use the PIPE dataset and PIPE_Masks:

```text
PIPE:
  source image
  target image
  object addition instruction
  image/object metadata

PIPE_Masks:
  object masks aligned by img_id and ann_id
```

The object mask is used to compute:

```text
object region
bounding box
bbox mask
pseudo target for fixed-location ablation
```

### Compared Methods

The final comparison includes four strategies:

| Method | Conditioning type | Trainable modules |
| --- | --- | --- |
| Strategy A | input-level bbox channel | U-Net LoRA + modified conv_in |
| Strategy B | bbox-weighted loss + inference constraint | U-Net LoRA |
| Strategy C | single-channel ControlNet condition | ControlNet branch |
| Strategy C+ | two-channel inner/outer ControlNet condition | ControlNet branch |

### Matched Training Setup

For fair comparison, the four strategies should be trained with matched high-level settings:

```text
model: paint-by-inpaint/add-base
max_train_samples: 256
epochs: 10
total steps: 2560
batch size: 1
image size: 512
learning rate: 1e-5
bbox loss weight: 8.0
seed: 1234
```

Checkpoints are stored on the large data disk, while logs and representative result images are stored in the project directory.

Unified script:

```bash
scripts/run_all_strategies_train_infer.sh
```

### Inference Setup

For qualitative comparison, all strategies use matched inference settings:

```text
split: test
indices: 0 / 1 / 2 by default
inference steps: 50
seed: 1234
guidance scale: 7.0
image guidance scale: 1.5
```

The output grid includes:

```text
source image
baseline output
strategy output
target image
```

Optional bbox sweep:

```text
top-left
top-right
center
bottom-left
bottom-right
original
```

### Fixed-Location Ablation

Purpose:

```text
Test whether the model learns bbox-conditioned spatial control
instead of memorizing the training location distribution.
```

Training:

```text
all pseudo-target objects are relocated to top-left
corresponding bbox/mask condition is also placed at top-left
```

Inference:

```text
same source image, prompt, seed, and checkpoint
change only the inference bbox:
zero / top-left / center / bottom-right / original
```

Key question:

```text
Can a model trained only with top-left placements generate the object at bottom-right during inference?
```

### Padding and Boundary Study

Observation from Strategy C:

```text
Tight bbox improves placement precision but may truncate fine details.
Hard bbox padding can improve boundary artifacts but may enlarge the object.
Inference-only soft padding can make objects look strange because the model was not trained with soft conditions.
```

This motivates Strategy C+:

```text
Use inner bbox for object extent.
Use outer bbox for edit/blending region.
```

### Evaluation Criteria

Qualitative evaluation focuses on:

```text
spatial accuracy:
  Does the object follow the bbox?

object quality:
  Is the object recognizable and realistic?

boundary quality:
  Are fine structures such as hair or thin parts truncated?

background compatibility:
  Are lighting, contact, shadow, and texture consistent?

background preservation:
  Is the area outside the edit region preserved?
```

Quantitative metrics can be added if time allows:

```text
training loss curve
bbox/object overlap proxy
outside-region reconstruction difference
CLIP-based text-image alignment
LPIPS or perceptual difference inside/outside bbox
```

## 4. Results

To be completed after all four matched strategies finish training and inference.

Recommended result figures:

```text
1. A/B/C/C+ qualitative comparison on the same test images
2. bbox sweep comparison
3. fixed-location ablation comparison
4. padding / boundary behavior comparison
```

## 5. Conclusion

To be completed after final result selection.
