# Final Presentation / Report Draft

Project title:

```text
BBox-Guided Object Addition with Diffusion-Based Image Editing
```

This document is written as a combined final report and slide-preparation draft. It explains the project goal, implementation, experiments, evaluation metrics, current results, and suggested presentation script.

Important result status:

```text
PIPE A/B/C/C+ main comparison:
  full PIPE test inference/evaluation is available for pipe_test_0 to pipe_test_751.
  Old pilot files test_0, test_1, and test_2 still exist in the result folder, so the clean table excludes them.

MagicBrush generalization:
  corrected full dev split inference/evaluation is available under magicbrush_generalization_maskdiff.
  BBoxes are derived from source-mask difference, with source-target difference as fallback.
```

## 1. Project Goal

The goal of this project is to build a controllable object-addition system for diffusion-based image editing.

Given:

```text
source image
text instruction
target bounding box
```

the system should generate an edited image where the requested object is added at the user-specified region while preserving the original background as much as possible.

In ordinary text-guided image editing, the user can describe what object to add, but the model often decides the location by itself. In this project, we study how to add an explicit bounding-box condition so that the user can control both:

```text
what object to add
where the object should appear
```

The desired output should satisfy four goals:

```text
1. The generated object matches the text instruction.
2. The generated object appears inside or near the requested bbox.
3. Regions outside the edit area remain close to the source image.
4. The object blends naturally with the scene.
```

### Slide Script

```text
This project focuses on controllable object addition.

Given a source image, a text instruction, and a user-specified bounding box,
we want the model to add the requested object at the specified location.

The task is not only about generating the correct object.
It also requires spatial control and background preservation.
Therefore, our project studies how to inject bbox information into a diffusion-based object-addition model.
```

## 2. Background and Motivation

This project is based on diffusion image editing. We use `paint-by-inpaint/add-base` as the base model. It is designed for object addition: given a source image and a text instruction, it generates a target image with a new object.

However, the original model does not provide explicit user control over the object location. This motivates our bbox-guided extension.

Relevant techniques:

```text
Diffusion image generation
Image-to-image editing
InstructPix2Pix-style conditional generation
Paint-by-Inpaint object addition
LoRA fine-tuning
ControlNet-style spatial conditioning
Mask and bbox based object localization
```

The key idea is to convert object masks into bounding boxes and use those boxes as spatial conditions during training and inference.

## 3. Dataset and Data Processing

### 3.1 PIPE Dataset

The main training and in-domain evaluation dataset is PIPE.

Resources:

```text
paint-by-inpaint/PIPE
paint-by-inpaint/PIPE_Masks
```

PIPE provides paired object-addition samples:

```text
source_img
target_img
text instruction
metadata such as img_id and ann_id
```

PIPE_Masks provides object masks. For each PIPE sample, the project matches the object mask using:

```text
(img_id, ann_id)
```

Then the preprocessing pipeline constructs:

```text
object_mask: binary mask of the added object
bbox: tight bounding box computed from object_mask
bbox_mask: binary rectangle mask generated from bbox
source_img: resized to 512 x 512
target_img: resized to 512 x 512
instruction: selected from available instruction fields
```

Implementation file:

```text
framework/sourcecode/pipe_bbox_dataset.py
```

Important functions/classes:

```text
load_pipe_split()
load_pipe_masks_split()
build_mask_index()
mask_to_bbox()
bbox_to_mask()
PIPEBBoxDataset
pipe_bbox_collate()
create_bbox_dataset()
```

### 3.2 MagicBrush Dataset

MagicBrush is used for cross-dataset generalization. The model is still trained on PIPE; no MagicBrush fine-tuning is performed.

Dataset:

```text
osunlp/MagicBrush
```

Fields used:

```text
source_img
target_img
instruction
mask_img
img_id
turn_index
```

For MagicBrush, the implementation now derives the object/edit mask from the difference between `source_img` and `mask_img`, then computes the bbox from that difference mask. During inspection, the original direct `mask_img` parsing was found to produce nearly full-image bbox overlays for most samples because `mask_img` is not a clean binary mask. If `mask_img` is unavailable or produces an invalid mask, the loader falls back to the source-target image difference.

Implementation additions:

```text
load_magicbrush_split()
diff_to_mask()
MagicBrushBBoxDataset
create_bbox_dataset(dataset_name="magicbrush")
```

The MagicBrush pipeline reuses the same inference and evaluator format as PIPE, so all automatic metrics can be computed in the same way.

## 4. System Overview

The overall pipeline is:

```text
source image
text instruction
object mask / bbox
        |
        v
strategy-specific bbox conditioning
        |
        v
diffusion image editing model
        |
        v
edited output image
        |
        v
automatic metrics + qualitative visualization
```

The project compares four strategies:

| Strategy | Main Idea | Trainable Part |
| --- | --- | --- |
| A | Input-level bbox conditioning | U-Net LoRA + modified input conv |
| B | BBox-weighted loss and inference-time constraint | U-Net LoRA |
| C | Single-channel ControlNet-style bbox conditioning | ControlNet branch |
| C+ | Inner/outer two-channel ControlNet conditioning | ControlNet branch |

All strategies share the same base model and are evaluated under matched inference settings when possible.

## 5. Strategy A: Input-Level BBox Conditioning

Strategy A injects the bbox condition directly into the denoising U-Net input.

Conceptually, the model receives:

```text
noisy target latent
source image latent
bbox latent / bbox condition
```

The U-Net input convolution is expanded so that the bbox information is available from the first denoising layer.

Implementation components:

```text
framework/sourcecode/model/modelA.py
framework/sourcecode/train.py
framework/sourcecode/infer_strategy_a.py
```

Training:

```text
1. Load paint-by-inpaint/add-base.
2. Add LoRA modules to the U-Net.
3. Modify the U-Net input path to accept bbox conditioning.
4. Encode target image into latent space.
5. Add random diffusion noise.
6. Predict noise with bbox-conditioned U-Net.
7. Optimize denoising loss, with bbox/object region emphasis.
```

Strength:

```text
The bbox signal is given directly to the denoising network.
```

Weakness:

```text
Changing the pretrained U-Net input distribution may make training less stable.
The model may not use the bbox deeply at multi-scale feature levels.
```

## 6. Strategy B: BBox-Weighted Loss and Inference Constraint

Strategy B keeps the pretrained U-Net architecture unchanged. Instead of modifying the model input, it uses bbox information through the loss function and inference-time constraints.

Main mechanisms:

```text
bbox-weighted denoising loss
latent blending outside the bbox during inference
optional pixel-level compositing outside the bbox
```

Implementation files:

```text
framework/sourcecode/model/modelB.py
framework/sourcecode/train_strategy_b.py
framework/sourcecode/infer_strategy_b.py
```

Weighted loss:

```text
per-pixel diffusion loss is computed in latent space
bbox mask is resized to latent resolution
inside-bbox pixels receive larger weight
outside-bbox pixels receive normal weight
```

Default bbox loss weight:

```text
bbox_loss_weight = 8.0
```

Inference constraint:

```text
During denoising, regions outside the bbox can be blended back toward the source latent.
After generation, the final image can be composited with the source outside the bbox.
```

Strength:

```text
It is lightweight and preserves the pretrained model architecture.
It gives very strong outside-bbox preservation.
```

Weakness:

```text
The bbox is not injected as a deep feature condition.
Therefore, it may preserve the background well but provide weaker generative spatial understanding.
```

## 7. Strategy C: Single-Channel ControlNet-Style Feature Conditioning

Strategy C adds a trainable ControlNet-style branch. The base editing U-Net is frozen, and the ControlNet branch learns to inject bbox-conditioned residual features into the denoising process.

Condition:

```text
single-channel bbox mask
```

Architecture idea:

```text
bbox mask
   |
   v
ControlNet-style condition encoder
   |
   v
multi-level residual features
   |
   v
frozen base U-Net denoising blocks
```

Implementation files:

```text
framework/sourcecode/model/modelC.py
framework/sourcecode/train_strategy_c.py
framework/sourcecode/infer_strategy_c.py
```

Training:

```text
1. Load the pretrained object-addition pipeline.
2. Freeze the original U-Net and VAE/text components.
3. Create a trainable ControlNet initialized from the U-Net structure.
4. Feed the bbox mask as the spatial conditioning image.
5. Inject residual features into the frozen U-Net.
6. Train only the ControlNet branch with bbox/object weighted denoising loss.
```

Additional training setting:

```text
bbox_shift_prob = 0.5
bbox_placement = random
```

This introduces bbox movement during training so the model does not only memorize original object locations.

Strength:

```text
Spatial information is injected at multiple feature levels.
The frozen base U-Net helps preserve pretrained editing ability.
```

Weakness:

```text
A single bbox channel may mix two meanings:
object extent and editable region.
If the bbox is enlarged to improve blending, the object size may also drift.
```

## 8. Strategy C+: Inner/Outer Two-Channel ControlNet Conditioning

Strategy C+ extends Strategy C by using two spatial condition channels:

```text
channel 1: inner bbox
  expected object location and scale

channel 2: outer padded bbox
  surrounding edit/blending region
```

Default padding:

```text
outer_bbox_padding = 24 pixels
```

Motivation:

```text
A tight bbox gives precise placement but may cut off boundary details.
A padded bbox gives more room for shadows and blending but may enlarge the generated object.
C+ separates object extent from edit region to reduce this conflict.
```

Implementation:

```text
same train_strategy_c.py and infer_strategy_c.py
control-conditioning-mode = inner-outer
```

Expected advantages:

```text
better object scale preservation
more room for local blending
less scale drift than a single padded bbox channel
improved separation between object location and edit area
```

## 9. Difficulties and Uniqueness

### 9.1 Difficulty: Spatial Control vs. Image Quality

The first difficulty is that spatial correctness and visual quality are not the same objective.

For bbox-guided object addition, a model may place the object near the requested bbox but still fail visually:

```text
the object boundary may be cut by the bbox
fine structures such as hair, legs, handles, or shadows may disappear
lighting and color may not match the source image
the generated object may have unrealistic scale
the local background may contain artifacts
```

Therefore, this project cannot evaluate only bbox alignment. It must also consider:

```text
semantic correctness
background preservation
perceptual similarity
boundary quality
object scale
```

This is why the project uses both automatic metrics and qualitative grids.

### 9.2 Difficulty: BBox Has Two Meanings

A bounding box can represent two different concepts:

```text
1. object extent:
   where the object should be and how large it should be

2. edit region:
   where the model is allowed to modify pixels for blending, shadows, and boundaries
```

If both concepts are represented by a single bbox channel, the model may confuse them. For example:

```text
tight bbox:
  better scale control, but may truncate object boundaries

padded bbox:
  more room for blending, but may enlarge the generated object
```

This difficulty motivates Strategy C+, where the inner bbox controls object extent and the outer bbox controls the edit/blending region.

### 9.3 Difficulty: Source-Image Location Priors

PIPE source images are created by removing objects from target images. Even after removal, the source image may still contain clues about the original object location:

```text
inpainting artifacts
background holes
remaining shadows
scene layout priors
object-context relationship
```

As a result, a model may appear to perform well simply because it restores the object near the original location, not because it truly follows the provided bbox.

To address this, the project includes a fixed-location ablation:

```text
train object placement: fixed top-left
test placement: top-left, center, bottom-right, original, zero
```

This ablation tests whether the model can follow unseen inference bboxes instead of memorizing a training location or relying on source-image priors.

### 9.4 Difficulty: Fair Evaluation of Generative Editing

Object addition is generative, so there can be multiple valid outputs for the same prompt and bbox. A generated image can be visually acceptable even if it is not pixel-identical to the dataset target.

This makes evaluation difficult:

```text
L1/MSE may penalize valid but different object appearances
CLIP may focus on global semantics rather than bbox placement
changed-region IoU can be misleading if the changed area is too large
LPIPS depends on perceptual backbone availability
qualitative examples can be biased if selected after seeing outputs
```

The project therefore separates:

```text
quantitative evaluation:
  same split, same indices, same seed, same inference settings

qualitative visualization:
  selected examples used only to explain behavior, not as the main metric
```

### 9.5 Uniqueness of This Project

The unique part of this project is not only adding bbox input to an image editing model. The project systematically studies bbox control at multiple levels of the diffusion pipeline:

```text
Strategy A:
  input-level bbox conditioning

Strategy B:
  bbox-weighted training loss and inference-time preservation

Strategy C:
  feature-level ControlNet-style spatial conditioning

Strategy C+:
  two-channel inner/outer feature-level conditioning
```

This comparison helps answer where bbox information should be injected for controllable object addition.

The project also includes two important experimental designs:

```text
1. fixed-location ablation:
   tests whether bbox control generalizes beyond the training placement distribution

2. MagicBrush generalization:
   evaluates whether PIPE-trained bbox control transfers to an external image-editing dataset
```

The proposed C+ design is the main architectural uniqueness:

```text
inner bbox:
  controls object location and scale

outer bbox:
  controls surrounding edit and blending area
```

This explicitly separates object extent from edit region, which is not handled by a simple single-channel bbox condition.

### Slide Script

```text
The main difficulty of this project is that bbox control is not just a localization problem.
The model must place the object correctly, but it also needs to preserve the background and generate natural boundaries.

Another challenge is that a bbox has two meanings.
It can describe the expected object size, but it can also describe the region where editing is allowed.
If we use only one bbox channel, these two meanings are mixed together.
This motivated our C+ design, where the inner bbox represents object extent and the outer bbox represents the blending region.

We also need to avoid being fooled by dataset priors.
In PIPE, the source image may still contain clues about the removed object location.
So we designed a fixed-location ablation to test whether the model truly follows the inference bbox.

The uniqueness of this project is that we compare bbox conditioning at four different levels,
and we further evaluate cross-dataset generalization on the full MagicBrush dev split.
```

## 10. Training Setup

Main matched training setup:

```text
base model: paint-by-inpaint/add-base
dataset: PIPE
image size: 512
training samples: 256
epochs: 10
batch size: 1
learning rate: 1e-5
bbox loss weight: 8.0
seed: 1234
inference steps: 50
guidance scale: 7.0
image guidance scale: 1.5
```

Main script:

```text
scripts/run_all_strategies_train_infer.sh
```

Outputs:

```text
checkpoints:
  /media/.../checkpoints/all_strategies_matched

results:
  results/all_strategies_matched

logs:
  logs/all_strategies_matched
```

Final training summary:

| Run | Steps | Epochs | Final Loss ↓ | Mean Loss ↓ |
| --- | ---: | ---: | ---: | ---: |
| strategy_b | 2560 | 10 | 0.0847 | 0.3221 |
| strategy_c | 2560 | 10 | 0.0252 | 0.2697 |
| strategy_cplus | 2560 | 10 | 0.0175 | 0.2723 |
| strategy_c_top-left_ablation | 2560 | 10 | 0.0304 | 0.2725 |
| strategy_cplus_top-left_ablation | 2560 | 10 | 0.0515 | 0.2722 |

Interpretation:

```text
Training loss alone is not a final quality metric.
It mainly confirms that each strategy can be optimized under the matched training budget.
Generation quality still requires image metrics and visual comparison.
```

## 11. Inference Pipeline

Each inference script loads:

```text
source image
target image
text instruction
bbox
trained checkpoint
fixed random seed
```

Then it saves:

```text
source image
target image
bbox overlay
generated output
grid image
info.txt metadata file
```

The `info.txt` file is important because the evaluator reads:

```text
prompt
split
index
bbox coordinates
strategy name
dataset name
inference settings
```

Main inference scripts:

```text
framework/sourcecode/infer_strategy_a.py
framework/sourcecode/infer_strategy_b.py
framework/sourcecode/infer_strategy_c.py
scripts/run_all_strategies_train_infer.sh
scripts/run_magicbrush_generalization.sh
```

BBox sweep modes:

```text
corners
small-corners
vertical
```

These modes test whether the output follows changed bbox positions while all other factors remain fixed.

## 12. Ablation Studies

### 12.1 Fixed-Location Ablation

Purpose:

```text
Test whether the model truly learns bbox-guided spatial control,
instead of relying on original object-location priors in PIPE.
```

Problem:

```text
PIPE source images are produced by object removal.
The source image may still contain removal artifacts, shadows, holes, or layout cues.
The model may learn to restore the object at the original location even without using bbox.
```

Ablation design:

```text
During training:
  paste the object crop into a fixed top-left location
  move the bbox/mask condition to the same top-left location

During inference:
  test top-left
  test center
  test bottom-right
  test original PIPE bbox
  test zero bbox
```

If the model follows center or bottom-right bboxes, then the bbox condition is acting as a real spatial control signal.

Script:

```text
scripts/run_fixed_location_ablation.sh
```

### 12.2 BBox Variation / Boundary Ablation

Purpose:

```text
Test how bbox size and location affect spatial control, object scale, and boundary quality.
```

Compared settings:

```text
original bbox
top-left
top-right
center
bottom-left
bottom-right
small-corner variants
```

This ablation is especially useful for comparing Strategy C and C+:

```text
C uses one bbox channel.
C+ uses inner/outer channels to separate object extent from edit region.
```

Script:

```text
scripts/run_bbox_variation_grid_infer.sh
```

## 13. Evaluation Metrics

Automatic evaluator:

```text
framework/sourcecode/evaluate.py
```

The evaluator reads generated results and computes metrics per output image and per strategy.

### 13.1 Inside-BBox Target Similarity

Metric:

```text
inside_output_target_l1
inside_output_target_mse
```

Meaning:

```text
Compares generated output with dataset target inside the requested bbox.
Lower is better.
```

Use:

```text
Approximates whether the object region resembles the target edit.
```

Limitation:

```text
Diffusion generation can have multiple valid outputs.
Pixel-level target similarity is useful but not perfect.
```

### 13.2 Outside-BBox Background Preservation

Metric:

```text
outside_output_source_l1
outside_output_source_mse
```

Meaning:

```text
Compares generated output with source image outside the bbox.
Lower is better.
```

Use:

```text
Measures unnecessary background changes.
```

Important note:

```text
Strategy B has a hard outside-bbox copy/compositing constraint.
Therefore, it can achieve extremely low or zero outside-bbox error by design.
This should be interpreted as a preservation property, not necessarily better generation quality.
```

### 13.3 Changed-Region BBox Alignment

The evaluator estimates changed pixels by comparing output and source. It then measures how the changed region relates to the requested bbox.

Metrics:

```text
changed_bbox_iou
changed_inside_bbox_ratio
changed_to_bbox_center_distance_norm
```

Meaning:

```text
changed_bbox_iou:
  overlap between changed region and requested bbox

changed_inside_bbox_ratio:
  fraction of changed pixels that lie inside the bbox

changed_to_bbox_center_distance_norm:
  normalized distance between changed-region center and bbox center
```

Limitations:

```text
If a method changes a very large region, IoU can be high even if the edit is not visually precise.
If a method makes a small localized edit inside a large mask, IoU can be low even when the object is reasonable.
Therefore these metrics must be read together with visual examples.
```

### 13.4 CLIP Text-Image Similarity

Metric:

```text
clip_output_prompt_similarity
```

Meaning:

```text
CLIP cosine similarity between generated output and text instruction.
Higher is better.
```

Use:

```text
Estimates semantic alignment between the generated image and the prompt.
```

Limitation:

```text
CLIP is a global image-text metric.
It may not focus specifically on whether the object appears inside the bbox.
```

### 13.5 LPIPS Perceptual Distance

Metric:

```text
lpips_output_target_whole
lpips_output_target_inside_bbox
```

Meaning:

```text
Perceptual distance between generated output and target image.
Lower is better.
```

Current status:

```text
PIPE and MagicBrush LPIPS values are available in the latest evaluation CSVs.
MagicBrush LPIPS is from the random-backbone fallback file, so it should be treated as an auxiliary trend unless standard pretrained LPIPS is available.
```

## 14. Current Quantitative Results

### 14.1 PIPE Main Comparison

Important status:

```text
This table is the clean full PIPE test evaluation.
It uses pipe_test_0 to pipe_test_751, for 752 outputs per strategy.
Old pilot outputs test_0, test_1, and test_2 are excluded from this table.
```

| Strategy | Num Outputs | Inside L1 ↓ | Outside L1 ↓ | CLIP ↑ | LPIPS Whole ↓ | Changed IoU ↑ | Changed Inside Ratio ↑ | Center Dist Norm ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strategy_a | 752 | 0.1638 | 0.0407 | 0.2488 | 0.1530 | 0.2001 | 0.3309 | 0.1540 |
| strategy_b | 752 | 0.1669 | 0.0040 | 0.2478 | 0.0823 | 0.3621 | 0.7566 | 0.0344 |
| strategy_c | 752 | 0.1834 | 0.0385 | 0.2520 | 0.1502 | 0.2647 | 0.4097 | 0.1383 |
| strategy_cplus | 752 | 0.1872 | 0.0350 | 0.2503 | 0.1422 | 0.2492 | 0.4136 | 0.1350 |

Observation:

```text
Strategy B has the strongest spatial localization and background preservation.
It achieves the lowest outside-bbox error, highest changed-region IoU, highest changed-inside-bbox ratio, and lowest center distance.
Strategy C has the highest CLIP score, suggesting slightly better text-image alignment.
Strategy C+ has the lowest whole-image LPIPS among the learned ControlNet-style variants.
```

### 14.2 Fixed-Location Ablation

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

Observation:

```text
The model can respond to unseen bbox placements such as center and bottom-right.
Bottom-right remains harder than center/original placements.
C+ improves bottom-right center distance over C, but results still require qualitative inspection.
```

### 14.3 Final Demo BBox Variation

| Run | Num BBoxes | Inside L1 ↓ | Outside L1 ↓ | Changed IoU ↑ | Changed Inside Ratio ↑ | Center Dist Norm ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strategy_b_corners | 6 | 0.0896 | 0.0039 | 0.1690 | 0.6989 | 0.1261 |
| strategy_b_small-corners | 6 | 0.0654 | 0.0000 | 0.0591 | 1.0000 | 0.0844 |
| strategy_c_corners | 6 | 0.1266 | 0.0364 | 0.2057 | 0.4628 | 0.1830 |
| strategy_c_small-corners | 6 | 0.1279 | 0.0296 | 0.1305 | 0.3797 | 0.1951 |
| strategy_cplus_corners | 6 | 0.1098 | 0.0346 | 0.2118 | 0.4735 | 0.1731 |
| strategy_cplus_small-corners | 6 | 0.1264 | 0.0297 | 0.1412 | 0.4059 | 0.1883 |

Observation:

```text
Small-corner bboxes are more difficult for C/C+ because the available object region becomes much smaller.
B preserves outside-bbox pixels strongly, but this also means its edit can be constrained by hard compositing.
```

### 14.4 MagicBrush Full-Dev Generalization

This is the corrected MagicBrush generalization result. The model is trained on PIPE and evaluated on the full MagicBrush dev split without MagicBrush fine-tuning. BBoxes are extracted from the difference between `source_img` and `mask_img`, with source-target difference as fallback.

Evaluation set:

```text
MagicBrush dev split
indices: 0-527
num outputs per strategy: 528
total outputs: 2112
```

Status:

```text
Corrected mask-difference run.
Use this table for MagicBrush generalization discussion.
LPIPS is from the random-backbone fallback evaluation file and should be treated as an auxiliary trend.
```

| Strategy | Num Outputs | Inside L1 ↓ | Outside L1 ↓ | CLIP ↑ | LPIPS Whole ↓ | Changed IoU ↑ | Changed Inside Ratio ↑ | Center Dist Norm ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strategy_a | 528 | 0.3595 | 0.3514 | 0.2471 | 0.2111 | 0.5242 | 0.5745 | 0.1203 |
| strategy_b | 528 | 0.2933 | 0.0000 | 0.2422 | 0.1175 | 0.7422 | 1.0000 | 0.0196 |
| strategy_c | 528 | 0.1573 | 0.0500 | 0.2445 | 0.0504 | 0.2151 | 0.7599 | 0.1275 |
| strategy_cplus | 528 | 0.1605 | 0.0445 | 0.2412 | 0.0497 | 0.2200 | 0.7748 | 0.1119 |

Observation:

```text
Strategy B remains the most spatially conservative method on MagicBrush.
It preserves the outside-bbox region almost perfectly and keeps all changed pixels inside the bbox.
Strategy C and C+ better match the target edited images, with much lower inside-bbox L1 and whole-image LPIPS than A/B.
C+ slightly improves over C in outside-bbox error, whole-image LPIPS, changed-region IoU, changed-inside-bbox ratio, and center distance.
```

## 15. Qualitative Result Discussion

The final presentation should include visual grids, because object addition can have multiple valid outputs and automatic metrics are imperfect.

Recommended visual comparisons:

```text
1. Same source/prompt/bbox across A/B/C/C+.
2. Original bbox vs moved bbox for spatial control.
3. C vs C+ comparison for object scale and boundary blending.
4. Fixed-location ablation examples.
5. MagicBrush generalization examples.
```

Main questions to discuss visually:

```text
Does the object follow the requested bbox?
Does the object identity match the prompt?
Does the object scale match the bbox?
Is the outside background preserved?
Are boundaries, shadows, and local blending natural?
Does C+ reduce the scale drift issue compared with single-channel C?
```

## 16. Fair Evaluation Protocol

For final reporting, use the following principles:

```text
1. Use the same evaluation split for all methods.
2. Use the same sample indices for all strategies.
3. Use the same seed, inference steps, image size, guidance scale, and checkpoint rule.
4. Report full-split metrics when feasible.
5. If only a subset is used, choose the subset before inspecting results.
6. Report the exact subset rule, for example first 100 samples or indices 0-99.
7. Use selected examples only for qualitative visualization, not as the main quantitative claim.
```

For the current project state:

```text
Full PIPE test evaluation is available for 752 samples per strategy.
Corrected MagicBrush full dev evaluation is available for 528 samples per strategy.
Fixed-location and bbox-variation experiments are still best treated as diagnostic/qualitative ablations.
```

## 17. Implementation Summary

Important project files:

| File | Purpose |
| --- | --- |
| `framework/sourcecode/pipe_bbox_dataset.py` | PIPE/MagicBrush dataset loading, mask-to-bbox processing |
| `framework/sourcecode/train.py` | Strategy A training |
| `framework/sourcecode/train_strategy_b.py` | Strategy B LoRA training |
| `framework/sourcecode/train_strategy_c.py` | Strategy C/C+ ControlNet training |
| `framework/sourcecode/infer_strategy_a.py` | Strategy A inference and bbox sweep |
| `framework/sourcecode/infer_strategy_b.py` | Strategy B inference, latent/pixel preservation |
| `framework/sourcecode/infer_strategy_c.py` | Strategy C/C+ inference and bbox conditioning |
| `framework/sourcecode/evaluate.py` | Automatic evaluation metrics |
| `scripts/run_all_strategies_train_infer.sh` | Main A/B/C/C+ experiment runner |
| `scripts/run_fixed_location_ablation.sh` | Fixed-location ablation runner |
| `scripts/run_bbox_variation_grid_infer.sh` | Demo bbox variation runner |
| `scripts/run_magicbrush_generalization.sh` | MagicBrush generalization runner |
| `DOC/experiment_run_guide.md` | Reproducible command guide |
| `results/evaluation/evaluation_summary.md` | Latest automatic evaluation summary |

Implementation highlights:

```text
1. Unified dataset API:
   create_bbox_dataset() supports both PIPE and MagicBrush.

2. Unified result format:
   inference scripts save source, target, output, bbox overlay, grid, and info.txt.

3. Unified evaluator:
   evaluate.py can compute all metrics from the generated result folders.

4. Strategy extensibility:
   Strategy C and C+ share the same ControlNet training script, selected by control-conditioning-mode.

5. Generalization support:
   MagicBrush can be evaluated without retraining, using the same checkpoints trained on PIPE.
```

## 18. Libraries and Open-Source Resources

| Library / Resource | Usage |
| --- | --- |
| PyTorch | Model training, tensor operations, denoising loss |
| torchvision | Image transforms |
| diffusers | Stable Diffusion / InstructPix2Pix / ControlNet pipeline |
| transformers | CLIP tokenizer, text encoder, CLIP evaluation |
| Hugging Face datasets | PIPE, PIPE_Masks, MagicBrush loading |
| PEFT / LoRA utilities | Lightweight U-Net adaptation |
| safetensors | Safe checkpoint storage |
| Pillow | Image loading, mask processing, bbox overlay, grids |
| tqdm | Training progress display |
| lpips | Perceptual distance evaluation |
| paint-by-inpaint/add-base | Base object-addition model |
| paint-by-inpaint/PIPE | Main object-addition dataset |
| paint-by-inpaint/PIPE_Masks | Object mask dataset |
| osunlp/MagicBrush | Cross-dataset generalization benchmark |

## 19. Difference from Existing Work

This project uses public pretrained models and datasets, but the project contribution is the bbox-guided object-addition study and implementation.

Differences:

```text
1. The original Paint-by-Inpaint model adds objects from image and text, but does not focus on explicit bbox placement control.

2. This project derives bbox conditions from PIPE_Masks and uses them during training/inference.

3. This project compares multiple bbox injection levels:
   input-level
   loss/inference-level
   feature-level ControlNet
   inner/outer feature-level ControlNet

4. Strategy C+ proposes a two-channel inner/outer condition to separate object extent from edit region.

5. The project includes fixed-location ablation to test whether the model truly follows bbox rather than source-image priors.

6. The project includes a corrected MagicBrush generalization pipeline using source-mask difference based bbox extraction.
```

## 20. Limitations

Current limitations:

```text
1. Automatic metrics cannot fully capture visual realism, object plausibility, and blending quality.
2. Pixel metrics are imperfect because diffusion outputs are not required to match the target exactly.
3. Changed-region IoU can be misleading when the changed region is very large or the bbox/mask is large.
4. CLIP is global and does not directly measure bbox alignment.
5. Strategy B outside-bbox preservation is partly enforced by hard compositing, so it is not a pure learned behavior.
6. Full-split inference requires significant GPU time, especially if the model is reloaded for every image.
7. Visual quality still needs human inspection for boundaries, shadows, and object realism.
```

## 21. Conclusion

This project studies bbox-guided object addition by adding explicit spatial control to a diffusion-based object-addition model.

The main findings are:

```text
1. BBox guidance can be injected at several levels, but feature-level conditioning is more effective for spatial control and generalization.
2. Strategy B preserves outside-bbox background extremely well because of hard inference constraints, but this does not necessarily imply better object generation.
3. On the full PIPE test set, Strategy B gives the strongest localization and background preservation, while Strategy C gives the best CLIP similarity.
4. Strategy C+ improves the separation between object extent and edit/blending region through inner/outer conditioning, and gives competitive perceptual quality.
5. Fixed-location ablation shows that bbox-conditioned models can respond to unseen bbox placements, though difficult locations such as bottom-right remain challenging.
6. On corrected MagicBrush generalization, C/C+ better match target edited images, while B remains the most spatially conservative method.
```

Final takeaway:

```text
Explicit bbox conditioning is useful for controllable object addition.
Among the tested designs, ControlNet-style feature conditioning, especially the C+ inner/outer design, provides the strongest current evidence for balancing spatial control, target similarity, and background preservation.
```

## 22. Suggested Final Slide Structure

```text
01 Project Goal
02 Task Definition and Motivation
03 Dataset and Preprocessing
04 System Overview
05 Strategy A/B/C/C+ Architecture
06 Training and Inference Pipeline
07 Ablation Study Design
08 Evaluation Metrics
09 PIPE Full-Test Results
10 Fixed-Location Ablation Results
11 MagicBrush Generalization Results
12 Qualitative Demo
13 Limitations
14 Conclusion
```

## 23. Suggested Project Goal Slide

Slide title:

```text
Project Goal: BBox-Guided Object Addition
```

Slide content:

```text
Input:
  source image + text instruction + bounding box

Output:
  edited image with the requested object added at the specified region

Goal:
  semantic correctness
  spatial controllability
  background preservation
  natural blending
```

Speaker script:

```text
This project aims to build a controllable image editing system for object addition.
Given a source image, a text instruction, and a user-specified bounding box,
the model should add the requested object at the specified location.

The key challenge is balancing three things:
the object must match the instruction,
the object must follow the bbox,
and the background outside the edit region should remain unchanged.

Therefore, we compare several bbox-conditioning strategies
and evaluate how well they control object placement and preserve image quality.
```

## 24. Member Contributions

Fill in actual member names before submission.

| Member | Contributions |
| --- | --- |
| Member 1 | Dataset processing, PIPE/PIPE_Masks loading, bbox extraction |
| Member 2 | Strategy A/B implementation, LoRA training, inference scripts |
| Member 3 | Strategy C/C+ ControlNet implementation, fixed-location ablation |
| Member 4 | Evaluation metrics, MagicBrush generalization, visualization, report/slides |

If the project is individual, replace the table with:

```text
All implementation, experiments, evaluation, and report preparation were completed by the author.
```

## 25. Confidentiality

No confidential data is used. The project uses public datasets, public pretrained models, and open-source libraries.
