# Fixed-Location BBox Training Ablation Study

## Purpose

This ablation study tests whether the model truly learns **bbox-conditioned spatial control**, rather than relying on source-image priors or memorizing the training location distribution.

Key question:

```text
If all training targets place the object in the top-left corner,
can the model still generate the object in the bottom-right corner at inference time?
```

If the model succeeds, the bbox condition may be interpreted as a generalizable spatial signal. If it fails, the model may only have learned training-location bias.

## Motivation

In the PIPE dataset, each source image is created by removing an object from a target image. Therefore, the source image may still contain strong clues about the original object location:

```text
object removal artifacts
background holes
shadows
contact surfaces
scene layout priors
```

Because of these cues, the model may learn to add the object back to its original location without using the bbox condition.

This ablation creates a stronger test by forcing all training targets to have the object in one fixed location, then checking whether inference-time bbox can move it elsewhere.

## Main Idea

During training, relocate every object to a fixed location, such as the top-left corner.

Original training sample:

```text
source image
target image
object mask
original bbox
instruction
```

Transformed training sample:

```text
source image
pseudo target image with object pasted into top-left
top-left object mask
top-left bbox
instruction
```

During inference, keep the source image, prompt, seed, and checkpoint fixed, but change only the bbox:

```text
zero bbox
top-left bbox
center bbox
bottom-right bbox
original bbox
```

## Important Rule

Do **not** only change the bbox while keeping the original target unchanged.

Wrong setup:

```text
target image: object remains at original location
bbox condition: moved to top-left
```

This creates contradictory supervision and encourages the model to ignore bbox.

Correct setup:

```text
target image: object is moved to top-left
bbox condition: top-left bbox
```

The target and the bbox condition must be spatially consistent.

## Training Data Construction

For each training sample:

1. Load source image.
2. Load target image.
3. Load object mask.
4. Compute the original bbox from the object mask.
5. Crop the object region from the target image.
6. Crop the corresponding object mask.
7. Choose a fixed placement, e.g. top-left.
8. Paste the object crop into the source image at the fixed placement.
9. Generate a new bbox mask at the pasted location.
10. Use the pseudo target and new bbox as the training pair.

## Computing the Fixed BBox

Given the original bbox:

```text
x1, y1, x2, y2
```

Object size:

```text
object_width  = x2 - x1
object_height = y2 - y1
```

For top-left placement:

```text
margin = 16
new_x1 = margin
new_y1 = margin
new_x2 = margin + object_width
new_y2 = margin + object_height
```

Clamp the bbox to image boundaries if necessary.

Possible placements:

```text
top-left
top-right
bottom-left
bottom-right
center
random
original
```

## Pseudo Target Generation

Let:

```text
source_img: image without the object
target_img: image with the object
object_mask: binary object mask
```

Crop the object:

```text
object_crop = target_img[:, y1:y2, x1:x2]
mask_crop   = object_mask[:, y1:y2, x1:x2]
```

Paste it into the fixed location:

```text
pseudo_target = source_img.clone()
source_region = pseudo_target[:, new_y1:new_y2, new_x1:new_x2]

pseudo_target[:, new_y1:new_y2, new_x1:new_x2] =
    object_crop * mask_crop + source_region * (1 - mask_crop)
```

Create the new condition mask:

```text
pseudo_mask = zeros_like(object_mask)
pseudo_mask[:, new_y1:new_y2, new_x1:new_x2] = mask_crop
```

Train using:

```text
source_img
pseudo_target
pseudo_mask or pseudo_bbox
instruction
```

## Recommended Strategy

This ablation is most meaningful for **Strategy C**, because Strategy C is the feature-level conditioning method from the proposal:

```text
frozen main U-Net
trainable ControlNet-style bbox branch
bbox mask injected as multi-level residual features
```

The same idea can also be tested on Strategy A and Strategy B.

## Current Repo Support

The repository now supports this ablation directly for Strategy C.

Training uses:

```bash
python framework/sourcecode/train_strategy_c.py \
  --output-dir checkpoints/strategy_c_ablation_top_left \
  --metrics-csv logs/strategy_c_ablation_top_left_metrics.csv \
  --max-train-samples 256 \
  --epochs 10 \
  --batch-size 1 \
  --num-workers 2 \
  --image-size 512 \
  --lr 1e-5 \
  --bbox-loss-weight 8.0 \
  --bbox-shift-prob 1.0 \
  --bbox-placement top-left \
  --bbox-placement-margin 16 \
  --save-every 1
```

Inference uses:

```bash
python framework/sourcecode/infer_strategy_c.py \
  --checkpoint-dir checkpoints/strategy_c_ablation_top_left \
  --checkpoint-tag final_epoch_0010_step_002560 \
  --split test \
  --index 0 \
  --steps 50 \
  --seed 1234 \
  --output-dir results/strategy_c_ablation_top_left \
  --bbox-sweep \
  --bbox-sweep-mode ablation
```

Or run the end-to-end helper:

```bash
scripts/run_strategy_c_fixed_bbox_ablation.sh
```

The helper trains with fixed top-left placement, finds the final checkpoint, and runs ablation inference for test indices `0 1 2` by default.

## Training Configuration

Recommended initial command:

```bash
export TMPDIR=/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def/tmp

python sourcecode/train_strategy_c.py \
  --batch-size 1 \
  --max-train-samples 4096 \
  --epochs 2 \
  --num-workers 2 \
  --lr 1e-5 \
  --bbox-loss-weight 16.0 \
  --bbox-shift-prob 1.0 \
  --bbox-placement top-left \
  --save-every 1
```

Important settings:

```text
--bbox-shift-prob 1.0
```

Every training sample is converted into a pseudo target.

```text
--bbox-placement top-left
```

Every pseudo target places the object in the top-left corner.

## Inference Evaluation

After training, fix:

```text
source image
prompt
random seed
model checkpoint
```

Only change the bbox condition.

Test locations:

```text
zero bbox
top-left
center
bottom-right
original bbox
```

Example:

```bash
python sourcecode/infer_strategy_c.py \
  --checkpoint-tag epoch_0002_step_008192 \
  --split test \
  --index 0 \
  --steps 50 \
  --seed 1234 \
  --bbox-sweep
```

The comparison grid should contain:

```text
source
baseline
zero bbox / zero output
top-left bbox / top-left output
center bbox / center output
bottom-right bbox / bottom-right output
original bbox / original output
target
```

## What to Observe

Key observations:

```text
Does top-left generation work?
Does bottom-right generation work?
Does center generation work?
Does zero bbox suppress object generation?
Does original bbox pull the object back to the source prior?
```

The most important comparison is:

```text
top-left vs bottom-right
```

The model was trained only with top-left targets, so bottom-right tests unseen-location generalization.

## Result Interpretation

### Case 1: Top-left works, bottom-right fails

The model learned training-location bias.

Interpretation:

```text
The model can learn a fixed spatial prior but does not generalize bbox control to unseen locations.
```

### Case 2: Top-left works, bottom-right also works

This is the desired outcome.

Interpretation:

```text
The model learns a generalizable bbox-conditioned spatial control mechanism.
```

### Case 3: Top-left fails

The model did not learn the fixed-location task.

Possible causes:

```text
pseudo target generation is incorrect
object mask is misaligned
bbox mask is not aligned with pasted object
learning rate is too low
training is too short
bbox loss weight is too weak
```

### Case 4: Zero bbox still generates an object

The model is still mainly driven by:

```text
prompt
source image prior
pretrained object addition prior
```

The bbox condition is not dominant.

## Metrics

Possible metrics:

```text
Edit-region IoU
Outside-region LPIPS
Inside-region LPIPS
CLIP score
```

For this ablation, the most important metric is spatial control:

```text
Does the generated object overlap the inference-time bbox?
```

A simple metric:

```text
Generated-object mask vs inference bbox IoU
```

If generated-object masks are unavailable, approximate evaluation can be done manually or with a segmentation model.

## Suggested Experiment Table

| Training placement | Inference bbox | Expected result if bbox control works |
|---|---|---|
| top-left | top-left | object appears top-left |
| top-left | center | object appears center |
| top-left | bottom-right | object appears bottom-right |
| top-left | zero | no object or much weaker object |
| top-left | original | object follows original bbox only if bbox is used |

## Report Wording

English:

```text
To evaluate whether the learned spatial condition generalizes beyond the training location distribution, we conduct a fixed-location training ablation. During training, all target objects are relocated to the top-left corner with corresponding top-left bbox conditions. During inference, we provide bbox conditions at unseen locations such as center and bottom-right. If the model truly learns bbox-conditioned spatial control, the generated object should follow the inference-time bbox despite never observing such placements during training.
```

Chinese:

```text
為了驗證模型是否真正學到可泛化的 bbox 空間控制，我們設計 fixed-location training ablation。訓練時將所有目標物件都移到左上角，並使用對應的左上角 bbox 作為條件；推論時則指定中央、右下角等訓練時未出現的位置。如果模型真的學到 bbox-conditioned spatial control，生成物件應該能跟隨推論時的 bbox，而不是固定出現在左上角。
```

## Expected Contribution

This ablation directly tests whether bbox conditioning is generalizable.

It distinguishes between:

```text
true spatial control
```

and:

```text
memorized training-location bias
source-image positional prior
```

Even if the result is negative, it is still valuable because it shows that object addition models require stronger counterfactual spatial supervision to learn controllable placement.

## Padding Sweep Follow-up

After the fixed-location ablation, Strategy C successfully places the object near unseen inference-time bbox locations such as the bottom-right corner. However, qualitative results show remaining visual issues:

```text
spatial placement: generally successful
object realism: imperfect
boundary quality: fine structures such as hair may be truncated
background compatibility: lighting, contact, and texture blending may be weak
```

An inference-only bbox padding sweep was tested to reduce hard-boundary artifacts:

```text
padding = 0 / 16 / 32 / 48
```

This improves boundary artifacts in some cases, but hard padding introduces a new failure mode:

```text
as the bbox is enlarged, the generated object also tends to become larger
```

This suggests that the current single-channel bbox condition is interpreted not only as an editable region, but also as an object extent / scale cue. Therefore, bbox padding creates a tradeoff:

```text
larger padding -> better boundary room and blending
larger padding -> more object scale drift and less precise size control
```

To reduce this issue, an inference-only soft padding variant was tested:

```bash
python framework/sourcecode/infer_strategy_c.py \
  --checkpoint-dir /media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def/checkpoints/strategy_c_ablation_top_left \
  --checkpoint-tag final_epoch_0010_step_002560 \
  --split test \
  --index 0 \
  --steps 50 \
  --seed 1234 \
  --output-dir results/strategy_c_soft_padding_sweep \
  --bbox-soft-padding-sweep 0 16 32 48
```

Soft padding keeps the original bbox at value 1.0 and gradually decays to 0 in the padded boundary region. The intended effect was to provide boundary/blending context without strongly telling the model that the whole padded rectangle is the object extent.

However, qualitative results show that inference-only soft padding can make the generated object look strange. This is likely because the current ControlNet branch was not trained with soft grayscale bbox conditions, so soft masks are out-of-distribution at inference time.

Current recommendation:

```text
Use hard padding only as a small inference-time sweep, e.g. 0 / 8 / 16 / 24.
Avoid large hard padding because it causes object scale drift.
Avoid inference-only soft padding unless the model is retrained with soft masks.
```

## Inner/Outer Conditioning Follow-up

The next model-level fix is to separate object size control from edit-region control during training. Instead of using a single condition channel, the model can be retrained with two ControlNet condition channels:

```text
channel 1: inner bbox
  - controls the expected object extent, size, and placement

channel 2: outer padded bbox
  - marks the surrounding edit/blending region
  - provides room for boundaries, hair, shadows, and local background adaptation
```

This is intended to address the main limitation of hard padding:

```text
single-channel padded bbox -> object may grow with bbox size
two-channel inner/outer bbox -> object size and blending region are represented separately
```

Training/inference helper:

```bash
scripts/run_strategy_c_inner_outer_ablation.sh
```

Default configuration:

```text
max_train_samples = 256
epochs = 10
steps = 2560
placement = top-left
control-conditioning-mode = inner-outer
outer-bbox-padding = 24
```

This trains a 2-channel ControlNet checkpoint and then evaluates the same fixed-location ablation sweep:

```text
zero / top_left / center / bottom_right / original
```

Helper script:

```bash
scripts/run_strategy_c_padding_sweep.sh
```

By default, this helper runs the conservative hard padding sweep. To reproduce the soft padding sweep:

```bash
MODE=soft scripts/run_strategy_c_padding_sweep.sh
```
