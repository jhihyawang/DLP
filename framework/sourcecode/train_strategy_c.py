import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sourcecode.model.modelC import (
    count_parameters,
    iter_trainable_parameters,
    load_strategy_c_pipeline,
)
from sourcecode.pipe_bbox_dataset import PIPEBBoxDataset, pipe_bbox_collate


DATA_DISK = Path("/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def")
DEFAULT_OUTPUT_DIR = DATA_DISK / "strategy_c_checkpoints"


def encode_target_latents(vae, images):
    images = images * 2.0 - 1.0
    latents = vae.encode(images).latent_dist.sample()
    return latents * vae.config.scaling_factor


def encode_source_condition_latents(vae, images):
    images = images * 2.0 - 1.0
    return vae.encode(images).latent_dist.mode()


def encode_prompts(tokenizer, text_encoder, prompts, device):
    text_inputs = tokenizer(
        list(prompts),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    return text_encoder(input_ids)[0]


def prepare_unet_input(noisy_latents, source_latents):
    return torch.cat([noisy_latents, source_latents], dim=1)


def mask_to_bbox(mask):
    positions = torch.nonzero(mask > 0.5, as_tuple=False)
    if positions.numel() == 0:
        return None
    y1 = int(positions[:, 0].min().item())
    y2 = int(positions[:, 0].max().item()) + 1
    x1 = int(positions[:, 1].min().item())
    x2 = int(positions[:, 1].max().item()) + 1
    return x1, y1, x2, y2


def random_shift_box(box, image_width, image_height, min_center_distance=64):
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0 or box_w >= image_width or box_h >= image_height:
        return box

    old_cx = (x1 + x2) / 2
    old_cy = (y1 + y2) / 2
    max_x = image_width - box_w
    max_y = image_height - box_h
    best_box = box
    best_distance = -1

    for _ in range(20):
        new_x1 = int(torch.randint(0, max_x + 1, (1,)).item())
        new_y1 = int(torch.randint(0, max_y + 1, (1,)).item())
        new_cx = new_x1 + box_w / 2
        new_cy = new_y1 + box_h / 2
        distance = ((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2) ** 0.5
        candidate = (new_x1, new_y1, new_x1 + box_w, new_y1 + box_h)
        if distance > best_distance:
            best_box = candidate
            best_distance = distance
        if distance >= min_center_distance:
            return candidate

    return best_box


def place_box(box, image_width, image_height, placement, margin=16):
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return box

    max_x = max(0, image_width - box_w)
    max_y = max(0, image_height - box_h)
    margin_x = min(max(0, margin), max_x)
    margin_y = min(max(0, margin), max_y)

    if placement == "original":
        new_x1, new_y1 = x1, y1
    elif placement == "top-left":
        new_x1, new_y1 = margin_x, margin_y
    elif placement == "top-right":
        new_x1, new_y1 = max(0, image_width - box_w - margin), margin_y
    elif placement == "bottom-left":
        new_x1, new_y1 = margin_x, max(0, image_height - box_h - margin)
    elif placement == "bottom-right":
        new_x1 = max(0, image_width - box_w - margin)
        new_y1 = max(0, image_height - box_h - margin)
    elif placement == "center":
        new_x1 = int(round((image_width - box_w) / 2))
        new_y1 = int(round((image_height - box_h) / 2))
    else:
        raise ValueError(f"Unknown bbox placement: {placement}")

    new_x1 = max(0, min(max_x, int(new_x1)))
    new_y1 = max(0, min(max_y, int(new_y1)))
    return new_x1, new_y1, new_x1 + box_w, new_y1 + box_h


def choose_new_box(box, image_width, image_height, placement, margin):
    if placement == "random":
        return random_shift_box(box, image_width, image_height)
    return place_box(box, image_width, image_height, placement, margin)


def make_shifted_bbox_batch(
    source_img,
    target_img,
    object_mask,
    shift_prob,
    placement="random",
    margin=16,
):
    if shift_prob <= 0:
        return target_img, object_mask

    batch_size, _, image_height, image_width = source_img.shape
    shifted_target = target_img.clone()
    shifted_mask = object_mask.clone()

    for idx in range(batch_size):
        if torch.rand((), device=source_img.device).item() > shift_prob:
            continue

        mask = object_mask[idx, 0]
        box = mask_to_bbox(mask)
        if box is None:
            continue

        x1, y1, x2, y2 = box
        new_x1, new_y1, new_x2, new_y2 = choose_new_box(
            box,
            image_width,
            image_height,
            placement,
            margin,
        )

        crop_mask = object_mask[idx : idx + 1, :, y1:y2, x1:x2]
        crop_object = target_img[idx : idx + 1, :, y1:y2, x1:x2]
        pseudo_target = source_img[idx : idx + 1].clone()
        pseudo_mask = torch.zeros_like(object_mask[idx : idx + 1])

        paste_region = pseudo_target[:, :, new_y1:new_y2, new_x1:new_x2]
        pseudo_target[:, :, new_y1:new_y2, new_x1:new_x2] = (
            crop_object * crop_mask + paste_region * (1.0 - crop_mask)
        )
        pseudo_mask[:, :, new_y1:new_y2, new_x1:new_x2] = crop_mask

        shifted_target[idx : idx + 1] = pseudo_target
        shifted_mask[idx : idx + 1] = pseudo_mask

    return shifted_target, shifted_mask


def bbox_to_mask_like(mask, box):
    bbox_mask = torch.zeros_like(mask)
    if box is None:
        return bbox_mask
    x1, y1, x2, y2 = box
    if x2 > x1 and y2 > y1:
        bbox_mask[..., y1:y2, x1:x2] = 1.0
    return bbox_mask


def expand_box(box, image_width, image_height, padding):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image_width, x2 + padding),
        min(image_height, y2 + padding),
    )


def make_bbox_mask_batch(object_mask, padding=0):
    batch_size, _, image_height, image_width = object_mask.shape
    bbox_mask = torch.zeros_like(object_mask)
    for idx in range(batch_size):
        box = mask_to_bbox(object_mask[idx, 0])
        box = expand_box(box, image_width, image_height, padding)
        bbox_mask[idx : idx + 1] = bbox_to_mask_like(object_mask[idx : idx + 1], box)
    return bbox_mask


def prepare_control_condition(object_mask, mode, outer_padding):
    if mode == "object-mask":
        return object_mask, object_mask

    inner_bbox = make_bbox_mask_batch(object_mask, padding=0)
    if mode == "bbox":
        return inner_bbox, inner_bbox

    if mode == "inner-outer":
        outer_bbox = make_bbox_mask_batch(object_mask, padding=outer_padding)
        return torch.cat([inner_bbox, outer_bbox], dim=1), inner_bbox

    raise ValueError(f"Unknown control conditioning mode: {mode}")


def make_latent_mask(mask, latent_shape, dtype):
    return F.interpolate(mask, size=latent_shape[-2:], mode="nearest").to(dtype=dtype)


def weighted_denoising_loss(noise_pred, noise, bbox_mask, bbox_loss_weight):
    per_pixel_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
    if bbox_loss_weight <= 1:
        return per_pixel_loss.mean()

    latent_bbox_mask = make_latent_mask(bbox_mask, noise_pred.shape, per_pixel_loss.dtype)
    weight = 1.0 + (bbox_loss_weight - 1.0) * latent_bbox_mask
    return (per_pixel_loss * weight).sum() / weight.sum().clamp_min(1.0)


def save_checkpoint(controlnet, output_dir, tag):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    controlnet.save_pretrained(output_dir / f"strategy_c_{tag}_controlnet")


def open_metrics_csv(metrics_csv):
    if not metrics_csv:
        return None, None

    metrics_path = Path(metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        metrics_file,
        fieldnames=["step", "epoch", "loss", "metric", "learning_rate"],
    )
    writer.writeheader()
    return metrics_file, writer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="paint-by-inpaint/add-base")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--bbox-loss-weight", type=float, default=8.0)
    parser.add_argument("--bbox-shift-prob", type=float, default=0.5)
    parser.add_argument(
        "--bbox-placement",
        default="random",
        choices=[
            "random",
            "original",
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
            "center",
        ],
    )
    parser.add_argument("--bbox-placement-margin", type=int, default=16)
    parser.add_argument(
        "--control-conditioning-mode",
        default="object-mask",
        choices=["object-mask", "bbox", "inner-outer"],
    )
    parser.add_argument("--outer-bbox-padding", type=int, default=24)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--metrics-csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        args.device = "cpu"

    dtype = torch.float16 if args.fp16 and args.device == "cuda" else torch.float32

    dataset = PIPEBBoxDataset(split="train", image_size=args.image_size)
    if args.max_train_samples:
        dataset = Subset(dataset, range(min(args.max_train_samples, len(dataset))))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        collate_fn=pipe_bbox_collate,
    )

    conditioning_channels = 2 if args.control_conditioning_mode == "inner-outer" else 1
    pipe, controlnet = load_strategy_c_pipeline(
        model_name=args.model_name,
        device=args.device,
        dtype=dtype,
        conditioning_channels=conditioning_channels,
    )

    print(f"Frozen U-Net parameters: {count_parameters(pipe.unet):,}")
    print(f"ControlNet parameters: {count_parameters(controlnet):,}")
    print(
        "Trainable ControlNet parameters: "
        f"{count_parameters(controlnet, trainable_only=True):,}"
    )

    optimizer = torch.optim.AdamW(iter_trainable_parameters(controlnet), lr=args.lr)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    steps_per_epoch = len(dataloader)
    print(f"Training samples: {len(dataset):,}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total optimization steps: {steps_per_epoch * args.epochs:,}")

    metrics_file, metrics_writer = open_metrics_csv(args.metrics_csv)
    try:
        for epoch in range(args.epochs):
            progress = tqdm(total=len(dataloader), desc=f"Strategy C Epoch {epoch+1}/{args.epochs}")
            for batch in dataloader:
                source_img = batch["source_img"].to(args.device, dtype=dtype)
                target_img = batch["target_img"].to(args.device, dtype=dtype)
                object_mask = batch["object_mask"].to(args.device, dtype=dtype)

                target_img, object_condition_mask = make_shifted_bbox_batch(
                    source_img,
                    target_img,
                    object_mask,
                    args.bbox_shift_prob,
                    args.bbox_placement,
                    args.bbox_placement_margin,
                )
                controlnet_cond, loss_mask = prepare_control_condition(
                    object_condition_mask,
                    args.control_conditioning_mode,
                    args.outer_bbox_padding,
                )

                with torch.no_grad():
                    target_latents = encode_target_latents(pipe.vae, target_img)
                    source_latents = encode_source_condition_latents(pipe.vae, source_img)
                    noise = torch.randn_like(target_latents)
                    timesteps = torch.randint(
                        0,
                        pipe.scheduler.config.num_train_timesteps,
                        (target_latents.shape[0],),
                        device=args.device,
                    ).long()
                    noisy_latents = pipe.scheduler.add_noise(target_latents, noise, timesteps)
                    encoder_hidden_states = encode_prompts(
                        pipe.tokenizer,
                        pipe.text_encoder,
                        batch["instruction"],
                        args.device,
                    )

                model_input = prepare_unet_input(noisy_latents, source_latents)
                down_residuals, mid_residual = controlnet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
                    conditioning_scale=args.controlnet_conditioning_scale,
                    return_dict=False,
                )
                noise_pred = pipe.unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_residuals,
                    mid_block_additional_residual=mid_residual,
                ).sample

                raw_loss = weighted_denoising_loss(
                    noise_pred,
                    noise,
                    loss_mask,
                    args.bbox_loss_weight,
                )
                if not torch.isfinite(raw_loss):
                    raise FloatingPointError("Training loss became NaN/Inf.")

                loss = raw_loss / args.gradient_accumulation_steps
                loss.backward()

                if (global_step + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            list(iter_trainable_parameters(controlnet)),
                            args.max_grad_norm,
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                raw_loss_value = raw_loss.item()
                if metrics_writer is not None:
                    metrics_writer.writerow(
                        {
                            "step": global_step,
                            "epoch": epoch + 1,
                            "loss": raw_loss_value,
                            "metric": "",
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )
                    metrics_file.flush()

                progress.update(1)
                progress.set_postfix(loss=f"{raw_loss_value:.4f}")

            progress.close()
            if args.save_every and (epoch + 1) % args.save_every == 0:
                tag = f"epoch_{epoch+1:04d}_step_{global_step:06d}"
                save_checkpoint(controlnet, args.output_dir, tag)
    finally:
        if metrics_file is not None:
            metrics_file.close()

    final_tag = f"final_epoch_{args.epochs:04d}_step_{global_step:06d}"
    save_checkpoint(controlnet, args.output_dir, final_tag)
    print(f"saved checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
