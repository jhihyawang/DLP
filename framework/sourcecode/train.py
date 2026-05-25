import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sourcecode.model.modelA import (
    count_parameters,
    iter_trainable_parameters,
    load_strategy_a_pipeline,
)
from sourcecode.pipe_bbox_dataset import PIPEBBoxDataset, pipe_bbox_collate


DATA_DISK = Path("/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def")
DEFAULT_OUTPUT_DIR = DATA_DISK / "strategy_a_checkpoints"


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


def apply_conditioning_dropout(prompts, source_latents, bbox_mask, dropout_prob):
    if dropout_prob <= 0:
        return prompts, source_latents, bbox_mask

    prompts = list(prompts)
    batch_size = source_latents.shape[0]
    random_p = torch.rand(batch_size, device=source_latents.device)

    prompt_drop_mask = random_p < 2 * dropout_prob
    image_drop_mask = (random_p >= dropout_prob) & (random_p < 3 * dropout_prob)

    for idx, should_drop in enumerate(prompt_drop_mask.tolist()):
        if should_drop:
            prompts[idx] = ""

    keep_image = (~image_drop_mask).float().view(batch_size, 1, 1, 1)
    source_latents = source_latents * keep_image.to(dtype=source_latents.dtype)
    bbox_mask = bbox_mask * keep_image.to(dtype=bbox_mask.dtype)
    return prompts, source_latents, bbox_mask


def prepare_unet_input(noisy_latents, source_latents, bbox_mask):
    bbox_latent = F.interpolate(
        bbox_mask,
        size=noisy_latents.shape[-2:],
        mode="nearest",
    ).to(dtype=noisy_latents.dtype)
    return torch.cat([noisy_latents, source_latents, bbox_latent], dim=1)


def make_latent_bbox_mask(bbox_mask, latent_shape, dtype):
    return F.interpolate(
        bbox_mask,
        size=latent_shape[-2:],
        mode="nearest",
    ).to(dtype=dtype)


def weighted_denoising_loss(noise_pred, noise, bbox_mask, bbox_loss_weight):
    per_pixel_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
    if bbox_loss_weight <= 1:
        return per_pixel_loss.mean()

    latent_bbox_mask = make_latent_bbox_mask(
        bbox_mask,
        noise_pred.shape,
        per_pixel_loss.dtype,
    )
    weight = 1.0 + (bbox_loss_weight - 1.0) * latent_bbox_mask
    return (per_pixel_loss * weight).sum() / weight.sum().clamp_min(1.0)


def save_checkpoint(pipe, output_dir, tag):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "conv_in": pipe.unet.conv_in.state_dict(),
            "unet_in_channels": pipe.unet.config.in_channels,
        },
        output_dir / f"strategy_a_{tag}_conv_in.pt",
    )
    pipe.unet.save_attn_procs(output_dir / f"strategy_a_{tag}_lora")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="paint-by-inpaint/add-base")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--conditioning-dropout-prob", type=float, default=0.05)
    parser.add_argument("--bbox-loss-weight", type=float, default=8.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
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

    pipe = load_strategy_a_pipeline(
        model_name=args.model_name,
        device=args.device,
        dtype=dtype,
        train_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    print(f"U-Net parameters: {count_parameters(pipe.unet):,}")
    print(f"Trainable U-Net parameters: {count_parameters(pipe.unet, trainable_only=True):,}")

    optimizer = torch.optim.AdamW(iter_trainable_parameters(pipe), lr=args.lr)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    steps_per_epoch = len(dataloader)
    print(f"Training samples: {len(dataset):,}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total optimization steps: {steps_per_epoch * args.epochs:,}")

    for epoch in range(args.epochs):
        progress = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in dataloader:
            source_img = batch["source_img"].to(args.device, dtype=dtype)
            target_img = batch["target_img"].to(args.device, dtype=dtype)
            bbox_mask = batch["bbox_mask"].to(args.device, dtype=dtype)

            with torch.no_grad():
                target_latents = encode_target_latents(pipe.vae, target_img)
                source_latents = encode_source_condition_latents(pipe.vae, source_img)
                prompts, source_latents, bbox_mask = apply_conditioning_dropout(
                    batch["instruction"],
                    source_latents,
                    bbox_mask,
                    args.conditioning_dropout_prob,
                )
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
                    prompts,
                    args.device,
                )

            model_input = prepare_unet_input(noisy_latents, source_latents, bbox_mask)
            noise_pred = pipe.unet(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            loss = weighted_denoising_loss(
                noise_pred,
                noise,
                bbox_mask,
                args.bbox_loss_weight,
            )
            if not torch.isfinite(loss):
                print(
                    "Non-finite loss detected. "
                    f"noise_pred finite={torch.isfinite(noise_pred).all().item()} "
                    f"model_input finite={torch.isfinite(model_input).all().item()} "
                    f"target_latents finite={torch.isfinite(target_latents).all().item()}"
                )
                raise FloatingPointError("Training loss became NaN/Inf.")

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        list(iter_trainable_parameters(pipe)),
                        args.max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}")

        progress.close()
        if args.save_every and (epoch + 1) % args.save_every == 0:
            tag = f"epoch_{epoch+1:04d}_step_{global_step:06d}"
            save_checkpoint(pipe, args.output_dir, tag)

    final_tag = f"final_epoch_{args.epochs:04d}_step_{global_step:06d}"
    save_checkpoint(pipe, args.output_dir, final_tag)
    print(f"saved checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()
