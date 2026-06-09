import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sourcecode.model.modelC import DEFAULT_MODEL_NAME, MODEL_CACHE, create_bbox_controlnet
from sourcecode.pipe_bbox_dataset import create_bbox_dataset


DATA_DISK = Path("/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def")
DATASET_CACHE = Path(os.environ.get("PIPE_HF_CACHE_DIR", DATA_DISK / "hf_cache")).expanduser()
DEFAULT_CHECKPOINT_DIR = DATA_DISK / "strategy_c_checkpoints"
DEFAULT_OUTPUT_DIR = DATA_DISK / "strategy_c_inference"


def configure_cache():
    if "PIPE_HF_CACHE_DIR" not in os.environ and not DATA_DISK.is_mount():
        raise RuntimeError(f"Data disk is not mounted at {DATA_DISK}")

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DATASET_CACHE))


def load_pipeline(model_name, device, dtype):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        safety_checker=None,
        cache_dir=str(MODEL_CACHE / "hub"),
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.unet.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_controlnet(pipe, checkpoint_dir, checkpoint_tag, device, dtype, conditioning_channels=1):
    if checkpoint_tag:
        controlnet_dir = Path(checkpoint_dir) / f"strategy_c_{checkpoint_tag}_controlnet"
        if not controlnet_dir.exists():
            raise FileNotFoundError(controlnet_dir)
        controlnet = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=dtype)
    else:
        controlnet = create_bbox_controlnet(
            pipe.unet,
            dtype=dtype,
            conditioning_channels=conditioning_channels,
        )

    controlnet.to(device=device, dtype=dtype)
    controlnet.eval()
    return controlnet


def encode_image_condition_latents(pipe, image):
    image = image * 2.0 - 1.0
    return pipe.vae.encode(image).latent_dist.mode()


def decode_latents(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return pipe.image_processor.numpy_to_pil(image)[0]


def encode_prompt(pipe, prompt, device, dtype):
    prompt_embeds = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
    )
    return prompt_embeds.to(dtype=dtype)


def tensor_to_pil(image_tensor):
    image = image_tensor.detach().cpu().clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


def clamp_bbox(bbox, image_size):
    width, height = image_size
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    return (x1, y1, x2, y2)


def bbox_to_mask_tensor(bbox, image_size):
    width, height = image_size
    x1, y1, x2, y2 = clamp_bbox(bbox, image_size)
    mask = torch.zeros(1, height, width, dtype=torch.float32)
    if x2 > x1 and y2 > y1:
        mask[:, y1:y2, x1:x2] = 1.0
    return mask


def expand_bbox(bbox, padding, image_size):
    if padding <= 0:
        return clamp_bbox(bbox, image_size)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    return clamp_bbox((x1 - padding, y1 - padding, x2 + padding, y2 + padding), image_size)


def bbox_to_soft_padding_mask_tensor(bbox, padding, image_size):
    if padding <= 0:
        return bbox_to_mask_tensor(bbox, image_size)

    width, height = image_size
    inner = clamp_bbox(bbox, image_size)
    outer = expand_bbox(inner, padding, image_size)
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer
    mask = torch.zeros(1, height, width, dtype=torch.float32)
    if ox2 <= ox1 or oy2 <= oy1:
        return mask

    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    dx = torch.maximum(torch.maximum(x1 - xx, xx - (x2 - 1)), torch.zeros_like(xx))
    dy = torch.maximum(torch.maximum(y1 - yy, yy - (y2 - 1)), torch.zeros_like(yy))
    distance = torch.maximum(dx, dy)
    soft = (1.0 - distance / float(padding)).clamp(0.0, 1.0)
    soft[:, :ox1] = 0.0
    soft[:, ox2:] = 0.0
    soft[:oy1, :] = 0.0
    soft[oy2:, :] = 0.0
    mask[0] = soft
    return mask


def bbox_to_inner_outer_mask_tensor(bbox, outer_padding, image_size):
    inner = bbox_to_mask_tensor(bbox, image_size)
    outer = bbox_to_mask_tensor(expand_bbox(bbox, outer_padding, image_size), image_size)
    return torch.cat([inner, outer], dim=0)


def draw_bbox_overlay(image, bbox, color=(255, 0, 0), alpha=100):
    overlay = image.convert("RGBA")
    x1, y1, x2, y2 = [int(v) for v in bbox]
    layer = Image.new("RGBA", overlay.size, (*color, 0))
    if x2 > x1 and y2 > y1:
        draw = ImageDraw.Draw(layer)
        draw.rectangle((x1, y1, x2, y2), fill=(*color, alpha), outline=(*color, 255), width=4)
    return Image.alpha_composite(overlay, layer).convert("RGB")


def make_grid(images, labels=None):
    width, height = images[0].size
    label_height = 28 if labels else 0
    grid = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(grid)
    for idx, image in enumerate(images):
        if labels:
            draw.text((idx * width + 8, 7), labels[idx], fill=(0, 0, 0))
        grid.paste(image.resize((width, height)), (idx * width, label_height))
    return grid


def move_bbox_to_center(bbox, center_x, center_y, image_size):
    width, height = image_size
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    new_x1 = int(round(center_x - box_w / 2))
    new_y1 = int(round(center_y - box_h / 2))
    new_x1 = max(0, min(width - box_w, new_x1))
    new_y1 = max(0, min(height - box_h, new_y1))
    return (new_x1, new_y1, new_x1 + box_w, new_y1 + box_h)


def make_bbox_sweep(original_bbox, image_size):
    width, height = image_size
    x1, y1, x2, y2 = [int(round(v)) for v in original_bbox]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    return {
        "top_left": move_bbox_to_center(original_bbox, box_w / 2, box_h / 2, image_size),
        "top_right": move_bbox_to_center(
            original_bbox,
            width - box_w / 2,
            box_h / 2,
            image_size,
        ),
        "center": move_bbox_to_center(original_bbox, width * 0.5, height * 0.5, image_size),
        "bottom_left": move_bbox_to_center(
            original_bbox,
            box_w / 2,
            height - box_h / 2,
            image_size,
        ),
        "bottom_right": move_bbox_to_center(
            original_bbox,
            width - box_w / 2,
            height - box_h / 2,
            image_size,
        ),
        "original": clamp_bbox(original_bbox, image_size),
    }


def resize_bbox_to_fit(bbox, max_width, max_height, image_size):
    x1, y1, x2, y2 = clamp_bbox(bbox, image_size)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    scale = min(1.0, max_width / box_w, max_height / box_h)
    resized_w = max(1, int(round(box_w * scale)))
    resized_h = max(1, int(round(box_h * scale)))
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    seed_bbox = (0, 0, resized_w, resized_h)
    return move_bbox_to_center(seed_bbox, center_x, center_y, image_size)


def make_small_corner_bbox_sweep(original_bbox, image_size):
    small_bbox = resize_bbox_to_fit(original_bbox, 224, 112, image_size)
    return make_bbox_sweep(small_bbox, image_size)


def make_vertical_bbox_sweep(original_bbox, image_size):
    width, height = image_size
    x1, y1, x2, y2 = clamp_bbox(original_bbox, image_size)
    center_x = (x1 + x2) / 2
    box_h = max(1, y2 - y1)
    return {
        "top": move_bbox_to_center(original_bbox, center_x, box_h / 2, image_size),
        "middle": move_bbox_to_center(original_bbox, center_x, height * 0.5, image_size),
        "bottom": move_bbox_to_center(
            original_bbox,
            center_x,
            height - box_h / 2,
            image_size,
        ),
        "original": clamp_bbox(original_bbox, image_size),
    }


def make_ablation_bbox_sweep(original_bbox, image_size):
    width, height = image_size
    x1, y1, x2, y2 = clamp_bbox(original_bbox, image_size)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    return {
        "zero": (0, 0, 0, 0),
        "top_left": move_bbox_to_center(original_bbox, box_w / 2, box_h / 2, image_size),
        "center": move_bbox_to_center(original_bbox, width * 0.5, height * 0.5, image_size),
        "bottom_right": move_bbox_to_center(
            original_bbox,
            width - box_w / 2,
            height - box_h / 2,
            image_size,
        ),
        "original": clamp_bbox(original_bbox, image_size),
    }


def select_bbox_sweep(original_bbox, image_size, mode):
    if mode == "ablation":
        return make_ablation_bbox_sweep(original_bbox, image_size)
    if mode == "small-corners":
        return make_small_corner_bbox_sweep(original_bbox, image_size)
    if mode == "vertical":
        return make_vertical_bbox_sweep(original_bbox, image_size)
    return make_bbox_sweep(original_bbox, image_size)


@torch.no_grad()
def generate_strategy_c(
    pipe,
    controlnet,
    source_img,
    bbox_mask,
    prompt,
    steps,
    guidance_scale,
    image_guidance_scale,
    controlnet_conditioning_scale,
    seed,
    device,
    dtype,
):
    source_img = source_img.to(device=device, dtype=dtype)
    bbox_mask = bbox_mask.to(device=device, dtype=dtype)

    image_latents = encode_image_condition_latents(pipe, source_img)
    prompt_embeds = encode_prompt(pipe, prompt, device, dtype)
    image_latents = torch.cat(
        [image_latents, image_latents, torch.zeros_like(image_latents)],
        dim=0,
    )
    controlnet_cond = torch.cat(
        [bbox_mask, bbox_mask, torch.zeros_like(bbox_mask)],
        dim=0,
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        image_latents[:1].shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    pipe.scheduler.set_timesteps(steps, device=device)
    latents = latents * pipe.scheduler.init_noise_sigma
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta=0.0)

    for timestep in pipe.scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 3, dim=0)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)
        model_input = torch.cat([latent_model_input, image_latents], dim=1)

        down_residuals, mid_residual = controlnet(
            model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=controlnet_cond,
            conditioning_scale=controlnet_conditioning_scale,
            return_dict=False,
        )
        noise_pred = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=False,
        )[0]

        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        noise_pred = (
            noise_pred_uncond
            + guidance_scale * (noise_pred_text - noise_pred_image)
            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
        )
        latents = pipe.scheduler.step(
            noise_pred,
            timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
        )[0]

    return decode_latents(pipe, latents)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--checkpoint-tag", default=None)
    parser.add_argument("--dataset-name", default="pipe", choices=["pipe", "magicbrush"])
    parser.add_argument("--magicbrush-dataset", default="osunlp/MagicBrush")
    parser.add_argument("--split", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--control-conditioning-mode",
        default="bbox",
        choices=["bbox", "inner-outer"],
    )
    parser.add_argument("--outer-bbox-padding", type=int, default=24)
    parser.add_argument("--bbox-sweep", action="store_true")
    parser.add_argument(
        "--bbox-padding-sweep",
        nargs="*",
        type=int,
        default=None,
        help="Run original bbox with each listed padding value, e.g. 0 16 32 48.",
    )
    parser.add_argument(
        "--bbox-soft-padding-sweep",
        nargs="*",
        type=int,
        default=None,
        help="Run original bbox with soft falloff padding values, e.g. 0 16 32 48.",
    )
    parser.add_argument(
        "--bbox-sweep-mode",
        default="corners",
        choices=["corners", "small-corners", "vertical", "ablation"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_cache()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        args.device = "cpu"

    dtype = torch.float32 if args.fp32 or args.device == "cpu" else torch.float16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = create_bbox_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        image_size=args.image_size,
        magicbrush_dataset_name=args.magicbrush_dataset,
    )
    split_label = args.split or ("dev" if args.dataset_name == "magicbrush" else "test")
    sample = dataset[args.index]
    prompt = args.prompt or sample["instruction"]
    source_pil = tensor_to_pil(sample["source_img"])
    target_pil = tensor_to_pil(sample["target_img"])
    source_batch = sample["source_img"].unsqueeze(0)

    conditioning_channels = 2 if args.control_conditioning_mode == "inner-outer" else 1
    pipe = load_pipeline(args.model_name, args.device, dtype)
    controlnet = load_controlnet(
        pipe,
        args.checkpoint_dir,
        args.checkpoint_tag,
        args.device,
        dtype,
        conditioning_channels=conditioning_channels,
    )

    baseline_output = None
    if not args.no_baseline:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        baseline_output = pipe(
            prompt,
            image=source_pil,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            num_inference_steps=args.steps,
            num_images_per_prompt=1,
            generator=generator,
        ).images[0]

    use_soft_padding = args.bbox_soft_padding_sweep is not None
    if use_soft_padding:
        boxes = {
            f"soft_pad_{padding:03d}": expand_bbox(
                sample["bbox"].tolist(),
                padding,
                source_pil.size,
            )
            for padding in args.bbox_soft_padding_sweep
        }
        mask_boxes = {
            f"soft_pad_{padding:03d}": sample["bbox"].tolist()
            for padding in args.bbox_soft_padding_sweep
        }
        padding_by_name = {
            f"soft_pad_{padding:03d}": padding
            for padding in args.bbox_soft_padding_sweep
        }
    elif args.bbox_padding_sweep is not None:
        boxes = {
            f"pad_{padding:03d}": expand_bbox(
                sample["bbox"].tolist(),
                padding,
                source_pil.size,
            )
            for padding in args.bbox_padding_sweep
        }
        mask_boxes = boxes
        padding_by_name = {}
    elif args.bbox_sweep:
        boxes = select_bbox_sweep(sample["bbox"].tolist(), source_pil.size, args.bbox_sweep_mode)
        mask_boxes = boxes
        padding_by_name = {}
    else:
        boxes = {"original": sample["bbox"].tolist()}
        mask_boxes = boxes
        padding_by_name = {}

    checkpoint_label = args.checkpoint_tag or "untrained_controlnet"
    stem = (
        f"strategy_c_{checkpoint_label}_{args.dataset_name}_{split_label}_"
        f"{args.index}_seed{args.seed}"
    )
    paths = {
        "source": output_dir / f"{stem}_source.png",
        "target": output_dir / f"{stem}_target.png",
        "info": output_dir / f"{stem}_info.txt",
    }
    source_pil.save(paths["source"])
    target_pil.save(paths["target"])

    grid_images = [source_pil]
    grid_labels = ["source"]
    if baseline_output is not None:
        paths["baseline"] = output_dir / f"{stem}_baseline.png"
        baseline_output.save(paths["baseline"])
        grid_images.append(baseline_output)
        grid_labels.append("baseline")

    for name, bbox in boxes.items():
        if use_soft_padding:
            bbox_mask = bbox_to_soft_padding_mask_tensor(
                mask_boxes[name],
                padding_by_name[name],
                source_pil.size,
            )
        elif args.control_conditioning_mode == "inner-outer":
            bbox_mask = bbox_to_inner_outer_mask_tensor(
                mask_boxes[name],
                args.outer_bbox_padding,
                source_pil.size,
            )
        else:
            bbox_mask = bbox_to_mask_tensor(mask_boxes[name], source_pil.size)
        bbox_batch = bbox_mask.unsqueeze(0)
        output = generate_strategy_c(
            pipe,
            controlnet,
            source_batch,
            bbox_batch,
            prompt,
            args.steps,
            args.guidance_scale,
            args.image_guidance_scale,
            args.controlnet_conditioning_scale,
            args.seed,
            args.device,
            dtype,
        )
        overlay = draw_bbox_overlay(source_pil, bbox)
        overlay_path = output_dir / f"{stem}_bbox_{name}_overlay.png"
        output_path = output_dir / f"{stem}_bbox_{name}_strategy_c.png"
        overlay.save(overlay_path)
        output.save(output_path)
        paths[f"bbox_{name}_overlay"] = overlay_path
        paths[f"bbox_{name}_strategy_c"] = output_path
        grid_images.extend([overlay, output])
        grid_labels.extend([f"{name} bbox", f"{name} out"])

    grid_images.append(target_pil)
    grid_labels.append("target")
    grid = make_grid(grid_images, grid_labels)
    paths["grid"] = output_dir / f"{stem}_grid.png"
    grid.save(paths["grid"])

    info = [
        f"model_name: {args.model_name}",
        f"dataset_name: {args.dataset_name}",
        f"magicbrush_dataset: {args.magicbrush_dataset}",
        f"checkpoint_tag: {args.checkpoint_tag}",
        f"split: {split_label}",
        f"index: {args.index}",
        f"img_id: {sample['img_id']}",
        f"ann_id: {sample['ann_id']}",
        f"bbox_xyxy: {sample['bbox'].tolist()}",
        f"prompt: {prompt}",
        f"seed: {args.seed}",
        f"steps: {args.steps}",
        f"guidance_scale: {args.guidance_scale}",
        f"image_guidance_scale: {args.image_guidance_scale}",
        f"controlnet_conditioning_scale: {args.controlnet_conditioning_scale}",
        f"control_conditioning_mode: {args.control_conditioning_mode}",
        f"outer_bbox_padding: {args.outer_bbox_padding}",
        f"bbox_sweep: {args.bbox_sweep}",
        f"bbox_sweep_mode: {args.bbox_sweep_mode}",
        f"bbox_padding_sweep: {args.bbox_padding_sweep}",
        f"bbox_soft_padding_sweep: {args.bbox_soft_padding_sweep}",
        f"grid: {paths['grid']}",
    ]
    paths["info"].write_text("\n".join(info) + "\n", encoding="utf-8")

    print(f"prompt: {prompt}")
    print(f"grid: {paths['grid']}")
    if baseline_output is not None:
        print(f"baseline: {paths['baseline']}")


if __name__ == "__main__":
    main()
