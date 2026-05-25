import argparse
import os
from pathlib import Path

import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
)
from PIL import Image


DATA_DISK = Path("/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def")
DATASET_CACHE = DATA_DISK / "hf_cache"
MODEL_CACHE = DATA_DISK / "huggingface"
OUTPUT_DIR = DATA_DISK / "paint_by_inpaint_outputs"

MODELS = {
    "add-base": "paint-by-inpaint/add-base",
    "add-finetuned": "paint-by-inpaint/add-finetuned-mb",
    "general-base": "paint-by-inpaint/general-base",
    "general-finetuned": "paint-by-inpaint/general-finetuned-mb",
}


def configure_huggingface_cache():
    if not DATA_DISK.is_mount():
        raise RuntimeError(f"Data disk is not mounted at {DATA_DISK}")

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DATASET_CACHE))


def load_dataset_image(split, index):
    from datasets import load_dataset

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    dataset = load_dataset("paint-by-inpaint/PIPE", cache_dir=str(DATASET_CACHE))
    sample = dataset[split][index]
    return sample["source_img"], sample["Instruction_VLM-LLM"]


def load_input_image(args):
    if args.image:
        if not args.prompt:
            raise ValueError("--prompt is required when --image is provided.")
        image = Image.open(args.image).convert("RGB")
        return image, args.prompt

    image, dataset_prompt = load_dataset_image(args.split, args.index)
    return image.convert("RGB"), args.prompt or dataset_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="add-base",
        choices=sorted(MODELS),
        help="Which Paint-by-Inpaint pretrained model to use.",
    )
    parser.add_argument("--prompt", default=None, help="Edit instruction.")
    parser.add_argument("--image", default=None, help="Optional local input image path.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    configure_huggingface_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    image, prompt = load_input_image(args)
    image = image.resize((512, 512))

    model_name = MODELS[args.model]
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        safety_checker=None,
        cache_dir=str(MODEL_CACHE / "hub"),
    ).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = pipe(
        prompt,
        image=image,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=args.steps,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.split}_{args.index}_seed{args.seed}"
    input_path = output_dir / f"{stem}_input.png"
    output_path = output_dir / f"{stem}_output.png"
    prompt_path = output_dir / f"{stem}_prompt.txt"

    image.save(input_path)
    output.save(output_path)
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    print(f"model: {model_name}")
    print(f"device: {device}")
    print(f"prompt: {prompt}")
    print(f"input: {input_path}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
