import os
from pathlib import Path

import torch
from diffusers import DDPMScheduler, StableDiffusionInstructPix2PixPipeline
from peft import LoraConfig


DATA_DISK = Path("/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def")
MODEL_CACHE = Path(os.environ.get("PIPE_MODEL_CACHE_DIR", DATA_DISK / "huggingface")).expanduser()
DEFAULT_MODEL_NAME = "paint-by-inpaint/add-base"


def freeze_module(module):
    module.requires_grad_(False)
    module.eval()


def add_unet_lora(unet, rank=4, alpha=4, dropout=0.0):
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=dropout,
    )
    unet.add_adapter(lora_config)
    return unet


def iter_trainable_parameters(pipe):
    return (param for param in pipe.unet.parameters() if param.requires_grad)


def count_parameters(module, trainable_only=False):
    params = module.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def load_strategy_b_pipeline(
    model_name=DEFAULT_MODEL_NAME,
    device="cuda",
    dtype=torch.float16,
    train_lora=True,
    lora_rank=4,
    lora_alpha=4,
):
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        safety_checker=None,
        cache_dir=str(MODEL_CACHE / "hub"),
    )
    pipe.scheduler = DDPMScheduler.from_pretrained(
        model_name,
        subfolder="scheduler",
        cache_dir=str(MODEL_CACHE / "hub"),
    )

    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    pipe.unet.requires_grad_(False)
    pipe.unet.train()

    if train_lora:
        add_unet_lora(pipe.unet, rank=lora_rank, alpha=lora_alpha)

    pipe.to(device)
    return pipe
