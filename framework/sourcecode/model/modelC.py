import os
from pathlib import Path

import torch
from diffusers import ControlNetModel, DDPMScheduler, StableDiffusionInstructPix2PixPipeline


DATA_DISK = Path("/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def")
MODEL_CACHE = Path(os.environ.get("PIPE_MODEL_CACHE_DIR", DATA_DISK / "huggingface")).expanduser()
DEFAULT_MODEL_NAME = "paint-by-inpaint/add-base"


STRATEGY_C_DESCRIPTION = (
    "Feature-level conditioning with a ControlNet-style bbox branch. "
    "The frozen editing U-Net receives multi-level residual features extracted "
    "from a trainable bbox-conditioned ControlNet."
)


def freeze_module(module):
    module.requires_grad_(False)
    module.eval()


def iter_trainable_parameters(controlnet):
    return (param for param in controlnet.parameters() if param.requires_grad)


def count_parameters(module, trainable_only=False):
    params = module.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def create_bbox_controlnet(unet, dtype=torch.float16, conditioning_channels=1):
    controlnet = ControlNetModel.from_unet(
        unet,
        conditioning_channels=conditioning_channels,
        load_weights_from_unet=True,
    )
    controlnet.to(device=unet.device, dtype=dtype)
    return controlnet


def load_strategy_c_pipeline(
    model_name=DEFAULT_MODEL_NAME,
    device="cuda",
    dtype=torch.float16,
    conditioning_channels=1,
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
    freeze_module(pipe.unet)

    pipe.to(device)
    controlnet = create_bbox_controlnet(
        pipe.unet,
        dtype=dtype,
        conditioning_channels=conditioning_channels,
    )
    controlnet.train()
    return pipe, controlnet
