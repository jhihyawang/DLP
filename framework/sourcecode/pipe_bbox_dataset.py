import os
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor


DATA_DISK = Path("/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def")
HF_CACHE = Path(os.environ.get("PIPE_HF_CACHE_DIR", DATA_DISK / "hf_cache")).expanduser()


def configure_huggingface_cache():
    if "PIPE_HF_CACHE_DIR" not in os.environ and not DATA_DISK.is_mount():
        raise RuntimeError(
            f"Dataset disk is not mounted at {DATA_DISK}. "
            "Mount the 3TB disk first, or set PIPE_HF_CACHE_DIR to another large disk."
        )

    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE / "home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))


def load_pipe_split(split):
    configure_huggingface_cache()
    return load_dataset("paint-by-inpaint/PIPE", split=split, cache_dir=str(HF_CACHE))


def load_pipe_masks_split(split):
    configure_huggingface_cache()
    return load_dataset("paint-by-inpaint/PIPE_Masks", split=split, cache_dir=str(HF_CACHE))


def load_magicbrush_split(split, dataset_name="osunlp/MagicBrush"):
    configure_huggingface_cache()
    return load_dataset(dataset_name, split=split, cache_dir=str(HF_CACHE))


def build_mask_index(mask_dataset):
    # Column access avoids decoding the mask images while building the index.
    return {
        (img_id, ann_id): idx
        for idx, (img_id, ann_id) in enumerate(
            zip(mask_dataset["img_id"], mask_dataset["ann_id"])
        )
    }


def binarize_mask(mask):
    return mask.convert("L").point(lambda pixel: 255 if pixel > 0 else 0)


def mask_to_bbox(mask):
    bbox = binarize_mask(mask).getbbox()
    if bbox is None:
        return (0, 0, 0, 0)
    return bbox


def bbox_to_mask(bbox, size):
    x1, y1, x2, y2 = bbox
    mask = Image.new("L", size, 0)
    if x2 > x1 and y2 > y1:
        mask.paste(255, (x1, y1, x2, y2))
    return mask


def diff_to_mask(source_img, target_img, threshold=16):
    source = source_img.convert("RGB")
    target = target_img.convert("RGB")
    if target.size != source.size:
        target = target.resize(source.size, Image.NEAREST)
    diff = Image.new("L", source.size, 0)
    source_pixels = source.load()
    target_pixels = target.load()
    diff_pixels = diff.load()
    width, height = source.size
    for y in range(height):
        for x in range(width):
            sr, sg, sb = source_pixels[x, y]
            tr, tg, tb = target_pixels[x, y]
            if (abs(sr - tr) + abs(sg - tg) + abs(sb - tb)) / 3 > threshold:
                diff_pixels[x, y] = 255
    return diff


def mask_bbox_area_ratio(mask):
    bbox = mask_to_bbox(mask)
    x1, y1, x2, y2 = bbox
    width, height = mask.size
    if width <= 0 or height <= 0:
        return 0.0
    return max(0, x2 - x1) * max(0, y2 - y1) / float(width * height)


def is_valid_mask(mask, min_bbox_area_ratio=0.0001, max_bbox_area_ratio=0.98):
    ratio = mask_bbox_area_ratio(mask)
    return min_bbox_area_ratio <= ratio <= max_bbox_area_ratio


def choose_magicbrush_object_mask(source_img, target_img, mask_img=None):
    if mask_img is not None:
        object_mask = diff_to_mask(source_img, mask_img)
        if is_valid_mask(object_mask):
            return object_mask
    return diff_to_mask(source_img, target_img)


def choose_instruction(sample):
    candidates = [
        sample.get("Instruction_VLM-LLM", ""),
        sample.get("Instruction_Class", ""),
        sample.get("Instruction_Ref_Dataset", ""),
    ]
    for instruction in candidates:
        if instruction:
            return instruction
    return ""


class PIPEBBoxDataset(Dataset):
    def __init__(
        self,
        split="train",
        pipe_dataset=None,
        mask_dataset=None,
        image_size=512,
        image_transform=to_tensor,
        mask_transform=to_tensor,
    ):
        self.split = split
        self.pipe_dataset = pipe_dataset if pipe_dataset is not None else load_pipe_split(split)
        self.mask_dataset = (
            mask_dataset if mask_dataset is not None else load_pipe_masks_split(split)
        )
        self.mask_index = build_mask_index(self.mask_dataset)
        self.image_size = image_size
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.pipe_dataset)

    def __getitem__(self, idx):
        sample = self.pipe_dataset[idx]
        key = (sample["img_id"], sample["ann_id"])
        if key not in self.mask_index:
            raise KeyError(f"No PIPE_Masks entry for img_id={key[0]} ann_id={key[1]}")

        mask_sample = self.mask_dataset[self.mask_index[key]]
        source_img = sample["source_img"].convert("RGB")
        target_img = sample["target_img"].convert("RGB")
        object_mask = binarize_mask(mask_sample["mask"])

        if self.image_size is not None:
            size = (self.image_size, self.image_size)
            source_img = source_img.resize(size, Image.BICUBIC)
            target_img = target_img.resize(size, Image.BICUBIC)
            object_mask = object_mask.resize(size, Image.NEAREST)

        bbox = mask_to_bbox(object_mask)
        bbox_mask = bbox_to_mask(bbox, source_img.size)

        if self.image_transform is not None:
            source_img = self.image_transform(source_img).contiguous().clone()
            target_img = self.image_transform(target_img).contiguous().clone()

        if self.mask_transform is not None:
            object_mask = self.mask_transform(object_mask).contiguous().clone()
            bbox_mask = self.mask_transform(bbox_mask).contiguous().clone()

        return {
            "source_img": source_img,
            "target_img": target_img,
            "object_mask": object_mask,
            "bbox_mask": bbox_mask,
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "instruction": choose_instruction(sample),
            "instruction_vlm_llm": sample["Instruction_VLM-LLM"],
            "instruction_class": sample["Instruction_Class"],
            "instruction_ref": sample["Instruction_Ref_Dataset"],
            "object_location": sample["object_location"],
            "target_img_dataset": sample["target_img_dataset"],
            "img_id": sample["img_id"],
            "ann_id": sample["ann_id"],
        }


class MagicBrushBBoxDataset(Dataset):
    def __init__(
        self,
        split="dev",
        dataset_name="osunlp/MagicBrush",
        magicbrush_dataset=None,
        image_size=512,
        image_transform=to_tensor,
        mask_transform=to_tensor,
    ):
        self.split = split
        self.dataset_name = dataset_name
        self.dataset = (
            magicbrush_dataset
            if magicbrush_dataset is not None
            else load_magicbrush_split(split, dataset_name)
        )
        self.image_size = image_size
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        source_img = sample["source_img"].convert("RGB")
        target_img = sample["target_img"].convert("RGB")
        object_mask = choose_magicbrush_object_mask(source_img, target_img)

        if self.image_size is not None:
            size = (self.image_size, self.image_size)
            source_img = source_img.resize(size, Image.BICUBIC)
            target_img = target_img.resize(size, Image.BICUBIC)
            object_mask = object_mask.resize(size, Image.NEAREST)

        bbox = mask_to_bbox(object_mask)
        bbox_mask = bbox_to_mask(bbox, source_img.size)

        if self.image_transform is not None:
            source_img = self.image_transform(source_img).contiguous().clone()
            target_img = self.image_transform(target_img).contiguous().clone()

        if self.mask_transform is not None:
            object_mask = self.mask_transform(object_mask).contiguous().clone()
            bbox_mask = self.mask_transform(bbox_mask).contiguous().clone()

        img_id = sample.get("img_id", idx)
        turn_index = sample.get("turn_index", 0)
        return {
            "source_img": source_img,
            "target_img": target_img,
            "object_mask": object_mask,
            "bbox_mask": bbox_mask,
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "instruction": sample.get("instruction", ""),
            "instruction_vlm_llm": sample.get("instruction", ""),
            "instruction_class": "",
            "instruction_ref": "",
            "object_location": "",
            "target_img_dataset": "osunlp/MagicBrush",
            "img_id": img_id,
            "ann_id": turn_index,
            "turn_index": turn_index,
            "dataset_name": self.dataset_name,
        }


def create_bbox_dataset(
    dataset_name="pipe",
    split=None,
    image_size=512,
    magicbrush_dataset_name="osunlp/MagicBrush",
):
    if dataset_name == "pipe":
        return PIPEBBoxDataset(split=split or "test", image_size=image_size)
    if dataset_name == "magicbrush":
        return MagicBrushBBoxDataset(
            split=split or "dev",
            image_size=image_size,
            dataset_name=magicbrush_dataset_name,
        )
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def pipe_bbox_collate(batch):
    tensor_keys = {"source_img", "target_img", "object_mask", "bbox_mask", "bbox"}
    collated = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key in tensor_keys:
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = default_collate(values)
    return collated


def create_bbox_dataloaders(batch_size=4, num_workers=4, pin_memory=True):
    train_dataset = PIPEBBoxDataset(split="train")
    test_dataset = PIPEBBoxDataset(split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pipe_bbox_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pipe_bbox_collate,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    dataset = PIPEBBoxDataset(split="test")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=pipe_bbox_collate,
    )
    batch = next(iter(loader))

    print("source_img:", batch["source_img"].shape)
    print("target_img:", batch["target_img"].shape)
    print("object_mask:", batch["object_mask"].shape)
    print("bbox_mask:", batch["bbox_mask"].shape)
    print("bbox:", batch["bbox"])
    print("instruction:", batch["instruction"][0])
