# https://huggingface.co/datasets/paint-by-inpaint/PIPE?library=datasets

import os
from pathlib import Path

DEFAULT_DATA_DISK = Path("/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def")
DEFAULT_CACHE_DIR = DEFAULT_DATA_DISK / "hf_cache"

cache_dir = Path(os.environ.get("PIPE_HF_CACHE_DIR", DEFAULT_CACHE_DIR)).expanduser()

if "PIPE_HF_CACHE_DIR" not in os.environ and not DEFAULT_DATA_DISK.is_mount():
    raise RuntimeError(
        f"Dataset disk is not mounted at {DEFAULT_DATA_DISK}. "
        "Mount the 3TB disk first, or set PIPE_HF_CACHE_DIR to another large disk."
    )

cache_dir.mkdir(parents=True, exist_ok=True)

# Keep every Hugging Face download/cache artifact on the large disk.
os.environ.setdefault("HF_HOME", str(cache_dir / "home"))
os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))
os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

if (cache_dir / "paint-by-inpaint___pipe").exists() and os.environ.get("PIPE_ONLINE") != "1":
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor


class PIPE_Dataset(Dataset):
    def __init__(self, pipe_dataset, split="train", image_transform=to_tensor):
        self.dataset = pipe_dataset[split]
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        source_img = sample["source_img"]
        target_img = sample["target_img"]

        if self.image_transform is not None:
            source_img = self.image_transform(source_img)
            target_img = self.image_transform(target_img)

        return {
            "source_img": source_img,
            "target_img": target_img,
            "instruction": sample["Instruction_VLM-LLM"],
            "instruction_class": sample["Instruction_Class"],
            "object_location": sample["object_location"],
            "img_id": sample["img_id"],
        }


def load_pipe_dataset():
    return load_dataset(
        "paint-by-inpaint/PIPE",
        cache_dir=str(cache_dir),
    )


def create_dataloaders(batch_size=32, num_workers=4, pipe_dataset=None):
    if pipe_dataset is None:
        pipe_dataset = load_pipe_dataset()

    train_dataset = PIPE_Dataset(pipe_dataset, split="train")
    test_dataset = PIPE_Dataset(pipe_dataset, split="test")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    pipe_dataset = load_pipe_dataset()
    print(pipe_dataset)

    train_dataloader, test_dataloader = create_dataloaders(
        batch_size=2,
        num_workers=0,
        pipe_dataset=pipe_dataset,
    )
    batch = next(iter(train_dataloader))

    print("Train source image batch:", batch["source_img"].shape)
    print("Train target image batch:", batch["target_img"].shape)
    print("First instruction:", batch["instruction"][0])
