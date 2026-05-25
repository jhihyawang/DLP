import argparse
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageDraw


DATA_DISK = Path("/media/zia/88d6caf3-71f2-49cd-b054-48a1711c5def")
HF_CACHE = DATA_DISK / "hf_cache"
DEFAULT_OUTPUT_DIR = DATA_DISK / "pipe_visualizations"


def configure_cache():
    if not DATA_DISK.is_mount():
        raise RuntimeError(f"Data disk is not mounted at {DATA_DISK}")

    os.environ.setdefault("HF_HOME", str(HF_CACHE / "home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))


def load_pipe(split):
    return load_dataset("paint-by-inpaint/PIPE", split=split, cache_dir=str(HF_CACHE))


def load_masks(split):
    return load_dataset("paint-by-inpaint/PIPE_Masks", split=split, cache_dir=str(HF_CACHE))


def build_mask_index(mask_dataset):
    return {
        (sample["img_id"], sample["ann_id"]): i
        for i, sample in enumerate(mask_dataset)
    }


def make_overlay(image, mask, color=(255, 40, 40), alpha=110):
    image = image.convert("RGBA")
    mask = mask.convert("L")

    color_layer = Image.new("RGBA", image.size, (*color, 0))
    color_layer.putalpha(mask.point(lambda p: alpha if p > 0 else 0))

    overlay = Image.alpha_composite(image, color_layer)

    bbox = mask.getbbox()
    if bbox:
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(bbox, outline=(255, 0, 0, 255), width=4)

    return overlay.convert("RGB"), bbox


def make_grid(source, target, mask, overlay):
    panels = [
        source.convert("RGB"),
        target.convert("RGB"),
        mask.convert("RGB"),
        overlay.convert("RGB"),
    ]
    width, height = panels[0].size
    grid = Image.new("RGB", (width * len(panels), height), "white")
    for i, panel in enumerate(panels):
        grid.paste(panel.resize((width, height)), (i * width, 0))
    return grid


def export_sample(pipe_dataset, mask_dataset, mask_index, idx, output_dir):
    sample = pipe_dataset[idx]
    key = (sample["img_id"], sample["ann_id"])

    if key not in mask_index:
        raise KeyError(f"No mask found for img_id={key[0]} ann_id={key[1]}")

    mask_sample = mask_dataset[mask_index[key]]
    source = sample["source_img"].convert("RGB")
    target = sample["target_img"].convert("RGB")
    mask = mask_sample["mask"].convert("L")
    overlay, bbox = make_overlay(target, mask)
    grid = make_grid(source, target, mask, overlay)

    stem = f"{idx:06d}_img{sample['img_id']}_ann{sample['ann_id']}"
    source.save(output_dir / f"{stem}_source.png")
    target.save(output_dir / f"{stem}_target.png")
    mask.save(output_dir / f"{stem}_mask.png")
    overlay.save(output_dir / f"{stem}_overlay.png")
    grid.save(output_dir / f"{stem}_grid.png")

    instruction = sample["Instruction_VLM-LLM"]
    info = [
        f"index: {idx}",
        f"img_id: {sample['img_id']}",
        f"ann_id: {sample['ann_id']}",
        f"bbox_xyxy_from_mask: {bbox}",
        f"instruction: {instruction}",
        f"instruction_class: {sample['Instruction_Class']}",
        f"object_location: {sample['object_location']}",
    ]
    (output_dir / f"{stem}_info.txt").write_text("\n".join(info) + "\n", encoding="utf-8")

    return stem, instruction, bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    configure_cache()

    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe_dataset = load_pipe(args.split)
    mask_dataset = load_masks(args.split)
    mask_index = build_mask_index(mask_dataset)

    end = min(args.start + args.count, len(pipe_dataset))
    for idx in range(args.start, end):
        stem, instruction, bbox = export_sample(
            pipe_dataset,
            mask_dataset,
            mask_index,
            idx,
            output_dir,
        )
        print(f"{stem}: bbox={bbox} instruction={instruction}")

    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
