import argparse
import ast
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image


class OptionalEvaluators:
    def __init__(self, args):
        self.device = args.device
        self.clip_model = None
        self.clip_processor = None
        self.clip_image_processor = None
        self.clip_tokenizer = None
        self.lpips_model = None

        if args.enable_clip:
            try:
                from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel, CLIPProcessor
            except ImportError as exc:
                raise SystemExit(
                    "CLIP evaluation requires transformers. "
                    "Activate the project conda environment or install transformers."
                ) from exc

            try:
                self.clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
            except OSError:
                self.clip_image_processor = CLIPImageProcessor.from_pretrained(args.clip_model)
                self.clip_tokenizer = AutoTokenizer.from_pretrained(args.clip_model)
            self.clip_model = CLIPModel.from_pretrained(args.clip_model).to(self.device)
            self.clip_model.eval()

        if args.enable_lpips:
            try:
                import lpips
            except ImportError as exc:
                raise SystemExit(
                    "LPIPS evaluation requires the lpips package. "
                    "Install it with `pip install lpips` or update the conda environment."
                ) from exc

            self.lpips_model = lpips.LPIPS(
                net=args.lpips_net,
                pnet_rand=args.lpips_random_backbone,
            ).to(self.device)
            self.lpips_model.eval()

    @torch.no_grad()
    def clip_similarity(self, image, text):
        if self.clip_model is None:
            return float("nan")
        if self.clip_processor is not None:
            inputs = self.clip_processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        else:
            image_inputs = self.clip_image_processor(images=image, return_tensors="pt")
            text_inputs = self.clip_tokenizer([text], return_tensors="pt", padding=True)
            inputs = {**image_inputs, **text_inputs}
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.clip_model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return (image_features * text_features).sum(dim=-1).item()

    @torch.no_grad()
    def lpips_distance(self, image_a, image_b):
        if self.lpips_model is None:
            return float("nan")
        tensor_a = pil_to_lpips_tensor(image_a).to(self.device)
        tensor_b = pil_to_lpips_tensor(image_b).to(self.device)
        return self.lpips_model(tensor_a, tensor_b).item()


def parse_info_file(path):
    info = {}
    for line in path.read_text().splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        info[key.strip()] = value.strip()
    if "bbox_xyxy" in info:
        info["bbox_xyxy"] = tuple(float(v) for v in ast.literal_eval(info["bbox_xyxy"]))
    if "index" in info:
        info["index"] = int(info["index"])
    return info


def image_to_tensor(path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return data.view(height, width, 3).permute(2, 0, 1).float() / 255.0


def pil_to_lpips_tensor(image):
    width, height = image.size
    data = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    tensor = data.view(height, width, 3).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0) * 2.0 - 1.0


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


def resize_bbox_to_fit(bbox, max_width, max_height, image_size):
    x1, y1, x2, y2 = clamp_bbox(bbox, image_size)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    scale = min(1.0, max_width / box_w, max_height / box_h)
    resized_w = max(1, int(round(box_w * scale)))
    resized_h = max(1, int(round(box_h * scale)))
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return move_bbox_to_center((0, 0, resized_w, resized_h), center_x, center_y, image_size)


def make_bbox_sweep(original_bbox, image_size):
    width, height = image_size
    x1, y1, x2, y2 = [int(round(v)) for v in original_bbox]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    return {
        "top_left": move_bbox_to_center(original_bbox, box_w / 2, box_h / 2, image_size),
        "top_right": move_bbox_to_center(original_bbox, width - box_w / 2, box_h / 2, image_size),
        "center": move_bbox_to_center(original_bbox, width * 0.5, height * 0.5, image_size),
        "bottom_left": move_bbox_to_center(original_bbox, box_w / 2, height - box_h / 2, image_size),
        "bottom_right": move_bbox_to_center(
            original_bbox,
            width - box_w / 2,
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


def make_small_corner_bbox_sweep(original_bbox, image_size):
    return make_bbox_sweep(resize_bbox_to_fit(original_bbox, 224, 112, image_size), image_size)


def make_vertical_bbox_sweep(original_bbox, image_size):
    width, height = image_size
    x1, y1, x2, y2 = clamp_bbox(original_bbox, image_size)
    center_x = (x1 + x2) / 2
    box_h = max(1, y2 - y1)
    return {
        "top": move_bbox_to_center(original_bbox, center_x, box_h / 2, image_size),
        "middle": move_bbox_to_center(original_bbox, center_x, height * 0.5, image_size),
        "bottom": move_bbox_to_center(original_bbox, center_x, height - box_h / 2, image_size),
        "original": clamp_bbox(original_bbox, image_size),
    }


def bbox_for_name(original_bbox, image_size, name):
    sweeps = {}
    sweeps.update(make_bbox_sweep(original_bbox, image_size))
    sweeps.update(make_small_corner_bbox_sweep(original_bbox, image_size))
    sweeps.update(make_vertical_bbox_sweep(original_bbox, image_size))
    sweeps.update(make_ablation_bbox_sweep(original_bbox, image_size))
    return sweeps.get(name, clamp_bbox(original_bbox, image_size))


def bbox_mask(bbox, shape):
    _, height, width = shape
    x1, y1, x2, y2 = clamp_bbox(bbox, (width, height))
    mask = torch.zeros(height, width, dtype=torch.bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def expand_bbox(bbox, padding, image_size):
    x1, y1, x2, y2 = clamp_bbox(bbox, image_size)
    return clamp_bbox((x1 - padding, y1 - padding, x2 + padding, y2 + padding), image_size)


def mean_or_nan(values):
    values = [v for v in values if v == v]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def crop_image(image, bbox):
    x1, y1, x2, y2 = clamp_bbox(bbox, image.size)
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def mse(a, b, mask):
    if mask.sum().item() == 0:
        return float("nan")
    diff = (a - b).pow(2).mean(dim=0)
    return diff[mask].mean().item()


def l1(a, b, mask):
    if mask.sum().item() == 0:
        return float("nan")
    diff = (a - b).abs().mean(dim=0)
    return diff[mask].mean().item()


def changed_region_metrics(source, output, bbox, threshold):
    _, height, width = source.shape
    requested = bbox_mask(bbox, source.shape)
    diff = (output - source).abs().mean(dim=0)
    changed = diff > threshold

    changed_count = changed.sum().item()
    bbox_count = requested.sum().item()
    intersection = (changed & requested).sum().item()
    union = (changed | requested).sum().item()

    if changed_count == 0:
        changed_center_x = float("nan")
        changed_center_y = float("nan")
        center_distance = float("nan")
        center_distance_norm = float("nan")
    else:
        ys, xs = torch.nonzero(changed, as_tuple=True)
        changed_center_x = xs.float().mean().item()
        changed_center_y = ys.float().mean().item()
        x1, y1, x2, y2 = clamp_bbox(bbox, (width, height))
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        center_distance = math.hypot(changed_center_x - bbox_center_x, changed_center_y - bbox_center_y)
        center_distance_norm = center_distance / math.hypot(width, height)

    return {
        "changed_pixels": changed_count,
        "bbox_pixels": bbox_count,
        "changed_bbox_iou": intersection / union if union else float("nan"),
        "changed_inside_bbox_ratio": intersection / changed_count if changed_count else float("nan"),
        "bbox_coverage_by_change": intersection / bbox_count if bbox_count else float("nan"),
        "changed_center_x": changed_center_x,
        "changed_center_y": changed_center_y,
        "changed_to_bbox_center_distance_px": center_distance,
        "changed_to_bbox_center_distance_norm": center_distance_norm,
    }


def output_name_and_strategy(path):
    stem = path.stem
    if stem.endswith("_strategy_a"):
        return "original", "strategy_a"
    for strategy in ("strategy_b", "strategy_c"):
        marker = f"_bbox_"
        suffix = f"_{strategy}"
        if marker in stem and stem.endswith(suffix):
            name = stem.split(marker, 1)[1][: -len(suffix)]
            return name, strategy
    return None, None


def find_outputs(info_path):
    directory = info_path.parent
    stem = info_path.stem.removesuffix("_info")
    candidates = []
    candidates.extend(directory.glob(f"{stem}_strategy_a.png"))
    candidates.extend(directory.glob(f"{stem}_bbox_*_strategy_b.png"))
    candidates.extend(directory.glob(f"{stem}_bbox_*_strategy_c.png"))
    return sorted(candidates)


def evaluate_output(info_path, output_path, threshold, outer_padding, evaluators):
    info = parse_info_file(info_path)
    source_path = info_path.with_name(info_path.name.replace("_info.txt", "_source.png"))
    target_path = info_path.with_name(info_path.name.replace("_info.txt", "_target.png"))
    source_pil = Image.open(source_path).convert("RGB")
    target_pil = Image.open(target_path).convert("RGB")
    output_pil = Image.open(output_path).convert("RGB")
    source = image_to_tensor(source_path)
    target = image_to_tensor(target_path)
    output = image_to_tensor(output_path)

    if source.shape != output.shape:
        raise ValueError(f"Shape mismatch: {source_path} {source.shape} vs {output_path} {output.shape}")

    _, height, width = source.shape
    bbox_name, output_strategy = output_name_and_strategy(output_path)
    bbox = bbox_for_name(info["bbox_xyxy"], (width, height), bbox_name)
    inner_mask = bbox_mask(bbox, source.shape)
    outer_bbox = expand_bbox(bbox, outer_padding, (width, height))
    outer_mask = bbox_mask(outer_bbox, source.shape)
    outside_mask = ~inner_mask
    outside_outer_mask = ~outer_mask

    row = {
        "result_dir": str(info_path.parent),
        "output_file": output_path.name,
        "strategy_dir": info_path.parent.name,
        "output_strategy": output_strategy,
        "bbox_name": bbox_name,
        "split": info.get("split", ""),
        "index": info.get("index", ""),
        "prompt": info.get("prompt", ""),
        "bbox_xyxy": list(bbox),
        "threshold": threshold,
        "inside_output_target_l1": l1(output, target, inner_mask),
        "inside_output_target_mse": mse(output, target, inner_mask),
        "outside_output_source_l1": l1(output, source, outside_mask),
        "outside_output_source_mse": mse(output, source, outside_mask),
        "outside_outer_output_source_l1": l1(output, source, outside_outer_mask),
        "outside_outer_output_source_mse": mse(output, source, outside_outer_mask),
        "whole_output_source_l1": l1(output, source, torch.ones(height, width, dtype=torch.bool)),
        "whole_output_target_l1": l1(output, target, torch.ones(height, width, dtype=torch.bool)),
        "clip_output_prompt_similarity": evaluators.clip_similarity(output_pil, info.get("prompt", "")),
        "clip_target_prompt_similarity": evaluators.clip_similarity(target_pil, info.get("prompt", "")),
        "lpips_output_target_whole": evaluators.lpips_distance(output_pil, target_pil),
        "lpips_output_source_whole": evaluators.lpips_distance(output_pil, source_pil),
    }

    output_crop = crop_image(output_pil, bbox)
    target_crop = crop_image(target_pil, bbox)
    source_crop = crop_image(source_pil, bbox)
    if output_crop is not None and target_crop is not None:
        row["lpips_output_target_inside_bbox"] = evaluators.lpips_distance(output_crop, target_crop)
    else:
        row["lpips_output_target_inside_bbox"] = float("nan")
    if output_crop is not None and source_crop is not None:
        row["lpips_output_source_inside_bbox"] = evaluators.lpips_distance(output_crop, source_crop)
    else:
        row["lpips_output_source_inside_bbox"] = float("nan")

    row.update(changed_region_metrics(source, output, bbox, threshold))
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)
    numeric_keys = []
    for key, value in rows[0].items():
        if isinstance(value, (int, float)):
            numeric_keys.append(key)
    for row in rows:
        groups[(row["strategy_dir"], row["bbox_name"])].append(row)

    summaries = []
    for (strategy_dir, bbox_name), group_rows in sorted(groups.items()):
        summary = {
            "strategy_dir": strategy_dir,
            "bbox_name": bbox_name,
            "num_outputs": len(group_rows),
        }
        for key in numeric_keys:
            summary[f"mean_{key}"] = mean_or_nan([row[key] for row in group_rows])
        summaries.append(summary)
    return summaries


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/all_strategies_matched")
    parser.add_argument("--output-csv", default="results/evaluation/per_image_metrics.csv")
    parser.add_argument("--summary-csv", default="results/evaluation/summary_metrics.csv")
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.08,
        help="Mean RGB absolute difference threshold for changed-region proxy.",
    )
    parser.add_argument(
        "--outer-padding",
        type=int,
        default=24,
        help="Padding used for outside-outer background preservation metrics.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enable-clip", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--enable-lpips", action="store_true")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument(
        "--lpips-random-backbone",
        action="store_true",
        help=(
            "Avoid downloading torchvision pretrained backbone weights. "
            "Use only as an offline fallback; standard LPIPS should omit this flag."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    evaluators = OptionalEvaluators(args)
    rows = []
    for info_path in sorted(results_root.rglob("*_info.txt")):
        for output_path in find_outputs(info_path):
            rows.append(
                evaluate_output(
                    info_path,
                    output_path,
                    threshold=args.change_threshold,
                    outer_padding=args.outer_padding,
                    evaluators=evaluators,
                )
            )

    if not rows:
        raise SystemExit(f"No evaluable outputs found under {results_root}")

    write_csv(Path(args.output_csv), rows)
    write_csv(Path(args.summary_csv), summarize(rows))
    print(f"wrote per-image metrics: {args.output_csv}")
    print(f"wrote summary metrics: {args.summary_csv}")


if __name__ == "__main__":
    main()
