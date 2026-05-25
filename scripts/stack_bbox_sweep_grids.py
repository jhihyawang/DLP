#!/usr/bin/env python
import argparse
from pathlib import Path

from PIL import Image, ImageDraw


STRATEGIES = ("a", "b", "c")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-tag", default="final_epoch_0010_step_002560")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/bbox_corner_compare")
    parser.add_argument("--strategy-suffix", default="bbox_corner_sweep")
    parser.add_argument("--output-prefix", default="bbox_corner_compare")
    return parser.parse_args()


def find_grid(results_dir, strategy, strategy_suffix, checkpoint_tag, index, seed):
    return (
        results_dir
        / f"strategy_{strategy}_{strategy_suffix}"
        / f"strategy_{strategy}_{checkpoint_tag}_test_{index}_seed{seed}_grid.png"
    )


def add_row_label(image, strategy):
    label_width = 118
    canvas = Image.new("RGB", (image.width + label_width, image.height), "white")
    canvas.paste(image, (label_width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, image.height // 2 - 8), f"Strategy {strategy.upper()}", fill=(0, 0, 0))
    return canvas


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for strategy in STRATEGIES:
        grid_path = find_grid(
            results_dir,
            strategy,
            args.strategy_suffix,
            args.checkpoint_tag,
            args.index,
            args.seed,
        )
        if not grid_path.exists():
            raise FileNotFoundError(grid_path)
        with Image.open(grid_path) as grid:
            rows.append(add_row_label(grid.convert("RGB"), strategy))

    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    comparison = Image.new("RGB", (width, height), "white")
    cursor_y = 0
    for row in rows:
        comparison.paste(row, (0, cursor_y))
        cursor_y += row.height

    output_path = output_dir / f"{args.output_prefix}_test_{args.index}_seed{args.seed}.png"
    comparison.save(output_path)
    print(f"comparison: {output_path}")


if __name__ == "__main__":
    main()
