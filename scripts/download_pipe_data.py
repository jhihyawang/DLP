#!/usr/bin/env python
import argparse
import json
import os
from datetime import datetime
from pathlib import Path


REPOS = {
    "PIPE": "paint-by-inpaint/PIPE",
    "PIPE_Masks": "paint-by-inpaint/PIPE_Masks",
}


def repo_metadata(repo_id):
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(repo_id, files_metadata=True)
    parquet_files = [
        {
            "path": sibling.rfilename,
            "size": sibling.size,
        }
        for sibling in info.siblings
        if sibling.rfilename.endswith(".parquet")
    ]
    return {
        "repo_id": repo_id,
        "sha": info.sha,
        "last_modified": str(info.last_modified),
        "downloads": info.downloads,
        "likes": info.likes,
        "num_parquet_files": len(parquet_files),
        "parquet_size_bytes": sum(item["size"] or 0 for item in parquet_files),
        "first_parquet_files": parquet_files[:8],
    }


def configure_cache(cache_dir):
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PIPE_HF_CACHE_DIR"] = str(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir / "home")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")
    return cache_dir


def download_split(repo_id, split, cache_dir):
    from datasets import load_dataset

    print(f"\n=== Loading {repo_id} split={split} ===", flush=True)
    dataset = load_dataset(
        repo_id,
        split=split,
        cache_dir=str(cache_dir),
    )
    print(f"{repo_id} {split} rows: {len(dataset):,}", flush=True)
    sample = dataset[0]
    summary = {
        "repo_id": repo_id,
        "split": split,
        "rows": len(dataset),
        "keys": list(sample.keys()),
    }
    for key, value in sample.items():
        if hasattr(value, "size") and hasattr(value, "mode"):
            summary[key] = {
                "type": type(value).__name__,
                "size": list(value.size),
                "mode": value.mode,
            }
        elif isinstance(value, str):
            summary[key] = value[:200]
        else:
            summary[key] = str(value)[:200]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default="data/hf_cache",
        help="Hugging Face cache directory for PIPE and PIPE_Masks.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "train"],
        choices=["train", "test"],
        help="Splits to download. Default downloads test first, then train.",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        default=list(REPOS),
        choices=list(REPOS),
        help="Datasets to download.",
    )
    args = parser.parse_args()

    cache_dir = configure_cache(args.cache_dir)
    print(f"cache_dir: {cache_dir}", flush=True)
    print(f"HF_HOME: {os.environ['HF_HOME']}", flush=True)
    print(f"HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}", flush=True)
    print(f"HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}", flush=True)
    print(f"HF_XET_CACHE: {os.environ['HF_XET_CACHE']}", flush=True)

    summary = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "cache_dir": str(cache_dir),
        "repos": {},
        "splits": [],
    }

    for repo_name in args.repos:
        repo_id = REPOS[repo_name]
        print(f"\n=== Metadata {repo_id} ===", flush=True)
        metadata = repo_metadata(repo_id)
        print(json.dumps(metadata, indent=2), flush=True)
        summary["repos"][repo_name] = metadata

    for split in args.splits:
        for repo_name in args.repos:
            repo_id = REPOS[repo_name]
            split_summary = download_split(repo_id, split, cache_dir)
            summary["splits"].append(split_summary)

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary_path = cache_dir / "download_pipe_data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary: {summary_path}", flush=True)
    print("download complete", flush=True)


if __name__ == "__main__":
    main()
