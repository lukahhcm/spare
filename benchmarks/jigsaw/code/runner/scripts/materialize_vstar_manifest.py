#!/usr/bin/env python3
"""Materialize a benchmark once so evaluator shards do not each load it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_no_tool_benchmarks import load_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task", default="vstar")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"{args.task}.jsonl"
    items = load_items(args.task, None)
    with manifest.open("w", encoding="utf-8") as f:
        for n, item in enumerate(items):
            image_paths: list[str] = []
            for image_index, image in enumerate(item.pop("images")):
                path = image_dir / f"{n:04d}_{image_index}.jpg"
                image.convert("RGB").save(path, format="JPEG", quality=95)
                image_paths.append(str(path))
            record = {"task": args.task, **item, "image_paths": image_paths}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
