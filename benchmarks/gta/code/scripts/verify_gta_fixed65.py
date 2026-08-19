#!/usr/bin/env python3
"""Fail closed if the frozen GTA 65-item split has changed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "./results/gta/"
            "fixed65_20260728_manifest.json"
        ),
    )
    args = parser.parse_args()
    frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = Path(frozen["dataset_root"])

    observed_ids: list[str] = []
    for shard in frozen["shards"]:
        shard_dir = root / f"shard_{shard['shard']:02d}"
        dataset_path = shard_dir / "dataset.json"
        source_manifest_path = shard_dir / "manifest.json"
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        ids = list(dataset)
        assert ids == shard["ids"], (
            f"shard {shard['shard']} IDs changed: {ids} != {shard['ids']}"
        )
        assert len(ids) == shard["count"]
        assert sha256(dataset_path) == shard["dataset_sha256"]
        assert sha256(source_manifest_path) == shard["manifest_sha256"]
        observed_ids.extend(ids)

    assert len(observed_ids) == frozen["total_samples"] == 65
    assert len(set(observed_ids)) == 65
    print(
        f"verified {frozen['name']}: "
        f"{len(observed_ids)} unique samples across {len(frozen['shards'])} shards"
    )


if __name__ == "__main__":
    main()
