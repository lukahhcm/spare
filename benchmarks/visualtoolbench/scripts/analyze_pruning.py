#!/usr/bin/env python3
"""Reasoning-side lifecycle token accounting for bundled VTB predictions."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("vtb_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for pattern in (
        str(path / "predictions.jsonl"),
        str(path / "shards/shard_*/predictions.jsonl"),
    ):
        for filename in glob.glob(pattern):
            for line in Path(filename).read_text().splitlines():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def generated_segments(runner, row: dict) -> list[tuple[str, str]]:
    segments = []
    for turn in row.get("raw_responses") or []:
        for response in turn if isinstance(turn, list) else [turn]:
            try:
                choice = response["choices"][0]
                text = runner.completion_text(choice)
                calls = (
                    runner.completion_tool_calls(choice)
                    or runner.parse_text_tool_calls(text)
                )
            except Exception:
                continue
            reasoning = runner.strip_text_tool_calls(text) if calls else text
            action = runner.extract_action_block(text)
            for call in calls:
                name = runner.normalize_tool_name(
                    (call.get("function") or {}).get("name")
                )
                if name != runner.SUMMARY_TOOL_NAME:
                    segments.append((reasoning, action))
    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=["tool-baseline", "delete-reasoning", "visual-evidence", "v5"],
    )
    args = parser.parse_args()
    runner = load_runner(Path(args.runner))
    rows = load_rows(Path(args.predictions))

    original = 0
    final = 0
    seen = set()
    changed_segments = 0
    for sample_id, row in rows.items():
        segments = generated_segments(runner, row)
        original += sum(runner.rough_text_tokens(reasoning) for reasoning, _ in segments)
        if args.method == "tool-baseline":
            final += sum(runner.rough_text_tokens(reasoning) for reasoning, _ in segments)
        elif args.method == "visual-evidence":
            for reasoning, action in segments:
                replacement, _ = runner.prune_segment_content(
                    {"reasoning": reasoning, "action_block": action}, "v3"
                )
                final += max(
                    0,
                    runner.rough_text_tokens(replacement)
                    - runner.rough_text_tokens(action),
                )
        elif args.method == "v5":
            final += sum(runner.rough_text_tokens(reasoning) for reasoning, _ in segments)
            for event in row.get("summary_kl_events") or []:
                if not isinstance(event, dict):
                    continue
                for item in event.get("applied_pruned_messages") or []:
                    key = (sample_id, item.get("segment_id"))
                    if item.get("applied") and key not in seen:
                        seen.add(key)
                        final -= int(item.get("saved_rough_text_tokens") or 0)
                        changed_segments += 1

    removed = original - final
    report = {
        "method": args.method,
        "num_examples": len(rows),
        "original_reasoning_tokens": original,
        "final_reasoning_tokens": final,
        "net_tokens_removed": removed,
        "pruning_ratio_percent": 100 * removed / original if original else None,
        "changed_segments": changed_segments,
        "tokenizer": "runner rough_text_tokens: ceil(characters/4)",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
