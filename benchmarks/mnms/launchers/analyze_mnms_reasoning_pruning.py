#!/usr/bin/env python3
"""Compute MNMS reasoning-only pruning statistics with the served model tokenizer."""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def post_json(url: str, payload: dict[str, Any], retries: int = 4) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY') or 'EMPTY'}",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"tokenization failed after {retries} attempts: {last_error!r}")


def reasoning_pair(step: dict[str, Any]) -> tuple[str, str]:
    # The *_before_* fields are set only when a method rewrites the segment.
    # Prefer the earliest immutable copy so repeated summary events count once.
    original = step.get("reasoning_before_step_policy")
    if original is None:
        original = step.get("reasoning_before_prune")
    if original is None:
        original = step.get("reasoning") or ""
    retained = step.get("reasoning") or ""
    return str(original), str(retained)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--per-sample-jsonl", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.predictions))
    samples: list[dict[str, Any]] = []
    unique_texts: set[str] = set()
    for row in rows:
        segments = []
        for step in row.get("steps") or []:
            action = step.get("action")
            if not isinstance(action, dict) or action.get("name") == "summarize_the_task":
                continue
            original, retained = reasoning_pair(step)
            unique_texts.add(original)
            unique_texts.add(retained)
            segments.append(
                {
                    "step": step.get("step"),
                    "action_id": action.get("id"),
                    "original_reasoning": original,
                    "final_retained_reasoning": retained,
                }
            )
        samples.append({"id": row.get("id"), "segments": segments})

    root_url = args.base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3]
    cache: dict[str, int] = {"": 0}
    for text in sorted(unique_texts):
        if not text or text in cache:
            continue
        tokenized = post_json(
            root_url + "/tokenize",
            {"model": args.model, "prompt": text, "add_special_tokens": False},
        )
        tokens = tokenized.get("tokens")
        if tokens is None:
            tokens = tokenized.get("input_ids") or []
        cache[text] = len(tokens)

    original_total = 0
    retained_total = 0
    segment_total = 0
    output_path = Path(args.per_sample_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for sample in samples:
            sample_original = 0
            sample_retained = 0
            segment_records = []
            for segment in sample["segments"]:
                original_count = cache[segment.pop("original_reasoning")]
                retained_count = cache[segment.pop("final_retained_reasoning")]
                removed = original_count - retained_count
                segment_records.append(
                    {
                        **segment,
                        "original_token_count": original_count,
                        "final_retained_token_count": retained_count,
                        "net_tokens_removed": removed,
                    }
                )
                sample_original += original_count
                sample_retained += retained_count
                segment_total += 1
            original_total += sample_original
            retained_total += sample_retained
            output.write(
                json.dumps(
                    {
                        "id": sample["id"],
                        "segments": segment_records,
                        "original_token_count": sample_original,
                        "final_retained_token_count": sample_retained,
                        "net_tokens_removed": sample_original - sample_retained,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    net_removed = original_total - retained_total
    ratio = None if original_total == 0 else 100.0 * net_removed / original_total
    metrics = {
        "method": args.method,
        "token_count_scope": "assistant_reasoning_before_each_persisted_real_action_only",
        "tokenizer_model": args.model,
        "num_samples": len(rows),
        "num_reasoning_segments": segment_total,
        "original_token_count": original_total,
        "final_retained_token_count": retained_total,
        "net_tokens_removed": net_removed,
        "benchmark_pruning_ratio": None if ratio is None else round(ratio, 6),
        "pruning_ratio_defined": ratio is not None,
        "excludes": [
            "action_json",
            "verifier_feedback",
            "tool_response",
            "image_tokens",
            "summary_text",
            "one_shot_v5_hint",
        ],
        "multiple_summary_rule": "each segment uses its earliest original reasoning and final persisted representation exactly once",
    }
    Path(args.output_json).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
