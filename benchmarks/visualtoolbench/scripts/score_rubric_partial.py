#!/usr/bin/env python3
"""Portable fixed-rubric partial-credit scorer for the bundled VTB-238 set."""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import re
import time
import urllib.request
from pathlib import Path


SYSTEM = """Judge each prediction against the supplied fixed benchmark rubric
items. Do not create, split, merge, remove, or reinterpret rubric items.

For every method, return one boolean per rubric item in the given order:
- true only if that rubric item is semantically satisfied;
- numbers must be bound to the correct object/entity;
- equivalent units, synonyms, and harmless rounding are accepted;
- incidental numbers in examples, coordinates, labels, or intermediate work do
  not count;
- an explicit contradiction makes the corresponding item false;
- failure on one item does not invalidate independently correct items;
- if an item requires recognizing insufficient information, inventing an answer
  is false.

Return exactly one JSON object:
{
  "methods": {
    "<method>": {
      "rubric_correct": [true, false, ...],
      "brief_reason": "short"
    }
  }
}
Include every supplied method exactly once. Every boolean list must have exactly
the same length as rubric_items."""


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--method must be NAME=PATH")
    return name, Path(path)


def load_predictions(path: Path, allowed: set[str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    patterns = [
        str(path / "predictions.jsonl"),
        str(path / "shards/shard_*/predictions.jsonl"),
    ]
    for pattern in patterns:
        for filename in glob.glob(pattern):
            for line in Path(filename).read_text().splitlines():
                row = json.loads(line)
                sample_id = str(row["id"])
                if sample_id in allowed:
                    rows[sample_id] = str((row.get("predictions") or [""])[0])
    return rows


def rubric_items(raw) -> list[dict]:
    obj = json.loads(raw) if isinstance(raw, str) else raw
    return [
        {
            "rubric_id": rubric_id,
            "description": item["description"],
            "weight": item.get("weight", 1),
        }
        for rubric_id, item in obj.items()
        if item.get("critical") == "yes" and item.get("final_answer") == "yes"
    ]


def crop(text: str, limit: int = 3800) -> str:
    if len(text) <= limit:
        return text
    return text[:1000] + "\n[...middle omitted...]\n" + text[-2800:]


def parse_response(text: str, methods: set[str], count: int) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start = text.find("{")
    if start < 0:
        raise ValueError("judge returned no JSON object")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    got = set((obj.get("methods") or {}).keys())
    if got != methods:
        raise ValueError(f"method mismatch: expected {methods}, got {got}")
    for method in methods:
        flags = obj["methods"][method].get("rubric_correct")
        if (
            not isinstance(flags, list)
            or len(flags) != count
            or not all(isinstance(flag, bool) for flag in flags)
        ):
            raise ValueError(f"invalid rubric flags for {method}")
    return obj


def judge(record: dict, base_url: str, api_key: str, model: str, timeout: float) -> dict:
    items = rubric_items(record["rubric"])
    prompt = {
        "id": record["id"],
        "question": record["question"],
        "ground_truth": record["ground_truth"],
        "rubric_items": items,
        "predictions": {
            method: crop(prediction)
            for method, prediction in record["predictions"].items()
        },
    }
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 1300,
        },
        ensure_ascii=False,
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    error = None
    for attempt in range(3):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=timeout) as response:
                raw = json.loads(response.read())
            audit = parse_response(
                raw["choices"][0]["message"]["content"],
                set(record["predictions"]),
                len(items),
            )
            return {
                "id": record["id"],
                "ok": True,
                "rubric_items": items,
                "audit": audit,
            }
        except Exception as exc:
            error = repr(exc)
            time.sleep(1 + attempt)
    return {"id": record["id"], "ok": False, "error": error}


def summarize(rows: dict[str, dict], methods: list[str]) -> dict:
    summary = {}
    for method in methods:
        scores = []
        weighted = []
        strict = 0
        for row in rows.values():
            flags = row["audit"]["methods"][method]["rubric_correct"]
            weights = [float(item.get("weight", 1)) for item in row["rubric_items"]]
            scores.append(sum(flags) / len(flags))
            weighted.append(
                sum(weight * flag for weight, flag in zip(weights, flags)) / sum(weights)
            )
            strict += int(all(flags))
        n = len(scores)
        summary[method] = {
            "n": n,
            "partial_macro": sum(scores) / n,
            "weighted_partial_macro": sum(weighted) / n,
            "strict_all_correct": strict / n,
            "strict_count": strict,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", required=True)
    parser.add_argument(
        "--eligible-manifest",
        default="",
        help="Optional JSON eligible_ids manifest used to score an exact pilot subset.",
    )
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DASHSCOPE_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    anchors = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in Path(args.anchors).read_text().splitlines()
            if line.strip()
        )
    }
    if args.eligible_manifest:
        eligible = {
            str(sample_id)
            for sample_id in json.loads(
                Path(args.eligible_manifest).read_text()
            ).get("eligible_ids", [])
        }
        anchors = {
            sample_id: row
            for sample_id, row in anchors.items()
            if sample_id in eligible
        }
    allowed = set(anchors)
    predictions = {
        name: load_predictions(path, allowed)
        for name, path in args.method
    }
    missing = {
        name: sorted(allowed - set(rows))
        for name, rows in predictions.items()
        if allowed - set(rows)
    }
    if missing:
        raise SystemExit(
            "Every method must cover all bundled IDs. Missing counts: "
            + json.dumps({name: len(ids) for name, ids in missing.items()})
        )

    records = []
    for sample_id, anchor in anchors.items():
        records.append(
            {
                **anchor,
                "predictions": {
                    name: predictions[name][sample_id]
                    for name, _ in args.method
                },
            }
        )

    out = Path(args.out)
    completed: dict[str, dict] = {}
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            if row.get("ok"):
                completed[row["id"]] = row
    pending = [record for record in records if record["id"] not in completed]
    print(json.dumps({"pending": len(pending), "resume": len(completed)}), flush=True)
    with out.open("a") as handle, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(
                judge,
                record,
                args.base_url,
                args.api_key,
                args.model,
                args.timeout,
            ): record
            for record in pending
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 20 == 0 or not result.get("ok"):
                print(
                    json.dumps(
                        {"done": index, "total": len(pending), "ok": result.get("ok")}
                    ),
                    flush=True,
                )

    final_rows = dict(completed)
    for line in out.read_text().splitlines():
        row = json.loads(line)
        if row.get("ok"):
            final_rows[row["id"]] = row
    if len(final_rows) != len(anchors):
        raise SystemExit(f"Only {len(final_rows)}/{len(anchors)} samples scored successfully.")
    report = {
        "judge_model": args.model,
        "judge_base_url": args.base_url,
        "num_examples": len(final_rows),
        "methods": summarize(final_rows, [name for name, _ in args.method]),
    }
    Path(args.summary_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
