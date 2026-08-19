#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


def canonical_prediction(value: Any) -> str:
    text = str(value or "").strip()
    response = re.fullmatch(r"<response>\s*(.*?)\s*</response>", text, flags=re.I | re.S)
    if response:
        text = response.group(1).strip()
    boxed = re.fullmatch(r"\\boxed\{(.*?)\}", text, flags=re.S)
    if boxed:
        text = boxed.group(1).strip()
    text = re.sub(r"^final\s*:\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip().lower()


def merge_candidate_with_visual(
    candidate: dict[str, Any],
    visual: dict[str, Any],
) -> dict[str, Any]:
    candidate_prediction = candidate.get("prediction", "")
    visual_prediction = visual.get("prediction", "")
    disagree = canonical_prediction(candidate_prediction) != canonical_prediction(visual_prediction)
    merged = copy.deepcopy(visual if disagree else candidate)
    merged["v5_visual_fallback"] = disagree
    merged["v5_candidate_prediction"] = candidate_prediction
    merged["v5_visual_reference_prediction"] = visual_prediction
    merged["v5_candidate_pruned_reasoning_tokens_est"] = candidate.get(
        "summary_kl_pruned_reasoning_tokens_est", 0
    )
    merged["v5_candidate_reasoning_tokens_before_prune_est"] = candidate.get(
        "reasoning_tokens_before_prune_est", 0
    )
    merged["v5_stability_policy"] = "fallback_to_visual_trace_on_answer_disagreement"
    return merged


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def pruning_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    before = [int(row.get("reasoning_tokens_before_prune_est", 0)) for row in rows]
    pruned = [int(row.get("summary_kl_pruned_reasoning_tokens_est", 0)) for row in rows]
    ratios = [min(1.0, max(0.0, p / b)) if b > 0 else 0.0 for p, b in zip(pruned, before)]
    nonempty = [ratio for ratio, b in zip(ratios, before) if b > 0]
    return {
        "total": len(rows),
        "accuracy": sum(float(row.get("score", 0.0)) for row in rows) / len(rows) if rows else 0.0,
        "visual_fallback_count": sum(bool(row.get("v5_visual_fallback")) for row in rows),
        "visual_fallback_rate": sum(bool(row.get("v5_visual_fallback")) for row in rows) / len(rows) if rows else 0.0,
        "pruned_reasoning_fraction_micro_est": sum(pruned) / sum(before) if sum(before) else 0.0,
        "pruned_reasoning_fraction_macro_all_est": sum(ratios) / len(ratios) if ratios else 0.0,
        "pruned_reasoning_fraction_macro_nonempty_est": sum(nonempty) / len(nonempty) if nonempty else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-jsonl", action="append", required=True)
    parser.add_argument("--visual-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    candidates = read_jsonl([Path(path) for path in args.candidate_jsonl])
    visuals = read_jsonl([Path(args.visual_jsonl)])
    candidate_by_idx = {str(row["idx"]): row for row in candidates}
    visual_by_idx = {str(row["idx"]): row for row in visuals}
    if len(candidate_by_idx) != len(candidates):
        raise ValueError("Candidate JSONL inputs contain duplicate idx values.")
    missing = sorted(set(candidate_by_idx) - set(visual_by_idx))
    if missing:
        raise ValueError(f"Visual reference is missing {len(missing)} candidate idx values.")

    merged = [
        merge_candidate_with_visual(candidate, visual_by_idx[str(candidate["idx"])])
        for candidate in candidates
    ]
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = pruning_summary(merged)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
