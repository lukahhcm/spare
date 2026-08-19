#!/usr/bin/env python3
"""Run AdaReasoner benchmark datasets as direct no-tool VQA.

This script intentionally bypasses AdaReasoner's tool orchestration and training
code. It loads the benchmark data, calls an OpenAI-compatible vision model, and
applies lightweight rule-based scoring matching the repository tasks where
possible.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


def _bypass_proxy_for_local_services() -> None:
    """Keep localhost vLLM traffic out of a globally configured HTTP proxy."""
    required = ("127.0.0.1", "localhost", "::1")
    for key in ("no_proxy", "NO_PROXY"):
        current = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        for host in required:
            if host not in current:
                current.append(host)
        os.environ[key] = ",".join(current)


_bypass_proxy_for_local_services()
from datasets import load_dataset
from PIL import Image


DEFAULT_BASE_URL = "http://10.100.20.37:8006/v1"
DEFAULT_MODEL = "qwen3-vl-8b"


TASK_SPECS = {
    "vsp": [("hitsmy/AdaEval-VSP", None, "navigation_test"), ("hitsmy/AdaEval-VSP", None, "verify_test")],
    "vspo": [("hitsmy/AdaEval-VSPO", None, "navigation_test"), ("hitsmy/AdaEval-VSPO", None, "verify_test")],
    "jigsaw_coco": [("hitsmy/AdaEval-Jigsaw-COCO", None, "test")],
    "jigsaw_blink": [("BLINK-Benchmark/BLINK", "Jigsaw", "val")],
    "web_guichat": [("AdaReasoner/AdaEval-web_guichat", None, "validation")],
    "webquest": [("AdaReasoner/AdaEval-webquest", None, "validation")],
    "vstar": [("lmms-lab/vstar-bench", None, "test")],
    "hrbench": [("DreamMr/HR-Bench", "hrbench_version_split", "hrbench_4k")],
}


QUESTION_BLINK = (
    "Given the first image (img_1) with one part missing, can you tell which one "
    "of the second image (img_2) or the third image (img_3) is the missing part? "
    "Imagine which image would be more appropriate to place in the missing spot. "
    "You can also carefully observe and compare the edges of the images.\n\n"
    "Select from the following choices.\n\n"
    "(A) The second image (img_2)\n(B) The third image (img_3)\n"
    "Your final answer should be formatted as \\boxed{Your Choice}."
)


def image_to_data_url(image: Image.Image) -> str:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def normalize_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str) and os.path.exists(value):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)}")


def call_model(base_url: str, model: str, text: str, images: list[Image.Image], max_tokens: int, temperature: float, timeout: int) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": "Question: " + text}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers={"Authorization": "Bearer EMPTY"}, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"].get("content") or ""


def extract_boxed(text: str) -> str | None:
    match = re.search(r"\\boxed\s*\{([^}]+)\}", text, flags=re.I)
    return match.group(1).strip() if match else None


def extract_option(text: str, letters: str = "abcd") -> str | None:
    text_l = (text or "").strip().lower()
    boxed = extract_boxed(text_l)
    if boxed:
        text_l = boxed.lower()
    for pat in [
        rf"\b([{letters}])[.、)）]?\b",
        rf"(?:option|answer)\s*:?\s*([{letters}])[.、)）]?\b",
        rf"(?:the answer is|i choose|i pick)\s*:?\s*([{letters}])[.、)）]?\b",
        rf"\(([{letters}])\)",
    ]:
        match = re.search(pat, text_l, flags=re.I)
        if match:
            return match.group(1).upper()
    if text_l.upper() in list(letters.upper()):
        return text_l.upper()
    return None


def extract_yes_no(text: str) -> str | None:
    text_l = (text or "").lower().strip()
    boxed = extract_boxed(text_l)
    if boxed:
        text_l = boxed.lower()
    if text_l in {"yes", "no", "y", "n"}:
        return {"y": "yes", "n": "no"}.get(text_l, text_l)
    has_yes = bool(re.search(r"\byes\b", text_l))
    has_no = bool(re.search(r"\bno\b", text_l))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return None


def extract_actions(text: str) -> list[str]:
    boxed = extract_boxed(text or "")
    source = boxed if boxed else text or ""
    match = re.search(r"([UDLR](?:\s*,\s*[UDLR])*)", source, flags=re.I)
    if match:
        return [x.strip().upper() for x in match.group(1).split(",") if x.strip().upper() in {"U", "D", "L", "R"}]
    return [c.upper() for c in source if c.upper() in {"U", "D", "L", "R"}]


def parse_map(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Unsupported map type: {type(value)}")


def score_navigation(prediction: str, item: dict[str, Any]) -> tuple[float, str]:
    actions = extract_actions(prediction)
    if not actions:
        return 0.0, "no valid actions"
    grid = parse_map(item["gym_map"])
    pos = None
    goal = None
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == "S":
                pos = [r, c]
            elif val == "G":
                goal = [r, c]
    if pos is None or goal is None:
        return 0.0, "missing start/goal"
    delta = {"L": (0, -1), "R": (0, 1), "U": (-1, 0), "D": (1, 0)}
    for action in actions:
        dr, dc = delta[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            pos = [nr, nc]
        cell = grid[pos[0]][pos[1]]
        if cell == "H":
            return 0.0, "path leads to hole"
        if cell == "G":
            return 1.0, f"path reaches goal in {len(actions)} actions"
    return 0.0, "path does not reach goal"


def levenshtein(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        cur = [i]
        for j, ca in enumerate(a, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(gold: str, pred: str, threshold: float = 0.5) -> float:
    gold = (gold or "").replace('"', "").strip().lower()
    pred = (pred or "").replace('"', "").strip().lower()
    if not pred:
        return 0.0
    if gold == pred:
        return 1.0
    score = 1.0 - levenshtein(pred, gold) / max(len(pred), len(gold), 1)
    return score if score >= threshold else 0.0


def canonical_item(task: str, split: str, raw: dict[str, Any], idx: int) -> dict[str, Any]:
    if task in {"vsp", "vspo"}:
        item = dict(raw)
        item["idx"] = raw.get("idx", f"{task}_{split}_{idx}")
        item["images"] = [normalize_image(raw["image"])]
        return item
    if task == "jigsaw_coco":
        return {
            "idx": raw.get("idx", f"jigsaw_coco_{idx}"),
            "text": raw["question_text"],
            "images": [normalize_image(raw["question_image"])] + [normalize_image(x) for x in raw["choice_images"]],
            "answer": raw["correct_answer"],
            "task_type": "jigsaw",
        }
    if task == "jigsaw_blink":
        choices = [x for x in [raw.get("image_2"), raw.get("image_3"), raw.get("image_4")] if x is not None]
        answer = str(raw["answer"]).strip().strip("()").upper()
        return {
            "idx": raw.get("idx", f"jigsaw_blink_{idx}"),
            "text": QUESTION_BLINK,
            "images": [normalize_image(raw["image_1"])] + [normalize_image(x) for x in choices],
            "answer": answer,
            "choices": raw.get("choices", []),
            "task_type": "jigsaw",
        }
    if task in {"web_guichat", "webquest"}:
        return {
            "idx": str(raw.get("id", f"{task}_{idx}")),
            "text": raw["question"],
            "images": [normalize_image(raw["image"])],
            "answer": raw["answer"],
            "task_type": "anls",
        }
    if task == "vstar":
        return {
            "idx": str(raw.get("question_id", f"vstar_{idx}")),
            "text": raw["text"],
            "images": [normalize_image(raw["image"])],
            "answer": raw["label"],
            "category": raw.get("category", "unknown"),
            "task_type": "option",
        }
    if task == "hrbench":
        options = {k: raw[k] for k in "ABCD" if k in raw and raw[k]}
        options_text = "".join(f"{k}. {v}\n" for k, v in options.items())
        image = Image.open(io.BytesIO(base64.b64decode(raw["image"]))).convert("RGB")
        return {
            "idx": f"hrbench_{idx}",
            "text": raw["question"].strip() + "\n" + options_text + "Answer the option letter directly.",
            "images": [image],
            "answer": raw["answer"],
            "category": raw.get("category", "unknown"),
            "cycle_category": raw.get("cycle_category", "unknown"),
            "task_type": "option",
        }
    raise ValueError(task)


def score_item(task: str, item: dict[str, Any], prediction: str) -> tuple[float, str]:
    if task in {"vsp", "vspo"} and item.get("task_type") == "navigation":
        return score_navigation(prediction, item)
    if task in {"vsp", "vspo"} and item.get("task_type") == "verify":
        pred = extract_yes_no(prediction)
        gold = {"y": "yes", "n": "no"}.get(str(item["answer"]).lower(), str(item["answer"]).lower())
        return (1.0, "correct") if pred == gold else (0.0, f"expected {gold}, got {pred}")
    if task in {"jigsaw_coco", "jigsaw_blink"}:
        pred = extract_option(prediction, "abc")
        return (1.0, "correct") if pred == str(item["answer"]).upper() else (0.0, f"expected {item['answer']}, got {pred}")
    if task in {"vstar", "hrbench"}:
        pred = extract_option(prediction, "abcd")
        return (1.0, "correct") if pred == str(item["answer"]).upper() else (0.0, f"expected {item['answer']}, got {pred}")
    if task in {"web_guichat", "webquest"}:
        return anls(str(item["answer"]), prediction), "anls"
    return 0.0, "unknown scorer"


def load_items(task: str, limit: int | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for repo, config, split in TASK_SPECS[task]:
        ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        for i, raw in enumerate(ds):
            if limit is not None and len(items) >= limit:
                return items
            item = canonical_item(task, split, raw, i)
            item["split"] = split
            items.append(item)
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="vspo,vsp,jigsaw_coco,jigsaw_blink,web_guichat,webquest,vstar,hrbench")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--out-dir", default="scripts/logs/no_tool_qwen3_vl_8b")
    parser.add_argument("--limit", type=int, default=None, help="Per-task item cap for smoke tests.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    for task in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        raw_path = out_dir / f"{task}.jsonl"
        done: dict[str, Any] = {}
        if args.resume and raw_path.exists():
            with raw_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    done[row["idx"]] = row

        items = load_items(task, args.limit)
        rows: list[dict[str, Any]] = list(done.values())
        correct_by_cat: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            correct_by_cat[row.get("category", row.get("task_type", "all"))].append(float(row["score"]))

        with raw_path.open("a", encoding="utf-8") as f:
            for n, item in enumerate(items, 1):
                if item["idx"] in done:
                    continue
                start = time.time()
                try:
                    prediction = call_model(args.base_url, args.model, item["text"], item["images"], args.max_tokens, args.temperature, args.timeout)
                    score, message = score_item(task, item, prediction)
                    error = None
                except Exception as exc:
                    prediction = ""
                    score = 0.0
                    message = "exception"
                    error = repr(exc)
                row = {
                    "idx": item["idx"],
                    "task": task,
                    "split": item.get("split"),
                    "task_type": item.get("task_type"),
                    "category": item.get("category", item.get("task_type", "all")),
                    "question": item["text"],
                    "gold": item.get("answer"),
                    "prediction": prediction,
                    "score": score,
                    "message": message,
                    "error": error,
                    "latency_sec": round(time.time() - start, 3),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                rows.append(row)
                correct_by_cat[row["category"]].append(float(score))
                print(f"[{task}] {len(rows)}/{len(items)} score={score} idx={item['idx']}", flush=True)

        scores = [float(x["score"]) for x in rows]
        summary[task] = {
            "total": len(rows),
            "score": sum(scores) / len(scores) if scores else 0.0,
            "by_category": {k: sum(v) / len(v) for k, v in sorted(correct_by_cat.items()) if v},
            "raw_path": str(raw_path),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({task: summary[task]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
