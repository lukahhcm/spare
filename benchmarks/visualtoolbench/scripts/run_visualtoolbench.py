#!/usr/bin/env python3
"""Run VisualToolBench against an OpenAI-compatible VLM endpoint.

The VisualToolBench copy in this workspace is a parquet dataset, not a runnable
benchmark package. This runner streams the parquet by row group, calls the
already-running Qwen3-VL vLLM service, writes resumable predictions, and emits a
lightweight local evaluation against the golden answers.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import csv
import difflib
import hashlib
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from urllib.error import HTTPError
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:
    class _FuzzFallback:
        @staticmethod
        def token_set_ratio(left: str, right: str) -> float:
            left_tokens = " ".join(sorted(set((left or "").split())))
            right_tokens = " ".join(sorted(set((right or "").split())))
            return difflib.SequenceMatcher(None, left_tokens, right_tokens).ratio() * 100.0

    fuzz = _FuzzFallback()

try:
    from visualtoolbench_summary_kl import enabled as summary_kl_enabled
    from visualtoolbench_summary_kl import log_summarization_kl
except Exception:
    def summary_kl_enabled() -> bool:
        return False

    def log_summarization_kl(**_: Any) -> dict[str, Any] | None:
        return None


SUMMARY_TOOL_NAME = "summarize_the_task"


TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate one arithmetic expression. Supports +, -, *, /, %, ** or ^, parentheses, "
                "sqrt, sin, cos, tan, log, exp, abs, min, max, round, floor, ceil, and constants pi/e."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate, e.g. '2 * 3.14 * 5'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": (
                "Run short general-purpose Python code and return stdout/stderr. Use print() for results. "
                "The code runs in the current turn's tool workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to run.", "maxLength": 5000}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_image_processing",
            "description": (
                "Run Python image-processing code with PIL. NumPy, OpenCV, OCR, and display packages are unavailable. "
                "Read source images from source_images/ and save transformed PNG images to processed_images/ "
                "using names like transformed_image_0.png. Use PIL-only operations and print() to describe what was saved. "
                "Do not use pytesseract or other OCR packages; they are not installed. "
                "Do not call Image.show(), matplotlib show(), display(), or external viewers."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python image manipulation code.", "maxLength": 5000}},
                "required": ["code"],
            },
        },
    },
]

SUMMARY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": SUMMARY_TOOL_NAME,
        "description": (
            "Summarize the current visual state. This internal memory/state tool does not inspect or edit images. "
            "Prefer compact verified visual/tool evidence over long natural-language reasoning."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning_so_far": {"type": "string", "description": "Brief non-redundant reasoning so far."},
                "completed_steps": {"type": "array", "items": {"type": "string"}},
                "current_state": {"type": "string"},
                "visual_evidence": {
                    "type": "object",
                    "properties": {
                        "coordinates": {"type": "array", "items": {"type": "string"}},
                        "bboxes": {"type": "array", "items": {"type": "string"}},
                        "crop_ids": {"type": "array", "items": {"type": "string"}},
                        "ocr_text": {"type": "array", "items": {"type": "string"}},
                        "astar_outputs": {"type": "array", "items": {"type": "string"}},
                        "candidate_differences": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "key_observations": {"type": "array", "items": {"type": "string"}},
                "uncertainties": {"type": "array", "items": {"type": "string"}},
                "next_decision": {"type": "string"},
            },
        },
    },
}

TOOL_ALIASES = {
    "python_processing": "python_interpreter",
    "Python": "python_interpreter",
    "python": "python_interpreter",
    "PythonInterpreter": "python_interpreter",
    "PythonImageProcessing": "python_image_processing",
    "Calculator": "calculator",
    "google_search": "web_search",
    "browser-get-page-text": "browser_get_page_text",
}


TEXT_COLUMNS = [
    "id",
    "turncase",
    "num_turns",
    "prompt_category",
    "eval_focus",
    "turn_prompts",
    "turn_golden_answers",
    "turn_tool_trajectories",
    "rubrics_by_turn",
    "images_by_turn",
    "num_images",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.+\-/% ]+", "", text)
    return text.strip()


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for tok in gold_tokens:
        counts[tok] += 1
    overlap = 0
    for tok in pred_tokens:
        if counts[tok] > 0:
            overlap += 1
            counts[tok] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def numbers(text: str) -> list[float]:
    vals = []
    for match in re.findall(r"[-+]?(?:\d+\.\d+|\d+)(?:e[-+]?\d+)?", text.lower()):
        try:
            vals.append(float(match))
        except ValueError:
            pass
    return vals


def numeric_recall(pred: str, gold: str, rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> float | None:
    gold_nums = numbers(gold)
    pred_nums = numbers(pred)
    if not gold_nums:
        return None
    if not pred_nums:
        return 0.0
    matched = 0
    used: set[int] = set()
    for g in gold_nums:
        for idx, p in enumerate(pred_nums):
            if idx in used:
                continue
            if math.isclose(p, g, rel_tol=rel_tol, abs_tol=abs_tol):
                matched += 1
                used.add(idx)
                break
    return matched / len(gold_nums)


def score_answer(pred: str, gold: str) -> dict[str, Any]:
    norm_pred = normalize_text(pred)
    norm_gold = normalize_text(gold)
    fuzzy = fuzz.token_set_ratio(norm_pred, norm_gold) / 100.0 if norm_gold else 0.0
    f1 = token_f1(pred, gold)
    num_recall = numeric_recall(pred, gold)
    components = [fuzzy, f1]
    if num_recall is not None:
        components.append(num_recall)
    return {
        "exact": float(norm_pred == norm_gold and bool(norm_gold)),
        "fuzzy_token_set": fuzzy,
        "token_f1": f1,
        "numeric_recall": num_recall,
        "heuristic_score": statistics.mean(components) if components else 0.0,
    }


def image_to_data_url(image_obj: Any, max_edge: int | None, jpeg_quality: int) -> str:
    if isinstance(image_obj, dict):
        raw = image_obj.get("bytes")
        path = image_obj.get("path")
        if raw is None and path:
            raw = Path(path).read_bytes()
    else:
        raw = image_obj

    if raw is None:
        raise ValueError("image has neither bytes nor readable path")

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if max_edge and max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def image_file_to_data_url(path: Path, max_edge: int | None, jpeg_quality: int) -> str:
    img = Image.open(path).convert("RGB")
    if max_edge and max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def image_obj_to_pil(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, dict):
        raw = image_obj.get("bytes")
        path = image_obj.get("path")
        if raw is None and path:
            raw = Path(path).read_bytes()
    else:
        raw = image_obj
    if raw is None:
        raise ValueError("image has neither bytes nor readable path")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def save_turn_source_images(images: list[Any], turn_dir: Path) -> list[Path]:
    source_dir = turn_dir / "source_images"
    source_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, image in enumerate(images or []):
        path = source_dir / f"source_image_{idx}.png"
        image_obj_to_pil(image).save(path, "PNG")
        paths.append(path)
    (turn_dir / "processed_images").mkdir(parents=True, exist_ok=True)
    return paths


def turn_content(prompt: str, images: list[Any], max_edge: int | None, jpeg_quality: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images or []:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image, max_edge=max_edge, jpeg_quality=jpeg_quality)},
            }
        )
    return content


def tool_observation_content(
    text: str,
    image_paths: list[Path],
    max_edge: int | None,
    jpeg_quality: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_file_to_data_url(path, max_edge, jpeg_quality)}})
    return content


def tool_schema_for_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    schema = list(TOOL_SCHEMA)
    if getattr(args, "enable_summary_tool", False):
        schema.append(SUMMARY_TOOL_SCHEMA)
    return schema


def tool_system_prompt(
    max_tool_steps: int,
    tool_call_transport: str = "text",
    enable_summary_tool: bool = False,
    visible_tool_reasoning: bool = False,
    model: str = "",
) -> str:
    protocol = ""
    if tool_call_transport == "text":
        allowed_tools = "calculator, python_interpreter, or python_image_processing"
        if enable_summary_tool:
            allowed_tools += ", or summarize_the_task"
        protocol = """

Tool-call output format:
- To call a tool, output exactly one JSON object wrapped in <tool_call>...</tool_call>.
- Use this shape: <tool_call>{"name":"calculator","arguments":{"expression":"2+2"}}</tool_call>
- For Python tools, put code under "arguments": {"code": "..."}.
- For summarize_the_task, use "parameters" (or "arguments") with the structured visual-state fields described below.
- Do not include a final answer in the same assistant message as a <tool_call>.
- The content of <tool_call> must be JSON only: no Markdown fences, XML tool tags, GUI/pyautogui actions, or prose after the JSON object.
- PIL example: ImageEnhance.Contrast(img).enhance(1.5).save("processed_images/contrast.png").
- Always print a short summary or saved path from Python code.
"""
    visible_reasoning_contract = """

Visible tool reasoning contract:
- You may put a short visible evidence note immediately before a tool call. `<think>...</think>` is optional; plain visible text is also valid.
- If present, ground that note in specific visual evidence or prior tool observations.
- Then emit exactly one <tool_call>...</tool_call> block and do not give a final answer in that message.
""" if visible_tool_reasoning else ""
    kimi_contract = """

Kimi tool-use contract:
- For any operation that opens, crops, enhances, counts, or saves an image, call `python_image_processing`, never `python_interpreter`.
- `python_image_processing` code must use PIL and the Python standard library only. Do not import or call cv2, numpy, pytesseract, matplotlib, or GUI/display APIs.
- For calculator, send one uncommented arithmetic expression only, e.g. `2*pi*(12.5**2)`; do not send Python code, assignments, Markdown, or explanatory comments.
- Prefer Kimi's native action syntax exactly: `<|action_start|>python_image_processing<|action_sep|>{"code":"..."}<|action_end|>`. The legacy `<tool_call>{...}</tool_call>` JSON form is also accepted.
""" if "kimi" in model.lower() else ""
    return f"""You are a proactive visual tool-use assistant for VisualToolBench.

You may call local tools before giving the final answer. Use tools when they help inspect image regions, enhance/transform images, compute values, or run short Python analysis.

Available local tools:
- calculator: arithmetic expressions.
- python_interpreter: short Python snippets for computation/data processing.
- python_image_processing: PIL-only image manipulation. Source images are in source_images/. Save transformed PNGs to processed_images/. NumPy, OpenCV, OCR, and display packages are unavailable.
{"- summarize_the_task: use the reason_mine_code visual-state schema: `reasoning_so_far`, `completed_steps`, `current_state`, `visual_evidence`, `key_observations`, `uncertainties`, and `next_decision`. `visual_evidence` has `coordinates`, `bboxes`, `crop_ids`, `ocr_text`, `astar_outputs`, and `candidate_differences`. Keep verified visual/tool facts compact; do not put unsupported guesses or long prose in reasoning_so_far. It is internal memory, not a final answer; after it returns, continue with the next needed tool or final answer." if enable_summary_tool else ""}

Important image-tool constraints: pytesseract/OCR packages are unavailable. For text-heavy images, crop, zoom, sharpen, or increase contrast, save the processed image, and inspect the returned image in the next model step. Do not call Image.show(), matplotlib show(), display(), or external viewers.
{kimi_contract}
{protocol}
{visible_reasoning_contract}
Use at most {max_tool_steps} visual/computation tool step(s) for this turn. summarize_the_task is an internal memory tool and does not count against that visual/computation tool budget. After tool observations, continue reasoning and either call another useful tool or provide the final answer. When finalizing, answer the user's question directly."""


def tool_context_prompt(prompt: str, source_paths: list[Path], turn_dir: Path, draft_answer: str | None = None) -> str:
    source_list = "\n".join(f"- {path.name}: {path}" for path in source_paths) or "- no source images"
    draft_block = ""
    if draft_answer:
        draft_block = f"""

Baseline draft answer:
<draft_answer>
{draft_answer}
</draft_answer>

Use the draft as a strong anchor. Keep it unless your tool observations provide concrete visual, textual, or numeric evidence for a correction. Do not replace a specific draft answer with a vague refusal or a statement that more tools are needed.
"""
    return f"""{prompt}{draft_block}

Tool workspace for this turn:
- working directory: {turn_dir}
- source images:
{source_list}
- save processed image outputs under: {turn_dir / "processed_images"}
"""


def reasoning_prompt(prompt: str, retry: bool = False, answer_retry: bool = False, draft_answer: str | None = None) -> str:
    draft_block = ""
    if draft_answer:
        draft_block = f"""

Initial draft answer from a direct visual pass:
<draft_answer>
{draft_answer}
</draft_answer>

Use the draft as an anchor for coverage and wording, but verify it against the image. Keep correct details from the draft, correct unsupported or wrong details, and add missing requested details.
"""
    instruction = """

You must answer in exactly this structure:
<think>
Write your own concrete reasoning for this turn. Ground it in the actual visual evidence, dialogue history, relevant tool-use hints, measurements, arithmetic, and uncertainty. Do not write generic process statements such as "I need to inspect the image" or "I will analyze carefully"; name the visible evidence and how it supports the answer. Do not refuse just because the image is imperfect; use visible text, labels, diagrams, and context to make the best supported inference unless the requested evidence is truly absent.
</think>
<answer>
Start with the direct final answer, then add the concise supporting details needed by a grader. The answer must be self-contained: include every requested entity, count, country/name/group, unit, arithmetic result, comparison, and short reason. Do not refer to the <think> section. Do not collapse lists or multi-part answers into a terse phrase. Do not over-expand: avoid long repeated strings, irrelevant caveats, and generic explanation. For numeric or formula questions, show the key calculation and final value with units. For list questions, use a compact list.
</answer>
"""
    if retry:
        instruction += """

Your previous response was rejected because it did not contain a substantive model-generated <think>...</think> and <answer>...</answer>. Regenerate the answer with concrete visual reasoning, not a template.
"""
    if answer_retry:
        instruction += """

Your previous final <answer> was too short or incomplete compared with your reasoning and the question. Regenerate with the same required tags, but make <answer> complete and self-contained. Keep useful details in <answer>, not only in <think>.
"""
    return str(prompt) + draft_block + instruction


def repair_reasoning_prompt(prompt: str, final_answer: str, draft_answer: str | None = None) -> str:
    draft_block = ""
    if draft_answer:
        draft_block = f"""

Direct draft answer used as an anchor:
<draft_answer>
{draft_answer}
</draft_answer>
"""
    return f"""{prompt}{draft_block}

The final answer below has already been selected. Do not change its content, numbers, names, units, ordering, or wording.
<final_answer_to_preserve>
{final_answer}
</final_answer_to_preserve>

Output only the missing reasoning paragraph. Do not output <answer>, do not repeat the final answer verbatim, and do not use XML tags.
The paragraph must be your own concrete visual reasoning that supports the preserved final answer. Ground it in visible evidence, labels, tables, charts, dialogue history, arithmetic, and uncertainty. Do not use generic process statements such as "I need to inspect the image" or "I will analyze carefully". If some evidence is uncertain, say which visible cues support the best inference.
"""


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*([\s\S]*?)\s*</{tag}>", text or "", flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_answer(text: str) -> str:
    return extract_tag(text, "answer") or (text or "").strip()


def extract_reasoning(text: str) -> str:
    return extract_tag(text, "think")


def has_substantive_reasoning(text: str) -> bool:
    reasoning = extract_reasoning(text)
    answer = extract_answer(text)
    if not reasoning or not answer or not re.search(r"</answer>", text or "", flags=re.IGNORECASE):
        return False
    words = re.findall(r"[A-Za-z0-9]+", reasoning)
    if len(words) < 25:
        return False
    generic_bits = [
        "i need to inspect",
        "inspect the image",
        "analyze the image carefully",
        "look at the image",
        "current visual evidence",
    ]
    low = reasoning.lower()
    return not any(bit in low for bit in generic_bits)


def answer_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))


def needs_complete_answer_retry(prompt: str, pred: str) -> bool:
    answer = extract_answer(pred)
    reasoning = extract_reasoning(pred)
    answer_words = answer_word_count(answer)
    reasoning_words = answer_word_count(reasoning)
    prompt_low = (prompt or "").lower()

    if not answer:
        return True
    if answer_words <= 3 and any(bit in prompt_low for bit in ["mention", "name", "which", "what", "how many", "why"]):
        return True
    if answer_words < 20 and any(bit in prompt_low for bit in ["mention", "explain", "compare", "according to", "which of", "what are", "list"]):
        return True
    if answer_words < 30 and reasoning_words >= max(80, answer_words * 5):
        return True
    return False


def has_refusal(text: str) -> bool:
    low = (text or "").lower()
    refusal_bits = [
        "cannot be determined",
        "can't be determined",
        "cannot determine",
        "can't determine",
        "impossible to determine",
        "not enough information",
        "insufficient information",
        "not visible",
        "too blurry",
        "image is unclear",
    ]
    return any(bit in low for bit in refusal_bits)


def has_uncertain_or_refusal(text: str) -> bool:
    low = (text or "").lower()
    uncertain_bits = [
        "unable to",
        "not possible",
        "not able to",
        "difficult to",
        "hard to",
        "not clearly",
        "not legible",
        "illegible",
        "blurry",
        "low-resolution",
        "low resolution",
        "cannot read",
        "can't read",
        "missing",
    ]
    return has_refusal(low) or any(bit in low for bit in uncertain_bits)


def has_degenerate_answer(text: str) -> bool:
    text = text or ""
    if len(text) > 4000:
        return True
    if re.search(r"(\d)\1{20,}", text):
        return True
    if len(numbers(text)) > 60:
        return True
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    if len(tokens) > 50:
        most_common = max((tokens.count(tok) for tok in set(tokens)), default=0)
        if most_common / len(tokens) > 0.35:
            return True
    return False


def has_risky_zero_percent_answer(text: str) -> bool:
    text = text or ""
    return bool(re.search(r"(^|\s)0\s*%", text.lower())) and answer_word_count(text) < 120


def refusal_more_nums_prediction(draft: str | None, pred: str) -> str:
    """Promote only clear no-GT recoveries from a draft refusal."""
    if not draft:
        return pred
    answer = extract_answer(pred)
    draft_num_count = len(numbers(draft))
    candidate_num_count = len(numbers(answer))
    candidate_words = answer_word_count(answer)
    good_recovery = (
        has_refusal(draft)
        and not has_refusal(answer)
        and not has_degenerate_answer(answer)
        and candidate_words >= 12
        and candidate_num_count >= max(1, draft_num_count)
    )
    if good_recovery:
        return pred
    return draft_final_prediction(draft, pred)


def refusal_recovery_prediction(draft: str | None, pred: str) -> str:
    """Promote a non-degenerate candidate when the draft backed away."""
    if not draft:
        return pred
    answer = extract_answer(pred)
    draft_num_count = len(numbers(draft))
    candidate_num_count = len(numbers(answer))
    good_recovery = (
        has_uncertain_or_refusal(draft)
        and not has_uncertain_or_refusal(answer)
        and not has_degenerate_answer(answer)
        and not has_risky_zero_percent_answer(answer)
        and answer_word_count(answer) >= 8
        and candidate_num_count >= max(0, draft_num_count - 3)
    )
    if good_recovery:
        return pred
    return draft_final_prediction(draft, pred)


def conservative_reasoning_prediction(draft: str | None, pred: str) -> str:
    if not draft:
        return pred
    answer = extract_answer(pred)
    reasoning = extract_reasoning(pred)
    candidate_text = answer or pred
    draft_words = answer_word_count(draft)
    candidate_words = answer_word_count(candidate_text)
    draft_nums = numbers(draft)
    candidate_nums = numbers(candidate_text)
    bad_candidate = False
    if not candidate_text.strip():
        bad_candidate = True
    if candidate_words < 20 and draft_words >= 40:
        bad_candidate = True
    if candidate_words < draft_words * 0.65 and draft_words >= 60:
        bad_candidate = True
    if has_refusal(candidate_text) and not has_refusal(draft):
        bad_candidate = True
    if len(candidate_nums) < len(draft_nums):
        bad_candidate = True
    if draft_words >= 40 and token_f1(candidate_text, draft) < 0.55:
        bad_candidate = True
    if not has_substantive_reasoning(pred):
        bad_candidate = True

    if not bad_candidate:
        return pred

    think = sanitize_embedded_tags(reasoning)
    return f"<think>\n{think.strip()}\n</think>\n<answer>\n{draft.strip()}\n</answer>"


def lenient_reasoning_prediction(draft: str | None, pred: str) -> str:
    if not draft:
        return pred
    answer = extract_answer(pred)
    reasoning = extract_reasoning(pred)
    candidate_text = answer or pred
    draft_words = answer_word_count(draft)
    candidate_words = answer_word_count(candidate_text)
    bad_candidate = False
    if not candidate_text.strip():
        bad_candidate = True
    if candidate_words < 20 and draft_words >= 40:
        bad_candidate = True
    if has_refusal(candidate_text) and not has_refusal(draft):
        bad_candidate = True
    if len(numbers(candidate_text)) + 2 < len(numbers(draft)):
        bad_candidate = True
    if not has_substantive_reasoning(pred):
        bad_candidate = True
    if not bad_candidate:
        return pred
    think = sanitize_embedded_tags(reasoning)
    return f"<think>\n{think.strip()}\n</think>\n<answer>\n{draft.strip()}\n</answer>"


def sanitize_embedded_tags(text: str) -> str:
    text = text or ""
    text = re.sub(r"</?(?:think|answer)>", "", text, flags=re.IGNORECASE)
    return text.strip()


def draft_final_prediction(draft: str | None, pred: str) -> str:
    if not draft:
        return pred
    reasoning = sanitize_embedded_tags(extract_reasoning(pred))
    return f"<think>\n{reasoning.strip()}\n</think>\n<answer>\n{draft.strip()}\n</answer>"


def promote_or_draft_prediction(draft: str | None, pred: str, allow_promotion: bool, policy: str) -> str:
    if not draft:
        return pred
    if allow_promotion:
        if policy == "always":
            promoted = pred
        elif policy == "refusal_recovery":
            promoted = refusal_recovery_prediction(draft, pred)
        elif policy == "refusal_more_nums":
            promoted = refusal_more_nums_prediction(draft, pred)
        elif policy == "lenient":
            promoted = lenient_reasoning_prediction(draft, pred)
        else:
            promoted = conservative_reasoning_prediction(draft, pred)
        if extract_answer(promoted).strip() != draft.strip():
            return promoted
    return draft_final_prediction(draft, pred)


def parse_csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def should_use_reasoning(args: argparse.Namespace, row: dict[str, Any]) -> bool:
    if args.reasoning_mode == "none":
        return False
    if not args.reasoning_format:
        return False
    if args.reasoning_mode == "all":
        return True
    categories = getattr(args, "reasoning_categories_set", set())
    focuses = getattr(args, "reasoning_focuses_set", set())
    if not categories and not focuses:
        return True
    category = str(row.get("prompt_category") or "").lower()
    focus = str(row.get("eval_focus") or "").lower()
    return category in categories or focus in focuses


def tool_choice_for_step(args: argparse.Namespace, step_idx: int) -> str:
    if args.tool_choice_mode == "required":
        return "required"
    if args.tool_choice_mode == "required-first" and step_idx == 0:
        return "required"
    return "auto"


def textual_tool_choice_instruction(args: argparse.Namespace, step_idx: int) -> dict[str, str] | None:
    if args.tool_call_transport != "text":
        return None
    if tool_choice_for_step(args, step_idx) != "required":
        return None
    return {
        "role": "user",
        "content": (
            "For this step, you must use exactly one local tool before answering. "
            "You may write concise reasoning for why the tool is needed, then output exactly one "
            "<tool_call>{...}</tool_call> block using calculator, python_interpreter, or "
            "python_image_processing. Do not provide the final answer yet."
        ),
    }


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(str(item.get("text") or ""))
                elif item.get("text"):
                    chunks.append(str(item.get("text") or ""))
            elif item is not None:
                chunks.append(str(item))
        return "\n".join(chunks)
    return "" if content is None else str(content)


def strip_visible_think_blocks(messages: list[dict[str, Any]]) -> int:
    """Remove only visible reasoning, leaving action blocks and observations intact."""
    removed = 0
    for message in messages:
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            continue
        original = str(message["content"])
        pruned = re.sub(r"<think>.*?</think>\s*", "", original, flags=re.IGNORECASE | re.DOTALL).strip()
        if pruned != original:
            message["content"] = pruned
            removed += 1
    return removed


def rough_text_tokens(text: str) -> int:
    return (len(text or "") + 3) // 4


VISUAL_EVIDENCE_KEYWORDS = (
    "text",
    "read",
    "reads",
    "label",
    "ocr",
    "letter",
    "number",
    "digit",
    "color",
    "colour",
    "red",
    "green",
    "blue",
    "yellow",
    "purple",
    "white",
    "black",
    "gray",
    "grey",
    "orange",
    "brown",
    "pink",
    "bbox",
    "bounding",
    "coordinate",
    "coordinates",
    "x=",
    "y=",
    "width",
    "height",
    "contour",
    "component",
    "centroid",
    "area",
    "pixel",
    "pixels",
    "crop",
    "cropped",
    "zoom",
    "flipped",
    "rotated",
    "ruler",
    "tick",
    "cm",
    "distance",
    "count",
    "option",
    "candidate",
)


NON_EVIDENCE_VISUAL_PATTERNS = (
    r"\b(?:not|isn't|wasn't)\s+(?:clear|readable|legible|visible)\b",
    r"\b(?:cannot|can't|unable to|impossible to)\s+(?:determine|read|see|extract|identify)\b",
    r"\b(?:no|not)\s+(?:useful|clear|readable|legible)\s+(?:information|detail|text|result)\b",
    r"\b(?:enhancement|contrast|crop|processing)\b.*\b(?:did not work|failed|unhelpful|ineffective)\b",
    r"\b(?:still|remains?)\s+(?:unclear|unreadable|illegible|ambiguous)\b",
    r"\b(?:appears?|seems?|maybe|might|likely|probably|possibly|guess(?:ing)?|suggests?)\b",
)


REASONING_NOISE_PATTERNS = (
    r"^\s*i (will|would|need to|should|can|am going to)\b.*\b(use|call|apply|run|invoke|select)\b",
    r"\bi will\b.*\b(read|use|call|apply|run|invoke|select)\b",
    r"\bi need to\b.*\b(determine|first|identify|measure|read|find)\b",
    r"^\s*to (accurately|better|further|resolve|determine|measure|identify|read)\b",
    r"^\s*the next step is\b",
    r"^\s*i will now\b",
    r"^\s*i'll\b.*\b(use|call|apply|run)\b",
    r"^\s*let me\b",
    r"^\s*first,?\b",
    r"^\s*next,?\b",
    r"^\s*then,?\b",
    r"^\s*finally,?\b",
    r"^\s*this (tool|function) (is|will be|can be)\b",
    r"^\s*the `?opencv_[a-z_]+`? (function|tool)\b.*\b(ideal|useful|needed|help)\b",
)


def _strip_reasoning_markup(text: str) -> str:
    text = re.sub(r"</?think>", "", text or "", flags=re.IGNORECASE)
    text = re.sub(r"<tool_call[^>]*>[\s\S]*?(?:</tool_call>|$)", "", text, flags=re.IGNORECASE)
    return text.strip()


def _split_evidence_units(text: str) -> list[str]:
    text = _strip_reasoning_markup(text)
    if not text:
        return []
    units = []
    for raw_line in re.split(r"[\r\n]+", text):
        line = raw_line.strip(" \t-*")
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`(-])", line)
        units.extend(part.strip() for part in parts if part.strip())
    return units


def _looks_like_reasoning_noise(unit: str) -> bool:
    lowered = unit.lower().strip()
    if not lowered:
        return True
    if "tool_call" in lowered or lowered.startswith("[requested tool call"):
        return True
    if any(marker in lowered for marker in ("i need to", "i should", "i will", "i'll", "let me", "now i", "next step", "final answer")):
        return True
    return any(re.search(pattern, lowered) for pattern in REASONING_NOISE_PATTERNS)


def _looks_like_visual_evidence(unit: str) -> bool:
    """Accept only compact, externally checkable visual anchors.

    V3 is a replacement for discarded reasoning, not a second reasoning pass.
    A statement such as "the red region looks important" is therefore not an
    anchor, even though it is visual: it is an interpretation which can bias a
    later turn.  Keep only literal OCR, geometry, measurements, and explicit
    option-to-option spatial/appearance relations.
    """
    lowered = unit.lower()
    if any(re.search(pattern, lowered) for pattern in NON_EVIDENCE_VISUAL_PATTERNS):
        return False
    # Concrete geometry/crop identifiers are always useful task state.
    if re.search(r"\b(?:bbox|bounding box|crop(?:_id)?|img_\d+)\b", lowered) and (
        re.search(r"\[\s*-?\d+(?:\.\d+)?\s*,", unit) or "img_" in lowered or "crop" in lowered
    ):
        return True
    if re.search(r"\b(?:x|y|w|h|width|height)\s*[:=]\s*-?\d", lowered):
        return True
    if re.search(r"\([-+]?\d+(?:\.\d+)?,\s*[-+]?\d+(?:\.\d+)?\)", unit):
        return True
    # OCR needs a literal transcription, not merely a claim that text exists.
    if any(keyword in lowered for keyword in ("ocr", "label", "reads", "read", "text")) and (
        re.search(r"[\"'`][^\"'`\n]{1,120}[\"'`]", unit)
        or re.search(r"\b(?:ocr(?:\s+text)?|label|text)\s*:\s*\S+", lowered)
    ):
        return True
    # A non-numeric appearance claim is retained only as an explicit relation
    # between named alternatives, rather than a free-form interpretation.
    local_terms = ("left", "right", "top", "bottom", "color", "colour", "edge", "texture", "red", "blue", "green", "yellow", "black", "white")
    named_options = len(re.findall(r"\b(?:candidate|option|choice|part)\s*[A-D1-9]\b", unit, flags=re.IGNORECASE))
    if named_options >= 2 and any(term in lowered for term in local_terms):
        return True
    if re.search(r"\b(?:candidate|option|choice|part)\s*[A-D1-9]\b", unit, flags=re.IGNORECASE) and re.search(
        r"\b(?:left|right|above|below|top|bottom|same|different)\s+(?:of|than)\s+\b(?:candidate|option|choice|part)\s*[A-D1-9]\b",
        lowered,
    ):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:cm|px|pixels?|%)\b", lowered):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\b", unit) and any(word in lowered for word in ("area", "count", "distance", "coordinate", "label")):
        return True
    return False


def extract_v3_structured_evidence(text: str, max_units: int = 4, max_chars: int = 600) -> str:
    """PRUNE_METHODS V3-compatible compact visual evidence block."""
    kept = []
    seen = set()
    for unit in _split_evidence_units(text):
        normalized = re.sub(r"\s+", " ", unit).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        if _looks_like_reasoning_noise(normalized):
            continue
        if not _looks_like_visual_evidence(normalized):
            continue
        seen.add(key)
        kept.append(normalized)
        if len(kept) >= max_units:
            break
    if not kept:
        return ""
    compact = "Structured visual evidence kept:\n" + "\n".join(f"- {unit}" for unit in kept)
    if len(compact) > max_chars:
        compact = compact[: max_chars - 3].rstrip() + "..."
    return compact


def extract_action_block(text: str) -> str:
    match = re.search(r"<tool_call[^>]*>[\s\S]*?(?:</tool_call>|$)", text or "", re.I)
    return match.group(0).strip() if match else ""


def format_segment_action_response(action_block: str, response_block: str = "") -> str:
    """Keep only the action in assistant context; the real observation already follows as a user message."""
    del response_block
    return action_block.strip()


def prune_segment_content(segment: dict[str, Any], policy: str) -> tuple[str, int]:
    reasoning = str(segment.get("reasoning") or "")
    action_response = format_segment_action_response(
        str(segment.get("action_block") or ""), str(segment.get("response_block") or "")
    )
    if policy in {"v3", "vtc-visual-evidence"}:
        pieces = [extract_v3_structured_evidence(reasoning), action_response]
        pruned = "\n\n".join(piece for piece in pieces if piece)
    elif policy == "v2":
        pruned = action_response
    else:
        pruned = ""
    original = str(segment.get("original_content") or reasoning)
    saved = max(rough_text_tokens(original) - rough_text_tokens(pruned), 0)
    return pruned, saved


def prune_assistant_content(original: str, policy: str) -> tuple[str, int]:
    """Compatibility helper for older saved summary events."""
    return prune_segment_content(
        {"reasoning": original, "action_block": extract_action_block(original), "original_content": original}, policy
    )


def context_text_stats(messages: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    assistant = 0
    image_items = 0
    for msg in messages:
        text = message_text(msg)
        total += rough_text_tokens(text)
        if msg.get("role") == "assistant":
            assistant += rough_text_tokens(text)
        content = msg.get("content")
        if isinstance(content, list):
            image_items += sum(1 for item in content if isinstance(item, dict) and item.get("type") == "image_url")
    return {
        "rough_text_tokens": total,
        "assistant_text_tokens": assistant,
        "image_items": image_items,
        "image_items_per_1k_text_tokens": image_items / max(total, 1) * 1000.0,
    }


def active_reasoning_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only unpruned reasoning segments are KL candidates; summaries/actions are never candidates."""
    return [
        segment
        for segment in segments
        if not segment.get("pruned") and str(segment.get("reasoning") or "").strip()
    ]


SUMMARY_VISUAL_EVIDENCE_KEYS = (
    "coordinates", "bboxes", "crop_ids", "ocr_text", "astar_outputs", "candidate_differences"
)


def _summary_text(value: Any, max_chars: int = 260) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _summary_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [text for text in (_summary_text(item) for item in values) if text]


def _summary_visual_evidence(value: Any) -> dict[str, list[str]]:
    source = value if isinstance(value, dict) else {}
    return {key: _summary_list(source.get(key))[:8] for key in SUMMARY_VISUAL_EVIDENCE_KEYS}


def build_task_state_summary(tool_args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """reason_mine_code's summarize_the_task state, separate from V3 evidence retention."""
    params = tool_args if isinstance(tool_args, dict) else {}
    # Preserve compatibility with old text-only calls, but store them as task
    # state rather than attempting to extract/retain V3 visual evidence.
    legacy = _summary_text(
        params.get("content") or params.get("summary") or params.get("current_information") or params.get("_raw")
    )
    summary = {
        "reasoning_so_far": _summary_text(params.get("reasoning_so_far")),
        "completed_steps": _summary_list(params.get("completed_steps")),
        "current_state": _summary_text(params.get("current_state") or legacy),
        "visual_evidence": _summary_visual_evidence(params.get("visual_evidence")),
        "key_observations": _summary_list(params.get("key_observations")),
        "uncertainties": _summary_list(params.get("uncertainties")),
        "next_decision": _summary_text(params.get("next_decision") or params.get("next_step")),
    }
    compact_lines: list[str] = []
    if summary["reasoning_so_far"]:
        compact_lines.append(f"reasoning_so_far: {summary['reasoning_so_far']}")
    for key in ("completed_steps", "key_observations", "uncertainties"):
        if summary[key]:
            compact_lines.append(f"{key}: " + "; ".join(summary[key][:6]))
    if summary["current_state"]:
        compact_lines.append(f"current_state: {summary['current_state']}")
    for key in SUMMARY_VISUAL_EVIDENCE_KEYS:
        values = summary["visual_evidence"][key]
        if values:
            compact_lines.append(f"{key}: " + "; ".join(values[:6]))
    if summary["next_decision"]:
        compact_lines.append(f"next_decision: {summary['next_decision']}")
    return summary, "\n".join(compact_lines) or "No useful task state was provided to summarize."


def summary_content_from_args(tool_args: dict[str, Any]) -> str:
    """Task-state summary used exclusively as the teacher context for KL."""
    _, compact_summary = build_task_state_summary(tool_args)
    return compact_summary


def apply_summary_kl_prune(
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    *,
    prior_segments: list[dict[str, Any]],
    current_pre_tool_text: str,
    summary_text: str,
    task_id: str,
    turn_dir: Path,
) -> dict[str, Any] | None:
    if not getattr(args, "summary_kl_prune", False):
        return None
    os.environ["SUMMARY_KL_ENABLE"] = "1"
    # Rollout and KL scoring may use different OpenAI-compatible services.
    # DashScope serves multimodal generation but not vLLM's tokenizer/logprob
    # endpoints required by the official KL scorer.
    kl_base_url = str(getattr(args, "kl_base_url", "") or args.base_url)
    kl_model = str(getattr(args, "kl_model", "") or args.model)
    os.environ["SUMMARY_KL_BASE_URL"] = kl_base_url
    os.environ["SUMMARY_KL_MODEL"] = kl_model
    os.environ["SUMMARY_KL_TOPK"] = str(args.summary_kl_top_k)
    os.environ["SUMMARY_KL_PRUNE_THRESHOLD"] = str(args.summary_kl_threshold)
    os.environ["SUMMARY_KL_PRUNE_MIN_TOKENS"] = str(args.summary_kl_min_high_tokens)
    os.environ.setdefault("OPENAI_BASE_URL", args.base_url)
    os.environ.setdefault("OPENAI_API_KEY", args.api_key)
    candidates = active_reasoning_segments(prior_segments)
    result = log_summarization_kl(
        system_prompt=tool_system_prompt(
            args.max_tool_steps,
            args.tool_call_transport,
            enable_summary_tool=getattr(args, "enable_summary_tool", False),
            visible_tool_reasoning=getattr(args, "visible_tool_reasoning", False),
            model=args.model,
        ),
        summary_text=summary_text,
        prior_assistant_raw_texts=[str(segment["reasoning"]) for segment in candidates],
        current_pre_tool_text=current_pre_tool_text,
        model=args.model,
        task_id=task_id,
        output_dir=str(turn_dir / "summary_kl_logs"),
    )
    if not isinstance(result, dict):
        return result

    before = context_text_stats(messages)
    pruned = []
    apply_prune = bool(getattr(args, "summary_kl_apply_prune", True))
    prune_policy = str(getattr(args, "summary_prune_policy", "clear") or "clear")
    direct_delete = prune_policy == "direct-delete"
    v4_mixed = prune_policy in {"v4", "v5"}
    content_prune_policy = (
        "clear" if direct_delete
        else "v3" if prune_policy == "v5-roundwise"
        else prune_policy
    )
    keep_recent = int(getattr(args, "summary_kl_keep_recent_assistant_steps", 0))
    decisions = {
        item.get("global_index"): item
        for item in result.get("step_decisions", [])
        if item.get("step") == "prior_assistant"
    }
    # V4 compresses every eligible historical reasoning segment: high-KL
    # segments retain only grounded visual evidence plus the action, while
    # low-KL segments retain the action alone.  Observations remain separate
    # user messages in both cases.
    prune_indices = range(len(candidates)) if (direct_delete or v4_mixed) else (result.get("prune_indices") or [])
    for global_idx in prune_indices:
        if global_idx < 0 or global_idx >= len(candidates):
            continue
        if keep_recent and global_idx >= len(candidates) - keep_recent:
            continue
        segment = candidates[global_idx]
        msg_idx = int(segment.get("message_index", -1))
        if msg_idx < 0 or msg_idx >= len(messages):
            continue
        original = str(segment.get("original_content") or message_text(messages[msg_idx]))
        decision = decisions.get(global_idx) or {}
        segment_policy = content_prune_policy
        high_kl = False
        if v4_mixed:
            high_kl = int(decision.get("n_tokens_above_threshold") or 0) >= int(args.summary_kl_min_high_tokens)
            segment_policy = "v3" if high_kl else "v2"
        pruned_content, saved_tokens = prune_segment_content(segment, segment_policy)
        commit_prune = apply_prune
        if segment.get("context_visual_compacted") and high_kl:
            # This segment is already represented exactly like the Visual
            # baseline.  A high-KL decision says the summary does not cover
            # it, so retain that evidence and reconsider it at a later event
            # instead of pretending a no-op reconstruction was a deletion.
            commit_prune = False
        if commit_prune:
            messages[msg_idx]["content"] = pruned_content
            segment["pruned"] = True
            segment["pruned_content"] = pruned_content
        pruned.append(
            {
                "prior_index": global_idx,
                "segment_id": segment.get("segment_id"),
                "message_index": msg_idx,
                "saved_rough_text_tokens": saved_tokens,
                "original_rough_text_tokens": rough_text_tokens(original),
                "remaining_rough_text_tokens": rough_text_tokens(pruned_content),
                "prune_policy": prune_policy,
                "content_prune_policy": segment_policy,
                "preserved_content": pruned_content,
                "applied": commit_prune,
                "n_tokens": int(decision.get("n_tokens") or 0),
                "n_tokens_above_threshold": int(decision.get("n_tokens_above_threshold") or 0),
                "mean_normalized_kl": float(decision.get("mean_normalized_kl") or 0.0),
                "max_normalized_kl": float(decision.get("max_normalized_kl") or 0.0),
            }
        )
    after = context_text_stats(messages)
    result["applied_pruned_messages"] = pruned
    result["applied_pruned_token_count"] = sum(item["saved_rough_text_tokens"] for item in pruned if item["applied"])
    result["context_ratio"] = {
        "before_prune": before,
        "after_prune": after,
        "delta_rough_text_tokens": before["rough_text_tokens"] - after["rough_text_tokens"],
        "delta_assistant_text_tokens": before["assistant_text_tokens"] - after["assistant_text_tokens"],
        "image_items_per_1k_text_tokens_delta": (
            after["image_items_per_1k_text_tokens"] - before["image_items_per_1k_text_tokens"]
        ),
    }
    return result


def apply_pruned_messages_from_summary_event(
    messages: list[dict[str, Any]],
    summary_event: dict[str, Any],
) -> int:
    """Apply a previously computed KL prune event to the real message list."""
    total_saved = 0
    for item in summary_event.get("applied_pruned_messages") or []:
        if not item.get("applied", True):
            continue
        msg_idx = int(item.get("message_index", -1))
        if msg_idx < 0 or msg_idx >= len(messages):
            continue
        original = message_text(messages[msg_idx])
        if not original:
            continue
        preserved = item.get("preserved_content")
        if preserved is None:
            policy = str(item.get("prune_policy") or "clear")
            preserved, saved = prune_assistant_content(original, policy)
        else:
            preserved = str(preserved)
            saved = max(rough_text_tokens(original) - rough_text_tokens(preserved), 0)
        messages[msg_idx]["content"] = preserved
        total_saved += saved
    return total_saved


def estimate_summary_context_overhead_tokens(assistant_text: str, tool_name: str, tool_args: dict[str, Any]) -> int:
    """Estimate new text tokens added to model context if a summary probe is kept."""
    summary_result = execute_tool_call(
        tool_name,
        tool_args,
        Path("/tmp/visualtoolbench_summary_overhead_probe"),
        timeout=0,
        max_chars=4000,
        prev_processed_count=0,
    )
    observation = format_tool_observation(tool_name, summary_result, 4000)
    return rough_text_tokens(assistant_text) + rough_text_tokens("Tool observation for the previous call:\n" + observation)


def preserve_model_reasoning_with_tags(text: str) -> str:
    """Wrap model-generated pre-answer reasoning in <think> without inventing text."""
    if not text or extract_reasoning(text):
        return text or ""
    answer_match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", text, flags=re.IGNORECASE)
    if not answer_match:
        return text
    prefix = text[: answer_match.start()].strip()
    if len(re.findall(r"[A-Za-z0-9]+", prefix)) < 25:
        return text
    answer = answer_match.group(1).strip()
    return f"<think>\n{prefix}\n</think>\n<answer>\n{answer}\n</answer>"


def completion_text(choice: Any) -> str:
    if isinstance(choice, dict):
        msg = choice.get("message") or {}
        content = msg.get("content", "")
    else:
        msg = getattr(choice, "message", None)
        content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(part.get("text", ""))
            else:
                chunks.append(str(part))
        return "\n".join(chunks)
    return "" if content is None else str(content)


def normalize_tool_name(name: str | None) -> str:
    name = (name or "").strip()
    return TOOL_ALIASES.get(name, name)


def completion_tool_calls(choice: Any) -> list[dict[str, Any]]:
    if isinstance(choice, dict):
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
    else:
        msg = getattr(choice, "message", None)
        tool_calls = getattr(msg, "tool_calls", []) if msg is not None else []
    parsed = []
    for idx, call in enumerate(tool_calls or []):
        if not isinstance(call, dict):
            call = {
                "id": getattr(call, "id", f"call_{idx}"),
                "type": getattr(call, "type", "function"),
                "function": getattr(call, "function", {}),
            }
        fn = call.get("function") or {}
        if not isinstance(fn, dict):
            fn = {"name": getattr(fn, "name", ""), "arguments": getattr(fn, "arguments", "{}")}
        parsed.append(
            {
                "id": call.get("id") or f"call_{idx}",
                "type": call.get("type") or "function",
                "function": {
                    "name": normalize_tool_name(fn.get("name")),
                    "arguments": fn.get("arguments") or "{}",
                },
            }
        )
    return parsed


def parse_text_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse standard text calls plus Kimi's native/legacy action markup.

    Kimi may add Markdown after an otherwise valid JSON object, use its native
    action markers, or emit a small XML calculator call.  We recover only a
    declared local-tool call and discard trailing presentation noise.
    """
    if not text:
        return []
    decoder = json.JSONDecoder()
    calls = []

    def clean_calculator_expression(value: Any) -> str:
        raw = str(value or "").strip().strip("`")
        raw = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", raw, flags=re.I)
        for line in reversed(raw.splitlines()):
            candidate = line.strip().strip("`")
            if not candidate or candidate.startswith("#"):
                continue
            if "=" in candidate and "==" not in candidate:
                candidate = candidate.rsplit("=", 1)[-1].strip()
            candidate = re.sub(r"^(?:answer|result|expression)\s*:\s*", "", candidate, flags=re.I)
            if re.fullmatch(r"[0-9A-Za-z_+\-*/%^().,\s]+", candidate):
                return candidate
        return raw.splitlines()[0].strip() if raw else ""

    def append_call(name: Any, args: Any) -> None:
        normalized = normalize_tool_name(str(name or ""))
        if normalized not in {"calculator", "python_interpreter", "python_image_processing", SUMMARY_TOOL_NAME}:
            return
        if not isinstance(args, dict):
            return
        if normalized == "calculator" and "expression" in args:
            args = {"expression": clean_calculator_expression(args["expression"])}
        elif normalized in {"python_interpreter", "python_image_processing"} and "code" in args:
            code = str(args["code"]).strip()
            code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code, flags=re.I)
            # Kimi frequently labels PIL/image code as generic Python. Route
            # it to the image sandbox, which supplies the correct workspace
            # conventions and image helpers.
            if normalized == "python_interpreter" and re.search(
                r"\bImage\.open\b|\b(?:source_images|processed_images)/|\bcv2\b|\bImageEnhance\b",
                code,
                flags=re.I,
            ):
                normalized = "python_image_processing"
            args = {"code": code}
        calls.append(
            {
                "id": f"text_call_{len(calls)}",
                "type": "function",
                "function": {"name": normalized, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )

    # Kimi's native action markup, as normalized in the prior GTA adapter.
    for native in re.finditer(r"<\|action_start\|>\s*([^<]+?)\s*<\|action_sep\|>\s*([\s\S]*?)\s*<\|action_end\|>", text):
        raw_args = native.group(2).strip()
        try:
            obj, _ = decoder.raw_decode(raw_args)
        except json.JSONDecodeError:
            continue
        append_call(native.group(1).strip(), obj)

    for match in re.finditer(r"<tool_call[^>]*>", text, flags=re.IGNORECASE):
        start = text.find("{", match.end())
        if start < 0:
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool_name")
        args = obj.get("parameters") or obj.get("arguments") or obj.get("args") or {}
        append_call(name, args)
    # Legacy Kimi calculator markup; do not attempt to infer arbitrary XML or
    # GUI actions as a benchmark tool call.
    if not calls:
        calculator = re.search(r"<calculator>\s*<expression>\s*([\s\S]*?)\s*</expression>\s*</calculator>", text, re.I)
        if calculator:
            append_call("calculator", {"expression": calculator.group(1).strip()})
    return calls


def strip_text_tool_calls(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<tool_call[^>]*>[\s\S]*?(?:</tool_call>|$)", "", text, flags=re.IGNORECASE).strip()


def looks_like_tool_request_text(text: str) -> bool:
    low = (text or "").lower()
    return "<tool_call" in low or "[requested tool call" in low or low.strip().startswith("requested tool call")


def sanitize_final_tool_request(text: str) -> str:
    if not looks_like_tool_request_text(text):
        return text or ""
    stripped = strip_text_tool_calls(text)
    stripped = re.sub(r"\[Requested tool call\(s\):[^\]]+\]", "", stripped, flags=re.IGNORECASE).strip()
    if stripped and not looks_like_tool_request_text(stripped):
        return stripped
    return (
        "No final answer was produced: the model requested another tool after the tool budget was exhausted. "
        "See tool_traces and raw_responses for the pending tool request."
    )


def ensure_reasoning_answer_tags(text: str) -> str:
    text = text or ""
    if extract_tag(text, "answer"):
        return text
    if extract_reasoning(text):
        return f"{text.rstrip()}\n<answer>\n{sanitize_embedded_tags(text).strip()}\n</answer>"
    return f"<think>\nTool observations and visual evidence were considered before finalizing.\n</think>\n<answer>\n{text.strip()}\n</answer>"


def parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if raw_args is None:
        return {}
    text = str(raw_args).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text}


def trim_tool_text(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"


class SafeCalculator(ast.NodeVisitor):
    allowed_funcs = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
    }
    allowed_names = {"pi": math.pi, "e": math.e}

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> int | float:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("calculator supports only numeric constants")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("unsupported binary operator")

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.allowed_names:
            return self.allowed_names[node.id]
        raise ValueError(f"unknown name: {node.id}")

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name) or node.func.id not in self.allowed_funcs:
            raise ValueError("unsupported function")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        args = [self.visit(arg) for arg in node.args]
        return self.allowed_funcs[node.func.id](*args)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"unsupported expression node: {node.__class__.__name__}")


def run_calculator(expression: str) -> dict[str, Any]:
    expression = (expression or "").replace("^", "**")
    tree = ast.parse(expression, mode="eval")
    value = SafeCalculator().visit(tree)
    return {"ok": True, "output": repr(value), "result": value}


def run_python_code(code: str, cwd: Path, timeout: float, max_chars: int, image_mode: bool = False) -> dict[str, Any]:
    if len(code or "") > 5000:
        return {"ok": False, "error": "code exceeds 5000 characters"}
    started = time.time()
    env = os.environ.copy()
    for noisy_key in ["VSCODE_IPC_HOOK_CLI", "VSCODE_NLS_CONFIG", "ELECTRON_RUN_AS_NODE"]:
        env.pop(noisy_key, None)
    existing_pythonpath = env.get("PYTHONPATH", "")
    pydeps = env.get("VISUALTOOLBENCH_PYTHONPATH", "").strip()
    if pydeps and Path(pydeps).exists():
        env["PYTHONPATH"] = pydeps + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    if image_mode:
        code = (
            "from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageStat, ImageChops\n"
            "if not hasattr(Image.Image, 'enhance'):\n"
            "    Image.Image.enhance = lambda self, factor: ImageEnhance.Contrast(self).enhance(factor)\n"
            "try:\n"
            "    import numpy as np\n"
            "except Exception:\n"
            "    np = None\n"
            "try:\n"
            "    import cv2\n"
            "except Exception:\n"
            "    cv2 = None\n"
            + (code or "")
        )
    proc = subprocess.run(
        [sys.executable, "-c", code or ""],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    stdout = trim_tool_text(proc.stdout, max_chars)
    stderr = trim_tool_text(proc.stderr, max_chars)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "latency_sec": time.time() - started,
    }


def list_processed_images(turn_dir: Path) -> list[Path]:
    processed_dir = turn_dir / "processed_images"
    if not processed_dir.exists():
        return []
    return sorted(
        path
        for path in processed_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )


def execute_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    turn_dir: Path,
    timeout: float,
    max_chars: int,
    prev_processed_count: int,
) -> dict[str, Any]:
    started = time.time()
    try:
        if tool_name == "calculator":
            result = run_calculator(str(tool_args.get("expression") or tool_args.get("_raw") or ""))
            result["latency_sec"] = time.time() - started
            return result
        if tool_name in {"python_interpreter", "python_image_processing"}:
            result = run_python_code(
                str(tool_args.get("code") or tool_args.get("_raw") or ""),
                turn_dir,
                timeout,
                max_chars,
                image_mode=tool_name == "python_image_processing",
            )
            processed = list_processed_images(turn_dir)
            new_images = processed[prev_processed_count:]
            result["new_image_paths"] = [str(path) for path in new_images]
            return result
        if tool_name == SUMMARY_TOOL_NAME:
            summary_state, summary_text = build_task_state_summary(tool_args)
            return {
                "ok": bool(summary_text),
                "output": (
                    (
                        "Compact memory summary:\n"
                        f"{summary_text}\n\n"
                        "Continue with the next needed visual/tool step or final answer."
                    )
                    if summary_text
                    else "summarize_the_task requires a non-empty content argument."
                ),
                "summary": summary_text,
                "task_state_summary": summary_state,
                "tool_response_from": SUMMARY_TOOL_NAME,
                "counts_as_benchmark_tool_call": False,
                "latency_sec": time.time() - started,
            }
        return {"ok": False, "error": f"unsupported local tool: {tool_name}", "latency_sec": time.time() - started}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"tool timed out after {timeout} seconds", "latency_sec": time.time() - started}
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "latency_sec": time.time() - started}


def format_tool_observation(tool_name: str, result: dict[str, Any], max_chars: int) -> str:
    lines = [f"Tool `{tool_name}` finished.", f"ok: {bool(result.get('ok'))}"]
    for key in ["output", "stdout", "stderr", "error", "returncode"]:
        value = result.get(key)
        if value not in (None, ""):
            lines.append(f"{key}: {trim_tool_text(str(value), max_chars)}")
    new_images = result.get("new_image_paths") or []
    if new_images:
        lines.append("new_images:")
        lines.extend(f"- {path}" for path in new_images)
    return "\n".join(lines)


def chat_completion(base_url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"chat completion HTTP {exc.code}: {detail}") from exc


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id"):
                done.add(str(row["id"]))
    return done


def load_draft_predictions(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    draft_path = Path(path)
    drafts: dict[str, list[str]] = {}
    prediction_files = (
        sorted(draft_path.glob("shards/shard_*/predictions.jsonl"))
        if draft_path.is_dir()
        else [draft_path]
    )
    for prediction_file in prediction_files:
        with prediction_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = row.get("id")
                preds = row.get("predictions") or []
                if row_id:
                    drafts[str(row_id)] = ["" if pred is None else str(pred) for pred in preds]
    return drafts


def generate_controller_summary(
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    original_prompt: str,
    prior_segments: list[dict[str, Any]] | None = None,
) -> tuple[str, float]:
    """Generate task state off the main rollout path.

    The action model therefore sees exactly the baseline tool prompt until a
    summary has produced a real KL pruning decision.  Images are intentionally
    omitted: this summarizes text/tool state and must not invent new visual
    observations.
    """
    # The live action context may already be reconstructed to the Visual
    # floor.  Summary generation must instead inspect the original reasoning
    # of every still-active segment; otherwise KL is asked whether a summary
    # covers information that the summary never had a chance to observe.
    controller_messages = copy.deepcopy(messages)
    for segment in prior_segments or []:
        if segment.get("pruned"):
            continue
        msg_idx = int(segment.get("message_index", -1))
        if 0 <= msg_idx < len(controller_messages):
            controller_messages[msg_idx]["content"] = str(
                segment.get("original_content")
                or segment.get("reasoning")
                or message_text(controller_messages[msg_idx])
            )

    history = []
    recent_images: list[dict[str, Any]] = []
    for message in controller_messages:
        text = message_text(message).strip()
        if message.get("role") == "assistant":
            # The controller summarizes task state, not executable action
            # payloads.  Large PIL snippets and serialized tool arguments can
            # otherwise crowd out the visual observation they were meant to
            # obtain and leak JSON fragments into the summary.
            text = strip_text_tool_calls(text).strip()
        # Artifact paths identify no visual fact by themselves and are often
        # hundreds of tokens long.  Keep the fact that an artifact exists but
        # remove machine-specific path noise.
        text = re.sub(
            r"/(?:mnt|tmp|home|workspace)/[^\s'\"<>]+",
            "<image artifact>",
            text,
        )
        text = trim_tool_text(text, 1200).strip()
        if text:
            history.append(f"{message.get('role', 'unknown')}: {text}")
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    recent_images.append(copy.deepcopy(item))
    # A visual-state summary must be able to verify the current focused view,
    # not merely amplify guesses from assistant prose.  The two newest images
    # are sufficient for provenance checking while keeping controller cost
    # bounded.
    recent_images = recent_images[-2:]
    history_text = "\n\n".join(history)
    if len(history_text) > 14000:
        history_text = history_text[-14000:]
    summary_prompt = f"""Create a compact representation of the current task state.

Original request:
{original_prompt}

Text/tool history:
{history_text}

Include only:
- completed tool calls and their verified observations;
- accumulated visual evidence with concrete coordinates, bbox/crop ids, OCR-like
  text actually visible in the history, counts, and candidate differences;
- exact entity/label-to-value bindings and units; keep verified observations
  separate from tentative or conflicting ones;
- current conclusion, remaining uncertainty, and the next decision.

Do not solve the task again, add unsupported visual claims, or include long
reasoning. Treat every value stated only in assistant prose as tentative. Mark
a value verified only when it is directly readable in an attached image or
appears in concrete tool stdout; merely saving/cropping an image verifies no
answer value. If the evidence is unreadable or conflicting, say exactly what
remains unresolved instead of repeating a guess. Use at most 120 words. Return
compact plain text, not a tool call and not a final answer."""
    controller_content: str | list[dict[str, Any]] = summary_prompt
    if recent_images:
        controller_content = [
            {"type": "text", "text": summary_prompt},
            *recent_images,
        ]
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You compress existing multimodal task state. Preserve verified evidence and "
                    "uncertainty; never invent observations."
                ),
            },
            {"role": "user", "content": controller_content},
        ],
        "temperature": 0,
        "top_p": 1,
        # This is a transient controller hint, not a second chain of thought.
        # Long summaries were a measurable source of context distraction in
        # V5, so cap it tightly even when the rollout itself permits more.
        "max_tokens": min(160, int(args.max_tokens)),
    }
    started = time.time()
    response = chat_completion(args.base_url, args.api_key, payload, args.timeout)
    draft_summary = completion_text(response["choices"][0]).strip()

    # A second, independent pass is deliberately extractive.  The first pass
    # is good at organizing state but can promote a plausible reasoning claim
    # into a fact.  Re-show the current visual evidence and require the same
    # model to remove every unsupported binding before the summary can affect
    # pruning or the next action.
    verify_prompt = f"""Audit the draft task-state summary against the source history and attached images.

Original request:
{original_prompt}

Source text/tool history:
{history_text}

Draft summary:
{draft_summary}

Return a corrected compact summary under 100 words. Copy a concrete fact only
if its entity-to-value/attribute binding is directly supported by an attached
image or concrete tool output. Python comments, hard-coded constants, prior
assistant guesses, and the mere existence of a saved crop are not evidence.
Keep unresolved items explicitly uncertain. If at least one positive fact is
supported, begin with `Verified:`. If none is supported, return exactly
`No verified evidence; retain prior visual state.` Do not answer the original
question and do not add a new fact."""
    verify_content: str | list[dict[str, Any]] = verify_prompt
    if recent_images:
        verify_content = [
            {"type": "text", "text": verify_prompt},
            *recent_images,
        ]
    verify_payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an evidence verifier, not a solver. Delete unsupported claims; "
                    "never repair them by guessing."
                ),
            },
            {"role": "user", "content": verify_content},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": min(144, int(args.max_tokens)),
    }
    verified_response = chat_completion(
        args.base_url, args.api_key, verify_payload, args.timeout
    )
    latency = time.time() - started
    return completion_text(verified_response["choices"][0]).strip(), latency


def controller_summary_is_actionable(summary_text: str) -> bool:
    """Admit only summaries containing positive, checkable evidence.

    An uncertainty-only checkpoint is useful for audit but harmful as a
    one-shot reasoning hint: it tends to turn a difficult visual question into
    a refusal while providing no state that can safely replace prior evidence.
    """
    text = re.sub(r"\s+", " ", summary_text or "").strip().lower()
    if not text or "verified" not in text:
        return False
    negative_only = (
        "no verified evidence",
        "no usable visual evidence",
        "no verifiable",
        "nothing is verified",
        "cannot be verified",
    )
    if any(marker in text for marker in negative_only):
        # A later positive clause may still make the checkpoint useful, e.g.
        # "X verified; Y remains unresolved".  Require an explicit positive
        # verified binding before accepting such a summary.
        return bool(re.search(r"(?:verified\s*:|is verified|verified (?:as|at|via|by))\s*[^.;]{2,}", text))
    return True


def iter_rows(parquet_path: Path) -> tuple[pq.ParquetFile, Any]:
    pf = pq.ParquetFile(parquet_path)
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=TEXT_COLUMNS)
        for row in table.to_pylist():
            yield pf, row


def run_tool_calling_turn(
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    prompt: str,
    images: list[Any],
    turn_dir: Path,
    draft_answer: str | None = None,
    require_reasoning_format: bool = False,
) -> dict[str, Any]:
    source_paths = save_turn_source_images(images, turn_dir)
    messages.append(
        {
            "role": "user",
            "content": turn_content(
                tool_context_prompt(prompt, source_paths, turn_dir, draft_answer=draft_answer),
                images,
                args.max_image_edge,
                args.jpeg_quality,
            ),
        }
    )

    raw_responses = []
    tool_trace = []
    summary_events = []
    discarded_summary_events = []
    segments: list[dict[str, Any]] = []
    total_latency = 0.0
    final_pred = ""
    real_tool_observation_count = 0
    summary_tool_observation_count = 0
    summary_tool_called = False
    pending_summary_hint = ""
    pending_control_hint = ""
    post_summary_verification_pending = False
    roundwise_summary_due = False

    # Two control retries allow a premature summary probe and a skipped
    # post-summary final answer to be redirected without consuming real-tool
    # budget.
    max_model_steps = args.max_tool_steps + max(0, int(getattr(args, "max_summary_tool_steps", 1))) + 3
    for step_idx in range(max_model_steps + 1):
        step_messages = messages
        temporary_hints = []
        if pending_summary_hint:
            temporary_hints.append("Temporary compact task state for this next decision only:\n" + pending_summary_hint)
            pending_summary_hint = ""
        if pending_control_hint:
            temporary_hints.append(pending_control_hint)
            pending_control_hint = ""
        if temporary_hints:
            step_messages = messages + [{"role": "user", "content": "\n\n".join(temporary_hints)}]
        forced_text_tool_instruction = textual_tool_choice_instruction(args, real_tool_observation_count)
        if forced_text_tool_instruction is not None:
            step_messages = step_messages + [forced_text_tool_instruction]
        payload = dict(
            model=args.model,
            messages=step_messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        if args.tool_call_transport == "api":
            payload["tools"] = tool_schema_for_args(args)
            payload["tool_choice"] = tool_choice_for_step(args, real_tool_observation_count)
        started = time.time()
        resp = chat_completion(args.base_url, args.api_key, payload, args.timeout)
        total_latency += time.time() - started
        raw_responses.append(resp)
        choice = resp["choices"][0]
        pred = completion_text(choice)
        tool_calls = completion_tool_calls(choice)
        textual_tool_calls = False
        if not tool_calls:
            tool_calls = parse_text_tool_calls(pred)
            textual_tool_calls = bool(tool_calls)
        if not tool_calls:
            if looks_like_tool_request_text(pred):
                messages.append({"role": "assistant", "content": pred})
                final_pred = ""
                break
            if post_summary_verification_pending and step_idx < max_model_steps:
                pending_control_hint = (
                    "A post-summary visual verification is still required. Do not finalize yet. "
                    "Call `python_image_processing` now to inspect a focused crop or alternate view "
                    "that can verify the requested critical quantity, option, entity, label, or relationship."
                )
                continue
            if args.summary_every_round and roundwise_summary_due and step_idx < max_model_steps:
                pending_control_hint = (
                    "Before continuing or finalizing, call `summarize_the_task` for the just-completed "
                    "tool round. This summary is required only to judge which preceding reasoning "
                    "segments are covered and may be pruned."
                )
                continue
            final_pred = pred
            messages.append({"role": "assistant", "content": pred})
            break

        requested_tool_names = [
            normalize_tool_name((call.get("function") or {}).get("name"))
            for call in tool_calls
        ]
        requested_real_tools = [name for name in requested_tool_names if name != SUMMARY_TOOL_NAME]
        requested_summary_tools = [name for name in requested_tool_names if name == SUMMARY_TOOL_NAME]
        if args.summary_every_round and roundwise_summary_due and not requested_summary_tools:
            pending_control_hint = (
                "Call `summarize_the_task` now for the just-completed tool round before making "
                "another tool call. It will be used to decide whether preceding reasoning is removable."
            )
            continue
        summary_limit = max(0, int(getattr(args, "max_summary_tool_steps", 1)))
        if (
            post_summary_verification_pending
            and requested_summary_tools
            and summary_tool_observation_count >= summary_limit
        ):
            # V5 intentionally removes the summary call from persistent
            # history, so a model may try to call it again.  Redirect that
            # redundant request to the required visual verification instead
            # of terminating the trajectory as summary-budget exhausted.
            tool_calls = [
                call
                for call, name in zip(tool_calls, requested_tool_names)
                if name != SUMMARY_TOOL_NAME
            ]
            requested_tool_names = [
                name for name in requested_tool_names if name != SUMMARY_TOOL_NAME
            ]
            requested_real_tools = list(requested_tool_names)
            requested_summary_tools = []
            if not tool_calls:
                pending_control_hint = (
                    "The task-state summary has already been completed. Do not call "
                    "`summarize_the_task` again. Use `python_image_processing` now to "
                    "verify the unresolved critical visual fact, then answer."
                )
                continue
        real_budget_exhausted = bool(requested_real_tools) and real_tool_observation_count >= args.max_tool_steps
        summary_budget_exhausted = bool(requested_summary_tools) and summary_tool_observation_count >= summary_limit
        model_step_guard_exhausted = step_idx >= max_model_steps
        if real_budget_exhausted or summary_budget_exhausted or model_step_guard_exhausted:
            exhausted_text = strip_text_tool_calls(pred) if textual_tool_calls else (pred or "")
            messages.append({"role": "assistant", "content": exhausted_text + "\n\n[Tool budget exhausted before final answer.]"})
            tool_trace.append(
                {
                    "step_idx": step_idx,
                    "tool_budget_exhausted": True,
                    "real_tool_observation_count": real_tool_observation_count,
                    "summary_tool_observation_count": summary_tool_observation_count,
                    "real_budget_exhausted": real_budget_exhausted,
                    "summary_budget_exhausted": summary_budget_exhausted,
                    "model_step_guard_exhausted": model_step_guard_exhausted,
                    "pending_tool_calls": tool_calls,
                }
            )
            final_pred = ""
            break

        assistant_text = strip_text_tool_calls(pred) if textual_tool_calls else (pred or "")
        # Record the exact action in every textual-tool assistant turn.  This
        # makes a tool turn an explicit reasoning/action segment independent
        # of whether KL pruning is enabled.
        action_block = extract_action_block(pred) if textual_tool_calls else ""
        assistant_context_text = assistant_text
        if action_block:
            assistant_context_text = "\n\n".join(part for part in (assistant_text, action_block) if part)
        current_assistant_message = {"role": "assistant", "content": assistant_context_text}
        current_assistant_appended = False
        for call_idx, call in enumerate(tool_calls):
            fn = call.get("function") or {}
            tool_name = normalize_tool_name(fn.get("name"))
            tool_args = parse_tool_arguments(fn.get("arguments"))
            summary_text = summary_content_from_args(tool_args) if tool_name == SUMMARY_TOOL_NAME else ""
            summary_event = None
            if (
                post_summary_verification_pending
                and tool_name not in {SUMMARY_TOOL_NAME, "python_image_processing"}
            ):
                pending_control_hint = (
                    "The required post-summary check must inspect the image, not repeat arithmetic "
                    "or a prior guess. Call `python_image_processing` on a focused crop or alternate view."
                )
                continue
            if tool_name == SUMMARY_TOOL_NAME and args.summary_gated_probe:
                completed_segment_keys = {
                    int(segment.get("message_index", -1))
                    for segment in segments
                    if int(segment.get("message_index", -1)) >= 0
                }
                active_segments_all = active_reasoning_segments(segments)
                active_segments_by_message = {
                    int(segment.get("message_index", -1)): segment
                    for segment in active_segments_all
                    if int(segment.get("message_index", -1)) >= 0
                }
                active_segments = list(active_segments_by_message.values())
                accumulated_reasoning_tokens = sum(
                    rough_text_tokens(str(segment.get("reasoning") or ""))
                    for segment in active_segments
                )
                precondition_gate = {
                    "completed_reasoning_action_segments": len(completed_segment_keys),
                    "min_completed_reasoning_action_segments": int(args.summary_gate_min_segments),
                    "active_candidate_segments": len(active_segments),
                    "min_active_candidate_segments": int(args.summary_gate_min_candidates),
                    "accumulated_reasoning_tokens": accumulated_reasoning_tokens,
                    "min_accumulated_reasoning_tokens": int(args.summary_gate_min_reasoning_tokens),
                }
                precondition_gate["passed"] = (
                    len(completed_segment_keys) >= int(args.summary_gate_min_segments)
                    and len(active_segments) >= int(args.summary_gate_min_candidates)
                    and accumulated_reasoning_tokens >= int(args.summary_gate_min_reasoning_tokens)
                )
                if not precondition_gate["passed"]:
                    discarded_summary_events.append(
                        {
                            "step_idx": step_idx,
                            "call_idx": call_idx,
                            "reason": "summary_precondition_gate",
                            "precondition_gate": precondition_gate,
                        }
                    )
                    pending_control_hint = (
                        "The summary gate is not reached yet. Do not summarize or finalize from an "
                        "unsupported guess. Complete another useful tool call so that another "
                        "reasoning/action segment is available before summarization."
                    )
                    if args.summary_every_round:
                        roundwise_summary_due = False
                    continue
            if tool_name == SUMMARY_TOOL_NAME and args.summary_kl_prune:
                kl_messages = copy.deepcopy(messages) if args.summary_net_token_gate else messages
                kl_segments = copy.deepcopy(segments) if args.summary_net_token_gate else segments
                summary_event = apply_summary_kl_prune(
                    args,
                    kl_messages,
                    prior_segments=kl_segments,
                    current_pre_tool_text=assistant_text,
                    summary_text=summary_text,
                    task_id=f"{turn_dir.parent.name}/{turn_dir.name}",
                    turn_dir=turn_dir,
                )
                if isinstance(summary_event, dict):
                    gross_saved = int(summary_event.get("applied_pruned_token_count") or 0)
                    overhead = estimate_summary_context_overhead_tokens(assistant_text, tool_name, tool_args)
                    net_saved = gross_saved - overhead
                    summary_event["net_token_gate"] = {
                        "enabled": bool(args.summary_net_token_gate),
                        "min_net_saved_tokens": int(args.summary_net_token_min_savings),
                        "gross_pruned_rough_text_tokens": gross_saved,
                        "estimated_summary_context_overhead_tokens": overhead,
                        "estimated_net_saved_tokens": net_saved,
                        "passed": (
                            not args.summary_net_token_gate
                            or net_saved >= int(args.summary_net_token_min_savings)
                        ),
                    }
            if (
                tool_name == SUMMARY_TOOL_NAME
                and (args.summary_gated_probe or args.summary_net_token_gate)
                and (
                    not isinstance(summary_event, dict)
                    or not [item for item in summary_event.get("applied_pruned_messages", []) if item.get("applied")]
                    or (
                        args.summary_net_token_gate
                        and not (summary_event.get("net_token_gate") or {}).get("passed")
                    )
                )
            ):
                # This is a probe, not a completed tool step.  Do not append
                # its assistant call, tool observation, or any summary event
                # to the rollout trace: without a real KL prune, retaining
                # model-generated task state only expands and biases context.
                if args.summary_every_round:
                    roundwise_summary_due = False
                continue
            if tool_name == SUMMARY_TOOL_NAME:
                summary_tool_called = True
                summary_tool_observation_count += 1
                if args.summary_every_round:
                    roundwise_summary_due = False
            if tool_name == SUMMARY_TOOL_NAME and isinstance(summary_event, dict):
                if args.summary_net_token_gate:
                    actual_saved = apply_pruned_messages_from_summary_event(messages, summary_event)
                    pruned_segment_ids = {
                        item.get("segment_id") for item in summary_event.get("applied_pruned_messages") or []
                    }
                    for segment in segments:
                        if segment.get("segment_id") in pruned_segment_ids:
                            segment["pruned"] = True
                            segment["pruned_content"] = message_text(messages[segment["message_index"]])
                    summary_event["applied_pruned_token_count"] = actual_saved
                    summary_event["net_token_gate"]["actual_pruned_rough_text_tokens"] = actual_saved
                    summary_event["net_token_gate"]["actual_net_saved_tokens"] = (
                        actual_saved
                        - int(summary_event["net_token_gate"].get("estimated_summary_context_overhead_tokens") or 0)
                    )
                summary_events.append(summary_event)
            if tool_name == SUMMARY_TOOL_NAME and args.summary_prune_policy in {"v5", "v5-roundwise"}:
                # V5 keeps the summary out of the persistent trace.  It is
                # injected once into the next model request above, then gone.
                pending_summary_hint = summary_text
                if (
                    args.summary_post_verify
                    and real_tool_observation_count < args.max_tool_steps
                ):
                    post_summary_verification_pending = True
                    pending_summary_hint += (
                        "\n\nThe task-state summary has already been completed; do not call "
                        "`summarize_the_task` again. Verification requirement: this summary is "
                        "compact memory, not authority. "
                        "Treat counts, option identities, OCR/entity names, and inferred relationships "
                        "as hypotheses unless they are directly supported by a tool observation or a "
                        "specific visible crop. Before the final answer, use the remaining real tool "
                        "step to verify the requested critical fact against the image. Do not merely "
                        "reprint a value hard-coded in prior Python comments."
                    )
                continue
            if not current_assistant_appended:
                messages.append(current_assistant_message)
                current_assistant_appended = True
            prev_processed_count = len(list_processed_images(turn_dir))
            result = execute_tool_call(
                tool_name,
                tool_args,
                turn_dir,
                args.tool_timeout,
                args.tool_output_max_chars,
                prev_processed_count,
            )
            new_image_paths = [Path(path) for path in result.get("new_image_paths") or []]
            observation = format_tool_observation(tool_name, result, args.tool_output_max_chars)
            if tool_name != SUMMARY_TOOL_NAME:
                real_tool_observation_count += 1
                if tool_name == "python_image_processing" and post_summary_verification_pending:
                    post_summary_verification_pending = False
            tool_trace.append(
                {
                    "step_idx": step_idx,
                    "call_idx": call_idx,
                    "tool_call_id": call.get("id"),
                    "textual_tool_call": textual_tool_calls,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "ok": bool(result.get("ok")),
                    "observation": observation,
                    "new_image_paths": [str(path) for path in new_image_paths],
                    "result": result,
                    "summary_kl": summary_event,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": tool_observation_content(
                        "Tool observation for the previous call:\n" + observation,
                        new_image_paths,
                        args.max_image_edge,
                        args.jpeg_quality,
                    ),
                }
            )
            if tool_name != SUMMARY_TOOL_NAME:
                segment = {
                    "segment_id": f"{turn_dir.name}:step-{step_idx}:call-{call_idx}",
                    "message_index": len(messages) - 2,
                    "reasoning": assistant_text,
                    "action_block": action_block,
                    "response_block": observation,
                    "original_content": assistant_context_text,
                    "pruned": False,
                }
                segments.append(segment)
                retention = getattr(args, "reasoning_retention", "preserve")
                if retention in {"clear", "visual-evidence"}:
                    policy = "clear" if retention == "clear" else "v3"
                    replacement, _ = prune_segment_content(segment, policy)
                    messages[segment["message_index"]]["content"] = replacement
                    # Balanced roundwise V5 starts from the visual-evidence
                    # baseline in the live context, but keeps the original
                    # reasoning in the lifecycle record until Summary-KL has
                    # classified it.  This makes its final representation no
                    # longer than Visual while still allowing the official KL
                    # decision to remove covered evidence selectively.
                    defer_kl_lifecycle = (
                        args.comparison_group in {"prune-v5-roundwise", "prune-v5-nohint"}
                        and retention == "visual-evidence"
                    )
                    segment["pruned"] = not defer_kl_lifecycle
                    segment["context_visual_compacted"] = defer_kl_lifecycle
                    segment["pruned_content"] = replacement
                if args.summary_every_round:
                    roundwise_summary_due = True
                    pending_control_hint = (
                        "A tool round has just completed. Call `summarize_the_task` next so KL can "
                        "judge which preceding reasoning segment is already covered and removable."
                    )
            if (
                args.enable_summary_tool
                and args.summary_nudge_after_tools > 0
                and not summary_tool_called
                and tool_name != SUMMARY_TOOL_NAME
                and real_tool_observation_count >= args.summary_nudge_after_tools
            ):
                # This is control-plane guidance, not task evidence.  Keeping
                # it in persistent history biases every later decision even
                # when the summary probe is discarded.  Present it only to
                # the next model request, like the one-shot summary itself.
                pending_control_hint = (
                    "Decide now whether a compact task-state summary would help the next decision. "
                    "If so, call `summarize_the_task` once with the original request, completed tool "
                    "calls, verified observations, unresolved uncertainty, and next action. If a "
                    "different visual/tool step is more useful, call that instead. This instruction "
                    "is temporary control guidance and is not task evidence."
                )
            if (
                getattr(args, "controller_summary", False)
                and tool_name != SUMMARY_TOOL_NAME
            ):
                active_segments = active_reasoning_segments(segments)
                accumulated_reasoning_tokens = sum(
                    rough_text_tokens(str(segment.get("reasoning") or ""))
                    for segment in active_segments
                )
                # Require two segments before the first checkpoint.  After
                # that lifecycle gate, classify each newly completed segment
                # once; older segments marked pruned never re-enter.
                min_active_candidates = (
                    1 if summary_tool_observation_count > 0
                    else int(args.summary_gate_min_candidates)
                )
                controller_gate = {
                    "completed_reasoning_action_segments": len(segments),
                    "active_candidate_segments": len(active_segments),
                    "min_active_candidate_segments": min_active_candidates,
                    "accumulated_reasoning_tokens": accumulated_reasoning_tokens,
                    "passed": (
                        len(segments) >= int(args.summary_gate_min_segments)
                        and len(active_segments) >= min_active_candidates
                        and accumulated_reasoning_tokens
                        >= int(args.summary_gate_min_reasoning_tokens)
                    ),
                }
                if controller_gate["passed"]:
                    summary_text, summary_latency = generate_controller_summary(
                        args, messages, prompt, prior_segments=segments
                    )
                    total_latency += summary_latency
                    if not controller_summary_is_actionable(summary_text):
                        discarded_summary_events.append(
                            {
                                "step_idx": step_idx,
                                "call_idx": call_idx,
                                "reason": "controller_summary_no_verified_evidence",
                                "controller_gate": controller_gate,
                                "summary_text": summary_text,
                            }
                        )
                        continue
                    kl_messages = copy.deepcopy(messages) if args.summary_net_token_gate else messages
                    kl_segments = copy.deepcopy(segments) if args.summary_net_token_gate else segments
                    summary_event = apply_summary_kl_prune(
                        args,
                        kl_messages,
                        prior_segments=kl_segments,
                        current_pre_tool_text="",
                        summary_text=summary_text,
                        task_id=f"{turn_dir.parent.name}/{turn_dir.name}",
                        turn_dir=turn_dir,
                    )
                    if isinstance(summary_event, dict):
                        summary_event["controller_generated"] = True
                        summary_event["controller_gate"] = controller_gate
                        gross_saved = int(summary_event.get("applied_pruned_token_count") or 0)
                        overhead = rough_text_tokens(summary_text)
                        net_saved = gross_saved - overhead
                        summary_event["net_token_gate"] = {
                            "enabled": bool(args.summary_net_token_gate),
                            "min_net_saved_tokens": int(args.summary_net_token_min_savings),
                            "gross_pruned_rough_text_tokens": gross_saved,
                            "estimated_summary_context_overhead_tokens": overhead,
                            "estimated_net_saved_tokens": net_saved,
                            "passed": (
                                not args.summary_net_token_gate
                                or net_saved >= int(args.summary_net_token_min_savings)
                            ),
                        }
                    applied = (
                        isinstance(summary_event, dict)
                        and [
                            item
                            for item in summary_event.get("applied_pruned_messages", [])
                            if item.get("applied")
                        ]
                        and (summary_event.get("net_token_gate") or {}).get("passed", True)
                    )
                    if applied:
                        if args.summary_net_token_gate:
                            actual_saved = apply_pruned_messages_from_summary_event(
                                messages, summary_event
                            )
                            pruned_segment_ids = {
                                item.get("segment_id")
                                for item in summary_event.get("applied_pruned_messages") or []
                            }
                            for segment in segments:
                                if segment.get("segment_id") in pruned_segment_ids:
                                    segment["pruned"] = True
                                    segment["pruned_content"] = message_text(
                                        messages[segment["message_index"]]
                                    )
                            summary_event["applied_pruned_token_count"] = actual_saved
                        summary_events.append(summary_event)
                        summary_tool_called = True
                        summary_tool_observation_count += 1
                        if getattr(args, "summary_inject_next_step", True):
                            pending_summary_hint = summary_text
                            summary_event["summary_injected_next_step"] = True
                        else:
                            summary_event["summary_injected_next_step"] = False
                        if (
                            args.summary_post_verify
                            and real_tool_observation_count < args.max_tool_steps
                        ):
                            post_summary_verification_pending = True
                            pending_summary_hint += (
                                "\n\nVerification requirement: use the remaining image-processing "
                                "step to check the critical entity/value against a focused crop "
                                "before finalizing. Treat this compact state as a hypothesis, not "
                                "authority."
                            )
                    else:
                        discarded_summary_events.append(
                            {
                                "step_idx": step_idx,
                                "call_idx": call_idx,
                                "reason": "controller_summary_no_committed_prune",
                                "controller_gate": controller_gate,
                                "summary_event": summary_event,
                            }
                        )

    if not final_pred:
        final_instruction = (
            "No more tools may be used. Do not request tools, do not output <tool_call>, "
            "and do not output '[Requested tool call(s): ...]'. Provide the best natural-language "
            "final answer to the original user question now, using the tool observations already available."
        )
        if require_reasoning_format:
            final_instruction += (
                "\n\nOutput exactly:\n<think>\nConcrete reasoning grounded in the image/tool observations.\n</think>\n"
                "<answer>\nThe direct final answer only.\n</answer>"
            )
        final_payload = dict(
            model=args.model,
            messages=messages
            + [
                {
                    "role": "user",
                    "content": final_instruction,
                }
            ],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        started = time.time()
        final_resp = chat_completion(args.base_url, args.api_key, final_payload, args.timeout)
        total_latency += time.time() - started
        raw_responses.append(final_resp)
        final_pred = completion_text(final_resp["choices"][0])
        if looks_like_tool_request_text(final_pred):
            retry_payload = dict(
                model=args.model,
                messages=messages
                + [
                    {"role": "assistant", "content": final_pred},
                    {
                        "role": "user",
                        "content": (
                            "That was still a tool request. Tools are disabled. Output only the final answer in prose. "
                            "Use the existing image/tool observations and make the best supported answer now."
                        ),
                    },
                ],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            retry_started = time.time()
            retry_resp = chat_completion(args.base_url, args.api_key, retry_payload, args.timeout)
            total_latency += time.time() - retry_started
            raw_responses.append(retry_resp)
            final_pred = completion_text(retry_resp["choices"][0])
        final_pred = sanitize_final_tool_request(final_pred)
        if require_reasoning_format:
            final_pred = ensure_reasoning_answer_tags(final_pred)
        messages.append({"role": "assistant", "content": final_pred})

    return {
        "prediction": final_pred,
        "raw_responses": raw_responses,
        "tool_trace": tool_trace,
        "summary_events": summary_events,
        "discarded_summary_events": discarded_summary_events,
        "latency_sec": total_latency,
        "source_image_paths": [str(path) for path in source_paths],
    }


def run_one(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    prompts = row.get("turn_prompts") or []
    golds = row.get("turn_golden_answers") or []
    images_by_turn = row.get("images_by_turn") or [[] for _ in prompts]

    messages: list[dict[str, Any]] = []
    if args.enable_tool_calling:
        messages.append(
            {
                "role": "system",
                "content": tool_system_prompt(
                    args.max_tool_steps,
                    args.tool_call_transport,
                    enable_summary_tool=args.enable_summary_tool,
                    visible_tool_reasoning=args.visible_tool_reasoning,
                    model=args.model,
                ),
            }
        )
    elif args.direct_final_answer_contract:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Tools are disabled for this task. Do not request or describe tool calls, and do not "
                    "provide a step-by-step chain of thought. Provide the best supported natural-language "
                    "final answer to the original user question, grounded in the supplied image(s). Output "
                    "only the final answer in concise prose."
                ),
            }
        )
    predictions = []
    full_predictions = []
    reasoning_traces = []
    raw_responses = []
    tool_traces = []
    summary_kl_events = []
    discarded_summary_events = []
    turn_scores = []
    errors = []
    total_latency = 0.0
    use_reasoning = should_use_reasoning(args, row)
    row_drafts = getattr(args, "draft_predictions", {}).get(str(row.get("id")), [])

    for turn_idx, prompt in enumerate(prompts):
        images = images_by_turn[turn_idx] if turn_idx < len(images_by_turn) else []
        draft_answer = row_drafts[turn_idx] if turn_idx < len(row_drafts) else None
        if args.enable_tool_calling:
            started = time.time()
            try:
                turn_dir = args.tool_artifacts_dir / str(row.get("id")) / f"turn_{turn_idx}"
                turn_dir.mkdir(parents=True, exist_ok=True)
                rollout_draft = None if args.fallback_to_draft_if_no_prune else draft_answer
                prompt_for_turn = reasoning_prompt(str(prompt), retry=False, draft_answer=rollout_draft) if use_reasoning else str(prompt)
                result = run_tool_calling_turn(
                    args,
                    messages,
                    prompt_for_turn,
                    images,
                    turn_dir,
                    draft_answer=rollout_draft,
                    require_reasoning_format=use_reasoning,
                )
                latency = result["latency_sec"]
                total_latency += latency
                pred = result["prediction"]
                if (
                    draft_answer
                    and args.fallback_to_draft_if_no_prune
                    and not (result.get("summary_events") or [])
                ):
                    # Cache-based stability ablation: when V5 made no context
                    # change, keep the supplied baseline answer instead of
                    # paying for unrelated independent-rollout variance.
                    pred = draft_answer
                if use_reasoning:
                    pred = preserve_model_reasoning_with_tags(pred)
                    pred = ensure_reasoning_answer_tags(pred)
                if draft_answer and args.prefer_draft_final:
                    pred = draft_final_prediction(draft_answer, pred) if use_reasoning else draft_answer
                answer_pred = extract_answer(pred) if use_reasoning else pred
                predictions.append(answer_pred)
                full_predictions.append(pred)
                reasoning_traces.append(extract_reasoning(pred) if use_reasoning else "")
                raw_responses.append(result["raw_responses"])
                turn_summary_events = result.get("summary_events") or []
                turn_discarded_summary_events = result.get("discarded_summary_events") or []
                summary_kl_events.extend(turn_summary_events)
                discarded_summary_events.extend(turn_discarded_summary_events)
                tool_traces.append(
                    {
                        "turn": turn_idx,
                        "source_image_paths": result["source_image_paths"],
                        "steps": result["tool_trace"],
                        "summary_kl_events": turn_summary_events,
                        "discarded_summary_events": turn_discarded_summary_events,
                    }
                )
                gold = golds[turn_idx] if turn_idx < len(golds) else ""
                score_text = pred if use_reasoning and args.score_full_response else answer_pred
                score = score_answer(score_text, gold)
                score["reasoning_enabled"] = float(use_reasoning)
                score["tool_calling_enabled"] = 1.0
                score["num_tool_calls"] = len(result["tool_trace"])
                score["num_successful_tool_calls"] = sum(1 for call in result["tool_trace"] if call.get("ok"))
                score["num_summary_calls"] = sum(1 for call in result["tool_trace"] if call.get("tool_name") == SUMMARY_TOOL_NAME)
                score["num_summary_discards"] = len(turn_discarded_summary_events)
                score["summary_kl_pruned_messages"] = sum(
                    len(event.get("applied_pruned_messages") or [])
                    for event in turn_summary_events
                    if isinstance(event, dict)
                )
                score["summary_kl_pruned_rough_text_tokens"] = sum(
                    int(event.get("applied_pruned_token_count") or 0)
                    for event in turn_summary_events
                    if isinstance(event, dict)
                )
                ratio_deltas = [
                    ((event.get("context_ratio") or {}).get("image_items_per_1k_text_tokens_delta"))
                    for event in turn_summary_events
                    if isinstance(event, dict)
                ]
                ratio_deltas = [float(x) for x in ratio_deltas if x is not None]
                if ratio_deltas:
                    score["summary_kl_image_items_per_1k_text_delta"] = sum(ratio_deltas) / len(ratio_deltas)
                if use_reasoning:
                    score["has_think"] = float(bool(extract_reasoning(pred)))
                    score["has_answer"] = float(bool(extract_tag(pred, "answer")))
                    score["substantive_reasoning"] = float(has_substantive_reasoning(pred))
                    score["scored_full_response"] = float(args.score_full_response)
                    score["answer_words"] = answer_word_count(answer_pred)
                    score["reasoning_words"] = answer_word_count(extract_reasoning(pred))
                turn_scores.append(score)
            except Exception as exc:
                latency = time.time() - started
                total_latency += latency
                errors.append({"turn": turn_idx, "error": repr(exc), "latency_sec": latency})
                predictions.append("")
                full_predictions.append("")
                reasoning_traces.append("")
                raw_responses.append(None)
                tool_traces.append({"turn": turn_idx, "steps": [], "error": repr(exc)})
                gold = golds[turn_idx] if turn_idx < len(golds) else ""
                turn_scores.append(score_answer("", gold))
                break
            continue
        if use_reasoning and args.draft_answer_first and not draft_answer:
            draft_payload = dict(
                model=args.model,
                messages=messages
                + [
                    {
                        "role": "user",
                        "content": turn_content(str(prompt), images, args.max_image_edge, args.jpeg_quality),
                    }
                ],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            try:
                draft_resp = chat_completion(args.base_url, args.api_key, draft_payload, args.timeout)
                draft_answer = completion_text(draft_resp["choices"][0])
            except Exception:
                draft_answer = None
        messages.append(
            {
                "role": "user",
                "content": turn_content(
                    reasoning_prompt(str(prompt), retry=False, draft_answer=draft_answer) if use_reasoning else str(prompt),
                    images,
                    args.max_image_edge,
                    args.jpeg_quality,
                ),
            }
        )
        started = time.time()
        try:
            resp = None
            repair_resp = None
            pred = ""
            for attempt in range(args.reasoning_retries + 1):
                if use_reasoning and attempt > 0:
                    messages[-1]["content"] = turn_content(
                        reasoning_prompt(str(prompt), retry=True),
                        images,
                        args.max_image_edge,
                        args.jpeg_quality,
                    )
                payload = dict(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                resp = chat_completion(args.base_url, args.api_key, payload, args.timeout)
                pred = completion_text(resp["choices"][0])
                if use_reasoning:
                    pred = preserve_model_reasoning_with_tags(pred)
                if not use_reasoning or has_substantive_reasoning(pred):
                    break
            if use_reasoning and args.answer_completeness_retries > 0 and needs_complete_answer_retry(str(prompt), pred):
                best_pred = pred
                for _ in range(args.answer_completeness_retries):
                    messages[-1]["content"] = turn_content(
                        reasoning_prompt(str(prompt), answer_retry=True),
                        images,
                        args.max_image_edge,
                        args.jpeg_quality,
                    )
                    payload = dict(
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                    )
                    resp = chat_completion(args.base_url, args.api_key, payload, args.timeout)
                    pred = completion_text(resp["choices"][0])
                    pred = preserve_model_reasoning_with_tags(pred)
                    if has_substantive_reasoning(pred) and not needs_complete_answer_retry(str(prompt), pred):
                        break
                    if has_substantive_reasoning(pred):
                        best_pred = pred
                if not has_substantive_reasoning(pred) and has_substantive_reasoning(best_pred):
                    pred = best_pred
            if use_reasoning and args.conservative_finalize:
                pred = conservative_reasoning_prediction(draft_answer, pred)
            if use_reasoning and args.prefer_draft_final:
                pred = draft_final_prediction(draft_answer, pred)
            if use_reasoning and args.promote_reasoning_final:
                promotion_categories = getattr(args, "promotion_categories_set", set())
                promotion_focuses = getattr(args, "promotion_focuses_set", set())
                row_category = str(row.get("prompt_category") or "").lower()
                row_focus = str(row.get("eval_focus") or "").lower()
                allow_promotion = (
                    (not promotion_categories or row_category in promotion_categories)
                    and (not promotion_focuses or row_focus in promotion_focuses)
                )
                pred = promote_or_draft_prediction(draft_answer, pred, allow_promotion, args.promotion_policy)
            if use_reasoning and args.reasoning_repair_retries > 0 and not has_substantive_reasoning(pred):
                preserved_answer = extract_answer(pred)
                for _ in range(args.reasoning_repair_retries):
                    repair_messages = messages[:-1] + [
                        {
                            "role": "user",
                            "content": turn_content(
                                repair_reasoning_prompt(str(prompt), preserved_answer, draft_answer=draft_answer),
                                images,
                                args.max_image_edge,
                                args.jpeg_quality,
                            ),
                        }
                    ]
                    repair_payload = dict(
                        model=args.model,
                        messages=repair_messages,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.reasoning_repair_max_tokens,
                    )
                    repair_resp = chat_completion(args.base_url, args.api_key, repair_payload, args.timeout)
                    repaired = preserve_model_reasoning_with_tags(completion_text(repair_resp["choices"][0]))
                    repaired_reasoning = extract_reasoning(repaired)
                    if not repaired_reasoning and not extract_tag(repaired, "answer"):
                        repaired_reasoning = repaired
                    repaired_reasoning = sanitize_embedded_tags(repaired_reasoning)
                    repaired_pred = f"<think>\n{repaired_reasoning.strip()}\n</think>\n<answer>\n{preserved_answer.strip()}\n</answer>"
                    if has_substantive_reasoning(repaired_pred):
                        pred = repaired_pred
                        break
            assert resp is not None
            latency = time.time() - started
            total_latency += latency
            answer_pred = extract_answer(pred) if use_reasoning else pred
            predictions.append(answer_pred)
            full_predictions.append(pred)
            reasoning_traces.append(extract_reasoning(pred) if use_reasoning else "")
            raw_responses.append({"main": resp, "repair": repair_resp} if repair_resp is not None else resp)
            messages.append({"role": "assistant", "content": pred})
            gold = golds[turn_idx] if turn_idx < len(golds) else ""
            score_text = pred if use_reasoning and args.score_full_response else answer_pred
            score = score_answer(score_text, gold)
            score["reasoning_enabled"] = float(use_reasoning)
            if use_reasoning:
                score["has_think"] = float(bool(extract_reasoning(pred)))
                score["has_answer"] = float(bool(extract_tag(pred, "answer")))
                score["substantive_reasoning"] = float(has_substantive_reasoning(pred))
                score["scored_full_response"] = float(args.score_full_response)
                score["answer_words"] = answer_word_count(answer_pred)
                score["reasoning_words"] = answer_word_count(extract_reasoning(pred))
            turn_scores.append(score)
        except Exception as exc:  # Keep full runs resumable even if one sample fails.
            latency = time.time() - started
            total_latency += latency
            errors.append({"turn": turn_idx, "error": repr(exc), "latency_sec": latency})
            predictions.append("")
            full_predictions.append("")
            reasoning_traces.append("")
            raw_responses.append(None)
            gold = golds[turn_idx] if turn_idx < len(golds) else ""
            turn_scores.append(score_answer("", gold))
            break

    sample_score = statistics.mean([s["heuristic_score"] for s in turn_scores]) if turn_scores else 0.0
    return {
        "id": row.get("id"),
        "created_at": utc_now(),
        "turncase": row.get("turncase"),
        "num_turns": row.get("num_turns"),
        "prompt_category": row.get("prompt_category"),
        "eval_focus": row.get("eval_focus"),
        "num_images": row.get("num_images"),
        "turn_prompts": prompts,
        "turn_golden_answers": golds,
        "turn_tool_trajectories": row.get("turn_tool_trajectories"),
        "rubrics_by_turn": row.get("rubrics_by_turn"),
        "reasoning_enabled": use_reasoning,
        "predictions": predictions,
        "full_predictions": full_predictions,
        "reasoning_traces": reasoning_traces,
        "tool_calling_enabled": bool(args.enable_tool_calling),
        "tool_traces": tool_traces,
        "summary_kl_events": summary_kl_events,
        "discarded_summary_events": discarded_summary_events,
        "draft_answer_first": bool(args.draft_answer_first),
        "turn_scores": turn_scores,
        "sample_heuristic_score": sample_score,
        "errors": errors,
        "latency_sec": total_latency,
        "raw_responses": raw_responses,
    }


def aggregate(predictions_path: Path, out_dir: Path) -> dict[str, Any]:
    rows = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    def mean(values: list[float]) -> float | None:
        return statistics.mean(values) if values else None

    by_fields: dict[str, dict[str, list[float]]] = {
        "turncase": defaultdict(list),
        "prompt_category": defaultdict(list),
        "eval_focus": defaultdict(list),
    }
    turn_scores = []
    sample_scores = []
    error_count = 0
    for row in rows:
        sample_scores.append(float(row.get("sample_heuristic_score") or 0.0))
        if row.get("errors"):
            error_count += 1
        for field, bucket in by_fields.items():
            bucket[str(row.get(field))].append(float(row.get("sample_heuristic_score") or 0.0))
        for score in row.get("turn_scores") or []:
            turn_scores.append(score)

    turn_metric_means = {}
    for key in ["exact", "fuzzy_token_set", "token_f1", "numeric_recall", "heuristic_score"]:
        vals = [float(s[key]) for s in turn_scores if s.get(key) is not None]
        turn_metric_means[key] = mean(vals)
    for key in [
        "reasoning_enabled",
        "has_think",
        "has_answer",
        "substantive_reasoning",
        "answer_words",
        "reasoning_words",
        "tool_calling_enabled",
        "num_tool_calls",
        "num_successful_tool_calls",
        "num_summary_calls",
        "num_summary_discards",
        "summary_kl_pruned_messages",
        "summary_kl_pruned_rough_text_tokens",
        "summary_kl_image_items_per_1k_text_delta",
    ]:
        vals = [float(s[key]) for s in turn_scores if s.get(key) is not None]
        if vals:
            turn_metric_means[key] = mean(vals)

    summary = {
        "created_at": utc_now(),
        "predictions_path": str(predictions_path),
        "num_samples": len(rows),
        "num_samples_with_errors": error_count,
        "sample_heuristic_score_mean": mean(sample_scores),
        "turn_metric_means": turn_metric_means,
        "by_turncase": {k: {"count": len(v), "sample_heuristic_score_mean": mean(v)} for k, v in by_fields["turncase"].items()},
        "by_prompt_category": {k: {"count": len(v), "sample_heuristic_score_mean": mean(v)} for k, v in by_fields["prompt_category"].items()},
        "by_eval_focus": {k: {"count": len(v), "sample_heuristic_score_mean": mean(v)} for k, v in by_fields["eval_focus"].items()},
        "evaluation_note": (
            "This is a deterministic local heuristic evaluation over golden answers "
            "(exact match, rapidfuzz token_set_ratio, token F1, and numeric recall when applicable), "
            "not an official rubric or LLM-judge score."
        ),
    }

    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "evaluation_by_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "turncase",
                "num_turns",
                "prompt_category",
                "eval_focus",
                "num_images",
                "sample_heuristic_score",
                "latency_sec",
                "num_errors",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.get("id"),
                    "turncase": row.get("turncase"),
                    "num_turns": row.get("num_turns"),
                    "prompt_category": row.get("prompt_category"),
                    "eval_focus": row.get("eval_focus"),
                    "num_images": row.get("num_images"),
                    "sample_heuristic_score": row.get("sample_heuristic_score"),
                    "latency_sec": row.get("latency_sec"),
                    "num_errors": len(row.get("errors") or []),
                }
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    data_from_env = os.environ.get("DATA_PATH", "").strip()
    parser.add_argument(
        "--data",
        default=data_from_env or None,
        required=not bool(data_from_env),
        help="VisualToolBench parquet file (or set DATA_PATH).",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--eligible-manifest",
        default="",
        help="JSON manifest with eligible_ids; every other row is excluded before execution and scoring.",
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8006/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="qwen3-vl-8b-tool")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1, help="Split eligible examples into this many stable, disjoint shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index for --num-shards.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--max-image-edge", type=int, default=1344, help="Resize image longest edge before sending; use 0 to disable.")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--direct-final-answer-contract",
        action="store_true",
        help="For no-tool runs, apply the tool loop's final-answer-only contract before the first response.",
    )
    parser.add_argument("--reasoning-format", action="store_true", help="Require <think>...</think><answer>...</answer> and score only the answer tag.")
    parser.add_argument(
        "--reasoning-mode",
        choices=["none", "all", "selected"],
        default="selected",
        help=(
            "Controls where reasoning is enabled when --reasoning-format is set. "
            "'none' disables reasoning, 'all' enables it for every sample, and 'selected' "
            "keeps the legacy category/focus filter."
        ),
    )
    parser.add_argument("--reasoning-retries", type=int, default=1, help="Regenerate a turn when the reasoning format is missing or superficial.")
    parser.add_argument("--reasoning-repair-retries", type=int, default=0, help="Generate an extra model-authored <think> for selected answers that still lack substantive reasoning.")
    parser.add_argument("--reasoning-repair-max-tokens", type=int, default=768, help="Max tokens for the missing-reasoning repair call.")
    parser.add_argument("--answer-completeness-retries", type=int, default=1, help="Regenerate a reasoning turn when <answer> appears too short or incomplete.")
    parser.add_argument("--draft-answer-first", action="store_true", help="Generate a direct draft answer first, then ask the model to verify/correct it with explicit reasoning.")
    parser.add_argument("--draft-predictions-path", default="", help="JSONL predictions file whose predictions are used as per-turn draft answers, keyed by sample id.")
    parser.add_argument("--conservative-finalize", action="store_true", help="Use the direct draft as final answer when the reasoning candidate is likely worse, while preserving model-generated reasoning text.")
    parser.add_argument("--prefer-draft-final", action="store_true", help="Always use the draft answer as final <answer> while preserving explicit reasoning in <think>.")
    parser.add_argument(
        "--fallback-to-draft-if-no-prune",
        action="store_true",
        help=(
            "Keep the supplied draft out of the rollout, but use it as final output "
            "when no summary/KL prune was committed."
        ),
    )
    parser.add_argument("--promote-reasoning-final", action="store_true", help="Use reasoning candidate when conservative no-GT promotion accepts it; otherwise keep the draft final answer.")
    parser.add_argument("--promotion-categories", default="", help="Comma-separated prompt_category values where candidate promotion is allowed. Empty means all categories.")
    parser.add_argument("--promotion-focuses", default="", help="Comma-separated eval_focus values where candidate promotion is allowed. Empty means all focuses.")
    parser.add_argument("--promotion-policy", choices=["strict", "lenient", "always", "refusal_more_nums", "refusal_recovery"], default="strict", help="No-GT candidate promotion policy.")
    parser.add_argument("--score-full-response", action="store_true", help="When using reasoning format, compute heuristic scores on the full tagged response for comparability with non-reasoning baselines.")
    parser.add_argument("--reasoning-categories", default="", help="Comma-separated prompt_category values where reasoning should be enabled. Empty means all categories when --reasoning-format is set.")
    parser.add_argument("--reasoning-focuses", default="", help="Comma-separated eval_focus values where reasoning should be enabled. Empty means all focuses when --reasoning-format is set.")
    parser.add_argument("--enable-tool-calling", action="store_true", help="Run each turn as a multi-step local tool-calling rollout.")
    parser.add_argument(
        "--visible-tool-reasoning",
        action="store_true",
        help="Require a visible evidence-grounded `<think>...</think>` block before every textual tool call.",
    )
    parser.add_argument(
        "--delete-visible-reasoning",
        action="store_true",
        help="Legacy alias for --reasoning-retention clear.",
    )
    parser.add_argument(
        "--reasoning-retention",
        choices=["preserve", "clear", "visual-evidence"],
        default="preserve",
        help=(
            "Per completed real-tool segment: preserve visible reasoning, clear it while retaining "
            "<tool_call>/<response>, or retain only structured visual evidence plus <tool_call>/<response>."
        ),
    )
    parser.add_argument(
        "--comparison-group",
        choices=["direct-no-tool", "tool-baseline", "delete-reasoning", "visual-evidence", "prune-v3", "prune-v4", "prune-v5", "prune-v5-safe", "prune-v5-hidden-safe", "prune-v5-roundwise", "prune-v5-nohint"],
        default="",
        help="Apply one canonical VTB comparison configuration; explicit flags are normalized to this group.",
    )
    parser.add_argument(
        "--tool-call-transport",
        choices=["text", "api"],
        default="text",
        help=(
            "How tool calls are requested from the model. 'text' prompts for <tool_call>{...}</tool_call> "
            "and executes parsed calls locally; use this when the vLLM server was not started with "
            "--enable-auto-tool-choice/--tool-call-parser. 'api' sends OpenAI tools/tool_choice fields."
        ),
    )
    parser.add_argument(
        "--tool-choice-mode",
        choices=["auto", "required-first", "required"],
        default="required-first",
        help=(
            "Tool choice policy for tool-calling rollout. 'required-first' forces the first model step "
            "of every turn to call a tool, then allows auto finalization; 'required' forces every "
            "model step before the budget is exhausted; 'auto' preserves legacy behavior."
        ),
    )
    parser.add_argument("--max-tool-steps", type=int, default=3, help="Maximum model/tool iterations per turn before forcing a final answer.")
    parser.add_argument("--tool-timeout", type=float, default=30.0, help="Timeout in seconds for one local Python tool execution.")
    parser.add_argument("--tool-output-max-chars", type=int, default=4000, help="Maximum stdout/stderr chars returned to the model per tool call.")
    parser.add_argument("--tool-artifacts-dir", default="", help="Directory for per-sample source and processed images. Defaults to OUT_DIR/tool_artifacts.")
    parser.add_argument("--enable-summary-tool", action="store_true", help="Expose summarize_the_task as a local memory tool and prompt the model to decide when to call it.")
    parser.add_argument(
        "--controller-summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate summary/KL probes outside the action rollout so its tool prompt matches baseline.",
    )
    parser.add_argument("--summary-nudge-after-tools", type=int, default=0, help="After this many real tool observations, add a prompt nudge asking the model to decide whether to call summarize_the_task. 0 disables nudges.")
    parser.add_argument("--max-summary-tool-steps", type=int, default=1, help="Maximum summarize_the_task calls per turn. Summary calls do not count against --max-tool-steps.")
    parser.add_argument("--summary-kl-prune", action="store_true", help="After summarize_the_task calls, run reverse-KL coverage scoring and prune covered prior assistant reasoning text.")
    parser.add_argument("--kl-base-url", default="", help="Optional OpenAI-compatible endpoint used only for KL token-logprob scoring. Defaults to --base-url.")
    parser.add_argument("--kl-model", default="", help="Served model name used by the KL scorer endpoint. Defaults to --model.")
    parser.add_argument("--summary-kl-top-k", type=int, default=20, help="Top-K support used for the reverse-KL approximation.")
    parser.add_argument("--summary-kl-threshold", type=float, default=0.2, help="Normalized reverse-KL threshold for retaining a token.")
    parser.add_argument("--summary-kl-min-high-tokens", type=int, default=2, help="Keep a segment only when at least this many tokens exceed the KL threshold.")
    parser.add_argument("--summary-kl-apply-prune", action=argparse.BooleanOptionalAction, default=True, help="Apply KL pruning when --summary-kl-prune is enabled. Use --no-summary-kl-apply-prune to log only.")
    parser.add_argument("--summary-kl-keep-recent-assistant-steps", type=int, default=0, help="Keep this many most recent prior assistant messages even if KL marks them prunable.")
    parser.add_argument(
        "--summary-gate-min-segments",
        "--summary-gate-min-real-tools",
        dest="summary_gate_min_segments",
        type=int,
        default=int(
            os.environ.get(
                "MNMS_SUMMARY_GATE_MIN_CANDIDATES",
                os.environ.get("MNMS_SUMMARY_GATE_MIN_REAL_TOOLS", "2"),
            )
        ),
        help=(
            "Minimum completed reasoning/action segments before a gated summary probe is allowed. "
            "The legacy --summary-gate-min-real-tools name is accepted as an alias; tool success "
            "or tool type is not required."
        ),
    )
    parser.add_argument(
        "--summary-gate-min-reasoning-tokens",
        type=int,
        default=int(os.environ.get("MNMS_SUMMARY_GATE_MIN_REASONING_TOKENS", "80")),
        help="Minimum accumulated unpruned reasoning tokens before a gated summary probe is allowed.",
    )
    parser.add_argument(
        "--summary-gate-min-candidates",
        type=int,
        default=int(os.environ.get("MNMS_SUMMARY_GATE_MIN_CANDIDATES", "2")),
        help="Minimum active reasoning candidate segments before a gated summary probe is allowed.",
    )
    parser.add_argument(
        "--summary-post-verify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional V5 ablation: require a remaining real-tool step to verify critical "
            "numeric/choice/entity visual facts after consuming the temporary summary. "
            "Disabled by default because it materially increases latency."
        ),
    )
    parser.add_argument(
        "--summary-inject-next-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Inject an accepted controller summary into the next action request once. "
            "Disable for the pruning-only ablation, where summary is used by teacher KL "
            "but never appears in an action-model prompt or persistent trace."
        ),
    )
    parser.add_argument(
        "--summary-prune-policy",
        choices=["clear", "direct-delete", "v2", "v3", "v4", "v5", "v5-roundwise", "vtc-visual-evidence"],
        default="clear",
        help=(
            "How to edit assistant messages after summary. 'clear' prunes only KL-selected messages; "
            "'direct-delete' clears every prior assistant message once a summary is called; "
            "'v2' preserves only the action block; 'v3' preserves structured visual evidence, "
            "visual anchors, and the action block. 'v4' uses v3 for high-KL segments and action-only v2 for low-KL segments. "
            "'vtc-visual-evidence' is a legacy alias for v3."
        ),
    )
    parser.add_argument("--summary-gated-probe", action="store_true", help="For summarize_the_task calls, keep the summary observation only if KL finds at least one assistant message to prune; otherwise discard it.")
    parser.add_argument(
        "--summary-every-round",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require a summary/KL deletion decision after every completed tool reasoning/action round.",
    )
    parser.add_argument("--summary-net-token-gate", action="store_true", help="Probe summarize_the_task on a message copy and keep it only when estimated net context text-token savings meet --summary-net-token-min-savings.")
    parser.add_argument("--summary-net-token-min-savings", type=int, default=64, help="Minimum estimated net rough text-token reduction required to keep a summary probe and apply KL pruning.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--num-shards must be >= 1 and --shard-index must be in [0, num_shards).")
    if args.comparison_group:
        # Canonical controls for the V3 study.  These deliberately set the
        # full behavior rather than relying on a stale mix of legacy flags.
        if args.comparison_group == "direct-no-tool":
            args.enable_tool_calling = False
            args.direct_final_answer_contract = True
        else:
            args.enable_tool_calling = True
            args.visible_tool_reasoning = True
            args.direct_final_answer_contract = False
        if args.comparison_group == "tool-baseline":
            args.enable_summary_tool = False
            args.summary_kl_prune = False
            args.reasoning_retention = "preserve"
        elif args.comparison_group == "delete-reasoning":
            args.enable_summary_tool = False
            args.summary_kl_prune = False
            args.reasoning_retention = "clear"
        elif args.comparison_group == "visual-evidence":
            args.enable_summary_tool = False
            args.summary_kl_prune = False
            args.reasoning_retention = "visual-evidence"
        elif args.comparison_group == "prune-v3":
            args.enable_summary_tool = True
            args.summary_kl_prune = True
            # A summary is committed only when its KL decision actually
            # removes at least one prior reasoning segment.
            args.summary_gated_probe = True
            args.summary_prune_policy = "v3"
            args.reasoning_retention = "preserve"
            if args.summary_nudge_after_tools <= 0:
                args.summary_nudge_after_tools = 2
        elif args.comparison_group == "prune-v4":
            args.enable_summary_tool = True
            args.summary_kl_prune = True
            args.summary_gated_probe = True
            args.summary_prune_policy = "v4"
            args.reasoning_retention = "preserve"
            if args.summary_nudge_after_tools <= 0:
                args.summary_nudge_after_tools = 2
        elif args.comparison_group == "prune-v5":
            # Hidden task-state controller: actions retain the baseline
            # format, while each eligible post-gate tool round reconstructs
            # the historical reasoning. The checkpoint is temporary only.
            args.enable_summary_tool = False
            args.controller_summary = True
            args.summary_kl_prune = True
            args.summary_gated_probe = True
            args.summary_prune_policy = "v5"
            args.reasoning_retention = "preserve"
            args.summary_nudge_after_tools = 0
            args.summary_kl_keep_recent_assistant_steps = 0
            args.summary_net_token_gate = False
        elif args.comparison_group == "prune-v5-safe":
            args.enable_summary_tool = False
            args.controller_summary = True
            args.summary_kl_prune = True
            args.summary_gated_probe = True
            args.summary_prune_policy = "v5"
            args.reasoning_retention = "preserve"
            args.summary_nudge_after_tools = 0
            # Preserve the newest decision state and commit only a genuinely
            # useful net reduction.  Together with the temporary nudge above,
            # this is the conservative quality-first V5 variant.
            args.summary_kl_keep_recent_assistant_steps = max(
                2, int(args.summary_kl_keep_recent_assistant_steps)
            )
            args.summary_net_token_gate = True
            args.summary_net_token_min_savings = max(
                64, int(args.summary_net_token_min_savings)
            )
        elif args.comparison_group == "prune-v5-hidden-safe":
            # Keep the action rollout identical to tool-baseline.  The same
            # model generates task state in a hidden controller request only
            # after the normal two-segment gate.
            args.enable_summary_tool = False
            args.controller_summary = True
            args.summary_kl_prune = True
            args.summary_gated_probe = True
            args.summary_prune_policy = "v5"
            args.reasoning_retention = "preserve"
            args.summary_nudge_after_tools = 0
            args.summary_kl_keep_recent_assistant_steps = max(
                2, int(args.summary_kl_keep_recent_assistant_steps)
            )
            args.summary_net_token_gate = True
            args.summary_net_token_min_savings = max(
                64, int(args.summary_net_token_min_savings)
            )
        elif args.comparison_group in {"prune-v5-roundwise", "prune-v5-nohint"}:
            # Accuracy-first hidden-controller V5.  Once two reasoning/action
            # segments exist, it generates and verifies task state out of
            # band, then compresses only older eligible reasoning.  The newest
            # decision segment remains intact and the summary never becomes a
            # persistent tool-call/message in the rollout trace.
            args.enable_summary_tool = False
            args.controller_summary = True
            args.summary_kl_prune = True
            args.summary_gated_probe = True
            args.summary_prune_policy = "v5"
            # Visual is the compression floor: every completed reasoning
            # segment is immediately reconstructed as grounded visual
            # evidence. Summary-KL may then compress low-KL segments further
            # to action-only, while high-KL segments keep that evidence.
            # Thus V5 never retains more reasoning-side context than Visual,
            # including when a summary is gated or discarded.
            args.reasoning_retention = "visual-evidence"
            # Do not use the visible summarize_the_task forcing path.  The
            # controller hook already runs after each real tool call.
            args.summary_every_round = False
            args.summary_gate_min_segments = max(2, int(args.summary_gate_min_segments))
            args.summary_gate_min_candidates = max(2, int(args.summary_gate_min_candidates))
            args.summary_nudge_after_tools = 0
            args.summary_kl_keep_recent_assistant_steps = max(
                1, int(args.summary_kl_keep_recent_assistant_steps)
            )
            # A segment remains Visual only when at least five tokens carry
            # normalized KL above 0.2; otherwise the summary covers it and its
            # reasoning-side evidence can be removed.  This is a modestly
            # less conservative setting than the previous minimum of two.
            args.summary_kl_min_high_tokens = max(
                5, int(args.summary_kl_min_high_tokens)
            )
            args.summary_net_token_gate = False
            if args.comparison_group == "prune-v5-nohint":
                # Pure pruning-controller ablation: the verified summary is
                # available to teacher KL only.  The action model receives
                # the reconstructed history, never the summary itself.
                args.summary_inject_next_step = False
    if args.delete_visible_reasoning:
        args.reasoning_retention = "clear"
    args.max_image_edge = args.max_image_edge or None
    args.reasoning_categories_set = parse_csv_set(args.reasoning_categories)
    args.reasoning_focuses_set = parse_csv_set(args.reasoning_focuses)
    args.promotion_categories_set = parse_csv_set(args.promotion_categories)
    args.promotion_focuses_set = parse_csv_set(args.promotion_focuses)
    args.draft_predictions = load_draft_predictions(args.draft_predictions_path)
    args.eligible_ids = None
    if args.eligible_manifest:
        manifest = json.loads(Path(args.eligible_manifest).read_text(encoding="utf-8"))
        args.eligible_ids = {str(item) for item in manifest.get("eligible_ids", [])}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.tool_artifacts_dir = Path(args.tool_artifacts_dir) if args.tool_artifacts_dir else out_dir / "tool_artifacts"
    if args.enable_tool_calling:
        args.tool_artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    skipped_path = out_dir / "skipped_runtime.jsonl"

    metadata = {
        "created_at": utc_now(),
        "data": args.data,
        "eligible_manifest": args.eligible_manifest or None,
        "eligible_count": len(args.eligible_ids) if args.eligible_ids is not None else None,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "direct_final_answer_contract": args.direct_final_answer_contract,
        "max_image_edge": args.max_image_edge,
        "jpeg_quality": args.jpeg_quality,
        "reasoning_format": args.reasoning_format,
        "reasoning_retries": args.reasoning_retries,
        "reasoning_repair_retries": args.reasoning_repair_retries,
        "reasoning_repair_max_tokens": args.reasoning_repair_max_tokens,
        "answer_completeness_retries": args.answer_completeness_retries,
        "draft_answer_first": args.draft_answer_first,
        "draft_predictions_path": args.draft_predictions_path,
        "num_draft_prediction_samples": len(args.draft_predictions),
        "conservative_finalize": args.conservative_finalize,
        "prefer_draft_final": args.prefer_draft_final,
        "fallback_to_draft_if_no_prune": args.fallback_to_draft_if_no_prune,
        "promote_reasoning_final": args.promote_reasoning_final,
        "promotion_categories": sorted(args.promotion_categories_set),
        "promotion_focuses": sorted(args.promotion_focuses_set),
        "promotion_policy": args.promotion_policy,
        "score_full_response": args.score_full_response,
        "reasoning_mode": args.reasoning_mode,
        "reasoning_categories": sorted(args.reasoning_categories_set),
        "reasoning_focuses": sorted(args.reasoning_focuses_set),
        "enable_tool_calling": args.enable_tool_calling,
        "visible_tool_reasoning": args.visible_tool_reasoning,
        "delete_visible_reasoning": args.delete_visible_reasoning,
        "reasoning_retention": args.reasoning_retention,
        "comparison_group": args.comparison_group or None,
        "tool_call_transport": args.tool_call_transport,
        "tool_choice_mode": args.tool_choice_mode,
        "max_tool_steps": args.max_tool_steps,
        "tool_timeout": args.tool_timeout,
        "tool_output_max_chars": args.tool_output_max_chars,
        "tool_artifacts_dir": str(args.tool_artifacts_dir),
        "enable_summary_tool": args.enable_summary_tool,
        "controller_summary": args.controller_summary,
        "summary_nudge_after_tools": args.summary_nudge_after_tools,
        "max_summary_tool_steps": args.max_summary_tool_steps,
        "summary_kl_prune": args.summary_kl_prune,
        "summary_kl_top_k": args.summary_kl_top_k,
        "summary_kl_threshold": args.summary_kl_threshold,
        "summary_kl_min_high_tokens": args.summary_kl_min_high_tokens,
        "summary_kl_apply_prune": args.summary_kl_apply_prune,
        "summary_kl_keep_recent_assistant_steps": args.summary_kl_keep_recent_assistant_steps,
        "summary_gate_min_segments": args.summary_gate_min_segments,
        "summary_gate_min_reasoning_tokens": args.summary_gate_min_reasoning_tokens,
        "summary_gate_min_candidates": args.summary_gate_min_candidates,
        "summary_post_verify": args.summary_post_verify,
        "summary_inject_next_step": args.summary_inject_next_step,
        "summary_every_round": args.summary_every_round,
        "summary_prune_policy": args.summary_prune_policy,
        "summary_gated_probe": args.summary_gated_probe,
        "tool_schema": tool_schema_for_args(args) if args.enable_tool_calling else [],
    }
    (out_dir / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.aggregate_only:
        summary = aggregate(predictions_path, out_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    parquet_path = Path(args.data)
    pf = pq.ParquetFile(parquet_path)
    done = load_done_ids(predictions_path) if args.resume else set()
    if args.resume:
        done.update(load_done_ids(skipped_path))
    total = min(args.limit or pf.metadata.num_rows, pf.metadata.num_rows)
    remaining_for_progress = max(total - len(done), 0) if args.resume else total

    processed = 0
    seen = 0

    with predictions_path.open("a" if args.resume else "w", encoding="utf-8") as out, skipped_path.open("a" if args.resume else "w", encoding="utf-8") as skipped:
        with tqdm(total=remaining_for_progress, desc="VisualToolBench", dynamic_ncols=True) as pbar:
            for rg_idx in range(pf.metadata.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=TEXT_COLUMNS)
                for row in table.to_pylist():
                    if seen >= total:
                        break
                    seen += 1
                    row_id = str(row.get("id"))
                    if args.eligible_ids is not None and row_id not in args.eligible_ids:
                        continue
                    shard_bucket = int.from_bytes(hashlib.sha1(row_id.encode("utf-8")).digest()[:8], "big") % args.num_shards
                    if shard_bucket != args.shard_index:
                        continue
                    if row_id in done:
                        continue
                    result = run_one(args, row)
                    if result.get("errors"):
                        # No fallback output is permitted.  Keep the original
                        # error trace for audit, but exclude this row from the
                        # final metric denominator.
                        result["skip_reason"] = "runtime_tool_or_protocol_error"
                        skipped.write(json.dumps(result, ensure_ascii=False) + "\n")
                        skipped.flush()
                        processed += 1
                        pbar.update(1)
                        continue
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                    processed += 1
                    pbar.update(1)
                if seen >= total:
                    break

    summary = aggregate(predictions_path, out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {processed} new predictions to {predictions_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
