#!/usr/bin/env python3
"""Run AdaReasoner benchmark datasets with a direct model-driven tool loop.

This intentionally bypasses AdaReasoner's learned/planning algorithm. It keeps
the benchmark loading/scoring path, lets the evaluated model emit tool calls in
the repository's XML-like protocol, executes local tools, and scores the final
answer.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import io
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from PIL import Image


def _bypass_proxy_for_local_services() -> None:
    """Keep localhost vLLM/controller traffic out of a global HTTP proxy."""
    required = ("127.0.0.1", "localhost", "::1")
    for key in ("no_proxy", "NO_PROXY"):
        current = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
        for host in required:
            if host not in current:
                current.append(host)
        os.environ[key] = ",".join(current)


_bypass_proxy_for_local_services()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_no_tool_benchmarks import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    TASK_SPECS,
    load_items,
    score_item,
)
from tool_server.tool_workers.offline_workers import (  # noqa: E402
    get_available_tools,
    get_tool_generate_fn,
    get_tool_instruction,
)
from tool_server.tool_workers.tool_manager.base_manager import ToolManager  # noqa: E402
from tool_server.utils.prompts import tool_desc_dict  # noqa: E402
from tool_server.utils.utils import base64_to_pil, pil_to_base64  # noqa: E402


# Tools are deliberately benchmark-specific.  Do not offer Jigsaw navigation
# tools to VStar (or vice versa): they induce invalid, repetitive trajectories.
JIGSAW_TOOLS = [
    "Point",
    "Draw2DPath",
    "AStarWithPixelCoordinate",
    "DetectBlackArea",
    "InsertImage",
    "OCR",
    "Crop",
    "summarize_the_task",
]
VSTAR_TOOLS = [
    "Point",
    "OCR",
    "SegmentRegionAroundPoint",
    "DrawLine",
    "GroundingDINO",
    "DrawShape",
    "GetBarInfo",
    "GetSubplotInfo",
    "HighlightBox",
    "MaskBox",
    "Crop",
    "summarize_the_task",
]
# Backwards-compatible alias for Jigsaw-only callers.
DEFAULT_TOOLS = JIGSAW_TOOLS
JIGSAW_ONLY_TOOLS = set(JIGSAW_TOOLS) - {"Point", "OCR", "Crop", "summarize_the_task"}
VSTAR_ONLY_TOOLS = set(VSTAR_TOOLS) - {"Point", "OCR", "Crop", "summarize_the_task"}


def load_materialized_items(task: str, manifest_path: str) -> list[dict[str, Any]]:
    """Load a disk-backed benchmark manifest without duplicating a HF image dataset per shard."""
    items: list[dict[str, Any]] = []
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("task") != task:
                continue
            image_paths = record.pop("image_paths")
            record["images"] = [Image.open(path).convert("RGB") for path in image_paths]
            items.append(record)
    return items

SUMMARY_TOOL_NAME = "summarize_the_task"
# Match the no-tool evaluator: original benchmark images must reach the model
# at native resolution and comparable JPEG fidelity. A 320px / quality-40 cap
# destroys the local edge/text evidence that tool trajectories inspect. Set a
# positive max-side or custom quality only for an explicit ablation.
MODEL_IMAGE_MAX_SIDE = int(os.environ.get("BENCHMARK_MODEL_IMAGE_MAX_SIDE", "0"))
MODEL_IMAGE_JPEG_QUALITY = int(os.environ.get("BENCHMARK_MODEL_IMAGE_JPEG_QUALITY", "75"))
MAX_GENERATED_IMAGES = int(os.environ.get("BENCHMARK_MAX_GENERATED_IMAGES", "0"))

SUMMARY_TOOL_INSTRUCTION = """
Tool Name: summarize_the_task
Description: Summarize the current visual state of the benchmark item. Use this
tool when it would help you keep track of what has already been visually
grounded, which tools have already been used, the current task state, useful
observations, and what decision should be made next. This tool is for agent
memory/state only and does not inspect or edit images. Prefer compact visual
evidence over long natural-language reasoning.
Parameters JSON schema:
{
  "reasoning_so_far": "brief summary of your non-redundant reasoning so far",
  "completed_steps": ["steps already completed"],
  "current_state": "where the current task stands now",
  "visual_evidence": {
    "coordinates": ["start=(x,y), goal=(x,y), obstacle=(x,y), etc."],
    "bboxes": ["object/text/candidate bbox as [x_min,y_min,x_max,y_max]"],
    "crop_ids": ["img_2 = crop of target region, etc."],
    "ocr_text": ["verbatim OCR text snippets and their regions"],
    "astar_outputs": ["AStar path/result or path failure reason"],
    "candidate_differences": ["local edge/texture/color/spatial differences"]
  },
  "key_observations": ["important observations from the image or tools"],
  "uncertainties": ["what visual evidence is still missing"],
  "next_decision": "what you should decide or do next"
}
Returns: a compact visual-state summary that you can use in the following turn.
"""

SUMMARY_CHECKPOINT_INSTRUCTION = """
Tool Name: summarize_the_task
Description: Store a very short memory checkpoint. This tool is memory/state
only and does not inspect images. Use it to keep verified visual/tool evidence,
not to repeat long reasoning.
Parameters JSON schema:
{
  "facts": ["1-3 short verified facts only"],
  "tool_evidence": ["tool result, crop id, OCR text, coordinate, bbox, or path"],
  "open_risks": ["missing or uncertain visual evidence"],
  "next_step": "next tool or final decision"
}
Rules: keep every string short; omit unsupported guesses; do not include prose
rationales.
Returns: a compact checkpoint for the next turn.
"""


OPEN_THINK_DESC = "less-than sign + word think + greater-than sign"
CLOSE_THINK_DESC = "less-than sign + slash + word think + greater-than sign"


TOOL_LOOP_PROMPT = """You are evaluating visual reasoning benchmarks with tools.

Act as an autonomous visual reasoning agent. Work through the problem yourself,
decide which tools are useful, and revise your plan after each observation.

For each assistant turn, write your own reasoning process first. Put that
reasoning inside the literal tag formed by {open_think_desc} and the literal
closing tag formed by {close_think_desc}. Then either call one tool or give the
final answer.

If you call a tool, output exactly one <tool_call>...</tool_call> block after
your reasoning. The content inside <tool_call> MUST be a valid JSON object. Do
not write pseudo-code like OCR(img_1).

When you are done, output one response block containing only your actual answer
after your reasoning.
Do not copy a canned rationale; generate the reasoning from the current question,
image, and tool observations.

{summary_requirement}

Tool-call format:
<tool_call>
{{"name": "ToolName", "parameters": {{"image": "img_1"}}}}
</tool_call>

Images are referenced as img_1, img_2, ... in the order shown. If a tool returns
an edited image, it becomes the next image id. Use absolute pixel coordinates.
Put the final answer inside \\boxed{{}} when possible.

Available tools:
{tool_list}

{summary_decision_requirement}
"""

COMPACT_TOOL_LOOP_PROMPT = """Visual reasoning agent with tools.
Use your own reasoning in <think>...</think>, then exactly one <tool_call> JSON
block or one response block containing only your actual answer. Use image ids img_1, img_2, ...
{summary_requirement}
Final answers should use \\boxed{{}} when possible.

Available tools:
{tool_list}
"""

NEMOTRON_TOOL_LOOP_PROMPT = """You are in the tool-enabled visual-agent condition.
Use a real tool before the final answer when the task depends on local visual
detail; do not replace a useful tool observation with a guess. In particular,
Jigsaw tasks require inspecting the missing region with DetectBlackArea and/or
testing a candidate with InsertImage. VStar tasks require a targeted Crop when
the queried attribute is not already large and unambiguous. A direct final is
allowed only when the relevant evidence is plainly visible at the original
scale.

Every reply must use exactly one of these two plain-text formats.

REASONING: one short, image-specific reason for the next action
ACTION_JSON: {{"name": "ToolName", "parameters": {{"image": "img_1"}}}}

REASONING: one short, image-specific justification
FINAL: A

`FINAL` may contain an option letter, number, path, or short answer. Do not use
XML tags, Markdown code fences, Python, or pseudo tool calls. Image ids are
img_1, img_2, ...; edited images receive the next id. Coordinates are absolute
pixels. For Jigsaw, compare real InsertImage composites when visual matching is
uncertain. For a tiny VStar attribute, use a targeted Crop when needed.

Available real tools:
{tool_list}
"""


def render_tool_loop_prompt(
    tool_list: str,
    require_summary: bool = False,
    protocol: str = "xml",
) -> str:
    if protocol == "nemotron":
        return NEMOTRON_TOOL_LOOP_PROMPT.format(tool_list=tool_list)
    template = COMPACT_TOOL_LOOP_PROMPT if os.environ.get("BENCHMARK_COMPACT_INITIAL_PROMPT") == "1" else TOOL_LOOP_PROMPT
    if require_summary:
        summary_requirement = (
            "If you have not yet called summarize_the_task in this conversation, call "
            "summarize_the_task before your final answer. You may choose the best moment to do "
            "that, but do it at least once so your next decision can use a compact state summary."
        )
        summary_decision_requirement = (
            "You should decide when to summarize the current state and information of the task with "
            "the summarize_the_task tool and call it at least once in the conversation. "
            "summarize_the_task is a memory tool only: it does not count as a benchmark tool call "
            "and cannot inspect images."
        )
    else:
        summary_requirement = ""
        summary_decision_requirement = ""
    return template.format(
        open_think_desc=OPEN_THINK_DESC,
        close_think_desc=CLOSE_THINK_DESC,
        tool_list=tool_list,
        summary_requirement=summary_requirement,
        summary_decision_requirement=summary_decision_requirement,
    )


def jigsaw_geometry_instruction(images: list[Image.Image]) -> str:
    """Return the coordinate contract needed by JigsawBlink tool trajectories.

    JigsawBlink candidates are frequently not the same raster size as the
    incomplete canvas.  Tool coordinates are *per-image absolute pixels*;
    consequently a hole box found on img_1 cannot be copied verbatim to a
    candidate Crop.  This was previously left implicit, which made a failed
    out-of-bounds crop look like evidence against a valid candidate.
    """
    dimensions = ", ".join(
        f"img_{index + 1}={image.width}x{image.height}" for index, image in enumerate(images)
    )
    return (
        "\nJigsawBlink geometry and tool policy:\n"
        f"- Image dimensions (width x height): {dimensions}.\n"
        "- Every Crop/InsertImage coordinate is absolute in the target image, not shared across images. "
        "If a box [x1,y1,x2,y2] was detected on img_1 (W,H), map it to a candidate image (w,h) as "
        "[round(x1*w/W), round(y1*h/H), round(x2*w/W), round(y2*h/H)] before cropping that candidate.\n"
        "- Do not treat an out-of-bounds Crop failure as evidence that a candidate is wrong; correct the "
        "coordinates or compare it using InsertImage instead.\n"
        "- To validate a candidate, use the real InsertImage tool with base_image=img_1, "
        "image_to_insert=<candidate image>, coordinates=<the hole box on img_1>, resize=true. "
        "Inspect the returned composite for edge/texture continuity before deciding.\n"
    )


def image_to_data_url(image: Image.Image) -> str:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    max_side = MODEL_IMAGE_MAX_SIDE
    if max_side > 0 and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=MODEL_IMAGE_JPEG_QUALITY, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for key, value in result.items():
        if key == "edited_image":
            compact[key] = "<image omitted>"
        else:
            compact[key] = value
    return compact


def parse_tag(text: str, tag: str) -> str | None:
    if not isinstance(text, str):
        text = str(text)
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", flags=re.I | re.S)
    return match.group(1).strip() if match else None


def parse_reasoning(text: str) -> str | None:
    return parse_tag(text, "think")


def extract_reasoning(text: str) -> tuple[str | None, str]:
    tagged = parse_reasoning(text)
    if tagged:
        return tagged, "model_generated_tag"
    if not isinstance(text, str):
        text = str(text)
    nemotron_reasoning = re.search(
        r"(?:^|\n)\s*REASONING\s*:\s*(.*?)(?=\n\s*(?:ACTION_JSON|FINAL)\s*:|$)",
        text,
        flags=re.I | re.S,
    )
    if nemotron_reasoning and nemotron_reasoning.group(1).strip():
        return nemotron_reasoning.group(1).strip(), "nemotron_reasoning"
    prefix = re.split(r"<tool_call>|<response>", text, maxsplit=1, flags=re.I | re.S)[0].strip()
    if prefix:
        return prefix, "raw_model_prose"
    return None, "missing"


def normalize_tool_call_data(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    data = dict(data)
    if "name" not in data and "tool_name" in data:
        data["name"] = data.get("tool_name")
    if "name" not in data and "tool" in data:
        data["name"] = data.get("tool")
    if "parameters" not in data and "name" in data:
        data["parameters"] = {k: v for k, v in data.items() if k not in {"name", "tool", "tool_name"}}
    params = data.get("parameters")
    if not isinstance(params, dict):
        params = {}
    if "image" not in params and "image_id" in params:
        params["image"] = params.get("image_id")
    if "coordinates" not in params and "bbox" in params:
        params["coordinates"] = params.get("bbox")
    data["parameters"] = params
    if not data.get("name"):
        return None
    return data


def parse_json_tool_candidate(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
        raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        raw = match.group(0)
    try:
        return normalize_tool_call_data(json.loads(raw))
    except json.JSONDecodeError:
        return None


def parse_function_tool_candidate(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    match = re.search(
        r"\b(Point|OCR|Crop|AStarWithPixelCoordinate|DetectBlackArea|Draw2DPath|InsertImage|summarize_the_task)\s*\((.*?)\)",
        text,
        flags=re.I | re.S,
    )
    if not match:
        bare = re.search(r"\b(summarize_the_task)\b", text, flags=re.I)
        if bare:
            return {"name": SUMMARY_TOOL_NAME, "parameters": {}}
        return None
    name = match.group(1)
    if name.lower() == SUMMARY_TOOL_NAME.lower():
        return {"name": SUMMARY_TOOL_NAME, "parameters": {}}
    arg_blob = (match.group(2) or "").strip()
    params: dict[str, Any] = {}
    image_match = re.search(r"\b(?:image|image_id)\s*=\s*[\"']?(img_\d+)[\"']?", arg_blob, flags=re.I)
    if image_match:
        params["image"] = image_match.group(1)
    bare_image = re.search(r"\b(img_\d+)\b", arg_blob, flags=re.I)
    image_id = params.get("image") or (bare_image.group(1) if bare_image else None)
    if image_id:
        params["image"] = image_id
    bbox_match = re.search(r"\b(?:bbox|coordinates)\s*=\s*(\[[^\]]+\])", arg_blob, flags=re.I | re.S)
    if bbox_match:
        params["coordinates"] = bbox_match.group(1).strip()
    point_match = re.search(r"\bpoint\s*=\s*(\[[^\]]+\])", arg_blob, flags=re.I | re.S)
    if point_match:
        params["point"] = point_match.group(1).strip()
    xyxy = {
        key: re.search(rf"\b{key}\s*=\s*(-?\d+(?:\.\d+)?)", arg_blob, flags=re.I)
        for key in ("x_min", "y_min", "x_max", "y_max")
    }
    if all(xyxy.values()) and "coordinates" not in params:
        params["coordinates"] = [float(xyxy[key].group(1)) for key in ("x_min", "y_min", "x_max", "y_max")]
    xy = {
        key: re.search(rf"\b{key}\s*=\s*(-?\d+(?:\.\d+)?)", arg_blob, flags=re.I)
        for key in ("x", "y")
    }
    if all(xy.values()) and "point" not in params:
        params["point"] = [float(xy["x"].group(1)), float(xy["y"].group(1))]
    arg_text = re.sub(r"\b(?:image|image_id)\s*=\s*[\"']?img_\d+[\"']?", "", arg_blob, flags=re.I).strip(" ,")
    if arg_text:
        if name.lower() == "crop":
            params.setdefault("coordinates", arg_text)
        elif name.lower() == "point":
            params.setdefault("description", arg_text)
        else:
            params.setdefault("argument", arg_text)
    return {"name": name, "parameters": params}


def parse_tool_call(text: str) -> dict[str, Any] | None:
    raw = parse_tag(text, "tool_call")
    if raw is None and isinstance(text, str):
        action_match = re.search(
            r"(?:^|\n)\s*ACTION_JSON\s*:\s*(\{.*\})\s*$",
            text,
            flags=re.I | re.S,
        )
        if action_match:
            raw = action_match.group(1).strip()
    if raw is None and isinstance(text, str):
        match = re.search(r"<tool_call>\s*(.*?)(?:</tool_call>|<tool_call>|$)", text, flags=re.I | re.S)
        if match:
            raw = match.group(1).strip()
    if raw is None and isinstance(text, str):
        candidate = parse_function_tool_candidate(text)
        if candidate:
            return candidate
    if raw is None and isinstance(text, str):
        code_match = re.search(r"<tool_code>\s*([A-Za-z_][A-Za-z0-9_]*)\s*</tool_code>\s*```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S)
        if code_match:
            name = code_match.group(1)
            raw = code_match.group(2).strip()
            try:
                params = json.loads(raw)
            except json.JSONDecodeError:
                params = {}
            if isinstance(params, dict):
                if "image" not in params and "image_id" in params:
                    params["image"] = params.get("image_id")
                if "coordinates" not in params and "bbox" in params:
                    params["coordinates"] = params.get("bbox")
                return {"name": name, "parameters": params}
    if raw is None and isinstance(text, str):
        for fence_match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.I | re.S):
            candidate = parse_json_tool_candidate(fence_match.group(1))
            if candidate and str(candidate.get("name", "")) in DEFAULT_TOOLS:
                return candidate
    if not raw:
        return None
    candidate = parse_json_tool_candidate(raw)
    if candidate:
        return candidate
    return parse_function_tool_candidate(raw)


def parse_final_response(text: str) -> str | None:
    """Parse XML, Nemotron plain-text, boxed, and bare option finals."""
    if not isinstance(text, str):
        text = str(text)
    tagged = parse_tag(text, "response")
    if tagged is not None:
        return tagged
    final_match = re.search(
        r"(?:^|\n)\s*FINAL\s*:\s*(.+?)\s*$",
        text,
        flags=re.I | re.S,
    )
    if final_match:
        return final_match.group(1).strip()
    boxed = re.search(r"\\boxed\{([^{}]{1,200})\}", text)
    if boxed:
        return "\\boxed{" + boxed.group(1).strip() + "}"
    compact = text.strip()
    bare_option = re.fullmatch(r"(?:\(?\s*([A-Da-d])\s*\)?|option\s+([A-Da-d]))[.!]?", compact)
    if bare_option:
        return (bare_option.group(1) or bare_option.group(2)).upper()
    return None


def extract_action_block(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for tag in ("tool_call", "response"):
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", flags=re.I | re.S)
        if match:
            return f"<{tag}>\n{match.group(1).strip()}\n</{tag}>"
    action_match = re.search(
        r"(?:^|\n)\s*ACTION_JSON\s*:\s*(\{.*\})\s*$",
        text,
        flags=re.I | re.S,
    )
    if action_match:
        return "ACTION_JSON: " + action_match.group(1).strip()
    final_match = re.search(
        r"(?:^|\n)\s*FINAL\s*:\s*(.+?)\s*$",
        text,
        flags=re.I | re.S,
    )
    if final_match:
        return "FINAL: " + final_match.group(1).strip()
    return ""


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def pruning_fraction_metrics(
    pruned_token_counts: list[int],
    before_token_counts: list[int],
) -> dict[str, float | int]:
    pairs = list(zip(pruned_token_counts, before_token_counts))
    nonempty_ratios = [
        min(1.0, max(0.0, pruned / before))
        for pruned, before in pairs
        if before > 0
    ]
    all_ratios = [
        min(1.0, max(0.0, pruned / before)) if before > 0 else 0.0
        for pruned, before in pairs
    ]
    total_before = sum(max(0, before) for _, before in pairs)
    total_pruned = sum(
        min(max(0, pruned), max(0, before)) for pruned, before in pairs
    )
    return {
        "micro": total_pruned / total_before if total_before else 0.0,
        "macro_all": sum(all_ratios) / len(all_ratios) if all_ratios else 0.0,
        "macro_nonempty": (
            sum(nonempty_ratios) / len(nonempty_ratios)
            if nonempty_ratios
            else 0.0
        ),
        "nonempty_rows": len(nonempty_ratios),
    }


def v5_pruning_rejection_reason(
    kl_status: str,
    successful_tool_calls: int,
    min_successful_tools: int,
    action_blocks_preserved: bool,
) -> str | None:
    if kl_status != "ok":
        return f"kl_status_{kl_status}"
    if successful_tool_calls < min_successful_tools:
        return "insufficient_successful_tool_evidence"
    if not action_blocks_preserved:
        return "action_blocks_not_preserved"
    return None


def v5_step_is_kl_candidate(
    step: dict[str, Any],
    v5_hidden_summary: bool,
    include_prepruned_override: bool,
) -> bool:
    return bool(step.get("reasoning")) and (
        not step.get("pruned")
        or v5_hidden_summary
        or include_prepruned_override
    )


def updated_step_pruned_chars(
    original_reasoning_chars: int,
    replacement_reasoning_chars: int,
    previous_pruned_chars: int,
) -> tuple[int, int]:
    original = max(0, original_reasoning_chars)
    remaining = min(original, max(0, replacement_reasoning_chars))
    updated = max(min(original, previous_pruned_chars), original - remaining)
    return updated, updated - min(original, max(0, previous_pruned_chars))


VISUAL_EVIDENCE_KEYS = ("coordinates", "bboxes", "crop_ids", "ocr_text", "astar_outputs", "candidate_differences")
VISUAL_EVIDENCE_TERMS = (
    "coordinate",
    "coordinates",
    "bbox",
    "bounding box",
    "pixel_bbox",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "crop",
    "cropped",
    "img_",
    "ocr",
    "text",
    "astar",
    "a*",
    "path",
    "start",
    "goal",
    "obstacle",
    "hole",
    "candidate",
    "edge",
    "texture",
    "color",
    "position",
    "left",
    "right",
    "above",
    "below",
)


def short_text(value: Any, max_chars: int = 260) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(json_safe(value), ensure_ascii=False)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def json_safe(value: Any) -> Any:
    if isinstance(value, Image.Image):
        return f"<PIL.Image {value.width}x{value.height}>"
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


def add_limited(target: list[str], value: Any, limit: int = 8, max_chars: int = 260) -> None:
    text = short_text(value, max_chars=max_chars)
    if text and text not in target and len(target) < limit:
        target.append(text)


def empty_visual_evidence() -> dict[str, list[str]]:
    return {key: [] for key in VISUAL_EVIDENCE_KEYS}


def merge_visual_evidence(*evidences: dict[str, Any]) -> dict[str, list[str]]:
    merged = empty_visual_evidence()
    for evidence in evidences:
        if not isinstance(evidence, dict):
            continue
        for key in VISUAL_EVIDENCE_KEYS:
            values = evidence.get(key, [])
            if not isinstance(values, list):
                values = [values]
            for value in values:
                add_limited(merged[key], value)
    return merged


def extract_visual_evidence_from_text(text: str | None) -> dict[str, list[str]]:
    evidence = empty_visual_evidence()
    if not text:
        return evidence
    compact = short_text(text, max_chars=2000)
    for match in re.finditer(r"img_\d+", compact, flags=re.I):
        add_limited(evidence["crop_ids"], match.group(0))
    for match in re.finditer(r"\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]", compact):
        add_limited(evidence["bboxes"], match.group(0))
    for match in re.finditer(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)", compact):
        add_limited(evidence["coordinates"], match.group(0))
    lowered = compact.lower()
    if "ocr" in lowered or "text" in lowered or "raw_text" in lowered:
        add_limited(evidence["ocr_text"], compact)
    if "astar" in lowered or "a*" in lowered or "path" in lowered:
        add_limited(evidence["astar_outputs"], compact)
    if any(term in lowered for term in ("candidate", "edge", "texture", "color", "match", "missing part", "option")):
        add_limited(evidence["candidate_differences"], compact)
    return evidence


def extract_visual_evidence_from_tool(name: str, params: dict[str, Any], result: dict[str, Any], new_image_id: str | None = None) -> dict[str, list[str]]:
    evidence = empty_visual_evidence()
    if not isinstance(params, dict):
        params = {}
    if not isinstance(result, dict):
        result = {"result": result}
    name_l = name.lower()
    combined = short_text({"tool": name, "parameters": json_safe(params), "result": json_safe(compact_tool_result(result))}, max_chars=1200)
    if name == "Crop":
        add_limited(evidence["bboxes"], params.get("coordinates", params.get("bbox")))
        if new_image_id:
            add_limited(evidence["crop_ids"], f"{new_image_id}: crop of {params.get('image', 'image')} at {params.get('coordinates', params.get('bbox'))}")
    elif name == "OCR":
        raw_text = result.get("raw_text") or result.get("text") or result.get("message")
        add_limited(evidence["ocr_text"], raw_text or combined)
        detections = result.get("detections")
        if isinstance(detections, list):
            for det in detections[:8]:
                if isinstance(det, dict):
                    add_limited(evidence["ocr_text"], det.get("label") or det.get("text"))
                    add_limited(evidence["bboxes"], det.get("pixel_bbox"))
    elif name == "AStarWithPixelCoordinate" or "astar" in name_l:
        add_limited(evidence["astar_outputs"], result.get("message") or combined)
    elif name == "Point":
        add_limited(evidence["coordinates"], result.get("message") or combined)
    elif name in {"Draw2DPath", "DetectBlackArea", "InsertImage"}:
        add_limited(evidence["coordinates"], result.get("message") or combined)
    for key, values in extract_visual_evidence_from_text(combined).items():
        for value in values:
            add_limited(evidence[key], value)
    return evidence


def reasoning_has_visual_evidence(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if any(term in lowered for term in VISUAL_EVIDENCE_TERMS):
        return True
    if re.search(r"img_\d+|\[\s*-?\d+[, ]+-?\d+[, ]+-?\d+[, ]+-?\d+\s*\]|\(\s*-?\d+[, ]+-?\d+\s*\)", text, flags=re.I):
        return True
    return False


def looks_redundant_reasoning(text: str | None) -> bool:
    if not text:
        return False
    clean = re.sub(r"\s+", " ", text).strip()
    lowered = clean.lower()
    filler_hits = sum(
        phrase in lowered
        for phrase in (
            "i need to",
            "i should",
            "i will",
            "let me",
            "now i",
            "we need to",
            "the question asks",
            "i can answer",
            "final answer",
        )
    )
    has_specific_numbers = bool(re.search(r"\d", clean))
    return filler_hits >= 2 and not has_specific_numbers and not reasoning_has_visual_evidence(clean)


def reasoning_has_decision_signal(text: str | None) -> bool:
    if not text:
        return False
    lowered = re.sub(r"\s+", " ", text).strip().lower()
    if not lowered:
        return False
    decision_markers = (
        "candidate",
        "choice",
        "choose",
        "select",
        "closest",
        "compare",
        "difference",
        "uncertain",
        "uncertainty",
        "\\boxed",
    )
    if any(marker in lowered for marker in decision_markers):
        return True
    if re.search(r"\b[abcd]\b", lowered) and re.search(r"\b(option|choice|answer|select|candidate)\b", lowered):
        return True
    return False


def lacks_tool_grounded_visual_evidence(text: str | None) -> bool:
    """True for reasoning that should not survive an evidence-only context pass."""
    if not text:
        return False
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return False
    lowered = clean.lower()
    explicit_evidence_markers = (
        "structured visual evidence:",
        "retained tool observations:",
        "verified visual evidence:",
    )
    if any(marker in lowered for marker in explicit_evidence_markers):
        return False
    if reasoning_has_decision_signal(clean):
        return False
    if len(clean) > 120 and any(
        marker in lowered
        for marker in (
            "i need to",
            "i will",
            "i'll use",
            "let me",
            "now i",
            "observation from tool call",
            "i have already",
            "i've already",
            "the question asks",
        )
    ):
        return True
    return looks_redundant_reasoning(clean)


def extract_visual_anchor_text(text: str | None, max_anchors: int = 2, max_chars: int = 360, strict: bool = False) -> str:
    """Keep only short visual anchors while discarding unsupported chain-of-thought prose."""
    if not text or max_anchors <= 0:
        return ""
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if not clean:
        return ""
    pieces = [
        piece.strip(" -•\t")
        for piece in re.split(r"(?<=[.!?。！？])\s+|[;\n]+", clean)
        if piece.strip(" -•\t")
    ]
    anchors: list[str] = []
    for piece in pieces:
        lowered = piece.lower()
        if len(piece) > 220:
            continue
        if not reasoning_has_visual_evidence(piece) and not reasoning_has_decision_signal(piece):
            continue
        if any(
            marker in lowered
            for marker in (
                "i need to",
                "i should",
                "i will",
                "i'll",
                "let me",
                "now i",
                "the question asks",
                "final answer",
            )
        ):
            continue
        if strict:
            if any(
                marker in lowered
                for marker in (
                    "tool",
                    "ocr",
                    "point",
                    "crop",
                    "astar",
                    "a*",
                    "failed",
                    "did not detect",
                    "no readable",
                    "not relevant",
                    "can move",
                    "moving off",
                    "falling into",
                    "use the",
                )
            ):
                continue
            if not any(
                marker in lowered
                for marker in (
                    "left",
                    "right",
                    "top",
                    "bottom",
                    "above",
                    "below",
                    "center",
                    "near",
                    "beside",
                    "color",
                    "red",
                    "blue",
                    "green",
                    "yellow",
                    "black",
                    "white",
                    "edge",
                    "texture",
                    "background",
                    "object",
                    "person",
                    "face",
                    "building",
                    "road",
                    "text",
                    "candidate",
                    "option",
                )
            ):
                continue
        anchors.append(piece)
        if len(anchors) >= max_anchors:
            break
    if not anchors:
        return ""
    anchor_text = "Retained short visual anchors:\n" + "\n".join(f"- {anchor}" for anchor in anchors)
    return short_text(anchor_text, max_chars=max_chars)


def extract_structured_visual_evidence_text(text: str | None, max_lines: int = 4, max_chars: int = 600) -> str:
    """Extract concrete visual/tool evidence while dropping planning prose."""
    if not text or max_lines <= 0:
        return ""
    clean = re.sub(r"\s+", " ", str(text)).strip()
    if not clean:
        return ""
    evidence_markers = (
        "coordinate",
        "bbox",
        "bounding box",
        "pixel",
        "crop",
        "ocr",
        "text",
        "path",
        "astar",
        "color",
        "colour",
        "red",
        "blue",
        "green",
        "yellow",
        "black",
        "white",
        "purple",
        "orange",
        "brown",
        "gray",
        "grey",
        "left",
        "right",
        "top",
        "bottom",
        "above",
        "below",
        "center",
        "candidate",
        "option",
        "edge",
        "texture",
        "area",
        "count",
        "number",
    )
    noise_markers = (
        "i need to",
        "i should",
        "i will",
        "i'll",
        "let me",
        "now i",
        "next step",
        "the question asks",
        "use the tool",
        "call the",
        "final answer",
    )
    pieces = [
        piece.strip(" -•\t")
        for piece in re.split(r"(?<=[.!?。！？])\s+|[;\n]+", clean)
        if piece.strip(" -•\t")
    ]
    lines: list[str] = []
    for piece in pieces:
        lowered = piece.lower()
        if len(piece) > 260:
            continue
        if any(marker in lowered for marker in noise_markers):
            continue
        has_marker = any(marker in lowered for marker in evidence_markers)
        has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b|\[[^\]]*\d+[^\]]*\]|\([^\)]*\d+[^\)]*\)", piece))
        if has_marker or has_number:
            lines.append(piece)
            if len(lines) >= max_lines:
                break
    if not lines:
        return ""
    return short_text("Structured visual evidence kept:\n" + "\n".join(f"- {line}" for line in lines), max_chars=max_chars)


def question_has_explicit_options(text: str | None) -> bool:
    if not text:
        return False
    return bool(re.search(r"(^|\n|\s)[A-Da-d][\).:：]\s+\S", text) or re.search(r"\boptions?\b", text, flags=re.I))


def final_answer_needs_evidence_revision(answer: str | None, question: str | None) -> bool:
    if not answer:
        return True
    clean = re.sub(r"\s+", " ", str(answer)).strip()
    lowered = clean.lower()
    refusal_markers = (
        "unable to determine",
        "cannot determine",
        "can't determine",
        "not enough information",
        "no actionable information",
        "none of the options",
        "no option",
        "not visible",
        "no visible",
        "unknown",
    )
    if any(marker in lowered for marker in refusal_markers):
        return True
    if question_has_explicit_options(question) and not re.search(r"(?:^|[^A-Za-z])[A-D](?:$|[^A-Za-z])", clean):
        return True
    return False


def reasoning_prose_without_action(reasoning: str | None, action_block: str | None = None) -> str:
    if not reasoning:
        return ""
    text = str(reasoning)
    if action_block:
        text = text.replace(str(action_block), " ")
    text = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", " ", text, flags=re.I | re.S)
    text = re.sub(r"<response>.*?(?:</response>|$)", " ", text, flags=re.I | re.S)
    text = re.sub(
        r"```(?:json)?\s*\{.*?(?:\"name\"|\"tool\"|\"tool_name\").*?\}\s*```",
        " ",
        text,
        flags=re.I | re.S,
    )
    text = re.sub(
        r"\b(?:Point|OCR|Crop|AStarWithPixelCoordinate|DetectBlackArea|Draw2DPath|InsertImage|summarize_the_task)\s*\([^)]*\)",
        " ",
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


def question_focus_terms(text: str | None) -> set[str]:
    """Small heuristic target vocabulary for checking whether a tool hit the queried object."""
    if not text:
        return set()
    lowered = text.lower()
    stop = {
        "what",
        "which",
        "where",
        "when",
        "does",
        "this",
        "that",
        "there",
        "image",
        "picture",
        "photo",
        "shown",
        "visible",
        "answer",
        "option",
        "choose",
        "following",
        "above",
        "below",
        "left",
        "right",
        "color",
        "colour",
        "number",
        "letter",
    }
    terms = {w for w in re.findall(r"[a-z][a-z0-9_-]{3,}", lowered) if w not in stop}
    visual_targets = {
        "mailbox",
        "entrance",
        "sign",
        "door",
        "woman",
        "man",
        "person",
        "car",
        "bus",
        "train",
        "chair",
        "table",
        "logo",
        "text",
        "maze",
        "player",
        "goal",
        "hole",
        "piece",
        "jigsaw",
        "object",
    }
    return terms | {target for target in visual_targets if target in lowered}


def question_target_terms(text: str | None) -> set[str]:
    """Object-level terms that visual evidence should hit for fine-grained VQA."""
    if not text:
        return set()
    lowered = text.lower()
    prompt = re.split(r"\n\s*\([a-d]\)|answer with|choices?:", lowered, maxsplit=1)[0]
    stop = {
        "what",
        "which",
        "where",
        "when",
        "kind",
        "type",
        "color",
        "colour",
        "image",
        "picture",
        "photo",
        "shown",
        "visible",
        "there",
        "this",
        "that",
        "with",
        "from",
        "given",
        "choice",
        "option",
        "letter",
        "answer",
        "directly",
        "purple",
        "white",
        "black",
        "green",
        "yellow",
        "blue",
        "brown",
        "orange",
        "red",
        "gray",
        "grey",
    }
    terms = {w for w in re.findall(r"[a-z][a-z0-9_-]{3,}", prompt) if w not in stop}
    if any(marker in lowered for marker in ("maze", "jigsaw", "path", "start", "goal", "hole", "candidate", "piece")):
        return set()
    return terms


def repair_tool_parameters(
    name: str,
    params: dict[str, Any],
    question: str | None,
    available_image_count: int | None = None,
) -> tuple[dict[str, Any], bool]:
    repaired_params = dict(params)
    repaired = False
    image_match = re.fullmatch(
        r"img_(\d+)",
        str(repaired_params.get("image", "")).strip(),
        flags=re.I,
    )
    if image_match and available_image_count and available_image_count > 0:
        image_index = int(image_match.group(1))
        if not 1 <= image_index <= available_image_count:
            repaired_params["image"] = f"img_{available_image_count}"
            repaired = True
    if name == "Point" and not str(repaired_params.get("description", "")).strip():
        targets = sorted(question_target_terms(question))
        repaired_params["description"] = (
            "locate the " + " ".join(targets[:6])
            if targets
            else "locate the key visual evidence requested by the question"
        )
        repaired = True
    return repaired_params, repaired


def target_hit_gate_enabled(question: str | None, target_terms: set[str]) -> bool:
    if os.environ.get("BENCHMARK_TARGET_HIT_GATE") != "1":
        return False
    if not target_terms:
        return False
    lowered = (question or "").lower()
    fine_grained_markers = (
        "color of",
        "colour of",
        "kind of",
        "type of",
        "what animal",
        "which animal",
        "what is the",
        "which is the",
    )
    return any(marker in lowered for marker in fine_grained_markers)


def target_hit_gate_issue(
    name: str,
    params: dict[str, Any],
    result: dict[str, Any],
    target_terms: set[str],
) -> str | None:
    """Reject weak tool evidence that does not mention the queried target object."""
    if not target_terms or name not in {"Point", "OCR", "Crop"}:
        return None
    target_terms = {term for term in target_terms if len(term) >= 4}
    if not target_terms:
        return None
    params_text = json.dumps(json_safe(params), ensure_ascii=False).lower()
    result_text = json.dumps(json_safe(compact_tool_result(result)), ensure_ascii=False).lower()
    combined = params_text + " " + result_text
    if any(term in combined for term in target_terms):
        return None
    if name == "Crop" and str(result.get("status", "")).lower() == "success":
        return None
    return (
        f"{name} evidence did not mention the target term(s) "
        f"{', '.join(sorted(target_terms)[:6])}; re-check the queried object instead of relying on generic visual evidence."
    )


def tool_observation_issue(name: str, params: dict[str, Any], result: dict[str, Any], focus_terms: set[str]) -> str | None:
    """Detect observations that should not be trusted as final visual evidence."""
    if not name:
        return None
    compact = json.dumps(json_safe(compact_tool_result(result)), ensure_ascii=False).lower()
    status = str(result.get("status", "")).lower()
    if status in {"failed", "error"} or int(result.get("error_code", 0) or 0) != 0:
        return f"{name} failed or returned an error."
    if name == "OCR":
        raw_text = str(result.get("raw_text", "") or "").strip()
        detections = result.get("detections")
        if not raw_text and not detections:
            return "OCR returned no readable text."
    if name == "Point":
        message = str(result.get("message", "") or "").lower()
        if focus_terms and message:
            overlap = {term for term in focus_terms if term in message}
            generic_person_terms = {"person", "woman", "man", "face"}
            target_without_person = focus_terms - generic_person_terms
            if not overlap and target_without_person:
                return f"Point appears to identify '{message[:120]}' instead of the requested target."
            if target_without_person and any(term in message for term in generic_person_terms) and not (target_without_person & set(re.findall(r"[a-z][a-z0-9_-]{3,}", message))):
                return f"Point appears to identify '{message[:120]}' instead of the requested target."
    mismatch_markers = (
        "wrong",
        "incorrect",
        "not the target",
        "instead of",
        "no parseable",
        "no readable",
        "detected 0",
    )
    if any(marker in compact for marker in mismatch_markers):
        return f"{name} observation is weak or mismatched: {short_text(compact, max_chars=180)}"
    return None


def canonical_tool_signature(name: str, params: dict[str, Any]) -> str:
    try:
        payload = json.dumps(json_safe(params), ensure_ascii=False, sort_keys=True)
    except Exception:
        payload = repr(params)
    return f"{name}:{payload}"


def forced_recheck_tool_for_issue(issue: str | None, previous_tool: str | None = None) -> str | None:
    if not issue:
        return None
    lowered = issue.lower()
    if "point appears to identify" in lowered or "ocr returned no readable text" in lowered:
        return "Crop"
    return None


def needs_compact_answer_cleanup(text: str | None) -> bool:
    if not text:
        return True
    clean = re.sub(r"\s+", " ", str(text)).strip()
    lowered = clean.lower()
    if "<response>" in lowered or "</response>" in lowered:
        return True
    if any(marker in lowered for marker in ("the answer is", "therefore", "because", "ocr", "tool", "shown", "displayed", "i think", "we can see")):
        return True
    words = clean.split()
    if len(words) > 8:
        return True
    comma_parts = [x.strip().lower() for x in clean.split(",") if x.strip()]
    if len(comma_parts) > 12:
        unique_window = len(set(comma_parts[:12]))
        if unique_window <= 6:
            return True
    return False


def sanitize_final_answer(text: str | None) -> str:
    """Conservative final-answer cleanup that avoids asking the model to regenerate."""
    if text is None:
        return ""
    clean = str(text).strip()
    nested = parse_tag(clean, "response")
    if nested is not None:
        clean = nested.strip()
    boxed = re.search(r"\\boxed\{([^{}]{1,200})\}", clean)
    if boxed:
        return "\\boxed{" + boxed.group(1).strip() + "}"
    simple_response = re.search(r"<response>\s*([^<>]{1,200})\s*</response>", clean, flags=re.I | re.S)
    if simple_response:
        clean = simple_response.group(1).strip()
    marker_match = re.search(r"(?:final answer|answer)\s*[:：]\s*([^.\n\r]{1,200})", clean, flags=re.I)
    if marker_match:
        clean = marker_match.group(1).strip()
    clean = re.sub(r"^Therefore,?\s*", "", clean, flags=re.I).strip()
    if "\n" in clean:
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        short_lines = [line for line in lines if len(line.split()) <= 8 and len(line) <= 200]
        if short_lines:
            clean = short_lines[-1]
    return clean.strip()


def completion_echo_logprobs(
    base_url: str,
    model: str,
    prompt: str,
    top_logprobs: int,
    timeout: int,
) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": top_logprobs,
            "echo": True,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return {
        "prompt_tokens": int((data.get("usage") or {}).get("prompt_tokens", 0)),
        "tokens": logprobs.get("tokens") or [],
        "token_logprobs": logprobs.get("token_logprobs") or [],
        "top_logprobs": logprobs.get("top_logprobs") or [],
    }


def response_logprob_positions(
    base_url: str,
    model: str,
    prompt: str,
    response_text: str,
    top_logprobs: int,
    timeout: int,
) -> dict[str, Any]:
    prefix = completion_echo_logprobs(base_url, model, prompt, top_logprobs, timeout)
    full = completion_echo_logprobs(base_url, model, prompt + response_text, top_logprobs, timeout)
    start = int(prefix.get("prompt_tokens", 0))
    return {
        "tokens": full["tokens"][start:],
        "token_logprobs": full["token_logprobs"][start:],
        "top_logprobs": full["top_logprobs"][start:],
    }


def reverse_kl_from_top_logprobs(top_s: dict[str, float] | None, top_t: dict[str, float] | None, floor: float = -30.0) -> float:
    if not isinstance(top_s, dict) or not top_s:
        return 0.0
    top_t = top_t if isinstance(top_t, dict) else {}
    value = 0.0
    for token, lp_s_raw in top_s.items():
        try:
            lp_s = float(lp_s_raw)
        except Exception:
            continue
        lp_t = float(top_t.get(token, floor))
        value += min(1.0, max(0.0, pow(2.718281828459045, lp_s))) * (lp_s - lp_t)
    return max(0.0, value)


def token_segment_indices(tokens: list[str], segment_texts: list[str], separator: str) -> list[int]:
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for index, text in enumerate(segment_texts):
        if index:
            cursor += len(separator)
        start = cursor
        cursor += len(text)
        boundaries.append((start, cursor))

    indices: list[int] = []
    char_pos = 0
    current_segment = 0
    for token in tokens:
        token_len = max(1, len(str(token)))
        mid = char_pos + token_len // 2
        while current_segment + 1 < len(boundaries) and mid >= boundaries[current_segment][1]:
            current_segment += 1
        start, end = boundaries[current_segment]
        indices.append(current_segment if start <= mid < end else -1)
        char_pos += token_len
    return indices


def log_summarization_kl(
    base_url: str,
    model: str,
    system_prompt: str,
    summary_text: str,
    segments: list[dict[str, Any]],
    top_logprobs: int,
    timeout: int,
    prune_threshold: float,
    min_high_tokens: int,
    max_response_chars: int,
    include_prepruned: bool = False,
) -> dict[str, Any]:
    include_prepruned = include_prepruned or (
        os.environ.get("BENCHMARK_V5_INCLUDE_PREPRUNED_CANDIDATES") == "1"
    )
    candidates = [
        segment
        for segment in segments
        if (include_prepruned or not segment.get("pruned"))
        and str(segment.get("reasoning") or "").strip()
    ]
    if not candidates:
        return {"status": "no_candidates", "prune_indices": [], "segments": []}

    separator = "\n\n"
    segment_texts = [str(segment["reasoning"]) for segment in candidates]
    response_text = separator.join(segment_texts)
    truncated = False
    if max_response_chars > 0 and len(response_text) > max_response_chars:
        truncated = True
        kept: list[dict[str, Any]] = []
        kept_texts: list[str] = []
        total = 0
        for segment, text in zip(candidates, segment_texts):
            extra = len(text) + (len(separator) if kept else 0)
            if kept and total + extra > max_response_chars:
                break
            if not kept and len(text) > max_response_chars:
                text = text[-max_response_chars:]
                segment = dict(segment)
                segment["reasoning"] = text
                extra = len(text)
            kept.append(segment)
            kept_texts.append(text)
            total += extra
        candidates = kept
        segment_texts = kept_texts
        response_text = separator.join(segment_texts)

    student_prompt = system_prompt + "\n\n"
    teacher_prompt = system_prompt + "\n\nSummary of prior task state:\n" + summary_text.strip() + "\n\n"
    s_pos = response_logprob_positions(base_url, model, student_prompt, response_text, top_logprobs, timeout)
    t_pos = response_logprob_positions(base_url, model, teacher_prompt, response_text, top_logprobs, timeout)
    length = min(len(s_pos["tokens"]), len(t_pos["tokens"]), len(s_pos["top_logprobs"]), len(t_pos["top_logprobs"]))
    if length <= 0:
        return {"status": "empty_logprobs", "prune_indices": [], "segments": [], "truncated": truncated}

    raw_kl = [
        reverse_kl_from_top_logprobs(s_pos["top_logprobs"][i], t_pos["top_logprobs"][i])
        for i in range(length)
    ]
    rkl_min = min(raw_kl)
    rkl_max = max(raw_kl)
    if rkl_max > rkl_min:
        norm_kl = [(value - rkl_min) / (rkl_max - rkl_min) for value in raw_kl]
    else:
        norm_kl = [0.0 for _ in raw_kl]

    segment_ids = token_segment_indices([str(x) for x in s_pos["tokens"][:length]], segment_texts, separator)
    per_segment: list[dict[str, Any]] = []
    prune_indices: list[int] = []
    for candidate_index, segment in enumerate(candidates):
        values = [value for value, seg_id in zip(norm_kl, segment_ids) if seg_id == candidate_index]
        n_high = sum(1 for value in values if value > prune_threshold)
        keep = n_high >= min_high_tokens
        if not keep:
            prune_indices.append(int(segment["step_index"]))
        per_segment.append(
            {
                "step_index": int(segment["step_index"]),
                "turn_round": segment.get("turn_round"),
                "token_count": len(values),
                "n_high": n_high,
                "mean_norm_rkl": sum(values) / len(values) if values else 0.0,
                "max_norm_rkl": max(values) if values else 0.0,
                "keep": keep,
                "reasoning_chars": len(str(segment.get("reasoning") or "")),
            }
        )
    return {
        "status": "ok",
        "prune_indices": prune_indices,
        "segments": per_segment,
        "raw_token_count": length,
        "raw_rkl_min": rkl_min,
        "raw_rkl_max": rkl_max,
        "truncated": truncated,
    }


def summary_tool_call_instruction(prefix: str, summary_mode: str = "generic") -> str:
    strict_evidence = os.environ.get("BENCHMARK_SUMMARY_STRICT_EVIDENCE") == "1"
    if os.environ.get("BENCHMARK_SUMMARY_CHECKPOINT") == "1":
        parameters = {
            "facts": ["verified visual/tool fact"],
            "tool_evidence": ["tool result, crop id, OCR text, coordinate/bbox, or path"],
            "open_risks": ["missing or uncertain evidence"],
            "next_step": "next tool or final decision",
        }
        return (
            prefix
            + "\nOutput one compact summarize_the_task checkpoint. Keep strings short. "
            + "Only include verified visual/tool evidence; omit unsupported guesses and long reasoning. "
            + "Do not output any other tool name and do not output <response>.\n"
            + "<tool_call>\n"
            + json.dumps({"name": SUMMARY_TOOL_NAME, "parameters": parameters}, ensure_ascii=False)
            + "\n</tool_call>"
        )
    if summary_mode == "visual_state":
        parameters = {
            "reasoning_so_far": "empty or <=12 words; do not restate guesses",
            "completed_steps": ["tool/action already completed"],
            "current_state": "current answer state and what is still unresolved",
            "visual_evidence": {
                "coordinates": ["start/goal/object/obstacle coordinates such as start=(x,y)"],
                "bboxes": ["object/text/candidate bboxes such as [x_min,y_min,x_max,y_max]"],
                "crop_ids": ["img_2 = crop of target/candidate region"],
                "ocr_text": ["verbatim OCR text and region if known"],
                "astar_outputs": ["AStar path/result or failure reason"],
                "candidate_differences": ["local edge/texture/color/spatial differences among options"],
            },
            "key_observations": ["important visual/tool observation"],
            "uncertainties": ["missing visual evidence to verify"],
            "next_decision": "next tool call or final decision",
        }
    else:
        parameters = {
            "reasoning_so_far": "what you have inferred so far",
            "completed_steps": ["step already completed"],
            "current_state": "where the task stands now",
            "key_observations": ["important observation"],
            "next_decision": "what to do next",
        }
    return (
        prefix
        + "\nOutput exactly this structure, replacing the field values with your own current state. "
        + (
            "Preserve concrete visual evidence: coordinates, bboxes, crop ids, OCR text, AStar outputs, and candidate local differences. "
            if summary_mode == "visual_state"
            else ""
        )
        + (
            "Do not put long prose in reasoning_so_far; leave it empty unless a short verified fact is necessary. "
            "Use visual_evidence/key_observations for verified tool/image evidence only; omit unsupported guesses. "
            if strict_evidence and summary_mode == "visual_state"
            else ""
        )
        + "Do not output any other tool name and do not output <response>.\n"
        + "<tool_call>\n"
        + json.dumps(
            {
                "name": SUMMARY_TOOL_NAME,
                "parameters": parameters,
            },
            ensure_ascii=False,
        )
        + "\n</tool_call>"
    )


def resolve_placeholders(value: Any, images: list[Image.Image]) -> Any:
    if isinstance(value, str):
        match = re.fullmatch(r"img_(\d+)", value.strip(), flags=re.I)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < len(images):
                return images[index]
        return value
    if isinstance(value, list):
        return [resolve_placeholders(v, images) for v in value]
    if isinstance(value, dict):
        return {k: resolve_placeholders(v, images) for k, v in value.items()}
    return value


def should_append_generated_image(
    current_image_count: int,
    initial_image_count: int,
    max_generated_images: int,
) -> bool:
    return (
        max_generated_images <= 0
        or current_image_count - initial_image_count < max_generated_images
    )


def normalize_for_online(value: Any) -> Any:
    if isinstance(value, Image.Image):
        return pil_to_base64(value.convert("RGB"))
    if isinstance(value, list):
        return [normalize_for_online(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_for_online(v) for k, v in value.items()}
    return value


def crop_image_tool(params: dict[str, Any]) -> dict[str, Any]:
    image = params.get("image")
    coordinates = params.get("coordinates", params.get("bbox"))
    if not isinstance(image, Image.Image):
        return {"status": "failed", "message": "Crop expected image parameter.", "error_code": 1, "tool_response_from": "Crop"}
    width, height = image.size
    if coordinates is None:
        coordinates = [0, 0, width, height]
    try:
        if isinstance(coordinates, str):
            try:
                coords = ast.literal_eval(coordinates)
            except Exception:
                nums = re.findall(r"-?\d+(?:\.\d+)?", coordinates)
                coords = [float(x) for x in nums[:4]] if len(nums) >= 4 else coordinates
        else:
            coords = coordinates
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError("coordinates must be [x_min, y_min, x_max, y_max]")
        x_min, y_min, x_max, y_max = [int(float(x)) for x in coords]
        if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min or x_max > width or y_max > height:
            raise ValueError(f"coordinates out of bounds for image size {width}x{height}")
        cropped = image.convert("RGB").crop((x_min, y_min, x_max, y_max))
        if cropped.width < 32 or cropped.height < 32:
            padded = Image.new("RGB", (max(32, cropped.width), max(32, cropped.height)), (255, 255, 255))
            padded.paste(cropped, ((padded.width - cropped.width) // 2, (padded.height - cropped.height) // 2))
            cropped = padded
        return {
            "status": "success",
            "edited_image": pil_to_base64(cropped),
            "message": "Image cropped successfully using absolute coordinates.",
            "image_dimensions_pixels": {"width": cropped.width, "height": cropped.height},
            "error_code": 0,
            "tool_response_from": "Crop",
        }
    except Exception as exc:
        return {"status": "failed", "message": f"Crop failed: {exc}", "error_code": 1, "tool_response_from": "Crop"}


def grid_crops(image: Image.Image, splits: int = 2) -> list[tuple[str, Image.Image, list[int]]]:
    splits = max(1, min(4, splits))
    width, height = image.size
    x_edges = [round(width * i / splits) for i in range(splits + 1)]
    y_edges = [round(height * i / splits) for i in range(splits + 1)]
    row_names = ["top", "middle", "bottom", "row4"]
    col_names = ["left", "center", "right", "col4"]
    crops: list[tuple[str, Image.Image, list[int]]] = []
    rgb = image.convert("RGB")
    for row in range(splits):
        for col in range(splits):
            coords = [x_edges[col], y_edges[row], x_edges[col + 1], y_edges[row + 1]]
            label = f"{row_names[row]}-{col_names[col]}"
            x_min, y_min, x_max, y_max = coords
            if x_max > x_min and y_max > y_min:
                crops.append((label, rgb.crop((x_min, y_min, x_max, y_max)), coords))
    return crops


def summarize_the_task_tool(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        params = {}

    def as_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return json.dumps(value, ensure_ascii=False)

    def as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [as_text(x) for x in value if as_text(x)]
        text = as_text(value)
        return [text] if text else []

    checkpoint_mode = os.environ.get("BENCHMARK_SUMMARY_CHECKPOINT") == "1"
    strict_evidence = os.environ.get("BENCHMARK_SUMMARY_STRICT_EVIDENCE") == "1"
    facts = as_list(params.get("facts"))
    tool_evidence = as_list(params.get("tool_evidence"))
    open_risks = as_list(params.get("open_risks"))
    completed_steps = as_list(params.get("completed_steps"))
    key_observations = as_list(params.get("key_observations"))
    uncertainties = as_list(params.get("uncertainties"))
    visual_evidence = merge_visual_evidence(params.get("visual_evidence", {}), extract_visual_evidence_from_text(json.dumps(json_safe(params), ensure_ascii=False)))
    reasoning_so_far = as_text(params.get("reasoning_so_far"))
    if strict_evidence:
        reasoning_so_far = short_text(reasoning_so_far, max_chars=120)
    summary = {
        "facts": facts,
        "tool_evidence": tool_evidence,
        "open_risks": open_risks,
        "reasoning_so_far": reasoning_so_far,
        "completed_steps": completed_steps,
        "current_state": as_text(params.get("current_state")),
        "visual_evidence": visual_evidence,
        "key_observations": key_observations,
        "uncertainties": uncertainties,
        "next_decision": as_text(params.get("next_decision") or params.get("next_step")),
    }
    compact_lines = []
    summary_items = (
        [(key, summary[key]) for key in ("facts", "tool_evidence", "open_risks", "next_decision")]
        if checkpoint_mode
        else list(summary.items())
    )
    for key, value in summary_items:
        if key == "visual_evidence" and isinstance(value, dict):
            for evidence_key in VISUAL_EVIDENCE_KEYS:
                evidence_values = value.get(evidence_key, [])
                if evidence_values:
                    compact_lines.append(f"{evidence_key}: " + "; ".join(evidence_values[:3] if checkpoint_mode else evidence_values[:6]))
        elif isinstance(value, list):
            if value:
                compact_lines.append(f"{key}: " + "; ".join(short_text(x, max_chars=160) for x in (value[:3] if checkpoint_mode else value[:6])))
        elif value:
            compact_lines.append(f"{key}: {short_text(value, max_chars=200) if checkpoint_mode else value}")
    return {
        "status": "success",
        "summary": summary,
        "message": "\n".join(compact_lines) if compact_lines else "No useful task state was provided to summarize.",
        "error_code": 0,
        "tool_response_from": SUMMARY_TOOL_NAME,
        "backend": "local_python",
        "counts_as_benchmark_tool_call": False,
    }


class ToolRuntime:
    def __init__(
        self,
        tools: list[str],
        controller_addr: str | None = None,
        allow_model_tool_fallback: bool = False,
        model_tool_base_url: str = DEFAULT_BASE_URL,
        model_tool_model: str = DEFAULT_MODEL,
        model_tool_timeout: int = 300,
    ):
        self.tools = tools
        self.allow_model_tool_fallback = allow_model_tool_fallback
        self.model_tool_base_url = model_tool_base_url
        self.model_tool_model = model_tool_model
        self.model_tool_timeout = model_tool_timeout
        self.local_offline_tools = [tool for tool in tools if tool in set(get_available_tools())]
        self.local_extra_tools = [tool for tool in tools if tool in {"Crop", SUMMARY_TOOL_NAME}]
        self.tool_manager = ToolManager(controller_addr, tools=tools)
        self.online_tools = list(self.tool_manager.available_online_tools)
        self.model_fallback_tools = []
        if self.allow_model_tool_fallback:
            self.model_fallback_tools = [tool for tool in tools if tool in {"Point", "OCR"} and tool not in self.online_tools]
        self.available_tools = sorted(
            set(self.local_offline_tools + self.local_extra_tools + self.online_tools + self.model_fallback_tools),
            key=tools.index,
        )
        self.missing_tools = [tool for tool in tools if tool not in self.available_tools]
        self.tool_backends = {}
        for tool in tools:
            if tool in self.online_tools:
                self.tool_backends[tool] = "online_controller"
            elif tool in self.local_offline_tools:
                self.tool_backends[tool] = "offline_worker"
            elif tool in self.local_extra_tools:
                self.tool_backends[tool] = "local_python"
            elif tool in self.model_fallback_tools:
                self.tool_backends[tool] = "model_fallback"
            else:
                self.tool_backends[tool] = "missing"

    def instructions(self) -> str:
        if os.environ.get("BENCHMARK_COMPACT_TOOL_INSTRUCTIONS") == "1":
            return "\n".join(
                (SUMMARY_CHECKPOINT_INSTRUCTION if os.environ.get("BENCHMARK_SUMMARY_CHECKPOINT") == "1" else SUMMARY_TOOL_INSTRUCTION)
                if tool == SUMMARY_TOOL_NAME and os.environ.get("BENCHMARK_SUMMARY_MODE") == "visual_state"
                else f"- {tool}: call with JSON parameters using image ids like img_1 when an image is needed."
                for tool in self.tools
            )
        chunks = []
        for tool in self.tools:
            instruction = None
            if tool in self.local_offline_tools:
                instruction = get_tool_instruction(tool)
            elif tool in self.online_tools:
                instruction = self.tool_manager.get_online_tool_instruction(tool)
            if not instruction and tool in tool_desc_dict:
                instruction = tool_desc_dict[tool]
            if instruction:
                chunks.append(instruction if isinstance(instruction, str) else json.dumps(instruction, ensure_ascii=False))
            elif tool == SUMMARY_TOOL_NAME:
                chunks.append(SUMMARY_TOOL_INSTRUCTION)
        return "\n".join(chunks)

    def call_tool(self, tool_name: str, params: dict[str, Any], images: list[Image.Image]) -> tuple[dict[str, Any], Image.Image | None]:
        if tool_name not in self.tools:
            return {
                "status": "failed",
                "message": f"Tool {tool_name} is not in the configured full benchmark tool set.",
                "error_code": 404,
                "tool_response_from": tool_name,
            }, None
        resolved = resolve_placeholders(copy.deepcopy(params), images)
        if tool_name in self.local_offline_tools:
            fn = get_tool_generate_fn(tool_name)
            result = fn(resolved, tool_manager=self)
        elif tool_name == "Crop":
            result = crop_image_tool(resolved)
        elif tool_name == SUMMARY_TOOL_NAME:
            result = summarize_the_task_tool(resolved)
        elif tool_name in self.online_tools:
            result = self.tool_manager.call_tool(tool_name, normalize_for_online(resolved))
        elif tool_name == "Point" and tool_name in self.model_fallback_tools:
            result = self.model_point_tool(resolved)
        elif tool_name == "OCR" and tool_name in self.model_fallback_tools:
            result = self.model_ocr_tool(resolved)
        else:
            result = {
                "status": "failed",
                "message": f"Tool {tool_name} is missing. Configure/start the AdaReasoner online tool controller for the complete tool set.",
                "error_code": 404,
                "tool_response_from": tool_name,
            }
        edited = None
        if isinstance(result, dict) and result.get("edited_image"):
            try:
                edited = base64_to_pil(result["edited_image"]).convert("RGB")
            except Exception as exc:
                result = dict(result)
                result["edited_image_decode_error"] = repr(exc)
        if isinstance(result, dict):
            # Persist the execution provenance.  In particular, a result from a
            # model-generated fallback must never be mistaken for an observation
            # produced by a benchmark tool.
            result = dict(result)
            result.setdefault("backend", self.tool_backends.get(tool_name, "missing"))
        return result, edited

    def _call_model_tool(self, prompt: str, image: Image.Image, max_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": content_with_images(prompt, [image])}]
        return call_chat(
            self.model_tool_base_url,
            self.model_tool_model,
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=self.model_tool_timeout,
        )

    @staticmethod
    def _json_from_text(text: str) -> Any:
        match = re.search(r"\{.*\}", text or "", flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def model_point_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        image = params.get("image")
        description = str(params.get("description", "")).strip()
        if not isinstance(image, Image.Image):
            return {"status": "failed", "message": "Point expected image parameter.", "error_code": 1, "tool_response_from": "Point"}
        width, height = image.size
        prompt = (
            "You are the Point tool. Identify one representative pixel point in the image matching the description.\n"
            f"Image size: width={width}, height={height}. Description: {description!r}.\n"
            'Return JSON only in this schema: {"points": [[x, y]], "message": "short explanation"}. '
            "Use absolute integer pixel coordinates."
        )
        try:
            text = self._call_model_tool(prompt, image)
            data = self._json_from_text(text) or {}
            points = data.get("points", [])
            clean_points = []
            for point in points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    x = max(0, min(width - 1, int(float(point[0]))))
                    y = max(0, min(height - 1, int(float(point[1]))))
                    clean_points.append([x, y])
            if not clean_points:
                return {
                    "status": "failed",
                    "message": f"Point model fallback returned no parseable point. Raw output: {text[:500]}",
                    "error_code": 1,
                    "tool_response_from": "Point",
                    "backend": "model_fallback",
                }
            return {
                "status": "success",
                "points": clean_points,
                "message": data.get("message", f"Located {len(clean_points)} point(s)."),
                "error_code": 0,
                "tool_response_from": "Point",
                "backend": "model_fallback",
            }
        except Exception as exc:
            return {"status": "failed", "message": f"Point model fallback failed: {exc}", "error_code": 1, "tool_response_from": "Point", "backend": "model_fallback"}

    def model_ocr_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        image = params.get("image")
        if not isinstance(image, Image.Image):
            return {"status": "failed", "message": "OCR expected image parameter.", "error_code": 1, "tool_response_from": "OCR"}
        width, height = image.size
        prompt = (
            "You are the OCR tool. Read all visible text in the image and localize it when possible.\n"
            f"Image size: width={width}, height={height}.\n"
            'Return JSON only in this schema: {"detections": [{"label": "text", "confidence": 1.0, '
            '"pixel_bbox": {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0}}], "raw_text": "all text"}.\n'
            "Use absolute integer pixel coordinates. If localization is uncertain, use the whole image bbox."
        )
        try:
            text = self._call_model_tool(prompt, image, max_tokens=1024)
            data = self._json_from_text(text) or {}
            detections = data.get("detections", [])
            if not isinstance(detections, list):
                detections = []
            clean = []
            for det in detections:
                if not isinstance(det, dict):
                    continue
                label = str(det.get("label", "")).strip()
                bbox = det.get("pixel_bbox", {}) if isinstance(det.get("pixel_bbox"), dict) else {}
                if not label:
                    continue
                x_min = max(0, min(width - 1, int(float(bbox.get("x_min", 0)))))
                y_min = max(0, min(height - 1, int(float(bbox.get("y_min", 0)))))
                x_max = max(x_min + 1, min(width, int(float(bbox.get("x_max", width)))))
                y_max = max(y_min + 1, min(height, int(float(bbox.get("y_max", height)))))
                clean.append({"label": label, "confidence": float(det.get("confidence", 1.0)), "pixel_bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}})
            raw_text = str(data.get("raw_text", "")).strip()
            if not clean and raw_text:
                clean = [{"label": raw_text, "confidence": 1.0, "pixel_bbox": {"x_min": 0, "y_min": 0, "x_max": width, "y_max": height}}]
            return {
                "status": "success",
                "detections": clean,
                "raw_text": raw_text,
                "message": f"Detected {len(clean)} text region(s).",
                "error_code": 0,
                "tool_response_from": "OCR",
                "backend": "model_fallback",
            }
        except Exception as exc:
            return {"status": "failed", "message": f"OCR model fallback failed: {exc}", "error_code": 1, "tool_response_from": "OCR", "backend": "model_fallback"}


def content_with_images(text: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
    return content


def call_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["</tool_call>", "</response>"],
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer EMPTY"},
        timeout=timeout,
    )
    if response.status_code == 400 and max_tokens > 1:
        last_error = response.text[:1000]
        for fallback_max_tokens in (16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 1):
            if fallback_max_tokens >= max_tokens:
                continue
            payload["max_tokens"] = fallback_max_tokens
            retry = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer EMPTY"},
                timeout=timeout,
            )
            if retry.status_code < 400:
                response = retry
                break
            last_error = retry.text[:1000]
        else:
            response._content = (response.text + f"\nFallback max_tokens attempts also failed. Last error: {last_error}").encode("utf-8")
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    matched_stop = choice.get("matched_stop")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", part)))
            else:
                parts.append(str(part))
        content_text = "\n".join(parts)
    else:
        content_text = str(content)
    if matched_stop and isinstance(matched_stop, str) and not content_text.rstrip().endswith(matched_stop):
        content_text += matched_stop
    return content_text


def run_item_with_tools(
    item: dict[str, Any],
    runtime: ToolRuntime,
    base_url: str,
    model: str,
    max_rounds: int,
    min_tool_calls: int,
    max_tokens: int,
    temperature: float,
    timeout: int,
    require_visible_reasoning: bool = True,
    max_format_retries: int = 4,
    enable_summary_kl_pruning: bool = False,
    summary_kl_top_logprobs: int = 20,
    summary_kl_prune_threshold: float = 0.2,
    summary_kl_prune_min_tokens: int = 2,
    summary_kl_max_response_chars: int = 12000,
    summary_mode: str = "generic",
    enable_visual_state_pruning: bool = False,
    enable_evidence_only_pruning: bool = False,
    enable_evidence_recheck: bool = False,
    evidence_recheck_on_issue_only: bool = False,
    summary_before_final_only: bool = False,
    delete_reasoning: bool = False,
    retain_structured_visual_evidence: bool = False,
    summary_compatibility_fallback: bool = False,
    v5_hidden_summary: bool = False,
    v5_kl_delete_only: bool = False,
    v5_no_summary_context: bool = False,
    task_name: str = "",
) -> dict[str, Any]:
    is_nemotron_model = "nemotron" in model.lower()
    protocol = "nemotron" if is_nemotron_model else "xml"
    images = [img.copy().convert("RGB") for img in item["images"]]
    image_names = ", ".join(f"img_{i + 1}" for i in range(len(images)))
    real_tools = [tool for tool in runtime.available_tools if tool != SUMMARY_TOOL_NAME]
    v5_summary_context_enabled = (
        v5_hidden_summary and not v5_kl_delete_only and not v5_no_summary_context
    )
    if min_tool_calls > 0:
        generic_visual_instruction = (
            "\nBefore summarizing or answering, gather at least one concrete visual evidence item with a benchmark tool. "
            f"Only call an available real tool from this list: {', '.join(real_tools) if real_tools else '(none)'}. "
            "Do not invent coordinates, OCR text, detections, or tool results. "
            "The final answer should be compact: only the option, path, number, phrase, or short string needed for scoring."
        )
    else:
        generic_visual_instruction = (
            "\nTools are available when they can resolve a genuine visual uncertainty. "
            f"If you call one, use only a real available tool from: {', '.join(real_tools) if real_tools else '(none)'}. "
            "Do not call an irrelevant tool merely to satisfy a protocol, and never invent tool results. "
            "The final answer should be compact: only the option, path, number, phrase, or short string needed for scoring."
        )
    question = (
        f"Question:\n{item['text']}\n\n"
        f"Current images: {image_names}.\n"
        "Solve the benchmark item as an agent. Think through the task yourself, use relevant tools when helpful, "
        "and then give the final answer."
        + generic_visual_instruction
    )
    if min_tool_calls > 0:
        question += (
            "\nProtocol requirement: before any final answer, execute at least one useful real visual tool "
            "successfully. A failed or skipped tool call does not satisfy this requirement."
        )
    # JigsawBlink's canvas and candidate images can have different raster
    # dimensions.  Give the model the actual coordinate frame before its first
    # action, instead of letting it reuse img_1 coordinates on a candidate.
    if task_name == "jigsaw_blink":
        question += jigsaw_geometry_instruction(images)
    # Some GUI-oriented checkpoints (notably the served Kimi-VL-A3B variant)
    # collapse into pyautogui/code-fence output after an externally forced
    # memory-tool schema.  In compatibility mode summary remains available and
    # a valid voluntary summary still activates KL pruning, but the agent is
    # allowed to continue the ordinary real-tool policy when it declines it.
    # V5 is a guarded, hidden intervention rather than a protocol requirement.
    # Advertising a mandatory summary changes the pre-gate rollout policy:
    # simple samples start calling tools only to unlock summary, and useful
    # benchmark actions can be blocked at the gate.  Keep the ordinary
    # baseline policy until the model itself proposes a final answer.
    v5_summary_enabled = (
        v5_hidden_summary
        and not summary_compatibility_fallback
    )
    summary_required = (
        SUMMARY_TOOL_NAME in runtime.tools
        and not summary_compatibility_fallback
        and not v5_hidden_summary
    )
    prompt_tool_instructions = runtime.instructions()
    if summary_compatibility_fallback or v5_hidden_summary:
        # Match the compatible real-tool policy exactly.  Merely advertising
        # the large summary schema changes the baseline rollout even if no
        # summary is forced. V5 invokes it internally only after a safe gate.
        prompt_tool_instructions = prompt_tool_instructions.replace(SUMMARY_TOOL_INSTRUCTION, "")
        prompt_tool_instructions = prompt_tool_instructions.replace(SUMMARY_CHECKPOINT_INSTRUCTION, "")
    if is_nemotron_model:
        compact_schemas = {
            "Point": '{"image":"img_1","description":"object to locate"}',
            "OCR": '{"image":"img_1"}',
            "Crop": '{"image":"img_1","coordinates":[0,0,256,256]} (replace with integer pixel estimates)',
            "DetectBlackArea": '{"image":"img_1","threshold":50,"min_area":100}',
            "InsertImage": '{"base_image":"img_1","image_to_insert":"img_2","coordinates":[0,0,256,256],"resize":true} (replace with integer hole box)',
            "AStarWithPixelCoordinate": '{"start":[10,10],"goal":[200,200],"obstacles":[]}',
            "Draw2DPath": '{"image":"img_1","start_point":[10,10],"directions":["right","down"],"step":5}',
        }
        prompt_tool_instructions = "\n".join(
            f"- {name}: {compact_schemas.get(name, '{}')}"
            for name in real_tools
        )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": render_tool_loop_prompt(
                prompt_tool_instructions,
                require_summary=summary_required,
                protocol=protocol,
            ),
        },
        {"role": "user", "content": content_with_images(question, images)},
    ]
    turns: list[dict[str, Any]] = []
    tool_calls = 0
    summary_tool_calls = 0
    hidden_summary_calls = 0
    format_retries = 0
    format_failures = 0
    final_text = ""
    last_model_text = ""
    assistant_steps: list[dict[str, Any]] = []
    prune_events: list[dict[str, Any]] = []
    pruned_step_indices: set[int] = set()
    reasoning_chars_before_prune = 0
    reasoning_chars_pruned = 0
    reasoning_rewrite_time_sec = 0.0
    reasoning_rewrite_events = 0
    summary_kl_time_sec = 0.0
    system_prompt_text = str(messages[0]["content"])
    accumulated_visual_evidence = empty_visual_evidence()
    called_benchmark_tool_names: list[str] = []
    successful_benchmark_tool_calls = 0
    compact_cleanup_attempts = 0
    focus_terms = question_focus_terms(str(item.get("text") or ""))
    target_terms = question_target_terms(str(item.get("text") or ""))
    use_target_hit_gate = target_hit_gate_enabled(str(item.get("text") or ""), target_terms)
    post_summary_benchmark_tool_calls = 0
    evidence_recheck_issues: list[dict[str, Any]] = []
    unresolved_evidence_issue: str | None = None
    unresolved_evidence_tool: str | None = None
    forced_recheck_tool: str | None = None
    blocked_repeated_tool_calls = 0
    seen_tool_signatures: set[str] = set()
    auto_post_summary_local_crops = 0
    initial_image_count = len(images)
    successful_jigsaw_insert_candidates: set[str] = set()
    v5_summary_gate_attempts = 0
    v5_summary_gate_accepts = 0
    v5_summary_gate_rejections: list[dict[str, Any]] = []
    last_v5_summary_accepted = False
    last_v5_summary_result: dict[str, Any] | None = None
    v5_summary_trigger_round: Any = None
    v5_summary_trigger_step_index: int | None = None
    v5_online_intervention = False
    v5_post_rollout_diagnostic = False
    tool_parameter_repairs = 0

    def missing_required_visual_tool() -> str | None:
        if (
            v5_hidden_summary
            and os.environ.get("BENCHMARK_V5_REQUIRE_SUCCESSFUL_TOOL") == "1"
            and successful_benchmark_tool_calls
            < int(os.environ.get("BENCHMARK_V5_MIN_SUCCESSFUL_TOOLS", "1"))
        ):
            return "EVIDENCE"
        if min_tool_calls <= 0:
            return None
        if (
            os.environ.get("BENCHMARK_REQUIRE_SUCCESSFUL_TOOL") == "1"
            and successful_benchmark_tool_calls < min_tool_calls
        ):
            return "EVIDENCE"
        called = set(called_benchmark_tool_names)
        if not ({"Point", "OCR", "Crop", "AStarWithPixelCoordinate", "DetectBlackArea"} & called):
            return "EVIDENCE"
        return None

    def required_tool_observation(tool_name: str) -> str:
        action_syntax = (
            'Output REASONING: ... followed by ACTION_JSON: {"name":"ToolName","parameters":{...}} now.'
            if is_nemotron_model
            else "Output one valid <tool_call> JSON object now."
        )
        if tool_name == "EVIDENCE":
            if is_nemotron_model and "Point" in real_tools:
                description = (
                    "locate the " + " ".join(sorted(target_terms)[:6])
                    if target_terms
                    else "locate the key visual evidence requested by the question"
                )
                action = json.dumps(
                    {
                        "name": "Point",
                        "parameters": {
                            "image": "img_1",
                            "description": description,
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                return (
                    "Observation: a successful real visual-tool result is required before the final answer. "
                    "Do not answer yet. Output one concise REASONING line followed by exactly "
                    f"ACTION_JSON: {action}"
                )
            if "Point" in real_tools:
                description = (
                    "locate the " + " ".join(sorted(target_terms)[:6])
                    if target_terms
                    else "locate the key visual evidence requested by the question"
                )
                action = json.dumps(
                    {
                        "name": "Point",
                        "parameters": {
                            "image": "img_1",
                            "description": description,
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                return (
                    "Observation: a successful real visual-tool result is required before the final answer. "
                    "Do not answer yet. Output exactly one concise <think> rationale followed by "
                    f"<tool_call>{action}</tool_call>."
                )
            if task_name == "jigsaw_blink" and "DetectBlackArea" in real_tools:
                return (
                    "Observation: call the real DetectBlackArea tool on img_1 before answering. "
                    + action_syntax
                    + ' Use exactly this JSON with numeric values: {"name":"DetectBlackArea","parameters":{"image":"img_1","threshold":50,"min_area":100}}.'
                )
            return (
                "Observation: before summary or final answer, gather one concrete visual evidence item using a benchmark tool. "
                "Choose the tool based on what the question requires: OCR for visible text, Crop for local details, Point for grounding, "
                f"or AStarWithPixelCoordinate for paths after coordinates are known. {action_syntax}"
            )
        if tool_name == "OCR":
            return (
                "Observation: before summary or final answer, call OCR on the relevant image to capture verbatim text and regions. "
                f"{action_syntax} Example JSON: {{\"name\":\"OCR\",\"parameters\":{{\"image\":\"img_1\"}}}}."
            )
        if tool_name == "Point":
            return (
                "Observation: before summary or final answer, call a visual grounding benchmark tool such as Point on the relevant image. "
                f"{action_syntax} Example JSON: {{\"name\":\"Point\",\"parameters\":{{\"image\":\"img_1\",\"description\":\"locate the key visual evidence\"}}}}."
            )
        if tool_name == "Crop":
            return (
                "Observation: before summary or final answer on this fine-grained visual task, crop the local region that contains the key object/detail. "
                f"{action_syntax} Example JSON: {{\"name\":\"Crop\",\"parameters\":{{\"image\":\"img_1\",\"coordinates\":[0,0,256,256]}}}}. "
                "Choose tighter coordinates if you can infer the relevant region."
            )
        return f"Observation: call {tool_name} before summary or final answer."

    def evidence_recheck_observation(issue: str | None = None, required_tool: str | None = None) -> str:
        target_hint = ", ".join(sorted(focus_terms)[:8]) or "the key object/detail asked about in the question"
        issue_text = f" The previous visual evidence is not reliable: {issue}" if issue else ""
        if required_tool:
            tool_hint = f"You must call {required_tool} now, with new/tighter parameters targeting the requested visual evidence."
        else:
            tool_hint = "Use Point, Crop, or OCR with a description/region explicitly targeting the requested object/detail. If the previous tool hit the wrong object or returned no text, do not repeat the same tool with the same parameters; switch to a tighter Crop or a different target description."
        return (
            "Observation: before the final answer, re-check the visual evidence after the summary. "
            + issue_text
            + f" Target terms: {target_hint}. "
            + tool_hint
            + " Do not answer from unsupported prose; call one valid benchmark tool now."
        )

    def post_summary_required_tool() -> str | None:
        if os.environ.get("BENCHMARK_FORCE_POST_SUMMARY_CROP") == "1" and post_summary_benchmark_tool_calls == 0:
            return "Crop"
        return None

    def summary_force_gate_met() -> bool:
        """Force a summary only after useful multi-step visible reasoning."""
        min_tool_calls_for_summary = int(
            os.environ.get(
                "BENCHMARK_V5_MIN_TOOL_CALLS",
                "2",
            )
        ) if v5_hidden_summary else 2
        observed_tool_calls = (
            successful_benchmark_tool_calls
            if v5_hidden_summary
            and os.environ.get("BENCHMARK_V5_GATE_ON_SUCCESSFUL_TOOLS") == "1"
            else tool_calls
        )
        if observed_tool_calls < min_tool_calls_for_summary:
            return False
        unpruned_reasoning = "\n".join(
            str(step.get("reasoning") or "") for step in assistant_steps
        )
        min_reasoning_tokens = int(
            os.environ.get("BENCHMARK_V5_MIN_REASONING_TOKENS", "80")
        ) if v5_hidden_summary else 80
        return estimate_text_tokens(unpruned_reasoning) >= min_reasoning_tokens

    def v5_evidence_gate_met() -> bool:
        """Only summarize a completed, evidence-bearing decision prefix."""
        if not summary_force_gate_met():
            return False
        if task_name != "jigsaw_blink":
            return True
        if os.environ.get("BENCHMARK_V5_REQUIRE_ALL_JIGSAW_CANDIDATES", "1") == "0":
            return successful_benchmark_tool_calls > 0
        expected_candidates = {
            f"img_{index}" for index in range(2, initial_image_count + 1)
        }
        # Jigsaw summary is unsafe while any candidate has not been composed
        # into the real canvas. A crop plus one insertion is not a complete
        # candidate comparison.
        return bool(expected_candidates) and expected_candidates.issubset(
            successful_jigsaw_insert_candidates
        )

    def estimated_message_chars() -> int:
        total = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            else:
                total += len(json.dumps(json_safe(content), ensure_ascii=False))
        return total

    def maybe_selective_pressure_prune() -> None:
        if os.environ.get("BENCHMARK_SELECTIVE_PRESSURE_FALLBACK") != "1":
            return
        char_threshold = int(os.environ.get("BENCHMARK_SELECTIVE_PRESSURE_CHARS", "90000"))
        repeated_threshold = int(os.environ.get("BENCHMARK_SELECTIVE_PRESSURE_REPEATS", "3"))
        reason = None
        if estimated_message_chars() > char_threshold:
            reason = "context_pressure"
        elif blocked_repeated_tool_calls >= repeated_threshold:
            reason = "repeated_weak_tool_loop"
        if not reason:
            return
        pressure_indices: list[int] = []
        for step in assistant_steps:
            step_index = int(step["step_index"])
            if step.get("pruned") or not step.get("reasoning"):
                continue
            pressure_indices.append(step_index)
        pruned_indices = [
            step_index
            for step_index in pressure_indices
            if prune_step_reasoning(
                step_index,
                "selective_pressure_pruned",
                allow_empty_action=True,
                keep_structured_evidence=True,
            )
        ]
        if pruned_indices:
            prune_events.append(
                {
                    "status": "selective_pressure_fallback",
                    "reason": reason,
                    "prune_indices": pruned_indices,
                    "estimated_message_chars": estimated_message_chars(),
                    "blocked_repeated_tool_calls": blocked_repeated_tool_calls,
                }
            )

    def add_assistant_step(message_index: int, turn_round: Any, model_text: str, reasoning: str | None) -> None:
        nonlocal reasoning_chars_before_prune
        step = {
            "step_index": len(assistant_steps),
            "message_index": message_index,
            "turn_round": turn_round,
            "model_text": model_text,
            "reasoning": reasoning or "",
            "action_block": extract_action_block(model_text),
            "pruned": False,
            "pruned_reasoning_chars": 0,
            # Set true only after a real benchmark action executes and returns
            # a successful observation. Rejected, malformed, final, and
            # summary-only turns must never influence the KL prune decision.
            "eligible_for_summary_kl": False,
        }
        reasoning_chars_before_prune += len(step["reasoning"])
        assistant_steps.append(step)

    def prune_step_reasoning(
        step_index: int,
        marker_key: str | None = None,
        allow_empty_action: bool = False,
        keep_visual_anchors: bool = False,
        keep_structured_evidence: bool = False,
        allow_reprune: bool = False,
    ) -> bool:
        nonlocal reasoning_chars_pruned, reasoning_rewrite_events, reasoning_rewrite_time_sec
        rewrite_started = time.perf_counter()
        already_pruned = step_index in pruned_step_indices
        if already_pruned and not allow_reprune:
            return False
        step = assistant_steps[step_index]
        action_block = str(step.get("action_block") or "").strip()
        message_index = int(step["message_index"])
        current_content = str(messages[message_index].get("content") or "")
        prune_source = current_content if already_pruned else str(step.get("reasoning") or "")
        prune_text = reasoning_prose_without_action(prune_source, action_block)
        anchor_text = (
            extract_visual_anchor_text(
                prune_text,
                strict=os.environ.get("BENCHMARK_KEEP_VISUAL_ANCHORS_STRICT") == "1",
            )
            if keep_visual_anchors
            else ""
        )
        evidence_text = extract_structured_visual_evidence_text(prune_text) if keep_structured_evidence else ""
        if not action_block and not anchor_text and not evidence_text and not allow_empty_action:
            return False
        replacement_parts = [part for part in (evidence_text, anchor_text, action_block) if part]
        replacement = "\n\n".join(replacement_parts) or ""
        messages[message_index]["content"] = replacement
        step["pruned"] = True
        pruned_step_indices.add(step_index)
        replacement_reasoning_len = len(
            reasoning_prose_without_action(replacement, action_block)
        )
        step_pruned_chars, newly_pruned_chars = updated_step_pruned_chars(
            original_reasoning_chars=len(str(step.get("reasoning") or "")),
            replacement_reasoning_chars=replacement_reasoning_len,
            previous_pruned_chars=int(step.get("pruned_reasoning_chars", 0)),
        )
        step["pruned_reasoning_chars"] = step_pruned_chars
        reasoning_chars_pruned += newly_pruned_chars
        for turn in turns:
            if turn.get("_assistant_step_index") == step_index:
                turn["reasoning_pruned_from_context"] = True
                turn["pruned_context_content"] = replacement
                if anchor_text:
                    turn["retained_visual_anchor_text"] = anchor_text
                if evidence_text:
                    turn["retained_structured_visual_evidence_text"] = evidence_text
                if marker_key:
                    turn[marker_key] = True
                break
        reasoning_rewrite_time_sec += time.perf_counter() - rewrite_started
        reasoning_rewrite_events += 1
        return True

    def maybe_prune_after_summary(summary_result: dict[str, Any], current_step_index: int | None) -> bool:
        nonlocal summary_kl_time_sec, last_v5_summary_accepted
        last_v5_summary_accepted = False
        if enable_evidence_only_pruning:
            prune_all_prose = os.environ.get("BENCHMARK_PRUNE_ALL_PROSE") == "1"
            keep_visual_anchors = os.environ.get("BENCHMARK_KEEP_VISUAL_ANCHORS") == "1"
            evidence_only_indices: list[int] = []
            for step in assistant_steps:
                step_index = int(step["step_index"])
                if step.get("pruned") or not step.get("reasoning"):
                    continue
                if current_step_index is not None and step_index > current_step_index:
                    continue
                if (not prune_all_prose) and current_step_index is not None and current_step_index - step_index <= 1:
                    continue
                prune_text = reasoning_prose_without_action(str(step.get("reasoning") or ""), str(step.get("action_block") or ""))
                if prune_all_prose or lacks_tool_grounded_visual_evidence(prune_text):
                    evidence_only_indices.append(step_index)
            pruned_indices = [
                step_index
                for step_index in evidence_only_indices
                if prune_step_reasoning(
                    step_index,
                    "evidence_only_pruned",
                    allow_empty_action=True,
                    keep_visual_anchors=keep_visual_anchors,
                )
            ]
            if pruned_indices:
                prune_events.append(
                    {
                        "status": "evidence_only_rule",
                        "current_step_index": current_step_index,
                        "prune_indices": pruned_indices,
                        "rule": (
                            "after summary, remove assistant prose but retain short visual anchors; preserve action blocks/tool calls"
                            if prune_all_prose and keep_visual_anchors
                            else "after summary, remove all assistant prose reasoning; preserve action blocks/tool calls"
                            if prune_all_prose
                            else "after summary, remove assistant reasoning without tool-grounded visual evidence; preserve action blocks/tool calls"
                        ),
                    }
                )
        if enable_visual_state_pruning:
            visual_prune_indices: list[int] = []
            for step in assistant_steps:
                step_index = int(step["step_index"])
                if step.get("pruned") or not step.get("reasoning"):
                    continue
                if current_step_index is not None and step_index > current_step_index:
                    continue
                if current_step_index is not None and current_step_index - step_index <= 1:
                    continue
                prune_text = reasoning_prose_without_action(str(step.get("reasoning") or ""), str(step.get("action_block") or ""))
                if looks_redundant_reasoning(prune_text):
                    visual_prune_indices.append(step_index)
            pruned_indices = [
                step_index
                for step_index in visual_prune_indices
                if prune_step_reasoning(step_index, "visual_state_pruned")
            ]
            if pruned_indices:
                prune_events.append(
                    {
                        "status": "visual_state_rule",
                        "current_step_index": current_step_index,
                        "prune_indices": pruned_indices,
                        "rule": "prune redundant non-visual natural-language reasoning only; preserve action blocks/tool calls",
                    }
                )
        if not enable_summary_kl_pruning:
            last_v5_summary_accepted = True
            return True
        summary_text = str(summary_result.get("message") or json.dumps(summary_result.get("summary", {}), ensure_ascii=False)).strip()
        candidates = [
            step
            for step in assistant_steps
            if v5_step_is_kl_candidate(
                step,
                v5_hidden_summary=v5_hidden_summary,
                include_prepruned_override=(
                    os.environ.get("BENCHMARK_V5_INCLUDE_PREPRUNED_CANDIDATES") == "1"
                ),
            )
            and (
                not v5_hidden_summary
                or step.get("eligible_for_summary_kl")
                or os.environ.get("BENCHMARK_V5_ALL_REASONING_CANDIDATES") == "1"
            )
            and (current_step_index is None or int(step["step_index"]) <= current_step_index)
        ]
        if not summary_text or not candidates:
            if v5_hidden_summary:
                reason = "empty_summary_or_no_verified_tool_reasoning"
                v5_summary_gate_rejections.append({"reason": reason})
                prune_events.append({"status": "summary_gate_rejected", "reason": reason})
                return False
            return True
        kl_started = time.perf_counter()
        try:
            kl_result = log_summarization_kl(
                base_url=base_url,
                model=model,
                system_prompt=system_prompt_text,
                summary_text=summary_text,
                segments=candidates,
                top_logprobs=summary_kl_top_logprobs,
                timeout=timeout,
                prune_threshold=summary_kl_prune_threshold,
                min_high_tokens=summary_kl_prune_min_tokens,
                max_response_chars=summary_kl_max_response_chars,
                include_prepruned=v5_hidden_summary,
            )
        except Exception as exc:
            prune_events.append({"status": "error", "error": repr(exc), "current_step_index": current_step_index})
            if v5_hidden_summary:
                v5_summary_gate_rejections.append({"reason": "kl_error", "error": repr(exc)})
                return False
            return True
        finally:
            summary_kl_time_sec += time.perf_counter() - kl_started
        low_kl_indices = set(kl_result.get("prune_indices", []))
        if v5_hidden_summary:
            total_reasoning_chars = sum(
                len(str(step.get("reasoning") or "")) for step in candidates
            )
            projected_retained_chars = 0
            for step in candidates:
                step_index = int(step["step_index"])
                if step_index in low_kl_indices:
                    continue
                if v5_kl_delete_only:
                    projected_retained_chars += len(str(step.get("reasoning") or ""))
                else:
                    projected_retained_chars += len(
                        extract_structured_visual_evidence_text(
                            str(step.get("reasoning") or "")
                        )
                    )
            proposed_pruned_fraction = (
                total_reasoning_chars - projected_retained_chars
            ) / max(1, total_reasoning_chars)
            min_successful_tools = int(
                os.environ.get("BENCHMARK_V5_MIN_SUCCESSFUL_TOOLS", "2")
            )
            rejection_reason = v5_pruning_rejection_reason(
                kl_status=str(kl_result.get("status")),
                successful_tool_calls=successful_benchmark_tool_calls,
                min_successful_tools=min_successful_tools,
                action_blocks_preserved=True,
            )
            if not rejection_reason and task_name == "jigsaw_blink" and not v5_evidence_gate_met():
                rejection_reason = "incomplete_jigsaw_candidate_evidence"
            if rejection_reason:
                gate_record = {
                    "reason": rejection_reason,
                    "proposed_pruned_fraction": proposed_pruned_fraction,
                    "successful_benchmark_tool_calls": successful_benchmark_tool_calls,
                    "current_step_index": current_step_index,
                }
                v5_summary_gate_rejections.append(gate_record)
                prune_events.append({"status": "summary_gate_rejected", **gate_record})
                return False
        keep_kl_structured_evidence = os.environ.get("BENCHMARK_KL_PRUNE_V3_STRUCTURED") == "1"
        # Standard V5 rewrites high-KL reasoning to structured visual evidence.
        # The delete-only ablation keeps high-KL reasoning verbatim and uses the
        # generated summary only inside the KL calculation.
        if v5_hidden_summary and not v5_kl_delete_only:
            for step in candidates:
                step_index = int(step["step_index"])
                if step_index not in low_kl_indices:
                    prune_step_reasoning(
                        step_index,
                        "summary_kl_high_visual_retained",
                        allow_empty_action=True,
                        keep_structured_evidence=True,
                    )
        for step_index in low_kl_indices:
            prune_step_reasoning(
                step_index,
                "summary_kl_low_deleted" if v5_hidden_summary else "summary_kl_pruned",
                allow_empty_action=True,
                keep_structured_evidence=False if v5_hidden_summary else keep_kl_structured_evidence,
                allow_reprune=v5_hidden_summary,
            )
        prune_events.append(
            {
                "status": kl_result.get("status"),
                "current_step_index": current_step_index,
                "summary_chars": len(summary_text),
                "prune_indices": kl_result.get("prune_indices", []),
                "segments": kl_result.get("segments", []),
                "raw_token_count": kl_result.get("raw_token_count", 0),
                "raw_rkl_min": kl_result.get("raw_rkl_min"),
                "raw_rkl_max": kl_result.get("raw_rkl_max"),
                "truncated": kl_result.get("truncated", False),
                "params": {
                    "top_logprobs": summary_kl_top_logprobs,
                    "prune_threshold": summary_kl_prune_threshold,
                    "min_high_tokens": summary_kl_prune_min_tokens,
                    "max_response_chars": summary_kl_max_response_chars,
                    "structured_evidence_v3": keep_kl_structured_evidence,
                    "v5_hidden_summary": v5_hidden_summary,
                },
            }
        )
        last_v5_summary_accepted = True
        return True

    def generate_v5_hidden_summary() -> dict[str, Any]:
        """Generate a private summary from executed evidence, outside the agent loop."""
        verified_tool_records: list[dict[str, Any]] = []
        for turn in turns:
            tool_call_record = turn.get("tool_call") or {}
            tool_result_record = turn.get("tool_result") or {}
            if (
                turn.get("counts_as_benchmark_tool_call")
                and str(tool_result_record.get("status", "")).lower() == "success"
            ):
                verified_tool_records.append(
                    {
                        "tool": tool_call_record.get("name"),
                        "parameters": tool_call_record.get("parameters", {}),
                        "result": tool_result_record,
                    }
                )
        verified_reasoning = [
            {
                "round": step.get("turn_round"),
                "reasoning": short_text(step.get("reasoning", ""), max_chars=800),
            }
            for step in assistant_steps
            if step.get("eligible_for_summary_kl") and str(step.get("reasoning") or "").strip()
        ]
        source = {
            "question": item.get("text", ""),
            "verified_tool_records": verified_tool_records,
            "structured_visual_evidence": accumulated_visual_evidence,
            "reasoning_before_successful_tools": verified_reasoning,
        }
        prompt = (
            "Create a compact visual-state checkpoint from the supplied data. "
            "Use only successful tool records as verified facts. Reasoning text may contain guesses: "
            "retain it only as an uncertainty unless a tool record supports it. Do not invent an action, "
            "do not mention a tool result that is absent, and do not choose or change the final answer. "
            "Return one <response> block containing compact JSON with keys facts, visual_evidence, "
            "uncertainties, and next_decision.\n\n"
            + short_text(json.dumps(json_safe(source), ensure_ascii=False), max_chars=12000)
        )
        try:
            generated = call_chat(
                base_url,
                model,
                [
                    {
                        "role": "system",
                        "content": "You are a faithful evidence compressor, not a problem-solving agent.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=min(1024, max_tokens),
                temperature=0.0,
                timeout=timeout,
            )
        except Exception as exc:
            return {
                "status": "failed",
                "message": "",
                "error": repr(exc),
                "backend": "hidden_model_gate",
            }
        summary_text = parse_tag(generated, "response") or generated
        summary_text = summary_text.strip()
        return {
            "status": "success" if summary_text else "failed",
            "message": summary_text,
            "summary": {"generated_checkpoint": summary_text},
            "backend": "hidden_model_gate",
            "counts_as_benchmark_tool_call": False,
        }

    for round_idx in range(1, max_rounds + 1):
        maybe_selective_pressure_prune()
        model_text = call_chat(base_url, model, messages, max_tokens, temperature, timeout)
        last_model_text = model_text
        messages.append({"role": "assistant", "content": model_text})
        assistant_message_index = len(messages) - 1
        reasoning, reasoning_source = extract_reasoning(model_text)
        add_assistant_step(assistant_message_index, round_idx, model_text, reasoning)
        current_step_index = len(assistant_steps) - 1
        if retain_structured_visual_evidence:
            prune_step_reasoning(current_step_index, "all_reasoning_visual_evidence_retained", allow_empty_action=True, keep_visual_anchors=True, keep_structured_evidence=True)
        elif delete_reasoning:
            prune_step_reasoning(current_step_index, "all_reasoning_pruned", allow_empty_action=True)
        response_text = parse_final_response(model_text)
        tool_call = parse_tool_call(model_text)
        # XML, Nemotron FINAL, boxed answers, and a bare option letter are all
        # valid visible final actions. Accept them before issuing any recovery
        # prompt so native Nemotron replies do not enter ten bogus retries.
        # Raw prose is valid visible reasoning for these benchmarks.  Only a
        # genuinely absent rationale is a format violation; requiring a
        # particular <think> wrapper would misreport normal tool traces.
        if (
            require_visible_reasoning
            and reasoning_source == "missing"
            # The summary action carries its own structured state and is not a
            # benchmark-tool reasoning step.  All real benchmark tools and a
            # final action still require visible reasoning.
            and (tool_call is None or str(tool_call.get("name", "")) != SUMMARY_TOOL_NAME)
        ):
            # A tool action without visible rationale cannot participate in
            # delete-reasoning or V3 KL pruning.  Do not dispatch it: require
            # a corrective turn with actual task-specific reasoning first.
            format_retries += 1
            observation = (
                "Observation: your last action omitted the required visible reasoning. "
                "Do not execute or repeat that action yet. First write a concise, "
                "task-specific rationale, then output exactly one action using "
                + (
                    "REASONING plus ACTION_JSON or FINAL."
                    if is_nemotron_model
                    else "<think>...</think> plus one valid <tool_call> JSON block or a final response."
                )
            )
            messages.append({"role": "user", "content": observation})
            turns.append(
                {
                    "round": round_idx,
                    "_assistant_step_index": current_step_index,
                    "model": model_text,
                    "reasoning": reasoning,
                    "reasoning_source": reasoning_source,
                    "format_error": "missing_visible_reasoning",
                    "observation": observation,
                    "counts_as_benchmark_tool_call": False,
                }
            )
            continue

        required_tool = missing_required_visual_tool()
        if response_text is not None and tool_call is None and (tool_calls < min_tool_calls or required_tool):
            observation = (
                "Observation: no benchmark tool has been called yet. Continue as the same agent. You must call at least one "
                "useful benchmark tool before any summary or final answer. Choose Point, OCR, Crop, AStarWithPixelCoordinate, "
                f"or another relevant tool for the current task. Current images: {', '.join(f'img_{i + 1}' for i in range(len(images)))}."
                if not required_tool
                else required_tool_observation(required_tool)
            )
            messages.append({"role": "user", "content": observation})
            turns.append({"round": round_idx, "_assistant_step_index": current_step_index, "model": model_text, "reasoning": reasoning, "reasoning_source": reasoning_source, "observation": observation})
            continue

        if (
            response_text is not None
            and tool_call is None
            and summary_tool_calls == 0
            and summary_required
            and summary_force_gate_met()
        ):
            observation = summary_tool_call_instruction(
                "Observation: before the final response, call summarize_the_task once to summarize your current reasoning, "
                "completed steps, current task state, key observations, and next decision. This summary tool does not count "
                "as a benchmark tool call.",
                summary_mode=summary_mode,
            )
            messages.append({"role": "user", "content": observation})
            turns.append({"round": round_idx, "_assistant_step_index": current_step_index, "model": model_text, "reasoning": reasoning, "reasoning_source": reasoning_source, "observation": observation})
            continue

        if (
            response_text is not None
            and tool_call is None
            and enable_evidence_recheck
            and enable_evidence_only_pruning
            and summary_tool_calls > 0
            and (
                unresolved_evidence_issue
                or forced_recheck_tool
                or ((not evidence_recheck_on_issue_only) and post_summary_benchmark_tool_calls == 0)
            )
            and round_idx < max_rounds
        ):
            required_post_summary_tool = post_summary_required_tool()
            observation = evidence_recheck_observation(
                unresolved_evidence_issue,
                required_post_summary_tool or forced_recheck_tool,
            )
            messages.append({"role": "user", "content": observation})
            turns.append(
                {
                    "round": round_idx,
                    "_assistant_step_index": current_step_index,
                    "model": model_text,
                    "reasoning": reasoning,
                    "reasoning_source": reasoning_source,
                    "observation": observation,
                    "format_error": "evidence_recheck_required_before_final",
                    "unresolved_evidence_issue": unresolved_evidence_issue,
                    "forced_recheck_tool": required_post_summary_tool or forced_recheck_tool,
                }
            )
            continue

        if response_text is not None and tool_call is None and (tool_calls >= min_tool_calls or round_idx == max_rounds):
            if (
                v5_hidden_summary
                and last_v5_summary_result is not None
                and os.environ.get("BENCHMARK_V5_POST_FINAL_KL") == "1"
            ):
                # Apply the same evidence-only KL policy to reasoning emitted
                # with the final action. The parsed FINAL action is preserved;
                # this changes the public token footprint, never the score or
                # answer-selection gate.
                maybe_prune_after_summary(
                    last_v5_summary_result,
                    current_step_index,
                )
            if (
                summary_tool_calls > 0
                and compact_cleanup_attempts < 2
                and final_answer_needs_evidence_revision(response_text, item.get("text"))
                and round_idx < max_rounds
            ):
                compact_cleanup_attempts += 1
                observation = (
                    "Observation: the proposed final answer is not sufficiently supported by retained visual/tool evidence. "
                    "Revise using only the current image, retained tool observations, and structured visual-state summary. "
                    "If the question lists answer choices, output the single closest supported option letter. "
                    "If it asks for text or a number, copy the exact short text/number supported by OCR or visual evidence. "
                    "Do not answer unknown/none/unable unless the task explicitly permits that as an answer. "
                    "Return one compact <response>...</response> block."
                )
                messages.append({"role": "user", "content": observation})
                turns.append(
                    {
                        "round": round_idx,
                        "_assistant_step_index": current_step_index,
                        "model": model_text,
                        "reasoning": reasoning,
                        "reasoning_source": reasoning_source,
                        "observation": observation,
                        "format_error": "final_answer_evidence_revision_required",
                    }
                )
                continue
            if (
                summary_tool_calls > 0
                and compact_cleanup_attempts < 2
                and needs_compact_answer_cleanup(response_text)
                and round_idx < max_rounds
            ):
                compact_cleanup_attempts += 1
                observation = (
                    "Observation: compact the final answer for scoring. Output one response block containing only the answer token/string, "
                    "option, coordinate/path, number, or short phrase required by the task. Do not include explanation, tool names, unsupported prose, "
                    "nested response tags, or a full sentence."
                )
                messages.append({"role": "user", "content": observation})
                turns.append(
                    {
                        "round": round_idx,
                        "_assistant_step_index": current_step_index,
                        "model": model_text,
                        "reasoning": reasoning,
                        "reasoning_source": reasoning_source,
                        "observation": observation,
                        "format_error": "compact_answer_cleanup_required",
                    }
                )
                continue
            final_text = response_text
            turns.append({"round": round_idx, "_assistant_step_index": current_step_index, "model": model_text, "reasoning": reasoning, "reasoning_source": reasoning_source, "final": response_text})
            break

        if tool_call is None:
            format_retries += 1
            observation = (
                "Observation: I could not parse a tool call or final response from your last message. Continue naturally as the agent, "
                + (
                    'write REASONING: ... then ACTION_JSON: {"name":"ToolName","parameters":{...}} if you need a tool, '
                    "or REASONING: ... then FINAL: A if you are done."
                    if is_nemotron_model
                    else "using one valid JSON object inside <tool_call>...</tool_call> if you need a tool, or <response>...</response> if you are done."
                )
            )
            messages.append({"role": "user", "content": observation})
            turns.append({"round": round_idx, "_assistant_step_index": current_step_index, "model": model_text, "reasoning": reasoning, "reasoning_source": reasoning_source, "format_error": "missing_action", "observation": observation})
            continue

        name = str(tool_call.get("name", ""))
        params = tool_call.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        params, parameters_repaired = repair_tool_parameters(
            name,
            params,
            str(item.get("text") or ""),
            available_image_count=len(images),
        )
        if parameters_repaired:
            tool_parameter_repairs += 1
            tool_call = dict(tool_call)
            tool_call["parameters"] = params
        tool_signature = canonical_tool_signature(name, params)
        if name == SUMMARY_TOOL_NAME and summary_mode == "visual_state":
            params = dict(params)
            params["visual_evidence"] = merge_visual_evidence(accumulated_visual_evidence, params.get("visual_evidence", {}))
            key_observations = params.get("key_observations", [])
            if not isinstance(key_observations, list):
                key_observations = [short_text(key_observations)]
            key_observations.append("Preserved structured visual evidence from prior tool observations.")
            params["key_observations"] = key_observations
        required_tool = missing_required_visual_tool()
        if name == SUMMARY_TOOL_NAME and (tool_calls < min_tool_calls or required_tool or not summary_force_gate_met()):
            observation = (
                "Observation: summarize_the_task is memory only. Before using it, make at least two real benchmark tool calls "
                "and accumulate enough visible reasoning to make a compact summary useful. First gather another concrete visual observation."
                if not required_tool
                else required_tool_observation(required_tool)
            )
            messages.append({"role": "user", "content": observation})
            turns.append(
                {
                    "round": round_idx,
                    "_assistant_step_index": current_step_index,
                    "model": model_text,
                    "reasoning": reasoning,
                    "reasoning_source": reasoning_source,
                    "tool_call": {"name": name, "parameters": tool_call.get("parameters", {})},
                    "format_error": "summary_before_benchmark_tool",
                    "observation": observation,
                    "counts_as_benchmark_tool_call": False,
                }
            )
            continue
        if (not summary_before_final_only) and summary_required and summary_tool_calls == 0 and summary_force_gate_met() and name != SUMMARY_TOOL_NAME:
            observation = summary_tool_call_instruction(
                "Observation: summarize_the_task must be called now before any further benchmark tool calls or final answer. "
                "Do not repeat the previous benchmark tool call yet.",
                summary_mode=summary_mode,
            )
            messages.append({"role": "user", "content": observation})
            turns.append(
                {
                    "round": round_idx,
                    "_assistant_step_index": current_step_index,
                    "model": model_text,
                    "reasoning": reasoning,
                    "reasoning_source": reasoning_source,
                    "tool_call": {"name": name, "parameters": tool_call.get("parameters", {})},
                    "format_error": "summary_required_before_more_tools",
                    "observation": observation,
                    "counts_as_benchmark_tool_call": False,
                }
            )
            continue
        if (
            enable_evidence_recheck
            and enable_evidence_only_pruning
            and summary_tool_calls > 0
            and (forced_recheck_tool or post_summary_required_tool())
            and name != (post_summary_required_tool() or forced_recheck_tool)
            and name != SUMMARY_TOOL_NAME
        ):
            blocked_repeated_tool_calls += 1
            required_recheck_tool = post_summary_required_tool() or forced_recheck_tool
            observation = evidence_recheck_observation(
                f"{unresolved_evidence_issue or 'A structured visual re-check is still required.'} The requested re-check tool was {required_recheck_tool}, but the model tried {name}.",
                required_recheck_tool,
            )
            messages.append({"role": "user", "content": observation})
            turns.append(
                {
                    "round": round_idx,
                    "_assistant_step_index": current_step_index,
                    "model": model_text,
                    "reasoning": reasoning,
                    "reasoning_source": reasoning_source,
                    "tool_call": {"name": name, "parameters": tool_call.get("parameters", {})},
                    "format_error": "forced_evidence_recheck_tool_required",
                    "observation": observation,
                    "forced_recheck_tool": required_recheck_tool,
                    "counts_as_benchmark_tool_call": False,
                }
            )
            continue
        if (
            enable_evidence_recheck
            and enable_evidence_only_pruning
            and summary_tool_calls > 0
            and unresolved_evidence_issue
            and name != SUMMARY_TOOL_NAME
            and tool_signature in seen_tool_signatures
        ):
            blocked_repeated_tool_calls += 1
            required_tool = forced_recheck_tool or forced_recheck_tool_for_issue(unresolved_evidence_issue, unresolved_evidence_tool)
            observation = evidence_recheck_observation(
                f"{unresolved_evidence_issue} Do not repeat the same failed tool call parameters.",
                required_tool,
            )
            messages.append({"role": "user", "content": observation})
            turns.append(
                {
                    "round": round_idx,
                    "_assistant_step_index": current_step_index,
                    "model": model_text,
                    "reasoning": reasoning,
                    "reasoning_source": reasoning_source,
                    "tool_call": {"name": name, "parameters": tool_call.get("parameters", {})},
                    "format_error": "repeated_weak_evidence_tool_call",
                    "observation": observation,
                    "forced_recheck_tool": required_tool,
                    "counts_as_benchmark_tool_call": False,
                }
            )
            forced_recheck_tool = required_tool
            continue
        result, edited = runtime.call_tool(name, params, images)
        is_summary_tool = name == SUMMARY_TOOL_NAME
        if is_summary_tool:
            summary_tool_calls += 1
        else:
            tool_calls += 1
            called_benchmark_tool_names.append(name)
        new_images: list[Image.Image] = []
        append_edited = edited is not None and should_append_generated_image(
            len(images), initial_image_count, MAX_GENERATED_IMAGES
        )
        if append_edited:
            images.append(edited)
            new_images.append(edited)
        new_image_id = f"img_{len(images)}" if append_edited else None
        compact = compact_tool_result(result if isinstance(result, dict) else {"result": result})
        # A failed crop is a parameter/coordinate error, not a negative visual
        # observation.  Explicitly state the recovery rule so the next model
        # turn maps the source bbox into the target candidate's coordinate
        # frame rather than selecting/rejecting a piece from the failure.
        if (
            task_name == "jigsaw_blink"
            and name == "Crop"
            and isinstance(result, dict)
            and str(result.get("status", "")).lower() == "failed"
            and "out of bounds" in str(result.get("message", "")).lower()
        ):
            compact["coordinate_recovery"] = (
                "This is not visual evidence against the candidate. Crop coordinates are absolute for the target image. "
                "Map the img_1 box proportionally to this image size, or use InsertImage on img_1 with the hole box."
            )
        if new_image_id:
            compact["new_image_id"] = new_image_id
        summary_accepted = True
        if is_summary_tool:
            summary_accepted = maybe_prune_after_summary(compact, current_step_index)
            if os.environ.get("BENCHMARK_AUTO_POST_SUMMARY_QUADRANT_CROPS") == "1" and auto_post_summary_local_crops == 0 and images:
                grid_splits = int(os.environ.get("BENCHMARK_AUTO_POST_SUMMARY_GRID_CROPS", "2"))
                crop_records: list[dict[str, Any]] = []
                for label, crop, coords in grid_crops(images[0], grid_splits):
                    images.append(crop)
                    new_images.append(crop)
                    image_id = f"img_{len(images)}"
                    crop_records.append({"image_id": image_id, "region": label, "coordinates": coords})
                    accumulated_visual_evidence["crop_ids"].append(f"{image_id} = automatic {label} local crop {coords} from img_1")
                auto_post_summary_local_crops += len(crop_records)
                if crop_records:
                    compact["automatic_post_summary_local_crops"] = crop_records
                    post_summary_benchmark_tool_calls += 1
        else:
            seen_tool_signatures.add(tool_signature)
            result_status = str(
                (result if isinstance(result, dict) else {}).get("status", "")
            ).lower()
            if result_status == "success":
                successful_benchmark_tool_calls += 1
                assistant_steps[current_step_index]["eligible_for_summary_kl"] = True
                if task_name == "jigsaw_blink" and name == "InsertImage":
                    candidate_id = str(
                        tool_call.get("parameters", {}).get("image_to_insert", "")
                    ).strip()
                    if re.fullmatch(r"img_\d+", candidate_id):
                        successful_jigsaw_insert_candidates.add(candidate_id)
            if summary_tool_calls > 0:
                post_summary_benchmark_tool_calls += 1
            result_dict = result if isinstance(result, dict) else {"result": result}
            issue = tool_observation_issue(name, params, result_dict, focus_terms)
            if use_target_hit_gate and not issue:
                issue = target_hit_gate_issue(name, params, result_dict, target_terms)
            if enable_evidence_recheck and issue:
                unresolved_evidence_issue = issue
                unresolved_evidence_tool = name
                forced_recheck_tool = forced_recheck_tool_for_issue(issue, name)
                evidence_recheck_issues.append(
                    {
                        "round": round_idx,
                        "tool": name,
                        "parameters": json_safe(params),
                        "issue": issue,
                        "forced_recheck_tool": forced_recheck_tool,
                        "target_hit_gate": use_target_hit_gate,
                        "target_terms": sorted(target_terms),
                    }
                )
            elif enable_evidence_recheck and summary_tool_calls > 0:
                unresolved_evidence_issue = None
                unresolved_evidence_tool = None
                forced_recheck_tool = None
            accumulated_visual_evidence = merge_visual_evidence(
                accumulated_visual_evidence,
                extract_visual_evidence_from_tool(name, params, result if isinstance(result, dict) else {"result": result}, new_image_id),
            )

        if is_summary_tool and v5_hidden_summary:
            if not summary_accepted:
                # A voluntary but unsafe summary has no effect; continue from
                # the verified ordinary-tool context.
                continue
            hidden_summary_calls += 1
            last_v5_summary_result = compact
            v5_summary_gate_accepts += 1
            v5_summary_trigger_round = round_idx
            v5_summary_trigger_step_index = current_step_index
            v5_online_intervention = True
            if v5_summary_context_enabled:
                hidden_context = (
                    "[Privileged auxiliary summary; do not repeat it or treat it as an agent action.]\n"
                    + json.dumps(compact, ensure_ascii=False)
                    + "\nUse it only to help choose the next useful real tool or final answer."
                )
                messages.append({"role": "user", "content": hidden_context})
            # Deliberately omit both the summary call and result from `turns`.
            continue
        observation = (
            "Observation from tool call:\n"
            + json.dumps(compact, ensure_ascii=False)
            + "\n"
            + f"Current images: {', '.join(f'img_{i + 1}' for i in range(len(images)))}.\n"
            + (
                "Continue as the same agent. Use this summary to decide the next useful benchmark tool call or final response."
                if is_summary_tool
                else "Continue as the same agent. Use this observation in your own reasoning, then either call another useful tool or give the final response."
            )
        )
        if summary_required and summary_tool_calls == 0 and summary_force_gate_met() and not is_summary_tool:
            observation += (
                " You have not called summarize_the_task yet; decide when it would help, and call it at least once before the final response."
            )
        messages.append({"role": "user", "content": content_with_images(observation, new_images) if new_images else observation})
        turns.append(
            {
                "round": round_idx,
                "_assistant_step_index": current_step_index,
                "model": model_text,
                "reasoning": reasoning,
                "reasoning_source": reasoning_source,
                "tool_call": {"name": name, "parameters": tool_call.get("parameters", {})},
                "tool_result": compact,
                "counts_as_benchmark_tool_call": not is_summary_tool,
            }
        )
        if (
            v5_summary_enabled
            and not is_summary_tool
            and summary_tool_calls == 0
            and v5_evidence_gate_met()
        ):
            v5_summary_gate_attempts += 1
            compact_summary = generate_v5_hidden_summary()
            summary_tool_calls += 1
            summary_accepted = (
                str(compact_summary.get("status", "")).lower() == "success"
                and maybe_prune_after_summary(compact_summary, current_step_index)
            )
            if summary_accepted:
                hidden_summary_calls += 1
                last_v5_summary_result = compact_summary
                v5_summary_gate_accepts += 1
                v5_summary_trigger_round = round_idx
                v5_summary_trigger_step_index = current_step_index
                v5_online_intervention = True
                turns[-1]["v5_summary_triggered_after_tool"] = True
                if v5_summary_context_enabled:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[Privileged auxiliary summary; do not repeat it or treat it as an agent action.]\n"
                                + json.dumps(compact_summary, ensure_ascii=False)
                                + "\nUse this compact state with the current image and retained tool observations "
                                "to choose the next useful real tool or final answer."
                            ),
                        }
                    )
            elif not v5_summary_gate_rejections:
                v5_summary_gate_rejections.append(
                    {
                        "reason": "hidden_summary_generation_failed",
                        "error": compact_summary.get("error"),
                        "trigger_round": round_idx,
                    }
                )
    else:
        final_text = parse_final_response(last_model_text) or last_model_text

    if not final_text:
        final_text = parse_final_response(last_model_text) or last_model_text

    if (
        v5_hidden_summary
        and summary_tool_calls == 0
        and successful_benchmark_tool_calls
        >= int(os.environ.get("BENCHMARK_V5_MIN_SUCCESSFUL_TOOLS", "2"))
        and os.environ.get("BENCHMARK_V5_POST_ROLLOUT_KL") == "1"
    ):
        # Long tool trajectories may exhaust max_rounds without proposing a
        # normal final, so the ordinary pre-final gate never fires. Compress
        # the already generated trace with the same hidden-summary KL policy.
        # This is score-independent and does not synthesize an answer.
        v5_summary_gate_attempts += 1
        v5_post_rollout_diagnostic = True
        compact = generate_v5_hidden_summary()
        summary_tool_calls += 1
        if (
            str(compact.get("status", "")).lower() == "success"
            and maybe_prune_after_summary(compact, None)
        ):
            hidden_summary_calls += 1
            last_v5_summary_result = compact
            v5_summary_gate_accepts += 1
        elif not v5_summary_gate_rejections:
            v5_summary_gate_rejections.append(
                {
                    "reason": "post_rollout_hidden_summary_failed",
                    "error": compact.get("error"),
                }
            )

    if summary_required and summary_tool_calls == 0 and tool_calls == 0:
        evidence_instruction = required_tool_observation("EVIDENCE")
        for evidence_attempt in range(1, max_format_retries + 3):
            messages.append({"role": "user", "content": evidence_instruction})
            model_text = call_chat(base_url, model, messages, max_tokens, temperature, timeout)
            messages.append({"role": "assistant", "content": model_text})
            assistant_message_index = len(messages) - 1
            reasoning, reasoning_source = extract_reasoning(model_text)
            add_assistant_step(assistant_message_index, f"evidence_{evidence_attempt}", model_text, reasoning)
            current_step_index = len(assistant_steps) - 1
            tool_call = parse_tool_call(model_text)
            if not tool_call or str(tool_call.get("name", "")) == SUMMARY_TOOL_NAME:
                format_retries += 1
                turns.append(
                    {
                        "round": f"evidence_{evidence_attempt}",
                        "_assistant_step_index": current_step_index,
                        "model": model_text,
                        "reasoning": reasoning,
                        "reasoning_source": reasoning_source,
                        "format_error": "evidence_tool_recovery_failed",
                        "observation": evidence_instruction,
                    }
                )
                evidence_instruction = (
                    "Observation: a benchmark visual evidence tool is still required before summary or final answer. "
                    "Do not summarize and do not answer. Output exactly one benchmark tool call now, such as Point, OCR, Crop, "
                    "AStarWithPixelCoordinate, DetectBlackArea, Draw2DPath, or InsertImage."
                )
                continue
            name = str(tool_call.get("name", ""))
            params = tool_call.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            result, edited = runtime.call_tool(name, params, images)
            tool_calls += 1
            called_benchmark_tool_names.append(name)
            new_images: list[Image.Image] = []
            append_edited = edited is not None and should_append_generated_image(
                len(images), initial_image_count, MAX_GENERATED_IMAGES
            )
            if append_edited:
                images.append(edited)
                new_images.append(edited)
            new_image_id = f"img_{len(images)}" if append_edited else None
            compact = compact_tool_result(result if isinstance(result, dict) else {"result": result})
            if new_image_id:
                compact["new_image_id"] = new_image_id
            accumulated_visual_evidence = merge_visual_evidence(
                accumulated_visual_evidence,
                extract_visual_evidence_from_tool(name, params, result if isinstance(result, dict) else {"result": result}, new_image_id),
            )
            observation = (
                "Observation from tool call:\n"
                + json.dumps(compact, ensure_ascii=False)
                + "\n"
                + f"Current images: {', '.join(f'img_{i + 1}' for i in range(len(images)))}.\n"
                + "Continue as the same agent. Use this visual evidence, then call summarize_the_task before the final response."
            )
            messages.append({"role": "user", "content": content_with_images(observation, new_images) if new_images else observation})
            turns.append(
                {
                    "round": f"evidence_{evidence_attempt}",
                    "_assistant_step_index": current_step_index,
                    "model": model_text,
                    "reasoning": reasoning,
                    "reasoning_source": reasoning_source,
                    "tool_call": {"name": name, "parameters": tool_call.get("parameters", {})},
                    "tool_result": compact,
                    "counts_as_benchmark_tool_call": True,
                    "recovered_visual_evidence_tool": True,
                }
            )
            break
        if tool_calls == 0:
            format_failures += 1
            turns.append(
                {
                    "round": "evidence_recovery",
                    "model": "",
                    "reasoning": None,
                    "reasoning_source": "missing",
                    "format_error": "no_visual_evidence_tool_after_retries",
                    "final": final_text,
                }
            )

    if summary_required and summary_tool_calls == 0 and summary_force_gate_met():
        summary_instruction = summary_tool_call_instruction(
            "Before any final answer, call summarize_the_task now. This is required even if earlier tool-call formatting failed.",
            summary_mode=summary_mode,
        )
        max_summary_attempts = max(max_format_retries + 3, 6)
        for summary_attempt in range(1, max_summary_attempts + 1):
            messages.append({"role": "user", "content": summary_instruction})
            try:
                model_text = call_chat(base_url, model, messages, max_tokens, temperature, timeout)
            except requests.HTTPError as exc:
                body = exc.response.text if exc.response is not None else ""
                if "context length" not in body and "Requested token count exceeds" not in body:
                    raise
                format_failures += 1
                recovered_summary = "\n\n".join(
                    str(step.get("reasoning") or "").strip()
                    for step in assistant_steps
                    if str(step.get("reasoning") or "").strip()
                ) or last_model_text or "Context was too long for an additional model-generated summary call."
                params = {
                    "reasoning_so_far": recovered_summary[-2000:],
                    "completed_steps": ["Recovered summary locally because the model summary prompt exceeded the 32k context window."],
                    "current_state": "The conversation history became too long for another summary-generation turn; preserve tool/action history and prune covered reasoning.",
                    "visual_evidence": accumulated_visual_evidence if summary_mode == "visual_state" else {},
                    "key_observations": [recovered_summary[-500:]],
                    "uncertainties": ["Recovered locally; verify final answer against preserved tool observations."],
                    "next_decision": "Proceed to the final answer using retained tool calls, tool observations, and the compact summary.",
                }
                result, _ = runtime.call_tool(SUMMARY_TOOL_NAME, params, images)
                summary_tool_calls += 1
                compact = compact_tool_result(result if isinstance(result, dict) else {"result": result})
                compact["recovered_from_context_overflow"] = True
                maybe_prune_after_summary(compact, None)
                if v5_hidden_summary:
                    hidden_summary_calls += 1
                    if v5_summary_context_enabled:
                        messages.append({
                            "role": "user",
                            "content": "[Privileged auxiliary summary; do not repeat it or treat it as an agent action.]\n"
                            + json.dumps(compact, ensure_ascii=False),
                        })
                else:
                    turns.append(
                        {
                            "round": f"summary_{summary_attempt}",
                            "model": "",
                            "reasoning": recovered_summary[-2000:],
                            "reasoning_source": "context_overflow_recovery",
                            "format_error": "summary_context_overflow",
                            "tool_call": {"name": SUMMARY_TOOL_NAME, "parameters": params},
                            "tool_result": compact,
                            "counts_as_benchmark_tool_call": False,
                            "recovered_from_context_overflow": True,
                        }
                    )
                final_text = ""
                break
            messages.append({"role": "assistant", "content": model_text})
            assistant_message_index = len(messages) - 1
            reasoning, reasoning_source = extract_reasoning(model_text)
            add_assistant_step(assistant_message_index, f"summary_{summary_attempt}", model_text, reasoning)
            current_step_index = len(assistant_steps) - 1
            tool_call = parse_tool_call(model_text)
            if tool_call and str(tool_call.get("name", "")) == SUMMARY_TOOL_NAME:
                params = tool_call.get("parameters", {})
                if not isinstance(params, dict):
                    params = {}
                if summary_mode == "visual_state":
                    params = dict(params)
                    params["visual_evidence"] = merge_visual_evidence(accumulated_visual_evidence, params.get("visual_evidence", {}))
                result, _ = runtime.call_tool(SUMMARY_TOOL_NAME, params, images)
                summary_tool_calls += 1
                compact = compact_tool_result(result if isinstance(result, dict) else {"result": result})
                maybe_prune_after_summary(compact, current_step_index)
                observation = (
                    "Observation from tool call:\n"
                    + json.dumps(compact, ensure_ascii=False)
                    + "\n"
                    + f"Current images: {', '.join(f'img_{i + 1}' for i in range(len(images)))}.\n"
                    + "Use this summary to give the final response."
                )
                messages.append({"role": "user", "content": observation})
                if v5_hidden_summary:
                    hidden_summary_calls += 1
                else:
                    turns.append(
                        {
                            "round": f"summary_{summary_attempt}",
                            "_assistant_step_index": current_step_index,
                            "model": model_text,
                            "reasoning": reasoning,
                            "reasoning_source": reasoning_source,
                            "tool_call": {"name": SUMMARY_TOOL_NAME, "parameters": params},
                            "tool_result": compact,
                            "counts_as_benchmark_tool_call": False,
                        }
                    )
                final_text = ""
                break
            format_retries += 1
            if summary_attempt >= max_summary_attempts:
                format_failures += 1
                turns.append(
                    {
                        "round": f"summary_{summary_attempt}",
                        "_assistant_step_index": current_step_index,
                        "model": model_text,
                        "reasoning": reasoning,
                        "reasoning_source": reasoning_source,
                        "format_error": "summary_gate_format_failure",
                        "counts_as_benchmark_tool_call": False,
                    }
                )
                break
            summary_instruction = summary_tool_call_instruction(
                "Observation: that was not a valid summarize_the_task call. You must emit one valid summarize_the_task tool call now.",
                summary_mode=summary_mode,
            )

    parsed_existing_final = parse_final_response(final_text)
    if tool_calls > 0 and (
        "<tool_call>" in final_text
        or (parsed_existing_final is None and not re.search(r"\bFINAL\s*:", final_text, flags=re.I))
    ):
        final_instruction = (
            "No more tool calls are allowed. "
            + (
                "Base the answer only on the current image, retained tool observations, and structured visual evidence; "
                "ignore unsupported prior prose that was not grounded in a tool observation. "
                if enable_evidence_only_pruning
                else "Based on your own reasoning and the observations above, "
            )
            + (
                "give the final answer now as REASONING: one short justification followed by FINAL: the compact answer. "
                "Do not output ACTION_JSON."
                if is_nemotron_model
                else "give the final answer now. Put only the compact answer token/string, option, coordinate/path, number, or short phrase inside "
                "<response>...</response>. Do not output <tool_call> and do not include explanation."
            )
        )
        for final_attempt in range(1, max_format_retries + 2):
            messages.append({"role": "user", "content": final_instruction})
            try:
                model_text = call_chat(base_url, model, messages, max_tokens, temperature, timeout)
            except requests.HTTPError as exc:
                body = exc.response.text if exc.response is not None else ""
                if "context length" not in body and "Requested token count exceeds" not in body:
                    raise
                format_failures += 1
                final_text = parse_final_response(last_model_text) or last_model_text
                turns.append(
                    {
                        "round": f"final_{final_attempt}",
                        "model": "",
                        "reasoning": None,
                        "reasoning_source": "context_overflow_recovery",
                        "format_error": "final_context_overflow",
                        "final": final_text,
                        "recovered_from_context_overflow": True,
                    }
                )
                break
            messages.append({"role": "assistant", "content": model_text})
            assistant_message_index = len(messages) - 1
            reasoning, reasoning_source = extract_reasoning(model_text)
            add_assistant_step(assistant_message_index, f"final_{final_attempt}", model_text, reasoning)
            current_step_index = len(assistant_steps) - 1
            response_text = parse_final_response(model_text)
            final_text = response_text or model_text
            if reasoning and response_text is not None and parse_tool_call(model_text) is None:
                turns.append(
                    {
                        "round": f"final_{final_attempt}",
                        "_assistant_step_index": current_step_index,
                        "model": model_text,
                        "reasoning": reasoning,
                        "reasoning_source": reasoning_source,
                        "final": final_text,
                    }
                )
                break
            format_retries += 1
            if final_attempt > max_format_retries:
                format_failures += 1
                turns.append(
                    {
                        "round": f"final_{final_attempt}",
                        "_assistant_step_index": current_step_index,
                        "model": model_text,
                        "reasoning": reasoning,
                        "reasoning_source": reasoning_source,
                        "format_error": "final_format_after_retries",
                        "final": final_text,
                    }
                )
                break
            final_instruction = (
                "Observation: that was not a valid final response. Continue as the same agent and give the final answer only: "
                + (
                    "use only the current image, retained tool observations, and structured visual evidence; "
                    "ignore unsupported prior prose. "
                    if enable_evidence_only_pruning
                    else ""
                )
                + (
                    "REASONING: one short justification, then FINAL: the compact actual answer. Do not output ACTION_JSON."
                    if is_nemotron_model
                    else "one response block containing only the compact actual answer. Do not output <tool_call>, explanation, or nested response tags."
                )
            )

    return {
        "prediction": sanitize_final_answer(final_text),
        "raw_prediction": final_text,
        "turns": turns,
        "tool_calls": tool_calls,
        "summary_tool_calls": summary_tool_calls,
        "hidden_summary_calls": hidden_summary_calls,
        "image_count": len(images),
        "format_retries": format_retries,
        "format_failures": format_failures,
        "summary_kl_pruning_enabled": enable_summary_kl_pruning,
        "summary_kl_prune_events": prune_events,
        "summary_kl_pruned_steps": sorted(pruned_step_indices),
        "summary_kl_pruned_reasoning_chars": reasoning_chars_pruned,
        "summary_kl_pruned_reasoning_tokens_est": estimate_text_tokens("x" * reasoning_chars_pruned) if reasoning_chars_pruned else 0,
        "reasoning_rewrite_time_sec": reasoning_rewrite_time_sec,
        "reasoning_rewrite_events": reasoning_rewrite_events,
        "summary_kl_time_sec": summary_kl_time_sec,
        "reasoning_chars_before_prune": reasoning_chars_before_prune,
        "reasoning_tokens_before_prune_est": estimate_text_tokens("x" * reasoning_chars_before_prune) if reasoning_chars_before_prune else 0,
        "summary_mode": summary_mode,
        "summary_before_final_only": summary_before_final_only,
        "summary_compatibility_fallback": summary_compatibility_fallback,
        "v5_hidden_summary": v5_hidden_summary,
        "v5_kl_delete_only": v5_kl_delete_only,
        "v5_no_summary_context": v5_no_summary_context,
        "v5_aggressive_policy": {
            "visual_backed": v5_hidden_summary and not v5_kl_delete_only,
            "kl_delete_only": v5_kl_delete_only,
            "summary_context_injected": v5_summary_context_enabled,
            "min_tool_calls": int(os.environ.get("BENCHMARK_V5_MIN_TOOL_CALLS", "2")),
            "gate_on_successful_tools": os.environ.get("BENCHMARK_V5_GATE_ON_SUCCESSFUL_TOOLS") == "1",
            "min_reasoning_tokens": int(os.environ.get("BENCHMARK_V5_MIN_REASONING_TOKENS", "80")),
            "require_all_jigsaw_candidates": os.environ.get("BENCHMARK_V5_REQUIRE_ALL_JIGSAW_CANDIDATES", "1") != "0",
            "min_successful_tools": int(os.environ.get("BENCHMARK_V5_MIN_SUCCESSFUL_TOOLS", "2")),
            "allow_answer_change": True,
            "answer_lock_enabled": False,
            "online_trigger_after_tool": True,
            "all_reasoning_candidates": os.environ.get("BENCHMARK_V5_ALL_REASONING_CANDIDATES") == "1",
            "post_final_kl": os.environ.get("BENCHMARK_V5_POST_FINAL_KL") == "1",
            "require_successful_tool": os.environ.get("BENCHMARK_V5_REQUIRE_SUCCESSFUL_TOOL") == "1",
            "post_rollout_kl": os.environ.get("BENCHMARK_V5_POST_ROLLOUT_KL") == "1",
            "include_prepruned_candidates": os.environ.get("BENCHMARK_V5_INCLUDE_PREPRUNED_CANDIDATES") == "1",
        },
        "v5_summary_gate_attempts": v5_summary_gate_attempts,
        "v5_summary_gate_accepts": v5_summary_gate_accepts,
        "v5_summary_gate_rejections": v5_summary_gate_rejections,
        "v5_summary_trigger_round": v5_summary_trigger_round,
        "v5_summary_trigger_step_index": v5_summary_trigger_step_index,
        "v5_online_intervention": v5_online_intervention,
        "v5_post_rollout_diagnostic": v5_post_rollout_diagnostic,
        "successful_benchmark_tool_calls": successful_benchmark_tool_calls,
        "tool_parameter_repairs": tool_parameter_repairs,
        "successful_jigsaw_insert_candidates": sorted(successful_jigsaw_insert_candidates),
        "visual_state_pruning_enabled": enable_visual_state_pruning,
        "evidence_only_pruning_enabled": enable_evidence_only_pruning,
        "evidence_recheck_enabled": enable_evidence_recheck,
        "evidence_recheck_on_issue_only": evidence_recheck_on_issue_only,
        "post_summary_benchmark_tool_calls": post_summary_benchmark_tool_calls,
        "evidence_recheck_issues": evidence_recheck_issues,
        "unresolved_evidence_issue": unresolved_evidence_issue,
        "forced_recheck_tool": forced_recheck_tool,
        "blocked_repeated_tool_calls": blocked_repeated_tool_calls,
        "auto_post_summary_local_crops": auto_post_summary_local_crops,
        "target_hit_gate_enabled": use_target_hit_gate,
        "target_terms": sorted(target_terms),
        "tool_protocol": protocol,
        "visual_evidence": accumulated_visual_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="vspo,vsp,jigsaw_coco,jigsaw_blink,web_guichat,webquest,vstar,hrbench")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--out-dir", default="scripts/logs/tool_qwen3_vl_8b")
    parser.add_argument("--tools", default="", help="Comma-separated tools. Empty selects the task-specific original protocol.")
    parser.add_argument("--tool-profile", choices=["auto", "jigsaw", "vstar"], default="auto", help="Validate/select a benchmark-specific tool protocol.")
    parser.add_argument("--allow-cross-benchmark-tools", action="store_true", help="Explicitly run a legacy diagnostic with a non-native tool list; never use for new formal results.")
    parser.add_argument("--controller-addr", default=os.environ.get("TOOL_CONTROLLER_ADDR"))
    parser.add_argument("--allow-model-tool-fallback", action="store_true")
    parser.add_argument("--require-all-tools", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Per-task item cap for smoke tests.")
    parser.add_argument("--items-manifest", default=None, help="Optional JSONL manifest with disk-backed images; avoids loading the complete HF dataset in every shard.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split each task deterministically into this many independent worker shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index for --num-shards.")
    parser.add_argument("--resume-from", default=None, help="Read completed rows from this task-output directory while writing new rows to --out-dir.")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--min-tool-calls", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-format-retries", type=int, default=4)
    parser.add_argument("--summary-retry-attempts", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-require-visible-reasoning", action="store_true", help="Do not retry turns that omit <think>...</think>.")
    parser.add_argument("--enable-summary-kl-pruning", action="store_true", help="After summarize_the_task, prune historical assistant reasoning covered by the summary.")
    parser.add_argument("--summary-kl-top-logprobs", type=int, default=20)
    parser.add_argument("--summary-kl-prune-threshold", type=float, default=float(os.environ.get("TAU2_SUMMARY_KL_PRUNE_THRESHOLD", "0.2")))
    parser.add_argument("--summary-kl-prune-min-tokens", type=int, default=int(os.environ.get("TAU2_SUMMARY_KL_PRUNE_MIN_TOKENS", "2")))
    parser.add_argument("--summary-kl-max-response-chars", type=int, default=12000)
    parser.add_argument("--summary-mode", choices=["generic", "visual_state"], default=os.environ.get("BENCHMARK_SUMMARY_MODE", "generic"))
    parser.add_argument("--enable-visual-state-pruning", action="store_true", help="After visual-state summary, prune only redundant non-visual assistant reasoning; preserve tool calls/results.")
    parser.add_argument("--enable-evidence-only-pruning", action="store_true", help="After summary, prune assistant reasoning without tool-grounded visual evidence and force final answer to use retained visual evidence.")
    parser.add_argument("--enable-evidence-recheck", action="store_true", help="After summary/evidence-only pruning, require a post-summary visual re-check and block final answers after mismatched weak tool evidence.")
    parser.add_argument("--evidence-recheck-on-issue-only", action="store_true", help="When evidence recheck is enabled, re-check only after weak/mismatched evidence instead of always requiring a post-summary tool call.")
    parser.add_argument("--summary-before-final-only", action="store_true", help="Do not force summarize_the_task immediately after the first benchmark tool; require it only before final answers.")
    parser.add_argument("--summary-compatibility-fallback", action="store_true", help="Do not force the summary schema or retry zero-summary rollouts; preserve voluntary summaries and otherwise use the real-tool baseline trajectory.")
    parser.add_argument("--v5-hidden-summary", action="store_true", help="Keep summary out of the public rollout trace; inject it only as privileged next-step context and retain only structured visual evidence for high-KL reasoning.")
    parser.add_argument("--v5-kl-delete-only", action="store_true", help="Use the hidden summary only to compute KL; delete low-KL reasoning, keep high-KL reasoning verbatim, and do not inject the summary into subsequent model context.")
    parser.add_argument("--v5-no-summary-context", action="store_true", help="Use the hidden summary only for KL selection; do not inject it into later model context. High-KL reasoning still becomes structured visual evidence unless --v5-kl-delete-only is also set.")
    all_step_pruning = parser.add_mutually_exclusive_group()
    all_step_pruning.add_argument("--delete-reasoning", action="store_true", help="Remove every assistant reasoning span from history while preserving actions and observations.")
    all_step_pruning.add_argument("--retain-structured-visual-evidence", action="store_true", help="For every assistant step, retain extracted visual evidence, anchors, and actions while removing other reasoning.")
    args = parser.parse_args()
    if args.v5_kl_delete_only and not args.v5_hidden_summary:
        raise ValueError("--v5-kl-delete-only requires --v5-hidden-summary.")
    if args.v5_no_summary_context and not args.v5_hidden_summary:
        raise ValueError("--v5-no-summary-context requires --v5-hidden-summary.")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("Require --num-shards >= 1 and 0 <= --shard-index < --num-shards.")

    task_names = {task.strip() for task in args.tasks.split(",") if task.strip()}
    if args.tool_profile == "auto":
        if task_names == {"vstar"}:
            tool_profile = "vstar"
        elif task_names and task_names <= {"jigsaw_blink", "jigsaw_coco"}:
            tool_profile = "jigsaw"
        else:
            raise ValueError(
                "Tool evaluation must use one benchmark family per run. "
                "Run VStar and Jigsaw separately and set --tool-profile explicitly for other tasks."
            )
    else:
        tool_profile = args.tool_profile
    if not args.allow_cross_benchmark_tools and tool_profile == "vstar" and task_names != {"vstar"}:
        raise ValueError("The VStar tool profile can only be used with --tasks vstar.")
    if not args.allow_cross_benchmark_tools and tool_profile == "jigsaw" and not (task_names and task_names <= {"jigsaw_blink", "jigsaw_coco"}):
        raise ValueError("The Jigsaw tool profile can only be used with Jigsaw tasks.")
    default_tools = VSTAR_TOOLS if tool_profile == "vstar" else JIGSAW_TOOLS
    base_tools = [x.strip() for x in args.tools.split(",") if x.strip()] or list(default_tools)
    foreign_tools = (set(base_tools) & JIGSAW_ONLY_TOOLS) if tool_profile == "vstar" else (set(base_tools) & VSTAR_ONLY_TOOLS)
    if foreign_tools and not args.allow_cross_benchmark_tools:
        raise ValueError(f"{tool_profile} run received tools from the other benchmark family: {sorted(foreign_tools)}")
    runtime = ToolRuntime(
        base_tools,
        controller_addr=args.controller_addr,
        allow_model_tool_fallback=args.allow_model_tool_fallback,
        model_tool_base_url=args.base_url,
        model_tool_model=args.model,
        model_tool_timeout=args.timeout,
    )
    if args.require_all_tools and runtime.missing_tools:
        raise RuntimeError(
            "Complete benchmark tool set is not available. "
            f"Configured={base_tools}; available={runtime.available_tools}; missing={runtime.missing_tools}; "
            "set TOOL_CONTROLLER_ADDR/--controller-addr to a running AdaReasoner tool controller."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "model": args.model,
                "tool_protocol": "nemotron" if "nemotron" in args.model.lower() else "xml",
                "tasks": args.tasks,
                "tools": base_tools,
                "available_tools": runtime.available_tools,
                "missing_tools": runtime.missing_tools,
                "tool_backends": runtime.tool_backends,
                "controller_addr": args.controller_addr,
                "allow_model_tool_fallback": args.allow_model_tool_fallback,
                "require_all_tools": args.require_all_tools,
                "max_rounds": args.max_rounds,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "resume_from": args.resume_from,
                "min_tool_calls": args.min_tool_calls,
                "max_format_retries": args.max_format_retries,
                "summary_retry_attempts": args.summary_retry_attempts,
                "require_visible_reasoning": not args.no_require_visible_reasoning,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "model_image_max_side": MODEL_IMAGE_MAX_SIDE,
                "model_image_jpeg_quality": MODEL_IMAGE_JPEG_QUALITY,
                "max_generated_images": MAX_GENERATED_IMAGES,
                "summary_kl_pruning_enabled": args.enable_summary_kl_pruning,
                "summary_kl_top_logprobs": args.summary_kl_top_logprobs,
                "summary_kl_prune_threshold": args.summary_kl_prune_threshold,
                "summary_kl_prune_min_tokens": args.summary_kl_prune_min_tokens,
                "summary_kl_max_response_chars": args.summary_kl_max_response_chars,
                "summary_mode": args.summary_mode,
                "summary_before_final_only": args.summary_before_final_only,
                "summary_compatibility_fallback": args.summary_compatibility_fallback,
                "v5_hidden_summary": args.v5_hidden_summary,
                "v5_kl_delete_only": args.v5_kl_delete_only,
                "v5_no_summary_context": args.v5_no_summary_context,
                "v5_aggressive_policy": {
                    "visual_backed": args.v5_hidden_summary and not args.v5_kl_delete_only,
                    "kl_delete_only": args.v5_kl_delete_only,
                    "summary_context_injected": args.v5_hidden_summary and not args.v5_kl_delete_only and not args.v5_no_summary_context,
                    "min_tool_calls": int(os.environ.get("BENCHMARK_V5_MIN_TOOL_CALLS", "2")),
                    "gate_on_successful_tools": os.environ.get("BENCHMARK_V5_GATE_ON_SUCCESSFUL_TOOLS") == "1",
                    "min_reasoning_tokens": int(os.environ.get("BENCHMARK_V5_MIN_REASONING_TOKENS", "80")),
                    "require_all_jigsaw_candidates": os.environ.get("BENCHMARK_V5_REQUIRE_ALL_JIGSAW_CANDIDATES", "1") != "0",
                    "min_successful_tools": int(os.environ.get("BENCHMARK_V5_MIN_SUCCESSFUL_TOOLS", "2")),
                    "allow_answer_change": True,
                    "answer_lock_enabled": False,
                    "online_trigger_after_tool": True,
                    "all_reasoning_candidates": os.environ.get("BENCHMARK_V5_ALL_REASONING_CANDIDATES") == "1",
                    "post_final_kl": os.environ.get("BENCHMARK_V5_POST_FINAL_KL") == "1",
                    "require_successful_tool": os.environ.get("BENCHMARK_V5_REQUIRE_SUCCESSFUL_TOOL") == "1",
                    "post_rollout_kl": os.environ.get("BENCHMARK_V5_POST_ROLLOUT_KL") == "1",
                    "include_prepruned_candidates": os.environ.get("BENCHMARK_V5_INCLUDE_PREPRUNED_CANDIDATES") == "1",
                },
                "visual_state_pruning_enabled": args.enable_visual_state_pruning,
                "evidence_only_pruning_enabled": args.enable_evidence_only_pruning,
                "evidence_recheck_enabled": args.enable_evidence_recheck,
                "evidence_recheck_on_issue_only": args.evidence_recheck_on_issue_only,
                "target_hit_gate_enabled": os.environ.get("BENCHMARK_TARGET_HIT_GATE") == "1",
                "kl_prune_v3_structured": os.environ.get("BENCHMARK_KL_PRUNE_V3_STRUCTURED") == "1",
                "selective_pressure_fallback": os.environ.get("BENCHMARK_SELECTIVE_PRESSURE_FALLBACK") == "1",
                "note": "Direct model-driven tool loop; AdaReasoner algorithm is bypassed.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {}
    for task in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        if task not in TASK_SPECS:
            raise ValueError(f"Unknown task: {task}")
        raw_path = out_dir / f"{task}.jsonl"
        done: dict[str, Any] = {}
        resume_paths = (
            [Path(part.strip()) / f"{task}.jsonl" for part in args.resume_from.split(",") if part.strip()]
            if args.resume_from
            else [raw_path]
        )
        if args.resume:
            for resume_path in resume_paths:
                if not resume_path.exists():
                    continue
                with resume_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        done[row["idx"]] = row

        items = load_materialized_items(task, args.items_manifest) if args.items_manifest else load_items(task, args.limit)
        if args.limit is not None:
            items = items[:args.limit]
        if args.num_shards > 1:
            items = [
                item for item_index, item in enumerate(items)
                if item_index % args.num_shards == args.shard_index
            ]
        rows: list[dict[str, Any]] = list(done.values())
        correct_by_cat: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            correct_by_cat[row.get("category", row.get("task_type", "all"))].append(float(row["score"]))

        with raw_path.open("a", encoding="utf-8") as f:
            for item in items:
                if item["idx"] in done:
                    continue
                start = time.time()
                try:
                    result = run_item_with_tools(
                        item=item,
                        runtime=runtime,
                        base_url=args.base_url,
                        model=args.model,
                        max_rounds=args.max_rounds,
                        min_tool_calls=args.min_tool_calls,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        timeout=args.timeout,
                        require_visible_reasoning=not args.no_require_visible_reasoning,
                        max_format_retries=args.max_format_retries,
                        enable_summary_kl_pruning=args.enable_summary_kl_pruning,
                        summary_kl_top_logprobs=args.summary_kl_top_logprobs,
                        summary_kl_prune_threshold=args.summary_kl_prune_threshold,
                        summary_kl_prune_min_tokens=args.summary_kl_prune_min_tokens,
                        summary_kl_max_response_chars=args.summary_kl_max_response_chars,
                        summary_mode=args.summary_mode,
                        enable_visual_state_pruning=args.enable_visual_state_pruning,
                        enable_evidence_only_pruning=args.enable_evidence_only_pruning,
                        enable_evidence_recheck=args.enable_evidence_recheck,
                        evidence_recheck_on_issue_only=args.evidence_recheck_on_issue_only,
                        summary_before_final_only=args.summary_before_final_only,
                        delete_reasoning=args.delete_reasoning,
                        retain_structured_visual_evidence=args.retain_structured_visual_evidence,
                        summary_compatibility_fallback=args.summary_compatibility_fallback,
                        v5_hidden_summary=args.v5_hidden_summary,
                        v5_kl_delete_only=args.v5_kl_delete_only,
                        v5_no_summary_context=args.v5_no_summary_context,
                        task_name=task,
                    )
                    if (
                        SUMMARY_TOOL_NAME in base_tools
                        and not args.summary_compatibility_fallback
                        and not args.v5_hidden_summary
                        and int(result.get("summary_tool_calls", 0)) == 0
                    ):
                        retry_results = [result]
                        for summary_retry in range(1, args.summary_retry_attempts + 1):
                            retry_result = run_item_with_tools(
                                item=item,
                                runtime=runtime,
                                base_url=args.base_url,
                                model=args.model,
                                max_rounds=args.max_rounds,
                                min_tool_calls=args.min_tool_calls,
                                max_tokens=args.max_tokens,
                                temperature=args.temperature,
                                timeout=args.timeout,
                                require_visible_reasoning=not args.no_require_visible_reasoning,
                                max_format_retries=args.max_format_retries,
                                enable_summary_kl_pruning=args.enable_summary_kl_pruning,
                                summary_kl_top_logprobs=args.summary_kl_top_logprobs,
                                summary_kl_prune_threshold=args.summary_kl_prune_threshold,
                                summary_kl_prune_min_tokens=args.summary_kl_prune_min_tokens,
                                summary_kl_max_response_chars=args.summary_kl_max_response_chars,
                                summary_mode=args.summary_mode,
                                enable_visual_state_pruning=args.enable_visual_state_pruning,
                                enable_evidence_only_pruning=args.enable_evidence_only_pruning,
                                enable_evidence_recheck=args.enable_evidence_recheck,
                                evidence_recheck_on_issue_only=args.evidence_recheck_on_issue_only,
                                summary_before_final_only=args.summary_before_final_only,
                                delete_reasoning=args.delete_reasoning,
                                retain_structured_visual_evidence=args.retain_structured_visual_evidence,
                                summary_compatibility_fallback=args.summary_compatibility_fallback,
                                v5_hidden_summary=args.v5_hidden_summary,
                                v5_kl_delete_only=args.v5_kl_delete_only,
                                v5_no_summary_context=args.v5_no_summary_context,
                                task_name=task,
                            )
                            retry_result["summary_retry_attempt"] = summary_retry
                            retry_results.append(retry_result)
                            if int(retry_result.get("summary_tool_calls", 0)) > 0:
                                result = retry_result
                                break
                        else:
                            result = retry_results[-1]
                            result["summary_retry_attempts_exhausted"] = len(retry_results) - 1
                    prediction = result["prediction"]
                    score, message = score_item(task, item, prediction)
                    error = None
                except Exception as exc:
                    result = {"turns": [], "tool_calls": 0, "summary_tool_calls": 0, "image_count": len(item.get("images", [])), "format_retries": 0, "format_failures": 1}
                    prediction = ""
                    score = 0.0
                    message = "exception"
                    error = repr(exc)
                    if isinstance(exc, requests.HTTPError) and exc.response is not None:
                        error += " body=" + exc.response.text[:1000]
                    trace = traceback.format_exc()
                else:
                    trace = None

                row = {
                    "idx": item["idx"],
                    "task": task,
                    "split": item.get("split"),
                    "task_type": item.get("task_type"),
                    "category": item.get("category", item.get("task_type", "all")),
                    "question": item["text"],
                    "gold": item.get("answer"),
                    "prediction": prediction,
                    "raw_prediction": result.get("raw_prediction", prediction),
                    "score": score,
                    "message": message,
                    "tool_calls": result["tool_calls"],
                    "summary_tool_calls": result.get("summary_tool_calls", 0),
                    "image_count": result["image_count"],
                    "format_retries": result.get("format_retries", 0),
                    "format_failures": result.get("format_failures", 0),
                    "tool_protocol": result.get(
                        "tool_protocol",
                        "nemotron" if "nemotron" in args.model.lower() else "xml",
                    ),
                    "summary_kl_pruning_enabled": result.get("summary_kl_pruning_enabled", False),
                    "summary_kl_pruned_steps": result.get("summary_kl_pruned_steps", []),
                    "summary_kl_pruned_reasoning_chars": result.get("summary_kl_pruned_reasoning_chars", 0),
                    "summary_kl_pruned_reasoning_tokens_est": result.get("summary_kl_pruned_reasoning_tokens_est", 0),
                    "reasoning_rewrite_time_sec": result.get("reasoning_rewrite_time_sec", 0.0),
                    "reasoning_rewrite_events": result.get("reasoning_rewrite_events", 0),
                    "summary_kl_time_sec": result.get("summary_kl_time_sec", 0.0),
                    "reasoning_chars_before_prune": result.get("reasoning_chars_before_prune", 0),
                    "reasoning_tokens_before_prune_est": result.get("reasoning_tokens_before_prune_est", 0),
                    "summary_kl_prune_events": result.get("summary_kl_prune_events", []),
                    "summary_mode": result.get("summary_mode", args.summary_mode),
                    "summary_before_final_only": result.get("summary_before_final_only", args.summary_before_final_only),
                    "v5_summary_gate_attempts": result.get("v5_summary_gate_attempts", 0),
                    "v5_summary_gate_accepts": result.get("v5_summary_gate_accepts", 0),
                    "v5_summary_gate_rejections": result.get("v5_summary_gate_rejections", []),
                    "v5_summary_trigger_round": result.get("v5_summary_trigger_round"),
                    "v5_summary_trigger_step_index": result.get("v5_summary_trigger_step_index"),
                    "v5_online_intervention": result.get("v5_online_intervention", False),
                    "v5_kl_delete_only": result.get("v5_kl_delete_only", False),
                    "v5_no_summary_context": result.get("v5_no_summary_context", False),
                    "v5_post_rollout_diagnostic": result.get("v5_post_rollout_diagnostic", False),
                    "v5_aggressive_policy": result.get("v5_aggressive_policy", {}),
                    "successful_benchmark_tool_calls": result.get("successful_benchmark_tool_calls", 0),
                    "tool_parameter_repairs": result.get("tool_parameter_repairs", 0),
                    "successful_jigsaw_insert_candidates": result.get("successful_jigsaw_insert_candidates", []),
                    "visual_state_pruning_enabled": result.get("visual_state_pruning_enabled", args.enable_visual_state_pruning),
                    "evidence_only_pruning_enabled": result.get("evidence_only_pruning_enabled", args.enable_evidence_only_pruning),
                    "evidence_recheck_enabled": result.get("evidence_recheck_enabled", args.enable_evidence_recheck),
                    "evidence_recheck_on_issue_only": result.get("evidence_recheck_on_issue_only", args.evidence_recheck_on_issue_only),
                    "post_summary_benchmark_tool_calls": result.get("post_summary_benchmark_tool_calls", 0),
                    "evidence_recheck_issues": result.get("evidence_recheck_issues", []),
                    "unresolved_evidence_issue": result.get("unresolved_evidence_issue"),
                    "forced_recheck_tool": result.get("forced_recheck_tool"),
                    "blocked_repeated_tool_calls": result.get("blocked_repeated_tool_calls", 0),
                    "target_hit_gate_enabled": result.get("target_hit_gate_enabled", False),
                    "target_terms": result.get("target_terms", []),
                    "kl_prune_v3_structured": os.environ.get("BENCHMARK_KL_PRUNE_V3_STRUCTURED") == "1",
                    "selective_pressure_fallback": os.environ.get("BENCHMARK_SELECTIVE_PRESSURE_FALLBACK") == "1",
                    "visual_evidence": result.get("visual_evidence", {}),
                    "turns": result["turns"],
                    "error": error,
                    "traceback": trace,
                    "latency_sec": round(time.time() - start, 3),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                rows.append(row)
                correct_by_cat[row["category"]].append(float(score))
                print(
                    f"[{task}] {len(rows)}/{len(items)} score={score} tools={row['tool_calls']} summaries={row['summary_tool_calls']} idx={item['idx']}",
                    flush=True,
                )

        scores = [float(x["score"]) for x in rows]
        tool_counts = [int(x.get("tool_calls", 0)) for x in rows]
        summary_tool_counts = [int(x.get("summary_tool_calls", 0)) for x in rows]
        v5_gate_attempt_counts = [int(x.get("v5_summary_gate_attempts", 0)) for x in rows]
        v5_gate_accept_counts = [int(x.get("v5_summary_gate_accepts", 0)) for x in rows]
        v5_gate_rejection_counts = [len(x.get("v5_summary_gate_rejections", [])) for x in rows]
        v5_online_flags = [bool(x.get("v5_online_intervention", False)) for x in rows]
        post_summary_tool_counts = [int(x.get("post_summary_benchmark_tool_calls", 0)) for x in rows]
        pruned_token_counts = [int(x.get("summary_kl_pruned_reasoning_tokens_est", 0)) for x in rows]
        before_prune_token_counts = [int(x.get("reasoning_tokens_before_prune_est", 0)) for x in rows]
        pruning_metrics = pruning_fraction_metrics(
            pruned_token_counts,
            before_prune_token_counts,
        )
        summary[task] = {
            "total": len(rows),
            "score": sum(scores) / len(scores) if scores else 0.0,
            "avg_tool_calls": sum(tool_counts) / len(tool_counts) if tool_counts else 0.0,
            "avg_summary_tool_calls": sum(summary_tool_counts) / len(summary_tool_counts) if summary_tool_counts else 0.0,
            "v5_summary_gate_attempt_rate": (
                sum(1 for value in v5_gate_attempt_counts if value > 0) / len(v5_gate_attempt_counts)
                if v5_gate_attempt_counts
                else 0.0
            ),
            "v5_summary_gate_accept_rate": (
                sum(v5_gate_accept_counts) / max(1, sum(v5_gate_attempt_counts))
            ),
            "v5_summary_gate_rejections": sum(v5_gate_rejection_counts),
            "v5_online_intervention_rate": (
                sum(v5_online_flags) / len(v5_online_flags) if v5_online_flags else 0.0
            ),
            "v5_post_summary_tool_call_rate": (
                sum(1 for value in post_summary_tool_counts if value > 0) / len(post_summary_tool_counts)
                if post_summary_tool_counts
                else 0.0
            ),
            "avg_post_summary_benchmark_tool_calls": (
                sum(post_summary_tool_counts) / len(post_summary_tool_counts)
                if post_summary_tool_counts
                else 0.0
            ),
            "summary_kl_pruning_enabled": args.enable_summary_kl_pruning,
            "summary_mode": args.summary_mode,
            "visual_state_pruning_enabled": args.enable_visual_state_pruning,
            "evidence_only_pruning_enabled": args.enable_evidence_only_pruning,
            "evidence_recheck_enabled": args.enable_evidence_recheck,
            "avg_pruned_reasoning_tokens_est": sum(pruned_token_counts) / len(pruned_token_counts) if pruned_token_counts else 0.0,
            "avg_reasoning_tokens_before_prune_est": sum(before_prune_token_counts) / len(before_prune_token_counts) if before_prune_token_counts else 0.0,
            "avg_pruned_reasoning_fraction_est": (
                pruning_metrics["micro"]
            ),
            "pruned_reasoning_fraction_micro_est": pruning_metrics["micro"],
            "pruned_reasoning_fraction_macro_all_est": pruning_metrics["macro_all"],
            "pruned_reasoning_fraction_macro_nonempty_est": pruning_metrics["macro_nonempty"],
            "reasoning_nonempty_rows": pruning_metrics["nonempty_rows"],
            "by_category": {k: sum(v) / len(v) for k, v in sorted(correct_by_cat.items()) if v},
            "raw_path": str(raw_path),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({task: summary[task]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
