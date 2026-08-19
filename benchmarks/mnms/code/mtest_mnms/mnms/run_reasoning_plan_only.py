import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mnms.constants import TOOL_METADATA
from mnms.verifier import Verifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model", default="qwen3-vl-8b-tool")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8007/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_done_ids(path: Path) -> set[Any]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def tool_metadata_text() -> str:
    lines = ["# TOOL LIST #"]
    for tool, meta in TOOL_METADATA.items():
        inputs = ", ".join(meta["input"]["arg_name"])
        outputs = ", ".join(meta["output"]["arg_name"])
        lines.append(f"- {tool}: {meta['description']} Input args: {inputs}. Output fields: {outputs}.")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an MNMS plan-only tool-use agent.

You must solve the user request by PLANNING tool calls only. Do not execute tools and do not depend on real observations.
Output exactly one JSON object and no markdown.

Your priority is PLAN QUALITY. Reasoning is a compact decision checklist, not a long explanation. First choose the minimal tool sequence that best matches MNMS benchmark conventions, then write a short rationale for each chosen tool.

Required output schema:
{
  "steps": [
    {
      "reasoning": "route=<task type>; use=<selected tool and why>; avoid=<nearest wrong tool or extra step>; args=<exact argument names and sources>; output=<expected field>",
      "content": "<think>same checklist reasoning</think>",
      "action": {
        "id": 0,
        "name": "tool name from the tool list",
        "args": {"arg_name": "arg_value"}
      }
    }
  ],
  "final_answer": "optional short natural-language answer if the plan itself directly answers the request"
}

Rules:
1. This is plan-only mode. Never say you observed a tool result and never invent tool outputs.
2. Produce the complete plan in one response, up to the max step count requested by the user.
3. Use valid MNMS references like <node-0>.text, <node-1>.image, or <node-2>.objects when later steps consume earlier planned outputs.
4. Every action must be one valid MNMS tool node. IDs must start at 0 and increase by 1.
5. Every reasoning must be short and decision-focused, using this format:
   "route=...; use=...; avoid=...; args=...; output=..."
   It must mention the selected tool name, exact input argument names, input sources/values, expected output fields, and the nearest wrong alternative to avoid.
6. Reasoning should prevent common MNMS mistakes, especially extra tools. Do not repeat the same generic template across steps.
7. Use as few steps as possible while covering the request.
8. There is no image loading tool. For image tasks, pass the image filename directly to tools such as image captioning, object detection, image segmentation, image crop, background blur, or color pop.
9. Do not add tools just because they sound helpful. Extra object detection/select object/text generation steps often hurt plan accuracy.

Tool-routing rules that should override vague intuition:
- If the user asks a question whose answer is contained in a provided text passage, use question answering with `question` and `text`. Do not use wikipedia simple search, get date fact, or text classification. Clues: "based on this text", "based on these details", "according to the passage", quoted passage followed by a question.
- If the user asks whether text/review/sentiment/tone is positive or negative, use text classification.
- If the user asks for text continuation, rewriting, explanation, story, caption-to-text, or free-form writing, use text generation.
- If the user asks to summarize text, use text summarization.
- If the user asks for historical events/facts on a calendar date, use get date fact. For a year, use get year fact. For an educational/math/classroom/kids/student fact about a number, use get math fact. For an explicitly "trivia" or "fun random" number fact, use get trivia fact.
- If the user asks movie details, plot, director, or genre, first use search movie. If they ask a specific question about returned movie details, follow with question answering over <node-0>.text. If they ask for a discussion or influence essay, follow with text generation.
- If the user asks a direct visual question about an image ("who", "what", "where", "is there", "which part"), use visual question answering with `image` and `question`. Do not decompose into object detection unless the request asks for boxes, objects, count, crop, segmentation, blur, or color pop.
- If the user asks to read, identify, extract, or transcribe visible text in an image, use optical character recognition. Do not add text generation unless the user asks to translate into a specific target language or rewrite the extracted text.
- If the user asks to describe an image, use image captioning. If they ask to classify an image category, use image classification.
- If the user asks to identify/list/categorize distinct objects, use object detection only. Do not add count or text generation unless the user explicitly asks "how many" or asks for prose after detection.
- If the user asks "how many objects/items are visible", use object detection then count. If they ask "how many distinct parts/regions/components/pieces/sections", use image segmentation then count.
- If the user asks to separate/segment components, regions, parts, masks, shapes, or isolate an object from the rest, use image segmentation. If a specific object must be selected after segmentation, then use select object on <node-0>.objects. Avoid object detection for mask/region wording.
- If the user asks to crop a known region such as top/bottom/left/right, use the matching image crop tool directly. For crop around a detected object, use object detection then image crop only when the request literally asks to crop/focus more closely around that object.
- If the user asks to edit/remove/add/replace/transform something in an image, use image editing directly with `image` set to the filename and `prompt` describing the edit. Do not add object detection/select object before image editing unless the requested final tool requires an `object`.
- If the user asks to create a creative representation based on an existing image/concept, use image editing on the given image, not image generation, when an image filename is provided.
- If the user asks to blur the background around a subject, use object detection, select object, then background blur. If they ask for one object in color and the rest black-and-white, use object detection, select object, then color pop.
- If the user provides longitude and latitude and asks weather, use get weather directly with `lon` and `lat`. Do not call get location.
- If audio should be visualized as an image, use automatic speech recognition then image generation directly. Do not insert text generation unless the user asks to rewrite or elaborate the transcript.
- If the user asks to create an image from a textual description, use image generation.

Good examples:
- "route=sentiment; use=text classification for polarity; avoid=question answering because no question over context is needed; args=`text` is the user's review; output=text sentiment label."
- "route=image edit; use=image editing directly for replacing an object; avoid=object detection/select object because image editing accepts an image filename and prompt; args=`image` is the filename, `prompt` describes the edit; output=image."
- "route=visual QA; use=visual question answering for a direct question about an image; avoid=object detection/crop/caption because no boxes or masks are requested; args=`image` is the filename, `question` is the user question; output=text answer."
- "route=number educational fact; use=get math fact for a classroom/student number fact; avoid=get trivia fact because MNMS uses math fact for educational number facts; args=`number` is the requested number; output=text fact."
"""


def initial_messages(user_request: str, max_steps: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + tool_metadata_text()},
        {
            "role": "user",
            "content": (
                f"# USER REQUEST #\n{user_request}\n\n"
                f"Generate a complete plan with at most {max_steps} tool steps. "
                "Before choosing steps, silently classify the request type using the routing rules. "
                "Return only the JSON object. Remember: plan only, do not execute tools and do not claim observations."
            ),
        },
    ]


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code} from {url}: {body[:1000]}") from err
    obj = json.loads(body)
    return obj["choices"][0]["message"].get("content") or ""


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:500]}")
    return json.loads(match.group(0))


def normalize_steps(parsed: dict[str, Any], max_steps: int) -> list[dict[str, Any]]:
    raw_steps = parsed.get("steps") or parsed.get("plan") or parsed.get("nodes") or []
    if isinstance(raw_steps, dict):
        raw_steps = raw_steps.get("steps") or raw_steps.get("nodes") or []
    if not isinstance(raw_steps, list):
        raise ValueError(f"steps must be a list: {type(raw_steps).__name__}")

    steps = []
    for idx, item in enumerate(raw_steps[:max_steps]):
        if not isinstance(item, dict):
            raise ValueError(f"step {idx} is not an object: {item!r}")
        action = item.get("action") or item.get("tool_call") or item
        if not isinstance(action, dict):
            raise ValueError(f"step {idx} action is not an object: {action!r}")
        action = dict(action)
        action["id"] = idx
        if "name" not in action or "args" not in action:
            raise ValueError(f"step {idx} action must contain name and args: {action!r}")
        if not isinstance(action["args"], dict):
            raise ValueError(f"step {idx} args must be an object: {action!r}")
        reasoning = str(item.get("reasoning") or item.get("rationale") or item.get("thought") or "").strip()
        content = str(item.get("content") or "").strip()
        match = re.search(r"<think>(.*?)</think>", content, re.S)
        if match and not reasoning:
            reasoning = " ".join(match.group(1).split())
        if not reasoning:
            reasoning = (
                f"I will call {action['name']} with the provided arguments because this planned step is needed "
                "to solve the user's request. The expected output will be consumed by the remaining plan or final answer."
            )
        content = f"<think>{' '.join(reasoning.split())}</think>"
        steps.append({"step": idx, "reasoning": " ".join(reasoning.split()), "assistant_content": content, "action": action})
    steps = remove_unsupported_image_loading_steps(steps)
    if not steps:
        raise ValueError("model returned no planned tool steps")
    return steps


def replace_refs(value: Any, ref_map: dict[int, int], image_replacements: dict[int, str]) -> Any:
    if isinstance(value, str):
        for old_id, image_value in image_replacements.items():
            value = value.replace(f"<node-{old_id}>.image", image_value)
        for old_id, new_id in sorted(ref_map.items(), reverse=True):
            value = value.replace(f"<node-{old_id}>.", f"<node-{new_id}>.")
        return value
    if isinstance(value, list):
        return [replace_refs(v, ref_map, image_replacements) for v in value]
    if isinstance(value, dict):
        return {k: replace_refs(v, ref_map, image_replacements) for k, v in value.items()}
    return value


def remove_unsupported_image_loading_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_replacements = {}
    kept = []
    ref_map = {}
    for step in steps:
        old_id = int(step["action"]["id"])
        action = step["action"]
        name = str(action.get("name", "")).lower()
        if name in {"image loading", "load image", "image loader"}:
            args = action.get("args", {})
            image_value = args.get("image") or args.get("image_path") or args.get("path") or args.get("file")
            if image_value:
                image_replacements[old_id] = str(image_value)
            continue
        ref_map[old_id] = len(kept)
        kept.append(step)

    fixed = []
    for new_id, step in enumerate(kept):
        step = dict(step)
        action = dict(step["action"])
        action["args"] = replace_refs(action.get("args", {}), ref_map, image_replacements)
        action["id"] = new_id
        step["action"] = action
        step["step"] = new_id
        step["reasoning"] = replace_refs(step.get("reasoning", ""), ref_map, image_replacements)
        step["assistant_content"] = f"<think>{step['reasoning']}</think>"
        fixed.append(step)
    return fixed


def validate_reasoning(step: dict[str, Any]) -> list[str]:
    reasoning = step.get("reasoning", "")
    action = step.get("action") or {}
    normalized = reasoning.lower()
    issues = []
    if len(reasoning.split()) < 12:
        issues.append("reasoning is too short")
    tool_name = str(action.get("name", ""))
    if tool_name.lower() not in normalized:
        issues.append(f"reasoning does not mention selected tool name '{tool_name}'")
    if not any(str(k).lower() in normalized for k in action.get("args", {}).keys()):
        issues.append("reasoning does not mention an action input argument")
    if not any(word in normalized for word in ["output", "return", "produce", "extract", "generate", "classify", "answer", "summarize", "image", "text"]):
        issues.append("reasoning does not mention expected output or purpose")
    return issues


class PlanParseError(Exception):
    def __init__(self, message: str, raw_response: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.raw_response = raw_response
        self.attempts = attempts


def rollout_one(client: dict[str, str], args: argparse.Namespace, verifier: Verifier, sample: dict[str, Any]) -> dict[str, Any]:
    task_id = sample["id"]
    user_request = sample.get("user_request") or sample.get("human_rewrite_request") or ""
    messages = initial_messages(user_request, args.max_steps)
    attempts = []
    assistant_text = ""
    latency = 0.0
    parsed = None
    steps = None
    for attempt_idx in range(args.parse_retries + 1):
        started = time.time()
        assistant_text = chat_completion(
            base_url=client["base_url"],
            api_key=client["api_key"],
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        latency += time.time() - started
        try:
            parsed = extract_json(assistant_text)
            steps = normalize_steps(parsed, args.max_steps)
            break
        except Exception as exc:
            attempts.append({"attempt": attempt_idx, "error": repr(exc), "raw_response": assistant_text})
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid parseable JSON for the required schema. "
                        f"Parser error: {repr(exc)}. "
                        "Return exactly one corrected JSON object and no markdown. "
                        "Use double quotes for all strings, escape all quotes inside string values, and do not include trailing commas. "
                        "The JSON object must contain a `steps` list; each step must contain `reasoning`, `content`, and `action`."
                    ),
                }
            )
    if parsed is None or steps is None:
        raise PlanParseError("model did not return parseable plan JSON after retries", assistant_text, attempts)

    verify_results = []
    reasoning_issues = []
    for step in steps:
        verify_results.append(verifier.verify_single_node(step["action"]))
        issues = validate_reasoning(step)
        if issues:
            reasoning_issues.append({"step": step["step"], "issues": issues})

    verify_status = all(v["status"] for v in verify_results)
    status = "planned" if verify_status and not reasoning_issues else "planned_with_warnings"
    if not verify_status:
        status = "verification_failed"

    return {
        "id": task_id,
        "user_request": user_request,
        "status": status,
        "prediction": [s["action"] for s in steps],
        "steps": steps,
        "reasoning": [s["reasoning"] for s in steps],
        "final_answer": parsed.get("final_answer"),
        "parse_status": True,
        "verify_status": verify_status,
        "verify_results": verify_results,
        "reasoning_issues": reasoning_issues,
        "raw_response": assistant_text,
        "latency_sec": latency,
        "plan_only": True,
        "executed_tools": False,
    }


def main():
    args = get_args()
    args.input_file = str(Path(args.input_file).resolve())
    args.output_file = str(Path(args.output_file).resolve())
    os.chdir(Path(__file__).resolve().parents[1])
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.input_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    done = load_done_ids(out_path) if args.resume else set()

    client = {"base_url": args.base_url, "api_key": args.api_key}
    verifier = Verifier(tool_metadata=TOOL_METADATA)

    with out_path.open("a" if args.resume else "w", encoding="utf-8") as wf:
        for idx, sample in enumerate(rows):
            if sample["id"] in done:
                continue
            print(f"Planning task {sample['id']} ({idx + 1}/{len(rows)})")
            try:
                result = rollout_one(client, args, verifier, sample)
            except PlanParseError as exc:
                result = {
                    "id": sample.get("id"),
                    "user_request": sample.get("user_request"),
                    "status": "plan_error",
                    "error": repr(exc),
                    "parse_status": False,
                    "prediction": [],
                    "steps": [],
                    "raw_response": exc.raw_response,
                    "parse_attempts": exc.attempts,
                    "plan_only": True,
                    "executed_tools": False,
                }
            except Exception as exc:
                result = {
                    "id": sample.get("id"),
                    "user_request": sample.get("user_request"),
                    "status": "plan_error",
                    "error": repr(exc),
                    "parse_status": False,
                    "prediction": [],
                    "steps": [],
                    "plan_only": True,
                    "executed_tools": False,
                }
            wf.write(json.dumps(result, ensure_ascii=False) + "\n")
            wf.flush()
            print(
                f"  status={result.get('status')} steps={len(result.get('steps', []))} "
                f"verify={result.get('verify_status')}"
            )


if __name__ == "__main__":
    main()
