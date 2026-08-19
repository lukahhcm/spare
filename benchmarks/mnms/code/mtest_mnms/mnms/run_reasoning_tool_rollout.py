import argparse
import ast
import json
import os
import pickle
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mnms.constants import TOOL_METADATA
from mnms.execution.executor import Executor
from mnms.execution.tool_api import _as_text
from mnms.summary_kl_core import (
    SUMMARY_TOOL_NAME,
    enabled as summary_kl_enabled,
    log_summarization_kl,
    summary_tool_metadata,
)
from mnms.verifier import Verifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--result-folder", required=True)
    parser.add_argument("--model", default="qwen3-vl-8b-tool")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8007/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-retries", type=int, default=2)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Run the official sequential planning protocol without executing tools. "
            "Each proposed MNMS action is still validated by Verifier and recorded; "
            "the next turn receives an explicit plan-verification acknowledgement, not a fabricated tool observation."
        ),
    )
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
    if os.environ.get("MNMS_NO_TOOL_BASELINE", "0").lower() in {"1", "true", "yes"}:
        return "# TOOL LIST #\nNo tools are available in this run."
    lines = ["# TOOL LIST #"]
    visible_tools = dict(TOOL_METADATA)
    summary_enabled = os.environ.get("MNMS_ENABLE_SUMMARY_TOOL", "1") != "0"
    summary_model_visible = os.environ.get("MNMS_SUMMARY_MODEL_VISIBLE", "1") != "0"
    if summary_enabled and summary_model_visible:
        visible_tools[SUMMARY_TOOL_NAME] = summary_tool_metadata()
    for tool, meta in visible_tools.items():
        inputs = ", ".join(meta["input"]["arg_name"])
        outputs = ", ".join(meta["output"]["arg_name"])
        lines.append(
            f"- {tool}: {meta['description']} Input args: {inputs}. Output fields: {outputs}."
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an MNMS tool-use agent.

You must solve the user request by interacting with tools step by step.
Before every tool call, you must write a concrete action rationale in the exact format:
<think>specific reason for this next tool call</think>

For each assistant turn, output exactly one JSON object and no markdown.

If another tool is needed, output:
{
  "content": "<think>specific reason why this tool should be called now, which input it should use, and what output is needed</think>",
  "reasoning": "same specific reason, without XML tags",
  "action": {
    "id": integer starting from 0 and increasing by 1,
    "name": "tool name from the tool list",
    "args": {"arg_name": "arg_value"}
  }
}

If the request is already fulfilled after observing tool outputs, output:
{
  "content": "<think>reason why the request is fulfilled</think>",
  "reasoning": "same reason, without XML tags",
  "final_answer": "short final answer for the user"
}

Rules:
1. Every action must be a single valid MNMS tool node.
2. Use references like <node-0>.text, <node-1>.image, or <node-2>.objects when using prior tool outputs.
3. Do not output a multi-step plan. Call only the next tool.
4. Do not call tools not listed in the tool list.
5. Use as few tools as possible, but execute the tools needed to answer the user request.
6. The reasoning must be specific, not generic. It must mention the selected tool name, the input source or prior node used, and the expected output field or purpose.
7. Do not fabricate observations. If a prior tool output is needed, reference it with <node-i>.field and wait for the real observation.
8. After an observation, your next reasoning must explicitly react to that observation. If a search/API tool returns empty text, no result, or tool_status unavailable, do not call the same tool again with only a rephrased query. Either use a different available evidence source or provide a final_answer that states the limitation.
9. Do not repeat the same rationale pattern across steps. Each reasoning block must explain what changed because of the latest observation.
10. You should decide when to summarize the current state and information of the task with the `summarize_the_task` tool and call `summarize_the_task` at least once in the conversation to help you make decisions. Its `content` argument must be a comprehensive, faithful recap that a fresh agent could continue from: include the original request, every completed tool call with arguments, key observations, known facts, unresolved uncertainty, and the single next intended action. This internal memory tool does not count as solving the user request, and after it returns you must continue with the next needed real tool or final_answer.

Reasoning quality examples:
- Bad: "I should use a tool next."
- Bad: "I will generate text."
- Good: "I will call text generation with the `text` argument containing the user's seed sentence, because the user asked for a continuation. The expected output is generated text that can be summarized or classified in the next step."
- Good: "I will call text summarization with `text` set to <node-0>.text, because the previous tool produced a long text and the user asked for a short digestible summary. The expected output is a concise text summary."
"""


def initial_messages(user_request: str, *, plan_only: bool = False, model_name: str = "") -> list[dict[str, str]]:
    if os.environ.get("MNMS_NO_TOOL_BASELINE", "0").lower() in {"1", "true", "yes"}:
        system_prompt = """You are an MNMS direct-answer baseline agent.

No tools are available in this run. Do not propose, simulate, or call any tool.
For each assistant turn, output exactly one JSON object and no markdown:
{
  "content": "<think>specific visible reasoning for answering without tools</think>",
  "reasoning": "same specific reason, without XML tags",
  "final_answer": "short final answer for the user"
}
Do not output an action field."""
        return [
            {"role": "system", "content": system_prompt + "\n\n" + tool_metadata_text()},
            {"role": "user", "content": f"# USER REQUEST #\n{user_request}\n\nAnswer now without using tools."},
        ]
    system_prompt = SYSTEM_PROMPT
    summary_enabled = os.environ.get("MNMS_ENABLE_SUMMARY_TOOL", "1") != "0"
    summary_model_visible = os.environ.get("MNMS_SUMMARY_MODEL_VISIBLE", "1") != "0"
    summary_prompt_only_short = os.environ.get("MNMS_SUMMARY_PROMPT_ONLY_SHORT", "0").lower() in {"1", "true", "yes"}
    long_summary_rule = (
        "10. You should decide when to summarize the current state and information of the task with the "
        "`summarize_the_task` tool and call `summarize_the_task` at least once in the conversation to help "
        "you make decisions. Its `content` argument must be a comprehensive, faithful recap that a fresh "
        "agent could continue from: include the original request, every completed tool call with arguments, "
        "key observations, known facts, unresolved uncertainty, and the single next intended action. This "
        "internal memory tool does not count as solving the user request, and after it returns you must "
        "continue with the next needed real tool or final_answer."
    )
    if not summary_enabled or not summary_model_visible:
        system_prompt = system_prompt.replace(
            long_summary_rule,
            "10. Do not call tools that are not listed in the tool list.",
        )
    elif summary_prompt_only_short:
        system_prompt = system_prompt.replace(
            long_summary_rule,
            "10. You should decide when to summarize the current state and information of the task with the `summarize_the_task` tool and call `summarize_the_task` at least once in the conversation to help you make decisions.",
        )
    system_prompt = system_prompt + "\n\n" + tool_metadata_text()
    model_lower = model_name.lower()
    if "kimi" in model_lower or "a3b" in model_lower:
        system_prompt += (
            "\n\nMODEL OUTPUT COMPATIBILITY CONTRACT (MANDATORY): "
            "The controller consumes your trajectory, not prose. Return exactly one valid JSON object "
            "and nothing else on every turn. Never use <action>, <tool>, <args>, XML, YAML, markdown, "
            "Python repr, or a provider-native tool-call format. For a tool step use exactly: "
            '{"content":"<think>specific reason</think>","reasoning":"specific reason",'
            '"action":{"id":0,"name":"EXACT TOOL NAME FROM TOOL LIST","args":{}}}. '
            "Use the exact tool name and argument names from # TOOL LIST #; increment action.id by one "
            "for each real tool step. For completion use exactly: "
            '{"content":"<think>reason</think>","reasoning":"reason","final_answer":"answer"}. '
            "Do not wrap JSON in a code fence and do not emit a multi-step trajectory."
        )
    token_threshold = int(os.environ.get("MNMS_SUMMARY_FORCE_TOKEN_THRESHOLD", "0"))
    if os.environ.get("MNMS_SUMMARY_RUNTIME_ENFORCE", "0") == "1" and token_threshold > 0:
        system_prompt += (
            "\n\nToken-threshold summary experiment override: do not call "
            "`summarize_the_task` immediately after the first observation just because Rule 10 says so. "
            "In this run, call `summarize_the_task` only when the controller/protocol message explicitly "
            "says a summary is due, or when the conversation has clearly accumulated enough unsummarized "
            f"assistant reasoning to exceed about {token_threshold} tokens. If no such protocol message is present, "
            "continue with the next needed real MNMS tool or final_answer."
        )
    if plan_only:
        # Preserve the semantics of the official MNMS step-by-step planner:
        # actions are plan nodes checked by Verifier, not tool executions.
        # The JSON envelope is only a model-compatibility adaptation.
        system_prompt = system_prompt.replace(
            "You must solve the user request by interacting with tools step by step.",
            "You must solve the user request by planning tool actions step by step.",
        )
        system_prompt = system_prompt.replace(
            "If the request is already fulfilled after observing tool outputs, output:",
            "If the planned action graph is complete, output:",
        )
        system_prompt = system_prompt.replace(
            "Use as few tools as possible, but execute the tools needed to answer the user request.",
            "Use as few planned tool actions as possible.",
        )
        system_prompt = system_prompt.replace(
            "Do not fabricate observations. If a prior tool output is needed, reference it with <node-i>.field and wait for the real observation.",
            "Do not fabricate tool results. Express dependencies on earlier planned nodes with <node-i>.field references.",
        )
        system_prompt = system_prompt.replace(
            "After an observation, your next reasoning must explicitly react to that observation. If a search/API tool returns empty text, no result, or tool_status unavailable, do not call the same tool again with only a rephrased query. Either use a different available evidence source or provide a final_answer that states the limitation.",
            "After verifier feedback, either add the next necessary plan node or finish the plan. Never repeat an accepted node merely because it was not executed.",
        )
        system_prompt += (
            "\n\nPlan-only protocol: MNMS actions are being constructed and verified one at a time, "
            "and tools are not executed in this benchmark. After every valid action, the controller returns "
            "`OBSERVATION: Verification succeeded.` If the plan is then complete, finish immediately; "
            "otherwise generate the next distinct THOUGHT and ACTION. Verification success means the action "
            "is already part of the plan: never call it again to try to execute it. Do not invent tool outputs "
            "or facts. Use <node-i>.field references for dependencies that would be resolved at execution time. "
            "Because this model uses a JSON compatibility envelope, finishing the plan is represented by a "
            "`final_answer` object and is semantically equivalent to the official `TERMINATE` response."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"# USER REQUEST #\n{user_request}\n\nGenerate step 0 now."},
    ]


def compact_observation(output: Any, max_chars: int = 1600) -> str:
    if isinstance(output, dict):
        compact = {}
        for key, value in output.items():
            if key == "image":
                compact[key] = str(value)
            elif key == "objects":
                compact[key] = value[:5] if isinstance(value, list) else value
            else:
                compact[key] = value
        text = json.dumps(compact, ensure_ascii=False)
    else:
        text = _as_text(output)
    return text[:max_chars]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + int(non_ascii_chars / 1.5))


def _infer_args(tool_name: str, text: str, user_request: str) -> dict[str, Any]:
    meta = TOOL_METADATA.get(tool_name, {})
    arg_names = list((meta.get("input") or {}).get("arg_name") or [])
    combined = f"{text}\n{user_request}"
    image_match = re.search(r"[\w./-]+\.(?:png|jpe?g|webp|gif)", combined, re.I)
    quoted = re.findall(r"['\"]([^'\"]{3,})['\"]", combined)
    args: dict[str, Any] = {}
    for arg in arg_names:
        if arg == "image":
            args[arg] = image_match.group(0) if image_match else ""
        elif arg in {"text", "prompt", "question"}:
            args[arg] = quoted[-1] if quoted else user_request
        elif arg == "object":
            args[arg] = "object"
        elif arg == "object_name":
            args[arg] = "object"
        elif arg == "objects":
            args[arg] = []
        elif arg in {"number", "year", "lon", "lat"}:
            number_match = re.search(r"\b-?\d+(?:\.\d+)?\b", combined)
            args[arg] = number_match.group(0) if number_match else "0"
        elif arg == "date":
            args[arg] = ""
        else:
            args[arg] = ""
    return args


def _parse_legacy_tool_markup(text: str, user_request: str = "") -> dict[str, Any]:
    """Convert common provider-native/XML tool traces into one MNMS object."""
    tool_match = re.search(r"<(?:tool|name|function)\s*>\s*([^<\n]+?)\s*</(?:tool|name|function)\s*>", text, re.I)
    if not tool_match:
        tool_match = re.search(r"(?:^|\n)\s*(?:tool|function)\s*[:=]\s*([^\n]+)", text, re.I)
    if not tool_match:
        # Some models emit a JSON-like content/reasoning object but omit the
        # action field. Recover only when the response explicitly names one
        # of the registered MNMS tools; never guess an unseen tool name.
        lowered = text.lower()
        mentioned = sorted(
            (name for name in TOOL_METADATA if name.lower() in lowered),
            key=len,
            reverse=True,
        )
        if not mentioned:
            raise ValueError(f"No MNMS action or legacy tool name found in response: {text[:500]}")
        tool_name = mentioned[0]
        reason = " ".join(text.split())[:1200]
        return {
            "content": f"<think>{reason}</think>",
            "reasoning": reason,
            "action": {"id": 0, "name": tool_name, "args": _infer_args(tool_name, text, user_request)},
            "_external_format": "tool_mentioned_without_action",
        }
    tool_name = tool_match.group(1).strip().strip('`"')

    args_text = ""
    args_match = re.search(r"<args\s*>(.*?)(?:</args\s*>|$)", text, re.I | re.S)
    if args_match:
        args_text = args_match.group(1).strip()
    else:
        args_match = re.search(r"(?:args|arguments|parameters)\s*[:=]\s*(\{.*?\})", text, re.I | re.S)
        if args_match:
            args_text = args_match.group(1).strip()
    args_text = re.sub(r"^```(?:json|python|text)?\s*|\s*```$", "", args_text.strip(), flags=re.I | re.S).strip()
    if not args_text:
        args: dict[str, Any] = {}
    else:
        try:
            args = json.loads(args_text)
        except json.JSONDecodeError:
            try:
                args = ast.literal_eval(args_text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Legacy tool args are not a JSON/Python dict: {args_text[:500]}") from exc
        if not isinstance(args, dict):
            raise ValueError(f"Legacy tool args must be a dict: {args!r}")

    reason_match = re.search(r"<(?:think|reason|reasoning)\s*>(.*?)</(?:think|reason|reasoning)\s*>", text, re.I | re.S)
    reason = " ".join(reason_match.group(1).split()) if reason_match else f"Call {tool_name} with the requested arguments."
    return {
        "content": f"<think>{reason}</think>",
        "reasoning": reason,
        "action": {"id": 0, "name": tool_name, "args": args},
        "_external_format": "legacy_tool_markup",
    }


def extract_json(text: str, user_request: str = "") -> dict[str, Any]:
    text = text.strip()
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict) and (
            any(key in candidate for key in ("action", "tool_call", "tool", "final_answer"))
            or not any(name.lower() in text.lower() for name in TOOL_METADATA)
        ):
            return candidate
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            candidate = json.loads(match.group(0))
            if isinstance(candidate, dict):
                if any(key in candidate for key in ("action", "tool_call", "tool", "final_answer")):
                    return candidate
                if not any(name.lower() in text.lower() for name in TOOL_METADATA):
                    return candidate
        except json.JSONDecodeError:
            pass
    return _parse_legacy_tool_markup(text, user_request=user_request)


def normalize_action(obj: dict[str, Any], expected_id: int) -> dict[str, Any] | None:
    action = obj.get("action") or obj.get("tool_call") or obj.get("tool")
    if action is None:
        return None
    if not isinstance(action, dict):
        raise ValueError(f"action is not a dict: {action!r}")
    action = dict(action)
    action["id"] = expected_id
    if "name" not in action or "args" not in action:
        raise ValueError(f"action must contain name and args: {action!r}")
    if not isinstance(action["args"], dict):
        raise ValueError(f"action args must be a dict: {action!r}")
    return action


def extract_reasoning(parsed: dict[str, Any]) -> tuple[str, str]:
    content = str(parsed.get("content", "") or "")
    reasoning = str(parsed.get("reasoning", "") or "").strip()
    match = re.search(r"<think>(.*?)</think>", content, re.S)
    if match:
        think_text = " ".join(match.group(1).split())
        if not reasoning:
            reasoning = think_text
        content = f"<think>{think_text}</think>"
    elif reasoning:
        reasoning = " ".join(reasoning.split())
        content = f"<think>{reasoning}</think>"
    elif content.strip():
        reasoning = " ".join(content.split())
        content = f"<think>{reasoning}</think>"
    else:
        reasoning = ""
        content = ""
    return reasoning, content


def validate_reasoning(reasoning: str, action: dict[str, Any] | None) -> list[str]:
    issues = []
    normalized = reasoning.lower().strip()
    if action is not None and str(action.get("name", "")) == SUMMARY_TOOL_NAME:
        if "content" not in action.get("args", {}):
            issues.append("summarize_the_task requires a content argument")
        if len(str(action.get("args", {}).get("content", "")).split()) < 20:
            issues.append("summarize_the_task content is too short")
        return issues
    generic_phrases = {
        "i need to decide the next tool call",
        "i need to use a tool",
        "i will use the tool",
        "use the tool to solve the problem",
    }
    if len(reasoning.split()) < 14:
        issues.append("reasoning is too short")
    if normalized in generic_phrases:
        issues.append("reasoning is generic")
    if action is not None:
        tool_name = str(action.get("name", ""))
        if tool_name.lower() not in normalized:
            issues.append(f"reasoning does not mention selected tool name '{tool_name}'")
        if not any(str(k).lower() in normalized for k in action.get("args", {}).keys()):
            issues.append("reasoning does not mention an action input argument")
        if not any(word in normalized for word in ["output", "return", "produce", "extract", "generate", "classify", "answer", "summarize", "image", "text"]):
            issues.append("reasoning does not mention expected output or purpose")
    return issues


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
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
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


def call_model_for_step(
    client: dict[str, str],
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    step_idx: int,
) -> tuple[str, dict[str, Any], str, str, float]:
    last_issues = []
    for attempt in range(args.reasoning_retries + 1):
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
        latency = time.time() - started
        try:
            parsed = extract_json(assistant_text, user_request=str(messages[1].get("content", "") if len(messages) > 1 else ""))
            action = normalize_action(parsed, expected_id=step_idx)
        except Exception as exc:
            last_issues = [f"invalid JSON/action format: {exc}"]
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response could not be parsed as one complete JSON object. "
                        "Rewrite the same step now as exactly one valid JSON object and nothing else. "
                        "Do not use markdown, XML-only tool_call blocks, comments, trailing text, or unescaped quotes inside JSON strings. "
                        "Use this shape for a tool call: "
                        '{"content":"<think>specific reason</think>","reasoning":"specific reason","action":{"id":'
                        f'{step_idx},"name":"tool name","args":{{"arg_name":"arg_value"}}}}. '
                        f"Parser problem: {exc}. "
                        f"Previous response prefix: {assistant_text[:500]}"
                    ),
                }
            )
            continue
        reasoning, content = extract_reasoning(parsed)
        issues = validate_reasoning(reasoning, action)
        if (
            issues
            and os.environ.get("MNMS_RELAX_REASONING_CHECK", "1") == "1"
            and (action is None or action.get("name") != SUMMARY_TOOL_NAME)
        ):
            return assistant_text, parsed, reasoning, content, latency
        if not issues:
            return assistant_text, parsed, reasoning, content, latency
        last_issues = issues
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Rewrite the same step as exactly one JSON object. Keep the same intended next action if it is valid, "
                    "but provide a concrete reasoning field and matching <think>...</think> content. "
                    f"Reasoning problems: {issues}. "
                    "The reasoning must literally mention the selected tool name, at least one input argument name such as `text` or `image`, "
                    "the input source/value such as the user request or <node-i>.field, and the expected output/purpose. "
                    "Example: \"I will call text generation with the `text` argument set to the user's seed sentence, because the user asked for a continuation. The expected output is generated text.\""
                ),
            }
        )
    raise ValueError(f"Model did not provide valid concrete reasoning after retries: {last_issues}")


def load_node_output(result_folder: Path, task_id: Any, node_id: int) -> tuple[Any, str | None]:
    task_dir = result_folder / str(task_id)
    json_path = task_dir / f"node-{node_id}.json"
    pkl_path = task_dir / f"node-{node_id}.pkl"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8")), str(json_path)
    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            return pickle.load(f), str(pkl_path)
    return None, None


def is_tool_error(output: Any) -> bool:
    if not isinstance(output, dict):
        return False
    message = str(output.get("message", "")).lower()
    error_markers = [
        "too many requests",
        "invalid api key",
        "missing credentials",
        "error",
        "not found",
        "compatibility unknown",
    ]
    return any(marker in message for marker in error_markers)


def is_uninformative_output(output: Any) -> bool:
    if output is None:
        return True
    if not isinstance(output, dict):
        return not bool(str(output).strip())
    if str(output.get("tool_status", "")).lower() in {"unavailable", "disabled"}:
        return True
    text = output.get("text")
    if text is not None and not str(text).strip():
        non_text_values = [
            value
            for key, value in output.items()
            if key not in {"text", "message", "tool_status"} and value not in (None, "", [], {})
        ]
        return not non_text_values
    return False


def should_auto_stop_after_observation(action: dict[str, Any], output: Any, steps: list[dict[str, Any]]) -> bool:
    tool_name = str(action.get("name", "")).lower()
    if not is_uninformative_output(output):
        return False
    if tool_name in {
        "wikipedia simple search",
        "search movie",
        "get weather",
        "get location",
        "get date fact",
        "get year fact",
        "get math fact",
        "get trivia fact",
        "love calculator",
    }:
        return True
    previous_same_empty = 0
    for step in reversed(steps[:-1]):
        prev_action = step.get("action") or {}
        if str(prev_action.get("name", "")).lower() != tool_name:
            break
        if not is_uninformative_output(step.get("tool_output")):
            break
        previous_same_empty += 1
    return previous_same_empty >= 1


def make_auto_stop_step(step_idx: int, action: dict[str, Any], output: Any, observation: str) -> dict[str, Any]:
    tool_name = action.get("name", "tool")
    reasoning = (
        f"The latest observation from {tool_name} is uninformative or unavailable: {observation}. "
        "Calling the same tool again with another paraphrased input would not add evidence, so I will stop the rollout and answer with the limitation instead of repeating a template search step."
    )
    final_answer = (
        f"I could not verify the requested information from the available tool output because {tool_name} returned no usable result."
    )
    return {
        "step": step_idx,
        "reasoning": reasoning,
        "assistant_content": f"<think>{reasoning}</think>",
        "final_answer": final_answer,
        "auto_stop": True,
        "stop_reason": "uninformative_or_unavailable_tool_output",
    }


def should_auto_stop_after_duplicate_observation(action: dict[str, Any], observation: str, steps: list[dict[str, Any]]) -> bool:
    if os.environ.get("MNMS_STOP_ON_DUPLICATE_OBSERVATION", "1") != "1":
        return False
    signature = json.dumps(
        {"name": action.get("name"), "args": action.get("args"), "observation": observation},
        ensure_ascii=False,
        sort_keys=True,
    )
    for step in steps[:-1]:
        prev_action = step.get("action")
        if not prev_action or prev_action.get("name") == SUMMARY_TOOL_NAME:
            continue
        prev_signature = json.dumps(
            {"name": prev_action.get("name"), "args": prev_action.get("args"), "observation": step.get("observation", "")},
            ensure_ascii=False,
            sort_keys=True,
        )
        if prev_signature == signature:
            return True
    return False


def make_duplicate_stop_step(step_idx: int, action: dict[str, Any], observation: str) -> dict[str, Any]:
    tool_name = action.get("name", "tool")
    reasoning = (
        f"The latest {tool_name} call repeated the same arguments and produced the same observation as before: {observation}. "
        "Repeating it again would not add new evidence, so I will stop and provide the final answer from the accumulated observations."
    )
    return {
        "step": step_idx,
        "reasoning": reasoning,
        "assistant_content": f"<think>{reasoning}</think>",
        "final_answer": f"Final answer based on the completed tool observations: {observation}",
        "auto_stop": True,
        "stop_reason": "duplicate_tool_observation",
    }


def _extract_summary_text(action: dict[str, Any]) -> str:
    args = action.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return args
    if isinstance(args, dict):
        for key in ("content", "summary", "current_information", "text"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(args, ensure_ascii=False)
    return str(args)


def _text_pad_enabled() -> bool:
    return os.environ.get("SUMMARY_KL_TEXT_PAD", os.environ.get("TAU2_SUMMARY_KL_TEXT_PAD", "0")).lower() in {
        "1",
        "true",
        "yes",
    }


def _make_inert_text_pad(original_text: Any) -> str:
    original = str(original_text or "")
    if not original.strip():
        return ""
    unit = " [INERT_TEXT_PAD]"
    target_chars = max(len(original), len(unit))
    return (unit * ((target_chars // len(unit)) + 1))[:target_chars]


def _clear_mnms_message_reasoning(message: dict[str, str], *, replacement_text: str = "") -> None:
    try:
        obj = json.loads(message.get("content") or "{}")
        if isinstance(obj, dict):
            obj["content"] = replacement_text
            obj["reasoning"] = replacement_text
            message["content"] = json.dumps(obj, ensure_ascii=False)
            return
    except Exception:
        pass
    message["content"] = replacement_text


def _extract_mnms_structured_visual_evidence(text: str, max_lines: int = 4, max_chars: int = 600) -> str:
    """Retain only explicit, structured visual facts.

    Generic planning language (including phrases such as ``text
    classification`` or ``image generation``) is not visual evidence.
    """
    if not text:
        return ""
    strong_visual_terms = (
        "bbox", "bounding box", "coordinate", "pixel", "crop id", "crop",
        "ocr", "optical character", "color", "colour", "left", "right",
        "top", "bottom", "edge", "texture", "mask", "segmentation",
    )
    planning_noise = (
        "i need to", "i should", "i will", "let me", "next step",
        "the user asks", "the user wants", "call the", "use the tool",
        "expected output", "appropriate tool", "will provide", "will generate",
    )
    clean = re.sub(r"</?think>", " ", str(text), flags=re.I)
    clean = re.sub(r"\s+", " ", clean).strip()
    evidence: list[str] = []

    filenames = sorted(
        set(
            re.findall(
                r"(?<![\w.-])[\w./-]+\.(?:png|jpe?g|webp|bmp|tiff?|gif)(?![\w.-])",
                clean,
                flags=re.I,
            )
        )
    )
    if filenames:
        evidence.append("image file(s): " + ", ".join(filenames[:4]))

    node_refs = sorted(
        set(
            re.findall(
                r"<node-\d+>\.(?:image|objects?|bbox(?:es)?|mask|text)",
                clean,
                flags=re.I,
            )
        )
    )
    if node_refs:
        evidence.append("visual dependency: " + ", ".join(node_refs[:6]))

    for piece in re.split(r"(?<=[.!?。！？])\s+|[;\n]+", clean):
        piece = piece.strip(" -•\t")
        low = piece.lower()
        if not piece or len(piece) > 300 or any(noise in low for noise in planning_noise):
            continue
        has_structured_numbers = bool(
            re.search(r"\[[^\]]*\d+[^\]]*\]|\([^\)]*\d+[^\)]*\)", piece)
        )
        if any(term in low for term in strong_visual_terms) or has_structured_numbers:
            evidence.append(piece)
            if len(evidence) >= max_lines:
                break
    if not evidence:
        return ""
    return (
        "Structured visual evidence kept:\n"
        + "\n".join(f"- {line}" for line in evidence[:max_lines])
    )[:max_chars]


def _apply_summary_kl_prune(
    *,
    args: argparse.Namespace,
    task_id: Any,
    messages: list[dict[str, str]],
    assistant_message_indices: list[int],
    prior_assistant_raw_texts: list[str],
    current_pre_tool_text: str,
    summary_text: str,
    steps: list[dict[str, Any]],
    candidate_index_offset: int = 0,
) -> dict[str, Any] | None:
    if not summary_kl_enabled():
        return None
    prune_strategy = os.environ.get("SUMMARY_KL_PRUNE_STRATEGY", os.environ.get("TAU2_SUMMARY_KL_PRUNE_STRATEGY", "kl")).strip().lower()
    if prune_strategy in {"direct_delete", "direct-delete", "delete_all", "delete-all", "all_reasoning"}:
        result = {
            "out_path": None,
            "prune_indices": [i for i, text in enumerate(prior_assistant_raw_texts) if text and text.strip()],
            "step_decisions": [
                {
                    "step": "prior_assistant",
                    "global_index": i,
                    "keep": False,
                    "prune_strategy": prune_strategy,
                }
                for i, text in enumerate(prior_assistant_raw_texts)
                if text and text.strip()
            ],
            "n_response_tokens": None,
            "reverse_kl_mean": None,
            "direct_delete": True,
        }
    else:
        previous_summary_kl_base_url = os.environ.get("SUMMARY_KL_BASE_URL")
        if previous_summary_kl_base_url is None:
            os.environ["SUMMARY_KL_BASE_URL"] = args.base_url
        try:
            result = log_summarization_kl(
                system_prompt=messages[0]["content"],
                summary_text=summary_text,
                prior_assistant_raw_texts=prior_assistant_raw_texts,
                current_pre_tool_text=current_pre_tool_text,
                model=args.model,
                task_id=task_id,
                output_dir=str(Path(args.output_file).resolve().parent / "summary_kl_logs"),
            )
        finally:
            if previous_summary_kl_base_url is None:
                os.environ.pop("SUMMARY_KL_BASE_URL", None)
            else:
                os.environ["SUMMARY_KL_BASE_URL"] = previous_summary_kl_base_url
    if not isinstance(result, dict):
        return result
    if os.environ.get("SUMMARY_KL_APPLY_PRUNE", "1").lower() in {"0", "false", "no"}:
        result["computed_prune_indices"] = list(result.get("prune_indices") or [])
        result["prune_indices"] = []
        result["prune_apply_disabled"] = True
    v5_mode = prune_strategy in {"v5", "kl_v5", "kl-v5"}
    decision_by_index = {
        int(decision["global_index"]) + candidate_index_offset: decision
        for decision in (result.get("step_decisions") or [])
        if decision.get("step") == "prior_assistant" and "global_index" in decision
    }
    low_kl_indices = set(result.get("prune_indices") or [])
    rewrite_indices = (
        [i for i, text in enumerate(prior_assistant_raw_texts) if text and text.strip()]
        if v5_mode
        else list(low_kl_indices)
    )
    pruned = []
    visual_rewritten = []
    for candidate_idx in rewrite_indices:
        history_idx = candidate_idx + candidate_index_offset
        if history_idx < 0 or history_idx >= len(assistant_message_indices):
            continue
        msg_idx = assistant_message_indices[history_idx]
        if msg_idx < 0 or msg_idx >= len(messages):
            continue
        original_text = ""
        if history_idx < len(steps):
            original_text = steps[history_idx].get("assistant_content") or steps[history_idx].get("reasoning") or ""
        if not original_text:
            original_text = messages[msg_idx].get("content") or ""
        low_kl = candidate_idx in low_kl_indices
        if v5_mode and not low_kl:
            replacement_text = _extract_mnms_structured_visual_evidence(original_text)
        else:
            replacement_text = _make_inert_text_pad(original_text) if _text_pad_enabled() else ""
        _clear_mnms_message_reasoning(messages[msg_idx], replacement_text=replacement_text)
        if history_idx < len(steps):
            steps[history_idx].setdefault(
                "assistant_content_before_prune",
                steps[history_idx].get("assistant_content"),
            )
            steps[history_idx]["assistant_content"] = replacement_text
            steps[history_idx].setdefault(
                "reasoning_before_prune",
                steps[history_idx].get("reasoning"),
            )
            steps[history_idx]["reasoning"] = replacement_text
            steps[history_idx]["summary_kl_pruned"] = True
            steps[history_idx]["summary_kl_low"] = low_kl
            if v5_mode and not low_kl:
                steps[history_idx]["summary_kl_v5_visual_only"] = True
            if replacement_text:
                steps[history_idx]["summary_kl_text_pad"] = True
                steps[history_idx]["summary_kl_text_pad_chars"] = len(replacement_text)
        record = {
            "prior_index": history_idx,
            "candidate_index": candidate_idx,
            "message_index": msg_idx,
        }
        if v5_mode and not low_kl:
            visual_rewritten.append(record)
        else:
            pruned.append(record)
    result["applied_pruned_messages"] = pruned
    result["applied_visual_only_messages"] = visual_rewritten
    result["v5_mode"] = v5_mode
    result["decision_by_index"] = decision_by_index
    result["candidate_index_offset"] = candidate_index_offset
    return result


def _build_controller_summary(user_request: str, steps: list[dict[str, Any]]) -> str:
    completed = []
    observations = []
    for step in steps:
        action = step.get("action")
        if not action or action.get("name") == SUMMARY_TOOL_NAME:
            continue
        completed.append(
            {
                "step": step.get("step"),
                "tool": action.get("name"),
                "args": action.get("args"),
            }
        )
        obs = step.get("observation")
        if obs:
            observations.append({"step": step.get("step"), "observation": str(obs)[:600]})
    return (
        "Controller summary for summarize_the_task. "
        f"Original request: {user_request}. "
        f"Completed real tool calls with arguments: {json.dumps(completed, ensure_ascii=False)}. "
        f"Key observations so far: {json.dumps(observations, ensure_ascii=False)}. "
        "Known facts are exactly the observations above; no unavailable external facts are assumed. "
        "Unresolved uncertainty: whether the current evidence is sufficient to produce the final answer. "
        "Single next intended action: continue from this compact memory and either call the next necessary MNMS tool or output final_answer if the request is already fulfilled."
    )


def rollout_one(
    client: dict[str, str],
    args: argparse.Namespace,
    executor: Executor,
    verifier: Verifier,
    sample: dict[str, Any],
) -> dict[str, Any]:
    task_id = sample["id"]
    user_request = sample.get("user_request") or sample.get("human_rewrite_request") or ""
    messages = initial_messages(user_request, plan_only=args.plan_only, model_name=args.model)
    steps = []
    assistant_message_indices: list[int] = []
    prior_assistant_raw_texts: list[str] = []
    summary_events: list[dict[str, Any]] = []
    discarded_summary_probes: list[dict[str, Any]] = []
    hidden_summary_messages: list[dict[str, str]] = []
    final_answer = None
    status = "max_steps"
    summary_nudge_sent = False
    summary_nudge_after = int(os.environ.get("MNMS_SUMMARY_NUDGE_AFTER", "0"))
    abort_on_model_connection_error = os.environ.get("MNMS_ABORT_ON_MODEL_CONNECTION_ERROR", "0").lower() in {"1", "true", "yes"}
    summary_force_repeat_after = int(os.environ.get("MNMS_SUMMARY_FORCE_REPEAT_AFTER", "0"))
    summary_force_token_threshold = int(os.environ.get("MNMS_SUMMARY_FORCE_TOKEN_THRESHOLD", "0"))
    summary_force_first_default = "0" if summary_force_token_threshold > 0 else "1"
    summary_force_first_after = int(os.environ.get("MNMS_SUMMARY_FORCE_FIRST_AFTER", summary_force_first_default))
    summary_runtime_enforce = os.environ.get("MNMS_SUMMARY_RUNTIME_ENFORCE", "0") == "1"
    summary_gate_enable = os.environ.get("MNMS_SUMMARY_GATE_ENABLE", "0").lower() in {"1", "true", "yes"}
    summary_gate_discard_no_prune = os.environ.get(
        "MNMS_SUMMARY_GATE_DISCARD_NO_PRUNE",
        "1" if summary_gate_enable else "0",
    ).lower() in {"1", "true", "yes"}
    summary_gate_min_real_tools = int(os.environ.get("MNMS_SUMMARY_GATE_MIN_REAL_TOOLS", "2"))
    summary_gate_min_reasoning_tokens = int(os.environ.get("MNMS_SUMMARY_GATE_MIN_REASONING_TOKENS", "80"))
    summary_gate_min_candidates = int(os.environ.get("MNMS_SUMMARY_GATE_MIN_CANDIDATES", "2"))
    no_tool_baseline = os.environ.get("MNMS_NO_TOOL_BASELINE", "0").lower() in {"1", "true", "yes"}
    summary_trace_hidden = os.environ.get("MNMS_SUMMARY_TRACE_HIDDEN", "0").lower() in {"1", "true", "yes"}
    last_summary_real_tool_count = 0
    last_summary_assistant_text_count = 0

    def unsummarized_reasoning_token_estimate() -> int:
        texts = prior_assistant_raw_texts[last_summary_assistant_text_count:]
        return estimate_tokens("\n\n".join(texts))

    def unsummarized_candidate_count() -> int:
        return sum(1 for text in prior_assistant_raw_texts[last_summary_assistant_text_count:] if text and text.strip())

    def summary_gate_allows(real_tool_count: int) -> bool:
        if not summary_gate_enable:
            return True
        return (
            real_tool_count >= summary_gate_min_real_tools
            and unsummarized_reasoning_token_estimate() >= summary_gate_min_reasoning_tokens
            and unsummarized_candidate_count() >= summary_gate_min_candidates
        )

    def summary_due(real_tool_count: int) -> bool:
        if not summary_runtime_enforce or not summary_kl_enabled():
            return False
        if not summary_gate_allows(real_tool_count):
            return False
        if (
            summary_force_token_threshold > 0
            and unsummarized_reasoning_token_estimate() >= summary_force_token_threshold
        ):
            return True
        if not summary_events:
            return summary_force_first_after > 0 and real_tool_count >= summary_force_first_after
        return (
            summary_force_repeat_after > 0
            and real_tool_count - last_summary_real_tool_count >= summary_force_repeat_after
        )

    def summary_due_label() -> str:
        token_estimate = unsummarized_reasoning_token_estimate()
        if summary_gate_enable and not summary_gate_allows(real_tool_count):
            return (
                "the gated summarize_the_task policy has not been satisfied: "
                f"real_tool_count={real_tool_count}, unsummarized_reasoning_tokens={token_estimate}, "
                f"unsummarized_candidate_count={unsummarized_candidate_count()}"
            )
        if summary_force_token_threshold > 0 and token_estimate >= summary_force_token_threshold:
            return (
                "the unsummarized assistant reasoning is estimated at "
                f"{token_estimate} tokens, above the {summary_force_token_threshold} token threshold"
            )
        if not summary_events:
            return "the mandatory first summarize_the_task call is due"
        return f"a periodic summarize_the_task call is due after every {summary_force_repeat_after} real tool observations"

    # Summary/KL probes are controller-side diagnostics and do not consume
    # the benchmark's real tool/action step budget.
    step_idx = 0
    protocol_rejections = 0
    verification_rejections = 0
    max_protocol_rejections = max(1, int(os.environ.get("MNMS_MAX_PROTOCOL_REJECTIONS", "5")))
    while step_idx < args.max_steps:
        real_tool_count = len([s for s in steps if "action" in s and s["action"].get("name") != SUMMARY_TOOL_NAME])
        if (
            os.environ.get("MNMS_SUMMARY_AUTO_INSERT", "0") == "1"
            and summary_due(real_tool_count)
        ):
            summary_text = _build_controller_summary(user_request, steps)
            summary_reasoning_token_estimate = unsummarized_reasoning_token_estimate()
            summary_incremental = os.environ.get(
                "MNMS_SUMMARY_INCREMENTAL",
                "0",
            ).lower() in {"1", "true", "yes"}
            candidate_start = last_summary_assistant_text_count if summary_incremental else 0
            candidate_reasoning = (
                prior_assistant_raw_texts[candidate_start:]
                if summary_incremental
                else prior_assistant_raw_texts
            )
            last_summary_real_tool_count = real_tool_count
            last_summary_assistant_text_count = len(prior_assistant_raw_texts)
            action = {
                "id": step_idx,
                "name": SUMMARY_TOOL_NAME,
                "args": {"content": summary_text},
            }
            event = _apply_summary_kl_prune(
                args=args,
                task_id=task_id,
                messages=messages,
                assistant_message_indices=assistant_message_indices,
                prior_assistant_raw_texts=candidate_reasoning,
                current_pre_tool_text="",
                summary_text=summary_text,
                steps=steps,
                candidate_index_offset=candidate_start,
            )
            prune_indices = []
            if isinstance(event, dict):
                prune_indices = event.get("prune_indices") or []
            if summary_gate_discard_no_prune and not prune_indices:
                discarded_summary_probes.append(
                    {
                        "step": step_idx,
                        "summary_real_tool_count": real_tool_count,
                        "summary_reasoning_token_estimate": summary_reasoning_token_estimate,
                        "summary_kl": event,
                        "discard_reason": "no_prune_indices",
                    }
                )
            else:
                observation = json.dumps({"summary": summary_text}, ensure_ascii=False)
                reasoning = "The controller inserted the mandatory summarize_the_task call because the summary schedule is due."
                step_record = {
                    "step": step_idx,
                    "reasoning": reasoning,
                    "assistant_content": f"<think>{reasoning}</think>",
                    "action": action,
                    "summary_tool": True,
                    "summary_auto_inserted": True,
                    "summary_real_tool_count": real_tool_count,
                    "summary_reasoning_token_estimate": summary_reasoning_token_estimate,
                    "summary_kl": event,
                    "observation": observation,
                    "raw_response": "",
                    "latency_sec": 0.0,
                }
                assistant_summary_message = {
                    "role": "assistant",
                    "content": json.dumps({"content": step_record["assistant_content"], "action": action}, ensure_ascii=False),
                }
                user_summary_message = {
                    "role": "user",
                    "content": (
                        f"SUMMARY OBSERVATION {step_idx}: {observation}\n"
                        "Continue from this compact memory. If the original user request is fulfilled, output final_answer. "
                        f"Otherwise output step {step_idx + 1} with a new <think> reasoning block and one action."
                    ),
                }
                if summary_trace_hidden:
                    # Record only audit metadata.  The summary text itself is
                    # one-shot working context and is absent from the trace.
                    summary_events.append({
                        "hidden": True,
                        "summary_auto_inserted": True,
                        "summary_real_tool_count": real_tool_count,
                        "summary_reasoning_token_estimate": summary_reasoning_token_estimate,
                        "summary_kl": event,
                    })
                    # Do not expose a synthetic assistant tool-call shell:
                    # models tend to imitate it and call the hidden probe
                    # again.  The state is supplied exactly once as controller
                    # context and removed after the next successful inference.
                    hidden_state_message = {
                        "role": "user",
                        "content": (
                            "[INTERNAL TASK-STATE PROBE; NOT A TOOL REQUEST]\n"
                            f"{summary_text}\n"
                            "Use this state only for the next plan decision. "
                            "Do not call summarize_the_task; output the next real "
                            "MNMS plan node or final_answer."
                        ),
                    }
                    messages.append(hidden_state_message)
                    hidden_summary_messages[:] = [hidden_state_message]
                else:
                    steps.append(step_record)
                    summary_events.append(step_record)
                    messages.extend([assistant_summary_message, user_summary_message])
                continue

        try:
            assistant_text, parsed, reasoning, content, latency = call_model_for_step(
                client=client,
                args=args,
                messages=messages,
                step_idx=step_idx,
            )
        except Exception as exc:
            protocol_rejections += 1
            if abort_on_model_connection_error and any(
                marker in repr(exc)
                for marker in (
                    "ConnectionRefusedError",
                    "RemoteDisconnected",
                    "ConnectionResetError",
                    "timed out",
                    "TimeoutError",
                )
            ):
                raise
            steps.append(
                {
                    "step": step_idx,
                    "error": repr(exc),
                    "messages_len": len(messages),
                    "num_prior_assistant_texts": len(prior_assistant_raw_texts),
                    "protocol_rejected": True,
                    "reject_reason": "invalid_model_response_recovered",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not usable by the MNMS controller. "
                        f"Retry step {step_idx} as exactly one valid JSON object and nothing else. "
                        "If the task is complete, use "
                        '{"content":"<think>specific reason</think>","reasoning":"specific reason","final_answer":"answer"}. '
                        "Otherwise use "
                        '{"content":"<think>specific reason</think>","reasoning":"specific reason","action":{"id":'
                        f'{step_idx},"name":"tool name","args":{{"arg_name":"arg_value"}}}}. '
                        "Do not output a JSON object with only `content`; include either `action` or `final_answer`."
                    ),
                }
            )
            if protocol_rejections >= max_protocol_rejections:
                status = "protocol_rejected"
                break
            continue
        if hidden_summary_messages:
            hidden_ids = {id(message) for message in hidden_summary_messages}
            messages[:] = [message for message in messages if id(message) not in hidden_ids]
            hidden_summary_messages.clear()
        protocol_rejections = 0
        action = normalize_action(parsed, expected_id=step_idx)
        if no_tool_baseline and action is not None:
            steps.append(
                {
                    "step": step_idx,
                    "reasoning": reasoning,
                    "assistant_content": content,
                    "attempted_action": action,
                    "raw_response": assistant_text,
                    "latency_sec": latency,
                    "protocol_rejected": True,
                    "reject_reason": "tool_call_in_no_tool_baseline",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Protocol violation: no tools are available in this baseline run. "
                        "Do not output an action field, do not simulate a tool call, and do not refer to tool outputs. "
                        "Reply now as one JSON object with content, reasoning, and final_answer only."
                    ),
                }
            )
            continue
        real_tool_count = len([s for s in steps if "action" in s and s["action"].get("name") != SUMMARY_TOOL_NAME])
        if action is None:
            if summary_due(real_tool_count):
                steps.append(
                    {
                        "step": step_idx,
                        "reasoning": reasoning,
                        "assistant_content": content,
                        "attempted_final_answer": parsed.get("final_answer", ""),
                        "raw_response": assistant_text,
                        "latency_sec": latency,
                        "protocol_rejected": True,
                        "reject_reason": "final_answer_before_summarize_the_task",
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Protocol violation: {summary_due_label()}, "
                            "but you attempted final_answer before calling summarize_the_task. "
                            f"Your next response MUST output step {step_idx} with a <think> block "
                            "and exactly one action named summarize_the_task. Its args.content must "
                            "summarize the original request, every completed tool call with arguments, "
                            "key observations, known facts, unresolved uncertainty, and the single next "
                            "intended action. Do not output final_answer until the summary tool has returned."
                        ),
                    }
                )
                continue
            final_answer = parsed.get("final_answer", "")
            status = "finished"
            steps.append(
                {
                    "step": step_idx,
                    "reasoning": reasoning,
                    "assistant_content": content,
                    "final_answer": final_answer,
                    "raw_response": assistant_text,
                    "latency_sec": latency,
                }
            )
            messages.append({"role": "assistant", "content": json.dumps({"content": content, "final_answer": final_answer}, ensure_ascii=False)})
            break

        if summary_due(real_tool_count) and action["name"] != SUMMARY_TOOL_NAME:
            steps.append(
                {
                    "step": step_idx,
                    "reasoning": reasoning,
                    "assistant_content": content,
                    "attempted_action": action,
                    "raw_response": assistant_text,
                    "latency_sec": latency,
                    "protocol_rejected": True,
                    "reject_reason": "tool_before_summarize_the_task",
                }
                )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Protocol violation: {summary_due_label()}, "
                        f"but tried to call {action['name']} before summarize_the_task. The attempted "
                        "tool call was not executed. Your next response MUST output one action named "
                        "summarize_the_task with args.content summarizing the original request, every "
                        "completed tool call with arguments, key observations, known facts, unresolved "
                        "uncertainty, and the single next intended action. Do not call any other real "
                        "tool until the summary tool has returned."
                    ),
                }
            )
            continue

        if action["name"] == SUMMARY_TOOL_NAME:
            if summary_trace_hidden:
                protocol_rejections += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Protocol violation: summarize_the_task is a hidden "
                            "controller probe in this condition and is not an "
                            "available plan node. Output the next real MNMS tool "
                            "node, or final_answer if the plan is complete."
                        ),
                    }
                )
                if protocol_rejections >= max_protocol_rejections:
                    status = "protocol_rejected"
                    break
                continue
            summary_text = _extract_summary_text(action)
            summary_reasoning_token_estimate = unsummarized_reasoning_token_estimate()
            last_summary_real_tool_count = real_tool_count
            last_summary_assistant_text_count = len(prior_assistant_raw_texts)
            event = _apply_summary_kl_prune(
                args=args,
                task_id=task_id,
                messages=messages,
                assistant_message_indices=assistant_message_indices,
                prior_assistant_raw_texts=prior_assistant_raw_texts,
                current_pre_tool_text=content or reasoning,
                summary_text=summary_text,
                steps=steps,
            )
            prune_indices = []
            if isinstance(event, dict):
                prune_indices = event.get("prune_indices") or []
            if summary_gate_discard_no_prune and not prune_indices:
                discarded_summary_probes.append(
                    {
                        "step": step_idx,
                        "summary_real_tool_count": real_tool_count,
                        "summary_reasoning_token_estimate": summary_reasoning_token_estimate,
                        "summary_kl": event,
                        "discard_reason": "model_summary_no_prune_indices",
                    }
                )
                continue
            observation = json.dumps({"summary": summary_text}, ensure_ascii=False)
            step_record = {
                "step": step_idx,
                "reasoning": reasoning,
                "assistant_content": content,
                "action": action,
                "summary_tool": True,
                "summary_real_tool_count": real_tool_count,
                "summary_reasoning_token_estimate": summary_reasoning_token_estimate,
                "summary_kl": event,
                "observation": observation,
                "raw_response": assistant_text,
                "latency_sec": latency,
            }
            steps.append(step_record)
            summary_events.append(step_record)
            messages.append({"role": "assistant", "content": json.dumps({"content": content, "action": action}, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"SUMMARY OBSERVATION {step_idx}: {observation}\n"
                        "Continue from this compact memory. If the original user request is fulfilled, output final_answer. "
                        f"Otherwise output step {step_idx + 1} with a new <think> reasoning block and one action."
                    ),
                }
            )
            continue

        action["id"] = len([s for s in steps if "action" in s and s["action"].get("name") != SUMMARY_TOOL_NAME])

        verify_result = verifier.verify_single_node(action)
        exec_result = {"status": False, "message": "Skipped because verification failed."}
        output = None
        output_path = None
        tool_error = False

        if verify_result["status"] and args.plan_only:
            exec_result = {
                "status": True,
                "message": verify_result["message"],
                "plan_only": True,
                "executed_tools": False,
            }
        elif verify_result["status"]:
            exec_result = executor.execute(task_id, action)
            output, output_path = load_node_output(Path(args.result_folder), task_id, action["id"])
            tool_error = (not exec_result["status"]) or is_tool_error(output)

        observation = (
            verify_result["message"]
            if args.plan_only
            else compact_observation(output if output is not None else exec_result["message"])
        )
        step_record = {
            "step": step_idx,
            "reasoning": reasoning,
            "assistant_content": content,
            "action": action,
            "verify": verify_result,
            "execution": exec_result,
            "tool_output": output,
            "tool_output_path": output_path,
            "tool_error": tool_error,
            "observation": observation,
            "raw_response": assistant_text,
            "latency_sec": latency,
        }
        if args.plan_only and not verify_result["status"]:
            # A rejected proposal is feedback/retry history, not part of the
            # accumulated plan evaluated by MNMS.
            step_record["attempted_action"] = step_record.pop("action")
        steps.append(step_record)

        if (
            not args.plan_only
            and verify_result["status"]
            and exec_result["status"]
            and should_auto_stop_after_observation(action, output, steps)
        ):
            stop_step = make_auto_stop_step(step_idx + 1, action, output, observation)
            steps.append(stop_step)
            final_answer = stop_step["final_answer"]
            status = "finished_tool_unavailable"
            messages.append({"role": "assistant", "content": json.dumps({"content": content, "action": action}, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"OBSERVATION {step_idx}: {observation}\n"
                        "The observation is empty or the tool is unavailable. Do not call the same tool again with a rephrased query. "
                        "Provide final_answer with the limitation."
                    ),
                }
            )
            break

        if (
            not args.plan_only
            and verify_result["status"]
            and exec_result["status"]
            and should_auto_stop_after_duplicate_observation(action, observation, steps)
        ):
            stop_step = make_duplicate_stop_step(step_idx + 1, action, observation)
            steps.append(stop_step)
            final_answer = stop_step["final_answer"]
            status = "finished"
            messages.append({"role": "assistant", "content": json.dumps({"content": content, "action": action}, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"OBSERVATION {step_idx}: {observation}\n"
                        "The same tool with the same arguments already produced this same observation. "
                        "Stop repeating it and provide final_answer from the accumulated evidence."
                    ),
                }
            )
            break

        messages.append({"role": "assistant", "content": json.dumps({"content": content, "action": action}, ensure_ascii=False)})
        assistant_message_indices.append(len(messages) - 1)
        prior_assistant_raw_texts.append(content or reasoning or assistant_text)
        step_reasoning_policy = os.environ.get("MNMS_STEP_REASONING_POLICY", "full").strip().lower()
        if (
            args.plan_only
            and verify_result["status"]
            and step_reasoning_policy in {"delete_all", "visual_only"}
        ):
            original_reasoning = step_record.get("assistant_content") or step_record.get("reasoning") or ""
            replacement_reasoning = (
                ""
                if step_reasoning_policy == "delete_all"
                else _extract_mnms_structured_visual_evidence(original_reasoning)
            )
            _clear_mnms_message_reasoning(
                messages[assistant_message_indices[-1]],
                replacement_text=replacement_reasoning,
            )
            step_record["reasoning_before_step_policy"] = step_record.get("reasoning")
            step_record["assistant_content_before_step_policy"] = step_record.get("assistant_content")
            step_record["reasoning"] = replacement_reasoning
            step_record["assistant_content"] = replacement_reasoning
            step_record["step_reasoning_policy"] = step_reasoning_policy
            step_record["step_reasoning_replacement_chars"] = len(replacement_reasoning)
        summary_nudge = ""
        real_tool_count = len([s for s in steps if "action" in s and s["action"].get("name") != SUMMARY_TOOL_NAME])
        if (
            summary_kl_enabled()
            and summary_nudge_after > 0
            and not summary_events
            and not summary_nudge_sent
            and real_tool_count >= summary_nudge_after
        ):
            summary_nudge = (
                "\nMemory note: You have already received real MNMS tool observations. "
                "For this summary-KL experiment, call summarize_the_task as your next action "
                "with a faithful summary of the original request, completed tool calls, observations, "
                "remaining uncertainty, and next intended action before continuing with real tools or final_answer."
            )
            summary_nudge_sent = True
        messages.append(
            {
                "role": "user",
                "content": (
                    f"OBSERVATION: {observation}\n"
                    + (
                        "If the request has been fulfilled by the planned action graph, output a JSON final_answer now "
                        "(the compatibility equivalent of TERMINATE). Otherwise generate the next distinct THOUGHT and ACTION. "
                        if args.plan_only
                        else "If the original user request is fulfilled, output final_answer. "
                    )
                    + (
                        f"The next accepted action id must be {step_idx + 1}."
                        if args.plan_only and verify_result["status"]
                        else (
                            f"Retry action id {step_idx} after fixing the verifier error."
                            if args.plan_only
                            else f"Otherwise output step {step_idx + 1} with a new <think> reasoning block and one action."
                        )
                    )
                    + summary_nudge
                ),
            }
        )

        if not verify_result["status"]:
            if args.plan_only:
                verification_rejections += 1
                if verification_rejections >= max_protocol_rejections:
                    status = "verification_failed"
                    break
                continue
            status = "verification_failed"
            break
        verification_rejections = 0
        if not exec_result["status"]:
            status = "execution_failed"
            break
        step_idx += 1

    return {
        "id": task_id,
        "user_request": user_request,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "reasoning": [s.get("reasoning", "") for s in steps],
        "prediction": [s["action"] for s in steps if "action" in s and s["action"].get("name") != SUMMARY_TOOL_NAME],
        "summary_events": summary_events,
        "discarded_summary_probes": discarded_summary_probes,
        "num_steps": len([s for s in steps if "action" in s and s["action"].get("name") != SUMMARY_TOOL_NAME]),
        "num_summary_calls": len(summary_events),
        "num_discarded_summary_probes": len(discarded_summary_probes),
        "num_tool_errors": sum(1 for s in steps if s.get("tool_error")),
    }


def main():
    args = get_args()
    args.input_file = str(Path(args.input_file).resolve())
    args.output_file = str(Path(args.output_file).resolve())
    args.result_folder = str(Path(args.result_folder).resolve())
    # MNMS Verifier loads "mnms/tool_graph.pkl" relative to the repo root.
    os.chdir(Path(__file__).resolve().parents[1])
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_folder).mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input_file)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"Invalid shard {args.shard_index}/{args.num_shards}; "
            "require num_shards >= 1 and 0 <= shard_index < num_shards"
        )
    rows = rows[args.shard_index :: args.num_shards]
    if args.limit is not None:
        rows = rows[: args.limit]
    done = load_done_ids(out_path) if args.resume else set()

    client = {"base_url": args.base_url, "api_key": args.api_key}
    executor = Executor(result_folder=args.result_folder)
    verifier = Verifier(tool_metadata=TOOL_METADATA)

    with out_path.open("a" if args.resume else "w", encoding="utf-8") as wf:
        for idx, sample in enumerate(rows):
            if sample["id"] in done:
                continue
            print(f"Rolling out task {sample['id']} ({idx + 1}/{len(rows)})")
            try:
                result = rollout_one(client, args, executor, verifier, sample)
            except Exception as exc:
                if os.environ.get("MNMS_ABORT_ON_MODEL_CONNECTION_ERROR", "0").lower() in {"1", "true", "yes"} and any(
                    marker in repr(exc)
                    for marker in (
                        "ConnectionRefusedError",
                        "RemoteDisconnected",
                        "ConnectionResetError",
                        "timed out",
                        "TimeoutError",
                    )
                ):
                    print(f"  aborting after model connection error: {exc!r}", flush=True)
                    raise
                result = {
                    "id": sample.get("id"),
                    "user_request": sample.get("user_request"),
                    "status": "rollout_error",
                    "error": repr(exc),
                    "steps": [],
                }
            wf.write(json.dumps(result, ensure_ascii=False) + "\n")
            wf.flush()
            print(f"  status={result.get('status')} steps={result.get('num_steps')} tool_errors={result.get('num_tool_errors')}")


if __name__ == "__main__":
    main()
