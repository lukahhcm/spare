import json
import hashlib
import math
import os
import random
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


SUMMARY_TOOL_NAME = "summarize_the_task"


def summary_tool_metadata() -> dict[str, Any]:
    return {
        "description": (
            "Summarize the original request, prior reasoning, completed tool calls, "
            "observations, known facts, remaining uncertainty, and the next intended action. "
            "This is an internal memory tool; it does not solve the user request by itself."
        ),
        "input": {"arg_name": ["content"]},
        "output": {"arg_name": ["summary"]},
    }


def enabled() -> bool:
    return os.environ.get("SUMMARY_KL_ENABLE", os.environ.get("TAU2_SUMMARY_KL_ENABLE", "")).lower() in {
        "1",
        "true",
        "yes",
    }


def _env(name: str, default: str) -> str:
    return os.environ.get(name, os.environ.get("TAU2_" + name, default))


def _post_json(url: str, payload: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY') or 'EMPTY'}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code} from {url}: {body[:800]}") from err


def _base_url() -> str:
    return (
        os.environ.get("SUMMARY_KL_BASE_URL")
        or os.environ.get("TAU2_VLLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://127.0.0.1:8000/v1"
    ).rstrip("/")


def _root_url() -> str:
    base = _base_url()
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _model(default_model: str | None) -> str | None:
    return os.environ.get("SUMMARY_KL_MODEL") or os.environ.get("TAU2_SUMMARY_KL_MODEL") or default_model


def _tokenize(text: str, model: str) -> list[int]:
    if not text:
        return []
    data = _post_json(
        _root_url() + "/tokenize",
        {"model": model, "prompt": text, "add_special_tokens": False},
        timeout=120,
    )
    return [int(x) for x in (data.get("tokens") or data.get("input_ids") or [])]


def _score_prompt_logprobs(full_ids: list[int], n_response: int, model: str, top_k: int) -> list[dict[str, Any]]:
    if n_response <= 0 or len(full_ids) <= 1:
        return []
    data = _post_json(
        _base_url() + "/completions",
        {
            "model": model,
            "prompt": full_ids,
            "max_tokens": 1,
            "echo": True,
            "temperature": 0.0,
            "logprobs": top_k,
        },
        timeout=600,
    )
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("vLLM /completions returned no choices")
    prompt_logprobs = choices[0].get("prompt_logprobs")
    if prompt_logprobs is None:
        logprobs = choices[0].get("logprobs") or {}
        prompt_logprobs = logprobs.get("top_logprobs")
    if prompt_logprobs is None:
        raise RuntimeError("vLLM /completions response missing prompt logprobs")

    out = []
    start = len(full_ids) - n_response
    for i in range(start, len(full_ids)):
        entry = prompt_logprobs[i] if i < len(prompt_logprobs) else None
        topk = {}
        if isinstance(entry, dict):
            for key, value in entry.items():
                logprob = value.get("logprob") if isinstance(value, dict) else value
                if logprob is not None:
                    topk[str(key)] = float(logprob)
        out.append({"target_id": int(full_ids[i]), "topk": topk})
    return out


def _tokenize_segments(texts: list[str], current_text: str, model: str) -> tuple[list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    sep = _tokenize("\n\n", model)
    ids: list[int] = []
    prior_ranges: list[tuple[int, int]] = []
    current_ranges: list[tuple[int, int]] = []

    def append(text: str) -> tuple[int, int] | None:
        if not text or not text.strip():
            return None
        if ids and sep:
            ids.extend(sep)
        part = _tokenize(text, model)
        if not part:
            return None
        start = len(ids)
        ids.extend(part)
        return start, len(ids)

    for text in texts:
        rng = append(text)
        if rng is not None:
            prior_ranges.append(rng)
    rng = append(current_text)
    if rng is not None:
        current_ranges.append(rng)
    return ids, prior_ranges, current_ranges


def _reverse_kl(student: list[dict[str, Any]], teacher: list[dict[str, Any]]) -> list[float]:
    values = []
    for s_pos, t_pos in zip(student, teacher):
        s_top = s_pos.get("topk") or {}
        t_top = t_pos.get("topk") or {}
        if not s_top or not t_top:
            values.append(0.0)
            continue
        t_floor = min(t_top.values()) - 5.0
        kl = 0.0
        for token_id, lp_s in s_top.items():
            lp_t = t_top.get(token_id, t_floor)
            kl += math.exp(lp_s) * (lp_s - lp_t)
        values.append(float(kl))
    return values


def _random_matched_prune_indices(
    *,
    candidates: list[int],
    count: int,
    task_id: Any,
    summary_text: str,
    prior_assistant_raw_texts: list[str],
) -> list[int]:
    if count <= 0 or not candidates:
        return []
    seed = os.environ.get("SUMMARY_KL_RANDOM_SEED", os.environ.get("TAU2_SUMMARY_KL_RANDOM_SEED", "0"))
    seed_material = json.dumps(
        {
            "seed": seed,
            "task_id": task_id,
            "summary_text": summary_text,
            "n_prior": len(prior_assistant_raw_texts),
            "prior_lengths": [len(text or "") for text in prior_assistant_raw_texts],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    return sorted(rng.sample(candidates, min(count, len(candidates))))


def log_summarization_kl(
    *,
    system_prompt: str,
    summary_text: str,
    prior_assistant_raw_texts: list[str],
    current_pre_tool_text: str,
    model: str | None,
    task_id: Any = None,
    output_dir: str | None = None,
) -> dict[str, Any] | None:
    if not enabled():
        return None
    model_name = _model(model)
    if not model_name:
        return None

    top_k = int(os.environ.get("SUMMARY_KL_TOPK", os.environ.get("TAU2_SUMMARY_KL_TOPK", "20")))
    max_length = int(os.environ.get("SUMMARY_KL_MAX_LENGTH", os.environ.get("TAU2_SUMMARY_KL_MAX_LENGTH", "8192")))
    threshold = float(os.environ.get("SUMMARY_KL_PRUNE_THRESHOLD", os.environ.get("TAU2_SUMMARY_KL_PRUNE_THRESHOLD", "0.2")))
    min_high_tokens = int(os.environ.get("SUMMARY_KL_PRUNE_MIN_TOKENS", os.environ.get("TAU2_SUMMARY_KL_PRUNE_MIN_TOKENS", "2")))
    prune_strategy = os.environ.get("SUMMARY_KL_PRUNE_STRATEGY", os.environ.get("TAU2_SUMMARY_KL_PRUNE_STRATEGY", "kl")).strip().lower()

    response_ids, prior_ranges, current_ranges = _tokenize_segments(
        prior_assistant_raw_texts, current_pre_tool_text, model_name
    )
    if not response_ids:
        return None

    student_prompt_ids = _tokenize(system_prompt, model_name)
    teacher_prompt_ids = _tokenize(system_prompt + "\n\n[summary_oracle]\n" + summary_text, model_name)
    if len(student_prompt_ids) + len(response_ids) > max_length:
        budget = max_length - len(response_ids)
        student_prompt_ids = student_prompt_ids[-max(1, budget) :]
    if len(teacher_prompt_ids) + len(response_ids) > max_length:
        budget = max_length - len(response_ids)
        teacher_prompt_ids = teacher_prompt_ids[-max(1, budget) :]

    n = len(response_ids)
    student_pos = _score_prompt_logprobs(student_prompt_ids + response_ids, n, model_name, top_k)
    teacher_pos = _score_prompt_logprobs(teacher_prompt_ids + response_ids, n, model_name, top_k)
    per_token_kl = _reverse_kl(student_pos, teacher_pos)
    if not per_token_kl:
        return None
    k_min = min(per_token_kl)
    k_max = max(per_token_kl)
    if k_max - k_min > 1e-12:
        norm = [(x - k_min) / (k_max - k_min) for x in per_token_kl]
    else:
        norm = [0.0 for _ in per_token_kl]

    kl_prune_indices = []
    step_decisions = []
    prior_nonempty = [i for i, text in enumerate(prior_assistant_raw_texts) if text and text.strip()]
    for local_i, (start, end) in enumerate(prior_ranges):
        end = min(end, len(norm))
        segment = norm[start:end]
        n_high = sum(1 for x in segment if x > threshold)
        keep = n_high >= min_high_tokens
        global_i = prior_nonempty[local_i] if local_i < len(prior_nonempty) else local_i
        decision = {
            "step": "prior_assistant",
            "local_index": local_i,
            "global_index": global_i,
            "token_start": start,
            "token_end": end,
            "n_tokens": max(0, end - start),
            "n_tokens_above_threshold": n_high,
            "mean_normalized_kl": sum(segment) / len(segment) if segment else 0.0,
            "max_normalized_kl": max(segment) if segment else 0.0,
            "keep": keep,
        }
        step_decisions.append(decision)
        if not keep:
            kl_prune_indices.append(global_i)

    if prune_strategy in {"kl", "", "v5", "kl_v5", "kl-v5"}:
        prune_indices = kl_prune_indices
    elif prune_strategy in {"random_matched", "random-matched", "random"}:
        prune_indices = _random_matched_prune_indices(
            candidates=prior_nonempty,
            count=len(kl_prune_indices),
            task_id=task_id,
            summary_text=summary_text,
            prior_assistant_raw_texts=prior_assistant_raw_texts,
        )
    else:
        raise RuntimeError(f"Unknown SUMMARY_KL_PRUNE_STRATEGY={prune_strategy!r}")

    for start, end in current_ranges:
        segment = norm[start : min(end, len(norm))]
        step_decisions.append(
            {
                "step": "current_pre_tool",
                "n_tokens": len(segment),
                "n_tokens_above_threshold": sum(1 for x in segment if x > threshold),
                "mean_normalized_kl": sum(segment) / len(segment) if segment else 0.0,
                "max_normalized_kl": max(segment) if segment else 0.0,
                "keep": True,
            }
        )

    out_root = Path(output_dir or os.environ.get("SUMMARY_KL_OUTPUT_DIR", os.environ.get("TAU2_SUMMARY_KL_OUTPUT_DIR", "./summary_kl_logs")))
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"summary_kl_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.json"
    payload = {
        "summary": {
            "task_id": task_id,
            "model": model_name,
            "summary_text": summary_text,
            "n_prior_assistant_turns": len(prior_assistant_raw_texts),
            "n_response_tokens": len(per_token_kl),
            "reverse_kl_mean": sum(per_token_kl) / len(per_token_kl),
            "reverse_kl_min": k_min,
            "reverse_kl_max": k_max,
            "reverse_kl_sum": sum(per_token_kl),
            "prune_threshold": threshold,
            "prune_min_high_tokens": min_high_tokens,
            "prune_strategy": prune_strategy or "kl",
            "kl_prune_indices": kl_prune_indices,
            "prune_indices": prune_indices,
        },
        "step_decisions": step_decisions,
        "prior_assistant_raw_texts": prior_assistant_raw_texts,
        "tokens": [
            {
                "resp_token_index": i,
                "per_token_reverse_kl": float(v),
                "per_token_reverse_kl_normalized": float(norm[i]),
            }
            for i, v in enumerate(per_token_kl)
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "out_path": str(out_path),
        "prune_indices": prune_indices,
        "step_decisions": step_decisions,
        "n_response_tokens": len(per_token_kl),
        "reverse_kl_mean": payload["summary"]["reverse_kl_mean"],
    }
