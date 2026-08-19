"""Exact continuation log-prob scorer for vLLM/OpenAI-compatible servers."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener


class OpenAICompletionScorer:
    """Uses `/completions` echo mode; no local weights or tokenizer fallback."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", top_logprobs: int = 20, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.top_logprobs = top_logprobs
        self.timeout = timeout

    def score(self, prompt: str, continuation: str) -> list[dict[str, float]]:
        payload = {
            "model": self.model,
            "prompt": prompt + continuation,
            "max_tokens": 0,
            "echo": True,
            "logprobs": self.top_logprobs,
            "temperature": 0,
        }
        request = Request(
            self.base_url + "/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        # Cluster shells often export an HTTP proxy which turns localhost into a
        # 503.  Model endpoints are intentionally contacted directly.
        with build_opener(ProxyHandler({})).open(request, timeout=self.timeout) as response:
            body: dict[str, Any] = json.load(response)
        choice = body["choices"][0]
        logprobs = choice.get("logprobs") or {}
        tokens = logprobs.get("tokens")
        offsets = logprobs.get("text_offset")
        prompt_top = choice.get("prompt_logprobs")
        if not isinstance(tokens, list) or not isinstance(offsets, list) or not isinstance(prompt_top, list):
            raise RuntimeError("server did not return echo prompt logprobs")
        if not (len(tokens) == len(offsets) == len(prompt_top)):
            raise RuntimeError("incompatible echo logprob arrays")
        # Do not locate the suffix with ``text_offset``. Some vLLM/tokenizer
        # combinations include decoded BOS/control-token lengths in those
        # offsets, so comparing them with Python character counts leaks prompt
        # positions into the continuation and gives student/teacher different
        # lengths. Ask the serving tokenizer for the exact continuation count
        # and take that many echo-logprob positions from the tail.
        tokenize_request = Request(
            self.base_url.removesuffix("/v1") + "/tokenize",
            data=json.dumps({
                "model": self.model,
                "prompt": continuation,
                "add_special_tokens": False,
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with build_opener(ProxyHandler({})).open(
                tokenize_request, timeout=self.timeout) as response:
            tokenized: dict[str, Any] = json.load(response)
        count = len(tokenized.get("tokens") or tokenized.get("input_ids") or [])
        decoded_positions = []
        for position in prompt_top:
            decoded = {}
            for value in (position or {}).values():
                if (isinstance(value, dict)
                        and "decoded_token" in value
                        and "logprob" in value):
                    decoded[str(value["decoded_token"])] = float(value["logprob"])
            if decoded:
                decoded_positions.append(decoded)
        maps = decoded_positions[-count:] if count else []
        if not maps:
            raise RuntimeError("cannot locate continuation prompt-logprob positions")
        return maps
