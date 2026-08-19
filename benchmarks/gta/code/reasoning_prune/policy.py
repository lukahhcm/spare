"""Summary-conditioned KL pruning without imposing an agent protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Callable, Mapping, Protocol, Sequence


class LogprobScorer(Protocol):
    """Returns top-K log-probability maps for every token in ``continuation``."""

    def score(self, prompt: str, continuation: str) -> list[dict[str, float]]: ...


@dataclass
class ReasoningSegment:
    """A native assistant turn split by its host adapter.

    ``action_text`` must be the exact action/function-call payload emitted by
    the official engine.  The pruner treats it as opaque text.
    """

    segment_id: str
    reasoning_text: str
    action_text: str
    native_index: int
    metadata: dict[str, object] = field(default_factory=dict)
    pruned: bool = False


@dataclass(frozen=True)
class PruneConfig:
    top_logprobs: int = 20
    threshold: float = 0.2
    min_high_tokens: int = 2
    logprob_floor: float = -30.0
    evidence_mode: str = "v2"  # v2: action only, v3: structured visual evidence + action

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.min_high_tokens < 1:
            raise ValueError("min_high_tokens must be positive")
        if self.evidence_mode not in {"v2", "v3"}:
            raise ValueError("evidence_mode must be 'v2' or 'v3'")


@dataclass
class PruneEvent:
    summary: str
    decisions: list[dict[str, object]]
    skipped_reason: str | None = None


def _reverse_kl(student: Mapping[str, float], teacher: Mapping[str, float], floor: float) -> float:
    value = 0.0
    for token, student_lp in student.items():
        p = min(1.0, math.exp(float(student_lp)))
        value += p * (float(student_lp) - float(teacher.get(token, floor)))
    return max(0.0, value)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


_EVIDENCE = re.compile(
    r"\b(coordinate|bbox|bounding box|pixel|crop|ocr|path|area|count|number|"
    r"red|blue|green|yellow|left|right|top|bottom|above|below|center|text|"
    r"candidate|option|edge|texture)\b|\d",
    re.IGNORECASE,
)
_PLANNING = re.compile(
    r"\b(i need to|i should|let me|now i|next step|the question asks|"
    r"use the tool|call the|i will)\b",
    re.IGNORECASE,
)
_IMPERATIVE_PLAN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:add|calculate|compare|determine|divide|find|"
    r"identify|inspect|look|multiply|select|use)\b",
    re.IGNORECASE,
)
_ENUMERATOR_ONLY = re.compile(r"^\s*(?:[-*]\s*)?\d+[.)]?\s*$")


def structured_evidence(
    text: str,
    limit: int | None = None,
    max_chars: int | None = None,
) -> str:
    """V3 evidence retention, copied conceptually from PRUNE_METHODS.md.

    It is intentionally format-agnostic: evidence is only prepended to the
    host's original action payload and never parsed as a new action.
    """
    keep: list[str] = []
    size = 0
    for sentence in _sentences(text):
        if (len(sentence) > 260
                or _PLANNING.search(sentence)
                or _IMPERATIVE_PLAN.search(sentence)
                or _ENUMERATOR_ONLY.fullmatch(sentence)
                or not _EVIDENCE.search(sentence)):
            continue
        if max_chars is not None and size + len(sentence) > max_chars:
            break
        keep.append(sentence)
        size += len(sentence)
        if limit is not None and len(keep) == limit:
            break
    return "Structured evidence retained:\n" + "\n".join(f"- {line}" for line in keep) if keep else ""


class SummaryKLPruner:
    """Applies the documented reverse-KL decision to native turn segments."""

    def __init__(self, scorer: LogprobScorer, config: PruneConfig | None = None):
        self.scorer = scorer
        self.config = config or PruneConfig()

    def prune(
        self,
        segments: Sequence[ReasoningSegment],
        summary: str,
        student_prompt: str,
        teacher_prompt: str,
    ) -> PruneEvent:
        candidates = [segment for segment in segments if not segment.pruned and segment.reasoning_text.strip()]
        if not candidates:
            return PruneEvent(summary=summary, decisions=[], skipped_reason="no_unpruned_reasoning")
        continuation = "\n\n".join(segment.reasoning_text for segment in candidates)
        try:
            student = self.scorer.score(student_prompt, continuation)
            teacher = self.scorer.score(teacher_prompt, continuation)
        except Exception as exc:  # safe fail: baseline-compatible history is retained
            return PruneEvent(summary=summary, decisions=[], skipped_reason=f"logprob_unavailable:{type(exc).__name__}")
        if len(student) != len(teacher) or not student:
            return PruneEvent(summary=summary, decisions=[], skipped_reason="incompatible_logprob_positions")

        raw = [_reverse_kl(s, t, self.config.logprob_floor) for s, t in zip(student, teacher)]
        low, high = min(raw), max(raw)
        normalized = [0.0] * len(raw) if high == low else [(value - low) / (high - low) for value in raw]
        # The scorer must expose continuation token counts per segment.  In
        # absence of tokenizer access, adapters provide an explicit map; a
        # character-proportional map would silently make the decision invalid.
        boundaries = self._boundaries(candidates, len(normalized))
        if boundaries is None:
            return PruneEvent(summary=summary, decisions=[], skipped_reason="missing_token_boundaries")

        decisions: list[dict[str, object]] = []
        for segment, (start, end) in zip(candidates, boundaries):
            values = normalized[start:end]
            high_count = sum(value > self.config.threshold for value in values)
            keep = high_count >= self.config.min_high_tokens
            replacement = "\n\n".join(part for part in (segment.reasoning_text, segment.action_text) if part)
            if not keep:
                segment.pruned = True
                evidence = structured_evidence(segment.reasoning_text) if self.config.evidence_mode == "v3" else ""
                replacement = "\n\n".join(part for part in (evidence, segment.action_text) if part)
            decisions.append({
                "segment_id": segment.segment_id,
                "native_index": segment.native_index,
                "keep": keep,
                "high_kl_tokens": high_count,
                "token_count": len(values),
                "replacement": replacement,
            })
        return PruneEvent(summary=summary, decisions=decisions)

    @staticmethod
    def _boundaries(segments: Sequence[ReasoningSegment], total: int) -> list[tuple[int, int]] | None:
        counts: list[int] = []
        for segment in segments:
            value = segment.metadata.get("reasoning_token_count")
            if not isinstance(value, int) or value < 0:
                return None
            counts.append(value)
        if sum(counts) != total:
            return None
        cursor = 0
        boundaries: list[tuple[int, int]] = []
        for count in counts:
            boundaries.append((cursor, cursor + count))
            cursor += count
        return boundaries
