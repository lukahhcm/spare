# Method definitions

All comparisons use the benchmark's native prompt, tool loop, parser, and
scorer. Only the reasoning-history policy changes.

## Conditions

`direct-no-tool` disables tools and asks for the final answer under the same
answer-format contract used by the agentic runs.

`tool-baseline` keeps the complete reasoning, action, and observation history.

`delete-reasoning` clears every completed assistant reasoning span. Tool calls,
tool observations, and image inputs remain unchanged.

`visual-evidence` replaces each completed reasoning span with compact,
structured visual facts. It does not use KL selection.

`SPARE` forms a compact task-state summary after the adaptive gate is met. The
same model scores the candidate history under the original and
summary-conditioned contexts. It drops low-residual-KL reasoning and preserves
high-residual-KL content as structured visual evidence.

## KL decision

The default configuration uses the student's top-20 vocabulary support. At
each replay position, SPARE computes reverse KL from the student distribution
to the summary-conditioned distribution, then min-max normalizes token scores
within the pruning event. A segment is evidence-critical when at least
`kappa=2` tokens exceed `tau=0.2`.

## Persistent context

The summary is privileged diagnostic state. It does not replace the original
trajectory. If a probe commits no pruning, it is discarded. When pruning is
accepted, the next decision can receive the compact state once, while the
persistent history stores the reconstructed reasoning, original actions, tool
observations, and images.

## Pruning ratio

The reported pruning ratio is computed over reasoning-side history tokens:

```text
100 * sum(original_reasoning_tokens - retained_reasoning_tokens)
    / sum(original_reasoning_tokens)
```

Action blocks, tool observations, and image tokens are excluded because SPARE
does not modify them. Each reasoning segment is counted once over its lifecycle.
