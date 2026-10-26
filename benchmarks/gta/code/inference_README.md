# Native-engine test-time reasoning pruning

This directory contains the test-time pruning layer used by the benchmark
adapters.  It is deliberately **not** an agent framework: each benchmark keeps
its official prompt, tool schema, parser, execution loop, and evaluator.

`reasoning_prune` only receives an already-produced native assistant turn and,
after that benchmark's own summary/memory event, decides whether the prose
reasoning portion of older turns can be replaced.  The native action block is
kept byte-for-byte.  Thus an adapter never converts ToolSandbox function calls,
Qwen-Agent messages, OpenCompass actions, or AdaReasoner XML into a common
format.

## Layout

- `reasoning_prune/`: portable summary-KL policy, OpenAI/vLLM log-prob scorer,
  evidence retaining V3 replacement, and result writer.
- `adapters/`: one thin hook per native benchmark engine.
- `tests/`: policy and adapter contract tests; these need no model weights.
- `upstreams/`: shallow, unmodified official repositories used only as a
  reference for each native integration.

## Guarantees enforced by the hooks

1. A summary must be produced by the native benchmark; this package never
   invents a summary tool or forces a model to call one.
2. Only an assistant turn's separately captured `reasoning_text` may be
   replaced.  Its `action_text` is retained exactly.
3. If token log-probs are unavailable or malformed, pruning is skipped and the
   run records the reason.  It never substitutes a heuristic score.
4. Baseline and pruned runs use the same native runner configuration and write
   machine-readable JSONL plus aggregate summaries.

See `adapters/README.md` for the exact host-side insertion point for GTA,
ToolVQA, m&m's, ToolSandbox, AdaReasoner/ReasonMine, and
VTC-Bench.
