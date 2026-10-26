# Method definitions

SPARE (Selective Pruning of Accumulated Reasoning with Visual Evidence
Preservation) manages reasoning history inside a multimodal agent's native tool
loop. The paper's main evaluation uses GTA, V*, BLINK-Jigsaw, and m&m's in a
planning-only setting. VisualToolBench is an additional released adapter.

## Conditions

| Condition | History policy |
| --- | --- |
| Direct answer | Answer without invoking tools |
| Full Trace | Retain complete reasoning, actions, and observations |
| Delete all reasoning | Remove completed reasoning while preserving tools and observations |
| Visual evidence-only | Extract compact visual evidence without KL selection |
| SPARE | Select segments using summary-conditioned residual sensitivity, then extract evidence from selected segments |

## Summary-conditioned replay

When the agent invokes `summarize_the_task`, SPARE creates a compact task-state
summary. The same model replays the same historical reasoning prefix under a
common context, with and without the summary. The summary serves as temporary
context for diagnosis and the next reasoning step. It is not stored in the
persistent trajectory.

## KL decision

The paper uses truncated reverse KL in the student-to-summary-conditioned
direction on the student's top-20 vocabulary support. Missing teacher log
probabilities use a fallback of −20. Scores are min-max normalized within each
pruning event with epsilon 1e-12. Constant scores normalize to zero.

A segment is selected for evidence extraction when at least `kappa=2` tokens
have normalized scores above `tau=0.2`. Low-sensitivity segments are removed.
This truncated score is a residual-sensitivity heuristic. It does not establish
semantic coverage or inherit every property of full-distribution KL.

## Evidence extraction and persistent context

Programmatic rules extract visual evidence from selected high-sensitivity
segments, retaining at most four items and 600 characters per segment. This is
not an additional LLM rewriting step. If no evidence is extracted, the segment
is removed. Original tool calls, tool observations, and images remain unchanged.
The next generated reasoning and action are recorded normally, without retaining
the temporary summary.

## Pruning ratio

Pruning is measured on each method's own generated trajectory:

```text
100 * sum(original_reasoning_tokens - retained_reasoning_tokens)
    / sum(original_reasoning_tokens)
```

Retained reasoning includes extracted evidence. Temporary summaries, tool calls,
observations, images, and final answers are excluded. Counts are aggregated
within each benchmark. The main table reports sample-weighted averages across
benchmarks, rather than an unweighted macro-average.

See the [paper](assets/paper.pdf) for the complete procedure and
[results](RESULTS.md) for the evaluation scopes.
