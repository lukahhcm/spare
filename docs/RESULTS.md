# Results

These results follow the updated [paper PDF](assets/paper.pdf).
The [arXiv record](https://arxiv.org/abs/2608.22963) provides the publication link.
All accuracy and pruning values below are percentages.

## Main evaluation

The main table evaluates four benchmarks. m&m's uses the planning-only setting.
Acc. and Pr. are sample-weighted averages across benchmarks. Pruning is measured
on each method's own generated reasoning, including retained evidence in the
remaining token count.

| Backbone | Method | GTA | V* | BLINK-Jigsaw | m&m's | Acc. | Pr. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-30B-A3B | Full Trace | 45.76 | 74.87 | 67.33 | 36.84 | 56.53 | 0.00 |
| | SPARE | 59.32 | 67.02 | 64.00 | 49.56 | 59.24 | 69.40 |
| Qwen3-VL-8B | Full Trace | 42.37 | 69.63 | 74.00 | 60.96 | 64.97 | 0.00 |
| | SPARE | 49.15 | 74.35 | 72.00 | 57.46 | 65.29 | 61.20 |
| Llama-3.1-Nemotron-Nano-VL-8B-V1 | Full Trace | 40.68 | 38.22 | 45.33 | 26.75 | 35.99 | 0.00 |
| | SPARE | 42.37 | 38.74 | 48.67 | 21.93 | 35.35 | 41.34 |

SPARE improves average accuracy on both Qwen backbones. Nemotron shows a small
accuracy decrease, from 35.99 to 35.35, while pruning 41.34% of reasoning tokens.

## Aggregate accuracy–pruning comparison

This comparison uses **three** benchmarks (GTA, BLINK-Jigsaw, and m&m's) across
all three backbones. Its scope differs from the four-benchmark table above.

| Method | Accuracy | Reasoning pruned |
| --- | ---: | ---: |
| Full Trace | 48.81 | 0.00 |
| Random-33% | 41.95 | 32.26 |
| Random-50% | 42.64 | 50.64 |
| KL-guided | 48.36 | 16.40 |
| Visual evidence-only | 42.87 | 75.62 |
| SPARE | 50.34 | 57.81 |

## Summarizer fine-tuning

The summarizer is fine-tuned on 376 VTC-Bench trajectories. On the out-of-domain
V* evaluation with Qwen3-VL-8B, SPARE achieves 74.35 accuracy with 70.20% pruning.
SFT + SPARE achieves 77.49 accuracy with 78.84% pruning under the same pruning
rule. Full Trace accuracy is 69.63.

## Controlled textual-interference diagnostic

The constructed GTA subset holds the question, images, tool calls, and tool
observations fixed while varying only retained reasoning history. All audited
cases are retained, including those in which pruning does not activate.
Random-1 results are averaged over random draws.

| Backbone | Method | Forced-choice accuracy ↑ | Stale-copy rate ↓ |
| --- | --- | ---: | ---: |
| Qwen3-VL-8B | Full Trace | 63.64 | 36.36 |
| | Random-1 | 72.73 | 27.27 |
| | SPARE | 90.91 | 9.09 |
| Qwen3-VL-30B-A3B | Full Trace | 54.55 | 45.45 |
| | Random-1 | 72.73 | 22.73 |
| | SPARE | 81.82 | 18.18 |

Stale-copy rate measures repetition of the earlier incorrect answer despite
later corrective evidence. Free-form answer accuracy remains unchanged between
Full Trace and SPARE. This diagnostic supports reduced reliance on conflicting
textual history, rather than a general improvement in task-solving ability.
See Appendix D.1 for the complete setup and definitions.

Generated trajectories and evaluation outputs are kept outside Git. See the
benchmark READMEs for launchers and scoring instructions.
