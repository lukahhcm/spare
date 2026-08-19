<div align="center">

# SPARE

### Buried in Textual Debt: Context Pruning with Visual Evidence Preservation for MLLM Agents

**Selective forgetting for multimodal tool-use agents.**

[![Project Page](https://img.shields.io/badge/Project-Page-ef7f45?style=for-the-badge)](https://lukahhcm.github.io/spare/)
[![Paper](https://img.shields.io/badge/Paper-PDF-4f7cff?style=for-the-badge)](https://lukahhcm.github.io/spare/assets/paper.pdf)
[![Pages](https://img.shields.io/github/actions/workflow/status/lukahhcm/spare/pages.yml?style=for-the-badge&label=Project%20Page)](https://github.com/lukahhcm/spare/actions/workflows/pages.yml)

<br>

<img src="docs/assets/spare-overview.png" alt="SPARE method overview" width="100%">

</div>

## TL;DR

Long multimodal-agent trajectories accumulate self-generated plans, descriptions,
and hypotheses. This **textual debt** can dominate the context and suppress newer
visual evidence. SPARE uses a compact task-state summary as privileged diagnostic
context, measures each historical segment's residual influence with
summary-conditioned reverse KL, and selectively removes redundant reasoning while
preserving images, tool calls, observations, and verifiable visual evidence.

> Across five visual tool-use benchmarks and three MLLM backbones, SPARE achieves
> the strongest macro-average accuracy among pruning methods while removing
> **37.89–64.58%** of persistent reasoning tokens.

## Contents

- [Method at a glance](#method-at-a-glance)
- [Results](#results)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Benchmark entry points](#benchmark-entry-points)
- [Reproducing matched comparisons](#reproducing-matched-comparisons)
- [Repository layout](#repository-layout)
- [Citation](#citation)

## Method at a glance

SPARE runs inside the native multi-round tool loop:

1. **Adaptive summary.** The agent summarizes the question, completed tools,
   grounded evidence, and remaining uncertainty only when context management is
   needed.
2. **Summary-conditioned replay.** The same backbone replays candidate reasoning
   with and without the task-state summary. No external verifier or reward model
   is required.
3. **Residual-sensitivity scoring.** Top-K reverse KL estimates which tokens still
   contain information beyond the compact task state.
4. **Evidence-preserving reconstruction.** Summary-covered reasoning is removed;
   evidence-critical content is retained as compact OCR strings, coordinates,
   crop identifiers, values, or candidate relations.

The original action blocks, tool observations, and image inputs are never pruned.

### Paper configuration

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `top_k` | `20` | Student vocabulary support used for reverse KL |
| `tau` | `0.2` | Normalized token-level residual-sensitivity threshold |
| `kappa` | `2` | Minimum number of high-sensitivity tokens needed to protect a segment |
| Trigger | model-selected | Short trajectories avoid unnecessary pruning |
| Decoder | greedy | Used for matched evaluation |

Exact policy semantics and token accounting are documented in
[`docs/METHODS.md`](docs/METHODS.md).

## Results

The main evaluation covers **GTA**, **V\***, **BLINK-Jigsaw**,
**VisualToolBench**, and **m&m's**.

| Backbone | Full Trace avg. acc. | SPARE avg. acc. | Reasoning pruned |
| --- | ---: | ---: | ---: |
| Qwen3-VL-30B-A3B-Instruct | 44.30 | **49.19** | 63.70% |
| Qwen3-VL-8B-Instruct | **54.27** | 53.47 | 37.89% |
| Llama-3.1-Nemotron-Nano-VL-8B-V1 | **30.10** | 29.97 | 64.58% |

SPARE provides the best average accuracy among the evaluated pruning methods.
The complete per-benchmark results, SFT study, ablations, controlled stale-context
diagnostic, and attention analysis are reported in the
[paper](https://lukahhcm.github.io/spare/assets/paper.pdf).

## Installation

### Requirements

- Linux with NVIDIA GPUs for model and vision-tool serving;
- Python 3.10 or newer;
- an OpenAI-compatible endpoint that can return top-K token log probabilities;
- the official datasets and assets for the benchmark you want to run.

Benchmark toolchains have different dependencies, so we recommend one virtual
environment per benchmark rather than a single monolithic environment.

```bash
git clone https://github.com/lukahhcm/spare.git
cd spare

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Example: VisualToolBench adapter
pip install -r benchmarks/visualtoolbench/requirements.txt
```

Dataset sources and expected local layouts are listed in
[`docs/DATA.md`](docs/DATA.md). Datasets, checkpoints, rollout traces, generated
images, logs, and credentials are intentionally excluded from Git.

## Quick start

VisualToolBench provides the most compact end-to-end entry point. First serve the
evaluated VLM and the KL/logprob endpoint, then export:

```bash
export DATA_PATH=/path/to/visualtoolbench_eval.parquet
export MANIFEST_PATH=/path/to/evaluation_manifest.json
export OUT_ROOT=/path/to/outputs/vtb

export BASE_URL=http://127.0.0.1:8000/v1
export MODEL_NAME=Qwen3-VL-8B-Instruct

export KL_BASE_URL=http://127.0.0.1:8001/v1
export KL_MODEL=Qwen3-VL-8B-Instruct
```

Run the four matched controls and SPARE:

```bash
bash benchmarks/visualtoolbench/run_4controls.sh
bash benchmarks/visualtoolbench/run_spare.sh
```

Score all conditions with the same manifest, rubric, and judge configuration:

```bash
export PRED_ROOT="$OUT_ROOT"
export ANCHORS_PATH=/path/to/anchor_records.jsonl
export JUDGE_BASE_URL=http://127.0.0.1:8012/v1
export JUDGE_MODEL=Qwen3-VL-30B-A3B-Instruct

bash benchmarks/visualtoolbench/score_all.sh
```

Use `LIMIT=2` where supported for a smoke test before launching a complete run.

## Benchmark entry points

Each adapter preserves its benchmark's native prompt, tools, parser, and scorer.
Only the reasoning-history policy changes.

| Benchmark | Data / setup | Primary launcher | What is included |
| --- | --- | --- | --- |
| [VisualToolBench](benchmarks/visualtoolbench/README.md) | [Official repo](https://github.com/xi1ngang/VisualToolBench) | [`run_spare.sh`](benchmarks/visualtoolbench/run_spare.sh) | Multi-round rollout, local image tools, KL pruning, rubric scoring |
| [GTA](benchmarks/gta/README.md) | [OpenCompass GTA](https://github.com/open-compass/GTA) | [`run_gta_8gpu.py`](benchmarks/gta/code/scripts/run_gta_8gpu.py) | OpenCompass adapter, sharding, fixed-set verification |
| [V*](benchmarks/vstar/README.md) | [vstar-bench](https://huggingface.co/datasets/lmms-lab/vstar-bench) | [`run_fiveway.sh`](benchmarks/vstar/run_fiveway.sh) | Visual search, Point/OCR/Crop workers, five matched methods |
| [BLINK-Jigsaw](benchmarks/jigsaw/README.md) | [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) | [`run_fiveway.sh`](benchmarks/jigsaw/run_fiveway.sh) | Jigsaw-native spatial and image-editing tools |
| [m&m's](benchmarks/mnms/README.md) | [mnms](https://huggingface.co/datasets/zixianma/mnms) | [`run_mnms_plan_v5_only.sh`](benchmarks/mnms/launchers/run_mnms_plan_v5_only.sh) | Multimodal plan execution, KL core, pruning analysis |

## Reproducing matched comparisons

The release implements five evaluation conditions:

| Condition | History policy |
| --- | --- |
| Direct answer | Answer without invoking tools |
| Full trace | Retain complete reasoning, actions, and observations |
| Delete all reasoning | Remove completed reasoning while preserving tools and observations |
| Visual evidence-only | Non-selectively rewrite reasoning as structured visual evidence |
| **SPARE** | Use summary-conditioned KL to select removal or evidence preservation |

For a valid comparison, keep the following fixed across conditions:

- model checkpoint, chat template, decoding parameters, and tool-round budget;
- benchmark manifest and available tool set;
- KL endpoint model, `top_k`, `tau`, and `kappa`;
- judge checkpoint, judge prompt, official parser, and scorer version.

The runners record predictions, tool traces, summary/KL events, latency, runtime
errors, and reasoning-token counts. Primary results should never use cached
answers or fabricated tool observations.

### Lightweight verification

Python modules and shell launchers can be checked without downloading datasets:

```bash
python -m compileall -q benchmarks
find benchmarks -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

Benchmark-specific regression commands are listed in the corresponding README.

## Repository layout

```text
spare/
├── benchmarks/
│   ├── visualtoolbench/   # rollout, SPARE policy, scoring, token analysis
│   ├── gta/               # GTA / OpenCompass integration and 8-GPU launcher
│   ├── vstar/             # V* visual search and Point/OCR tool workers
│   ├── jigsaw/            # BLINK-Jigsaw spatial tool loop
│   └── mnms/              # m&m's planning and execution
├── docs/
│   ├── index.html         # deployed project page
│   ├── DATA.md            # dataset sources and preparation
│   ├── METHODS.md         # policy definitions and pruning metric
│   └── RESULTS.md         # release-level result note
├── .github/workflows/     # GitHub Pages deployment
└── README.md
```

## Citation

If SPARE is useful for your work, please cite the paper. Author and venue fields
will be updated with the archival version.

```bibtex
@misc{spare2026,
  title = {Buried in Textual Debt: Context Pruning with Visual Evidence
           Preservation for MLLM Agents},
  year  = {2026}
}
```

Please also cite the original benchmark and framework releases used by the
adapter you run.
