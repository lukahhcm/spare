# VisualToolBench adapter

This directory contains the VisualToolBench multi-round rollout, local image
tool, SPARE summary/KL policy, pruning analysis, and fixed-rubric scorer.

The dataset is available from
[xi1ngang/VisualToolBench](https://github.com/xi1ngang/VisualToolBench). Dataset
files and generated manifests belong in `data/`, which is ignored by Git. You
can also set `DATA_PATH` and `MANIFEST_PATH` to any local location.

Run the four controls with `run_4controls.sh`, the paper method with
`run_spare.sh`, and all five methods through the matched rubric scorer with
`score_all.sh`. The
older hidden-controller experiment remains available as `run_v5.sh`. All launchers accept
`PYTHON_BIN` and OpenAI-compatible endpoint settings through environment
variables. The repository root README defines the shared five-method protocol.
