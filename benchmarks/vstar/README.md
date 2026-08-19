# V*Bench

This directory contains the benchmark-native V*Bench evaluator, multi-round
tool loop, KL-Visual V5 policy, and the Point-fixed Molmo worker.

## Download the data

The runner loads [`lmms-lab/vstar-bench`](https://huggingface.co/datasets/lmms-lab/vstar-bench),
split `test`. For parallel runs, materialize the 191 items once:

```bash
cd code/runner/scripts
python materialize_vstar_manifest.py --task vstar --out-dir ../../../data/vstar
```

This creates `data/vstar/vstar.jsonl` and `data/vstar/images/`. Both are ignored
by Git.

## Start the model and tools

Serve the evaluated vision model through an OpenAI-compatible endpoint. For
example, with vLLM:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b \
  --host 127.0.0.1 --port 8011 \
  --max-model-len 32768 --max-logprobs 20 --enforce-eager
```

Start the controller and official Molmo Point worker from `code/runner/`:

```bash
python -m tool_server.tool_workers.online_workers.controller \
  --host 127.0.0.1 --port 20001

CUDA_VISIBLE_DEVICES=0 python -m \
  tool_server.tool_workers.online_workers.molmo_point_worker \
  --host 127.0.0.1 --port 20027 \
  --controller_addr http://127.0.0.1:20001 \
  --model_path allenai/Molmo-7B-D-0924 \
  --model_name Point --max_concurrency 4
```

The verified lightweight VStar profile uses real `Point`, `OCR`, and local
`Crop`. Start the included OCR worker when evaluating that profile:

```bash
CUDA_VISIBLE_DEVICES=1 python -m \
  tool_server.tool_workers.online_workers.ocr_worker \
  --host 127.0.0.1 --port 20009 \
  --controller-addr http://127.0.0.1:20001 \
  --model-name OCR --gpu-ids 0 --max-concurrency 4
```

## Run five methods

```bash
BASE_URL=http://127.0.0.1:8011/v1 \
MODEL=qwen3-vl-8b \
CONTROLLER_ADDR=http://127.0.0.1:20001 \
ITEMS_MANIFEST="$PWD/data/vstar/vstar.jsonl" \
bash run_fiveway.sh
```

Set `LIMIT=2` for a smoke test. Formal runs use `LIMIT` unset, temperature 0,
8,192 output tokens, 10 maximum rounds, and no model-generated tool fallback.

## Test

```bash
cd code/runner
python -m unittest \
  tests.test_generated_image_cap \
  tests.test_tool_parameter_repair \
  tests.test_point_worker \
  tests.test_v5_pruning_policy -v
```
