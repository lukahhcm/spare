#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${DATA_PATH:-$BUNDLE_DIR/data/visualtoolbench_eval_1204.parquet}"
MANIFEST_PATH="${MANIFEST_PATH:-$BUNDLE_DIR/data/manifest_238.json}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT}"
BASE_URL="${BASE_URL:?Set BASE_URL, for example http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the served model name}"
API_KEY="${API_KEY:-EMPTY}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_TOOL_STEPS="${MAX_TOOL_STEPS:-3}"
TIMEOUT="${TIMEOUT:-240}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUT_ROOT/logs"
methods=(direct-no-tool tool-baseline delete-reasoning visual-evidence)
for method in "${methods[@]}"; do
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    out="$OUT_ROOT/$method/shards/shard_$shard"
    mkdir -p "$out"
    env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
      "$PYTHON_BIN" "$BUNDLE_DIR/scripts/run_visualtoolbench.py" \
      --data "$DATA_PATH" \
      --eligible-manifest "$MANIFEST_PATH" \
      --out-dir "$out" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      --base-url "$BASE_URL" \
      --api-key "$API_KEY" \
      --model "$MODEL_NAME" \
      --temperature 0 \
      --top-p 1 \
      --max-tokens "$MAX_TOKENS" \
      --timeout "$TIMEOUT" \
      --max-retries 1 \
      --tool-call-transport text \
      --tool-choice-mode required-first \
      --max-tool-steps "$MAX_TOOL_STEPS" \
      --comparison-group "$method" \
      > "$OUT_ROOT/logs/${method}_shard_$shard.log" 2>&1 &
  done
  wait
done
