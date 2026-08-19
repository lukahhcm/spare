#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${DATA_PATH:-$BUNDLE_DIR/data/visualtoolbench_eval_1204.parquet}"
MANIFEST_PATH="${MANIFEST_PATH:-$BUNDLE_DIR/data/manifest_238.json}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT}"
BASE_URL="${BASE_URL:?Set rollout BASE_URL}"
MODEL_NAME="${MODEL_NAME:?Set rollout MODEL_NAME}"
KL_BASE_URL="${KL_BASE_URL:?Set the logprob-capable KL scorer endpoint}"
KL_MODEL="${KL_MODEL:?Set the served KL scorer model name}"
API_KEY="${API_KEY:-EMPTY}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TIMEOUT="${TIMEOUT:-240}"
# `prune-v5` is the quality-first hidden-controller implementation.  It
# preserves baseline action formatting, commits only net-positive pruning, and
# keeps the two most recent reasoning segments intact.
V5_GROUP="${V5_GROUP:-prune-v5}"

mkdir -p "$OUT_ROOT/$V5_GROUP/logs"
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  out="$OUT_ROOT/$V5_GROUP/shards/shard_$shard"
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
    --kl-base-url "$KL_BASE_URL" \
    --kl-model "$KL_MODEL" \
    --temperature 0 \
    --top-p 1 \
    --max-tokens "$MAX_TOKENS" \
    --timeout "$TIMEOUT" \
    --max-retries 1 \
    --tool-call-transport text \
    --tool-choice-mode required-first \
    --max-tool-steps 3 \
    --comparison-group "$V5_GROUP" \
    > "$OUT_ROOT/$V5_GROUP/logs/shard_$shard.log" 2>&1 &
done
wait
