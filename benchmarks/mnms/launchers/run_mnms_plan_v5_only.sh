#!/usr/bin/env bash
# Run only V5 on the fixed MNMS user-ID 228 subset.
# Usage: run_mnms_plan_v5_only.sh MODEL_TAG PORT SERVED_MODEL JOBS WAIT_PID
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 MODEL_TAG PORT SERVED_MODEL JOBS WAIT_PID" >&2
  exit 2
fi

MODEL_TAG="$1"
PORT="$2"
SERVED_MODEL="$3"
JOBS="$4"
WAIT_PID="$5"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
CODE="$PROJECT_ROOT/code/mtest_mnms"
EVAL="$PROJECT_ROOT/code/benchmark_runs_code/eval_mnms_plan_local_nondep.py"
ANALYZE="$PROJECT_ROOT/launchers/analyze_mnms_reasoning_pruning.py"
DATASET="$PROJECT_ROOT/data/plan_only_user_subset_20260728/dataset.jsonl"
OUT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/${MODEL_TAG}/user228/v5"
BASE_URL="http://127.0.0.1:${PORT}/v1"

export PYTHONPATH="$CODE"
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY="EMPTY"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MNMS_USE_HEAVY_MODELS=0
export MNMS_ALLOW_NETWORK_TOOLS=0
export MNMS_MAX_PROTOCOL_REJECTIONS=5

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 10
done
# Also cover an orphaned last shard if its queue parent exited first.
while pgrep -f "plan_only_subset_comparison_20260728/${MODEL_TAG}/user228/(full_trace|no_tool_direct|prune_all_reasoning|visual_only)/" >/dev/null; do
  sleep 10
done
until curl --noproxy '*' -fsS -o /dev/null --max-time 3 "http://127.0.0.1:${PORT}/health"; do
  sleep 5
done

mkdir -p "$OUT/results"
pids=()
for shard in $(seq 0 $((JOBS - 1))); do
  mkdir -p "$OUT/results/shard_${shard}"
  env \
    MNMS_NO_TOOL_BASELINE=0 \
    MNMS_STEP_REASONING_POLICY=full \
    MNMS_ENABLE_SUMMARY_TOOL=1 \
    MNMS_SUMMARY_MODEL_VISIBLE=0 \
    SUMMARY_KL_ENABLE=1 \
    SUMMARY_KL_PRUNE_STRATEGY=v5 \
    SUMMARY_KL_PRUNE_THRESHOLD=0.2 \
    SUMMARY_KL_PRUNE_MIN_TOKENS=2 \
    SUMMARY_KL_APPLY_PRUNE=1 \
    SUMMARY_KL_MAX_LENGTH=8192 \
    MNMS_SUMMARY_RUNTIME_ENFORCE=1 \
    MNMS_SUMMARY_AUTO_INSERT=1 \
    MNMS_SUMMARY_FORCE_FIRST_AFTER=2 \
    MNMS_SUMMARY_FORCE_REPEAT_AFTER=0 \
    MNMS_SUMMARY_GATE_ENABLE=1 \
    MNMS_SUMMARY_GATE_MIN_REAL_TOOLS=2 \
    MNMS_SUMMARY_GATE_MIN_REASONING_TOKENS=1 \
    MNMS_SUMMARY_GATE_MIN_CANDIDATES=2 \
    MNMS_SUMMARY_GATE_DISCARD_NO_PRUNE=0 \
    MNMS_SUMMARY_TRACE_HIDDEN=1 \
    "$PY" -m mnms.run_reasoning_tool_rollout \
      --input-file "$DATASET" \
      --output-file "$OUT/predictions_shard_${shard}.jsonl" \
      --result-folder "$OUT/results/shard_${shard}" \
      --model "$SERVED_MODEL" --base-url "$BASE_URL" --api-key EMPTY \
      --max-steps 5 --timeout 60 --max-tokens 8192 --temperature 0 \
      --plan-only --resume --num-shards "$JOBS" --shard-index "$shard" \
      >"$OUT/run_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "One or more V5 shards failed for ${MODEL_TAG}" >&2
  exit 1
fi

: >"$OUT/predictions.jsonl"
for shard in $(seq 0 $((JOBS - 1))); do
  sed -n '1,$p' "$OUT/predictions_shard_${shard}.jsonl" >>"$OUT/predictions.jsonl"
done
"$PY" "$EVAL" \
  --gt-file "$DATASET" --preds-file "$OUT/predictions.jsonl" \
  --output-csv "$OUT/metrics.csv" --output-json "$OUT/metrics.json" \
  --per-sample-jsonl "$OUT/per_sample.jsonl" >"$OUT/eval.log" 2>&1
"$PY" "$ANALYZE" \
  --predictions "$OUT/predictions.jsonl" \
  --output-json "$OUT/pruning_metrics.json" \
  --per-sample-jsonl "$OUT/pruning_per_sample.jsonl" \
  --base-url "$BASE_URL" --model "$SERVED_MODEL" --method v5 \
  >"$OUT/pruning_eval.log" 2>&1
