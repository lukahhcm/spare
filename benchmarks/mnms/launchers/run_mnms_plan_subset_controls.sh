#!/usr/bin/env bash
# Queue the three remaining MNMS controls for one model on the fixed user-ID 228 subset.
# Usage: run_mnms_plan_subset_controls.sh MODEL_TAG PORT SERVED_MODEL JOBS WAIT_PID
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
ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/${MODEL_TAG}/user228"
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

# Do not add load until this model's already-running baseline/V5 queue exits.
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 10
done

until curl --noproxy '*' -fsS -o /dev/null --max-time 3 "http://127.0.0.1:${PORT}/health"; do
  sleep 5
done

# The already-running pair launcher predates the pruning-report hook. Backfill
# its two reasoning-only reports before starting the additional controls.
for existing_mode in full_trace v5; do
  existing_out="$ROOT/$existing_mode"
  if [[ -s "$existing_out/predictions.jsonl" && ! -s "$existing_out/pruning_metrics.json" ]]; then
    "$PY" "$ANALYZE" \
      --predictions "$existing_out/predictions.jsonl" \
      --output-json "$existing_out/pruning_metrics.json" \
      --per-sample-jsonl "$existing_out/pruning_per_sample.jsonl" \
      --base-url "$BASE_URL" --model "$SERVED_MODEL" --method "$existing_mode" \
      >"$existing_out/pruning_eval.log" 2>&1
  fi
done

run_control() {
  local mode="$1"
  shift
  local out="$ROOT/$mode"
  mkdir -p "$out/results"

  local -a pids=()
  for shard in $(seq 0 $((JOBS - 1))); do
    mkdir -p "$out/results/shard_${shard}"
    env "$@" "$PY" -m mnms.run_reasoning_tool_rollout \
      --input-file "$DATASET" \
      --output-file "$out/predictions_shard_${shard}.jsonl" \
      --result-folder "$out/results/shard_${shard}" \
      --model "$SERVED_MODEL" --base-url "$BASE_URL" --api-key EMPTY \
      --max-steps 5 --timeout 60 --max-tokens 8192 --temperature 0 \
      --plan-only --resume --num-shards "$JOBS" --shard-index "$shard" \
      >"$out/run_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "One or more shards failed for ${MODEL_TAG}/${mode}" >&2
    return 1
  fi

  : >"$out/predictions.jsonl"
  for shard in $(seq 0 $((JOBS - 1))); do
    sed -n '1,$p' "$out/predictions_shard_${shard}.jsonl" >>"$out/predictions.jsonl"
  done
  "$PY" "$EVAL" \
    --gt-file "$DATASET" \
    --preds-file "$out/predictions.jsonl" \
    --output-csv "$out/metrics.csv" \
    --output-json "$out/metrics.json" \
    --per-sample-jsonl "$out/per_sample.jsonl" \
    >"$out/eval.log" 2>&1
  "$PY" "$ANALYZE" \
    --predictions "$out/predictions.jsonl" \
    --output-json "$out/pruning_metrics.json" \
    --per-sample-jsonl "$out/pruning_per_sample.jsonl" \
    --base-url "$BASE_URL" --model "$SERVED_MODEL" --method "$mode" \
    >"$out/pruning_eval.log" 2>&1
}

run_control no_tool_direct \
  MNMS_NO_TOOL_BASELINE=1 MNMS_STEP_REASONING_POLICY=full \
  MNMS_ENABLE_SUMMARY_TOOL=0 SUMMARY_KL_ENABLE=0 \
  MNMS_SUMMARY_RUNTIME_ENFORCE=0 MNMS_SUMMARY_AUTO_INSERT=0

run_control prune_all_reasoning \
  MNMS_NO_TOOL_BASELINE=0 MNMS_STEP_REASONING_POLICY=delete_all \
  MNMS_ENABLE_SUMMARY_TOOL=0 SUMMARY_KL_ENABLE=0 \
  MNMS_SUMMARY_RUNTIME_ENFORCE=0 MNMS_SUMMARY_AUTO_INSERT=0

run_control visual_only \
  MNMS_NO_TOOL_BASELINE=0 MNMS_STEP_REASONING_POLICY=visual_only \
  MNMS_ENABLE_SUMMARY_TOOL=0 SUMMARY_KL_ENABLE=0 \
  MNMS_SUMMARY_RUNTIME_ENFORCE=0 MNMS_SUMMARY_AUTO_INSERT=0
