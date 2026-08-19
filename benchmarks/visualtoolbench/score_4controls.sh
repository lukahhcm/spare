#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRED_ROOT="${PRED_ROOT:?Set PRED_ROOT to the output of run_4controls.sh}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:?Set JUDGE_BASE_URL}"
JUDGE_MODEL="${JUDGE_MODEL:?Set JUDGE_MODEL}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
SCORE_ROOT="${SCORE_ROOT:-$PRED_ROOT/rubric_scores}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$SCORE_ROOT"
"$PYTHON_BIN" "$BUNDLE_DIR/scripts/score_rubric_partial.py" \
  --anchors "$BUNDLE_DIR/data/anchor_records_238.jsonl" \
  --method "direct_no_tool=$PRED_ROOT/direct-no-tool" \
  --method "tool_baseline=$PRED_ROOT/tool-baseline" \
  --method "delete_reasoning=$PRED_ROOT/delete-reasoning" \
  --method "visual_evidence=$PRED_ROOT/visual-evidence" \
  --out "$SCORE_ROOT/per_example.jsonl" \
  --summary-out "$SCORE_ROOT/summary.json" \
  --base-url "$JUDGE_BASE_URL" \
  --api-key "$JUDGE_API_KEY" \
  --model "$JUDGE_MODEL" \
  --workers "$JUDGE_WORKERS"
