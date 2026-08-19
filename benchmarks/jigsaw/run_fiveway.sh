#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_URL="${BASE_URL:?set BASE_URL to the OpenAI-compatible model endpoint}"
MODEL="${MODEL:?set MODEL to the served model name}"
ITEMS_MANIFEST="${ITEMS_MANIFEST:-$ROOT/data/jigsaw_blink/jigsaw_blink.jsonl}"
OUT="${OUT:-$ROOT/results/$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT="${LIMIT:-}"

if [[ ! -s "$ITEMS_MANIFEST" ]]; then
  echo "Missing JigsawBlink manifest: $ITEMS_MANIFEST" >&2
  echo "Create it with code/runner/scripts/materialize_vstar_manifest.py." >&2
  exit 2
fi

common=(
  --tasks jigsaw_blink
  --items-manifest "$ITEMS_MANIFEST"
  --tool-profile jigsaw
  --base-url "$BASE_URL"
  --model "$MODEL"
  --require-all-tools
  --max-rounds 10
  --min-tool-calls 1
  --max-tokens 8192
  --temperature 0
  --timeout 600
)
if [[ -n "$LIMIT" ]]; then
  common+=(--limit "$LIMIT")
fi

runner="$ROOT/code/runner/scripts/run_tool_benchmarks.py"
direct_runner="$ROOT/code/runner/scripts/run_no_tool_benchmarks.py"
tools="Crop,DetectBlackArea,AStarWithPixelCoordinate,Draw2DPath,InsertImage"
mkdir -p "$OUT"

direct_args=(
  --tasks jigsaw_blink --base-url "$BASE_URL" --model "$MODEL"
  --max-tokens 8192 --temperature 0 --timeout 600
  --out-dir "$OUT/no_tool_direct"
)
if [[ -n "$LIMIT" ]]; then
  direct_args+=(--limit "$LIMIT")
fi
"$PYTHON_BIN" "$direct_runner" "${direct_args[@]}"

BENCHMARK_REQUIRE_SUCCESSFUL_TOOL=1 "$PYTHON_BIN" "$runner" \
  "${common[@]}" --tools "$tools" --out-dir "$OUT/tool_baseline"

BENCHMARK_REQUIRE_SUCCESSFUL_TOOL=1 "$PYTHON_BIN" "$runner" \
  "${common[@]}" --tools "$tools" --delete-reasoning \
  --out-dir "$OUT/delete_reasoning"

BENCHMARK_REQUIRE_SUCCESSFUL_TOOL=1 "$PYTHON_BIN" "$runner" \
  "${common[@]}" --tools "$tools" --retain-structured-visual-evidence \
  --out-dir "$OUT/visual_evidence"

BENCHMARK_REQUIRE_SUCCESSFUL_TOOL=1 "$PYTHON_BIN" "$runner" \
  "${common[@]}" --tools "$tools,summarize_the_task" \
  --summary-mode visual_state --enable-summary-kl-pruning \
  --summary-kl-top-logprobs 20 --summary-kl-prune-threshold 0.8 \
  --summary-kl-prune-min-tokens 2 --summary-kl-max-response-chars 12000 \
  --v5-hidden-summary --v5-no-summary-context \
  --out-dir "$OUT/kl_visual_v5"
