#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CASES="${CASES:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen/cases.jsonl}"
SFT_RUN_ID="${SFT_RUN_ID:-sft-r4-all-full-seed42-v1}"
DPO_RUN_ID="${DPO_RUN_ID:-dpo-v1p4-rollout-quality-screen800-beta0p1-seed42-from-r4-v1}"
EVAL_SUFFIX="${EVAL_SUFFIX:-}"
ANALYSIS_SUFFIX="${ANALYSIS_SUFFIX:-$EVAL_SUFFIX}"
ROOT="experiments/local/1p5b"
OUTPUT_ROOT="$ROOT/$DPO_RUN_ID/analysis${ANALYSIS_SUFFIX}"
SFT_RESULTS="$ROOT/$SFT_RUN_ID/screen${EVAL_SUFFIX}_eval/per_sample.jsonl"
STEP10="$ROOT/$DPO_RUN_ID/screen_checkpoint10${EVAL_SUFFIX}_eval/per_sample.jsonl"
STEP25="$ROOT/$DPO_RUN_ID/screen_checkpoint25${EVAL_SUFFIX}_eval/per_sample.jsonl"
STEP45="$ROOT/$DPO_RUN_ID/screen_checkpoint45${EVAL_SUFFIX}_eval/per_sample.jsonl"

for path in "$CASES" "$SFT_RESULTS" "$STEP10" "$STEP25" "$STEP45"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required result: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite immutable analysis: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" scripts/ecommerce/compare_1p5b_screen_runs.py \
  --cases "$CASES" \
  --run "sft=$SFT_RESULTS" \
  --run "dpo_step10=$STEP10" \
  --run "dpo_step25=$STEP25" \
  --run "dpo_step45=$STEP45" \
  --seed 20260809 \
  --resamples 10000 \
  --output "$OUTPUT_ROOT/paired_comparison.json"

"$PYTHON_BIN" scripts/ecommerce/analyze_1p5b_dpo_failures.py \
  --cases "$CASES" \
  --baseline "$SFT_RESULTS" \
  --run "dpo_step10=$STEP10" \
  --run "dpo_step25=$STEP25" \
  --run "dpo_step45=$STEP45" \
  --output "$OUTPUT_ROOT/failure_transitions.json"

echo "DPO v1.4 screen analysis completed: $OUTPUT_ROOT"
