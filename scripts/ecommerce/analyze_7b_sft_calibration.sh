#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CASES="${CASES:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen/cases.jsonl}"
RUN_SUFFIX="${RUN_SUFFIX:-v1}"
SEED="${SEED:-42}"
ROOT="experiments/local/7b"
INITIAL="$ROOT/initial-screen-${RUN_SUFFIX}/screen_eval/per_sample.jsonl"
R8="$ROOT/sft-r8-all-cal100-seed${SEED}-${RUN_SUFFIX}/calibration_screen_eval/per_sample.jsonl"
R16="$ROOT/sft-r16-all-cal100-seed${SEED}-${RUN_SUFFIX}/calibration_screen_eval/per_sample.jsonl"
OUTPUT_ROOT="$ROOT/sft-calibration-analysis-seed${SEED}-${RUN_SUFFIX}"

for path in "$CASES" "$INITIAL" "$R8" "$R16"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required calibration result: $path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite immutable calibration analysis: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

"$PYTHON_BIN" scripts/ecommerce/compare_1p5b_screen_runs.py \
  --cases "$CASES" \
  --run "initial=$INITIAL" \
  --run "sft_r8=$R8" \
  --run "sft_r16=$R16" \
  --seed 20260809 \
  --resamples 10000 \
  --output "$OUTPUT_ROOT/paired_comparison.json"

"$PYTHON_BIN" scripts/ecommerce/analyze_1p5b_dpo_failures.py \
  --cases "$CASES" \
  --baseline "$INITIAL" \
  --run "sft_r8=$R8" \
  --run "sft_r16=$R16" \
  --output "$OUTPUT_ROOT/failure_transitions_vs_initial.json"

echo "7B SFT calibration analysis completed: $OUTPUT_ROOT"
