#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CASES="${CASES:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen/cases.jsonl}"
RUN_SUFFIX="${RUN_SUFFIX:-v1}"
ROOT="experiments/local/1p5b"
RESPONSE_RUN="$ROOT/dpo-v1p4-response-only-matched-n173-beta0p1-seed42-${RUN_SUFFIX}"
MULTI_RUN="$ROOT/dpo-v1p4-multigranularity-matched-n173-beta0p1-seed42-${RUN_SUFFIX}"
SFT="$ROOT/sft-r4-all-full-seed42-v1/screen_negation_v2_eval/per_sample.jsonl"
OUTPUT_ROOT="$ROOT/dpo-v1p4-composition-matched-n173-analysis-${RUN_SUFFIX}"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite immutable composition analysis: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

runs=("sft=$SFT")
for step in 5 10 20; do
  response="$RESPONSE_RUN/screen_checkpoint${step}_eval/per_sample.jsonl"
  multi="$MULTI_RUN/screen_checkpoint${step}_eval/per_sample.jsonl"
  for path in "$response" "$multi"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing required result: $path" >&2
      exit 2
    fi
  done
  runs+=("response_step${step}=$response" "multi_step${step}=$multi")
  "$PYTHON_BIN" scripts/ecommerce/analyze_1p5b_dpo_failures.py \
    --cases "$CASES" \
    --baseline "$response" \
    --run "multi_step${step}=$multi" \
    --output "$OUTPUT_ROOT/composition_transitions_step${step}.json"
done

command=("$PYTHON_BIN" scripts/ecommerce/compare_1p5b_screen_runs.py --cases "$CASES")
for run in "${runs[@]}"; do
  command+=(--run "$run")
done
command+=(--seed 20260809 --resamples 10000 --output "$OUTPUT_ROOT/paired_comparison.json")
"${command[@]}"

echo "DPO v1.4 composition ablation analysis completed: $OUTPUT_ROOT"
