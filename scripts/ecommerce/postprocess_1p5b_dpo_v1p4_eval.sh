#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CASES_ROOT="${CASES_ROOT:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen}"
SFT_RUN_ID="${SFT_RUN_ID:-sft-r4-all-full-seed42-v1}"
DPO_RUN_ID="${DPO_RUN_ID:-dpo-v1p4-rollout-quality-screen800-beta0p1-seed42-from-r4-v1}"
ROOT="experiments/local/1p5b"
DPO_ROOT="$ROOT/$DPO_RUN_ID"
EVAL_SUFFIX="_negation_v2"

for step in 10 25 45; do
  trace="$DPO_ROOT/screen_checkpoint${step}_traces.jsonl"
  while [[ ! -f "$trace" ]]; do
    sleep 10
  done
done

declare -a trace_specs=(
  "$ROOT/$SFT_RUN_ID/screen_traces.jsonl:$ROOT/$SFT_RUN_ID/screen${EVAL_SUFFIX}_eval"
  "$DPO_ROOT/screen_checkpoint10_traces.jsonl:$DPO_ROOT/screen_checkpoint10${EVAL_SUFFIX}_eval"
  "$DPO_ROOT/screen_checkpoint25_traces.jsonl:$DPO_ROOT/screen_checkpoint25${EVAL_SUFFIX}_eval"
  "$DPO_ROOT/screen_checkpoint45_traces.jsonl:$DPO_ROOT/screen_checkpoint45${EVAL_SUFFIX}_eval"
)
for spec in "${trace_specs[@]}"; do
  traces="${spec%%:*}"
  output_dir="${spec#*:}"
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite evaluator-v2 output: $output_dir" >&2
    exit 2
  fi
  "$PYTHON_BIN" scripts/ecommerce/evaluate_rollout_v1.py \
    --cases "$CASES_ROOT/evaluator_cases.jsonl" \
    --traces "$traces" \
    --output-dir "$output_dir"
done

EVAL_SUFFIX="$EVAL_SUFFIX" bash scripts/ecommerce/analyze_1p5b_dpo_v1p4_screen.sh
touch "$DPO_ROOT/POSTPROCESS_NEGATION_V2_COMPLETED"
echo "DPO v1.4 evaluator-v2 postprocessing completed"
