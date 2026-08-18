#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-1.5B-Instruct}"
CASES_ROOT="${CASES_ROOT:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SFT_TAG="${SFT_TAG:?Set SFT_TAG to the composition matrix parent tag}"
SEED="${SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:-v1}"
mkdir -p logs/ecommerce/1p5b

run_ids=(
  "dpo-response-only-matched-beta0p1-seed${SEED}-from-${SFT_TAG}-${RUN_SUFFIX}"
  "dpo-multigranularity-matched-beta0p1-seed${SEED}-from-${SFT_TAG}-${RUN_SUFFIX}"
  "dpo-multigranularity-full-beta0p1-seed${SEED}-from-${SFT_TAG}-${RUN_SUFFIX}"
)
gpus=(0 1 2)
pids=()

for index in "${!run_ids[@]}"; do
  run_id="${run_ids[$index]}"
  run_dir="experiments/local/1p5b/$run_id"
  traces="$run_dir/screen_traces.jsonl"
  eval_dir="$run_dir/screen_eval"
  if [[ ! -f "$run_dir/adapter/adapter_config.json" ]]; then
    echo "Missing completed DPO adapter for $run_id" >&2
    exit 2
  fi
  if [[ -e "$traces" || -e "$eval_dir" ]]; then
    echo "Refusing to overwrite DPO screen evaluation for $run_id" >&2
    exit 2
  fi
  (
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$PYTHON_BIN" scripts/ecommerce/run_ecommerce_rollout.py \
      --cases "$CASES_ROOT/cases.jsonl" \
      --base-model "$MODEL_PATH" \
      --adapter "$run_dir/adapter" \
      --output "$traces" \
      --device cuda:0 \
      --max-new-tokens 512 \
      --max-steps 6
    "$PYTHON_BIN" scripts/ecommerce/evaluate_rollout_v1.py \
      --cases "$CASES_ROOT/evaluator_cases.jsonl" \
      --traces "$traces" \
      --output-dir "$eval_dir"
  ) >"logs/ecommerce/1p5b/${run_id}.screen.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "DPO screen evaluation completed ${run_ids[$index]}"
  else
    echo "DPO screen evaluation failed ${run_ids[$index]}" >&2
    status=1
  fi
done
exit "$status"
