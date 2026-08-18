#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-7B-Instruct}"
CASES_ROOT="${CASES_ROOT:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN_SUFFIX="${RUN_SUFFIX:-v1}"
SEED="${SEED:-42}"
LOG_ROOT="logs/ecommerce/7b"
mkdir -p "$LOG_ROOT"

ranks=(8 16)
gpus=(0 1)
pids=()
run_ids=()

for index in "${!ranks[@]}"; do
  rank="${ranks[$index]}"
  run_id="sft-r${rank}-all-cal100-seed${SEED}-${RUN_SUFFIX}"
  run_dir="experiments/local/7b/$run_id"
  traces="$run_dir/calibration_screen_traces.jsonl"
  eval_dir="$run_dir/calibration_screen_eval"
  if [[ ! -f "$run_dir/adapter/adapter_config.json" ]]; then
    echo "Missing completed adapter for $run_id" >&2
    exit 2
  fi
  if [[ -e "$traces" || -e "$eval_dir" ]]; then
    echo "Refusing to overwrite calibration evaluation for $run_id" >&2
    exit 2
  fi
  run_ids+=("$run_id")
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
  ) >"$LOG_ROOT/${run_id}.calibration-screen.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "7B SFT calibration evaluation completed: ${run_ids[$index]}"
  else
    echo "7B SFT calibration evaluation failed: ${run_ids[$index]}" >&2
    status=1
  fi
done
exit "$status"
