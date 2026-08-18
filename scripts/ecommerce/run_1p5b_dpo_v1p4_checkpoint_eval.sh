#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-1.5B-Instruct}"
CASES_ROOT="${CASES_ROOT:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN_ID="${RUN_ID:-dpo-v1p4-rollout-quality-screen800-beta0p1-seed42-from-r4-v1}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-10 25 45}"
EXPERIMENT_SCALE="${EXPERIMENT_SCALE:-1p5b}"
RUN_DIR="${RUN_DIR:-experiments/local/$EXPERIMENT_SCALE/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-logs/ecommerce/$EXPERIMENT_SCALE/$RUN_ID}"
read -r -a steps <<<"$CHECKPOINT_STEPS"
gpus=(0 1 2)

if [[ "${#steps[@]}" -ne 3 ]]; then
  echo "CHECKPOINT_STEPS must contain exactly three steps for GPUs 0,1,2" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"
pids=()
for index in 0 1 2; do
  step="${steps[$index]}"
  adapter="$RUN_DIR/adapter/checkpoint-$step"
  traces="$RUN_DIR/screen_checkpoint${step}_traces.jsonl"
  eval_dir="$RUN_DIR/screen_checkpoint${step}_eval"
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    echo "Missing checkpoint adapter: $adapter" >&2
    exit 2
  fi
  if [[ -e "$traces" || -e "$eval_dir" ]]; then
    echo "Refusing to overwrite checkpoint-$step evaluation" >&2
    exit 2
  fi
  (
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" "$PYTHON_BIN" scripts/ecommerce/run_ecommerce_rollout.py \
      --cases "$CASES_ROOT/cases.jsonl" \
      --base-model "$MODEL_PATH" \
      --adapter "$adapter" \
      --output "$traces" \
      --device cuda:0 \
      --max-new-tokens 512 \
      --max-steps 6
    "$PYTHON_BIN" scripts/ecommerce/evaluate_rollout_v1.py \
      --cases "$CASES_ROOT/evaluator_cases.jsonl" \
      --traces "$traces" \
      --output-dir "$eval_dir"
  ) >"$LOG_ROOT/checkpoint-${step}-screen.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed checkpoint-${steps[$index]} screen evaluation"
  else
    echo "Failed checkpoint-${steps[$index]} screen evaluation" >&2
    status=1
  fi
done
exit "$status"
