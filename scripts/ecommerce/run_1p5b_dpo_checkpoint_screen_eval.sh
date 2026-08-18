#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-1.5B-Instruct}"
CASES_ROOT="${CASES_ROOT:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SFT_TAG="${SFT_TAG:?Set SFT_TAG to the composition matrix parent tag}"
SEED="${SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:?Set RUN_SUFFIX to the immutable DPO run suffix}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:?Set CHECKPOINT_STEP to an existing saved step}"

if [[ ! "$CHECKPOINT_STEP" =~ ^[1-9][0-9]*$ ]]; then
  echo "CHECKPOINT_STEP must be a positive integer" >&2
  exit 2
fi

run_ids=(
  "dpo-response-only-matched-beta0p1-seed${SEED}-from-${SFT_TAG}-${RUN_SUFFIX}"
  "dpo-multigranularity-matched-beta0p1-seed${SEED}-from-${SFT_TAG}-${RUN_SUFFIX}"
  "dpo-multigranularity-full-beta0p1-seed${SEED}-from-${SFT_TAG}-${RUN_SUFFIX}"
)
gpus=(0 1 2)
pids=()
mkdir -p logs/ecommerce/1p5b

for index in "${!run_ids[@]}"; do
  run_id="${run_ids[$index]}"
  run_dir="experiments/local/1p5b/$run_id"
  adapter="$run_dir/adapter/checkpoint-$CHECKPOINT_STEP"
  traces="$run_dir/screen_checkpoint${CHECKPOINT_STEP}_traces.jsonl"
  eval_dir="$run_dir/screen_checkpoint${CHECKPOINT_STEP}_eval"
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    echo "Missing checkpoint adapter for $run_id at step $CHECKPOINT_STEP" >&2
    exit 2
  fi
  if [[ -e "$traces" || -e "$eval_dir" ]]; then
    echo "Refusing to overwrite checkpoint evaluation for $run_id at step $CHECKPOINT_STEP" >&2
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
  ) >"logs/ecommerce/1p5b/${run_id}.screen-checkpoint${CHECKPOINT_STEP}.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Checkpoint screen evaluation completed ${run_ids[$index]} step $CHECKPOINT_STEP"
  else
    echo "Checkpoint screen evaluation failed ${run_ids[$index]} step $CHECKPOINT_STEP" >&2
    status=1
  fi
done
exit "$status"
