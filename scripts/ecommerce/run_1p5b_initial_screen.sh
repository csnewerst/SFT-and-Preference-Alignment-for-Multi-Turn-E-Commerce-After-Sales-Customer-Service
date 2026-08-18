#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-1.5B-Instruct}"
CASES_ROOT="${CASES_ROOT:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen}"
CONFIG_PATH="${CONFIG_PATH:-configs/ecommerce/experiments_1p5b_v1.json}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
GPU="${GPU:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-v1}"
EXPERIMENT_SCALE="${EXPERIMENT_SCALE:-1p5b}"
RUN_ID="${RUN_ID:-initial-screen-${RUN_SUFFIX}}"
RUN_DIR="${RUN_DIR:-experiments/local/$EXPERIMENT_SCALE/$RUN_ID}"
TRACES="$RUN_DIR/screen_traces.jsonl"
EVAL_DIR="$RUN_DIR/screen_eval"
LOG_ROOT="${LOG_ROOT:-logs/ecommerce/$EXPERIMENT_SCALE}"
LOG_PATH="$LOG_ROOT/${RUN_ID}.log"

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"

COMMAND="CUDA_VISIBLE_DEVICES=$GPU $PYTHON_BIN scripts/ecommerce/run_ecommerce_rollout.py --cases $CASES_ROOT/cases.jsonl --base-model $MODEL_PATH --output $TRACES --device cuda:0 --max-new-tokens 512 --max-steps 6"

"$PYTHON_BIN" scripts/ecommerce/capture_experiment_manifest.py \
  --output-dir "$RUN_DIR" \
  --run-id "$RUN_ID" \
  --config "$CONFIG_PATH" \
  --input "$MODEL_PATH" \
  --input "$CASES_ROOT/cases.jsonl" \
  --input "$CASES_ROOT/evaluator_cases.jsonl" \
  --command "$COMMAND"

{
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" scripts/ecommerce/run_ecommerce_rollout.py \
    --cases "$CASES_ROOT/cases.jsonl" \
    --base-model "$MODEL_PATH" \
    --output "$TRACES" \
    --device cuda:0 \
    --max-new-tokens 512 \
    --max-steps 6
  "$PYTHON_BIN" scripts/ecommerce/evaluate_rollout_v1.py \
    --cases "$CASES_ROOT/evaluator_cases.jsonl" \
    --traces "$TRACES" \
    --output-dir "$EVAL_DIR"
} 2>&1 | tee "$LOG_PATH"

echo "Initial screen evaluation completed: $RUN_DIR"
