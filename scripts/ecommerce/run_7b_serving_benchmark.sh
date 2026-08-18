#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-7B-Instruct}"
ADAPTER="${ADAPTER:?Set ADAPTER to the frozen final 7B adapter}"
CASES="${CASES:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen/cases.jsonl}"
MAX_CASES="${MAX_CASES:-100}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
RUN_ID="${RUN_ID:-serving-benchmark-7b-dpo-seed42-hf-greedy-100-v1}"
RUN_DIR="experiments/local/7b/$RUN_ID"

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite serving benchmark: $RUN_DIR" >&2
  exit 2
fi
for path in "$MODEL_PATH" "$ADAPTER/adapter_config.json" "$CASES"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing serving benchmark input: $path" >&2
    exit 2
  fi
done

"$PYTHON_BIN" scripts/ecommerce/capture_experiment_manifest.py \
  --output-dir "$RUN_DIR" --run-id "$RUN_ID" --config configs/ecommerce/experiments_7b_v1.json \
  --input "$MODEL_PATH" --input "$ADAPTER" --input "$CASES" \
  --command "bash scripts/ecommerce/run_7b_serving_benchmark.sh"

export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false
scripts/ecommerce/monitor_gpu.sh "$RUN_DIR/hardware.csv" "$CUDA_VISIBLE_DEVICES" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

"$PYTHON_BIN" scripts/ecommerce/run_ecommerce_rollout.py \
  --cases "$CASES" --base-model "$MODEL_PATH" --adapter "$ADAPTER" \
  --output "$RUN_DIR/traces.jsonl" --metrics-output "$RUN_DIR/latency_metrics.json" \
  --device cuda:0 --max-new-tokens 512 --max-steps 6 --max-cases "$MAX_CASES" \
  2>&1 | tee "$RUN_DIR/benchmark.log"
