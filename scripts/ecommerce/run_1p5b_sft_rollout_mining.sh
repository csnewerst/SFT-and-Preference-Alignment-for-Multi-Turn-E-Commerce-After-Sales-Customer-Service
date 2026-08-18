#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-1.5B-Instruct}"
SFT_RUN_ID="${SFT_RUN_ID:-sft-r4-all-full-seed42-v1}"
ADAPTER_PATH="${ADAPTER_PATH:-experiments/local/1p5b/$SFT_RUN_ID/adapter}"
CASES_ROOT="${CASES_ROOT:-data/ecommerce/rollout_mining_v1_csds_dch2_train_2000}"
CONFIG_PATH="${CONFIG_PATH:-configs/ecommerce/dpo_v1_4_quality.json}"
RUN_ID="${RUN_ID:-sft-r4-rollout-mining-csds-dch2-train2000-v1}"
RUN_DIR="experiments/local/1p5b/$RUN_ID"
LOG_ROOT="logs/ecommerce/1p5b/$RUN_ID"
SHARD_COUNT="${SHARD_COUNT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_STEPS="${MAX_STEPS:-6}"

if [[ "$SHARD_COUNT" -ne 4 ]]; then
  echo "This reproducible runner requires exactly four shards/GPUs" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite immutable run directory: $RUN_DIR" >&2
  exit 2
fi
for path in "$MODEL_PATH" "$ADAPTER_PATH/adapter_config.json" "$CASES_ROOT/cases.jsonl" "$CASES_ROOT/evaluator_cases.jsonl"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done
mkdir -p "$LOG_ROOT"

COMMAND="CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/ecommerce/run_1p5b_sft_rollout_mining.sh"
"$PYTHON_BIN" scripts/ecommerce/capture_experiment_manifest.py \
  --output-dir "$RUN_DIR" \
  --run-id "$RUN_ID" \
  --config "$CONFIG_PATH" \
  --input "$MODEL_PATH" \
  --input "$ADAPTER_PATH" \
  --input "$CASES_ROOT/cases.jsonl" \
  --input "$CASES_ROOT/evaluator_cases.jsonl" \
  --command "$COMMAND"

"$PYTHON_BIN" scripts/ecommerce/shard_rollout_cases.py prepare \
  --cases "$CASES_ROOT/cases.jsonl" \
  --output-dir "$RUN_DIR/shards" \
  --shard-count "$SHARD_COUNT"

bash scripts/ecommerce/monitor_gpu.sh "$RUN_DIR/gpu_samples.csv" >"$LOG_ROOT/gpu-monitor.log" 2>&1 &
monitor_pid="$!"
pids=()
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

for index in 0 1 2 3; do
  shard_tag="$(printf '%02d' "$index")"
  (
    CUDA_VISIBLE_DEVICES="$index" "$PYTHON_BIN" scripts/ecommerce/run_ecommerce_rollout.py \
      --cases "$RUN_DIR/shards/cases-${shard_tag}.jsonl" \
      --base-model "$MODEL_PATH" \
      --adapter "$ADAPTER_PATH" \
      --output "$RUN_DIR/shards/traces-${shard_tag}.jsonl" \
      --device cuda:0 \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --max-steps "$MAX_STEPS"
  ) >"$LOG_ROOT/shard-${shard_tag}.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Rollout mining shard $index failed" >&2
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

merge_args=()
for index in 0 1 2 3; do
  shard_tag="$(printf '%02d' "$index")"
  merge_args+=(--trace "$RUN_DIR/shards/traces-${shard_tag}.jsonl")
done
"$PYTHON_BIN" scripts/ecommerce/shard_rollout_cases.py merge \
  --cases "$CASES_ROOT/cases.jsonl" \
  "${merge_args[@]}" \
  --output "$RUN_DIR/traces.jsonl"

"$PYTHON_BIN" scripts/ecommerce/evaluate_rollout_v1.py \
  --cases "$CASES_ROOT/evaluator_cases.jsonl" \
  --traces "$RUN_DIR/traces.jsonl" \
  --output-dir "$RUN_DIR/eval"
touch "$RUN_DIR/COMPLETED"
echo "SFT rollout mining completed: $RUN_DIR"
