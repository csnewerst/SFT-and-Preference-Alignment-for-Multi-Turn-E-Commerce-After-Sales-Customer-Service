#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TRAIN_SEED="${TRAIN_SEED:-43}"
PLAN_CONFIG="${PLAN_CONFIG:-configs/ecommerce/experiments_7b_seed43_replication.json}"
MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-7B-Instruct}"
FORMAL_ROOT="${FORMAL_ROOT:-data/ecommerce/formal_test_v2}"
BASELINE_RUN="${BASELINE_RUN:-experiments/local/7b/formal-test-v2-7b-final-v2}"
SFT_ADAPTER="${SFT_ADAPTER:?Set the pre-registered seed replication SFT adapter}"
DPO_ADAPTER="${DPO_ADAPTER:?Set the pre-registered seed replication DPO adapter}"
RUN_ID="${RUN_ID:-formal-test-v2-7b-seed${TRAIN_SEED}-replication-v1}"
RUN_DIR="experiments/local/7b/$RUN_ID"

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite seed replication evaluation: $RUN_DIR" >&2
  exit 2
fi
for path in "$MODEL_PATH" "$FORMAL_ROOT" "$BASELINE_RUN/initial_eval/per_sample.jsonl"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing replication evaluation input: $path" >&2
    exit 2
  fi
done
for adapter in "$SFT_ADAPTER" "$DPO_ADAPTER"; do
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    echo "Missing frozen replication adapter: $adapter" >&2
    exit 2
  fi
done

"$PYTHON_BIN" scripts/ecommerce/capture_experiment_manifest.py \
  --output-dir "$RUN_DIR" --run-id "$RUN_ID" \
  --config "$PLAN_CONFIG" \
  --input "$MODEL_PATH" --input "$FORMAL_ROOT" --input "$SFT_ADAPTER" --input "$DPO_ADAPTER" \
  --command "bash scripts/ecommerce/run_7b_seed_replication_eval.sh"
mkdir -p "$RUN_DIR/logs"

run_one() {
  local label="$1"
  local gpu="$2"
  local adapter="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/ecommerce/run_ecommerce_rollout.py \
    --cases "$FORMAL_ROOT/cases.jsonl" --base-model "$MODEL_PATH" --adapter "$adapter" \
    --output "$RUN_DIR/${label}_traces.jsonl" --device cuda:0 --max-new-tokens 512 --max-steps 6
  "$PYTHON_BIN" scripts/ecommerce/evaluate_rollout_v1.py \
    --cases "$FORMAL_ROOT/evaluator_cases.jsonl" --traces "$RUN_DIR/${label}_traces.jsonl" \
    --output-dir "$RUN_DIR/${label}_eval"
}

run_one sft 0 "$SFT_ADAPTER" >"$RUN_DIR/logs/sft.log" 2>&1 & sft_pid=$!
run_one dpo 1 "$DPO_ADAPTER" >"$RUN_DIR/logs/dpo.log" 2>&1 & dpo_pid=$!
status=0
wait "$sft_pid" || status=1
wait "$dpo_pid" || status=1
if [[ "$status" -ne 0 ]]; then
  echo "Seed replication formal evaluation failed" >&2
  exit "$status"
fi

"$PYTHON_BIN" scripts/ecommerce/compare_1p5b_screen_runs.py \
  --cases "$FORMAL_ROOT/cases.jsonl" \
  --run "initial=$BASELINE_RUN/initial_eval/per_sample.jsonl" \
  --run "sft=$RUN_DIR/sft_eval/per_sample.jsonl" \
  --run "dpo=$RUN_DIR/dpo_eval/per_sample.jsonl" \
  --seed 20260809 --resamples 10000 --output "$RUN_DIR/paired_comparison.json"

"$PYTHON_BIN" scripts/ecommerce/analyze_1p5b_dpo_failures.py \
  --cases "$FORMAL_ROOT/cases.jsonl" --baseline "$RUN_DIR/sft_eval/per_sample.jsonl" \
  --run "dpo=$RUN_DIR/dpo_eval/per_sample.jsonl" --output "$RUN_DIR/failure_transitions_dpo_vs_sft.json"

echo "Seed $TRAIN_SEED replication completed: $RUN_DIR"
