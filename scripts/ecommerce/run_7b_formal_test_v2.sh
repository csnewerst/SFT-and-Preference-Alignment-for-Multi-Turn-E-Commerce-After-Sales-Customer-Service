#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-7B-Instruct}"
FORMAL_ROOT="${FORMAL_ROOT:-data/ecommerce/formal_test_v2}"
SCREEN_ROOT="${SCREEN_ROOT:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen}"
GATE_ROOT="${GATE_ROOT:-data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/gate}"
SFT_ROOT="${SFT_ROOT:-data/ecommerce/domain_train_v1_3_2_zh}"
DPO_ROOT="${DPO_ROOT:-data/ecommerce/dpo_v1_4_rollout_quality_screen_800_v2}"
SFT_ADAPTER="${SFT_ADAPTER:?Set SFT_ADAPTER to the frozen 7B SFT adapter}"
DPO_ADAPTER="${DPO_ADAPTER:?Set DPO_ADAPTER to the frozen 7B DPO adapter}"
RUN_ID="${RUN_ID:-formal-test-v2-7b-final-v1}"
RUN_DIR="experiments/local/7b/$RUN_ID"

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite formal evaluation: $RUN_DIR" >&2
  exit 2
fi
for path in "$MODEL_PATH" "$FORMAL_ROOT" "$SCREEN_ROOT" "$GATE_ROOT" "$SFT_ROOT" "$DPO_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing formal evaluation input: $path" >&2
    exit 2
  fi
done
for adapter in "$SFT_ADAPTER" "$DPO_ADAPTER"; do
  if [[ ! -f "$adapter/adapter_config.json" ]]; then
    echo "Missing frozen adapter: $adapter" >&2
    exit 2
  fi
done

"$PYTHON_BIN" scripts/ecommerce/capture_experiment_manifest.py \
  --output-dir "$RUN_DIR" --run-id "$RUN_ID" \
  --config configs/ecommerce/experiments_7b_v1.json \
  --input "$MODEL_PATH" --input "$FORMAL_ROOT" \
  --input "$SFT_ADAPTER" --input "$DPO_ADAPTER" \
  --command "bash scripts/ecommerce/run_7b_formal_test_v2.sh"
mkdir -p "$RUN_DIR/logs"

"$PYTHON_BIN" scripts/ecommerce/validate_formal_test_v2.py \
  --candidate-dir "$FORMAL_ROOT" \
  --development-dir "$SCREEN_ROOT" \
  --development-dir "$GATE_ROOT" \
  --reference-jsonl "$SFT_ROOT/sft/train/data.jsonl" \
  --reference-jsonl "$SFT_ROOT/sft/validation/data.jsonl" \
  --reference-jsonl "$DPO_ROOT/train/records.jsonl" \
  --reference-jsonl "$DPO_ROOT/validation/records.jsonl" \
  --minimum-cases 600 \
  --output "$RUN_DIR/formal_validation.json"

run_one() {
  local label="$1"
  local gpu="$2"
  local adapter="${3:-}"
  local command=(
    "$PYTHON_BIN" scripts/ecommerce/run_ecommerce_rollout.py
    --cases "$FORMAL_ROOT/cases.jsonl"
    --base-model "$MODEL_PATH"
    --output "$RUN_DIR/${label}_traces.jsonl"
    --device cuda:0
    --max-new-tokens 512
    --max-steps 6
  )
  if [[ -n "$adapter" ]]; then
    command+=(--adapter "$adapter")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"
  "$PYTHON_BIN" scripts/ecommerce/evaluate_rollout_v1.py \
    --cases "$FORMAL_ROOT/evaluator_cases.jsonl" \
    --traces "$RUN_DIR/${label}_traces.jsonl" \
    --output-dir "$RUN_DIR/${label}_eval"
}

pids=()
run_one initial 0 >"$RUN_DIR/logs/initial.log" 2>&1 & pids+=("$!")
run_one sft 1 "$SFT_ADAPTER" >"$RUN_DIR/logs/sft.log" 2>&1 & pids+=("$!")
run_one dpo 2 "$DPO_ADAPTER" >"$RUN_DIR/logs/dpo.log" 2>&1 & pids+=("$!")

status=0
labels=(initial sft dpo)
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed ${labels[$index]} formal evaluation"
  else
    echo "Failed ${labels[$index]} formal evaluation" >&2
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PYTHON_BIN" scripts/ecommerce/compare_1p5b_screen_runs.py \
  --cases "$FORMAL_ROOT/cases.jsonl" \
  --run "initial=$RUN_DIR/initial_eval/per_sample.jsonl" \
  --run "sft=$RUN_DIR/sft_eval/per_sample.jsonl" \
  --run "dpo=$RUN_DIR/dpo_eval/per_sample.jsonl" \
  --seed 20260809 --resamples 10000 \
  --output "$RUN_DIR/paired_comparison.json"

"$PYTHON_BIN" scripts/ecommerce/analyze_1p5b_dpo_failures.py \
  --cases "$FORMAL_ROOT/cases.jsonl" \
  --baseline "$RUN_DIR/initial_eval/per_sample.jsonl" \
  --run "sft=$RUN_DIR/sft_eval/per_sample.jsonl" \
  --run "dpo=$RUN_DIR/dpo_eval/per_sample.jsonl" \
  --output "$RUN_DIR/failure_transitions_vs_initial.json"

"$PYTHON_BIN" scripts/ecommerce/analyze_1p5b_dpo_failures.py \
  --cases "$FORMAL_ROOT/cases.jsonl" \
  --baseline "$RUN_DIR/sft_eval/per_sample.jsonl" \
  --run "dpo=$RUN_DIR/dpo_eval/per_sample.jsonl" \
  --output "$RUN_DIR/failure_transitions_dpo_vs_sft.json"

echo "Formal test v2 completed: $RUN_DIR"
