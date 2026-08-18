#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-0.5B-Instruct}"
RUN_ROOT="${RUN_ROOT:-outputs/ecommerce/05b_pilot}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/rollout_eval_v1}"
CASES_ROOT="${CASES_ROOT:-data/ecommerce/rollout_dev_v1}"
CASES_FILE="${CASES_FILE:-$CASES_ROOT/cases.jsonl}"
BUILD_DEV_CASES="${BUILD_DEV_CASES:-1}"
SFT_ADAPTER="${SFT_ADAPTER:-$RUN_ROOT/sft}"
DPO_ADAPTER="${DPO_ADAPTER:-$RUN_ROOT/dpo}"
MAX_STEPS="${MAX_STEPS:-6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"

for required in "$MODEL_PATH" "$SFT_ADAPTER/adapter_config.json" "$DPO_ADAPTER/adapter_config.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EVAL_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

if [[ "$BUILD_DEV_CASES" == "1" ]]; then
  "$PYTHON_BIN" scripts/ecommerce/build_rollout_dev_v1.py --output-dir "$CASES_ROOT"
elif [[ ! -f "$CASES_FILE" ]]; then
  echo "Missing rollout cases file: $CASES_FILE" >&2
  exit 2
fi

for stage in initial sft sft_dpo; do
  adapter_args=()
  if [[ "$stage" == "sft" ]]; then
    adapter_args=(--adapter "$SFT_ADAPTER")
  elif [[ "$stage" == "sft_dpo" ]]; then
    adapter_args=(--adapter "$DPO_ADAPTER")
  fi

  "$PYTHON_BIN" scripts/ecommerce/run_ecommerce_rollout.py \
    --cases "$CASES_FILE" \
    --base-model "$MODEL_PATH" \
    "${adapter_args[@]}" \
    --max-steps "$MAX_STEPS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output "$EVAL_ROOT/${stage}_traces.jsonl"

  "$PYTHON_BIN" scripts/ecommerce/evaluate_rollout_v1.py \
    --cases "$CASES_FILE" \
    --traces "$EVAL_ROOT/${stage}_traces.jsonl" \
    --output-dir "$EVAL_ROOT/$stage"
done

"$PYTHON_BIN" - "$EVAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for stage in ("initial", "sft", "sft_dpo"):
    with (root / stage / "summary.json").open("r", encoding="utf-8") as input_file:
        summary[stage] = json.load(input_file)
with (root / "comparison_summary.json").open("w", encoding="utf-8", newline="\n") as output_file:
    json.dump(summary, output_file, ensure_ascii=False, indent=2, sort_keys=True)
    output_file.write("\n")
print(json.dumps({stage: result["metrics"]["task_success_rate"] for stage, result in summary.items()}, sort_keys=True))
PY
