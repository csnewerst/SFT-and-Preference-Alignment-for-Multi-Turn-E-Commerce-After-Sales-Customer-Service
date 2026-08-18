#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ID="${RUN_ID:?Set RUN_ID to an existing 7B run}"
ADAPTER="${ADAPTER:?Set ADAPTER to the adapter or checkpoint to evaluate}"
EVAL_TAG="${EVAL_TAG:-screen}"
GPU="${GPU:-0}"
RUN_DIR="experiments/local/7b/$RUN_ID"
TRACES="$RUN_DIR/${EVAL_TAG}_traces.jsonl"
EVAL_DIR="$RUN_DIR/${EVAL_TAG}_eval"
if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
  echo "Missing adapter: $ADAPTER" >&2
  exit 2
fi
if [[ -e "$TRACES" || -e "$EVAL_DIR" ]]; then
  echo "Refusing to overwrite adapter evaluation: $RUN_ID/$EVAL_TAG" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="$GPU" /root/miniconda3/bin/python scripts/ecommerce/run_ecommerce_rollout.py \
  --cases data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen/cases.jsonl \
  --base-model models/base/Qwen2.5-7B-Instruct \
  --adapter "$ADAPTER" --output "$TRACES" --device cuda:0 --max-new-tokens 512 --max-steps 6
/root/miniconda3/bin/python scripts/ecommerce/evaluate_rollout_v1.py \
  --cases data/ecommerce/rollout_prefreeze_v1_1_zh_1p5b_split/screen/evaluator_cases.jsonl \
  --traces "$TRACES" --output-dir "$EVAL_DIR"
