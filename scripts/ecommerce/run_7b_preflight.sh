#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-7B-Instruct}"
RUN_ID="${RUN_ID:-preflight-bf16-lora-r8-seq1024-b2-v1}"
OUTPUT="experiments/local/7b/$RUN_ID/preflight.json"
if [[ -e "$(dirname "$OUTPUT")" ]]; then
  echo "Refusing to overwrite immutable preflight: $(dirname "$OUTPUT")" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  /root/miniconda3/bin/python scripts/ecommerce/preflight_7b_lora.py \
    --model-path "$MODEL_PATH" --output "$OUTPUT" --max-length 1024 --batch-size 2
