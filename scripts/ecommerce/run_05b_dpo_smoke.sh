#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-0.5B-Instruct}"
DATA_ROOT="${DATA_ROOT:-data/ecommerce/domain_pilot_v1_1_1}"
RUN_ROOT="${RUN_ROOT:-outputs/ecommerce/05b_smoke}"
LOG_ROOT="${LOG_ROOT:-logs/ecommerce/05b_smoke}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SFT_ADAPTER="${SFT_ADAPTER:-$RUN_ROOT/sft}"
DPO_OUTPUT="${DPO_OUTPUT:-$RUN_ROOT/dpo}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-64}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-24}"
MAX_STEPS="${MAX_STEPS:-10}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" cache

if [[ ! -f "$SFT_ADAPTER/adapter_config.json" ]]; then
  echo "Missing SFT adapter: $SFT_ADAPTER/adapter_config.json" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" training/dpo_training.py \
  --model_name_or_path "$MODEL_PATH" \
  --device_map none \
  --peft_path "$SFT_ADAPTER" \
  --train_file_dir "$DATA_ROOT/dpo/train" \
  --validation_file_dir "$DATA_ROOT/dpo/validation" \
  --do_train True \
  --do_eval True \
  --use_peft True \
  --beta 0.1 \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_eval_samples "$MAX_EVAL_SAMPLES" \
  --max_source_length 4096 \
  --max_target_length 512 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-6 \
  --warmup_steps 1 \
  --weight_decay 0.01 \
  --logging_steps 1 \
  --eval_strategy steps \
  --eval_steps 5 \
  --save_steps 10 \
  --gradient_checkpointing True \
  --preprocessing_num_workers 4 \
  --tool_format default \
  --torch_dtype bfloat16 \
  --bf16 True \
  --fp16 False \
  --seed 42 \
  --report_to tensorboard \
  --output_dir "$DPO_OUTPUT" 2>&1 | tee "$LOG_ROOT/dpo.log"
