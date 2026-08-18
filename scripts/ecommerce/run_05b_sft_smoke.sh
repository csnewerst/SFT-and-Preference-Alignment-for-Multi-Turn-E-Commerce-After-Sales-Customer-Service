#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-0.5B-Instruct}"
DATA_ROOT="${DATA_ROOT:-data/ecommerce/domain_pilot_v1_1_1}"
RUN_ROOT="${RUN_ROOT:-outputs/ecommerce/05b_smoke}"
LOG_ROOT="${LOG_ROOT:-logs/ecommerce/05b_smoke}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-128}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-32}"
MAX_STEPS="${MAX_STEPS:-20}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" cache

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" training/supervised_finetuning.py \
  --model_name_or_path "$MODEL_PATH" \
  --device_map none \
  --train_file_dir "$DATA_ROOT/sft/train" \
  --validation_file_dir "$DATA_ROOT/sft/validation" \
  --do_train True \
  --do_eval True \
  --use_peft True \
  --target_modules all \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_eval_samples "$MAX_EVAL_SAMPLES" \
  --model_max_length 1024 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --warmup_steps 2 \
  --weight_decay 0.01 \
  --logging_strategy steps \
  --logging_steps 1 \
  --logging_first_step True \
  --eval_strategy steps \
  --eval_steps 10 \
  --save_strategy steps \
  --save_steps 20 \
  --save_total_limit 1 \
  --gradient_checkpointing True \
  --preprocessing_num_workers 4 \
  --tool_format default \
  --torch_dtype bfloat16 \
  --bf16 True \
  --fp16 False \
  --seed 42 \
  --data_seed 42 \
  --report_to tensorboard \
  --run_name ecommerce-05b-sft-smoke \
  --cache_dir cache \
  --output_dir "$RUN_ROOT/sft" 2>&1 | tee "$LOG_ROOT/sft.log"
