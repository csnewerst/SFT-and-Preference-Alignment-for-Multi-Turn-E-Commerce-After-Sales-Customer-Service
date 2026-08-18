#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PLAN_CONFIG="${PLAN_CONFIG:-configs/ecommerce/experiments_1p5b_v1.json}"
MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-1.5B-Instruct}"
DATA_ROOT="${DATA_ROOT:-data/ecommerce/domain_train_v1_3_2_zh}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN_ID="${RUN_ID:?Set RUN_ID to a unique immutable experiment ID}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-$((LORA_RANK * 2))}"
TARGET_MODULES="${TARGET_MODULES:-all}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"
MICRO_BATCH="${MICRO_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:--1}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
RUN_DIR="${RUN_DIR:-experiments/local/1p5b/$RUN_ID}"

if [[ ! -d "$MODEL_PATH" || ! -d "$DATA_ROOT/sft/train" || ! -d "$DATA_ROOT/sft/validation" ]]; then
  echo "Missing model or SFT data directories" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false

train_command=(
  "$PYTHON_BIN" training/supervised_finetuning.py
  --model_name_or_path "$MODEL_PATH"
  --device_map none
  --train_file_dir "$DATA_ROOT/sft/train"
  --validation_file_dir "$DATA_ROOT/sft/validation"
  --do_train True --do_eval True --use_peft True
  --target_modules "$TARGET_MODULES"
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" --lora_dropout 0.05
  --model_max_length "$MODEL_MAX_LENGTH"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --max_steps "$MAX_STEPS"
  --per_device_train_batch_size "$MICRO_BATCH"
  --per_device_eval_batch_size "$MICRO_BATCH"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "$LEARNING_RATE" --lr_scheduler_type cosine --warmup_steps "$WARMUP_STEPS" --weight_decay 0.01
  --logging_strategy steps --logging_steps 5 --logging_first_step True
  --eval_strategy "$EVAL_STRATEGY" --eval_steps "$EVAL_STEPS"
  --save_strategy "$SAVE_STRATEGY" --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT"
  --gradient_checkpointing True --preprocessing_num_workers 8
  --tool_format default --torch_dtype bfloat16 --bf16 True --fp16 False
  --seed "$SEED" --data_seed "$DATA_SEED"
  --report_to tensorboard --run_name "$RUN_ID"
  --cache_dir cache --output_dir "$RUN_DIR/adapter"
)
printf -v command_text '%q ' "${train_command[@]}"

"$PYTHON_BIN" scripts/ecommerce/capture_experiment_manifest.py \
  --output-dir "$RUN_DIR" --run-id "$RUN_ID" --config "$PLAN_CONFIG" \
  --input "$MODEL_PATH" --input "$DATA_ROOT/sft/train" --input "$DATA_ROOT/sft/validation" \
  --command "$command_text"

scripts/ecommerce/monitor_gpu.sh "$RUN_DIR/hardware.csv" "$CUDA_VISIBLE_DEVICES" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

"${train_command[@]}" 2>&1 | tee "$RUN_DIR/train.log"
