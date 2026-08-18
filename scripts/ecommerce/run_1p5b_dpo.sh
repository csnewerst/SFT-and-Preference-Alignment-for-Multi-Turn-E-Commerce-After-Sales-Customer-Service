#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PLAN_CONFIG="${PLAN_CONFIG:-configs/ecommerce/experiments_1p5b_v1.json}"
MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-1.5B-Instruct}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the selected DPO dataset variant}"
SFT_ADAPTER="${SFT_ADAPTER:?Set SFT_ADAPTER to the selected immutable SFT run}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN_ID="${RUN_ID:?Set RUN_ID to a unique immutable experiment ID}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-1024}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-128}"
MAX_STEPS="${MAX_STEPS:?Set MAX_STEPS for one controlled pass over this DPO variant}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BETA="${BETA:-0.1}"
SEED="${SEED:-42}"
MICRO_BATCH="${MICRO_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-$(((MAX_STEPS * 3 + 99) / 100))}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
RUN_DIR="${RUN_DIR:-experiments/local/1p5b/$RUN_ID}"

if [[ ! -d "$MODEL_PATH" || ! -f "$SFT_ADAPTER/adapter_config.json" || ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/validation" ]]; then
  echo "Missing model, SFT adapter, or DPO data directories" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES TOKENIZERS_PARALLELISM=false

train_command=(
  "$PYTHON_BIN" training/dpo_training.py
  --model_name_or_path "$MODEL_PATH" --device_map none --peft_path "$SFT_ADAPTER"
  --train_file_dir "$DATA_ROOT/train" --validation_file_dir "$DATA_ROOT/validation"
  --do_train True --do_eval True --use_peft True --beta "$BETA"
  --max_source_length "$MAX_SOURCE_LENGTH" --max_target_length "$MAX_TARGET_LENGTH"
  --max_steps "$MAX_STEPS"
  --per_device_train_batch_size "$MICRO_BATCH" --per_device_eval_batch_size "$MICRO_BATCH"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "$LEARNING_RATE" --lr_scheduler_type cosine --warmup_steps "$WARMUP_STEPS" --weight_decay 0.01
  --logging_steps "$LOGGING_STEPS" --eval_strategy steps --eval_steps "$EVAL_STEPS" --save_steps "$SAVE_STEPS"
  --gradient_checkpointing True --preprocessing_num_workers 8
  --tool_format default --torch_dtype bfloat16 --bf16 True --fp16 False
  --verify_reference_logps True --reference_logps_tolerance 1e-4
  --seed "$SEED" --report_to tensorboard --output_dir "$RUN_DIR/adapter"
)
printf -v command_text '%q ' "${train_command[@]}"

"$PYTHON_BIN" scripts/ecommerce/capture_experiment_manifest.py \
  --output-dir "$RUN_DIR" --run-id "$RUN_ID" --config "$PLAN_CONFIG" \
  --input "$MODEL_PATH" --input "$DATA_ROOT/train" --input "$DATA_ROOT/validation" --input "$SFT_ADAPTER" \
  --command "$command_text"

scripts/ecommerce/monitor_gpu.sh "$RUN_DIR/hardware.csv" "$CUDA_VISIBLE_DEVICES" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

"${train_command[@]}" 2>&1 | tee "$RUN_DIR/train.log"
