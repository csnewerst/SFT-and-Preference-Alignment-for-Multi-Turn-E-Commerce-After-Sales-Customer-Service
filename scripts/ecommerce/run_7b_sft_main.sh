#!/usr/bin/env bash
set -euo pipefail

LORA_RANK="${LORA_RANK:?Set LORA_RANK to the winning 7B calibration rank}"
TRAIN_SEED="${TRAIN_SEED:-42}"
DATA_SEED="${DATA_SEED:-$TRAIN_SEED}"
PLAN_CONFIG="${PLAN_CONFIG:-configs/ecommerce/experiments_7b_v1.json}"
RUN_SUFFIX="${RUN_SUFFIX:-v1}"
RUN_ID="${RUN_ID:-sft-r${LORA_RANK}-all-epoch1-seed${TRAIN_SEED}-${RUN_SUFFIX}}"

PLAN_CONFIG="$PLAN_CONFIG" \
MODEL_PATH=models/base/Qwen2.5-7B-Instruct \
DATA_ROOT=data/ecommerce/domain_train_v1_3_2_zh \
RUN_ID="$RUN_ID" RUN_DIR="experiments/local/7b/$RUN_ID" \
LORA_RANK="$LORA_RANK" LORA_ALPHA="$((LORA_RANK * 2))" TARGET_MODULES=all \
MICRO_BATCH=2 GRAD_ACCUM=16 MAX_STEPS=-1 NUM_TRAIN_EPOCHS=1 \
LEARNING_RATE=1e-5 WARMUP_STEPS=10 \
EVAL_STRATEGY=steps EVAL_STEPS=100 SAVE_STRATEGY=steps SAVE_STEPS=100 SAVE_TOTAL_LIMIT=4 \
SEED="$TRAIN_SEED" DATA_SEED="$DATA_SEED" \
  exec bash scripts/ecommerce/run_1p5b_sft.sh
