#!/usr/bin/env bash
set -euo pipefail

SFT_ADAPTER="${SFT_ADAPTER:?Set SFT_ADAPTER to the selected immutable 7B SFT adapter}"
TRAIN_SEED="${TRAIN_SEED:-42}"
PLAN_CONFIG="${PLAN_CONFIG:-configs/ecommerce/experiments_7b_v1.json}"
RUN_SUFFIX="${RUN_SUFFIX:-v1}"
RUN_ID="${RUN_ID:-dpo-v1p4-full720-beta0p1-lr2e6-seed${TRAIN_SEED}-${RUN_SUFFIX}}"

PLAN_CONFIG="$PLAN_CONFIG" \
MODEL_PATH=models/base/Qwen2.5-7B-Instruct \
DATA_ROOT=data/ecommerce/dpo_v1_4_rollout_quality_screen_800_v2 \
SFT_ADAPTER="$SFT_ADAPTER" RUN_ID="$RUN_ID" RUN_DIR="experiments/local/7b/$RUN_ID" \
MAX_STEPS=20 SAVE_STEPS=5 EVAL_STEPS=5 LOGGING_STEPS=5 \
WARMUP_STEPS=1 BETA=0.1 LEARNING_RATE=2e-6 SEED="$TRAIN_SEED" \
MICRO_BATCH=2 GRAD_ACCUM=8 \
  exec bash scripts/ecommerce/run_1p5b_dpo.sh
