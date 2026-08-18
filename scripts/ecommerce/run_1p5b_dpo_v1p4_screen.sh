#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export PLAN_CONFIG="${PLAN_CONFIG:-configs/ecommerce/dpo_v1_4_quality.json}"
export DATA_ROOT="${DATA_ROOT:-data/ecommerce/dpo_v1_4_rollout_quality_screen_800_v2}"
export SFT_ADAPTER="${SFT_ADAPTER:-experiments/local/1p5b/sft-r4-all-full-seed42-v1/adapter}"
export RUN_ID="${RUN_ID:-dpo-v1p4-rollout-quality-screen800-beta0p1-seed42-from-r4-v1}"
export MAX_STEPS="${MAX_STEPS:-45}"
export SAVE_STEPS="${SAVE_STEPS:-5}"
export EVAL_STEPS="${EVAL_STEPS:-5}"
export LOGGING_STEPS="${LOGGING_STEPS:-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-2}"
export BETA="${BETA:-0.1}"
export SEED="${SEED:-42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec bash scripts/ecommerce/run_1p5b_dpo.sh
