#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-7B-Instruct}"
export EXPERIMENT_SCALE=7b
export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-5 10 20}"
export RUN_ID="${RUN_ID:?Set RUN_ID to the 7B DPO calibration run}"

exec bash scripts/ecommerce/run_1p5b_dpo_v1p4_checkpoint_eval.sh
