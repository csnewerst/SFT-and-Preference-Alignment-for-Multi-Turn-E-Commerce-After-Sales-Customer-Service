#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-7B-Instruct}"
export CONFIG_PATH="${CONFIG_PATH:-configs/ecommerce/experiments_7b_v1.json}"
export EXPERIMENT_SCALE=7b
export RUN_ID="${RUN_ID:-initial-screen-v1}"

exec bash scripts/ecommerce/run_1p5b_initial_screen.sh
