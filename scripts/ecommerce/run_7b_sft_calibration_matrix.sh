#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-models/base/Qwen2.5-7B-Instruct}"
DATA_ROOT="${DATA_ROOT:-data/ecommerce/domain_train_v1_3_2_zh}"
RUN_SUFFIX="${RUN_SUFFIX:-v1}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-100}"
LOG_ROOT="logs/ecommerce/7b"
mkdir -p "$LOG_ROOT"

ranks=(8 16)
gpus=(0 1)
pids=()
run_ids=()

for index in "${!ranks[@]}"; do
  rank="${ranks[$index]}"
  run_id="sft-r${rank}-all-cal100-seed${SEED}-${RUN_SUFFIX}"
  run_dir="experiments/local/7b/$run_id"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite immutable run directory: $run_dir" >&2
    exit 2
  fi
  run_ids+=("$run_id")
  (
    PLAN_CONFIG=configs/ecommerce/experiments_7b_v1.json \
    MODEL_PATH="$MODEL_PATH" DATA_ROOT="$DATA_ROOT" \
    RUN_ID="$run_id" RUN_DIR="$run_dir" \
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" \
    LORA_RANK="$rank" LORA_ALPHA="$((rank * 2))" TARGET_MODULES=all \
    MICRO_BATCH=2 GRAD_ACCUM=16 MAX_STEPS="$MAX_STEPS" \
    NUM_TRAIN_EPOCHS=1 LEARNING_RATE=1e-5 WARMUP_STEPS=3 \
    SEED="$SEED" DATA_SEED="$SEED" \
    bash scripts/ecommerce/run_1p5b_sft.sh
  ) >"$LOG_ROOT/${run_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "7B SFT calibration completed: ${run_ids[$index]}"
  else
    echo "7B SFT calibration failed: ${run_ids[$index]}" >&2
    status=1
  fi
done
exit "$status"
