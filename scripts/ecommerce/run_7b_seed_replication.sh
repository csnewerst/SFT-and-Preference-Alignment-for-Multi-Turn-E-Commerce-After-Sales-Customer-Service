#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

TRAIN_SEED="${TRAIN_SEED:-43}"
PLAN_CONFIG="${PLAN_CONFIG:-configs/ecommerce/experiments_7b_seed43_replication.json}"
SFT_RUN_ID="${SFT_RUN_ID:-sft-r8-all-epoch1-seed${TRAIN_SEED}-replication-v1}"
DPO_RUN_ID="${DPO_RUN_ID:-dpo-v1p4-full720-beta0p1-lr2e6-seed${TRAIN_SEED}-replication-v1}"
SFT_ADAPTER="experiments/local/7b/$SFT_RUN_ID/adapter/checkpoint-200"
DPO_ADAPTER="experiments/local/7b/$DPO_RUN_ID/adapter/checkpoint-5"

if [[ -e "experiments/local/7b/$SFT_RUN_ID" || -e "experiments/local/7b/$DPO_RUN_ID" ]]; then
  echo "Refusing to overwrite an immutable seed replication run" >&2
  exit 2
fi

PLAN_CONFIG="$PLAN_CONFIG" TRAIN_SEED="$TRAIN_SEED" DATA_SEED="$TRAIN_SEED" LORA_RANK=8 RUN_ID="$SFT_RUN_ID" \
  bash scripts/ecommerce/run_7b_sft_main.sh

if [[ ! -f "$SFT_ADAPTER/adapter_config.json" ]]; then
  echo "Missing pre-registered SFT checkpoint-200: $SFT_ADAPTER" >&2
  exit 2
fi

PLAN_CONFIG="$PLAN_CONFIG" TRAIN_SEED="$TRAIN_SEED" SFT_ADAPTER="$SFT_ADAPTER" RUN_ID="$DPO_RUN_ID" \
  bash scripts/ecommerce/run_7b_dpo_calibration.sh

if [[ ! -f "$DPO_ADAPTER/adapter_config.json" ]]; then
  echo "Missing pre-registered DPO checkpoint-5: $DPO_ADAPTER" >&2
  exit 2
fi

PLAN_CONFIG="$PLAN_CONFIG" TRAIN_SEED="$TRAIN_SEED" SFT_ADAPTER="$SFT_ADAPTER" DPO_ADAPTER="$DPO_ADAPTER" \
  bash scripts/ecommerce/run_7b_seed_replication_eval.sh
