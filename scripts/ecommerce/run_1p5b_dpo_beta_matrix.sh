#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

SFT_RUN_ID="${SFT_RUN_ID:?Set SFT_RUN_ID to the gate-selected immutable SFT run}"
SFT_TAG="${SFT_TAG:?Set SFT_TAG to the selected SFT tag}"
DATA_ROOT="${DATA_ROOT:-data/ecommerce/domain_train_v1_3_2_zh_dpo_ablation_1p5b/multigranularity_full}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SEED="${SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:-v1}"
SFT_ADAPTER="experiments/local/1p5b/$SFT_RUN_ID/adapter"

if [[ ! "$SFT_TAG" =~ ^[a-z0-9-]+$ ]]; then
  echo "SFT_TAG must contain only lowercase letters, digits, and hyphens" >&2
  exit 2
fi
if [[ ! -f "$SFT_ADAPTER/adapter_config.json" ]]; then
  echo "Missing selected SFT adapter: $SFT_ADAPTER" >&2
  exit 2
fi

betas=(0.05 0.3)
beta_tags=(0p05 0p3)
gpus=(0 1)
pids=()
mkdir -p logs/ecommerce/1p5b

for index in "${!betas[@]}"; do
  run_id="dpo-multigranularity-full-beta${beta_tags[$index]}-seed${SEED}-from-${SFT_TAG}-${RUN_SUFFIX}"
  run_dir="experiments/local/1p5b/$run_id"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite immutable run directory: $run_dir" >&2
    exit 2
  fi
  (
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" \
    PYTHON_BIN="$PYTHON_BIN" \
    DATA_ROOT="$DATA_ROOT" \
    SFT_ADAPTER="$SFT_ADAPTER" \
    RUN_ID="$run_id" \
    MAX_STEPS=278 \
    BETA="${betas[$index]}" \
    SEED="$SEED" \
      bash scripts/ecommerce/run_1p5b_dpo.sh
  ) >"logs/ecommerce/1p5b/${run_id}.launcher.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "DPO beta run completed ${betas[$index]}"
  else
    echo "DPO beta run failed ${betas[$index]}" >&2
    status=1
  fi
done
exit "$status"
