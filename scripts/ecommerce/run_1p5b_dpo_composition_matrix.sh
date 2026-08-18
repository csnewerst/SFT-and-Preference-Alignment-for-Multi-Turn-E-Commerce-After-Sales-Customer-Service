#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

SFT_RUN_ID="${SFT_RUN_ID:?Set SFT_RUN_ID to the gate-selected immutable SFT run}"
SFT_TAG="${SFT_TAG:?Set SFT_TAG to a short identifier such as r4-all or r16-all}"
DATA_VARIANTS_ROOT="${DATA_VARIANTS_ROOT:-data/ecommerce/domain_train_v1_3_2_zh_dpo_ablation_1p5b}"
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

variants=(
  "response_only_matched"
  "multigranularity_matched"
  "multigranularity_full"
)
run_names=(
  "response-only-matched"
  "multigranularity-matched"
  "multigranularity-full"
)
max_steps=(99 99 278)
gpus=(0 1 2)
pids=()
mkdir -p logs/ecommerce/1p5b

for index in "${!variants[@]}"; do
  run_id="dpo-${run_names[$index]}-beta0p1-seed${SEED}-from-${SFT_TAG}-${RUN_SUFFIX}"
  run_dir="experiments/local/1p5b/$run_id"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite immutable run directory: $run_dir" >&2
    exit 2
  fi
  (
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" \
    PYTHON_BIN="$PYTHON_BIN" \
    DATA_ROOT="$DATA_VARIANTS_ROOT/${variants[$index]}" \
    SFT_ADAPTER="$SFT_ADAPTER" \
    RUN_ID="$run_id" \
    MAX_STEPS="${max_steps[$index]}" \
    BETA=0.1 \
    SEED="$SEED" \
      bash scripts/ecommerce/run_1p5b_dpo.sh
  ) >"logs/ecommerce/1p5b/${run_id}.launcher.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "DPO composition run completed ${run_names[$index]}"
  else
    echo "DPO composition run failed ${run_names[$index]}" >&2
    status=1
  fi
done
exit "$status"
