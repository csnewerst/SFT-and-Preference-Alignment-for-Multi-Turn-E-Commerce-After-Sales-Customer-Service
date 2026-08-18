#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"
RUN_SUFFIX="${RUN_SUFFIX:-v1}"
mkdir -p logs/ecommerce/1p5b

run_ids=(
  "sft-r4-all-full-seed${SEED}-${RUN_SUFFIX}"
  "sft-r16-all-full-seed${SEED}-${RUN_SUFFIX}"
  "sft-r64-all-full-seed${SEED}-${RUN_SUFFIX}"
  "sft-r16-qv-full-seed${SEED}-${RUN_SUFFIX}"
)
gpus=(0 1 2 3)
ranks=(4 16 64 16)
alphas=(8 32 128 32)
targets=(all all all q_proj,v_proj)

pids=()
for index in "${!run_ids[@]}"; do
  run_id="${run_ids[$index]}"
  run_dir="experiments/local/1p5b/$run_id"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite immutable run directory: $run_dir" >&2
    exit 2
  fi
  echo "Launching $run_id on GPU ${gpus[$index]}"
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" \
    RUN_ID="$run_id" \
    LORA_RANK="${ranks[$index]}" \
    LORA_ALPHA="${alphas[$index]}" \
    TARGET_MODULES="${targets[$index]}" \
    SEED="$SEED" DATA_SEED="$DATA_SEED" MAX_STEPS=-1 \
    bash scripts/ecommerce/run_1p5b_sft.sh \
    >"logs/ecommerce/1p5b/${run_id}.launcher.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed ${run_ids[$index]}"
  else
    echo "Failed ${run_ids[$index]}" >&2
    status=1
  fi
done
exit "$status"
