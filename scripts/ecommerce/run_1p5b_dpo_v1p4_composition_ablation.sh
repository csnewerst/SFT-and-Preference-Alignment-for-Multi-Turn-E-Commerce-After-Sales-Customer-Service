#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

SOURCE_ROOT="${SOURCE_ROOT:-data/ecommerce/dpo_v1_4_rollout_quality_screen_800_v2}"
VARIANT_ROOT="${VARIANT_ROOT:-data/ecommerce/dpo_v1_4_rollout_quality_composition_matched_v1}"
SFT_ADAPTER="${SFT_ADAPTER:-experiments/local/1p5b/sft-r4-all-full-seed42-v1/adapter}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RUN_SUFFIX="${RUN_SUFFIX:-v1}"

if [[ ! -e "$VARIANT_ROOT" ]]; then
  "$PYTHON_BIN" scripts/ecommerce/build_1p5b_dpo_variants.py \
    --input-root "$SOURCE_ROOT" \
    --output-root "$VARIANT_ROOT" \
    --seed 20260809
fi

variants=(response_only_matched multigranularity_matched)
gpus=(0 1)
pids=()
run_ids=()
mkdir -p logs/ecommerce/1p5b

for index in "${!variants[@]}"; do
  variant="${variants[$index]}"
  tag="${variant%_matched}"
  tag="${tag//_/-}"
  run_id="dpo-v1p4-${tag}-matched-n173-beta0p1-seed42-${RUN_SUFFIX}"
  run_dir="experiments/local/1p5b/$run_id"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite immutable run directory: $run_dir" >&2
    exit 2
  fi
  run_ids+=("$run_id")
  (
    PLAN_CONFIG=configs/ecommerce/dpo_v1_4_composition_ablation.json \
    DATA_ROOT="$VARIANT_ROOT/$variant" \
    SFT_ADAPTER="$SFT_ADAPTER" RUN_ID="$run_id" \
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" \
    MAX_STEPS=20 SAVE_STEPS=5 EVAL_STEPS=5 LOGGING_STEPS=5 \
    WARMUP_STEPS=1 BETA=0.1 SEED=42 \
    bash scripts/ecommerce/run_1p5b_dpo.sh
  ) >"logs/ecommerce/1p5b/${run_id}.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Composition ablation training completed: ${run_ids[$index]}"
  else
    echo "Composition ablation training failed: ${run_ids[$index]}" >&2
    status=1
  fi
done
exit "$status"
