#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_SUFFIX="${RUN_SUFFIX:-v1}"
variants=(response-only multigranularity)

for variant in "${variants[@]}"; do
  RUN_ID="dpo-v1p4-${variant}-matched-n173-beta0p1-seed42-${RUN_SUFFIX}" \
  CHECKPOINT_STEPS="5 10 20" \
    bash scripts/ecommerce/run_1p5b_dpo_v1p4_checkpoint_eval.sh
done
