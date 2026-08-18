#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 OUTPUT.csv [GPU_INDEX]" >&2
  exit 2
fi

output="$1"
gpu_index="${2:-}"
mkdir -p "$(dirname "$output")"
echo "timestamp,index,name,uuid,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w,temperature_c" > "$output"
command=(nvidia-smi)
if [[ -n "$gpu_index" ]]; then
  command+=(-i "$gpu_index")
fi
exec "${command[@]}" \
  --query-gpu=timestamp,index,name,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits \
  --loop-ms=1000 >> "$output"
