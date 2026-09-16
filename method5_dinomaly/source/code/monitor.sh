#!/usr/bin/env bash
# Usage: ./monitor.sh <label> <interval_seconds>
# Logs GPU temp/util/VRAM at a fixed interval to ../result/<label>.csv until killed.
# Restarting with the same label appends to the existing file instead of overwriting it
# (safe to stop/resume without losing prior data).
LABEL="${1:?label required, e.g. batch16_default_fan}"
INTERVAL="${2:-10}"
OUT="$(dirname "$0")/../result/${LABEL}.csv"
if [ ! -f "$OUT" ]; then
  echo "timestamp,temp_c,util_pct,mem_used_mib,mem_total_mib" > "$OUT"
fi
while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -v ts="$ts" -F', *' '{print ts","$1","$2","$3","$4}' >> "$OUT"
  sleep "$INTERVAL"
done
