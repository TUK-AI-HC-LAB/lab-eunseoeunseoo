#!/bin/bash
# 피드백 4: batch=16(gradient checkpointing 없음) 짧은 재현 — VRAM 전용/공유 메모리 원시 카운터 + nvidia-smi 10초 간격 로그.
# transistor 1개 카테고리, 30 iter. 사용: bash bench_batch16_gpu_log.sh
OUT="$(dirname "$0")/../result"
SMI="$OUT/bench_batch16_gpu.csv"
CTR="$OUT/bench_batch16_gpu_memory_counter.txt"
echo "timestamp,temp_c,util_pct,mem_used_mib,mem_total_mib,power_w,sm_clock_mhz" > "$SMI"
: > "$CTR"
( while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm --format=csv,noheader,nounits | tr -d ' ')" >> "$SMI"
    { echo "== $(date '+%Y-%m-%d %H:%M:%S')"; powershell.exe -NoProfile -Command "(Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage').CounterSamples | Where-Object { \$_.CookedValue -gt 50MB } | ForEach-Object { '{0}  {1:N0} MiB' -f \$_.Path, (\$_.CookedValue/1MB) }"; } >> "$CTR" 2>&1
    sleep 4
  done ) &
MON=$!
cd "$(dirname "$0")/../dinomaly"
"/c/Users/kelly/anaconda3/envs/dinomaly/python.exe" -u bench_sep_batch16.py \
    --data_path 'C:/ai_local/diad_dataset' --save_dir 'C:/ai_local/dinomaly_run/saved_results' \
    --save_name bench_sep_batch16 --items transistor 2>&1 | tee "$OUT/bench_batch16_stdout.txt"
kill $MON
