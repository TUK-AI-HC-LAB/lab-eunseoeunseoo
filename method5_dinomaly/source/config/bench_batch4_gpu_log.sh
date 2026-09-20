#!/bin/bash
# 피드백 4: batch=4 GPU 로그 재측정(짧은 실행). transistor 1개 카테고리, 200 iter, gradient checkpointing 없음, 10초 간격 GPU 로그.
# 사용: bash bench_batch4_gpu_log.sh   (batch=16 로그와 같은 GPU 로그 형식 + power/clock 열)
OUT="$(dirname "$0")/../result"
LOG="$OUT/bench_batch4_gpu.csv"
echo "timestamp,temp_c,util_pct,mem_used_mib,mem_total_mib,power_w,sm_clock_mhz" > "$LOG"
( while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm --format=csv,noheader,nounits | tr -d ' ')" >> "$LOG"
    sleep 10
  done ) &
MON=$!
cd "$(dirname "$0")/../dinomaly"
"/c/Users/kelly/anaconda3/envs/dinomaly/python.exe" -u bench_sep_batch4.py \
    --data_path 'C:/ai_local/diad_dataset' --save_dir 'C:/ai_local/dinomaly_run/saved_results' \
    --save_name bench_sep_batch4 --items transistor 2>&1 | tee "$OUT/bench_batch4_stdout.txt"
kill $MON
