#!/bin/bash
# H3 4차 재평가(epoch 34)용 평가 실행 스크립트. run_eval_epoch7/16/24.sh와 동일한 test.py, 체크포인트만 교체.
# 결과: method3_diad/source/result/eval_results_epoch34.csv
cd "$(dirname "$0")/../code/DiAD"
"/c/Users/kelly/anaconda3/envs/diad/python.exe" test.py \
  --resume_path "C:/ai_local/diad_val_ckpt/step_step=15650.ckpt"
