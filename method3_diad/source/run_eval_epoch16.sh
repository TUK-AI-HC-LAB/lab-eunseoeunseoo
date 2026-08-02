#!/bin/bash
# H3 재평가(epoch 16)용 평가 실행 스크립트. run_eval_epoch7.sh와 동일한 test.py, 체크포인트만 교체.
# 결과: method3_diad/source/eval_results_epoch16.csv
# 분석: method3_diad/markdown/h3_h4_evaluation.md
cd "$(dirname "$0")/DiAD"
"/c/Users/kelly/anaconda3/envs/diad/python.exe" test.py \
  --resume_path "C:/ai_local/diad_val_ckpt/step_step=7400.ckpt"
