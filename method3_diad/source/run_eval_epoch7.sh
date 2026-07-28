#!/bin/bash
# H3/H4 검증용 평가 실행 스크립트.
# 결과: method3_diad/source/eval_results_epoch7.csv (수동으로 log에서 옮겨 적음)
# 분석: method3_diad/markdown/h3_h4_evaluation.md
cd "$(dirname "$0")/DiAD"
"/c/Users/kelly/anaconda3/envs/diad/python.exe" test.py \
  --resume_path "C:/ai_local/diad_val_ckpt/step_step=3600.ckpt"
