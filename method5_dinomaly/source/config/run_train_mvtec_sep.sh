#!/bin/bash
# Dinomaly class-separated(sep) MVTec-AD 학습 실행 스크립트 — 15개 카테고리 순차 학습(카테고리당 5000 iter, 원 논문 설정 batch=16 그대로).
# 로컬 하드웨어 대응은 dinomaly_mvtec_uni.py와 동일 패턴을 dinomaly_mvtec_sep.py에도 적용:
#   - num_workers 4->0 (Windows spawn multiprocessing 이슈)
#   - device cuda:1 -> cuda:0 (이 노트북은 GPU 1개)
# batch_size=16은 원 논문 설정 그대로(스크립트 하드코딩) — uni 학습 때 batch=16으로 VRAM 7.8GB선에서 안정적으로 돈 것 확인됨.
# 기존 MVTec-AD 데이터셋(diad_dataset, DiAD 재현 때 받은 것)을 그대로 재사용.
DATA_PATH='C:/ai_local/diad_dataset'
SAVE_DIR='C:/ai_local/dinomaly_run/saved_results'

cd "$(dirname "$0")/../dinomaly"
"/c/Users/kelly/anaconda3/envs/dinomaly/python.exe" dinomaly_mvtec_sep.py \
    --data_path "$DATA_PATH" \
    --save_dir "$SAVE_DIR" \
    --save_name "dinomaly_mvtec_sep_b16_v2"
