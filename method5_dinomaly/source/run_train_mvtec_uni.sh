#!/bin/bash
# Dinomaly multi-class(uni) MVTec-AD 학습 실행 스크립트.
# protocol change 세부는 dinomaly_mvtec_uni.py 상단 주석과 method5_dinomaly/markdown/setup_notes.md 참고:
#   - batch_size 16->4 (원 저자 RTX 3090 24GB 기준 -> 로컬 8GB GPU)
#   - num_workers 4->0 (Windows spawn multiprocessing 이슈, DiAD/GLAD와 동일)
#   - device cuda:1 -> cuda:0 (원 코드가 멀티 GPU 두 번째 장치를 하드코딩, 이 노트북은 GPU 1개)
# 기존 MVTec-AD 데이터셋(diad_dataset, DiAD 재현 때 받은 것)을 그대로 재사용.
DATA_PATH='C:/ai_local/diad_dataset'
SAVE_DIR='C:/ai_local/dinomaly_run/saved_results'

cd "$(dirname "$0")/dinomaly"
"/c/Users/kelly/anaconda3/envs/dinomaly/python.exe" dinomaly_mvtec_uni.py \
    --data_path "$DATA_PATH" \
    --save_dir "$SAVE_DIR" \
    --save_name "dinomaly_mvtec_uni_b4_local8gb"
