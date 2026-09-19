#!/bin/bash
# Reverse Distillation(공식 코드 저장소 이름은 RD4AD) class-separated MVTec-AD 학습 실행 스크립트(공식 repo 그대로, 카테고리별 별도 모델).
# 원 논문/공식 코드는 multi-class 옵션이 없음 -- GLAD/Dinomaly(multi-class)와 protocol이 다름을 감안하고 참고치로 사용.
# 원본 main.py 대비 변경점 -- 로컬 환경:
#   - './mvtec/<cls>' 하드코딩 경로 -> --data_path 인자로 실제 데이터셋(glad_dataset, 다른 방법들과 동일 폴더) 사용
#   - './checkpoints/' 하드코딩 -> --ckpt_dir 인자
#   - device 'cuda' 고정 -> --gpu 인자(cuda:N)
#   - 카테고리 학습 완료 시 result_csv에 기록, 이미 기록된 카테고리는 재실행 시 스킵(크래시 후 재개용)
#   - test.py: np.bool(제거됨)->bool, DataFrame.append(제거됨)->list+concat로 교체(numpy/pandas 신버전 호환)
# 나머지 하이퍼파라미터(batch=16, epochs=200, lr=0.005, encoder=wide_resnet50_2, image=256)는 원 논문 설정 그대로.
DATA_PATH='C:/ai_local/glad_dataset/MVTec-AD'
CKPT_DIR='C:/ai_local/rd4ad_run/checkpoints'
RESULT_CSV="$(dirname "$0")/../result/eval_results_sep.csv"

cd "$(dirname "$0")/../code/RD4AD"
"/c/ai_local/miniconda3/envs/rd4ad/python.exe" main.py \
    --data_path "$DATA_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --result_csv "$RESULT_CSV" \
    --gpu 0
