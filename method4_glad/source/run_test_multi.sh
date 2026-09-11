#!/bin/bash
# GLAD 체크포인트 평가 실행 스크립트 (multi-category, DiAD와 동일 조건).
# 사용법: CHECKPOINT_STEP=600 bash run_test_multi.sh
# 근거: run_train_multi.sh와 동일한 output_dir 태그를 써야 같은 model/ 폴더를 찾음.
# protocol change: main_multi.py의 eval 분기가 checkpoint_step=20000을 하드코딩하고
# 있어(main_multi.py 부근, MVTec-AD 분기) 중간 체크포인트를 평가할 수 없었음.
# --checkpoint_step_override 인자를 추가해 임의 체크포인트를 지정할 수 있게 패치함
# (세부는 method4_glad/markdown/setup_notes.md).
export CHECKPOINT_STEP="${CHECKPOINT_STEP:?사용법: CHECKPOINT_STEP=600 bash run_test_multi.sh}"
# protocol change: the eval branch picks dataset-specific hyperparameters
# (denoise_step/threshold/checkpoint_step/CLSNAMES) by checking the literal
# substring 'MVTec-AD' in instance_data_dir. Our dataset folder is named
# diad_dataset (inherited from method3_diad, and its name is also baked into
# the training checkpoint folder path via instance_data_dir.split('/')[-1] --
# so it must stay 'diad_dataset' here to match, and the 'MVTec-AD' substring
# check in main_multi.py was patched to also accept 'diad_dataset' instead.
export INSTANCE_DIR='C:/ai_local/diad_dataset'
export ANOMALY_DIR='C:/ai_local/glad_dataset/dtd'
export PRETRAINED='C:/ai_local/glad_models/stable-diffusion-v1-4'
export OUTPUT_DIR='bs1_grad4_eps_anomaly2_multiclass'
export SEED=0

GLAD_SRC="$(cd "$(dirname "$0")/GLAD" && pwd)"
mkdir -p /c/ai_local/glad_run
cd /c/ai_local/glad_run
"/c/Users/kelly/anaconda3/envs/glad/python.exe" -m accelerate.commands.launch --num_processes=1 --mixed_precision=fp16 "$GLAD_SRC/main_multi.py" \
    --pretrained_model_name_or_path="$PRETRAINED" \
    --instance_data_dir="$INSTANCE_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --anomaly_data_dir="$ANOMALY_DIR" \
    --checkpoint_step_override="$CHECKPOINT_STEP" \
    --test_batch_size=2 \
    --dataloader_num_workers=0 \
    --pre_compute_text_embeddings \
    --seed=$SEED
