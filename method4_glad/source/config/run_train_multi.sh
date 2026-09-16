#!/bin/bash
# GLAD multi-category(=DiAD와 동일 조건) 학습 실행 스크립트.
# 원본 train_multi.sh(공식 repo) 대비 변경점 -- 로컬 8GB GPU 제약:
#   - train_batch_size 32 -> 1, gradient_accumulation_steps 1 -> 4 (원 effective batch 32를
#     그대로 맞추면 optimizer step 하나에 32 x ~5s ≈ 160s -- DiAD 때와 같은 이유로 "논문과
#     동일한 학습량"보다 "판단 가능한 시점을 실제로 도달하는 것"을 우선해 축소)
#   - dataloader_num_workers 기본값(8) -> 0 (Windows spawn multiprocessing 버그, DiAD와 동일 이슈)
#   - pretrained_model_name_or_path를 HF 허브 대신 로컬 다운로드 경로로 변경
#   - instance_data_dir: DiAD 재현 때 받은 MVTec-AD를 그대로 재사용(동일 폴더 구조)
#   - anomaly_data_dir: DTD 데이터셋 로컬 경로
#   - output_dir: OneDrive 동기화 폴더 밖(체크포인트 용량 문제, DiAD와 동일 이슈)
# protocol change: main_multi.py hardcodes checkpoint output to a *relative*
# './model/<...>' path (line 529, os.path.join('model', ...)) -- it ignores
# --output_dir as an absolute path and instead uses it as a short suffix tag.
# Running with cwd inside the OneDrive-synced repo would put multi-GB
# checkpoints under OneDrive sync scope (same stall issue hit in DiAD).
# Fix: cd into a local-only run dir before launching so the relative
# './model/' lands in C:/ai_local instead.
export INSTANCE_DIR='C:/ai_local/diad_dataset'
export ANOMALY_DIR='C:/ai_local/glad_dataset/dtd'
export PRETRAINED='C:/ai_local/glad_models/stable-diffusion-v1-4'
export OUTPUT_DIR='bs1_grad4_eps_anomaly2_multiclass'
export DENOISE_STEP=500
export MAX_TRAIN_STEP=20000
export SEED=0

GLAD_SRC="$(cd "$(dirname "$0")/../GLAD" && pwd)"
mkdir -p /c/ai_local/glad_run
cd /c/ai_local/glad_run
"/c/Users/kelly/anaconda3/envs/glad/python.exe" -m accelerate.commands.launch --num_processes=1 --mixed_precision=fp16 "$GLAD_SRC/main_multi.py" \
    --train=True \
    --pretrained_model_name_or_path="$PRETRAINED" \
    --instance_data_dir="$INSTANCE_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --anomaly_data_dir="$ANOMALY_DIR" \
    --checkpointing_steps=200 \
    --denoise_step=$DENOISE_STEP \
    --instance_prompt="a photo of sks" \
    --resolution=256 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --use_8bit_adam \
    --mixed_precision="fp16" \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=$MAX_TRAIN_STEP \
    --dataloader_num_workers=0 \
    --pre_compute_text_embeddings \
    --seed=$SEED
