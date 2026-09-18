#!/bin/bash
# SimpleNet class-separated MVTec-AD 학습 실행 스크립트(공식 repo 그대로, 카테고리별 별도 모델).
# 원본 run.sh 대비 변경점 -- 로컬 환경:
#   --gpu 4 -> 0 (이 노트북은 GPU 1개)
#   num_workers 기본값(2) -> 0 (Windows spawn multiprocessing 이슈, DiAD/GLAD/Dinomaly와 동일 이슈)
#   datapath: 기존 MVTec-AD 데이터셋(diad_dataset, DiAD/GLAD/Dinomaly와 동일 폴더) 재사용
# 나머지 하이퍼파라미터(batch=8, meta_epochs=40, gan_epochs=4, image 288 등)는 원 논문 설정 그대로.
DATAPATH='C:/ai_local/diad_dataset'
datasets=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

cd "$(dirname "$0")/../code/SimpleNet"
"/c/Users/kelly/anaconda3/envs/simplenet/python.exe" main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_mvtec \
--log_project MVTecAD_Results \
--results_path results \
--run_name run \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 40 \
--embedding_size 256 \
--gan_epochs 4 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 8 \
--num_workers 0 \
--resize 329 \
--imagesize 288 "${dataset_flags[@]}" mvtec "$DATAPATH"
