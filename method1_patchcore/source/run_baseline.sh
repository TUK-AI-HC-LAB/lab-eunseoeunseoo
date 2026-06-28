#!/bin/bash
# Run PatchCore baseline on all 15 MVTec AD categories.
# Usage: bash run_baseline.sh
# Requires: conda activate patchcore, run from source/ directory

# Set this to your local MVTec AD root (the folder containing bottle/, cable/, etc.)
DATAPATH="/path/to/mvtec"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR/patchcore-inspection"

python bin/run_patchcore.py \
  --gpu 0 --seed 0 \
  --log_group baseline_all \
  results \
  patch_core \
    -b wideresnet50 \
    -le layer2 -le layer3 \
    --pretrain_embed_dimension 1024 \
    --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 \
    --patchsize 3 \
  sampler \
    -p 0.1 approx_greedy_coreset \
  dataset \
    --resize 256 --imagesize 224 \
    -d bottle -d cable -d capsule -d carpet -d grid \
    -d hazelnut -d leather -d metal_nut -d pill -d screw \
    -d tile -d toothbrush -d transistor -d wood -d zipper \
    mvtec "$DATAPATH"