#!/bin/bash
# Run WinCLIP+ (1-shot) on all 15 MVTec-AD categories, on THIS machine.
# See run_all_zeroshot_local.sh for why this wrapper exists (path override
# instead of editing run_all_1shot.py's default).
# Usage: bash run_all_1shot_local.sh
# Requires: conda env "winclip" (C:/ai_local/miniconda3/envs/winclip)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../code/WinCLIP"

CONDA_ENV_PYTHON="/c/ai_local/miniconda3/envs/winclip/python.exe"
export WINCLIP_DATA_ROOT="C:/ai_local/glad_dataset/MVTec-AD"

"$CONDA_ENV_PYTHON" run_all_1shot.py
