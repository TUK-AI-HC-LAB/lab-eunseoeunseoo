#!/bin/bash
# Run zero-shot WinCLIP on all 15 MVTec-AD categories, on THIS machine.
# (run_all_zeroshot.py's default path was written for the other machine's
#  No_Submit/Dataset layout; this wrapper points it at the local dataset
#  via WINCLIP_DATA_ROOT instead of editing that default.)
# Usage: bash run_all_zeroshot_local.sh
# Requires: conda env "winclip" (created under C:/ai_local/miniconda3/envs/winclip
#            — torch 2.11.0+cu128 to match this machine's RTX 5070)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../code/WinCLIP"

CONDA_ENV_PYTHON="/c/ai_local/miniconda3/envs/winclip/python.exe"
export WINCLIP_DATA_ROOT="C:/ai_local/glad_dataset/MVTec-AD"

"$CONDA_ENV_PYTHON" run_all_zeroshot.py
