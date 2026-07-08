#!/bin/bash
# Run WinCLIP+ (1-shot) on MVTec AD pill category only (H2 few-shot follow-up).
# Usage: bash run_pill_1shot.sh
# Requires: conda env "patchcore" (torch+cuda already installed there)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/WinCLIP"

# conda is not reliably on PATH in this shell; call the env's python directly.
CONDA_ENV_PYTHON="/c/Users/kelly/anaconda3/envs/patchcore/python.exe"
"$CONDA_ENV_PYTHON" run_pill_1shot.py
