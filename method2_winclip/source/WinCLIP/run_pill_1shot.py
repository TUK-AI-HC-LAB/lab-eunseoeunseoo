"""WinCLIP+ (1-shot) on MVTec AD pill category only.

Follow-up to run_pill_zeroshot.py: zero-shot WinCLIP refuted H2 on pill
(I-AUROC 0.812 vs PatchCore 0.968). This checks whether adding a single
normal reference image (language + minimal visual reference) closes
the gap.

Usage: run from this directory with the patchcore conda env:
    conda run -n patchcore python run_pill_1shot.py
"""
import csv
import os

import numpy as np
import torch

import main as winclip_main

DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "No_Submit", "Dataset")
)

if __name__ == "__main__":
    np.random.seed(10)
    torch.manual_seed(10)

    config = {
        "datasetname": "mvtec",
        "dataset_root_dir": os.path.dirname(DATA_ROOT),
        "data_dir": DATA_ROOT,
        "obj_type": "pill",
        "shot": 1,
    }

    with torch.no_grad(), torch.cuda.amp.autocast():
        gt_list, score_list, auroc, aupr, f1_max = winclip_main.run(config)

    print(f"pill 1-shot: I-AUROC={auroc:.4f} AUPR={aupr:.4f} F1-max={f1_max:.4f}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "pill_1shot.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "shot", "i_auroc", "aupr", "f1_max"])
        w.writerow(["pill", 1, auroc, aupr, f1_max])
    print(f"saved: {out_path}")
