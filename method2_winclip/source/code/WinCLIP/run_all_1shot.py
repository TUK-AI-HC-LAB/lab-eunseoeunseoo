"""WinCLIP+ (1-shot) on all 15 MVTec-AD categories.

Same setup as run_all_zeroshot.py (seed=10, ViT-B-16-plus-240) but with a
single normal reference image added per category, mirroring run_pill_1shot.py's
zero-shot vs 1-shot follow-up, extended to the full category sweep.

Usage: run from this directory with the winclip conda env:
    conda run -n winclip python run_all_1shot.py
"""
import csv
import os
import time

import numpy as np
import torch

import main as winclip_main
from datasets.mvtec_dataset import OBJECT_TYPE

DATA_ROOT = os.environ.get("WINCLIP_DATA_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "No_Submit", "Dataset")
)

if __name__ == "__main__":
    np.random.seed(10)
    torch.manual_seed(10)

    categories = [c for c in OBJECT_TYPE if c != "all"]
    rows = []
    t0 = time.time()

    for cat in categories:
        cat_t0 = time.time()
        config = {
            "datasetname": "mvtec",
            "dataset_root_dir": os.path.dirname(DATA_ROOT),
            "data_dir": DATA_ROOT,
            "obj_type": cat,
            "shot": 1,
        }
        with torch.no_grad(), torch.cuda.amp.autocast():
            gt_list, score_list, auroc, aupr, f1_max = winclip_main.run(config)
        elapsed = time.time() - cat_t0
        print(f"{cat}: I-AUROC={auroc:.4f} AUPR={aupr:.4f} F1-max={f1_max:.4f} ({elapsed:.1f}s)")
        rows.append([cat, 1, auroc, aupr, f1_max, elapsed])

    total_elapsed = time.time() - t0
    print(f"total elapsed: {total_elapsed:.1f}s for {len(categories)} categories")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "mvtec_all_1shot.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "shot", "i_auroc", "aupr", "f1_max", "elapsed_sec"])
        w.writerows(rows)
        mean_auroc = np.mean([r[2] for r in rows])
        mean_aupr = np.mean([r[3] for r in rows])
        mean_f1 = np.mean([r[4] for r in rows])
        w.writerow(["mean", 1, mean_auroc, mean_aupr, mean_f1, total_elapsed])
    print(f"saved: {out_path}")
