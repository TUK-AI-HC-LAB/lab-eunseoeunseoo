"""W39 LabTask #55 - convert VisA into the MVTec directory layout.

The shared framework maps dataset key `visa` to datasets/mvtec.py:MVTecDataset
(component_registry.py:DATASET_REGISTRY), so VisA has to be laid out like MVTec:

    <out>/<category>/train/good/*.JPG
    <out>/<category>/test/good/*.JPG
    <out>/<category>/test/bad/*.JPG
    <out>/<category>/ground_truth/bad/*.png

IMPORTANT: datasets/mvtec.py pairs an image with its mask by *sorted index*, not
by name:

    anomaly_files      = sorted(os.listdir(anomaly_path))
    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
    ... data_tuple.append(maskpaths_per_class[classname][anomaly][i])

so the i-th sorted test/bad image must line up with the i-th sorted
ground_truth/bad mask. We therefore rename both sides to a shared zero-padded
stem (0000.JPG / 0000.png) and verify the pairing afterwards.

Splits come from VisA's official split_csv/1cls.csv (the 1-class protocol).
Files are hard-linked when possible (same NTFS volume, no extra disk), else copied.

Usage:
    python results_w40/w55_visa_to_mvtec.py \
        --src C:/ai_local/glad_dataset/VisA_raw \
        --out C:/ai_local/glad_dataset/VisA_mvtec
"""
import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return "skip"
    try:
        os.link(src, dst)
        return "link"
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="extracted VisA root")
    ap.add_argument("--out", required=True, help="destination in MVTec layout")
    ap.add_argument("--split-csv", default=None,
                    help="default: <src>/split_csv/1cls.csv")
    args = ap.parse_args()

    src, out = args.src, args.out
    split_csv = args.split_csv or os.path.join(src, "split_csv", "1cls.csv")
    if not os.path.exists(split_csv):
        sys.exit(f"split csv not found: {split_csv}")

    # ---- read the official 1-class split
    rows = defaultdict(list)          # (category, split, label) -> [(image, mask)]
    with open(split_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["object"], r["split"], r["label"])].append(
                (r["image"].strip(), (r["mask"] or "").strip()))

    categories = sorted({k[0] for k in rows})
    print(f"categories ({len(categories)}): {', '.join(categories)}")
    print(f"split csv : {split_csv}")
    print(f"src       : {src}")
    print(f"out       : {out}")
    print()

    stats, modes = [], defaultdict(int)

    for cat in categories:
        train_normal = sorted(rows.get((cat, "train", "normal"), []))
        test_normal = sorted(rows.get((cat, "test", "normal"), []))
        test_anom = sorted(rows.get((cat, "test", "anomaly"), []))

        # VisA's 1-class protocol puts no anomalies in train; assert rather than assume
        assert not rows.get((cat, "train", "anomaly")), \
            f"{cat}: unexpected anomalies in train split"

        d_tr = os.path.join(out, cat, "train", "good")
        d_te_good = os.path.join(out, cat, "test", "good")
        d_te_bad = os.path.join(out, cat, "test", "bad")
        d_gt_bad = os.path.join(out, cat, "ground_truth", "bad")
        for d in (d_tr, d_te_good, d_te_bad, d_gt_bad):
            os.makedirs(d, exist_ok=True)

        # ---- train/good and test/good : renamed to a stable zero-padded stem
        for i, (img, _) in enumerate(train_normal):
            modes[link_or_copy(os.path.join(src, img),
                               os.path.join(d_tr, f"{i:04d}.JPG"))] += 1
        for i, (img, _) in enumerate(test_normal):
            modes[link_or_copy(os.path.join(src, img),
                               os.path.join(d_te_good, f"{i:04d}.JPG"))] += 1

        # ---- test/bad + ground_truth/bad : SAME index => sorted order matches
        missing_mask = 0
        for i, (img, mask) in enumerate(test_anom):
            modes[link_or_copy(os.path.join(src, img),
                               os.path.join(d_te_bad, f"{i:04d}.JPG"))] += 1
            if mask:
                modes[link_or_copy(os.path.join(src, mask),
                                   os.path.join(d_gt_bad, f"{i:04d}.png"))] += 1
            else:
                missing_mask += 1

        stats.append({
            "category": cat,
            "train_good": len(train_normal),
            "test_good": len(test_normal),
            "test_bad": len(test_anom),
            "masks": len(test_anom) - missing_mask,
        })
        print(f"  {cat:<12} train/good={len(train_normal):<5} "
              f"test/good={len(test_normal):<4} test/bad={len(test_anom):<4} "
              f"masks={len(test_anom) - missing_mask}")

    # ---- verification: the framework's sorted-index pairing must line up
    print("\nverifying sorted-index image<->mask pairing (as datasets/mvtec.py does)")
    bad = 0
    for cat in categories:
        imgs = sorted(os.listdir(os.path.join(out, cat, "test", "bad")))
        masks = sorted(os.listdir(os.path.join(out, cat, "ground_truth", "bad")))
        if len(imgs) != len(masks):
            print(f"  MISMATCH {cat}: {len(imgs)} images vs {len(masks)} masks")
            bad += 1
            continue
        off = [(a, b) for a, b in zip(imgs, masks)
               if os.path.splitext(a)[0] != os.path.splitext(b)[0]]
        if off:
            print(f"  MISALIGNED {cat}: e.g. {off[0]}")
            bad += 1
    print("  all categories aligned" if bad == 0 else f"  {bad} categories need attention")

    tot = {k: sum(s[k] for s in stats)
           for k in ("train_good", "test_good", "test_bad", "masks")}
    print(f"\ntotals: train/good={tot['train_good']} test/good={tot['test_good']} "
          f"test/bad={tot['test_bad']} masks={tot['masks']}")
    print(f"file ops: {dict(modes)}")
    print(f"\nuse in experiment.yaml:\n  dataset: visa\n  data_path: {out}")


if __name__ == "__main__":
    main()
