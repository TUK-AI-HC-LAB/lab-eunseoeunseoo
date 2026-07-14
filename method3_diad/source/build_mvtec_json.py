"""Build train.json / test.json for DiAD's MVTecDataset from the local MVTec-AD layout
in No_Submit/Dataset/<category>/{train,test,ground_truth}/...

DiAD's mvtecad_dataloader.py reads these files as JSON-Lines relative to
`./training/MVTec-AD/`, and joins `data_path` (train.py) with each entry's
"filename"/"maskname" to build the actual image path. So filenames here are
written relative to DATA_ROOT below, not to this script's location.
"""
import json
import os

DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "No_Submit", "Dataset")
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "DiAD", "training", "MVTec-AD")

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


def list_images(dir_path):
    if not os.path.isdir(dir_path):
        return []
    return sorted(f for f in os.listdir(dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg")))


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_entries = []
    test_entries = []

    for cls in CATEGORIES:
        cls_dir = os.path.join(DATA_ROOT, cls)
        if not os.path.isdir(cls_dir):
            print(f"skip {cls}: not found under {DATA_ROOT}")
            continue

        for fname in list_images(os.path.join(cls_dir, "train", "good")):
            train_entries.append({
                "filename": f"{cls}/train/good/{fname}",
                "label": 0,
                "clsname": cls,
            })

        test_dir = os.path.join(cls_dir, "test")
        for defect_type in sorted(os.listdir(test_dir)):
            defect_dir = os.path.join(test_dir, defect_type)
            if not os.path.isdir(defect_dir):
                continue
            is_good = defect_type == "good"
            for fname in list_images(defect_dir):
                entry = {
                    "filename": f"{cls}/test/{defect_type}/{fname}",
                    "label": 0 if is_good else 1,
                    "clsname": cls,
                }
                if not is_good:
                    stem, _ = os.path.splitext(fname)
                    mask_name = f"{cls}/ground_truth/{defect_type}/{stem}_mask.png"
                    if os.path.isfile(os.path.join(DATA_ROOT, mask_name)):
                        entry["maskname"] = mask_name
                test_entries.append(entry)

    train_path = os.path.join(OUT_DIR, "train.json")
    test_path = os.path.join(OUT_DIR, "test.json")
    with open(train_path, "w") as f:
        for e in train_entries:
            f.write(json.dumps(e) + "\n")
    with open(test_path, "w") as f:
        for e in test_entries:
            f.write(json.dumps(e) + "\n")

    print(f"data root: {DATA_ROOT}")
    print(f"train.json: {len(train_entries)} entries -> {train_path}")
    print(f"test.json:  {len(test_entries)} entries -> {test_path}")


if __name__ == "__main__":
    build()
