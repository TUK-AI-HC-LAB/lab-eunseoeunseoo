# method1_patchcore — Source

## Environment Setup

```bash
conda activate patchcore
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install timm
cd patchcore-inspection
pip install -e .
pip install -r requirements.txt
```

## Dataset

Download MVTec AD from: https://www.mvtec.com/company/research/datasets/mvtec-ad

Extract so that the folder structure looks like:

```
<your_datapath>/
├── bottle/
├── cable/
├── capsule/
├── ...
└── zipper/
```

Then set `DATAPATH` in `run_baseline.sh` to your local path.

## Running the Baseline

```bash
# Run from source/ directory, with patchcore env active
bash run_baseline.sh
```

Results will be saved to `results/baseline_20260628.csv`.