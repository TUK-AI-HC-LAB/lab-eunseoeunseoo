"""Identical to main.py, except the wideresnet50 backbone loads ImageNet V1 weights.

backbones.py loads wide_resnet50_2(weights="DEFAULT"), which torchvision resolves to
IMAGENET1K_V2. V2 features are ~9x larger in scale than V1, and SimpleNet's fixed
noise_std=0.015 (tuned for V1) becomes too small to separate real from fake features,
so its discriminator never learns. The official SimpleNet code uses pretrained=True (V1).
Used only for simple; see W40_method_runs/result/README.md.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torchvision.models as models  # noqa: E402

import backbones  # noqa: E402

backbones._BACKBONES["wideresnet50"] = lambda: models.wide_resnet50_2(weights="IMAGENET1K_V1")

import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main.main(sys.argv[1:]))
