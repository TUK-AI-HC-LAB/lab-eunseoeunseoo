"""W39 LabTask #55 - probe every registered method for import/construction blockers.

For each of the 11 methods this tries, in order:
  1. import the trainer module            (component_registry MethodSpec.load_class)
  2. validate the 5-method Trainer contract
  3. load the backbone
  4. construct the trainer and call .load(...)   <- where most model code runs

and records exactly where it stops and why. Nothing is trained; this is a
static-ish feasibility sweep so that "실행하지 못한 범위와 이유" is evidence-backed.

Run from the codebase root:
    ./.venv-cpu/Scripts/python.exe results_w40/w55_probe_methods.py
"""
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results_w40", "capture")
os.makedirs(OUT, exist_ok=True)

import torch

import utils
from component_registry import get_method_spec, list_methods
from main import resolve_args

lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def short(exc):
    """One-line reason, keeping the bit that names the missing thing."""
    name = type(exc).__name__
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{name}: {msg}"[:160]


device = torch.device("cpu")
results = []

P("=" * 110)
P("W55 method feasibility probe  (import -> contract -> backbone -> construct+load)")
P("=" * 110)
P(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  device={device}")
P("")

for method in list_methods():
    row = {"method": method, "import": "-", "contract": "-",
           "backbone": "-", "construct": "-", "reason": ""}
    spec = get_method_spec(method)

    # --- build an args namespace for this method (picks up configs/<method>.yaml)
    try:
        args = resolve_args(["--method", method,
                             "--data-path", "C:/ai_local/glad_dataset/MVTec-AD",
                             "--category", "bottle"])
        args.gpu = []
    except Exception as exc:
        row["import"] = "FAIL"
        row["reason"] = f"resolve_args: {short(exc)}"
        results.append(row)
        P(f"[{method:<10}] args   FAIL  {row['reason']}")
        continue

    # 1. import module + 2. contract check (load_class does both)
    try:
        trainer_class = spec.load_class()
        row["import"] = "ok"
        row["contract"] = "ok"
    except Exception as exc:
        row["import"] = "FAIL"
        row["reason"] = short(exc)
        results.append(row)
        P(f"[{method:<10}] import FAIL  {row['reason']}")
        continue

    # 3. backbone
    try:
        backbone = spec.load_backbone(args.backbone_names[0])
        backbone.name = args.backbone_names[0]
        backbone.seed = None
        row["backbone"] = "ok"
    except Exception as exc:
        row["backbone"] = "FAIL"
        row["reason"] = short(exc)
        results.append(row)
        P(f"[{method:<10}] backbone FAIL  {row['reason']}")
        continue

    # 4. construct + load  (this is where each method builds its real modules)
    try:
        input_shape = (3, args.imagesize, args.imagesize)
        context = {"args": args, "input_shape": input_shape,
                   "backbone_name": args.backbone_names[0],
                   "layers_to_extract_from": list(args.layers_to_extract_from)}
        trainer = spec.create(device, context)
        trainer.load(
            backbone=backbone,
            layers_to_extract_from=list(args.layers_to_extract_from),
            device=device, input_shape=input_shape,
            pretrain_embed_dimension=args.pretrain_embed_dimension,
            target_embed_dimension=args.target_embed_dimension,
            patchsize=args.patchsize, meta_epochs=args.meta_epochs,
            aed_meta_epochs=args.aed_meta_epochs, gan_epochs=args.gan_epochs,
            noise_std=args.noise_std, dsc_layers=args.dsc_layers,
            dsc_hidden=args.dsc_hidden, dsc_margin=args.dsc_margin,
            dsc_lr=args.dsc_lr, auto_noise=args.auto_noise,
            train_backbone=args.train_backbone, cos_lr=args.cos_lr,
            pre_proj=args.pre_proj, proj_layer_type=args.proj_layer_type,
            mix_noise=args.mix_noise, onnx=args.onnx, args=args,
        )
        row["construct"] = "ok"
    except Exception as exc:
        row["construct"] = "FAIL"
        row["reason"] = short(exc)
        with open(os.path.join(OUT, f"w55_traceback_{method}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(traceback.format_exc())
    results.append(row)
    state = row["construct"]
    P(f"[{method:<10}] import ok  backbone ok  construct {state}"
      + (f"  {row['reason']}" if row["reason"] else ""))

# ---------------------------------------------------------------- summary
P("")
P("=" * 110)
P("SUMMARY")
P("=" * 110)
P(f"  {'method':<11}{'import':<9}{'contract':<11}{'backbone':<11}{'construct':<11}reason")
P("  " + "-" * 106)
for r in results:
    P(f"  {r['method']:<11}{r['import']:<9}{r['contract']:<11}"
      f"{r['backbone']:<11}{r['construct']:<11}{r['reason']}")

ready = [r["method"] for r in results if r["construct"] == "ok"]
blocked = [r for r in results if r["construct"] != "ok"]
P("")
P(f"  constructible now ({len(ready)}/11): {', '.join(ready) or '(none)'}")
P(f"  blocked ({len(blocked)}/11):")
for r in blocked:
    P(f"    {r['method']:<11} {r['reason']}")
P("")
P("  NOTE: 'construct ok' means the model builds and .load() completes.")
P("        It does NOT prove the method trains to completion or fits in VRAM.")

with open(os.path.join(OUT, "w55_method_probe.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "w55_method_probe.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nlog ->", os.path.join(OUT, "w55_method_probe.txt"))
