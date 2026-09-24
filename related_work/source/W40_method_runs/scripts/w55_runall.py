"""W39 LabTask #55 - run everything that is left, no wall-clock deadline.

There is no overall time limit here: the machine will simply become unavailable
at some point, and whatever has finished by then is kept. Every step is
resumable (w55_run_batch.py skips jobs already recorded as ok/failed and retries
timeouts), so this can be relaunched at any time and will pick up where it
stopped.

Ordering is by scarcity, not by cost:

  1. dinomaly            peaks ~11.8GB -> the 8GB laptop cannot run it at all,
                         so this machine is the only place it can happen.
  2. MVTec, everything else
  3. VisA, everything

Within a method total_iter / meta_epochs is held fixed so per-category numbers
stay comparable; the values match what is already recorded for that method.

Per-job caps are still present (a hung job must not block the queue) but are set
well above the observed worst case rather than tuned to a budget.
"""
import argparse
import datetime as dt
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv-gpu", "Scripts", "python.exe")
RUNNER = os.path.join(ROOT, "results_w40", "w55_run_batch.py")
LOG = os.path.join(ROOT, "results_w55", "day_plan.log")

# categories dinomaly still owes, cheapest-risk first
DINO_UNATTEMPTED = "tile,transistor,wood,zipper"
DINO_TIMED_OUT = "capsule,leather,metal_nut,pill,toothbrush"


def log(msg):
    line = f"[{dt.datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def step(tag, methods, dataset, categories, timeout, **kw):
    cmd = [PY, "-u", RUNNER, "--tag", tag, "--methods", methods,
           "--dataset", dataset, "--categories", categories,
           "--timeout", str(timeout), "--gpu", "0", "--num-workers", "0"]
    for k, v in kw.items():
        if v is not None:
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    log(f"START {tag}: {methods} {dataset}/[{categories}] cap={timeout}s {kw}")
    try:
        subprocess.run(cmd, cwd=ROOT)
    except Exception as exc:                      # keep the queue moving
        log(f"ERROR {tag}: {type(exc).__name__}: {exc}")
    log(f"END   {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dinomaly", action="store_true")
    args = ap.parse_args()

    log("=" * 78)
    log("run-all started (no deadline; resumable)")

    # ---- 1. dinomaly: only this machine can run it (11.8GB peak)
    if not args.skip_dinomaly:
        step("mvtec_dinomaly", "dinomaly", "mvtec", DINO_UNATTEMPTED,
             timeout=3600, total_iter=200)
        step("mvtec_dinomaly", "dinomaly", "mvtec", DINO_TIMED_OUT,
             timeout=3600, total_iter=200)

    # ---- 2. the rest of MVTec
    step("mvtec_light", "patchcore,padim,winclip,promptad", "mvtec", "all",
         timeout=1800)
    step("mvtec_glass", "glass", "mvtec", "all", timeout=1800, meta_epochs=1)
    step("mvtec_simple", "simple", "mvtec", "all", timeout=1800, meta_epochs=14)
    step("mvtec_uniad", "uniad", "mvtec", "all", timeout=1800, meta_epochs=105)
    step("mvtec_coad", "coad", "mvtec", "all", timeout=1800, meta_epochs=1)

    # ---- 3. VisA (converted to MVTec layout by w55_visa_to_mvtec.py)
    step("visa_light", "patchcore,padim,winclip,promptad", "visa", "all",
         timeout=1800)
    step("visa_rd_orig", "rd_orig", "visa", "all", timeout=1800, meta_epochs=35)
    step("visa_rd", "rd", "visa", "all", timeout=1800, meta_epochs=10)
    step("visa_glass", "glass", "visa", "all", timeout=1800, meta_epochs=1)
    step("visa_simple", "simple", "visa", "all", timeout=1800, meta_epochs=14)
    step("visa_uniad", "uniad", "visa", "all", timeout=1800, meta_epochs=105)
    step("visa_coad", "coad", "visa", "all", timeout=1800, meta_epochs=1)
    step("visa_dinomaly", "dinomaly", "visa", "all", timeout=3600, total_iter=200)

    log("run-all finished (everything queued has been attempted)")


if __name__ == "__main__":
    main()
