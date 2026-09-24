"""W39 LabTask #55 - revised afternoon plan for the remaining desktop GPU time.

Why this replaces the original ordering
---------------------------------------
The first pass put dinomaly first, but at total_iter=200 a category costs
680-900s while the job cap was 900s, so 6 of the first 10 categories were killed
at the cap and produced no row at all. A timeout is strictly worse than an
under-trained result: it yields no number to report.

rd_orig and rd are far cheaper per category and their cost is known exactly from
the two-point fit, so they are run first at a fixed per-category budget and are
guaranteed to finish all 15 categories. dinomaly then gets whatever remains,
with a cap raised above its observed worst case so the runs that do start
actually complete.

Measured inputs (results_w55/timing_per_run.csv):
    rd_orig   fixed 11s + 5.4 s/epoch
    rd        fixed 26s + 17.0 s/epoch
    dinomaly  ~727s per category at total_iter=200 (4 completed runs)
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv-gpu", "Scripts", "python.exe")
RUNNER = os.path.join(ROOT, "results_w40", "w55_run_batch.py")
LOG = os.path.join(ROOT, "results_w55", "day_plan.log")


def log(msg):
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def step(tag, methods, dataset, categories, deadline, timeout,
         meta_epochs=None, total_iter=None):
    if time.time() >= deadline:
        log(f"SKIP {tag}: past deadline")
        return
    cmd = [PY, "-u", RUNNER, "--tag", tag, "--methods", ",".join(methods),
           "--dataset", dataset, "--categories", categories,
           "--timeout", str(timeout), "--gpu", "0", "--num-workers", "0"]
    if meta_epochs is not None:
        cmd += ["--meta-epochs", str(meta_epochs)]
    if total_iter is not None:
        cmd += ["--total-iter", str(total_iter)]
    log(f"START {tag}: {methods} {dataset}/{categories} "
        f"meta_epochs={meta_epochs} total_iter={total_iter} cap={timeout}s")
    try:
        subprocess.run(cmd, cwd=ROOT, timeout=max(60, int(deadline - time.time())))
    except subprocess.TimeoutExpired:
        log(f"STOP {tag}: day deadline reached mid-step; finished jobs are kept")
    log(f"END   {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="18:00")
    args = ap.parse_args()
    hh, mm = (int(x) for x in args.deadline.split(":"))
    d = dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    if d <= dt.datetime.now():
        d += dt.timedelta(days=1)
    deadline = d.timestamp()

    log("=" * 78)
    log(f"afternoon plan; deadline {d:%H:%M} "
        f"({(deadline - time.time())/3600:.2f}h left)")

    # 200s per category -> epochs from the measured fits
    step("mvtec_rd_orig", ["rd_orig"], "mvtec", "all", deadline,
         timeout=600, meta_epochs=35)
    step("mvtec_rd", ["rd"], "mvtec", "all", deadline,
         timeout=600, meta_epochs=10)

    # cap raised to 1200s: observed worst completed dinomaly run was ~900s
    step("mvtec_dinomaly", ["dinomaly"], "mvtec", "all", deadline,
         timeout=1200, total_iter=200)

    # anything left goes to the cheap methods, which also run on the 8GB laptop
    step("mvtec_light", ["patchcore", "padim", "winclip", "promptad"],
         "mvtec", "all", deadline, timeout=900)
    step("mvtec_glass", ["glass"], "mvtec", "all", deadline,
         timeout=900, meta_epochs=1)

    log("afternoon plan finished")


if __name__ == "__main__":
    main()
