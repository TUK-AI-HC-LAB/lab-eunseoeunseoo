"""W39 LabTask #55 - unattended day plan for the desktop (RTX 5070 / 12GB).

Runs the sweeps in priority order until a wall-clock deadline, then stops
cleanly. Everything is resumable: each step appends to its own summary CSV and
already-recorded jobs are skipped, so this can be re-launched at any time.

Priority rationale (peak VRAM measured in the smoke step):
  rd 11.8GB / dinomaly 11.7GB / rd_orig 8.3GB  -> only the 12GB desktop can run these
  patchcore 2.3 / padim 1.7 / simple 3.3 / winclip 3.8 GB -> the 8GB laptop can too
so the desktop-only methods go first and the cheap ones fill in afterwards.

Usage:
    python results_w40/w55_day_plan.py --deadline 18:00
    python results_w40/w55_day_plan.py --deadline 18:00 --skip-wait   # smoke already done
"""
import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PY = os.path.join(ROOT, ".venv-gpu", "Scripts", "python.exe")
RUNNER = os.path.join(ROOT, "results_w40", "w55_run_batch.py")
W55 = os.path.join(ROOT, "results_w55")
PLAN_LOG = os.path.join(W55, "day_plan.log")

TRAIN_METHODS = ["simple", "rd_orig", "rd", "dinomaly", "uniad",
                 "glass", "promptad", "coad"]
DESKTOP_ONLY = ["dinomaly", "rd_orig", "rd"]       # >7GB peak
LAPTOP_OK = ["patchcore", "padim", "winclip", "simple"]
REMAINING = ["uniad", "glass", "coad"]

# dinomaly is driven by args.total_iter (trainer_dinomaly.py:92,117), not by
# meta_epochs, so its two timing points are iteration counts instead of epochs.
DINO_ITER_LO = 200
DINO_ITER_HI = 600


def log(msg):
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    os.makedirs(W55, exist_ok=True)
    with open(PLAN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_summary(tag):
    p = os.path.join(W55, tag, f"summary_{tag}.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wait_for_smoke(deadline):
    """The smoke batch may still be running from an earlier launch."""
    while time.time() < deadline:
        rows = read_summary("smoke")
        if len(rows) >= 11:
            log(f"smoke complete ({len(rows)} rows)")
            return True
        log(f"waiting for smoke ... {len(rows)}/11 done")
        time.sleep(60)
    return False


def epoch_budget(method, budget_s):
    """epochs that fit budget_s, from the 1-epoch and 5-epoch measurements.

        total(E) = fixed + E * per_epoch
    Two points give per_epoch and fixed; anything missing falls back to 1 epoch,
    which is honest rather than optimistic.
    """
    t1 = t5 = None
    for r in read_summary("smoke"):
        if r["method"] == method and r["status"] == "ok":
            t1 = float(r["wall_s"])
    for r in read_summary("timing5"):
        if r["method"] == method and r["status"] == "ok":
            t5 = float(r["wall_s"])
    if t1 is None or t5 is None or t5 <= t1:
        return 1, "no 2-point timing"
    per_epoch = (t5 - t1) / 4.0
    fixed = max(0.0, t1 - per_epoch)
    if per_epoch <= 0:
        return 1, "degenerate fit"
    n = int((budget_s - fixed) // per_epoch)
    n = max(1, n)
    return n, f"fixed={fixed:.0f}s per_epoch={per_epoch:.1f}s"


def dinomaly_iter_budget(budget_s):
    """Same two-point fit as epoch_budget, but over total_iter."""
    def wall(tag):
        for r in read_summary(tag):
            if r["method"] == "dinomaly" and r["status"] == "ok":
                return float(r["wall_s"])
        return None

    t_lo, t_hi = wall("smoke_dino1"), wall("smoke_dino2")
    if t_lo is None or t_hi is None or t_hi <= t_lo:
        return DINO_ITER_LO
    per_iter = (t_hi - t_lo) / (DINO_ITER_HI - DINO_ITER_LO)
    fixed = max(0.0, t_lo - per_iter * DINO_ITER_LO)
    if per_iter <= 0:
        return DINO_ITER_LO
    n = int((budget_s - fixed) // per_iter)
    return max(50, n)


def run_step(tag, methods, dataset, categories, meta_epochs, timeout, deadline,
             total_iter=None):
    if time.time() >= deadline:
        log(f"SKIP {tag}: past deadline")
        return
    cmd = [PY, "-u", RUNNER, "--tag", tag,
           "--methods", ",".join(methods) if isinstance(methods, list) else methods,
           "--dataset", dataset, "--categories", categories,
           "--timeout", str(timeout), "--gpu", "0", "--num-workers", "0"]
    if meta_epochs is not None:
        cmd += ["--meta-epochs", str(meta_epochs)]
    if total_iter is not None:
        cmd += ["--total-iter", str(total_iter)]
    log(f"START {tag}: {dataset}/{categories} methods={methods} "
        f"meta_epochs={meta_epochs} total_iter={total_iter} timeout={timeout}s")
    remaining = max(60, int(deadline - time.time()))
    try:
        subprocess.run(cmd, cwd=ROOT, timeout=remaining)
    except subprocess.TimeoutExpired:
        log(f"STOP {tag}: hit the day deadline mid-step (results so far are kept)")
    log(f"END   {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="18:00", help="HH:MM local, stop by then")
    ap.add_argument("--skip-wait", action="store_true")
    ap.add_argument("--per-job-budget", type=int, default=420,
                    help="target seconds per training job, used to pick epochs")
    ap.add_argument("--job-timeout", type=int, default=900)
    args = ap.parse_args()

    hh, mm = (int(x) for x in args.deadline.split(":"))
    today = dt.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    if today <= dt.datetime.now():
        today += dt.timedelta(days=1)
    deadline = today.timestamp()
    log("=" * 78)
    log(f"day plan start; deadline {today:%Y-%m-%d %H:%M} "
        f"({(deadline - time.time())/3600:.1f}h from now)")

    if not args.skip_wait and not wait_for_smoke(deadline):
        log("smoke never finished before deadline; stopping")
        return

    # 0) fill the two gaps the first smoke pass left:
    #    glass failed on a missing albumentations (now installed), and dinomaly
    #    ignores meta_epochs entirely - it loops on args.total_iter, so a
    #    1-"epoch" run actually ran all 5000 iterations and hit the cap.
    run_step("smoke", ["glass"], "mvtec", "bottle", 1, 900, deadline)
    run_step("smoke_dino1", ["dinomaly"], "mvtec", "bottle", None, 900, deadline,
             total_iter=DINO_ITER_LO)

    # 1) second timing point so training cost can be extrapolated honestly
    train_wo_dino = [m for m in TRAIN_METHODS if m != "dinomaly"]
    run_step("timing5", train_wo_dino, "mvtec", "bottle", 5, 900, deadline)
    run_step("smoke_dino2", ["dinomaly"], "mvtec", "bottle", None, 1800, deadline,
             total_iter=DINO_ITER_HI)

    budgets = {}
    for m in train_wo_dino:
        n, why = epoch_budget(m, args.per_job_budget)
        budgets[m] = n
        log(f"  epoch budget {m:<10} = {n:<4} ({why})")
    dino_iters = dinomaly_iter_budget(args.per_job_budget)
    log(f"  iter budget  dinomaly   = {dino_iters}")

    # 2) desktop-only methods first: the 8GB laptop cannot run these at all
    #    (dinomaly 11.8GB / rd 11.8GB / rd_orig 8.3GB peak)
    run_step("mvtec_dinomaly", ["dinomaly"], "mvtec", "all", None,
             args.job_timeout, deadline, total_iter=dino_iters)
    for m in ["rd_orig", "rd"]:
        run_step(f"mvtec_{m}", [m], "mvtec", "all", budgets.get(m),
                 args.job_timeout, deadline)

    # 3) cheap methods across MVTec
    run_step("mvtec_light", ["patchcore", "padim", "winclip", "promptad"],
             "mvtec", "all", None, args.job_timeout, deadline)
    run_step("mvtec_simple", ["simple"], "mvtec", "all",
             budgets.get("simple"), args.job_timeout, deadline)

    # 4) whatever is left on MVTec
    for m in REMAINING:
        run_step(f"mvtec_{m}", [m], "mvtec", "all", budgets.get(m),
                 args.job_timeout, deadline)

    # 5) VisA, same order of preference
    run_step("visa_light", ["patchcore", "padim", "winclip", "promptad"],
             "visa", "all", None, args.job_timeout, deadline)
    run_step("visa_dinomaly", ["dinomaly"], "visa", "all", None,
             args.job_timeout, deadline, total_iter=dino_iters)
    for m in ["rd_orig", "rd", "simple"] + REMAINING:
        run_step(f"visa_{m}", [m], "visa", "all", budgets.get(m),
                 args.job_timeout, deadline)

    log("day plan finished")


if __name__ == "__main__":
    main()
