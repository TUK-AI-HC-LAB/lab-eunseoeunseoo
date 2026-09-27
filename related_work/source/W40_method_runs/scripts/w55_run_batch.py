"""W39 LabTask #55 - batch runner over (method x dataset x category).

Runs main.py as a subprocess per job so that one method crashing or hanging
cannot take the rest down, and records for every job:
  exit code, wall time, peak VRAM, the metrics row, or the failure reason.

Everything lands in one CSV so the report table and the "실행하지 못한 범위와
이유" section are both backed by the same raw file.

Usage examples:
  # timing smoke test: all methods, one category, 1 epoch, 10 min cap
  python results_w40/w55_run_batch.py --tag smoke --methods all \
      --dataset mvtec --categories bottle --meta-epochs 1 --timeout 600

  # full sweep for the cheap methods
  python results_w40/w55_run_batch.py --tag mvtec_full \
      --methods patchcore,padim,winclip --dataset mvtec --categories all
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import threading
import time

# Windows spawns a visible conhost.exe window for every console subprocess.
# The VRAM sampler alone fires every 2s for hours, so without this flag the
# screen flashes thousands of times during a sweep.
_NOWINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PY = os.path.join(ROOT, ".venv-gpu", "Scripts", "python.exe")

MVTEC = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
         "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
         "wood", "zipper"]
VISA = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
ALL_METHODS = ["patchcore", "padim", "winclip", "simple", "rd_orig", "rd",
               "dinomaly", "uniad", "glass", "promptad", "coad"]

DATA_PATHS = {
    "mvtec": "C:/ai_local/glad_dataset/MVTec-AD",
    "visa": "C:/ai_local/glad_dataset/VisA_mvtec",
}


class VramSampler(threading.Thread):
    """Polls nvidia-smi while a job runs and keeps the peak used-MiB.

    Peak VRAM is the number that decides what can also run on the 8GB laptop,
    so it is recorded per job rather than guessed.
    """

    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10, **_NOWINDOW)
                v = int(out.stdout.strip().splitlines()[0])
                self.peak = max(self.peak, v)
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)
        return self.peak


def read_metrics(results_path, method):
    """main_run.py writes results/<method>__<layers>_<csv_save_name>/results_<method>.csv

    The <layers> part is method-specific (patchcore uses layer2_layer3, coad uses
    2_3_5_6_7_8_9, ...), so glob for it rather than assuming the default.
    """
    hits = glob.glob(os.path.join(results_path, "*", f"results_{method}.csv"))
    if not hits:
        return None, ""
    f = max(hits, key=os.path.getmtime)
    with open(f, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return (rows[-1] if rows else None), f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--methods", default="all")
    ap.add_argument("--dataset", default="mvtec", choices=["mvtec", "visa"])
    ap.add_argument("--categories", default="bottle")
    ap.add_argument("--meta-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--total-iter", type=int, default=None,
                    help="dinomaly loops on args.total_iter "
                         "(trainer_dinomaly.py:92,117), not meta_epochs")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per job")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="dataloader workers; the default of 1 makes JPEG "
                         "decode the bottleneck, not the GPU")
    ap.add_argument("--gpu", default="0",
                    help="GPU index, or 'cpu' to force CPU")
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    methods = (ALL_METHODS if args.methods == "all"
               else [m.strip() for m in args.methods.split(",") if m.strip()])
    if args.categories == "all":
        cats = MVTEC if args.dataset == "mvtec" else VISA
    else:
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    out_root = args.out_root or os.path.join(ROOT, "results_w55", args.tag)
    logs_dir = os.path.join(out_root, "run_logs")
    os.makedirs(logs_dir, exist_ok=True)
    summary_csv = os.path.join(out_root, f"summary_{args.tag}.csv")

    jobs = [(m, c) for m in methods for c in cats]
    print(f"tag={args.tag} dataset={args.dataset} jobs={len(jobs)} "
          f"timeout={args.timeout}s per job")
    print(f"out -> {out_root}\n")

    fields = ["tag", "dataset", "method", "category", "status", "exit_code",
              "wall_s", "peak_vram_mb", "meta_epochs", "total_iter",
              "batch_size", "auroc_mean",
              "pixel_auroc_mean", "sal_f1_mean", "reason", "log_path",
              "results_csv"]
    done = set()
    if os.path.exists(summary_csv):
        with open(summary_csv, newline="", encoding="utf-8") as _fh:
            for _r in csv.DictReader(_fh):
                # a timeout is worth retrying with a bigger cap; ok/failed/oom are final
                if _r.get("status") != "timeout":
                    done.add((_r.get("method"), _r.get("category")))
    if done:
        print(f"resume: {len(done)} job(s) already recorded, skipping those\n")

    write_header = not os.path.exists(summary_csv)
    fh = open(summary_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if write_header:
        writer.writeheader()
        fh.flush()

    for n, (method, cat) in enumerate(jobs, 1):
        if (method, cat) in done:
            print(f"[{n}/{len(jobs)}] {method:<10} {args.dataset}/{cat:<12} ... skipped (done)")
            continue
        job_results = os.path.join(out_root, "results", f"{method}__{cat}")
        os.makedirs(job_results, exist_ok=True)
        log_path = os.path.join(logs_dir, f"{method}__{args.dataset}__{cat}.log")

        cmd = [PY, "main.py",
               "--method", method,
               "--dataset", args.dataset,
               "--category", cat,
               "--data-path", DATA_PATHS[args.dataset],
               "--results-path", job_results]
        if args.gpu != "cpu":
            cmd += ["--gpu", str(args.gpu)]
        if args.meta_epochs is not None:
            cmd += ["--meta-epochs", str(args.meta_epochs)]
        if args.total_iter is not None:
            cmd += ["--total-iter", str(args.total_iter)]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        cmd += ["--num-workers", str(args.num_workers)]

        print(f"[{n}/{len(jobs)}] {method:<10} {args.dataset}/{cat:<12} ... ",
              end="", flush=True)
        t0 = time.time()
        status, code, reason = "ok", "", ""
        sampler = VramSampler()
        sampler.start()
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            lf.write("CMD: " + " ".join(cmd) + "\n\n")
            lf.flush()
            try:
                p = subprocess.run(cmd, cwd=ROOT, stdout=lf,
                                   stderr=subprocess.STDOUT,
                                   timeout=args.timeout, **_NOWINDOW)
                code = p.returncode
                if code != 0:
                    status = "failed"
            except subprocess.TimeoutExpired:
                status, code = "timeout", ""
                reason = f"exceeded {args.timeout}s"
        wall = round(time.time() - t0, 1)
        peak_vram = sampler.stop()

        row = {"tag": args.tag, "dataset": args.dataset, "method": method,
               "category": cat, "status": status, "exit_code": code,
               "wall_s": wall, "peak_vram_mb": peak_vram,
               "meta_epochs": args.meta_epochs,
               "total_iter": args.total_iter,
               "batch_size": args.batch_size, "log_path": log_path,
               "auroc_mean": "", "pixel_auroc_mean": "", "sal_f1_mean": "",
               "reason": reason, "results_csv": ""}

        if status == "failed":
            # pull the last exception line out of the log for the report
            try:
                tail = open(log_path, encoding="utf-8", errors="replace").read()
                errs = [l.strip() for l in tail.splitlines()
                        if ("Error" in l or "Exception" in l) and "Traceback" not in l]
                if "CUDA out of memory" in tail or "OutOfMemoryError" in tail:
                    row["reason"] = "CUDA OOM"
                    row["status"] = status = "oom"
                else:
                    row["reason"] = errs[-1][:200] if errs else "see log"
            except Exception:
                row["reason"] = "see log"

        m, m_path = read_metrics(job_results, method)
        if m:
            row["auroc_mean"] = m.get("auroc_mean", "")
            row["pixel_auroc_mean"] = m.get("pixel_auroc_mean", "")
            row["sal_f1_mean"] = m.get("sal_f1_mean", "")
            row["results_csv"] = m_path
            if status == "ok":
                print(f"{status:<8} {wall:>7.1f}s {peak_vram:>6}MiB  "
                      f"I-AUROC={row['auroc_mean'][:7]} "
                      f"P-AUROC={row['pixel_auroc_mean'][:7]}")
            else:
                print(f"{status:<8} {wall:>7.1f}s  {row['reason'][:60]}")
        else:
            print(f"{status:<8} {wall:>7.1f}s  {row['reason'][:60]}")

        writer.writerow(row)
        fh.flush()

    fh.close()
    print(f"\nsummary -> {summary_csv}")


if __name__ == "__main__":
    main()
