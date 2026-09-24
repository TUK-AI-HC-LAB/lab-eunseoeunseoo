"""W39 LabTask #55 - per-method timing report.

Reads every summary_*.csv under results_w55/ plus the matching run log and
produces one timing table. Total wall time alone does not explain much, so each
run is also split into phases using loguru timestamps that were verified to be
printed by all 11 methods:

    startup_s ... wall minus the rest: interpreter + torch import + CUDA init
                  (happens before the first log line, so it has no timestamp)
    data_s ...... first log line -> "Dataset <name>: train=.. test=.."
    model_s ..... that -> "Training models (i/n)"   (backbone + trainer build)
    run_s ....... that -> "<name>_test_auroc_mean"  (train + inference + eval)

Additionally, methods that inherit the shared predict path report their own
per-image inference cost ("Average inference time"). Per the W40 analysis only
patchcore and simple do; for the other nine those two columns stay empty, which
is expected rather than a parsing failure.

Run any time, including while the day plan is still going:
    python results_w40/w55_timing_report.py
"""
import csv
import datetime as dt
import glob
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W55 = os.path.join(ROOT, "results_w55")
OUT_CSV = os.path.join(W55, "timing_per_run.csv")
OUT_TXT = os.path.join(W55, "timing_report.txt")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")

# markers verified to appear for every one of the 11 methods
M_DATA = re.compile(r"get_dataloaders:\d+ - Dataset ")   # dataloaders built
M_MODEL = re.compile(r"Training models \(")              # trainer constructed
M_DONE = re.compile(r"_test_auroc_mean")                 # metrics computed

# only emitted by methods that inherit the shared predict path
# (W40 finding: patchcore and simple only)
INFER = re.compile(r"Average inference time:\s*([0-9.]+)")
PIXEL = re.compile(r"Average pixel inference time:\s*([0-9.]+)")

lines_out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines_out.append(s)


def parse_log(path):
    """Phase split + the framework's own per-image inference numbers."""
    out = {"startup_s": "", "data_s": "", "model_s": "", "run_s": "",
           "infer_s_per_img": "", "pixel_infer_s_per_img": ""}
    if not path or not os.path.exists(path):
        return out
    try:
        raw = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return out
    text = ANSI.sub("", raw)

    stamps = [dt.datetime.strptime(m, "%Y-%m-%d %H:%M:%S.%f")
              for m in TS.findall(text)]
    if not stamps:
        return out
    t_first, t_last = stamps[0], stamps[-1]

    def stamp_near(pattern, last=False):
        found = None
        for line in text.splitlines():
            if pattern.search(line):
                m = TS.search(line)
                if m:
                    found = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
                    if not last:
                        return found
        return found

    t_data = stamp_near(M_DATA, last=True)    # dataloaders ready
    t_model = stamp_near(M_MODEL)             # trainer built
    t_done = stamp_near(M_DONE, last=True)    # metrics computed

    if t_data:
        out["data_s"] = round((t_data - t_first).total_seconds(), 1)
    if t_data and t_model:
        out["model_s"] = round((t_model - t_data).total_seconds(), 1)
    if t_model:
        end = t_done or t_last
        out["run_s"] = round((end - t_model).total_seconds(), 1)

    m = INFER.search(text)
    if m:
        out["infer_s_per_img"] = float(m.group(1))
    m = PIXEL.search(text)
    if m:
        out["pixel_infer_s_per_img"] = float(m.group(1))
    return out


def main():
    summaries = sorted(glob.glob(os.path.join(W55, "*", "summary_*.csv")))
    if not summaries:
        sys.exit(f"no summary csv under {W55}")

    rows = []
    for s in summaries:
        with open(s, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r.update(parse_log(r.get("log_path")))
                # whatever wall time the logged phases do not account for is
                # interpreter + torch import + CUDA init, which happens before
                # the first timestamped line exists
                try:
                    acc = sum(float(r[k]) for k in ("data_s", "model_s", "run_s")
                              if r.get(k) not in ("", None))
                    r["startup_s"] = round(float(r["wall_s"]) - acc, 1)
                except (TypeError, ValueError):
                    r["startup_s"] = ""
                rows.append(r)

    fields = ["tag", "dataset", "method", "category", "status", "wall_s",
              "startup_s", "data_s", "model_s", "run_s", "infer_s_per_img",
              "pixel_infer_s_per_img", "peak_vram_mb", "meta_epochs",
              "auroc_mean", "pixel_auroc_mean"]
    os.makedirs(W55, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    P("=" * 108)
    P(f"W55 timing report   generated {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    P("=" * 108)
    P(f"  runs recorded: {len(rows)}   source: results_w55/*/summary_*.csv")
    P("")

    # ---- per run, grouped by tag
    for tag in sorted({r["tag"] for r in rows}):
        sub = [r for r in rows if r["tag"] == tag]
        P(f"[{tag}]  {len(sub)} run(s)")
        P(f"  {'method':<11}{'cat':<13}{'status':<8}{'wall_s':>8}{'start':>7}{'data':>6}"
          f"{'model':>7}{'run':>9}{'s/img':>8}{'VRAM MiB':>10}{'ep':>5}")
        for r in sorted(sub, key=lambda x: (x["method"], x["category"])):
            P(f"  {r['method']:<11}{r['category'][:12]:<13}{r['status']:<8}"
              f"{str(r['wall_s']):>8}{str(r['startup_s']):>7}{str(r['data_s']):>6}"
              f"{str(r['model_s']):>7}{str(r['run_s']):>9}"
              f"{str(r['infer_s_per_img']):>8}{str(r['peak_vram_mb']):>10}"
              f"{str(r.get('meta_epochs','')):>5}")
        P("")

    # ---- per method aggregate over successful runs
    P("=" * 108)
    P("PER-METHOD SUMMARY (status=ok only)")
    P("=" * 108)
    P(f"  {'method':<11}{'runs':>5}{'wall mean':>11}{'wall min':>10}{'wall max':>10}"
      f"{'run mean':>10}{'VRAM max':>10}  laptop-8GB?")
    P("    wall = total process time; run = train+infer+eval only "
      "(excludes dataset scan and model build)")
    ok = [r for r in rows if r["status"] == "ok"]
    for m in sorted({r["method"] for r in ok}):
        sub = [r for r in ok if r["method"] == m]
        walls = [float(r["wall_s"]) for r in sub if r["wall_s"]]
        vram = [int(r["peak_vram_mb"]) for r in sub
                if str(r.get("peak_vram_mb", "")).isdigit()]
        runs_s = [float(r["run_s"]) for r in sub if r["run_s"] != ""]
        vmax = max(vram) if vram else 0
        # 8GB card, keep ~1GB for desktop/driver overhead
        fits = "yes" if vmax and vmax < 7000 else ("NO" if vmax else "?")
        P(f"  {m:<11}{len(sub):>5}{statistics.mean(walls):>11.1f}"
          f"{min(walls):>10.1f}{max(walls):>10.1f}"
          f"{(statistics.mean(runs_s) if runs_s else 0):>10.1f}{vmax:>10}  {fits}")

    # ---- anything that did not succeed
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        P("")
        P("=" * 108)
        P("NOT SUCCESSFUL")
        P("=" * 108)
        for r in sorted(bad, key=lambda x: (x["method"], x["category"])):
            P(f"  {r['method']:<11}{r['dataset']:<7}{r['category'][:12]:<13}"
              f"{r['status']:<9}{str(r['wall_s']):>8}s  {r.get('reason','')[:60]}")

    P("")
    P(f"per-run csv -> {OUT_CSV}")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")
    print(f"text report -> {OUT_TXT}")


if __name__ == "__main__":
    main()
