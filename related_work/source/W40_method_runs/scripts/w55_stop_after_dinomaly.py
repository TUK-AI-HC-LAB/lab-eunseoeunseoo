"""W39 LabTask #55 - stop the run-all queue once dinomaly/MVTec is done.

The queue (w55_runall.py) would continue into the MVTec light methods after
dinomaly. Those all fit on the 8GB laptop, so they do not need this machine;
dinomaly does. This watchdog therefore lets the current dinomaly step finish and
then shuts the whole process tree down cleanly.

"Cleanly" matters: killing only the parent leaves the main.py child running and
competing for VRAM, which is exactly what produced a run of bogus timeouts
earlier today. Children are killed before parents here.

Usage (detached, so it survives the controlling session):
    python results_w40/w55_stop_after_dinomaly.py --pid <runall pid>
"""
import argparse
import csv
import datetime as dt
import os
import subprocess
import time

# Windows spawns a visible conhost.exe window for every console subprocess.
# The VRAM sampler alone fires every 2s for hours, so without this flag the
# screen flashes thousands of times during a sweep.
_NOWINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "results_w55", "day_plan.log")
SUMMARY = os.path.join(ROOT, "results_w55", "mvtec_dinomaly",
                       "summary_mvtec_dinomaly.csv")
TARGETS = {"metal_nut", "pill", "toothbrush"}


def log(msg):
    line = f"[{dt.datetime.now():%m-%d %H:%M:%S}] watchdog: {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def children_of(pid):
    """Descendant PIDs, via PowerShell CIM.

    wmic.exe is absent on current Windows builds, so it cannot be used here -
    it fails with WinError 2 and silently reports no children, which would let
    main.py survive as an orphan holding ~11.8GB of VRAM.
    """
    ps = (f"Get-CimInstance Win32_Process -Filter 'ParentProcessId={pid}' "
          f"| Select-Object -ExpandProperty ProcessId")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True, **_NOWINDOW)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def kill_tree(pid):
    """taskkill /T kills the whole tree; the explicit walk is a belt-and-braces
    second pass in case a grandchild was reparented in between."""
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                   capture_output=True, text=True, **_NOWINDOW)
    for kid in children_of(pid):
        kill_tree(kid)
    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                   capture_output=True, text=True, **_NOWINDOW)


def recorded():
    if not os.path.exists(SUMMARY):
        return set()
    with open(SUMMARY, newline="", encoding="utf-8") as f:
        # a category counts as settled once it has any non-timeout row
        return {r["category"] for r in csv.DictReader(f)
                if r["status"] != "timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--poll", type=int, default=30)
    args = ap.parse_args()

    log(f"armed; will stop pid {args.pid} once {sorted(TARGETS)} are recorded")
    while True:
        time.sleep(args.poll)

        # the queue moving on by itself is also a stop condition
        try:
            tail = open(LOG, encoding="utf-8").read()
        except OSError:
            tail = ""
        moved_on = "START mvtec_light" in tail

        left = TARGETS - recorded()
        if not left or moved_on:
            why = "queue moved past dinomaly" if moved_on else "all 3 recorded"
            log(f"stopping ({why}); remaining were {sorted(left)}")
            kill_tree(args.pid)
            time.sleep(3)
            log("process tree terminated; results are on disk and resumable")
            return

        log(f"waiting; still to finish: {sorted(left)}")


if __name__ == "__main__":
    main()
