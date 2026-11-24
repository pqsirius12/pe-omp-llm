#!/usr/bin/env python3
"""pe_speed.py - single-thread speed of every kernel on the default core
vs. forced onto the E-cores (macOS 'taskpolicy -b' = background QoS, which
macOS runs only on efficiency cores). Answers: how much slower is an E-core
for each kind of loop? Reuses run.py's build and run helpers.

  python3 pe_speed.py            (about 6 minutes; Mac must be idle)
Writes results/pe_speed.csv.
"""
import csv, os, statistics, time
import run   # run.py in the same folder

REPS = 3

def main():
    sizes = run.load_sizes()
    os.makedirs(run.BDIR, exist_ok=True)
    os.makedirs(run.RDIR, exist_ok=True)
    out = open(os.path.join(run.RDIR, "pe_speed.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["kernel", "mode", "rep", "time_s"])
    print(f"{'kernel':11s} {'default s':>9s} {'E-core s':>9s} {'E/default':>9s}")
    for name, src in run.kernels().items():
        exe = os.path.join(run.BDIR, f"ref_{name}")
        if not os.path.exists(exe):
            ok, err = run.build(src, exe)
            if not ok:
                print(name, "BUILD FAILED", err[:200]); continue
        res = {"default": [], "ecore": []}
        for r in range(REPS):                     # alternate modes to spread drift
            for mode in ("default", "ecore"):
                st, t, _ = run.run_once(exe, 1, sizes.get(name, []), 300,
                                        "background" if mode == "ecore" else None)
                if st == "ok":
                    res[mode].append(t); w.writerow([name, mode, r, t]); out.flush()
                time.sleep(1.0)
        d, e = statistics.median(res["default"]), statistics.median(res["ecore"])
        print(f"{name:11s} {d:9.3f} {e:9.3f} {e / d:9.2f}", flush=True)
    out.close()

if __name__ == "__main__":
    main()
