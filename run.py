#!/usr/bin/env python3
"""run.py - build, verify and time OpenMP kernel variants.

Layout it expects (relative to this file):
  kernels/kNN_name.c          serial reference for each kernel
  variants/<name>/<label>.c   parallel versions (LLM-written or baselines)
  results/                    created; CSV output goes here

Commands:
  python3 run.py calibrate               time every serial reference once
  python3 run.py run [options]           build + verify + time all variants

Standard library only, so nothing needs installing.
"""
import argparse, csv, glob, json, os, platform, random, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
KDIR, VDIR = os.path.join(ROOT, "kernels"), os.path.join(ROOT, "variants")
BDIR, RDIR = os.path.join(ROOT, "build"), os.path.join(ROOT, "results")
LINE = re.compile(r"kernel=(\S+) time_s=([0-9.eE+-]+) checksum=([0-9.eE+-]+)")
# Pure-integer kernels: any correct parallel version must match exactly.
EXACT = {"histogram", "prefix", "primes", "collatz"}
# Integer counts decided by floating-point tests (escape radius, distance
# cutoff): a correct rewrite may flip a handful of boundary cases.
COUNT_TOL = {"mandelbrot": 1e-6, "cutoff": 1e-6}
# Everything else sums floating-point values in a possibly different order.
REL_TOL = 1e-9
# -ffp-contract=off forbids fusing a*b+c into one FMA instruction, so every
# variant rounds identically on any compiler (clang on arm64 fuses by default).
FP = "-ffp-contract=off"
# Schedules tried on expert (schedule(runtime)) versions. Loops with
# millions of cheap iterations need large chunks, or the cost of handing
# out chunks dominates; loops over a few thousand heavy rows need small ones.
FINE = ["static", "dynamic,1024", "dynamic,65536", "guided"]
COARSE = ["static", "dynamic,1", "dynamic,16", "guided"]
SCHEDULES = {k: FINE for k in ("saxpy", "histogram", "norm", "spmv", "primes",
                               "collatz", "cutoff", "kmeans")}
SCHEDULES.update({k: COARSE for k in ("matmul", "jacobi2d", "trisum",
                                     "mandelbrot", "nbody", "conv2d")})
SCHEDULES["prefix"] = ["static"]   # block scan: splits statically by design
# Per-kernel extra compiler flags, applied to EVERY build of that kernel (serial
# reference included) so all variants are compared under the same compiler.
# histogram: Apple clang 21's -O2 loop vectorizer miscompiles the per-thread
# bin-update loop inside OpenMP regions (wrong counts; correct with
# -fno-vectorize or -O0; no undefined behaviour found) - see results/diag_hist*.txt
NOVEC = "-fno-vectorize" if platform.system() == "Darwin" else "-fno-tree-vectorize"
EXTRA_FLAGS = {"histogram": [NOVEC]}


def compiler_cmd():
    """Compile command for this machine. macOS: Apple clang + Homebrew libomp
    (verified on the M4: OpenMP 5.1). Linux: gcc -fopenmp."""
    if platform.system() == "Darwin":
        pre = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True,
                             text=True, check=True).stdout.strip()
        return ["clang", "-O2", FP, "-Xpreprocessor", "-fopenmp", f"-I{pre}/include",
                f"-L{pre}/lib", "-lomp", "-lm", f"-I{KDIR}"]
    return ["gcc", "-O2", FP, "-fopenmp", "-lm", f"-I{KDIR}"]


def build(src, out, kernel=""):
    """Compile one file; returns (ok, error_text). Link flags go after the source."""
    cmd = compiler_cmd()
    flags = [c for c in cmd[1:] if not c.startswith(("-l", "-L"))]
    flags += EXTRA_FLAGS.get(kernel, [])
    libs = [c for c in cmd[1:] if c.startswith(("-l", "-L"))]
    r = subprocess.run([cmd[0], *flags, src, "-o", out, *libs],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr[-2000:]


def run_once(exe, threads, args, timeout, qos, sched=""):
    """Run a binary once. Returns (status, time_s, checksum)."""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    env.pop("OMP_SCHEDULE", None)
    if sched:                         # only expert variants read this
        env["OMP_SCHEDULE"] = sched
    cmd = [exe, *args]
    if qos == "background":           # macOS: background QoS -> E-cores only
        cmd = ["taskpolicy", "-b", *cmd]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", None, None  # hangs and deadlocks land here
    m = LINE.search(r.stdout)
    if r.returncode != 0 or not m:
        return f"crash(rc={r.returncode})", None, None
    return "ok", float(m.group(2)), float(m.group(3))


def checksum_ok(kernel, got, ref):
    if kernel in EXACT:
        return got == ref
    tol = COUNT_TOL.get(kernel, REL_TOL)
    return abs(got - ref) <= tol * max(1.0, abs(ref))


def kernels():
    """Map kernel name (e.g. 'saxpy') -> serial source path."""
    out = {}
    for path in sorted(glob.glob(os.path.join(KDIR, "k[0-9][0-9]_*.c"))):
        out[os.path.basename(path)[4:-2]] = path
    return out


def load_sizes():
    """Optional per-kernel argument overrides set after calibration."""
    p = os.path.join(ROOT, "sizes.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def references(sizes, timeout, only=None):
    """Build and run each selected serial kernel once: its checksum is ground truth."""
    os.makedirs(BDIR, exist_ok=True)
    refs = {}
    for name, src in kernels().items():
        if only and name not in only:
            continue
        exe = os.path.join(BDIR, f"ref_{name}")
        ok, err = build(src, exe, name)
        if not ok:
            sys.exit(f"serial reference {name} failed to build:\n{err}")
        st, t, c = run_once(exe, 1, sizes.get(name, []), timeout, None)
        if st != "ok":
            sys.exit(f"serial reference {name} failed: {st}")
        refs[name] = {"time_s": t, "checksum": c}
        print(f"  ref {name:12s} {t:8.3f} s  checksum={c:.12e}", flush=True)
    return refs


def cmd_calibrate(a):
    print("Serial reference times (aim: 1-3 s each on the target machine):")
    refs = references(load_sizes(), a.timeout)
    os.makedirs(RDIR, exist_ok=True)
    json.dump(refs, open(os.path.join(RDIR, "reference.json"), "w"), indent=1)


def cmd_run(a):
    sizes = load_sizes()
    only = set(a.kernels.split(",")) if a.kernels else None
    # the serial run is the slowest; give it 3x the per-run timeout
    refs = references(sizes, 3 * a.timeout, only)
    threads = [int(t) for t in a.threads.split(",")]
    jobs, broken = [], []
    vfilter = re.compile(a.variants) if a.variants else None
    for name in kernels():
        if only and name not in only:
            continue
        for src in sorted(glob.glob(os.path.join(VDIR, name, "*.c"))):
            label = os.path.basename(src)[:-2]
            if vfilter and not vfilter.search(label):
                continue
            exe = os.path.join(BDIR, f"v_{name}__{label}")
            ok, err = build(src, exe, name)
            if not ok:
                broken.append((name, label, err))
                edir = os.path.join(RDIR, "compile_errors")
                os.makedirs(edir, exist_ok=True)
                with open(os.path.join(edir, f"{name}__{label}.txt"), "w") as ef:
                    ef.write(err)
                continue
            scheds = SCHEDULES.get(name, [""]) if label.startswith("expert") else [""]
            jobs += [(name, label, exe, t, sc) for t in threads for sc in scheds]
    os.makedirs(RDIR, exist_ok=True)
    out = os.path.join(RDIR, a.out)
    new = not os.path.exists(out)
    if a.skip_done and not new:
        done = {}
        for row in csv.DictReader(open(out)):
            k = (row["kernel"], row["variant"], row["schedule"], row["threads"])
            # a timeout already answered the question ("over 10x serial"): count it as complete
            done[k] = done.get(k, 0) + (10**6 if row["status"] == "timeout" else 1)
        before = len(jobs)
        jobs = [j for j in jobs if done.get((j[0], j[1], j[4], str(j[3])), 0) < a.reps]
        broken = [b for b in broken if (b[0], b[1], "", "") not in done]
        print(f"skip-done: {before - len(jobs)} configs already measured, {len(jobs)} to run")
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "kernel", "variant", "schedule", "threads", "rep", "qos",
                        "status", "time_s", "checksum", "correct", "serial_time_s"])
        for name, label, err in broken:     # compile failures are results too
            w.writerow([time.time(), name, label, "", "", "", a.qos, "compile_error",
                        "", "", False, refs[name]["time_s"]])
            print(f"  COMPILE ERROR {name}/{label}: {err.splitlines()[0] if err else ''}")
        timed_out = set()
        for rep in range(a.reps):
            random.shuffle(jobs)            # spread heat/drift over all configs
            for name, label, exe, t, sc in jobs:
                key = (name, label, sc, t)
                if key in timed_out:        # far slower than serial already: do not repeat
                    continue
                limit = max(20.0, 10.0 * refs[name]["time_s"])
                st, ts, c = run_once(exe, t, sizes.get(name, []), limit, a.qos, sc)
                if st == "timeout":
                    timed_out.add(key)
                good = st == "ok" and checksum_ok(name, c, refs[name]["checksum"])
                w.writerow([time.time(), name, label, sc, t, rep, a.qos, st, ts, c, good,
                            refs[name]["time_s"]])
                f.flush()                   # survive an interrupted run
                print(f"  rep {rep} {name}/{label} [{sc or '-'}] t={t:2d} {st} "
                      f"{'' if ts is None else f'{ts:.3f}s'} {'OK' if good else 'WRONG'}",
                      flush=True)
                time.sleep(a.cooldown)
    print(f"done: {len(jobs)} configs x {a.reps} reps -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--timeout", type=float, default=300)
    r = sub.add_parser("run")
    r.add_argument("--threads", default="1,2,4,6,8,10")
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--cooldown", type=float, default=1.0, help="seconds between runs")
    r.add_argument("--timeout", type=float, default=120)
    r.add_argument("--kernels", default="", help="comma list; default all")
    r.add_argument("--qos", choices=["default", "background"], default="default")
    r.add_argument("--out", default="timings.csv")
    r.add_argument("--variants", default="",
                   help="regex on variant label (e.g. '^(claude|serial|expert)'); default all")
    r.add_argument("--skip-done", action="store_true",
                   help="only run configs with fewer than --reps rows in the CSV")
    a = p.parse_args()
    cmd_calibrate(a) if a.cmd == "calibrate" else cmd_run(a)


if __name__ == "__main__":
    main()
