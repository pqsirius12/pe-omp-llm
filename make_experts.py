#!/usr/bin/env python3
"""make_experts.py - write the expert (human baseline) parallel version of
every kernel into variants/<kernel>/expert.c.

Each expert version uses schedule(runtime), so run.py can try static,
dynamic and guided scheduling on the SAME code through the OMP_SCHEDULE
environment variable. Edits are exact string replacements on the serial
kernel; the script stops if any expected line is missing.
"""
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
P = "_Pragma(\"omp parallel for schedule(runtime)%s\")\n"

def pf(indent, extra=""):
    return " " * indent + P % extra

# kernel file -> list of (exact text to find, replacement)
EDITS = {
 "k01_saxpy": [("        for (long i = 0; i < n; i++)\n            y[i]",
                pf(8) + "        for (long i = 0; i < n; i++)\n            y[i]")],
 "k02_matmul": [("    for (int i = 0; i < n; i++) {", pf(4) + "    for (int i = 0; i < n; i++) {")],
 "k03_jacobi2d": [("        for (int i = 1; i < n - 1; i++)", pf(8) + "        for (int i = 1; i < n - 1; i++)")],
 "k04_trisum": [("    for (long i = 0; i < n; i++) {\n        double s", pf(4) + "    for (long i = 0; i < n; i++) {\n        double s")],
 "k05_mandelbrot": [("    for (int py = 0; py < h; py++) {", pf(4) + "    for (int py = 0; py < h; py++) {")],
 "k06_spmv": [("        for (long i = 0; i < n; i++) {", pf(8) + "        for (long i = 0; i < n; i++) {")],
 "k07_nbody": [("    for (int i = 0; i < n; i++) {\n        double sx", pf(4) + "    for (int i = 0; i < n; i++) {\n        double sx")],
 "k08_histogram": [("        for (long i = 0; i < n; i++)\n            bins",
                    pf(8, " reduction(+:bins[:NBINS])") + "        for (long i = 0; i < n; i++)\n            bins")],
 "k09_norm": [("        for (long i = 0; i < n; i++)\n            s +=",
               pf(8, " reduction(+:s)") + "        for (long i = 0; i < n; i++)\n            s +=")],
 "k11_primes": [("    for (long i = 2; i < n; i++) {", pf(4, " reduction(+:count)") + "    for (long i = 2; i < n; i++) {")],
 "k12_collatz": [("    for (long i = 1; i < n; i++) {", pf(4) + "    for (long i = 1; i < n; i++) {")],
 "k13_conv2d": [("    for (int y = 0; y < oh; y++)", pf(4) + "    for (int y = 0; y < oh; y++)")],
 "k14_kmeans": [("        for (long i = 0; i < n; i++) {               /* assignment */",
                 pf(8) + "        for (long i = 0; i < n; i++) {               /* assignment */"),
                ("        for (long i = 0; i < n; i++) {\n            cnt[label[i]]++;",
                 pf(8, " reduction(+:sum[:K * D], cnt[:K])") +
                 "        for (long i = 0; i < n; i++) {\n            cnt[label[i]]++;")],
 "k15_cutoff": [("    for (long i = 0; i < n; i++) {\n        int c = 0;", pf(4) + "    for (long i = 0; i < n; i++) {\n        int c = 0;")],
}

# prefix sum cannot use a plain parallel for: replace the kernel body with a
# two-pass block scan (each thread scans its block, then adds the total of
# all blocks before it). It splits work statically by design.
PREFIX_OLD = """    for (int r = 0; r < reps; r++) {
        long run = 0;
        for (long i = 0; i < n; i++) {
            run += x[i];
            out[i] = run;
        }
    }"""
PREFIX_NEW = """    long partial[1024];                      /* block totals, one per thread */
    for (int r = 0; r < reps; r++) {
        #pragma omp parallel
        {
            int t = omp_get_thread_num(), nt = omp_get_num_threads();
            long lo = n * t / nt, hi = n * (t + 1) / nt, run = 0;
            for (long i = lo; i < hi; i++) { run += x[i]; out[i] = run; }  /* pass 1 */
            partial[t + 1] = run;
            #pragma omp barrier
            #pragma omp single
            { partial[0] = 0; for (int k = 1; k <= nt; k++) partial[k] += partial[k - 1]; }
            long off = partial[t];                                  /* implied barrier above */
            for (long i = lo; i < hi; i++) out[i] += off;           /* pass 2 */
        }
    }"""

def main():
    for name, edits in list(EDITS.items()) + [("k10_prefix", [(PREFIX_OLD, PREFIX_NEW)])]:
        src = open(os.path.join(ROOT, "kernels", name + ".c")).read()
        for old, new in edits:
            if src.count(old) != 1:
                sys.exit(f"{name}: expected text found {src.count(old)} times:\n{old}")
            src = src.replace(old, new)
        if name == "k10_prefix":
            src = src.replace('#include "common.h"', '#include <omp.h>\n#include "common.h"')
        out_dir = os.path.join(ROOT, "variants", name[4:])
        os.makedirs(out_dir, exist_ok=True)
        open(os.path.join(out_dir, "expert.c"), "w").write(src)
        print("wrote", name[4:])

if __name__ == "__main__":
    main()
