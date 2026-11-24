#!/usr/bin/env python3
"""make_calib.py - write variants/<kernel>/calib.c: the calibrated uneven
split baseline (pe_split.h) for the five equal-work kernels with a single
independent outer loop. Uneven-work kernels are left out on purpose: a
split by core speed alone cannot fix work that differs per iteration."""
import os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
INC = ('#include "common.h"', '#include "common.h"\n#include "pe_split.h"')

EDITS = {
 "k01_saxpy": [("""    for (int r = 0; r < reps; r++)
        for (long i = 0; i < n; i++)
            y[i] = a * x[i] + y[i];""",
"""    pe_calibrate();
    #pragma omp parallel
    {
        long b, e;
        pe_range(0, n, &b, &e);
        for (int r = 0; r < reps; r++)       /* each element is independent: no barrier needed */
            for (long i = b; i < e; i++)
                y[i] = a * x[i] + y[i];
    }""")],
 "k02_matmul": [("    for (int i = 0; i < n; i++) {\n        for (int j = 0; j < n; j++) C",
"""    pe_calibrate();
    #pragma omp parallel
    {
    long b, e;
    pe_range(0, n, &b, &e);
    for (int i = (int)b; i < (int)e; i++) {
        for (int j = 0; j < n; j++) C"""),
  ("                C[(long)i * n + j] += a * B[(long)k * n + j];\n        }\n    }\n}",
   "                C[(long)i * n + j] += a * B[(long)k * n + j];\n        }\n    }\n    }\n}")],
 "k07_nbody": [("    for (int i = 0; i < n; i++) {\n        double sx",
"""    pe_calibrate();
    #pragma omp parallel
    {
    long b, e;
    pe_range(0, n, &b, &e);
    for (int i = (int)b; i < (int)e; i++) {
        double sx"""),
  ("        ax[i] = sx; ay[i] = sy; az[i] = sz;\n    }\n}",
   "        ax[i] = sx; ay[i] = sy; az[i] = sz;\n    }\n    }\n}")],
 "k09_norm": [("""    double s = 0.0;
    for (int r = 0; r < reps; r++)
        for (long i = 0; i < n; i++)
            s += sqrt(fabs(x[i]));""",
"""    double s = 0.0;
    pe_calibrate();
    #pragma omp parallel reduction(+:s)
    {
        long b, e;
        pe_range(0, n, &b, &e);
        for (int r = 0; r < reps; r++)
            for (long i = b; i < e; i++)
                s += sqrt(fabs(x[i]));
    }""")],
 "k13_conv2d": [("    for (int y = 0; y < oh; y++)\n        for (int x = 0; x < ow; x++) {",
"""    pe_calibrate();
    #pragma omp parallel
    {
    long b, e;
    pe_range(0, oh, &b, &e);
    for (int y = (int)b; y < (int)e; y++)
        for (int x = 0; x < ow; x++) {"""),
  ("            out[(long)y * ow + x] = s;\n        }\n}",
   "            out[(long)y * ow + x] = s;\n        }\n    }\n}")],
}

for name, edits in EDITS.items():
    src = open(os.path.join(ROOT, "kernels", name + ".c")).read()
    for old, new in [INC] + edits:
        if src.count(old) != 1:
            sys.exit(f"{name}: expected text found {src.count(old)} times:\n{old}")
        src = src.replace(old, new)
    d = os.path.join(ROOT, "variants", name[4:])
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "calib.c"), "w").write(src)
    print("wrote", name[4:], "calib")
