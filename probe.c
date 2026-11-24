/* probe.c - where does macOS put OpenMP threads, and how fast is each one?
 * Every thread runs the SAME fixed work and times itself. Fast threads sat
 * on P-cores, slow ones on E-cores (or were shared). Run with
 * OMP_NUM_THREADS = 1..10; prints each thread's time, sorted.
 * Two workloads: an integer dependency chain (like collatz) and a
 * floating-point one (like nbody), since core types differ more on some. */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Read at run time, so the compiler cannot precompute the loops. */
volatile double FA = 0.9999999, FB = 1e-7;
volatile unsigned long IA = 3, IB = 1;
double sink[256];          /* each thread's final value is kept here and printed */

static int cmp(const void *a, const void *b) {
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}

int main(int argc, char **argv) {
    long work = argc > 1 ? atol(argv[1]) : 300000000L;
    for (int kind = 0; kind < 2; kind++) {
        double t[256];
        int nt = 1;
        #pragma omp parallel
        {
            #pragma omp single
            nt = omp_get_num_threads();       /* implied barrier: start together */
            int id = omp_get_thread_num();
            double t0 = omp_get_wtime();
            if (kind == 0) {                  /* integer chain: each step needs the last */
                unsigned long v = 27 + id, a = IA, b = IB;
                for (long i = 0; i < work; i++) v = (v & 1) ? a * v + b : (v >> 1) | 7;
                sink[id] = (double)v;
            } else {                          /* floating-point chain: each step needs the last */
                double x = 1.0 + id, a = FA, b = FB;
                for (long i = 0; i < work / 2; i++) x = x * a + b;
                sink[id] = x;
            }
            t[id] = omp_get_wtime() - t0;
        }
        qsort(t, nt, sizeof t[0], cmp);
        printf("%s threads=%2d times:", kind ? "float" : "int  ", nt);
        for (int i = 0; i < nt; i++) printf(" %.3f", t[i]);
        double chk = 0; for (int i = 0; i < nt; i++) chk += sink[i];
        printf("  (check %.3g)\n", chk);   /* printing the results keeps the work alive */
    }
    return 0;
}
