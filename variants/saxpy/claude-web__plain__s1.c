/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
#define SAXPY_BLOCK 2048L   /* 2048 doubles of x + y = 32 KB, stays in L1/L2 */

static void kernel(long n, int reps, double a, const double *x, double *y) {
    if (n <= 0 || reps <= 0) return;
    long nblocks = (n + SAXPY_BLOCK - 1) / SAXPY_BLOCK;

    #pragma omp parallel for schedule(static)
    for (long b = 0; b < nblocks; b++) {
        long start = b * SAXPY_BLOCK;
        long end = start + SAXPY_BLOCK;
        if (end > n) end = n;
        const double *restrict xb = x + start;
        double *restrict yb = y + start;
        long len = end - start;
        /* All repetitions on this cache-resident block. Each y[i] receives the
         * same sequence of operations as in the serial code, so the result is
         * bit-identical. */
        for (int r = 0; r < reps; r++) {
            #pragma omp simd
            for (long i = 0; i < len; i++)
                yb[i] = a * xb[i] + yb[i];
        }
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 20000000);
    int reps = (int)arg_long(argc, argv, 2, 20);
    double *x = xmalloc(n * sizeof *x), *y = xmalloc(n * sizeof *y);
    rng_seed(1);
    for (long i = 0; i < n; i++) { x[i] = rng_unit(); y[i] = rng_unit(); }

    double t0 = now_sec();
    kernel(n, reps, 0.5, x, y);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 0; i < n; i++) sum += y[i];
    report("saxpy", t1 - t0, sum);
    free(x); free(y);
    return 0;
}
