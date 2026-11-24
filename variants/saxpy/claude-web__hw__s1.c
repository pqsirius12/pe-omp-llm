/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
/* Loop interchange + cache blocking:
 *  - Each element's update chain (reps applications of y = a*x + y) is
 *    independent of every other element, so we process the array in
 *    L1-sized blocks and apply all reps to a block while it is cache-hot.
 *    Per element the exact same operations happen in the same order as the
 *    serial code, so results are bit-identical.
 *  - DRAM traffic drops from ~reps*3*n*8 bytes to ~3*n*8 bytes.
 *  - schedule(dynamic) balances work between fast P-cores and slow E-cores. */
#define SAXPY_BLOCK 2048L   /* 2048 doubles of x + y = 32 KiB: fits L1 on P and E cores */

static void kernel(long n, int reps, double a, const double *x, double *y) {
    if (n <= 0 || reps <= 0) return;
    long nblocks = (n + SAXPY_BLOCK - 1) / SAXPY_BLOCK;

    #pragma omp parallel for schedule(dynamic, 4)
    for (long b = 0; b < nblocks; b++) {
        long lo = b * SAXPY_BLOCK;
        long hi = lo + SAXPY_BLOCK;
        if (hi > n) hi = n;
        const double *restrict xb = x + lo;
        double *restrict yb = y + lo;
        long len = hi - lo;
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
