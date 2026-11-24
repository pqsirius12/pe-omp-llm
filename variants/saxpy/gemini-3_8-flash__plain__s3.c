/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    const double * __restrict__ rx = x;
    double * __restrict__ ry = y;
    const long TILE = 1024;

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i += TILE) {
        long end = (i + TILE < n) ? i + TILE : n;
        for (int r = 0; r < reps; r++) {
            #pragma clang loop vectorize(enable)
            for (long j = i; j < end; j++) {
                ry[j] = a * rx[j] + ry[j];
            }
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
