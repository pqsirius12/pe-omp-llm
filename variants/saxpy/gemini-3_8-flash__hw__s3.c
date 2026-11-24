/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
#define TILE_SIZE 2048

static void kernel(long n, int reps, double a, const double *x, double *y) {
    if (n <= 0 || reps <= 0) return;

    const double * restrict px = x;
    double * restrict py = y;
    long num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    #pragma omp parallel for schedule(dynamic, 8)
    for (long t = 0; t < num_tiles; t++) {
        long b = t * TILE_SIZE;
        long b_end = (b + TILE_SIZE < n) ? (b + TILE_SIZE) : n;
        for (int r = 0; r < reps; r++) {
            #pragma clang loop vectorize(enable)
            for (long i = b; i < b_end; i++) {
                py[i] = a * px[i] + py[i];
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
