/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
#if defined(__aarch64__) || defined(__arm64__)
#define SAXPY_TILE 32   /* 32 doubles = 16 NEON regs, of 32 available */
#else
#define SAXPY_TILE 16   /* 16 doubles = 8 SSE / 4 AVX regs, of 16 available */
#endif

static void kernel(long n, int reps, double a, const double *x, double *y) {
    if (n <= 0 || reps <= 0) return;
    const long ntiles = n / SAXPY_TILE;
    const long tail = ntiles * SAXPY_TILE;

    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (long t = 0; t < ntiles; t++) {
            const double *xb = x + t * SAXPY_TILE;
            double *yb = y + t * SAXPY_TILE;
            double yv[SAXPY_TILE];
            double xv[SAXPY_TILE];

            #pragma clang loop unroll(full)
            for (int k = 0; k < SAXPY_TILE; k++) { yv[k] = yb[k]; xv[k] = xb[k]; }

            /* All repetitions on a register-resident tile. Each element sees
             * exactly the same operation sequence as the serial code. */
            for (int r = 0; r < reps; r++) {
                #pragma clang loop unroll(full)
                for (int k = 0; k < SAXPY_TILE; k++)
                    yv[k] = a * xv[k] + yv[k];
            }

            #pragma clang loop unroll(full)
            for (int k = 0; k < SAXPY_TILE; k++) yb[k] = yv[k];
        }

        /* Leftover elements (< SAXPY_TILE): one thread handles them. */
        #pragma omp single nowait
        for (long i = tail; i < n; i++) {
            double xi = x[i], yi = y[i];
            for (int r = 0; r < reps; r++)
                yi = a * xi + yi;
            y[i] = yi;
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
