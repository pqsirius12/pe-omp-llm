/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int r = 0; r < reps; r++) {
            /* nothing */
        }
        /* Each thread handles a contiguous static chunk across all reps,
         * maximizing locality and minimizing false sharing between threads. */
        long chunk = (n + omp_get_num_threads() - 1) / omp_get_num_threads();
        long lo = (long)omp_get_thread_num() * chunk;
        long hi = lo + chunk;
        if (hi > n) hi = n;
        for (int r = 0; r < reps; r++) {
            for (long i = lo; i < hi; i++)
                y[i] = a * x[i] + y[i];
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
