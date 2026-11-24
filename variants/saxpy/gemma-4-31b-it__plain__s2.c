/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    /* 
     * The original code has a loop over 'reps' (outer) and 'n' (inner).
     * Since n is very large (20M), the arrays x and y do not fit in the cache.
     * In the original version, the arrays are streamed from memory 'reps' times.
     * 
     * By interchanging the loops, we process each element y[i] for all 'reps'
     * iterations while it is still in a register or L1 cache. This reduces 
     * memory traffic from O(reps * n) to O(n), providing a massive speedup.
     * 
     * The operation sequence for each y[i] remains identical to the serial version,
     * ensuring the same floating-point results.
     */
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        double xi = x[i];
        double yi = y[i];
        for (int r = 0; r < reps; r++) {
            yi = a * xi + yi;
        }
        y[i] = yi;
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

