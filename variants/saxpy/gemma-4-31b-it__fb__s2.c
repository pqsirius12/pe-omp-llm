/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    /* 
     * The original kernel has a loop over 'reps' as the outer loop and 'n' as the inner.
     * With n=20M, the arrays x and y (approx. 320MB total) exceed most L3 caches.
     * This means the arrays are streamed from main memory 'reps' times, creating
     * a bottleneck of O(reps * n) memory traffic.
     * 
     * By interchanging the loops (processing the inner loop over 'reps' for each element i),
     * we reduce memory traffic from O(reps * n) to O(n). We load x[i] and y[i] once, 
     * perform all 'reps' additions in registers, and store the result once.
     * 
     * To ensure bit-for-bit identity with the serial version, we maintain the exact 
     * sequence of floating-point additions. Pre-calculating (a * x[i]) as a constant 
     * for the inner loop preserves this sequence while reducing the number of 
     * multiplications from (reps * n) to (n).
     * 
     * Using '#pragma omp parallel for simd' ensures that the outer loop is 
     * parallelized across threads and vectorized using SIMD instructions.
     */
    #pragma omp parallel for simd schedule(static)
    for (long i = 0; i < n; i++) {
        double xi = x[i];
        double yi = y[i];
        double ax = a * xi;
        for (int r = 0; r < reps; r++) {
            yi = ax + yi;
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

