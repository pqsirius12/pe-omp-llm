/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    /* 
     * The original program has a nested loop structure:
     * for (int r = 0; r < reps; r++)
     *     for (long i = 0; i < n; i++)
     *         y[i] = a * x[i] + y[i];
     * 
     * This is memory-bound because it reads and writes the entire x and y 
     * arrays 'reps' times. To maximize performance, we swap the loops.
     * By processing each element y[i] for all 'reps' iterations before moving 
     * to the next element, we reduce memory traffic by a factor of 'reps', 
     * keeping the data in registers/L1 cache.
     * 
     * Regarding the hardware (Apple M4 P-cores and E-cores):
     * Because the CPU cores have different performance characteristics, 
     * a 'static' schedule would lead to load imbalance (P-cores finishing 
     * and waiting for E-cores). 'schedule(guided)' is used to dynamically 
     * balance the load by starting with large chunks and decreasing them over 
     * time, minimizing scheduling overhead while maintaining balance.
     */
    #pragma omp parallel for schedule(guided)
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

