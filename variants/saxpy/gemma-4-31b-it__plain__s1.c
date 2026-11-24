/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    /* 
     * The original code has a memory-bound structure where the entire array is 
     * streamed from memory 'reps' times. To maximize performance, we use 
     * loop interchange to bring the 'i' loop to the outside. This allows 
     * each element of x and y to be loaded into registers once, updated 
     * 'reps' times, and written back once, reducing memory traffic by a 
     * factor of 'reps'.
     *
     * We use '#pragma omp parallel for simd' to distribute the workload across 
     * available threads and enable vectorization (SIMD).
     */
    #pragma omp parallel for simd
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
