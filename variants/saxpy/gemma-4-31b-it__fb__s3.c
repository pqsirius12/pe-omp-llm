/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    /* 
     * The original program performs y[i] = a*x[i] + y[i] for all i, repeated 'reps' times.
     * This is a memory-bound operation. To maximize performance, we swap the loops.
     * Instead of streaming the entire arrays 'reps' times, we load each element 
     * of x and y once, perform the operation 'reps' times in registers, and write back.
     * This reduces memory bandwidth requirements from O(reps * n) to O(n).
     *
     * We use '#pragma omp parallel for simd' to both distribute the workload across 
     * cores and encourage the compiler to use SIMD instructions for the outer loop.
     * 'schedule(static)' is used to minimize scheduling overhead for this uniform workload.
     */
    #pragma omp parallel for simd schedule(static)
    for (long i = 0; i < n; i++) {
        double ax = a * x[i];
        double yi = y[i];
        // The inner loop must be kept to ensure bitwise identical results to the serial version,
        // as floating-point addition is not associative.
        for (int r = 0; r < reps; r++) {
            yi += ax;
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

