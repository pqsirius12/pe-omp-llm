/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    /* 
     * To maximize performance, we swap the loops to improve cache locality.
     * The original version streams the arrays from memory 'reps' times.
     * The swapped version loads each element once, processes it 'reps' times, 
     * and writes it back, reducing memory bandwidth requirements by a factor of 'reps'.
     * Each y[i] is independent, making this perfectly parallelizable across i.
     */
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        double ax = a * x[i];
        double yi = y[i];
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
