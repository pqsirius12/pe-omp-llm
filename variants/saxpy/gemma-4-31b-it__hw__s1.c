/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    /* 
     * The original algorithm is memory-bound, accessing the arrays 'x' and 'y' 
     * 'reps' times. By swapping the order of the loops (tiling), we can 
     * load each element of 'x' and 'y' into the cache once and perform all 
     * 'reps' updates before moving to the next element.
     *
     * On the Apple M4 (heterogeneous P-cores and E-cores), 'schedule(guided)' 
     * is used to balance the load, ensuring that faster P-cores handle more 
     * work than slower E-cores, preventing the "convoy effect".
     */
    #pragma omp parallel for schedule(guided)
    for (long i = 0; i < n; i++) {
        double xi = x[i];
        double yi = y[i];
        double term = a * xi;
        for (int r = 0; r < reps; r++) {
            yi += term;
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
