/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    /* 
     * This kernel is heavily memory-bound. The original serial version reads x[i] and y[i] 
     * from memory 'reps' times. By interchanging the loops (processing the 'reps' loop 
     * inside the 'n' loop), we reduce memory traffic by a factor of 'reps', as we 
     * load each element of x and y once, perform all updates in registers, and write 
     * the result back once.
     *
     * The Apple M4 has a heterogeneous architecture (P-cores and E-cores). A 'static' 
     * schedule would lead to load imbalance because E-cores are slower. 'guided' 
     * scheduling is used here to dynamically balance the load across the different 
     * core types while minimizing synchronization overhead.
     */
    #pragma omp parallel for schedule(guided)
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

