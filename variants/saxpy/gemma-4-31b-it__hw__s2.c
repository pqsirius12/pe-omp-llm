/* k01_saxpy: y = a*x + y, repeated. UNIFORM cost per iteration, memory-bound. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, double a, const double *x, double *y) {
    /* 
     * The original program performs the operation y[i] = a * x[i] + y[i] for all i, 
     * repeated 'reps' times. In the original loop order, the arrays are read from 
     * and written to memory 'reps' times, making the program heavily memory-bound.
     * 
     * By swapping the loop order (processing all repetitions for a single index i 
     * before moving to the next), we reduce the memory traffic from O(reps * n) 
     * to O(n). The values x[i] and y[i] are loaded into registers once, 
     * updated 'reps' times, and written back once.
     * 
     * To handle the heterogeneous nature of the Apple M4 (4 P-cores and 6 E-cores),
     * we use 'schedule(guided)'. This balances the load by dynamically assigning
     * work chunks, ensuring that faster P-cores do more work and don't wait
     * on the slower E-cores.
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
