/* k09_norm: sum of sqrt(|x_i|) over a large array.
 * UNIFORM cost; tests whether the LLM uses a reduction clause.
 * Floating-point sums in a different order differ in the last digits,
 * so the harness compares this checksum with a relative tolerance. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static double kernel(long n, int reps, const double *x) {
    double s = 0.0;

    /* 
     * The original program performs 'reps' passes over the array 'x'.
     * Since 'x' is read-only, we can process each element once and multiply 
     * by 'reps', which dramatically reduces memory bandwidth pressure (the primary bottleneck).
     * 
     * We use 'schedule(guided)' to balance the load across the Apple M4's 
     * heterogeneous architecture (4 Performance cores and 6 Efficiency cores), 
     * as the OS may assign threads to either type of core.
     * 
     * 'reduction(+:s)' is used to avoid data races and is more numerically 
     * stable than a global atomic sum.
     */
    #pragma omp parallel for reduction(+:s) schedule(guided)
    for (long i = 0; i < n; i++) {
        s += (double)reps * sqrt(fabs(x[i]));
    }

    return s;
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 50000000);
    int reps = (int)arg_long(argc, argv, 2, 6);
    double *x = xmalloc(n * sizeof *x);
    rng_seed(9);
    for (long i = 0; i < n; i++) x[i] = rng_unit() - 0.5;

    double t0 = now_sec();
    double s = kernel(n, reps, x);
    double t1 = now_sec();

    report("norm", t1 - t0, s);
    free(x);
    return 0;
}
