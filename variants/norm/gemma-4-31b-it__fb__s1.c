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
     * The computation sqrt(fabs(x[i])) is independent of the 'reps' loop.
     * By hoisting the 'reps' loop and calculating the value once per element,
     * we reduce the number of expensive sqrt calls from (n * reps) to (n).
     * We then use OpenMP's parallel for reduction to distribute the load
     * across threads and efficiently aggregate the sum.
     */
    #pragma omp parallel for reduction(+:s)
    for (long i = 0; i < n; i++) {
        s += sqrt(fabs(x[i])) * reps;
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

