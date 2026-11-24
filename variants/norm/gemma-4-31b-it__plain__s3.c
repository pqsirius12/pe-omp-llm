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
     * We parallelize the loop over the array elements (n), which is the largest 
     * dimension. By swapping the loop order (processing the repetition count 
     * 'reps' in the inner loop), we significantly improve cache locality, 
     * reading each element of x from memory only once instead of 'reps' times. 
     * We use an OpenMP reduction clause to ensure a thread-safe and efficient 
     * summation of the total.
     */
    #pragma omp parallel for reduction(+:s)
    for (long i = 0; i < n; i++) {
        double val = sqrt(fabs(x[i]));
        for (int r = 0; r < reps; r++) {
            s += val;
        }
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
