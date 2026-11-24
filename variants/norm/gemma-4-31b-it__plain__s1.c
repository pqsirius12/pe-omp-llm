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
     * We use collapse(2) to combine the outer loop (reps) and inner loop (n) into a single 
     * iteration space of size (n * reps). This ensures that all available OpenMP threads 
     * are utilized even when 'reps' is smaller than the number of threads.
     * A reduction clause is used to safely aggregate the partial sums across threads.
     */
    #pragma omp parallel for reduction(+:s) collapse(2)
    for (int r = 0; r < reps; r++) {
        for (long i = 0; i < n; i++) {
            s += sqrt(fabs(x[i]));
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
