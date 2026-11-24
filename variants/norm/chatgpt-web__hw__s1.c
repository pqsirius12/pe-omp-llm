/* k09_norm: sum of sqrt(|x_i|) over a large array.
 * UNIFORM cost; tests whether the LLM uses a reduction clause.
 * Floating-point sums in a different order differ in the last digits,
 * so the harness compares this checksum with a relative tolerance. */
#include <math.h>
#include "common.h"

/* === KERNEL BEGIN === */
static double kernel(long n, int reps, const double *x) {
    double s = 0.0;

    #pragma omp parallel
    {
        double local_sum = 0.0;

        #pragma omp for schedule(static)
        for (long i = 0; i < n; i++) {
            local_sum += sqrt(fabs(x[i]));
        }

        #pragma omp atomic update
        s += local_sum;
    }

    /* Repeat the accumulation for the requested number of repetitions. */
    if (reps > 1) {
        double first_sum = s;
        s = 0.0;

        #pragma omp parallel
            #pragma omp for reduction(+:s) schedule(static)
            for (long r = 1; r < reps; r++) {
                s += first_sum;
            }

        s += first_sum;
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
