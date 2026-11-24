/* k09_norm: sum of sqrt(|x_i|) over a large array.
 * UNIFORM cost; tests whether the LLM uses a reduction clause.
 * Floating-point sums in a different order differ in the last digits,
 * so the harness compares this checksum with a relative tolerance. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static double kernel(long n, int reps, const double *x) {
    if (reps <= 0 || n <= 0)
        return 0.0;

    /* Every repetition sums exactly the same terms, so the array is
     * traversed once (the kernel is memory-bound: this cuts traffic by
     * a factor of reps). The single pass is split statically across
     * threads and vectorized; partial sums are combined by the reduction. */
    double t = 0.0;
    #pragma omp parallel for simd reduction(+:t) schedule(static)
    for (long i = 0; i < n; i++)
        t += sqrt(fabs(x[i]));

    /* Accumulate the per-pass sum reps times, as the serial version does. */
    double s = 0.0;
    for (int r = 0; r < reps; r++)
        s += t;
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
