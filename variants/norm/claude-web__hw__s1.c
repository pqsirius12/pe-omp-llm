/* k09_norm: sum of sqrt(|x_i|) over a large array.
 * UNIFORM cost; tests whether the LLM uses a reduction clause.
 * Floating-point sums in a different order differ in the last digits,
 * so the harness compares this checksum with a relative tolerance. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static double kernel(long n, int reps, const double *x) {
    /* Block of 4096 doubles = 32 KB: fits in L1 on both P- and E-cores,
     * so all reps over a block hit cache instead of re-streaming DRAM. */
    const long B = 4096;
    long nblocks = (n + B - 1) / B;
    double s = 0.0;

    if (n <= 0 || reps <= 0) return 0.0;

    /* Dynamic scheduling balances work between fast P-cores and slower
     * E-cores; chunk of 8 blocks (256 KB) keeps scheduling overhead low. */
    #pragma omp parallel for schedule(dynamic, 8) reduction(+:s)
    for (long b = 0; b < nblocks; b++) {
        long lo = b * B;
        long hi = lo + B < n ? lo + B : n;
        double t = 0.0;
        for (int r = 0; r < reps; r++) {
            double u = 0.0;
            #pragma omp simd reduction(+:u)
            for (long i = lo; i < hi; i++)
                u += sqrt(fabs(x[i]));
            t += u;
        }
        s += t;
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
