/* k09_norm: sum of sqrt(|x_i|) over a large array.
 * UNIFORM cost; tests whether the LLM uses a reduction clause.
 * Floating-point sums in a different order differ in the last digits,
 * so the harness compares this checksum with a relative tolerance. */
#include <math.h>
#include "common.h"

/* === KERNEL BEGIN === */
static double kernel(long n, int reps, const double *x) {
    double s = 0.0;
    /*
     * The target machine is an Apple M4 with heterogeneous cores (P-cores and E-cores).
     * To maximize performance:
     * 1. Use 'collapse(2)' to merge the loops into a single large iteration space, 
     *    reducing overhead and maximizing the distribution of work.
     * 2. Use 'reduction(+:s)' to safely and efficiently accumulate the sum across threads.
     * 3. Use 'schedule(guided)' to handle the performance imbalance between P-cores 
     *    and E-cores; guided scheduling starts with large chunks and reduces them, 
     *    allowing faster cores to take more work.
     * 4. Use 'simd' to encourage the compiler to use NEON/AMX vector instructions 
     *    for the sqrt and fabs operations.
     */
    #pragma omp parallel for simd reduction(+:s) schedule(guided) collapse(2)
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

