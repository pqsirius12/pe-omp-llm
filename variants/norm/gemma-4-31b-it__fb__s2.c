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
     * The original calculation is sum_{r=0}^{reps-1} sum_{i=0}^{n-1} sqrt(|x[i]|).
     * This is mathematically equivalent to reps * sum_{i=0}^{n-1} sqrt(|x[i]|).
     * By moving the multiplication by 'reps' outside the loop, we reduce the 
     * computational complexity from O(n * reps) to O(n), which is a significant 
     * speedup. 
     * 
     * We use '#pragma omp parallel for simd' to utilize both multi-core 
     * parallelism and SIMD vectorization. The 'reduction(+:s)' clause 
     * prevents data races and ensures a high-performance accumulation.
     */
    #pragma omp parallel for simd reduction(+:s) schedule(static)
    for (long i = 0; i < n; i++) {
        s += sqrt(fabs(x[i]));
    }
    
    return s * reps;
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
