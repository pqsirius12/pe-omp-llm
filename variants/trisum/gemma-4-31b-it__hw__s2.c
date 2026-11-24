/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, const double *x, double *out) {
    /* 
     * The workload increases linearly with i, meaning the total work for row i 
     * is O(i). This creates a severe load imbalance if using static scheduling.
     * Furthermore, the target machine has asymmetric P-cores and E-cores.
     * 
     * 'schedule(dynamic, 1)' is chosen because:
     * 1. It perfectly handles the triangular work distribution by letting threads 
     *    request new rows as they finish, ensuring no core is idle while work remains.
     * 2. It automatically balances the P/E core asymmetry: faster P-cores will 
     *    naturally process more rows than the slower E-cores.
     * 3. With n = 40,000, the overhead of the dynamic scheduler (atomic increments) 
     *    is negligible compared to the O(n^2) total floating-point divisions.
     */
    #pragma omp parallel for schedule(dynamic, 1)
    for (long i = 0; i < n; i++) {
        double s = 0.0;
        /* 
         * The inner loop is a simple reduction. We use #pragma omp simd to 
         * encourage the compiler to utilize SIMD units for the divisions 
         * and additions, which is critical for compute-bound kernels.
         */
        #pragma omp simd reduction(+:s)
        for (long j = 0; j <= i; j++) {
            s += x[j] / (double)(1 + i - j);
        }
        out[i] = s;
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 40000);
    double *x = xmalloc(n * sizeof *x), *out = xmalloc(n * sizeof *out);
    rng_seed(4);
    for (long i = 0; i < n; i++) x[i] = rng_unit();

    double t0 = now_sec();
    kernel(n, x, out);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 0; i < n; i++) sum += out[i];
    report("trisum", t1 - t0, sum);
    free(x); free(out);
    return 0;
}

