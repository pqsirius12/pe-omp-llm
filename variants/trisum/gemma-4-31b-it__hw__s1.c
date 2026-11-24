/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, const double *x, double *out) {
    /* 
     * The workload for each iteration i is O(i), meaning it grows linearly.
     * A static schedule would cause severe load imbalance, especially given the 
     * P-core and E-core asymmetry on the M4. 
     * 
     * 'dynamic' scheduling is ideal here because it allows faster cores (P-cores) 
     * and iterations with smaller workloads to be processed more quickly, while 
     * the heavier iterations at the end are distributed more evenly.
     * 
     * We use a small chunk size (e.g., 16) to balance the overhead of task 
     * distribution with the need for fine-grained load balancing.
     */
    #pragma omp parallel for schedule(dynamic, 16)
    for (long i = 0; i < n; i++) {
        double s = 0.0;
        /* 
         * The inner loop is a simple reduction. Modern compilers (like clang -O2)
         * can vectorize this division-heavy loop using SIMD instructions.
         */
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
