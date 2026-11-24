/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(long n, const double *x, double *out) {
    /* Row i costs i+1. Use guided scheduling to balance the triangular work:
       guided hands out large chunks first and shrinks them toward the end,
       which minimizes the tail imbalance on the heavy rows. */
    #pragma omp parallel for schedule(guided, 64)
    for (long i = 0; i < n; i++) {
        double s = 0.0;
        for (long j = 0; j <= i; j++)
            s += x[j] / (double)(1 + i - j);
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
