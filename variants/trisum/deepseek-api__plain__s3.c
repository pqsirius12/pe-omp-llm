/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(long n, const double *x, double *out) {
    /* Triangular workload: row i costs (i+1) flops. Use dynamic scheduling
     * over row-blocks so load balances across asymmetric cores.
     * Small blocks amortize overhead while keeping balance fine-grained. */
    long block = 256;
    long nblocks = (n + block - 1) / block;
    #pragma omp parallel for schedule(dynamic, 1)
    for (long b = 0; b < nblocks; b++) {
        long i0 = b * block;
        long i1 = i0 + block;
        if (i1 > n) i1 = n;
        for (long i = i0; i < i1; i++) {
            double s = 0.0;
            for (long j = 0; j <= i; j++)
                s += x[j] / (double)(1 + i - j);
            out[i] = s;
        }
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
