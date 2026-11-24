/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
#define RB 8

static void kernel(long n, const double *x, double *out) {
    long nb = n / RB;          /* full blocks of RB rows */
    long rem_start = nb * RB;  /* leftover rows handled scalar */

    #pragma omp parallel
    {
        /* Blocks processed largest-first with dynamic scheduling for balance.
         * Within a block, each row's terms are accumulated in exactly the
         * same order (j ascending) as the serial code, so results are
         * bit-identical. */
        #pragma omp for schedule(dynamic, 1) nowait
        for (long bb = 0; bb < nb; bb++) {
            long b = nb - 1 - bb;
            long i0 = b * RB;
            double s[RB];
            for (int k = 0; k < RB; k++) s[k] = 0.0;

            /* Common range j = 0..i0, shared by all RB rows. */
            for (long j = 0; j <= i0; j++) {
                double xj = x[j];
                double base = (double)(1 + i0 - j);
                #pragma omp simd
                for (int k = 0; k < RB; k++)
                    s[k] += xj / (base + (double)k);
            }
            /* Tails: row i0+k additionally needs j = i0+1 .. i0+k, in order. */
            for (int k = 1; k < RB; k++) {
                long i = i0 + k;
                double sk = s[k];
                for (long j = i0 + 1; j <= i; j++)
                    sk += x[j] / (double)(1 + i - j);
                s[k] = sk;
            }
            for (int k = 0; k < RB; k++) out[i0 + k] = s[k];
        }

        /* Remaining rows (fewer than RB), each done exactly as serial. */
        #pragma omp for schedule(dynamic, 1)
        for (long i = rem_start; i < n; i++) {
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
