/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, const double *x, double *out) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();

        /* Cost of row i is proportional to i+1.
           Partition by equal cumulative work rather than equal row count.
           The cumulative work through row k is k*(k+1)/2. */
        long total_work = n * (n + 1) / 2;

        long lo_work = (long)((double)total_work * tid / nt);
        long hi_work = (long)((double)total_work * (tid + 1) / nt);

        long lo = (long)((sqrt(1.0 + 8.0 * lo_work) - 1.0) * 0.5);
        long hi = (long)((sqrt(1.0 + 8.0 * hi_work) - 1.0) * 0.5);

        if (lo < 0) lo = 0;
        if (hi >= n) hi = n - 1;

        /* Correct rounding at the partition boundaries. */
        while (lo > 0 && lo * (lo + 1) / 2 > lo_work) lo--;
        while (lo < n && lo * (lo + 1) / 2 < lo_work) lo++;
        while (hi > 0 && hi * (hi + 1) / 2 > hi_work) hi--;
        while (hi < n && hi * (hi + 1) / 2 < hi_work) hi++;

        #pragma omp for schedule(static)
        for (long i = 0; i < n; i++) {
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
