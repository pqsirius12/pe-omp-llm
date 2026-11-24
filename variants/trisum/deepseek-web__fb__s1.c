/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"
#include <math.h>
#include <omp.h>

/* === KERNEL BEGIN === */

/* Smallest m >= 0 such that m*(m+1)/2 >= target, where m*(m+1)/2 is the
   cumulative work done by rows 0..m-1 (row i costs i+1 operations). */
static inline long trisum_row_geq(double target) {
    if (target <= 0.0) return 0;
    long m = (long)(0.5 * (-1.0 + sqrt(1.0 + 8.0 * target)));
    if (m < 0) m = 0;
    while ((double)m * (double)(m + 1) * 0.5 < target) m++;
    while (m > 0 && (double)(m - 1) * (double)m * 0.5 >= target) m--;
    return m;
}

static void kernel(long n, const double *x, double *out) {
    /* Total work = sum_{i=0}^{n-1} (i+1) = n*(n+1)/2. */
    const double W = (double)n * (double)(n + 1) * 0.5;

    /* Manual static partition: each thread gets a contiguous range of rows
       whose total (triangular) work is ~ W/P.  Boundaries are computed from
       the same cumulative-work function, so adjacent ranges meet exactly. */
    #pragma omp parallel
    {
        const int P = omp_get_num_threads();
        const int t = omp_get_thread_num();

        long a = (t == 0) ? 0
                          : trisum_row_geq(W * (double)t / (double)P);
        long b = (t == P - 1) ? n
                              : trisum_row_geq(W * (double)(t + 1) / (double)P);
        if (a < 0) a = 0;
        if (b > n) b = n;
        if (a > b) a = b;

        for (long i = a; i < b; i++) {
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
