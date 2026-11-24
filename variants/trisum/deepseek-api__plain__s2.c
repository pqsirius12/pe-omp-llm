/* k04_trisum: out[i] = sum_{j<=i} x[j] / (1 + i - j).
 * IRREGULAR: row i costs i+1 operations, so cost grows linearly with i.
 * A plain static split gives the last thread far more work, even before
 * P/E-core asymmetry is added on top. Compute-bound, no large matrix. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, const double *x, double *out) {
    /* Cost of row i is (i+1), so total work ~ n^2/2. Schedule by chunks of
     * rows whose cumulative cost is balanced: since cost is linear in i,
     * equal ranges of i have roughly equal cost if we split on i itself.
     * But to actually balance, use dynamic scheduling with a reasonable
     * chunk; the linear-cost distribution means small chunks near the
     * start and larger absolute work per chunk near the end -- a guided
     * schedule naturally gives bigger chunks first (where work per row
     * is small) and smaller chunks later (where work per row is large),
     * which balances well. */
    #pragma omp parallel for schedule(guided, 16)
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
