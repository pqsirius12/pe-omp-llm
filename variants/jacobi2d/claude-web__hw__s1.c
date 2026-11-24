/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
/* Rows per dynamic chunk: big enough to amortize scheduling and keep
 * hardware prefetch streams long, small enough that faster P-cores
 * naturally take more rows than the slower E-cores (no static imbalance). */
#define JAC_CHUNK 8

static void kernel(int n, int steps, double *u, double *v) {
    double *const cu = u;   /* caller's u */
    double *const cv = v;   /* caller's v */
    const long N = n;

    /* One parallel region for all sweeps: threads are created once.
     * Each thread keeps private copies of the u/v pointers and swaps them
     * itself; the implicit barrier at the end of each "omp for" is the
     * only synchronization needed between sweeps. */
    #pragma omp parallel firstprivate(u, v)
    {
        for (int t = 0; t < steps; t++) {
            const double *restrict a = u;
            double *restrict b = v;

            #pragma omp for schedule(dynamic, JAC_CHUNK)
            for (int i = 1; i < n - 1; i++) {
                const double *restrict up  = a + (long)(i - 1) * N;
                const double *restrict mid = a + (long)i * N;
                const double *restrict dn  = a + (long)(i + 1) * N;
                double *restrict out = b + (long)i * N;
                #pragma omp simd
                for (int j = 1; j < n - 1; j++)
                    out[j] = 0.2 * (mid[j] + up[j] + dn[j] +
                                    mid[j - 1] + mid[j + 1]);
            }   /* implicit barrier: sweep t complete before t+1 */

            double *tmp = u; u = v; v = tmp;
        }

        /* After an odd number of sweeps the newest data is in the caller's v:
         * copy it back so the caller always reads the result from u. */
        if (steps % 2 == 1) {
            const long nn = N * N;
            #pragma omp for schedule(static)
            for (long k = 0; k < nn; k++)
                cu[k] = cv[k];
        }
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    int n = (int)arg_long(argc, argv, 1, 2048);
    int steps = (int)arg_long(argc, argv, 2, 100);
    long nn = (long)n * n;
    double *u = xmalloc(nn * sizeof *u), *v = xmalloc(nn * sizeof *v);
    rng_seed(3);
    for (long k = 0; k < nn; k++) { u[k] = rng_unit(); v[k] = u[k]; }

    double t0 = now_sec();
    kernel(n, steps, u, v);
    double t1 = now_sec();

    double sum = 0.0;
    for (long k = 0; k < nn; k++) sum += u[k];
    report("jacobi2d", t1 - t0, sum);
    free(u); free(v);
    return 0;
}
