/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    for (int t = 0; t < steps; t++) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < n - 1; i++) {
            const double * __restrict__ up = u + (long)i * n;
            const double * __restrict__ um = u + (long)(i - 1) * n;
            const double * __restrict__ uq = u + (long)(i + 1) * n;
            double * __restrict__ vp = v + (long)i * n;
            for (int j = 1; j < n - 1; j++)
                vp[j] = 0.2 * (up[j] + um[j] + uq[j] + up[j - 1] + up[j + 1]);
        }
        double *tmp = u; u = v; v = tmp;   /* result of sweep t feeds sweep t+1 */
    }
    /* After an odd number of sweeps the newest data is in the caller's v:
     * copy it back so the caller always reads the result from u. */
    if (steps % 2 == 1) {
        long total = (long)n * n;
        #pragma omp parallel for schedule(static)
        for (long k = 0; k < total; k++) v[k] = u[k];
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
