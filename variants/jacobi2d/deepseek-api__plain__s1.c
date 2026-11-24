/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    for (int t = 0; t < steps; t++) {
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < n - 1; i++) {
            const double * __restrict up = u;
            double * __restrict vp = v;
            for (int j = 1; j < n - 1; j++)
                vp[(long)i * n + j] = 0.2 * (up[(long)i * n + j] +
                    up[(long)(i - 1) * n + j] + up[(long)(i + 1) * n + j] +
                    up[(long)i * n + j - 1] + up[(long)i * n + j + 1]);
        }
        double *tmp = u; u = v; v = tmp;
    }
    if (steps % 2 == 1) {
        #pragma omp parallel for schedule(static)
        for (long k = 0; k < (long)n * n; k++) v[k] = u[k];
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
