/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    /* Persistent parallel region: avoids fork/join per sweep.
     * Strip-mined over rows; interior rows [1, n-2] split statically. */
    const int inner = n - 2;
    #pragma omp parallel if(steps > 0)
    {
        #pragma omp single
        {
            for (int t = 0; t < steps; t++) {
                #pragma omp taskloop grainsize(8) shared(u, v, n)
                for (int i = 1; i < n - 1; i++) {
                    const double *restrict up = u + (long)(i - 1) * n;
                    const double *restrict uc = u + (long)i * n;
                    const double *restrict un = u + (long)(i + 1) * n;
                    double *restrict vc = v + (long)i * n;
                    for (int j = 1; j < inner + 1; j++)
                        vc[j] = 0.2 * (uc[j] + up[j] + un[j] + uc[j - 1] + uc[j + 1]);
                }
                double *tmp = u; u = v; v = tmp;
            }
            if (steps % 2 == 1) {
                long nn = (long)n * n;
                #pragma omp taskloop grainsize(4096) shared(u, v)
                for (long k = 0; k < nn; k++) v[k] = u[k];
            }
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
