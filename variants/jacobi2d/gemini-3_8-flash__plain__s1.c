/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    long nn = (long)n * n;

    #pragma omp parallel
    {
        double *cur_u = u;
        double *cur_v = v;

        for (int t = 0; t < steps; t++) {
            #pragma omp for schedule(static)
            for (int i = 1; i < n - 1; i++) {
                const double * restrict ru_row = cur_u + (long)i * n;
                const double * restrict ru_prev = ru_row - n;
                const double * restrict ru_next = ru_row + n;
                double * restrict rv_row = cur_v + (long)i * n;

                #pragma omp simd
                for (int j = 1; j < n - 1; j++) {
                    rv_row[j] = 0.2 * (ru_row[j] +
                        ru_prev[j] + ru_next[j] +
                        ru_row[j - 1] + ru_row[j + 1]);
                }
            }
            /* Implicit barrier at the end of 'omp for' ensures all reads and writes
             * for sweep t are finished before any thread updates cur_u / cur_v. */
            double *tmp = cur_u; cur_u = cur_v; cur_v = tmp;
        }

        /* After an odd number of sweeps the newest data is in caller's v (now cur_u):
         * copy it back so caller reads the result from u (now cur_v). */
        if (steps % 2 == 1) {
            #pragma omp for schedule(static)
            for (long k = 0; k < nn; k++) {
                cur_v[k] = cur_u[k];
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
