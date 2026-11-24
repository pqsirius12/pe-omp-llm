/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    #pragma omp parallel
    {
        double *cur_u = u;
        double *cur_v = v;

        for (int t = 0; t < steps; t++) {
            #pragma omp for schedule(static)
            for (int i = 1; i < n - 1; i++) {
                const double * __restrict__ u_m1 = cur_u + (long)(i - 1) * n;
                const double * __restrict__ u_0  = cur_u + (long)i * n;
                const double * __restrict__ u_p1 = cur_u + (long)(i + 1) * n;
                double * __restrict__ v_0        = cur_v + (long)i * n;

                #pragma omp simd
                for (int j = 1; j < n - 1; j++) {
                    v_0[j] = 0.2 * (u_0[j] +
                                    u_m1[j] +
                                    u_p1[j] +
                                    u_0[j - 1] +
                                    u_0[j + 1]);
                }
            }

            double *tmp = cur_u;
            cur_u = cur_v;
            cur_v = tmp;
        }

        if (steps % 2 == 1) {
            #pragma omp for schedule(static)
            for (long k = 0; k < (long)n * n; k++) {
                u[k] = v[k];
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
