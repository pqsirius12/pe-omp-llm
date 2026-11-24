/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    double *buf[2] = {u, v};
    long nn = (long)n * n;

    #pragma omp parallel
    {
        for (int t = 0; t < steps; t++) {
            double *src = buf[t % 2];
            double *dst = buf[(t + 1) % 2];

            #pragma omp for schedule(static)
            for (int i = 1; i < n - 1; i++) {
                long row = (long)i * n;
                double *s_prev = &src[row - n];
                double *s_curr = &src[row];
                double *s_next = &src[row + n];
                double *d_curr = &dst[row];

                #pragma omp simd
                for (int j = 1; j < n - 1; j++) {
                    d_curr[j] = 0.2 * (s_curr[j] + s_prev[j] + s_next[j] + 
                                       s_curr[j - 1] + s_curr[j + 1]);
                }
            }
        }

        /* After an odd number of sweeps the newest data is in the caller's v (buf[1]):
         * copy it back so the caller always reads the result from u (buf[0]). */
        if (steps % 2 == 1) {
            double *src = buf[1];
            double *dst = buf[0];
            #pragma omp for schedule(static)
            for (long k = 0; k < nn; k++) {
                dst[k] = src[k];
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

