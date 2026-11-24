/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    for (int t = 0; t < steps; t++) {
        /* Parallelize the spatial loops for each time step. 
           Since the work per sweep is substantial (n*n iterations), 
           the overhead of creating/joining the parallel region is negligible. */
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < n - 1; i++) {
            long row = (long)i * n;
            long prev = row - n;
            long next = row + n;
            for (int j = 1; j < n - 1; j++) {
                v[row + j] = 0.2 * (u[row + j] +
                                   u[prev + j] + u[next + j] +
                                   u[row + j - 1] + u[row + j + 1]);
            }
        }
        /* Swap pointers so the result of sweep t feeds sweep t+1. */
        double *tmp = u; u = v; v = tmp;
    }
    /* After an odd number of sweeps the newest data is in the caller's v:
     * copy it back so the caller always reads the result from u. */
    if (steps % 2 == 1) {
        #pragma omp parallel for schedule(static)
        for (long k = 0; k < (long)n * n; k++) {
            v[k] = u[k];
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

