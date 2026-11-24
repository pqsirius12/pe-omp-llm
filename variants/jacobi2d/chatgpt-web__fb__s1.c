/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    const long nn = (long)n * n;

    /* Keep one OpenMP team alive across all sweeps to avoid repeated
       parallel-region creation overhead. */
    #pragma omp parallel
    {
        for (int t = 0; t < steps; t++) {
            #pragma omp for schedule(static)
            for (int i = 1; i < n - 1; i++) {
                const long row = (long)i * n;
                const long prev = row - n;
                const long next = row + n;

                #pragma omp simd
                for (int j = 1; j < n - 1; j++) {
                    v[row + j] = 0.2 * (
                        u[row + j] +
                        u[prev + j] +
                        u[next + j] +
                        u[row + j - 1] +
                        u[row + j + 1]
                    );
                }
            }

            /* Implicit barrier at omp-for guarantees sweep completion. */
            double *tmp = u;
            u = v;
            v = tmp;
        }

        /* Restore result to caller's u for odd sweep counts. */
        if (steps & 1) {
            #pragma omp for schedule(static)
            for (long k = 0; k < nn; k++)
                v[k] = u[k];
        }
    }
/* === KERNEL END === */
}

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
