/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    /* 
     * We use a single parallel region to avoid the overhead of spawning/joining 
     * threads in every sweep. The pointers u and v are shared among threads.
     */
    #pragma omp parallel
    {
        for (int t = 0; t < steps; t++) {
            /* 
             * Use guided scheduling to handle the heterogeneous nature of the Apple M4 
             * (4 P-cores and 6 E-cores). Guided scheduling dynamically balances the load 
             * while keeping overhead lower than dynamic scheduling.
             * 
             * We parallelize the i-loop to allow the compiler to better vectorize 
             * the inner j-loop via SIMD.
             */
            #pragma omp for schedule(guided)
            for (int i = 1; i < n - 1; i++) {
                long row = (long)i * n;
                long prev_row = (long)(i - 1) * n;
                long next_row = (long)(i + 1) * n;
                for (int j = 1; j < n - 1; j++) {
                    v[row + j] = 0.2 * (u[row + j] +
                                        u[prev_row + j] + u[next_row + j] +
                                        u[row + j - 1] + u[row + j + 1]);
                }
            }
            /* 
             * The pointer swap must be performed by a single thread. 
             * The implicit barrier at the end of the 'single' block ensures that 
             * all threads see the updated pointers before the next sweep begins.
             */
            #pragma omp single
            {
                double *tmp = u; u = v; v = tmp;
            }
        }

        /* 
         * If the number of steps is odd, the final result is in the current 'u' 
         * (which was the original 'v'). We copy it back to the original 'u' 
         * (now pointed to by 'v').
         */
        if (steps % 2 == 1) {
            #pragma omp for schedule(static)
            for (long k = 0; k < (long)n * n; k++) {
                v[k] = u[k];
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

