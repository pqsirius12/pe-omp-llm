/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    /* We open a single parallel region to avoid the overhead of spawning and 
     * joining threads in every single sweep of the Jacobi iteration. */
    #pragma omp parallel
    {
        for (int t = 0; t < steps; t++) {
            /* The i-loop is parallelized. Each thread processes a set of rows. 
             * A static schedule is used because the workload per row is identical. */
            #pragma omp for schedule(static)
            for (int i = 1; i < n - 1; i++) {
                long row = (long)i * n;
                long prev_row = (long)(i - 1) * n;
                long next_row = (long)(i + 1) * n;
                
                /* The inner j-loop is kept sequential to allow the compiler's 
                 * auto-vectorizer (SIMD) to optimize the contiguous memory access. */
                for (int j = 1; j < n - 1; j++) {
                    v[row + j] = 0.2 * (u[row + j] +
                                       u[prev_row + j] + u[next_row + j] +
                                       u[row + j - 1] + u[row + j + 1]);
                }
            }
            
            /* The result of sweep t becomes the source for sweep t+1.
             * The pointer swap must be synchronized across all threads. 
             * 'omp single' with its implicit barrier ensures all threads 
             * finish writing to v and see the updated pointers before the next t. */
            #pragma omp single
            {
                double *tmp = u;
                u = v;
                v = tmp;
            }
        }

        /* If steps was odd, the final result is currently in the local pointer u,
         * but the caller expects the result to be in the original u. Since we
         * swapped them, we must copy the result back to the original v. */
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
