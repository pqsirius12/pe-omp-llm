/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    /* Use a parallel region once and perform the sweeps inside to reduce overhead.
     * The P-cores and E-cores of the M4 suggest using a dynamic or guided schedule
     * to balance the load across heterogeneous cores. */
    #pragma omp parallel shared(u, v, n, steps)
    {
        /* Local array to manage the alternating source and destination pointers.
         * ptr[0] always refers to the original 'u' and ptr[1] to the original 'v'. */
        double *ptr[2] = {u, v};

        for (int t = 0; t < steps; t++) {
            /* The source and destination swap every sweep. */
            double *src = ptr[t % 2];
            double *dst = ptr[(t + 1) % 2];

            /* Parallelize the spatial loops. 'guided' scheduling is ideal for 
             * mixed performance cores. The implicit barrier at the end of 
             * 'omp for' ensures all threads finish the sweep before the next begins. */
            #pragma omp for schedule(guided)
            for (int i = 1; i < n - 1; i++) {
                long row = (long)i * n;
                long prev = (long)(i - 1) * n;
                long next = (long)(i + 1) * n;
                
                /* The inner loop is contiguous in memory, facilitating vectorization. */
                for (int j = 1; j < n - 1; j++) {
                    dst[row + j] = 0.2 * (src[row + j] +
                                          src[prev + j] + src[next + j] +
                                          src[row + j - 1] + src[row + j + 1]);
                }
            }
        }

        /* After an odd number of sweeps, the result resides in ptr[1] (original v).
         * The original code requires the final result to be in the original u pointer. */
        if (steps % 2 == 1) {
            #pragma omp for schedule(static)
            for (long k = 0; k < (long)n * n; k++) {
                ptr[0][k] = ptr[1][k];
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
