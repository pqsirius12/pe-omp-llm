/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    /* Use restrict pointers to inform the compiler that there is no aliasing,
     * which can help the auto-vectorizer and optimizer. */
    double * restrict ur = u;
    double * restrict vr = v;
    long n_long = n;

    /* Open a single parallel region for the entire duration of the kernel 
     * to minimize thread creation/synchronization overhead. */
    #pragma omp parallel
    {
        for (int t = 0; t < steps; t++) {
            /* Instead of swapping pointers in an 'omp single' block, which 
             * requires an additional synchronization barrier, we use the 
             * iteration count to determine the source and destination buffers. */
            double *src = (t % 2 == 0) ? ur : vr;
            double *dst = (t % 2 == 0) ? vr : ur;

            /* Parallelize the outer loop using a static schedule for balanced workload. */
            #pragma omp for schedule(static)
            for (int i = 1; i < n - 1; i++) {
                long row = (long)i * n_long;
                long prev_row = (long)(i - 1) * n_long;
                long next_row = (long)(i + 1) * n_long;

                /* Ensure the inner loop is vectorized. The 5-point stencil
                 * access pattern is highly amenable to SIMD. */
                #pragma omp simd
                for (int j = 1; j < n - 1; j++) {
                    dst[row + j] = 0.2 * (src[row + j] +
                                         src[prev_row + j] + src[next_row + j] +
                                         src[row + j - 1] + src[row + j + 1]);
                }
            }
            /* An implicit barrier here ensures all threads finish writing 
             * to the destination buffer before the next sweep starts. */
        }

        /* If the number of sweeps was odd, the latest result is in vr.
         * We must copy it back to ur so the caller always sees the result in u. */
        if (steps % 2 == 1) {
            #pragma omp for schedule(static)
            for (long k = 0; k < n_long * n_long; k++) {
                ur[k] = vr[k];
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
