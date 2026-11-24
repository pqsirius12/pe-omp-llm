/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    double *u_orig = u;
    double *v_orig = v;

    /* 
     * Move the parallel region outside the time-step loop to avoid the overhead
     * of repeatedly creating and joining threads for every sweep.
     */
    #pragma omp parallel
    {
        for (int t = 0; t < steps; t++) {
            /* Determine source and destination buffers based on current step parity.
               t=0: src=u_orig, dst=v_orig;
               t=1: src=v_orig, dst=u_orig;
               and so on. */
            double *src = (t % 2 == 0) ? u_orig : v_orig;
            double *dst = (t % 2 == 0) ? v_orig : u_orig;

            /* 
             * Parallelize the spatial loop. schedule(static) is used to ensure
             * consistent memory access patterns across sweeps, aiding cache reuse.
             */
            #pragma omp for schedule(static)
            for (int i = 1; i < n - 1; i++) {
                long row = (long)i * n;
                long prev = row - n;
                long next = row + n;
                
                /* The inner loop is a prime candidate for SIMD vectorization. */
                #pragma omp simd
                for (int j = 1; j < n - 1; j++) {
                    dst[row + j] = 0.2 * (src[row + j] +
                                         src[prev + j] + src[next + j] +
                                         src[row + j - 1] + src[row + j + 1]);
                }
            }
            /* 
             * An implicit barrier exists at the end of the '#pragma omp for' loop,
             * ensuring all threads finish sweep t before anyone starts sweep t+1.
             */
        }
    }

    /* After an odd number of sweeps, the most recent result is in v_orig.
     * Copy it back to u_orig so the caller always reads the result from u. */
    if (steps % 2 == 1) {
        #pragma omp parallel for schedule(static)
        for (long k = 0; k < (long)n * n; k++) {
            u_orig[k] = v_orig[k];
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

