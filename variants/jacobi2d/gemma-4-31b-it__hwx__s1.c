/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(int n, int steps, double *u, double *v) {
    double *u_ptr = u;
    double *v_ptr = v;

    /* Use a single parallel region to minimize the overhead of creating 
     * and destroying threads for every sweep. */
    #pragma omp parallel
    {
        for (int t = 0; t < steps; t++) {
            /* Since the target machine has both performance (P) and efficiency (E) cores,
             * schedule(guided) is used to dynamically balance the load, allowing the 
             * faster P-cores to process more rows than the slower E-cores. */
            #pragma omp for schedule(guided)
            for (int i = 1; i < n - 1; i++) {
                long row = (long)i * n;
                long row_prev = row - n;
                long row_next = row + n;
                for (int j = 1; j < n - 1; j++) {
                    v_ptr[row + j] = 0.2 * (u_ptr[row + j] +
                                           u_ptr[row_prev + j] + u_ptr[row_next + j] +
                                           u_ptr[row + j - 1] + u_ptr[row + j + 1]);
                }
            }
            /* The pointers u and v are shared. Swapping them inside a 'single' 
             * block ensures that only one thread updates the pointers and 
             * that all threads synchronize at the implicit barrier before 
             * the next sweep begins. */
            #pragma omp single
            {
                double *tmp = u_ptr; u_ptr = v_ptr; v_ptr = tmp;
            }
        }

        /* After an odd number of sweeps, the result is stored in the buffer 
         * that was originally passed as 'v'. We copy it back to 'u' to ensure 
         * consistency with the serial version's expectations. */
        if (steps % 2 == 1) {
            #pragma omp for schedule(guided)
            for (long k = 0; k < (long)n * n; k++) {
                v_ptr[k] = u_ptr[k];
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
