/* k07_nbody: all-pairs gravitational acceleration, one step.
 * UNIFORM cost per body (each body visits all n others), compute-bound. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(int n, const double *px, const double *py, const double *pz,
                   const double *m, double *ax, double *ay, double *az) {
    /* 
     * The workload is uniform per outer loop iteration i, so static scheduling
     * is optimal for load balancing.
     */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        /* Hoist constants for the inner loop to registers to minimize array access. */
        const double pxi = px[i];
        const double pyi = py[i];
        const double pzi = pz[i];
        double sx = 0.0, sy = 0.0, sz = 0.0;

        /* 
         * The inner loop is the compute bottleneck. We use #pragma omp simd to 
         * explicitly instruct the compiler to vectorize the calculations. 
         * This is critical for using AVX/AVX-512 units on modern CPUs, as 
         * clang -O2 may not automatically vectorize sqrt and division.
         * 
         * The reduction clause allows the compiler to maintain partial sums 
         * in vector registers, maximizing throughput.
         */
        #pragma omp simd reduction(+:sx, sy, sz)
        for (int j = 0; j < n; j++) {
            double dx = px[j] - pxi;
            double dy = py[j] - pyi;
            double dz = pz[j] - pzi;
            double r2 = dx * dx + dy * dy + dz * dz + 1e-9;   /* softening */
            
            /* 
             * The expression m[j] / (r2 * sqrt(r2)) calculates the interaction 
             * strength. Using a single division and a single sqrt per iteration 
             * allows for better pipelining in the CPU.
             */
            double inv = m[j] / (r2 * sqrt(r2));
            sx += dx * inv;
            sy += dy * inv;
            sz += dz * inv;
        }
        ax[i] = sx;
        ay[i] = sy;
        az[i] = sz;
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    int n = (int)arg_long(argc, argv, 1, 12000);
    double *b = xmalloc(7L * n * sizeof *b);
    double *px = b, *py = b + n, *pz = b + 2L * n, *m = b + 3L * n;
    double *ax = b + 4L * n, *ay = b + 5L * n, *az = b + 6L * n;
    rng_seed(7);
    for (int i = 0; i < n; i++) { px[i] = rng_unit(); py[i] = rng_unit(); pz[i] = rng_unit(); m[i] = rng_unit(); }

    double t0 = now_sec();
    kernel(n, px, py, pz, m, ax, ay, az);
    double t1 = now_sec();

    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += fabs(ax[i]) + fabs(ay[i]) + fabs(az[i]);
    report("nbody", t1 - t0, sum);
    free(b);
    return 0;
}
