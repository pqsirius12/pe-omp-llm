/* k07_nbody: all-pairs gravitational acceleration, one step.
 * UNIFORM cost per body (each body visits all n others), compute-bound. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(int n, const double *px, const double *py, const double *pz,
                   const double *m, double *ax, double *ay, double *az) {
    /* 
     * The problem is O(n^2) and compute-bound. 
     * The outer loop iterates over bodies receiving acceleration (i), and the 
     * inner loop over bodies exerting it (j). Each 'i' iteration is independent.
     * Target machine has heterogeneous cores (P-cores and E-cores), so 
     * schedule(guided) is used to balance the load.
     */
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; i++) {
        const double pix = px[i];
        const double piy = py[i];
        const double piz = pz[i];
        double sx = 0.0, sy = 0.0, sz = 0.0;

        /* 
         * The inner loop is a prime candidate for SIMD vectorization.
         * We use omp simd with reductions for sx, sy, and sz.
         * Access patterns for px, py, pz, m are linear, favoring cache locality.
         */
        #pragma omp simd reduction(+:sx, sy, sz)
        for (int j = 0; j < n; j++) {
            double dx = px[j] - pix;
            double dy = py[j] - piy;
            double dz = pz[j] - piz;
            double r2 = dx * dx + dy * dy + dz * dz + 1e-9; /* softening */
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

