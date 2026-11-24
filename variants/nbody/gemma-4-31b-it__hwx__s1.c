/* k07_nbody: all-pairs gravitational acceleration, one step.
 * UNIFORM cost per body (each body visits all n others), compute-bound. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(int n, const double *px, const double *py, const double *pz,
                   const double *m, double *ax, double *ay, double *az) {
    /* 
     * The outer loop is embarrassingly parallel. Because we are targeting a 
     * heterogeneous architecture (Apple M4 with P-cores and E-cores), 
     * a static schedule would cause the slower E-cores to become bottlenecks.
     * 'schedule(guided)' is used to dynamically balance the load: it starts 
     * with large chunks to minimize overhead and finishes with small chunks 
     * to ensure that faster cores can take over remaining work from slower ones.
     */
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < n; i++) {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        
        /* Hoist constant values for the current body i to reduce array indexing overhead 
           and assist the compiler's vectorization analysis. */
        const double pix = px[i];
        const double piy = py[i];
        const double piz = pz[i];

        for (int j = 0; j < n; j++) {
            double dx = px[j] - pix;
            double dy = py[j] - piy;
            double dz = pz[j] - piz;
            
            /* Softening factor 1e-9 prevents division by zero. */
            double r2 = dx * dx + dy * dy + dz * dz + 1e-9;
            
            /* 
             * Compute-bound core: r2 * sqrt(r2) is r^3. 
             * clang -O2 on Apple Silicon will typically vectorize this loop 
             * using NEON/AMX instructions. 
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
