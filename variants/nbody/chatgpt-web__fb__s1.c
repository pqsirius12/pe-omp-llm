/* k07_nbody: all-pairs gravitational acceleration, one step.
 * UNIFORM cost per body (each body visits all n others), compute-bound. */
#include <math.h>
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(int n, const double *px, const double *py, const double *pz,
                   const double *m, double *ax, double *ay, double *az) {
    #pragma omp parallel
    {
        /* Cache the target body's coordinates locally. */
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            const double xi = px[i], yi = py[i], zi = pz[i];
            double sx = 0.0, sy = 0.0, sz = 0.0;

            /* Multiple independent accumulators improve instruction-level
               parallelism and reduce dependency-chain latency. */
            double sx0 = 0.0, sx1 = 0.0;
            double sy0 = 0.0, sy1 = 0.0;
            double sz0 = 0.0, sz1 = 0.0;

            int j = 0;
            for (; j + 1 < n; j += 2) {
                double dx0 = px[j] - xi;
                double dy0 = py[j] - yi;
                double dz0 = pz[j] - zi;
                double r20 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + 1e-9;
                double inv0 = m[j] / (r20 * sqrt(r20));

                double dx1 = px[j + 1] - xi;
                double dy1 = py[j + 1] - yi;
                double dz1 = pz[j + 1] - zi;
                double r21 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + 1e-9;
                double inv1 = m[j + 1] / (r21 * sqrt(r21));

                sx0 += dx0 * inv0; sx1 += dx1 * inv1;
                sy0 += dy0 * inv0; sy1 += dy1 * inv1;
                sz0 += dz0 * inv0; sz1 += dz1 * inv1;
            }

            sx = sx0 + sx1;
            sy = sy0 + sy1;
            sz = sz0 + sz1;

            /* Handle odd n. */
            if (j < n) {
                double dx = px[j] - xi;
                double dy = py[j] - yi;
                double dz = pz[j] - zi;
                double r2 = dx * dx + dy * dy + dz * dz + 1e-9;
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
}

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
