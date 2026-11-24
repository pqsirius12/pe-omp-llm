/* k07_nbody: all-pairs gravitational acceleration, one step.
 * UNIFORM cost per body (each body visits all n others), compute-bound. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
#define NB_TILE 4

static void kernel(int n, const double *px, const double *py, const double *pz,
                   const double *m, double *ax, double *ay, double *az) {
    int nblk = n / NB_TILE;

    /* Main part: blocks of NB_TILE bodies. Each body's j-loop runs in the
     * original serial order, so results are bitwise identical. */
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (int b = 0; b < nblk; b++) {
            int i0 = b * NB_TILE;
            double xi[NB_TILE], yi[NB_TILE], zi[NB_TILE];
            double sx[NB_TILE], sy[NB_TILE], sz[NB_TILE];
            for (int k = 0; k < NB_TILE; k++) {
                xi[k] = px[i0 + k]; yi[k] = py[i0 + k]; zi[k] = pz[i0 + k];
                sx[k] = 0.0; sy[k] = 0.0; sz[k] = 0.0;
            }
            for (int j = 0; j < n; j++) {
                double xj = px[j], yj = py[j], zj = pz[j], mj = m[j];
                for (int k = 0; k < NB_TILE; k++) {
                    double dx = xj - xi[k], dy = yj - yi[k], dz = zj - zi[k];
                    double r2 = dx * dx + dy * dy + dz * dz + 1e-9;   /* softening */
                    double inv = mj / (r2 * sqrt(r2));
                    sx[k] += dx * inv; sy[k] += dy * inv; sz[k] += dz * inv;
                }
            }
            for (int k = 0; k < NB_TILE; k++) {
                ax[i0 + k] = sx[k]; ay[i0 + k] = sy[k]; az[i0 + k] = sz[k];
            }
        }

        /* Remainder bodies (fewer than NB_TILE), one per thread. */
        #pragma omp for schedule(static)
        for (int i = nblk * NB_TILE; i < n; i++) {
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (int j = 0; j < n; j++) {
                double dx = px[j] - px[i], dy = py[j] - py[i], dz = pz[j] - pz[i];
                double r2 = dx * dx + dy * dy + dz * dz + 1e-9;   /* softening */
                double inv = m[j] / (r2 * sqrt(r2));
                sx += dx * inv; sy += dy * inv; sz += dz * inv;
            }
            ax[i] = sx; ay[i] = sy; az[i] = sz;
        }
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
