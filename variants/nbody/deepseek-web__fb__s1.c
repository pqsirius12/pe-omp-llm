/* k07_nbody: all-pairs gravitational acceleration, one step.
 * UNIFORM cost per body (each body visits all n others), compute-bound. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(int n, const double *px, const double *py, const double *pz,
                   const double *m, double *ax, double *ay, double *az) {
    /* Block of i values processed per parallel task. Each j-iteration loads
     * px[j],py[j],pz[j],m[j] once and reuses them BI times, cutting memory
     * traffic by a factor of BI. BI=8 keeps 3*BI accumulators + i-tile data
     * comfortably in registers/L1. */
    const int BI = 8;

    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < n; ib += BI) {
        int iend = ib + BI; if (iend > n) iend = n;
        int nb = iend - ib;

        double pxi[8], pyi[8], pzi[8];
        double sx[8], sy[8], sz[8];
        for (int t = 0; t < nb; t++) {
            pxi[t] = px[ib+t]; pyi[t] = py[ib+t]; pzi[t] = pz[ib+t];
            sx[t] = 0.0; sy[t] = 0.0; sz[t] = 0.0;
        }

        for (int j = 0; j < n; j++) {
            double pxj = px[j], pyj = py[j], pzj = pz[j], mj = m[j];
            for (int t = 0; t < nb; t++) {
                double dx = pxj - pxi[t];
                double dy = pyj - pyi[t];
                double dz = pzj - pzi[t];
                double r2 = dx*dx + dy*dy + dz*dz + 1e-9;
                double inv = mj / (r2 * sqrt(r2));
                sx[t] += dx * inv;
                sy[t] += dy * inv;
                sz[t] += dz * inv;
            }
        }

        for (int t = 0; t < nb; t++) {
            ax[ib+t] = sx[t]; ay[ib+t] = sy[t]; az[ib+t] = sz[t];
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
