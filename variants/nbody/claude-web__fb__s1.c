/* k07_nbody: all-pairs gravitational acceleration, one step.
 * UNIFORM cost per body (each body visits all n others), compute-bound. */
#include <math.h>
#include <string.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
/* 4 bodies per vector; each lane accumulates its own body in the original
 * serial j order, and packed sqrt/div/mul/add are correctly rounded per lane,
 * so results are bitwise identical to the serial version. */
typedef double vd4 __attribute__((ext_vector_type(4)));

static inline vd4 vsqrt4(vd4 x) {
#ifdef __has_builtin
#if __has_builtin(__builtin_elementwise_sqrt)
#define NB_HAVE_VSQRT 1
#endif
#endif
#ifdef NB_HAVE_VSQRT
    return __builtin_elementwise_sqrt(x);
#else
    vd4 r;
    r.x = __builtin_sqrt(x.x); r.y = __builtin_sqrt(x.y);
    r.z = __builtin_sqrt(x.z); r.w = __builtin_sqrt(x.w);
    return r;
#endif
}

static void kernel(int n, const double *px, const double *py, const double *pz,
                   const double *m, double *ax, double *ay, double *az) {
    int nblk = n / 4;

    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (int b = 0; b < nblk; b++) {
            int i0 = b * 4;
            vd4 xi, yi, zi;
            memcpy(&xi, px + i0, sizeof xi);
            memcpy(&yi, py + i0, sizeof yi);
            memcpy(&zi, pz + i0, sizeof zi);
            vd4 sx = 0.0, sy = 0.0, sz = 0.0;
            for (int j = 0; j < n; j++) {
                vd4 dx = px[j] - xi, dy = py[j] - yi, dz = pz[j] - zi;
                vd4 r2 = dx * dx + dy * dy + dz * dz + 1e-9;   /* softening */
                vd4 inv = m[j] / (r2 * vsqrt4(r2));
                sx += dx * inv; sy += dy * inv; sz += dz * inv;
            }
            memcpy(ax + i0, &sx, sizeof sx);
            memcpy(ay + i0, &sy, sizeof sy);
            memcpy(az + i0, &sz, sizeof sz);
        }

        /* Remainder bodies (fewer than 4). */
        #pragma omp for schedule(static)
        for (int i = nblk * 4; i < n; i++) {
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
