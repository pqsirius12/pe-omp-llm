/* k07_nbody: all-pairs gravitational acceleration, one step.
 * UNIFORM cost per body (each body visits all n others), compute-bound. */
#include <math.h>
#include <omp.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#include "common.h"

/* === KERNEL BEGIN === */
/* Scalar computation for one body: identical to the serial code. */
static inline void body_scalar(int i, int n, const double *px, const double *py,
                               const double *pz, const double *m,
                               double *ax, double *ay, double *az) {
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int j = 0; j < n; j++) {
        double dx = px[j] - px[i], dy = py[j] - py[i], dz = pz[j] - pz[i];
        double r2 = dx * dx + dy * dy + dz * dz + 1e-9;   /* softening */
        double inv = m[j] / (r2 * sqrt(r2));
        sx += dx * inv; sy += dy * inv; sz += dz * inv;
    }
    ax[i] = sx; ay[i] = sy; az[i] = sz;
}

static void kernel(int n, const double *px, const double *py, const double *pz,
                   const double *m, double *ax, double *ay, double *az) {
    const int nb = n / 4;          /* blocks of 4 bodies */

    #pragma omp parallel
    {
        /* Dynamic scheduling: P-cores grab more blocks than E-cores. */
        #pragma omp for schedule(dynamic, 4) nowait
        for (int b = 0; b < nb; b++) {
            const int i = 4 * b;
#ifdef __aarch64__
            /* SIMD across i (4 bodies, 2 NEON vectors); the j order per body
             * is unchanged, so every sum matches the serial version. */
            const float64x2_t xi0 = vld1q_f64(px + i), xi1 = vld1q_f64(px + i + 2);
            const float64x2_t yi0 = vld1q_f64(py + i), yi1 = vld1q_f64(py + i + 2);
            const float64x2_t zi0 = vld1q_f64(pz + i), zi1 = vld1q_f64(pz + i + 2);
            const float64x2_t eps = vdupq_n_f64(1e-9);
            float64x2_t sx0 = vdupq_n_f64(0.0), sx1 = vdupq_n_f64(0.0);
            float64x2_t sy0 = vdupq_n_f64(0.0), sy1 = vdupq_n_f64(0.0);
            float64x2_t sz0 = vdupq_n_f64(0.0), sz1 = vdupq_n_f64(0.0);
            for (int j = 0; j < n; j++) {
                const float64x2_t xj = vdupq_n_f64(px[j]);
                const float64x2_t yj = vdupq_n_f64(py[j]);
                const float64x2_t zj = vdupq_n_f64(pz[j]);
                const float64x2_t mj = vdupq_n_f64(m[j]);

                float64x2_t dx0 = xj - xi0, dy0 = yj - yi0, dz0 = zj - zi0;
                float64x2_t dx1 = xj - xi1, dy1 = yj - yi1, dz1 = zj - zi1;
                float64x2_t r20 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0 + eps;
                float64x2_t r21 = dx1 * dx1 + dy1 * dy1 + dz1 * dz1 + eps;
                float64x2_t inv0 = mj / (r20 * vsqrtq_f64(r20));
                float64x2_t inv1 = mj / (r21 * vsqrtq_f64(r21));
                sx0 += dx0 * inv0; sy0 += dy0 * inv0; sz0 += dz0 * inv0;
                sx1 += dx1 * inv1; sy1 += dy1 * inv1; sz1 += dz1 * inv1;
            }
            vst1q_f64(ax + i, sx0); vst1q_f64(ax + i + 2, sx1);
            vst1q_f64(ay + i, sy0); vst1q_f64(ay + i + 2, sy1);
            vst1q_f64(az + i, sz0); vst1q_f64(az + i + 2, sz1);
#else
            for (int k = 0; k < 4; k++)
                body_scalar(i + k, n, px, py, pz, m, ax, ay, az);
#endif
        }

        /* Leftover bodies when n is not a multiple of 4. */
        #pragma omp for schedule(dynamic, 1)
        for (int i = 4 * nb; i < n; i++)
            body_scalar(i, n, px, py, pz, m, ax, ay, az);
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
