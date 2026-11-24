/* k07_nbody: all-pairs gravitational acceleration, one step.
 * UNIFORM cost per body (each body visits all n others), compute-bound. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(int n, const double *px, const double *py, const double *pz,
                   const double *m, double *ax, double *ay, double *az) {
    /* 
     * To maximize performance, we use the following strategies:
     * 1. Use local restrict pointers to eliminate potential aliasing, 
     *    allowing the compiler to optimize loads and stores.
     * 2. Parallelize the outer loop with a static schedule, as the workload 
     *    per body is identical.
     * 3. Hoist the current body's coordinates to registers to minimize repeated 
     *    memory accesses in the inner loop.
     * 4. Use 'omp simd' with a reduction clause to enable vectorization.
     * 5. Use a mathematical identity (1/r^3 = (1/sqrt(r^2))^3) to potentially 
     *    leverage the 'reciprocal square root' (rsqrt) instruction, which is 
     *    often faster than separate sqrt and division on modern CPUs.
     */
    const double * restrict rpx = px;
    const double * restrict rpy = py;
    const double * restrict rpz = pz;
    const double * restrict rm = m;
    double * restrict rax = ax;
    double * restrict ray = ay;
    double * restrict raz = az;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const double pix = rpx[i];
        const double piy = rpy[i];
        const double piz = rpz[i];
        double sx = 0.0, sy = 0.0, sz = 0.0;

        #pragma omp simd reduction(+:sx, sy, sz)
        for (int j = 0; j < n; j++) {
            const double dx = rpx[j] - pix;
            const double dy = rpy[j] - piy;
            const double dz = rpz[j] - piz;
            const double r2 = dx * dx + dy * dy + dz * dz + 1e-9; /* softening */
            
            /* 
             * 1.0 / sqrt(r2) is a common pattern that compilers can often optimize 
             * using the rsqrt hardware instruction.
             */
            const double inv_r = 1.0 / sqrt(r2);
            const double inv = rm[j] * (inv_r * inv_r * inv_r);
            
            sx += dx * inv;
            sy += dy * inv;
            sz += dz * inv;
        }
        rax[i] = sx;
        ray[i] = sy;
        raz[i] = sz;
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
