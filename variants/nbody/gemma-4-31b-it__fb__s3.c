/* k07_nbody: all-pairs gravitational acceleration, one step.
 * UNIFORM cost per body (each body visits all n others), compute-bound. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(int n, const double *px, const double *py, const double *pz,
                   const double *m, double *ax, double *ay, double *az) {
    /* Use restrict pointers to inform the compiler that arrays do not overlap,
       which allows for more aggressive SIMD vectorization. */
    const double * __restrict__ px_r = px;
    const double * __restrict__ py_r = py;
    const double * __restrict__ pz_r = pz;
    const double * __restrict__ m_r = m;
    double * __restrict__ ax_r = ax;
    double * __restrict__ ay_r = ay;
    double * __restrict__ az_r = az;

    /* Parallelize the outer loop. Since the workload per iteration is constant,
       static scheduling is the most efficient. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        const double pix = px_r[i];
        const double piy = py_r[i];
        const double piz = pz_r[i];
        
        /* Vectorize the inner loop. Using a reduction on the accumulators allows 
           the compiler to use SIMD registers for the partial sums. */
        #pragma omp simd reduction(+:sx, sy, sz)
        for (int j = 0; j < n; j++) {
            double dx = px_r[j] - pix;
            double dy = py_r[j] - piy;
            double dz = pz_r[j] - piz;
            double r2 = dx * dx + dy * dy + dz * dz + 1e-9;   /* softening */
            
            /* This expression is compute-heavy. Clang -O2 will typically optimize 
               sqrt() and division using SIMD instructions. */
            double inv = m_r[j] / (r2 * sqrt(r2));
            
            sx += dx * inv; 
            sy += dy * inv; 
            sz += dz * inv;
        }
        ax_r[i] = sx; 
        ay_r[i] = sy; 
        az_r[i] = sz;
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
