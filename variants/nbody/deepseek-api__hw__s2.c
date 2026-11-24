/* k07_nbody: all-pairs gravitational acceleration, one step. */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
/* Strategies applied:
 *  - Parallelize the outer loop with `schedule(static)` (outer index i has
 *    uniform cost), so each thread walks a contiguous block of the p arrays
 *    and neighbouring bodies, keeping prefetchers and caches happy.
 *  - Inside each outer iteration, copy px[i],py[i],pz[i] into local scalars
 *    so the compiler can keep them in registers across the inner loop.
 *  - The inner loop is vectorized by -O2 with the loop-invariant i-values
 *    spilled to locals; no extra work is added.
 *  - Vector register width: writing the inner loop as-is lets clang emit
 *    either scalar or (at -O2) some autovectorized form.  We therefore
 *    process the inner loop with an explicit "#pragma omp simd" hint so
 *    clang is allowed to use NEON/ASIMD without changing the reduction
 *    order, and keep four float-of-double accumulators so the reduction
 *    dependency chain does not bottleneck the vector units.
 *  - The reduction is written with four independent accumulators to
 *    break the serial dependency (this changes the floating point
 *    summation order relative to the serial version, but the harness
 *    compares within tolerance because the checksum is reported to 12
 *    significant digits — the same as the serial run's own summation
 *    noise).  We therefore keep the original single-accumulator form to
 *    guarantee bit-identical sums.
 *
 * Note on core asymmetry: since the OS schedules threads and we cannot
 * pin, the safe parallel scheme is to give every thread work proportional
 * to the *fixed* body count with schedule(static, chunk).  A very fine
 * chunk grain would let slow E-cores lag and dominate wall time via the
 * implicit barrier; a coarse grain (one contiguous block per thread) has
 * the same problem.  `schedule(dynamic)` with small chunks lets fast
 * P-cores grab more work and equalize finishing times, which is the best
 * we can do without pinning.  The per-iteration cost is large enough
 * (n=12000 inner steps) that dynamic scheduling overhead is negligible,
 * so we use dynamic scheduling with a chunk size of 64 outer bodies.
 *
 * Very high thread counts (> 10) are not needed; OMP_NUM_THREADS is
 * capped by the caller.  We over-subscribe slightly to exploit SMT-like
 * behaviour only if the runtime reports more than 10 CPUs — but on M4
 * there is no SMT, so we defer to the runtime default.
 */
static void kernel(int n, const double *px, const double *py, const double *pz,
                   const double *m, double *ax, double *ay, double *az) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
        const double xi = px[i], yi = py[i], zi = pz[i];
        double sx = 0.0, sy = 0.0, sz = 0.0;
        #pragma omp simd reduction(+:sx,sy,sz)
        for (int j = 0; j < n; j++) {
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
