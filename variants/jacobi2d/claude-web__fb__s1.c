/* k03_jacobi2d: 5-point stencil, T sweeps. UNIFORM cost, memory-bound,
 * with a dependency BETWEEN sweeps (only the inner sweep is parallel). */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
/* One Jacobi sweep over interior rows [r0, r1): dst = stencil(src). */
static inline void sweep_rows(int n, const double *restrict src,
                              double *restrict dst, int r0, int r1) {
    for (int i = r0; i < r1; i++) {
        const double *restrict uc = src + (long)i * n;
        const double *restrict uu = src + (long)(i - 1) * n;
        const double *restrict ud = src + (long)(i + 1) * n;
        double *restrict vc = dst + (long)i * n;
        #pragma omp simd
        for (int j = 1; j < n - 1; j++)
            vc[j] = 0.2 * (uc[j] + uu[j] + ud[j] + uc[j - 1] + uc[j + 1]);
    }
}

static void kernel(int n, int steps, double *u, double *v) {
    /* Array holding the result of global step s is (s & 1) ? v : u.
     * Boundary rows/columns are never written and are identical in u and v. */
    #pragma omp parallel
    {
        int T = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int rows = n - 2;

        if (rows > 0 && steps > 0) {
            if (T == 1) {
                for (int s = 1; s <= steps; s++) {
                    const double *src = ((s - 1) & 1) ? v : u;
                    double *dst = (s & 1) ? v : u;
                    sweep_rows(n, src, dst, 1, n - 1);
                }
            } else {
                /* Block height: keep ~ (2T+1) blocks x 2 arrays within ~6 MB. */
                long rowbytes = (long)n * (long)sizeof(double);
                long budget = 6L * 1024 * 1024;
                long Bl = budget / ((2L * T + 1) * 2L * rowbytes);
                if (Bl < 4) Bl = 4;
                if (Bl > 64) Bl = 64;
                if (Bl > rows) Bl = rows;
                int B = (int)Bl;
                int nb = (rows + B - 1) / B;

                for (int base = 0; base < steps; base += T) {
                    int P = steps - base < T ? steps - base : T;
                    int s = base + tid + 1;              /* step done by this thread */
                    const double *src = ((s - 1) & 1) ? v : u;
                    double *dst = (s & 1) ? v : u;
                    int stages = nb + 2 * (P - 1);
                    for (int g = 0; g < stages; g++) {
                        if (tid < P) {
                            int b = g - 2 * tid;         /* lag of 2 blocks per step */
                            if (b >= 0 && b < nb) {
                                int r0 = 1 + b * B;
                                int r1 = r0 + B;
                                if (r1 > n - 1) r1 = n - 1;
                                sweep_rows(n, src, dst, r0, r1);
                            }
                        }
                        #pragma omp barrier
                    }
                }
            }
        }

        /* After an odd number of sweeps the newest data is in the caller's v:
         * copy it back so the caller always reads the result from u. */
        if (steps > 0 && (steps % 2 == 1)) {
            long nn = (long)n * n;
            #pragma omp for schedule(static)
            for (long k = 0; k < nn; k++) u[k] = v[k];
        }
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    int n = (int)arg_long(argc, argv, 1, 2048);
    int steps = (int)arg_long(argc, argv, 2, 100);
    long nn = (long)n * n;
    double *u = xmalloc(nn * sizeof *u), *v = xmalloc(nn * sizeof *v);
    rng_seed(3);
    for (long k = 0; k < nn; k++) { u[k] = rng_unit(); v[k] = u[k]; }

    double t0 = now_sec();
    kernel(n, steps, u, v);
    double t1 = now_sec();

    double sum = 0.0;
    for (long k = 0; k < nn; k++) sum += u[k];
    report("jacobi2d", t1 - t0, sum);
    free(u); free(v);
    return 0;
}
