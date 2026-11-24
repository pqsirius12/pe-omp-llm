/* k14_kmeans: Lloyd's k-means, fixed number of iterations.
 * Assignment step: UNIFORM and easy to parallelize.
 * Update step: every point adds into a shared cluster sum - a reduction
 * TRAP, like the histogram. Sum order changes floating-point results, so
 * the harness compares with a relative tolerance. */
#include "common.h"
#include <string.h>
#include <stdlib.h>

#define D 8
#define K 16

/* === KERNEL BEGIN === */
static void kernel(long n, int iters, const double *pts, double *cent, int *label) {
    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    double *lsum = (double *)xmalloc((size_t)nthreads * K * D * sizeof(double));
    long   *lcnt = (long   *)xmalloc((size_t)nthreads * K * sizeof(long));

    for (int it = 0; it < iters; it++) {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++) {               /* assignment */
            const double *p = pts + i * D;
            int best = 0; double bd = 1e300;
            for (int c = 0; c < K; c++) {
                const double *cc = cent + c * D;
                double d0 = p[0]-cc[0], d1 = p[1]-cc[1], d2 = p[2]-cc[2], d3 = p[3]-cc[3];
                double d4 = p[4]-cc[4], d5 = p[5]-cc[5], d6 = p[6]-cc[6], d7 = p[7]-cc[7];
                double d = d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7;
                if (d < bd) { bd = d; best = c; }
            }
            label[i] = best;
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double *s = lsum + (size_t)tid * K * D;
            long   *c = lcnt + (size_t)tid * K;
            for (int j = 0; j < K * D; j++) s[j] = 0.0;
            for (int j = 0; j < K; j++) c[j] = 0;

            #pragma omp for schedule(static) nowait
            for (long i = 0; i < n; i++) {
                int cc = label[i];
                c[cc]++;
                const double *p = pts + i * D;
                double *dst = s + cc * D;
                dst[0] += p[0]; dst[1] += p[1]; dst[2] += p[2]; dst[3] += p[3];
                dst[4] += p[4]; dst[5] += p[5]; dst[6] += p[6]; dst[7] += p[7];
            }
        }

        /* reduce per-thread partials and update cent */
        for (int cc = 0; cc < K; cc++) {
            long total = 0;
            double acc[D];
            for (int k = 0; k < D; k++) acc[k] = 0.0;
            for (int t = 0; t < nthreads; t++) {
                total += lcnt[(size_t)t * K + cc];
                const double *s = lsum + (size_t)t * K * D + cc * D;
                for (int k = 0; k < D; k++) acc[k] += s[k];
            }
            if (total > 0) {
                double inv = 1.0 / (double)total;
                for (int k = 0; k < D; k++) cent[cc * D + k] = acc[k] * inv;
            }
        }
    }

    free(lsum);
    free(lcnt);
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 2000000);
    int iters = (int)arg_long(argc, argv, 2, 10);
    double *pts = xmalloc(n * D * sizeof *pts), cent[K * D];
    int *label = xmalloc(n * sizeof *label);
    rng_seed(14);
    for (long k = 0; k < n * D; k++) pts[k] = rng_unit();
    for (int k = 0; k < K * D; k++) cent[k] = pts[k];   /* first K points */

    double t0 = now_sec();
    kernel(n, iters, pts, cent, label);
    double t1 = now_sec();

    double sum = 0.0;
    for (int k = 0; k < K * D; k++) sum += cent[k] * (k + 1);
    report("kmeans", t1 - t0, sum);
    free(pts); free(label);
    return 0;
}
