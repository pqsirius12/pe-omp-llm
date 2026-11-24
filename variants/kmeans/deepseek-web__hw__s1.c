/* k14_kmeans: Lloyd's k-means, fixed number of iterations.
 * Assignment step: UNIFORM and easy to parallelize.
 * Update step: every point adds into a shared cluster sum - a reduction
 * TRAP, like the histogram. Sum order changes floating-point results, so
 * the harness compares with a relative tolerance. */
#include "common.h"
#include <string.h>
#include <omp.h>

#define D 8
#define K 16

/* === KERNEL BEGIN === */
static void kernel(long n, int iters, const double *pts, double *cent, int *label) {
    /* Per-thread private accumulators for the update step to avoid atomics. */
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
    double *tsum = (double *)xmalloc((size_t)nthreads * K * D * sizeof(double));
    long   *tcnt = (long   *)xmalloc((size_t)nthreads * K * sizeof(long));

    double sum[K * D];
    long cnt[K];

    for (int it = 0; it < iters; it++) {
        /* ---- Assignment step: fully parallel over points. ---- */
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++) {
            int best = 0; double bd = 1e300;
            const double *p = pts + i * D;
            for (int c = 0; c < K; c++) {
                const double *cc = cent + c * D;
                double d = 0.0;
                for (int k = 0; k < D; k++) {
                    double t = p[k] - cc[k];
                    d += t * t;
                }
                if (d < bd) { bd = d; best = c; }
            }
            label[i] = best;
        }

        /* ---- Update step: parallel reduction via per-thread buffers. ---- */
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double *ls = tsum + (size_t)tid * K * D;
            long   *lc = tcnt + (size_t)tid * K;
            memset(ls, 0, (size_t)K * D * sizeof(double));
            memset(lc, 0, (size_t)K * sizeof(long));

            #pragma omp for schedule(static) nowait
            for (long i = 0; i < n; i++) {
                int c = label[i];
                lc[c]++;
                const double *p = pts + i * D;
                double *s = ls + c * D;
                for (int k = 0; k < D; k++) s[k] += p[k];
            }

            #pragma omp for schedule(static)
            for (int c = 0; c < K * D; c++) sum[c] = 0.0;
            #pragma omp for schedule(static)
            for (int c = 0; c < K; c++) cnt[c] = 0;
            #pragma omp barrier

            /* Deterministic reduction over threads, sequential per c/k. */
            #pragma omp single
            {
                for (int t = 0; t < nthreads; t++) {
                    const double *ls = tsum + (size_t)t * K * D;
                    const long   *lc = tcnt + (size_t)t * K;
                    for (int c = 0; c < K; c++) {
                        cnt[c] += lc[c];
                        double *s = sum + c * D;
                        const double *ls2 = ls + c * D;
                        for (int k = 0; k < D; k++) s[k] += ls2[k];
                    }
                }
            }
        }

        for (int c = 0; c < K; c++)
            if (cnt[c] > 0)
                for (int k = 0; k < D; k++) cent[c * D + k] = sum[c * D + k] / cnt[c];
    }

    free(tsum);
    free(tcnt);
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
