/* k14_kmeans: Lloyd's k-means, fixed number of iterations.
 * Assignment step: UNIFORM and easy to parallelize.
 * Update step: every point adds into a shared cluster sum - a reduction
 * TRAP, like the histogram. Sum order changes floating-point results, so
 * the harness compares with a relative tolerance. */
#include "common.h"
#include <string.h>

#define D 8
#define K 16

/* === KERNEL BEGIN === */
static void kernel(long n, int iters, const double *pts, double *cent, int *label) {
    /* Per-thread accumulators to avoid contention in the update step. */
    int nthreads_max = 64;
    double *tsum = xmalloc((size_t)nthreads_max * K * D * sizeof(double));
    long *tcnt = xmalloc((size_t)nthreads_max * K * sizeof(long));

    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        double *mysum = tsum + (size_t)tid * K * D;
        long *mycnt = tcnt + (size_t)tid * K;

        for (int it = 0; it < iters; it++) {
            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {               /* assignment */
                int best = 0; double bd = 1e300;
                for (int c = 0; c < K; c++) {
                    double d = 0.0;
                    for (int k = 0; k < D; k++) {
                        double t = pts[i * D + k] - cent[c * D + k];
                        d += t * t;
                    }
                    if (d < bd) { bd = d; best = c; }
                }
                label[i] = best;
            }

            for (int c = 0; c < K * D; c++) mysum[c] = 0.0;
            for (int c = 0; c < K; c++) mycnt[c] = 0;

            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                int l = label[i];
                mycnt[l]++;
                const double *p = pts + i * D;
                double *s = mysum + l * D;
                for (int k = 0; k < D; k++) s[k] += p[k];
            }

            #pragma omp barrier
            #pragma omp single
            {
                double sum[K * D];
                long cnt[K];
                int nt = omp_get_num_threads();
                for (int c = 0; c < K * D; c++) sum[c] = 0.0;
                for (int c = 0; c < K; c++) cnt[c] = 0;
                for (int t = 0; t < nt; t++) {
                    double *s = tsum + (size_t)t * K * D;
                    long *cc = tcnt + (size_t)t * K;
                    for (int c = 0; c < K; c++) cnt[c] += cc[c];
                    for (int c = 0; c < K * D; c++) sum[c] += s[c];
                }
                for (int c = 0; c < K; c++)
                    if (cnt[c] > 0)
                        for (int k = 0; k < D; k++) cent[c * D + k] = sum[c * D + k] / cnt[c];
            }
            #pragma omp barrier
        }
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
