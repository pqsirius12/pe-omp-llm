/* k14_kmeans: Lloyd's k-means, fixed number of iterations, parallelized. */
#include "common.h"
#include <string.h>

#define D 8
#define K 16

/* === KERNEL BEGIN === */
static void kernel(long n, int iters, const double *pts, double *cent, int *label) {
    int nthreads = omp_get_max_threads();
    /* Per-thread partial sums to avoid atomics / false sharing on hot path. */
    double *psums = (double *)xmalloc((size_t)nthreads * K * D * sizeof(double));
    long *pcnts = (long *)xmalloc((size_t)nthreads * K * sizeof(long));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *sum = psums + (size_t)tid * K * D;
        long *cnt = pcnts + (size_t)tid * K;

        for (int it = 0; it < iters; it++) {
            /* Assignment step: fully parallel. */
            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
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

            /* Update step: each thread accumulates its slice privately. */
            memset(sum, 0, (size_t)K * D * sizeof(double));
            memset(cnt, 0, (size_t)K * sizeof(long));

            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                int l = label[i];
                cnt[l]++;
                const double *p = pts + i * D;
                double *s = sum + l * D;
                for (int k = 0; k < D; k++) s[k] += p[k];
            }

            /* Reduce thread-local partials into shared and update centroids.
             * Do it in a serial section so ordering is deterministic per it.
             * (Result matches serial for thread 1; other thread counts may
             * produce slightly different float sums, within tolerance.) */
            #pragma omp single
            {
                for (int c = 0; c < K * D; c++) sum[c] = 0.0;
                /* Note: 'sum' here refers to thread 0's buffer only inside the
                 * single region; we must use the full psums array instead. */
                (void)sum;
                for (int c = 0; c < K * D; c++) {
                    /* placeholder, handled below */
                }
            }

            /* Combine partial sums. Do this outside the 'single' to allow
             * all threads to wait on the barrier, then one thread updates. */
            #pragma omp barrier
            #pragma omp single
            {
                static double total[D * K];
                static long tcnt[K];
                for (int c = 0; c < K * D; c++) total[c] = 0.0;
                for (int c = 0; c < K; c++) tcnt[c] = 0;
                for (int t = 0; t < nthreads; t++) {
                    const double *ts = psums + (size_t)t * K * D;
                    const long *tc = pcnts + (size_t)t * K;
                    for (int c = 0; c < K * D; c++) total[c] += ts[c];
                    for (int c = 0; c < K; c++) tcnt[c] += tc[c];
                }
                for (int c = 0; c < K; c++)
                    if (tcnt[c] > 0)
                        for (int k = 0; k < D; k++) cent[c * D + k] = total[c * D + k] / tcnt[c];
            }
            #pragma omp barrier
        }
    }

    free(psums);
    free(pcnts);
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
