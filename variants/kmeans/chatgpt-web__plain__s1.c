/* k14_kmeans: Lloyd’s k-means, fixed number of iterations.
 * Assignment step: UNIFORM and easy to parallelize.
 * Update step: every point adds into a shared cluster sum - a reduction
 * TRAP, like the histogram. Sum order changes floating-point results, so
 * the harness compares with a relative tolerance. */
#include "common.h"

#define D 8
#define K 16

/* === KERNEL BEGIN === */
static void kernel(long n, int iters, const double *pts, double *cent, int *label) {
    double sum[K * D];
    long cnt[K];

    /* Thread-local cluster accumulators to avoid contention/atomics.
       10 threads max, matching the benchmark's OMP_NUM_THREADS range. */
    #define MAX_THREADS 10
    double local_sum[MAX_THREADS][K * D];
    long local_cnt[MAX_THREADS][K];

    for (int it = 0; it < iters; it++) {

        /* Assignment: each point independently finds its closest centroid. */
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++) {
            int best = 0;
            double bd = 1e300;

            for (int c = 0; c < K; c++) {
                double d = 0.0;
                for (int k = 0; k < D; k++) {
                    double t = pts[i * D + k] - cent[c * D + k];
                    d += t * t;
                }
                if (d < bd) {
                    bd = d;
                    best = c;
                }
            }
            label[i] = best;
        }

        /* Initialize thread-local accumulators. */
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            for (int c = 0; c < K * D; c++)
                local_sum[tid][c] = 0.0;
            for (int c = 0; c < K; c++)
                local_cnt[tid][c] = 0;

            /* Each thread accumulates only its own assigned points. */
            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                int cl = label[i];
                local_cnt[tid][cl]++;
                const double *p = pts + i * D;
                double *s = local_sum[tid] + cl * D;
                for (int k = 0; k < D; k++)
                    s[k] += p[k];
            }
        }

        /* Merge the small per-thread cluster accumulators. */
        for (int c = 0; c < K * D; c++)
            sum[c] = 0.0;
        for (int c = 0; c < K; c++)
            cnt[c] = 0;

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < K; c++) {
            long total_count = 0;
            double totals[D] = {0.0};

            for (int t = 0; t < MAX_THREADS; t++) {
                total_count += local_cnt[t][c];
                for (int k = 0; k < D; k++)
                    totals[k] += local_sum[t][c * D + k];
            }

            cnt[c] = total_count;
            for (int k = 0; k < D; k++)
                sum[c * D + k] = totals[k];
        }

        /* Update centroids. */
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < K; c++) {
            if (cnt[c] > 0) {
                for (int k = 0; k < D; k++)
                    cent[c * D + k] = sum[c * D + k] / cnt[c];
            }
        }
    }

    #undef MAX_THREADS
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
