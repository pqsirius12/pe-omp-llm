/* k14_kmeans: Lloyd's k-means, fixed number of iterations.
 * Assignment step: UNIFORM and easy to parallelize.
 * Update step: every point adds into a shared cluster sum - a reduction
 * TRAP, like the histogram. Sum order changes floating-point results, so
 * the harness compares with a relative tolerance. */
#include "common.h"
#include <omp.h>

#define D 8
#define K 16

/* === KERNEL BEGIN === */
static void kernel(long n, int iters, const double *pts, double *cent, int *label) {
    double sum[K * D];
    long cnt[K];

    /* 
     * We use a single parallel region outside the iteration loop to avoid 
     * the overhead of repeated thread spawning and joining.
     * 
     * The Apple M4 has an asymmetric architecture (4 P-cores, 6 E-cores).
     * To handle this, we use 'schedule(guided)', which dynamically distributes
     * work to keep faster cores busy without incurring the high overhead 
     * of 'schedule(dynamic)'.
     */
    #pragma omp parallel
    {
        /* Each thread maintains its own local buffers for reduction to avoid 
           contention and false sharing during the update phase. */
        double lsum[K * D];
        long lcnt[K];

        for (int it = 0; it < iters; it++) {
            /* Assignment Step: Perfectly uniform workload per point. */
            #pragma omp for schedule(guided)
            for (long i = 0; i < n; i++) {
                int best = 0;
                double bd = 1e300;
                const double *p = &pts[i * D];
                for (int c = 0; c < K; c++) {
                    double d = 0.0;
                    const double *cp = &cent[c * D];
                    for (int k = 0; k < D; k++) {
                        double t = p[k] - cp[k];
                        d += t * t;
                    }
                    if (d < bd) {
                        bd = d;
                        best = c;
                    }
                }
                label[i] = best;
            }

            /* Update Step:
               1. Reset global sum/cnt and local buffers.
               2. Accumulate points into local buffers.
               3. Merge local buffers into global sum/cnt.
               4. Recalculate centroids. */

            #pragma omp single
            {
                for (int c = 0; c < K * D; c++) sum[c] = 0.0;
                for (int c = 0; c < K; c++) cnt[c] = 0;
            }
            /* Implicit barrier at end of 'single' ensures reset is complete. */

            /* Reset local buffers for this iteration. */
            for (int c = 0; c < K * D; c++) lsum[c] = 0.0;
            for (int c = 0; c < K; c++) lcnt[c] = 0;

            #pragma omp for schedule(guided)
            for (long i = 0; i < n; i++) {
                int lb = label[i];
                lcnt[lb]++;
                const double *p = &pts[i * D];
                double *ls = &lsum[lb * D];
                for (int k = 0; k < D; k++) {
                    ls[k] += p[k];
                }
            }

            /* Merge local results into global buffers.
               Since T is small (up to 10), a critical section is efficient. */
            #pragma omp critical
            {
                for (int c = 0; c < K * D; c++) sum[c] += lsum[c];
                for (int c = 0; c < K; c++) cnt[c] += lcnt[c];
            }

            #pragma omp single
            {
                for (int c = 0; c < K; c++) {
                    if (cnt[c] > 0) {
                        double inv_cnt = 1.0 / (double)cnt[c];
                        for (int k = 0; k < D; k++) {
                            cent[c * D + k] = sum[c * D + k] * inv_cnt;
                        }
                    }
                }
            }
            /* Implicit barrier at end of 'single' ensures centroids are updated before next iteration. */
        }
    }
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
