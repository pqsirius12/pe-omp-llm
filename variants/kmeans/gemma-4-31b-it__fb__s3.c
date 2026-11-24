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
static void kernel(long n, int iters, const double * restrict pts, double * restrict cent, int * restrict label) {
    double sum[K * D];
    long cnt[K];

    for (int it = 0; it < iters; it++) {
        // Reset sums and counts for the update step
        for (int c = 0; c < K * D; c++) sum[c] = 0.0;
        for (int c = 0; c < K; c++) cnt[c] = 0;

        /* 
         * Merge assignment and aggregation into a single pass over the points array.
         * This reduces memory bandwidth pressure by reading 'pts' once per iteration.
         * Using OpenMP array reduction for the cluster sums and counts.
         * Each thread maintains local buffers for 'sum' and 'cnt' to avoid race 
         * conditions and contention, which are then combined at the end of the loop.
         */
        #pragma omp parallel for reduction(+:sum[0:K*D], cnt[0:K])
        for (long i = 0; i < n; i++) {
            const double * restrict p = &pts[i * D];
            int best = 0;
            double bd = 1e300;

            // Assignment Step: find the nearest centroid
            // With K and D as small constants, the compiler can aggressively unroll 
            // and vectorize these loops.
            for (int c = 0; c < K; c++) {
                double d = 0.0;
                const double * restrict cp = &cent[c * D];
                #pragma omp simd reduction(+:d)
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

            // Aggregation Step: accumulate coordinates for the next centroids
            cnt[best]++;
            double * restrict sp = &sum[best * D];
            #pragma omp simd
            for (int k = 0; k < D; k++) {
                sp[k] += p[k];
            }
        }

        // Update Step: calculate new centroid positions
        for (int c = 0; c < K; c++) {
            if (cnt[c] > 0) {
                double inv_cnt = 1.0 / (double)cnt[c];
                for (int k = 0; k < D; k++) {
                    cent[c * D + k] = sum[c * D + k] * inv_cnt;
                }
            }
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

