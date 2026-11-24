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
    for (int it = 0; it < iters; it++) {
        // Reset global accumulators for this iteration
        for (int c = 0; c < K * D; c++) sum[c] = 0.0;
        for (int c = 0; c < K; c++) cnt[c] = 0;

        #pragma omp parallel
        {
            // Use thread-local accumulators to avoid contention and false sharing
            double t_sum[K * D] = {0};
            long t_cnt[K] = {0};

            #pragma omp for
            for (long i = 0; i < n; i++) {
                const double *p = &pts[i * D];
                int best = 0; 
                double bd = 1e300;
                
                // Assignment: find nearest centroid
                for (int c = 0; c < K; c++) {
                    double d = 0.0;
                    const double *cent_c = &cent[c * D];
                    for (int k = 0; k < D; k++) {
                        double t = p[k] - cent_c[k];
                        d += t * t;
                    }
                    if (d < bd) { 
                        bd = d; 
                        best = c; 
                    }
                }
                label[i] = best;

                // Partial Update: accumulate coordinates and count
                t_cnt[best]++;
                double *t_sum_best = &t_sum[best * D];
                for (int k = 0; k < D; k++) {
                    t_sum_best[k] += p[k];
                }
            }

            // Merge thread-local results into global accumulators
            #pragma omp critical
            {
                for (int c = 0; c < K * D; c++) sum[c] += t_sum[c];
                for (int c = 0; c < K; c++) cnt[c] += t_cnt[c];
            }
        }

        // Update step: calculate new centroids
        for (int c = 0; c < K; c++) {
            if (cnt[c] > 0) {
                for (int k = 0; k < D; k++) {
                    cent[c * D + k] = sum[c * D + k] / cnt[c];
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
