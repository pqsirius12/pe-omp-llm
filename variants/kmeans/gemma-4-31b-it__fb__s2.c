/* k14_kmeans: Lloyd's k-means, fixed number of iterations.
 * Assignment step: UNIFORM and easy to parallelize.
 * Update step: every point adds into a shared cluster sum - a reduction
 * TRAP, like the histogram. Sum order changes floating-point results, so
 * the harness compares with a relative tolerance. */
#include "common.h"
#include <omp.h>
#include <string.h>

#define D 8
#define K 16

/* === KERNEL BEGIN === */
static void kernel(long n, int iters, const double *pts, double *cent, int *label) {
    double sum[K * D];
    long cnt[K];
    // Local rearranged centroids to improve SIMD vectorization
    // Format: [dim0_cluster0, dim0_cluster1, ..., dim0_cluster15, dim1_cluster0, ...]
    double cent_rearranged[D * K];

    for (int it = 0; it < iters; it++) {
        // Rearrange centroids for the current iteration to optimize inner loop
        for (int c = 0; c < K; c++) {
            for (int k = 0; k < D; k++) {
                cent_rearranged[k * K + c] = cent[c * D + k];
            }
        }

        // Reset global accumulators
        memset(sum, 0, sizeof(sum));
        memset(cnt, 0, sizeof(cnt));

        #pragma omp parallel
        {
            // Thread-local accumulators to avoid contention and false sharing
            double t_sum[K * D];
            long t_cnt[K];
            memset(t_sum, 0, sizeof(t_sum));
            memset(t_cnt, 0, sizeof(t_cnt));

            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                const double *p = &pts[i * D];
                double dists[K] = {0};

                // Calculate distance to all centroids using the rearranged layout
                // This inner loop is highly vectorizable.
                for (int k = 0; k < D; k++) {
                    double pk = p[k];
                    const double *cent_k = &cent_rearranged[k * K];
                    #pragma omp simd
                    for (int c = 0; c < K; c++) {
                        double t = pk - cent_k[c];
                        dists[c] += t * t;
                    }
                }

                // Find the cluster with the minimum distance
                int best = 0;
                double bd = dists[0];
                for (int c = 1; c < K; c++) {
                    if (dists[c] < bd) {
                        bd = dists[c];
                        best = c;
                    }
                }
                label[i] = best;

                // Accumulate point data into thread-local cluster sums
                t_cnt[best]++;
                double *t_sum_best = &t_sum[best * D];
                #pragma omp simd
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

        // Compute new centroids for the next iteration
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

