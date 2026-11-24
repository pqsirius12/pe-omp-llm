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

    // Initialize shared sums and counts
    for (int c = 0; c < K * D; c++) sum[c] = 0.0;
    for (int c = 0; c < K; c++) cnt[c] = 0;

    #pragma omp parallel
    {
        // Thread-local buffers to avoid contention during the update step
        double local_sum[K * D];
        long local_cnt[K];

        for (int it = 0; it < iters; it++) {
            // 1. Assignment Step
            // Workload per point is uniform; guided schedule helps handle P-core/E-core disparity.
            #pragma omp for schedule(guided)
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

            // Reset thread-local buffers for this iteration
            for (int c = 0; c < K * D; c++) local_sum[c] = 0.0;
            for (int c = 0; c < K; c++) local_cnt[c] = 0;

            // 2. Local Update Step
            #pragma omp for schedule(guided)
            for (long i = 0; i < n; i++) {
                int lb = label[i];
                local_cnt[lb]++;
                for (int k = 0; k < D; k++) {
                    local_sum[lb * D + k] += pts[i * D + k];
                }
            }

            // Global reduction: add local buffers into the shared sum/cnt arrays
            #pragma omp critical
            {
                for (int c = 0; c < K; c++) {
                    cnt[c] += local_cnt[c];
                    for (int k = 0; k < D; k++) {
                        sum[c * D + k] += local_sum[c * D + k];
                    }
                }
            }

            // Ensure all threads have contributed to the global sums before calculating centroids
            #pragma omp barrier

            // 3. Centroid Update and Reset
            #pragma omp single
            {
                for (int c = 0; c < K; c++) {
                    if (cnt[c] > 0) {
                        for (int k = 0; k < D; k++) {
                            cent[c * D + k] = sum[c * D + k] / cnt[c];
                        }
                    }
                }
                // Reset global trackers for the next iteration
                for (int c = 0; c < K * D; c++) sum[c] = 0.0;
                for (int c = 0; c < K; c++) cnt[c] = 0;
            }
            // Implicit barrier at the end of 'omp single' ensures centroids are ready for next iter.
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
