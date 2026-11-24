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
    double global_sum[K * D];
    long global_cnt[K];
    memset(global_sum, 0, sizeof(global_sum));
    memset(global_cnt, 0, sizeof(global_cnt));

    #pragma omp parallel
    {
        /* Use thread-local buffers to avoid reduction overhead and false sharing */
        double local_sum[K * D];
        long local_cnt[K];

        for (int it = 0; it < iters; it++) {
            /* Reset local buffers for each iteration */
            memset(local_sum, 0, sizeof(local_sum));
            memset(local_cnt, 0, sizeof(local_cnt));

            #pragma omp for
            for (long i = 0; i < n; i++) {
                const double *p = &pts[i * D];
                int best = 0;
                double bd = 1e300;

                /* Assignment: Find nearest centroid */
                for (int c = 0; c < K; c++) {
                    const double *cp = &cent[c * D];
                    double d = 0.0;
                    /* Manually unrolling the D-loop for performance and SIMD friendliness */
                    double t0 = p[0] - cp[0]; d += t0 * t0;
                    double t1 = p[1] - cp[1]; d += t1 * t1;
                    double t2 = p[2] - cp[2]; d += t2 * t2;
                    double t3 = p[3] - cp[3]; d += t3 * t3;
                    double t4 = p[4] - cp[4]; d += t4 * t4;
                    double t5 = p[5] - cp[5]; d += t5 * t5;
                    double t6 = p[6] - cp[6]; d += t6 * t6;
                    double t7 = p[7] - cp[7]; d += t7 * t7;
                    
                    if (d < bd) {
                        bd = d;
                        best = c;
                    }
                }
                label[i] = best;

                /* Local accumulation of cluster sums and counts */
                local_cnt[best]++;
                double *ls_best = &local_sum[best * D];
                ls_best[0] += p[0];
                ls_best[1] += p[1];
                ls_best[2] += p[2];
                ls_best[3] += p[3];
                ls_best[4] += p[4];
                ls_best[5] += p[5];
                ls_best[6] += p[6];
                ls_best[7] += p[7];
            }

            /* Merge local thread buffers into global buffers */
            #pragma omp critical
            {
                for (int j = 0; j < K * D; j++) global_sum[j] += local_sum[j];
                for (int j = 0; j < K; j++) global_cnt[j] += local_cnt[j];
            }
            
            /* Ensure all threads have merged before updating centroids */
            #pragma omp barrier

            #pragma omp single
            {
                /* Update centroids based on global sums and counts */
                for (int c = 0; c < K; c++) {
                    if (global_cnt[c] > 0) {
                        double inv_cnt = 1.0 / (double)global_cnt[c];
                        for (int k = 0; k < D; k++) {
                            cent[c * D + k] = global_sum[c * D + k] * inv_cnt;
                        }
                    }
                }
                /* Reset global buffers for the next iteration */
                memset(global_sum, 0, sizeof(global_sum));
                memset(global_cnt, 0, sizeof(global_cnt));
            }
            /* Implicit barrier at the end of 'single' ensures all threads see updated centroids */
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

