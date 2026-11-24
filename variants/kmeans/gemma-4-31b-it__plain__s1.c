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
    double global_sum[K * D];
    long global_cnt[K];

    #pragma omp parallel
    {
        for (int it = 0; it < iters; it++) {
            /* Reset global accumulation buffers for each iteration */
            #pragma omp single
            {
                for (int c = 0; c < K * D; c++) global_sum[c] = 0.0;
                for (int c = 0; c < K; c++) global_cnt[c] = 0;
            }

            /* Combined Assignment and Accumulation phase */
            #pragma omp for reduction(+:global_sum[0:K*D], global_cnt[0:K])
            for (long i = 0; i < n; i++) {
                const double *p = &pts[i * D];
                int best = 0;
                double bd = 1e300;
                
                /* Assignment: Find nearest centroid */
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
                
                /* Accumulation: Update cluster sums and counts */
                global_cnt[best]++;
                for (int k = 0; k < D; k++) {
                    global_sum[best * D + k] += p[k];
                }
            }

            /* Normalization: Update centroids for the next iteration */
            #pragma omp single
            {
                for (int c = 0; c < K; c++) {
                    if (global_cnt[c] > 0) {
                        for (int k = 0; k < D; k++) {
                            cent[c * D + k] = global_sum[c * D + k] / (double)global_cnt[c];
                        }
                    }
                }
            }
            /* Implicit barrier at the end of 'single' ensures all threads see 
               updated centroids before the next iteration starts. */
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
