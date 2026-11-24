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
    for (int it = 0; it < iters; it++) {
        double global_sum[K * D] = {0};
        long global_cnt[K] = {0};

        /* We use a single parallel region per iteration to avoid creating 
           too many threads. To balance the load between P-cores and E-cores,
           guided scheduling is typically effective on Apple Silicon. */
        #pragma omp parallel
        {
            /* Each thread maintains its own local sums and counts to prevent 
               false sharing and avoid expensive atomic operations inside the hot loop. */
            double local_sum[K * D] = {0};
            long local_cnt[K] = {0};

            #pragma omp for schedule(guided)
            for (long i = 0; i < n; i++) {
                int best = 0; 
                double bd = 1e300;
                const double *p = &pts[i * D];

                /* Assignment Step: Find the closest centroid */
                for (int c = 0; c < K; c++) {
                    double d = 0.0;
                    const double *cen = &cent[c * D];
                    for (int k = 0; k < D; k++) {
                        double t = p[k] - cen[k];
                        d += t * t;
                    }
                    if (d < bd) { 
                        bd = d; 
                        best = c; 
                    }
                }
                label[i] = best;

                /* Update Step (Combined): Accumulate values for the new centroid 
                   calculation. Doing this here instead of in a separate loop 
                   reduces the number of passes over the large 'pts' array. */
                local_cnt[best]++;
                for (int k = 0; k < D; k++) {
                    local_sum[best * D + k] += p[k];
                }
            }

            /* Final Reduction: Merge thread-local buffers into the global totals.
               Since K*D is small (128), this critical section is very fast. */
            #pragma omp critical
            {
                for (int c = 0; c < K * D; c++) global_sum[c] += local_sum[c];
                for (int c = 0; c < K; c++) global_cnt[c] += local_cnt[c];
            }
        }

        /* Recalculate centroid positions based on the averaged sums. */
        for (int c = 0; c < K; c++) {
            if (global_cnt[c] > 0) {
                double inv_cnt = 1.0 / (double)global_cnt[c];
                for (int k = 0; k < D; k++) {
                    cent[c * D + k] = global_sum[c * D + k] * inv_cnt;
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

