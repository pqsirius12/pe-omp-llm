#include "common.h"
#include <omp.h>

#define D 8
#define K 16

/* === KERNEL BEGIN === */
static void kernel(long n, int iters, const double *pts, double *cent, int *label) {
    for (int it = 0; it < iters; it++) {
        double sum[K * D] = {0};
        long cnt[K] = {0};

        /* 
         * We combine the assignment and the sum/count update into a single parallel pass 
         * over the points. This reduces memory traffic by avoiding writing and 
         * then reading back the labels, and reduces the number of parallel regions.
         * 
         * We use 'schedule(guided)' to balance the load across the asymmetric 
         * P-cores and E-cores of the Apple M4. 
         * 
         * The array reduction (OpenMP 4.5+) is used to aggregate local sums and 
         * counts efficiently without atomic contention.
         */
        #pragma omp parallel for reduction(+:sum, cnt) schedule(guided)
        for (long i = 0; i < n; i++) {
            const double *p_i = &pts[i * D];
            int best = 0;
            double bd = 1e300;

            /* Assignment step: Find the closest centroid */
            for (int c = 0; c < K; c++) {
                const double *p_c = &cent[c * D];
                double d = 0.0;
                /* The compiler will vectorize this tight loop since D is a constant */
                for (int k = 0; k < D; k++) {
                    double t = p_i[k] - p_c[k];
                    d += t * t;
                }
                if (d < bd) {
                    bd = d;
                    best = c;
                }
            }
            label[i] = best;

            /* Update step: Accumulate the point into its assigned cluster */
            cnt[best]++;
            for (int k = 0; k < D; k++) {
                sum[best * D + k] += p_i[k];
            }
        }

        /* Calculate new centroids based on the sums and counts (fast serial step) */
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

