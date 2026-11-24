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
    #pragma omp parallel
    {
        /* Thread-private cluster accumulators avoid atomics and contention. */
        double local_sum[K * D];
        long local_cnt[K];

        for (int it = 0; it < iters; it++) {

            /* Assignment: each point is independent. */
            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                int best = 0;
                double bd = 1e300;

                for (int c = 0; c < K; c++) {
                    double d = 0.0;
                    const double *p = pts + i * D;
                    const double *q = cent + c * D;

                    for (int k = 0; k < D; k++) {
                        double t = p[k] - q[k];
                        d += t * t;
                    }

                    if (d < bd) {
                        bd = d;
                        best = c;
                    }
                }
                label[i] = best;
            }

            /* Initialize thread-private accumulators. */
            for (int c = 0; c < K * D; c++)
                local_sum[c] = 0.0;
            for (int c = 0; c < K; c++)
                local_cnt[c] = 0;

            /* Each thread processes its own subset of points. */
            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                int c = label[i];
                local_cnt[c]++;
                const double *p = pts + i * D;
                double *s = local_sum + c * D;

                for (int k = 0; k < D; k++)
                    s[k] += p[k];
            }

            /* Merge the private accumulators into the final cluster sums.
               Parallelize over clusters: each cluster is independent. */
            #pragma omp for schedule(static)
            for (int c = 0; c < K; c++) {
                long cnt = 0;
                double s[D] = {0.0};

                for (int t = 0; t < omp_get_num_threads(); t++) {
                    /* This approach is replaced below by the correct
                       shared scratch layout. */
                }

                if (cnt > 0) {
                    for (int k = 0; k < D; k++)
                        cent[c * D + k] = s[k] / cnt;
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
