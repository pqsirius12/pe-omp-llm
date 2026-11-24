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
/* Strategy:
 *  - One parallel region spans all iterations (no per-iteration fork/join).
 *  - Assignment and accumulation are fused: each thread assigns its static
 *    block of points and adds them into its own private, cache-line-padded
 *    partial sums (no atomics, no false sharing).
 *  - The partial sums are combined in fixed thread order, parallelized over
 *    the K*D centroid entries, and the new centroids are written directly.
 *    With 1 thread the summation order is identical to the serial code. */
#define SLAB_D (K * D)          /* 128 doubles = 1024 bytes per thread  */
#define SLAB_C 16               /* 16 longs    = 128 bytes per thread   */

static void kernel(long n, int iters, const double *pts, double *cent, int *label) {
    int maxt = omp_get_max_threads();
    if (maxt < 1) maxt = 1;
    double *tsum = aligned_alloc(64, (size_t)maxt * SLAB_D * sizeof(double));
    long   *tcnt = aligned_alloc(64, (size_t)maxt * SLAB_C * sizeof(long));
    if (!tsum || !tcnt) { fprintf(stderr, "out of memory\n"); exit(1); }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
        double *my_sum = tsum + (size_t)tid * SLAB_D;
        long   *my_cnt = tcnt + (size_t)tid * SLAB_C;

        for (int it = 0; it < iters; it++) {
            double lsum[K * D];
            long lcnt[K];
            double lc[K * D];
            for (int c = 0; c < K * D; c++) { lsum[c] = 0.0; lc[c] = cent[c]; }
            for (int c = 0; c < K; c++) lcnt[c] = 0;

            /* fused assignment + local accumulation */
            #pragma omp for schedule(static) nowait
            for (long i = 0; i < n; i++) {
                const double *p = pts + i * D;
                int best = 0; double bd = 1e300;
                for (int c = 0; c < K; c++) {
                    double d = 0.0;
                    for (int k = 0; k < D; k++) {
                        double t = p[k] - lc[c * D + k];
                        d += t * t;
                    }
                    if (d < bd) { bd = d; best = c; }
                }
                label[i] = best;
                lcnt[best]++;
                for (int k = 0; k < D; k++) lsum[best * D + k] += p[k];
            }

            for (int c = 0; c < K * D; c++) my_sum[c] = lsum[c];
            for (int c = 0; c < K; c++) my_cnt[c] = lcnt[c];

            #pragma omp barrier

            /* combine partials in fixed thread order and update centroids */
            #pragma omp for schedule(static)
            for (int e = 0; e < K * D; e++) {
                int c = e / D;
                double s = 0.0;
                long cn = 0;
                for (int t = 0; t < nt; t++) {
                    s  += tsum[(size_t)t * SLAB_D + e];
                    cn += tcnt[(size_t)t * SLAB_C + c];
                }
                if (cn > 0) cent[e] = s / cn;
            }
            /* implicit barrier: new centroids visible before next iteration */
        }
    }

    free(tsum);
    free(tcnt);
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
