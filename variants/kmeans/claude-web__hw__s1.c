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
 *  - One parallel region for all iterations, so threads are not re-created.
 *  - The points are split into fixed-size blocks. Each block is handed out
 *    dynamically, so fast P-cores take more blocks than slow E-cores.
 *  - Assignment and update are fused: while labelling its points, a block
 *    also builds its own partial cluster sums and counts in private buffers.
 *    This avoids a second pass over the data and avoids shared-sum contention.
 *  - Partials are stored per BLOCK, not per thread, and combined in fixed
 *    block order. The result is the same for every thread count and every
 *    schedule. Distances use the same arithmetic order as the serial code,
 *    so for equal centroids the labels are identical.
 */
#define KM_BS 2048L   /* points per block */

static void kernel(long n, int iters, const double *pts, double *cent, int *label) {
    long nb = (n + KM_BS - 1) / KM_BS;
    double *psum = xmalloc((size_t)(nb > 0 ? nb : 1) * K * D * sizeof *psum);
    long   *pcnt = xmalloc((size_t)(nb > 0 ? nb : 1) * K * sizeof *pcnt);

    #pragma omp parallel
    {
        for (int it = 0; it < iters; it++) {
            /* Fused assignment + per-block partial update. */
            #pragma omp for schedule(dynamic, 1)
            for (long b = 0; b < nb; b++) {
                double cl[K * D];
                double s[K * D];
                long cn[K];
                for (int c = 0; c < K * D; c++) { cl[c] = cent[c]; s[c] = 0.0; }
                for (int c = 0; c < K; c++) cn[c] = 0;

                long i0 = b * KM_BS;
                long i1 = i0 + KM_BS;
                if (i1 > n) i1 = n;

                for (long i = i0; i < i1; i++) {
                    double p[D];
                    for (int k = 0; k < D; k++) p[k] = pts[i * D + k];
                    int best = 0; double bd = 1e300;
                    for (int c = 0; c < K; c++) {
                        double d = 0.0;
                        for (int k = 0; k < D; k++) {
                            double t = p[k] - cl[c * D + k];
                            d += t * t;
                        }
                        if (d < bd) { bd = d; best = c; }
                    }
                    label[i] = best;
                    cn[best]++;
                    for (int k = 0; k < D; k++) s[best * D + k] += p[k];
                }

                double *ps = psum + b * (K * D);
                long   *pc = pcnt + b * K;
                for (int c = 0; c < K * D; c++) ps[c] = s[c];
                for (int c = 0; c < K; c++) pc[c] = cn[c];
            } /* implicit barrier: all partials are ready */

            /* Combine the partials in fixed block order and update the centroids. */
            #pragma omp for schedule(static)
            for (int c = 0; c < K; c++) {
                double s[D];
                long cnt = 0;
                for (int k = 0; k < D; k++) s[k] = 0.0;
                for (long b = 0; b < nb; b++) {
                    cnt += pcnt[b * K + c];
                    const double *ps = psum + b * (K * D) + c * D;
                    for (int k = 0; k < D; k++) s[k] += ps[k];
                }
                if (cnt > 0)
                    for (int k = 0; k < D; k++) cent[c * D + k] = s[k] / cnt;
            } /* implicit barrier: new centroids are visible before the next iteration */
        }
    }

    free(psum);
    free(pcnt);
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
