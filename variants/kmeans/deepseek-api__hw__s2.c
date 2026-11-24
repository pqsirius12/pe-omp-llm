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
    /* Per-thread partial sums to avoid contention in the update step.
     * Size = max threads * K * D doubles and max threads * K longs. */
    int nth = omp_get_max_threads();
    if (nth < 1) nth = 1;
    double *partial = (double *)xmalloc((size_t)nth * K * D * sizeof(double));
    long *pcnt = (long *)xmalloc((size_t)nth * K * sizeof(long));
    double sum[K * D];
    long cnt[K];

    for (int it = 0; it < iters; it++) {
        /* Assignment step: fully independent per point. */
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++) {
            int best = 0; double bd = 1e300;
            for (int c = 0; c < K; c++) {
                double d = 0.0;
                for (int k = 0; k < D; k++) {
                    double t = pts[i * D + k] - cent[c * D + k];
                    d += t * t;
                }
                if (d < bd) { bd = d; best = c; }
            }
            label[i] = best;
        }

        /* Update step: parallel accumulation into per-thread buffers. */
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            double *ls = partial + (size_t)tid * K * D;
            long *lc = pcnt + (size_t)tid * K;
            for (int c = 0; c < K * D; c++) ls[c] = 0.0;
            for (int c = 0; c < K; c++) lc[c] = 0;

            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                int lbl = label[i];
                lc[lbl]++;
                const double *pp = pts + i * D;
                double *sp = ls + lbl * D;
                for (int k = 0; k < D; k++) sp[k] += pp[k];
            }

            /* Reduce per-thread buffers into shared sum/cnt. */
            #pragma omp critical
            {
                if (tid == 0) {
                    for (int c = 0; c < K * D; c++) sum[c] = 0.0;
                    for (int c = 0; c < K; c++) cnt[c] = 0;
                }
            }
            #pragma omp barrier
            #pragma omp critical
            {
                for (int c = 0; c < K * D; c++) sum[c] += ls[c];
                for (int c = 0; c < K; c++) cnt[c] += lc[c];
            }
            (void)nt;
        }

        for (int c = 0; c < K; c++)
            if (cnt[c] > 0)
                for (int k = 0; k < D; k++) cent[c * D + k] = sum[c * D + k] / cnt[c];
    }

    free(partial);
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
