/* k14_kmeans: Lloyd's k-means, fixed number of iterations.
 * Assignment step: UNIFORM and easy to parallelize.
 * Update step: every point adds into a shared cluster sum - a reduction
 * TRAP, like the histogram. Sum order changes floating-point results, so
 * the harness compares with a relative tolerance. */
#include "common.h"
#include <string.h>

#define D 8
#define K 16

/* === KERNEL BEGIN === */
static void kernel(long n, int iters, const double *pts, double *cent, int *label) {
    const int NT = 64;  /* thread-local accumulators */
    for (int it = 0; it < iters; it++) {
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++) {               /* assignment */
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

        double sum_local[64 * K * D];
        long cnt_local[64 * K];
        memset(sum_local, 0, sizeof(double) * NT * K * D);
        memset(cnt_local, 0, sizeof(long) * NT * K);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int slot = tid < NT ? tid : (tid % NT);
            (void)nt;
            double *sl = sum_local + (size_t)slot * K * D;
            long *cl = cnt_local + (size_t)slot * K;

            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                int l = label[i];
                cl[l]++;
                const double *p = pts + i * D;
                for (int k = 0; k < D; k++) sl[l * D + k] += p[k];
            }
        }

        /* reduce local accumulators. Number of active slots = number of threads */
        int nslots = omp_get_max_threads();
        if (nslots > NT) nslots = NT;
        double sum[K * D];
        long cnt[K];
        for (int c = 0; c < K * D; c++) sum[c] = 0.0;
        for (int c = 0; c < K; c++) cnt[c] = 0;
        for (int s = 0; s < nslots; s++) {
            double *sl = sum_local + (size_t)s * K * D;
            long *cl = cnt_local + (size_t)s * K;
            for (int c = 0; c < K; c++) {
                cnt[c] += cl[c];
                for (int k = 0; k < D; k++) sum[c * D + k] += sl[c * D + k];
            }
        }

        for (int c = 0; c < K; c++)
            if (cnt[c] > 0)
                for (int k = 0; k < D; k++) cent[c * D + k] = sum[c * D + k] / cnt[c];
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
