/* k15_cutoff: for each point, count neighbours within radius r, scanning
 * points sorted by x and stopping once x is out of range.
 * IRREGULAR: points are clustered, so points inside dense clusters scan
 * far more neighbours than isolated ones (like molecular dynamics). */
#include <math.h>
#include <string.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(long n, double r, const double *x, const double *y, int *nb) {
    if (n <= 0) return;
    const double r2 = r * r;
    const int maxt = omp_get_max_threads();
    int *buf = (int *)xmalloc((size_t)maxt * (size_t)n * sizeof(int));

    #pragma omp parallel
    {
        const int team = omp_get_num_threads();
        const int t = omp_get_thread_num();
        int *restrict acc = buf + (size_t)t * (size_t)n;
        const double *restrict X = x;
        const double *restrict Y = y;
        memset(acc, 0, (size_t)n * sizeof(int));

        /* Each unordered pair (i<j) is tested exactly once; both endpoints
         * are credited.  Identical arithmetic to the serial version, so the
         * counts are bit-for-bit the same. */
        #pragma omp for schedule(dynamic, 64)
        for (long i = 0; i < n; i++) {
            const double xi = X[i], yi = Y[i];
            /* first j > i with x[j] - x[i] > r (x sorted => monotone) */
            long lo = i + 1, hi = n;
            while (lo < hi) {
                long mid = lo + ((hi - lo) >> 1);
                if (X[mid] - xi <= r) lo = mid + 1; else hi = mid;
            }
            const long e = lo;
            int c = 0;
            for (long j = i + 1; j < e; j++) {
                double dx = X[j] - xi, dy = Y[j] - yi;
                int m = (dx * dx + dy * dy <= r2);
                c += m;
                acc[j] += m;
            }
            acc[i] += c;
        }
        /* implicit barrier: all private counts complete */

        #pragma omp for schedule(static)
        for (long i = 0; i < n; i++) {
            int s = 0;
            for (int k = 0; k < team; k++) s += buf[(size_t)k * (size_t)n + i];
            nb[i] = s;
        }
    }
    free(buf);
}
/* === KERNEL END === */

static int cmp_x(const void *a, const void *b) {
    double d = ((const double *)a)[0] - ((const double *)b)[0];
    return (d > 0) - (d < 0);
}

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 400000);
    double r = 0.002;
    double *xy = xmalloc(2 * n * sizeof *xy);
    rng_seed(15);
    /* 70% of points in 8 tight Gaussian-ish clusters, 30% spread uniformly */
    for (long i = 0; i < n; i++) {
        if (rng_unit() < 0.7) {
            int c = (int)(rng_next() % 8);
            double cx = 0.1 + 0.1 * c, cy = 0.5;
            double a = rng_unit() + rng_unit() + rng_unit() - 1.5;   /* ~bell */
            double b = rng_unit() + rng_unit() + rng_unit() - 1.5;
            xy[2 * i] = cx + 0.01 * a; xy[2 * i + 1] = cy + 0.01 * b;
        } else { xy[2 * i] = rng_unit(); xy[2 * i + 1] = rng_unit(); }
    }
    qsort(xy, n, 2 * sizeof *xy, cmp_x);   /* sort (x,y) pairs by x */
    double *x = xmalloc(n * sizeof *x), *y = xmalloc(n * sizeof *y);
    for (long i = 0; i < n; i++) { x[i] = xy[2 * i]; y[i] = xy[2 * i + 1]; }
    int *nb = xmalloc(n * sizeof *nb);

    double t0 = now_sec();
    kernel(n, r, x, y, nb);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 0; i < n; i++) sum += nb[i];
    report("cutoff", t1 - t0, sum);
    free(xy); free(x); free(y); free(nb);
    return 0;
}
