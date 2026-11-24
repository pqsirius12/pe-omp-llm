#include <math.h>
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, double r, const double *x, const double *y, int *nb) {
    const double r2 = r * r;
    /* 
     * The workload is irregular because points are clustered, meaning iterations 
     * within clusters require significantly more distance checks. Additionally, 
     * the target machine has asymmetric cores (P-cores and E-cores).
     * 
     * 'schedule(dynamic, 64)' is chosen to ensure load balancing across 
     * heterogeneous cores and to prevent threads from getting stuck on 
     * dense clusters (which are contiguous in the sorted x-array). 
     * A chunk size of 64 reduces scheduling overhead while maintaining flexibility.
     */
    #pragma omp parallel for schedule(dynamic, 64)
    for (long i = 0; i < n; i++) {
        int c = 0;
        const double xi = x[i];
        const double yi = y[i];

        /* Scan forward in sorted x-array until x is out of range */
        for (long j = i + 1; j < n; j++) {
            const double dx = x[j] - xi;
            if (dx > r) break;
            const double dy = y[j] - yi;
            if (dx * dx + dy * dy <= r2) c++;
        }

        /* Scan backward in sorted x-array until x is out of range */
        for (long j = i - 1; j >= 0; j--) {
            const double dx = xi - x[j];
            if (dx > r) break;
            const double dy = yi - y[j];
            if (dx * dx + dy * dy <= r2) c++;
        }
        nb[i] = c;
    }
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
