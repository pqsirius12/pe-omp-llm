/* k06_spmv: y = A*x for a sparse CSR matrix, repeated.
 * IRREGULAR: row lengths follow a heavy-tailed distribution, so a few
 * rows hold most of the work (like real web or social graphs). */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *rowptr, const int *col,
                   const double *val, const double *x, double *y) {
    /* 
     * The workload per row varies significantly (heavy-tailed distribution).
     * To optimize, we swap the 'reps' and 'i' loops. This provides two major benefits:
     * 1. Cache Locality: Instead of reading the huge 'col' and 'val' arrays 
     *    'reps' times, each segment is read once and reused 'reps' times while
     *    it's likely still in the L1/L2 cache.
     * 2. Reduced Synchronization: We move the OpenMP parallelization to the 'i' 
     *    loop, reducing the number of implicit barriers from (reps) to 1.
     * 
     * 'schedule(guided)' is used to balance the irregular row lengths across threads.
     */
    #pragma omp parallel for schedule(guided)
    for (long i = 0; i < n; i++) {
        const long start = rowptr[i];
        const long end = rowptr[i + 1];
        
        for (int r = 0; r < reps; r++) {
            double s = 0.0;
            for (long k = start; k < end; k++) {
                s += val[k] * x[col[k]];
            }
            y[i] = s;
        }
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 200000);
    int reps = (int)arg_long(argc, argv, 2, 20);
    long *rowptr = xmalloc((n + 1) * sizeof *rowptr);
    rng_seed(6);
    /* Row length = 4 / u^1.2 for u uniform in (0,1], capped at 20000:
     * most rows have a few entries, a handful have thousands. */
    rowptr[0] = 0;
    for (long i = 0; i < n; i++) {
        double u = 1.0 - rng_unit();
        long len = (long)(4.0 / pow(u, 1.2));
        if (len > 20000) len = 20000;
        rowptr[i + 1] = rowptr[i] + len;
    }
    long nnz = rowptr[n];
    int *col = xmalloc(nnz * sizeof *col);
    double *val = xmalloc(nnz * sizeof *val);
    double *x = xmalloc(n * sizeof *x), *y = xmalloc(n * sizeof *y);
    for (long k = 0; k < nnz; k++) { col[k] = (int)(rng_next() % n); val[k] = rng_unit(); }
    for (long i = 0; i < n; i++) x[i] = rng_unit();

    double t0 = now_sec();
    kernel(n, reps, rowptr, col, val, x, y);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 0; i < n; i++) sum += y[i];
    report("spmv", t1 - t0, sum);
    free(rowptr); free(col); free(val); free(x); free(y);
    return 0;
}

