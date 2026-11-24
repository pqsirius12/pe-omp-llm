/* k06_spmv: y = A*x for a sparse CSR matrix, repeated.
 * IRREGULAR: row lengths follow a heavy-tailed distribution, so a few
 * rows hold most of the work (like real web or social graphs). */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
static inline void spmv_rows(long lo, long hi, const long *rowptr,
                             const int *col, const double *val,
                             const double *x, double *y) {
    for (long i = lo; i < hi; i++) {
        double s = 0.0;
        const long kend = rowptr[i + 1];
        for (long k = rowptr[i]; k < kend; k++)
            s += val[k] * x[col[k]];          /* same order as serial */
        y[i] = s;
    }
}

static void kernel(long n, int reps, const long *rowptr, const int *col,
                   const double *val, const double *x, double *y) {
    int nt = omp_get_max_threads();
    if (nt <= 1 || n < 2) {
        for (int r = 0; r < reps; r++)
            spmv_rows(0, n, rowptr, col, val, x, y);
        return;
    }

    /* Cost model: nnz plus a small per-row overhead.
     * Split rows into chunks of roughly equal cost (rows are never split,
     * so each y[i] is summed in exactly the serial order). Many more chunks
     * than threads + dynamic scheduling lets fast P-cores take more work
     * than slow E-cores. */
    const long ROW_COST = 2;
    long total = rowptr[n] + ROW_COST * n;
    long C = (long)nt * 48;
    if (C > n) C = n;

    long *bounds = xmalloc((size_t)(C + 1) * sizeof *bounds);
    bounds[0] = 0;
    bounds[C] = n;
    for (long c = 1; c < C; c++) {
        long target = (long)((double)total * (double)c / (double)C);
        long lo = bounds[c - 1], hi = n;   /* first i with cost(i) >= target */
        while (lo < hi) {
            long mid = lo + (hi - lo) / 2;
            if (rowptr[mid] + ROW_COST * mid < target) lo = mid + 1;
            else hi = mid;
        }
        bounds[c] = lo;
    }

    #pragma omp parallel
    {
        for (int r = 0; r < reps; r++) {
            #pragma omp for schedule(dynamic, 1)
            for (long c = 0; c < C; c++)
                spmv_rows(bounds[c], bounds[c + 1], rowptr, col, val, x, y);
        }
    }

    free(bounds);
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
