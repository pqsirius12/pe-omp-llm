/* k06_spmv: y = A*x for a sparse CSR matrix, repeated.
 * IRREGULAR: row lengths follow a heavy-tailed distribution, so a few
 * rows hold most of the work (like real web or social graphs). */
#include <math.h>
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *rowptr, const int *col,
                   const double *val, const double *x, double *y) {
    /* Block-cyclic distribution so heavy rows get spread across threads.
     * Within a block we write into a private buffer, then flush in order
     * to keep results deterministic (each y[i] computed once, same value). */
    for (int r = 0; r < reps; r++) {
        #pragma omp parallel
        {
            const long B = 1024;
            double *buf = (double *)malloc((size_t)B * sizeof *buf);
            if (!buf) { fprintf(stderr, "oom\n"); exit(1); }
            #pragma omp for schedule(static, 1)
            for (long blk = 0; blk * B < n; blk++) {
                long lo = blk * B;
                long hi = lo + B;
                if (hi > n) hi = n;
                for (long i = lo; i < hi; i++) {
                    double s = 0.0;
                    for (long k = rowptr[i]; k < rowptr[i + 1]; k++)
                        s += val[k] * x[col[k]];
                    buf[i - lo] = s;
                }
                for (long i = lo; i < hi; i++)
                    y[i] = buf[i - lo];
            }
            free(buf);
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
