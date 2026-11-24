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
     * The SpMV operation is repeated 'reps' times. Since the input vector 'x' 
     * and the matrix (rowptr, col, val) are constant across repetitions, 
     * the resulting vector 'y' is the same for every repetition.
     * 
     * To maximize performance, we flip the order of the repetition loop (r) 
     * and the row loop (i). This provides two major benefits:
     * 1. It eliminates the overhead of 'reps' implicit barriers.
     * 2. It significantly improves cache locality. For a given row, the 
     *    corresponding 'val' and 'col' entries, as well as the gathered 
     *    values from 'x', are loaded into the cache and reused for 
     *    all 'reps' iterations, rather than being re-loaded from memory 
     *    in every repetition.
     */
    #pragma omp parallel
    {
        /* 'guided' scheduling is used to handle the heavy-tailed row 
           distribution, ensuring that threads processing short rows 
           don't idle while others handle long rows. */
        #pragma omp for schedule(guided)
        for (long i = 0; i < n; i++) {
            const long start = rowptr[i];
            const long end = rowptr[i + 1];
            
            for (int r = 0; r < reps; r++) {
                double s = 0.0;
                /* Use SIMD to vectorize the inner dot product. 
                    Modern compilers will use gather instructions for x[col[k]]. */
                #pragma omp simd reduction(+:s)
                for (long k = start; k < end; k++) {
                    s += val[k] * x[col[k]];
                }
                y[i] = s;
            }
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
