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
     * The workload is highly irregular due to the heavy-tailed row length distribution.
     * We implement a manual balanced static schedule. Since the matrix structure 
     * (rowptr) is constant across repetitions, we can calculate the optimal 
     * division of rows once and reuse it for all 'reps' repetitions.
     * This eliminates the overhead of dynamic scheduling (like 'guided') 
     * and ensures a near-perfect load balance.
     */
    #pragma omp parallel
    {
        int num_t = omp_get_num_threads();
        int tid = omp_get_thread_num();
        long *bounds = NULL;

        /* The master thread calculates the row boundaries to balance the 
         * number of non-zero elements (NNZ) handled by each thread. */
        #pragma omp single
        {
            bounds = (long *)xmalloc((num_t + 1) * sizeof(long));
            long total_nnz = rowptr[n];
            long target = total_nnz / num_t;
            long row = 0;
            for (int t = 0; t < num_t; t++) {
                bounds[t] = row;
                long count = 0;
                while (row < n && count < target) {
                    count += (rowptr[row + 1] - rowptr[row]);
                    row++;
                }
            }
            bounds[num_t] = n;
        }

        /* Each thread processes its assigned range of rows for every repetition. */
        long start_row = bounds[tid];
        long end_row = bounds[tid + 1];

        for (int r = 0; r < reps; r++) {
            for (long i = start_row; i < end_row; i++) {
                double s = 0.0;
                const long start = rowptr[i];
                const long end = rowptr[i + 1];
                
                /* The inner loop is a sparse dot product. We use #pragma omp simd 
                 * to encourage the compiler to use gather instructions (AVX/AVX-512). */
                #pragma omp simd reduction(+:s)
                for (long k = start; k < end; k++) {
                    s += val[k] * x[col[k]];
                }
                y[i] = s;
            }
        }

        /* Free the boundaries array after all threads have finished the work. */
        #pragma omp single
        {
            free(bounds);
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

