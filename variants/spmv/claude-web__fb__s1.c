/* k06_spmv: y = A*x for a sparse CSR matrix, repeated.
 * IRREGULAR: row lengths follow a heavy-tailed distribution, so a few
 * rows hold most of the work (like real web or social graphs). */
#include <math.h>
#include <omp.h>
#include "common.h"

/* === KERNEL BEGIN === */
/* Smallest row index i in [0, n] with rowptr[i] + i >= target.
 * rowptr[i] + i is strictly increasing, so binary search works. */
static long find_split(long n, const long *rowptr, long target) {
    long lo = 0, hi = n;
    while (lo < hi) {
        long mid = lo + (hi - lo) / 2;
        if (rowptr[mid] + mid >= target) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

/* Nonzeros per cache block: ~8K nnz = ~96 KB of val+col, which stays
 * resident in L2 across all repetitions of that block. */
#define BLOCK_NNZ 8192L

static void kernel(long n, int reps, const long *restrict rowptr,
                   const int *restrict col, const double *restrict val,
                   const double *restrict x, double *restrict y) {
    #pragma omp parallel
    {
        int T = omp_get_num_threads();
        int t = omp_get_thread_num();
        /* Work weight: nonzeros plus one unit per row (loop overhead). */
        long total = rowptr[n] + n;
        long lo = (t == 0) ? 0 : find_split(n, rowptr, (long)((__int128)total * t / T));
        long hi = (t == T - 1) ? n : find_split(n, rowptr, (long)((__int128)total * (t + 1) / T));

        /* Each thread owns rows [lo, hi). x is read-only and rows are
         * disjoint, so repetitions of different rows are independent:
         * process a cache-sized block of rows for all reps, then move on.
         * Every row is still computed reps times, in the same order of
         * additions, so y is bit-identical to the serial version. */
        long b = lo;
        while (b < hi) {
            long e = b + 1;                       /* at least one row */
            long start = rowptr[b];
            while (e < hi && rowptr[e + 1] - start <= BLOCK_NNZ) e++;

            for (int r = 0; r < reps; r++) {
                for (long i = b; i < e; i++) {
                    double s = 0.0;
                    long kend = rowptr[i + 1];
                    for (long k = rowptr[i]; k < kend; k++)
                        s += val[k] * x[col[k]];
                    y[i] = s;
                }
            }
            b = e;
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
