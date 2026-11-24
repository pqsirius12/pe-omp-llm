/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"
#include <string.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *x, long *out) {
    if (n <= 0) return;

    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    long *block_sum = (long *)malloc((size_t)nthreads * sizeof(long));

    for (int r = 0; r < reps; r++) {
        /* Pass 1: each thread computes the sum of its contiguous chunk,
         * plus a local inclusive scan of that chunk into out[]. */
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nth = omp_get_num_threads();
            long start = (long)((double)n * tid / nth);
            long end   = (long)((double)n * (tid + 1) / nth);
            long run = 0;
            for (long i = start; i < end; i++) {
                run += x[i];
                out[i] = run;
            }
            block_sum[tid] = run;
        }

        /* Pass 2: exclusive prefix over block sums (serial; nthreads small). */
        long acc = 0;
        for (int t = 0; t < nthreads; t++) {
            long s = block_sum[t];
            block_sum[t] = acc;
            acc += s;
        }

        /* Pass 3: add each block's exclusive offset to its elements. */
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nth = omp_get_num_threads();
            long start = (long)((double)n * tid / nth);
            long end   = (long)((double)n * (tid + 1) / nth);
            long off = block_sum[tid];
            if (off != 0) {
                for (long i = start; i < end; i++) out[i] += off;
            }
        }
    }

    free(block_sum);
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 100000000);
    int reps = (int)arg_long(argc, argv, 2, 5);
    long *x = xmalloc(n * sizeof *x), *out = xmalloc(n * sizeof *out);
    rng_seed(10);
    for (long i = 0; i < n; i++) x[i] = (long)(rng_next() % 100);

    double t0 = now_sec();
    kernel(n, reps, x, out);
    double t1 = now_sec();

    double sum = 0.0;   /* sample every 1000th entry plus the last one */
    for (long i = 0; i < n; i += 1000) sum += (double)out[i];
    sum += (double)out[n - 1];
    report("prefix", t1 - t0, sum);
    free(x); free(out);
    return 0;
}
