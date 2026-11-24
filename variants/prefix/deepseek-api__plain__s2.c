/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *x, long *out) {
    /* Two-pass block scan. Compute per-block sums, scan them, then
     * scan each block with the offset from the prefix of block sums. */
    #pragma omp parallel
    {
        long nthreads = (long)omp_get_num_threads();
        long tid = (long)omp_get_thread_num();
        long *block_sums = NULL;
        long *block_prefix = NULL;
        #pragma omp single
        {
            block_sums = (long *)xmalloc((size_t)nthreads * sizeof(long));
            block_prefix = (long *)xmalloc((size_t)nthreads * sizeof(long));
        }

        for (int r = 0; r < reps; r++) {
            /* Each thread handles a contiguous chunk [lo, hi). */
            long chunk = (n + nthreads - 1) / nthreads;
            long lo = tid * chunk;
            long hi = lo + chunk;
            if (hi > n) hi = n;
            if (lo > n) lo = n;

            /* Pass 1: sum of the local chunk. */
            long local_sum = 0;
            for (long i = lo; i < hi; i++) local_sum += x[i];
            block_sums[tid] = local_sum;

            #pragma omp barrier

            /* Pass 2: compute prefix of block sums (sequentially in thread 0,
             * writing to block_prefix). Use a simple sequential scan since
             * nthreads is small. */
            #pragma omp single
            {
                long run = 0;
                for (long t = 0; t < nthreads; t++) {
                    block_prefix[t] = run;
                    run += block_sums[t];
                }
            }

            #pragma omp barrier

            /* Pass 3: local scan with offset. */
            long offset = block_prefix[tid];
            long run = offset;
            for (long i = lo; i < hi; i++) {
                run += x[i];
                out[i] = run;
            }

            #pragma omp barrier
        }

        #pragma omp single
        {
            free(block_sums);
            free(block_prefix);
        }
    }
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
