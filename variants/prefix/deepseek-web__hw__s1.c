/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"
#include <omp.h>
#include <string.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *x, long *out) {
    if (n <= 0) return;

    int nt = omp_get_max_threads();
    if (nt < 1) nt = 1;

    /* Block scan: split [0,n) into nt contiguous chunks, each thread does
     * a serial inclusive scan of its chunk, records its total, then a
     * serial (cheap) scan of chunk totals gives the offsets, and finally
     * each thread adds its offset to its chunk. */
    long *chunk_tot = (long *)xmalloc((size_t)nt * sizeof(long));
    long *offsets   = (long *)xmalloc((size_t)nt * sizeof(long));

    for (int r = 0; r < reps; r++) {
        #pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            long lo = (long)((long long)n * tid / nthreads);
            long hi = (long)((long long)n * (tid + 1) / nthreads);

            /* Pass 1: local inclusive scan of this chunk. */
            long run = 0;
            for (long i = lo; i < hi; i++) {
                run += x[i];
                out[i] = run;
            }
            chunk_tot[tid] = run;

            #pragma omp barrier

            /* Pass 2 (single thread): exclusive prefix of chunk totals. */
            #pragma omp single
            {
                long acc = 0;
                for (int t = 0; t < nthreads; t++) {
                    offsets[t] = acc;
                    acc += chunk_tot[t];
                }
            }

            #pragma omp barrier

            /* Pass 3: add offset to this chunk. */
            long off = offsets[tid];
            if (off != 0) {
                for (long i = lo; i < hi; i++) out[i] += off;
            }
        }
    }

    free(chunk_tot);
    free(offsets);
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
