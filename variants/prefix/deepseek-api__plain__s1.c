/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"
#include <string.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *x, long *out) {
    if (n <= 0) return;
    const int max_threads = 64;
    long nblocks = 0;
    long *block_sums = NULL;

    for (int r = 0; r < reps; r++) {
        long nchunks = 0, chunk = 0;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nth = omp_get_num_threads();

            #pragma omp single
            {
                chunk = (n + nth - 1) / nth;
                if (chunk < 4096) chunk = 4096;
                nchunks = (n + chunk - 1) / chunk;
                (void)max_threads;
                if (block_sums == NULL) block_sums = xmalloc((size_t)nchunks * sizeof(long));
            }

            #pragma omp for schedule(static)
            for (long c = 0; c < nchunks; c++) {
                long lo = c * chunk;
                long hi = lo + chunk;
                if (hi > n) hi = n;
                long run = 0;
                for (long i = lo; i < hi; i++) {
                    run += x[i];
                    out[i] = run;
                }
                block_sums[c] = run;
            }

            #pragma omp barrier

            /* sequential-ish prefix over block sums: do it redundantly per thread
             * since nchunks is small (nth <= 10, chunk >= 4096 -> nchunks ~ n/4096)
             * Actually nchunks can be large for big n; do it once by thread 0. */
            #pragma omp single
            {
                long acc = 0;
                for (long c = 0; c < nchunks; c++) {
                    long v = block_sums[c];
                    block_sums[c] = acc;
                    acc += v;
                }
            }

            #pragma omp barrier

            (void)tid;

            #pragma omp for schedule(static)
            for (long c = 0; c < nchunks; c++) {
                long lo = c * chunk;
                long hi = lo + chunk;
                if (hi > n) hi = n;
                long base = block_sums[c];
                if (base != 0) {
                    for (long i = lo; i < hi; i++) out[i] += base;
                }
            }
        }
    }
    free(block_sums);
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
