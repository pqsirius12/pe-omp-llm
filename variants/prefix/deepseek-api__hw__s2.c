/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
#include <stdlib.h>

static void kernel(long n, int reps, const long *x, long *out) {
    /* Cache block sums per (chunk, rep) so the second pass can prefix-scan
     * them per rep, then apply offsets to the data. */
    int nt = omp_get_max_threads();
    long bs = 4096;
    long nchunks = (n + bs - 1) / bs;
    long *chunk_sums = (long *)xmalloc((size_t)nchunks * sizeof(long) * reps);

    for (int r = 0; r < reps; r++) {
        long *sums_r = chunk_sums + (size_t)r * nchunks;

        #pragma omp parallel for schedule(static)
        for (long c = 0; c < nchunks; c++) {
            long lo = c * bs;
            long hi = lo + bs < n ? lo + bs : n;
            long s = 0;
            for (long i = lo; i < hi; i++) s += x[i];
            sums_r[c] = s;
        }

        /* Sequential prefix over chunks (nchunks is small). */
        long prefix = 0;
        for (long c = 0; c < nchunks; c++) {
            long v = sums_r[c];
            sums_r[c] = prefix;
            prefix += v;
        }

        #pragma omp parallel for schedule(static)
        for (long c = 0; c < nchunks; c++) {
            long lo = c * bs;
            long hi = lo + bs < n ? lo + bs : n;
            long run = sums_r[c];
            for (long i = lo; i < hi; i++) {
                run += x[i];
                out[i] = run;
            }
        }
    }
    free(chunk_sums);
    (void)nt;
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
