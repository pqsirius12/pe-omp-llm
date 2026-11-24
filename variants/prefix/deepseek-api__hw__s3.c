/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *x, long *out) {
    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();

        /* Carve the array into ~nparts contiguous chunks, one per thread.
         * Use a larger chunk count than threads for better load balance
         * on the heterogeneous M4 (4 P + 6 E cores). We use a fixed
         * number of chunks = 4*nt, each thread processes multiple chunks
         * in a strided fashion so all threads get a mix of work. */
        int nchunks = nt * 8;
        if ((long)nchunks > n) nchunks = (int)n;
        long *chunk_off = NULL;
        long *chunk_sum = NULL;

        /* Allocate per-thread scratch only via shared arrays allocated once
         * by thread 0 in a single region. We allocate on first rep. */
        /* Use static-like storage: allocate a small shared buffer sized nchunks. */
        /* To avoid re-alloc each rep, allocate outside the rep loop below. */

        /* We'll do the real work in a separate structured region per rep
         * below; this parallel region is entered reps times via the rep loop
         * being outside. Restructure: */

        (void)chunk_off; (void)chunk_sum; (void)tid;
    }

    /* We restructure: allocate scan scratch once. */
    int nt = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nt = omp_get_num_threads();
    }
    int nchunks = nt * 8;
    if ((long)nchunks > n) nchunks = (int)n;
    if (nchunks < 1) nchunks = 1;

    long *chunk_start = (long *)xmalloc(sizeof(long) * (size_t)(nchunks + 1));
    long *chunk_prefix = (long *)xmalloc(sizeof(long) * (size_t)nchunks);

    for (int r = 0; r < reps; r++) {
        /* Compute chunk boundaries: chunk c covers [chunk_start[c], chunk_start[c+1]) */
        for (int c = 0; c <= nchunks; c++) {
            chunk_start[c] = (long)((long long)c * (long long)n / (long long)nchunks);
        }

        /* Pass 1: each chunk does a local scan into out[], and stores its total. */
        #pragma omp parallel for schedule(dynamic, 1)
        for (int c = 0; c < nchunks; c++) {
            long s = chunk_start[c];
            long e = chunk_start[c + 1];
            long run = 0;
            for (long i = s; i < e; i++) {
                run += x[i];
                out[i] = run;
            }
            chunk_prefix[c] = run;
        }

        /* Pass 2: exclusive prefix over chunk totals (serial, small: nchunks). */
        long acc = 0;
        for (int c = 0; c < nchunks; c++) {
            long t = chunk_prefix[c];
            chunk_prefix[c] = acc;
            acc += t;
        }

        /* Pass 3: add chunk offset to each chunk's output. */
        #pragma omp parallel for schedule(dynamic, 1)
        for (int c = 0; c < nchunks; c++) {
            long off = chunk_prefix[c];
            if (off == 0) continue;
            long s = chunk_start[c];
            long e = chunk_start[c + 1];
            for (long i = s; i < e; i++) {
                out[i] += off;
            }
        }
    }

    free(chunk_start);
    free(chunk_prefix);
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
