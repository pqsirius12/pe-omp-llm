/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
/* Cache-tiled two-pass block scan.
 * The array is processed in tiles of T*B elements. Within a tile each thread
 * owns a contiguous sub-block of about B elements (fits in L2). For each tile:
 *   pass 1: each thread sums its sub-block (x streamed from DRAM once),
 *   barrier,
 *   pass 2: each thread computes its offset from the tile carry plus the sums
 *           of the threads before it, then scans its sub-block (x is still in
 *           cache) and writes out.
 * The sums buffer is double-buffered by tile parity, so one barrier per tile
 * is enough. Every thread computes the same running carry redundantly.
 * Integer arithmetic keeps the result identical to the serial version. */
#define PS_BLOCK 32768L   /* elements per thread per tile (256 KB of longs) */
#define PS_PAD   8        /* pad per-thread slots to one cache line */

static void kernel(long n, int reps, const long *x, long *out) {
    int maxt = omp_get_max_threads();
    if (maxt <= 1 || n < 4 * PS_BLOCK) {
        for (int r = 0; r < reps; r++) {
            long run = 0;
            for (long i = 0; i < n; i++) {
                run += x[i];
                out[i] = run;
            }
        }
        return;
    }

    long *sums = xmalloc((size_t)2 * maxt * PS_PAD * sizeof *sums);

    #pragma omp parallel
    {
        const int t = omp_get_thread_num();
        const int T = omp_get_num_threads();
        const long tile = PS_BLOCK * (long)T;

        for (int r = 0; r < reps; r++) {
            long carry = 0;
            long k = 0;
            for (long base = 0; base < n; base += tile, k++) {
                long len = (n - base < tile) ? (n - base) : tile;
                long lo = base + len * t / T;
                long hi = base + len * (t + 1) / T;
                long *buf = sums + (size_t)(k & 1) * maxt * PS_PAD;

                /* pass 1: local sum */
                long s = 0;
                for (long i = lo; i < hi; i++) s += x[i];
                buf[t * PS_PAD] = s;

                #pragma omp barrier

                /* offsets */
                long off = carry, tot = carry;
                for (int j = 0; j < T; j++) {
                    long v = buf[j * PS_PAD];
                    if (j < t) off += v;
                    tot += v;
                }

                /* pass 2: scan with offset (data is cache-hot) */
                long run = off;
                for (long i = lo; i < hi; i++) {
                    run += x[i];
                    out[i] = run;
                }
                carry = tot;
                /* no second barrier: next tile writes the other buffer, and
                 * reuse of this buffer (tile k+2) happens only after every
                 * thread has passed the barrier of tile k+1. */
            }
            #pragma omp barrier  /* keep reps cleanly separated */
        }
    }

    free(sums);
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
