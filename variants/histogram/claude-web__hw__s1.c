/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <string.h>
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    for (int b = 0; b < NBINS; b++) bins[b] = 0;
    if (reps <= 0 || n <= 0) return;

    /* Every repetition counts the same data, so the serial result equals
     * reps * (histogram of one pass). Compute one pass in parallel. */
    long acc[NBINS];
    for (int b = 0; b < NBINS; b++) acc[b] = 0;

    const long CH = 1L << 20;               /* 1 MiB chunks: dynamic balance P/E cores */
    const long nch = (n + CH - 1) / CH;

    #pragma omp parallel reduction(+:acc[:NBINS])
    {
        uint32_t h[4][NBINS];

        #pragma omp for schedule(dynamic, 1) nowait
        for (long c = 0; c < nch; c++) {
            memset(h, 0, sizeof h);
            long s = c * CH;
            long e = s + CH;
            if (e > n) e = n;
            const unsigned char *p = data + s;
            long len = e - s;
            long i = 0;
            /* 4 interleaved sub-histograms hide store->load dependencies
             * on repeated bins; 8-byte loads cut load count. */
            for (; i + 8 <= len; i += 8) {
                uint64_t w;
                memcpy(&w, p + i, 8);
                h[0][(w      ) & 0xFF]++;
                h[1][(w >>  8) & 0xFF]++;
                h[2][(w >> 16) & 0xFF]++;
                h[3][(w >> 24) & 0xFF]++;
                h[0][(w >> 32) & 0xFF]++;
                h[1][(w >> 40) & 0xFF]++;
                h[2][(w >> 48) & 0xFF]++;
                h[3][(w >> 56)       ]++;
            }
            for (; i < len; i++) h[0][p[i]]++;

            /* flush 32-bit chunk counts into 64-bit per-thread totals */
            for (int b = 0; b < NBINS; b++)
                acc[b] += (long)h[0][b] + h[1][b] + h[2][b] + h[3][b];
        }
    }

    for (int b = 0; b < NBINS; b++) bins[b] = acc[b] * (long)reps;
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 200000000);
    int reps = (int)arg_long(argc, argv, 2, 20);
    unsigned char *data = xmalloc(n);
    long bins[NBINS];
    rng_seed(8);
    for (long i = 0; i < n; i++) data[i] = (unsigned char)(rng_next() >> 56);

    double t0 = now_sec();
    kernel(n, reps, data, bins);
    double t1 = now_sec();

    double sum = 0.0;   /* weighted so a value landing in the wrong bin is caught */
    for (int b = 0; b < NBINS; b++) sum += (double)bins[b] * (b + 1);
    report("histogram", t1 - t0, sum);
    free(data);
    return 0;
}
