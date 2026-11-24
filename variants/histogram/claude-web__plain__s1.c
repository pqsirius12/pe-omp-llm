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

    /* Every repetition produces the identical histogram: compute it once
     * (in parallel) and scale by reps. Integer arithmetic -> exact. */
    long total[NBINS];
    for (int b = 0; b < NBINS; b++) total[b] = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
        long chunk = (n + nt - 1) / nt;
        long lo = (long)tid * chunk;
        long hi = lo + chunk;
        if (lo > n) lo = n;
        if (hi > n) hi = n;

        /* 4 sub-histograms break store-to-load dependency chains */
        long h[4][NBINS];
        memset(h, 0, sizeof(h));

        long i = lo;
        for (; i + 8 <= hi; i += 8) {
            uint64_t w;
            memcpy(&w, data + i, 8);
            h[0][(w      ) & 0xFF]++;
            h[1][(w >>  8) & 0xFF]++;
            h[2][(w >> 16) & 0xFF]++;
            h[3][(w >> 24) & 0xFF]++;
            h[0][(w >> 32) & 0xFF]++;
            h[1][(w >> 40) & 0xFF]++;
            h[2][(w >> 48) & 0xFF]++;
            h[3][(w >> 56)       ]++;
        }
        for (; i < hi; i++) h[0][data[i]]++;

        long local[NBINS];
        for (int b = 0; b < NBINS; b++)
            local[b] = h[0][b] + h[1][b] + h[2][b] + h[3][b];

        #pragma omp critical
        {
            for (int b = 0; b < NBINS; b++) total[b] += local[b];
        }
    }

    for (int b = 0; b < NBINS; b++) bins[b] = total[b] * (long)reps;
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
