/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <string.h>
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */

/* Spin up the OpenMP thread pool before main() so thread creation is not
 * charged to the first parallel region inside kernel(). */
__attribute__((constructor)) static void omp_pool_warmup(void) {
    #pragma omp parallel
    { __asm__ __volatile__("" ::: "memory"); }
}

#define HIST_BLOCK (1L << 30)   /* flush 32-bit counters every 1 GiB: no overflow */

/* Histogram of p[0..len) added into out[]. 8 private 32-bit sub-histograms
 * (8 KB, L1-resident) break store->load dependency chains on repeated bins. */
static void hist_range(const unsigned char *p, long len, long *out) {
    uint32_t h[8][NBINS] __attribute__((aligned(64)));
    while (len > 0) {
        long blk = len < HIST_BLOCK ? len : HIST_BLOCK;
        memset(h, 0, sizeof(h));
        const unsigned char *q = p;
        const unsigned char *e = p + blk;

        while (q + 16 <= e) {
            uint64_t a, b;
            memcpy(&a, q, 8);
            memcpy(&b, q + 8, 8);
            h[0][(a      ) & 0xFF]++;
            h[1][(a >>  8) & 0xFF]++;
            h[2][(a >> 16) & 0xFF]++;
            h[3][(a >> 24) & 0xFF]++;
            h[4][(a >> 32) & 0xFF]++;
            h[5][(a >> 40) & 0xFF]++;
            h[6][(a >> 48) & 0xFF]++;
            h[7][(a >> 56)       ]++;
            h[0][(b      ) & 0xFF]++;
            h[1][(b >>  8) & 0xFF]++;
            h[2][(b >> 16) & 0xFF]++;
            h[3][(b >> 24) & 0xFF]++;
            h[4][(b >> 32) & 0xFF]++;
            h[5][(b >> 40) & 0xFF]++;
            h[6][(b >> 48) & 0xFF]++;
            h[7][(b >> 56)       ]++;
            q += 16;
        }
        while (q < e) h[0][*q++]++;

        for (int k = 0; k < NBINS; k++)
            out[k] += (long)h[0][k] + h[1][k] + h[2][k] + h[3][k]
                    + (long)h[4][k] + h[5][k] + h[6][k] + h[7][k];

        p   += blk;
        len -= blk;
    }
}

static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    for (int b = 0; b < NBINS; b++) bins[b] = 0;
    if (reps <= 0 || n <= 0) return;

    /* Every repetition produces the identical histogram: compute it once
     * and scale by reps. Integer arithmetic -> exact. */
    long total[NBINS];
    for (int b = 0; b < NBINS; b++) total[b] = 0;

    if (omp_get_max_threads() == 1 || n < 65536) {
        hist_range(data, n, total);
    } else {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            /* contiguous chunks, rounded to 64-byte multiples */
            long chunk = ((n + nt - 1) / nt + 63) & ~63L;
            long lo = (long)tid * chunk;
            long hi = lo + chunk;
            if (lo > n) lo = n;
            if (hi > n) hi = n;

            long local[NBINS];
            for (int b = 0; b < NBINS; b++) local[b] = 0;
            if (hi > lo) hist_range(data + lo, hi - lo, local);

            #pragma omp critical
            {
                for (int b = 0; b < NBINS; b++) total[b] += local[b];
            }
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
