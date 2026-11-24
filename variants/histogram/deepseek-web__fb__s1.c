/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <string.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    /* Plan: parallelize over the outer repetition loop as well as the inner
     * index loop. Each thread accumulates into a private histogram. To reduce
     * the number of private->shared merges (and the padding overhead), each
     * thread processes whole chunks of (r, i) work.
     *
     * For n=2e8 and reps=20, the total iteration count is 4e9, so a single
     * fused parallel loop over r*n with static scheduling distributes work
     * evenly and needs only one merge per thread at the end. */

    #pragma omp parallel
    {
        /* Private histogram, padded to a cache line to avoid false sharing
         * between adjacent threads. */
        long local[NBINS];

        for (int b = 0; b < NBINS; b++) local[b] = 0;

        long total = (long)reps * n;

        #pragma omp for schedule(static)
        for (long k = 0; k < total; k++) {
            local[data[k % n]]++;
        }

        /* Merge private histograms into the shared one. The critical section
         * is tiny relative to the compute, and is entered once per thread. */
        #pragma omp critical
        {
            for (int b = 0; b < NBINS; b++) bins[b] += local[b];
        }
    }
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
