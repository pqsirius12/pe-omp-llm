/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    // Initialize global bins to 0
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    /* 
     * Strategy:
     * 1. Use a private local bin array for each thread to avoid cache line bouncing 
     *    and contention (False Sharing). 256 * 8 bytes = 2KB, which fits in L1 cache.
     * 2. Algorithmic Optimization: The original loop is bins[data[i]]++ repeated 'reps' times.
     *    This is mathematically equivalent to bins[data[i]] += reps.
     *    This reduces the complexity from O(n * reps) to O(n).
     * 3. Load Balancing: Use 'schedule(guided)' to handle the heterogeneity 
     *    of Apple M4's P-cores and E-cores.
     */
    #pragma omp parallel
    {
        long local_bins[NBINS] = {0};

        #pragma omp for schedule(guided)
        for (long i = 0; i < n; i++) {
            local_bins[data[i]] += reps;
        }

        // Combine local results into the global bins array
        for (int b = 0; b < NBINS; b++) {
            #pragma omp atomic
            bins[b] += local_bins[b];
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

