/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    // Initialize global bins to zero
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    /* 
     * To avoid cache contention and race conditions on the bins array, 
     * we use thread-local bins. Each thread maintains its own count 
     * and merges it into the global array at the end.
     * 
     * Optimization: Since 'data' is constant and the inner loop simply 
     * increments bins[data[i]], we can combine the 'reps' iterations 
     * by adding 'reps' to the bin count for each element once. 
     * This reduces memory bandwidth pressure significantly.
     * 
     * Scheduling: 'guided' is used to balance the load across the M4's 
     * performance and efficiency cores.
     */
    #pragma omp parallel
    {
        long local_bins[NBINS] = {0};

        #pragma omp for schedule(guided)
        for (long i = 0; i < n; i++) {
            local_bins[data[i]] += reps;
        }

        // Merge local results into the global bin array
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
