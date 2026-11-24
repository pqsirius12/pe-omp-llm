/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    // Initialize bins to 0. This is a small operation and doesn't need parallelization.
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    /* 
     * Optimization:
     * The original code performs:
     * for (int r = 0; r < reps; r++)
     *     for (long i = 0; i < n; i++)
     *         bins[data[i]]++;
     * 
     * This is mathematically equivalent to iterating through 'data' once and 
     * adding 'reps' to the bin for each element:
     * for (long i = 0; i < n; i++)
     *     bins[data[i]] += reps;
     * 
     * This transformation reduces memory access and total operations by a factor of 'reps'.
     * 
     * Parallelization Strategy:
     * 1. Use 'reduction(+:bins[0:NBINS])' to create a thread-local copy of the bins array.
     *    This prevents race conditions and eliminates the need for expensive atomic operations.
     * 2. Use 'schedule(guided)' to balance the load between the 4 P-cores and 6 E-cores
     *    on the Apple M4. Guided scheduling starts with large chunks and reduces them over time,
     *    allowing faster cores to process more work.
     */
    #pragma omp parallel for reduction(+:bins[0:NBINS]) schedule(guided)
    for (long i = 0; i < n; i++) {
        bins[data[i]] += reps;
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
