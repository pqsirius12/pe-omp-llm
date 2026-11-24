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
     * The original serial code executes:
     * for (int r = 0; r < reps; r++)
     *     for (long i = 0; i < n; i++)
     *         bins[data[i]]++;
     * 
     * This is logically equivalent to iterating through the 'data' array once 
     * and adding 'reps' to the corresponding bin for each element.
     * This optimization reduces memory bandwidth pressure significantly by 
     * reducing the number of times the 'data' array is read from reps to 1.
     * 
     * We use an OpenMP array reduction to handle the race condition where 
     * multiple threads attempt to update the same bin. 
     * reduction(+:bins[0:NBINS]) creates a private copy of the bins array 
     * for each thread and combines them at the end.
     */
    #pragma omp parallel for reduction(+:bins[0:NBINS])
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
