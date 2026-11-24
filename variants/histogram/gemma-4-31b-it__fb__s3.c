/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    // Initialize the global bins array
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    /* 
     * Optimization 1: Algebraic reduction.
     * The original serial code repeats the counting process 'reps' times.
     * Since the 'data' array is read-only, we can count each element once 
     * and multiply its contribution by 'reps'. This transforms the 
     * time complexity from O(n * reps) to O(n), providing a massive speedup.
     *
     * Optimization 2: OpenMP Array Reduction.
     * We use the OpenMP 4.5+ reduction clause for arrays: reduction(+:bins[0:NBINS]).
     * This instructs the compiler to create thread-local copies of the bins array.
     * Each thread updates its local copy (which fits in L1 cache), eliminating 
     * cache contention and race conditions. The local arrays are then summed 
     * together at the end of the parallel region.
     *
     * Optimization 3: SIMD.
     * By adding the 'simd' hint, we encourage the compiler to use vector 
     * instructions (such as AVX-512 conflict detection) if available to 
     * further accelerate the histogram process.
     */
    #pragma omp parallel for simd reduction(+:bins[0:NBINS])
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

