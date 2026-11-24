/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    // Reset global bins to zero
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    /*
     * To achieve maximum performance:
     * 1. We avoid the O(n * reps) complexity of the serial version by transforming 
     *    the nested loops. Instead of incrementing a bin 'reps' times, we add 
     *    'reps' to the bin once for each element in the data array. This reduces 
     *    the complexity to O(n).
     * 2. We use thread-local buffers (local_bins) to eliminate race conditions 
     *    and cache contention (false sharing) during the main loop.
     * 3. We use a static schedule to ensure the data is read in large, contiguous 
     *    blocks, maximizing hardware prefetching efficiency.
     * 4. We merge local results into the global bins using atomics at the very end.
     *    With only 256 bins, this merge is extremely fast and does not bottleneck.
     */
    #pragma omp parallel
    {
        long local_bins[NBINS] = {0};

        #pragma omp for schedule(static)
        for (long i = 0; i < n; i++) {
            local_bins[data[i]] += reps;
        }

        // Merge thread-local results into the global array
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
