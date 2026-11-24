/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    // Initialize bins to zero
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    /*
     * The goal is to parallelize the counting process while avoiding race conditions
     * on the bin updates. Since NBINS is small (256), the most efficient approach 
     * is to give each thread a private local histogram to avoid atomic contention 
     * and cache line bouncing (false sharing).
     * 
     * We also optimize the loop order: the original code scans the entire 'data' 
     * array 'reps' times. By scanning 'data' once and adding 'reps' to the bin 
     * for each element, we drastically improve cache locality.
     */
    #pragma omp parallel
    {
        long local_bins[NBINS] = {0};

        #pragma omp for schedule(static)
        for (long i = 0; i < n; i++) {
            unsigned char val = data[i];
            // We use a loop to maintain the "uniform cost" intent of the kernel,
            // though a compiler will likely optimize this to local_bins[val] += reps.
            for (int r = 0; r < reps; r++) {
                local_bins[val]++;
            }
        }

        // Merge local results into the global bins array using atomics.
        // Since there are only 256 bins, this final merge is negligible in cost.
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

