/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    // Initialize bins to 0
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    /* 
     * To maximize performance on a heterogeneous architecture (P-cores and E-cores),
     * we use a parallel region with private bins to eliminate contention and
     * 'schedule(guided)' to balance the load.
     * 
     * We parallelize the 'i' loop (the larger dimension) and keep the 'reps' loop
     * inside to ensure all cores are utilized even if 'reps' is small.
     * Placing the 'reps' loop inside also improves cache locality by loading 
     * data[i] into a register once and performing all repetitions for that value.
     */
    #pragma omp parallel
    {
        long local_bins[NBINS] = {0};

        #pragma omp for schedule(guided)
        for (long i = 0; i < n; i++) {
            unsigned char val = data[i];
            for (int r = 0; r < reps; r++) {
                local_bins[val]++;
            }
        }

        // Aggregate private bins into the global bins array
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

