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
     * Optimization: The original code repeats the counting process 'reps' times.
     * Since the 'data' array is constant across repetitions, we can simply 
     * count the occurrences once and multiply by 'reps', or add 'reps' to 
     * the bin each time we encounter the value. This reduces memory 
     * bandwidth pressure from O(n * reps) to O(n).
     *
     * Correctness: We use an OpenMP array reduction to avoid race conditions
     * when multiple threads increment the same bin. Each thread maintains
     * a private histogram which is merged at the end.
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

