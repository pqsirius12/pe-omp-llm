/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    #pragma omp parallel
    {
        /* Thread-private histograms eliminate contention on shared bins. */
        long local[NBINS] = {0};

        #pragma omp for schedule(static)
        for (long i = 0; i < n; i++) {
            local[data[i]]++;
        }

        /* Each thread processes its private histogram for every repetition.
           The first pass already counted once; remaining passes add locally. */
        for (int r = 1; r < reps; r++) {
            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                local[data[i]]++;
            }
        }

        /* Merge private histograms into the global result. */
        #pragma omp critical
        {
            for (int b = 0; b < NBINS; b++) {
                bins[b] += local[b];
            }
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
