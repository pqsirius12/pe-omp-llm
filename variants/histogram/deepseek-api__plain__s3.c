/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <string.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    for (int b = 0; b < NBINS; b++) bins[b] = 0;
    long total = (long)reps * n;
    if (total <= 0) return;

    #pragma omp parallel
    {
        long local[NBINS];
        memset(local, 0, sizeof(local));

        /* Merge all reps into a single flat loop over the (reps x n) iteration
         * space so load is balanced across threads. */
        #pragma omp for schedule(static)
        for (long k = 0; k < total; k++) {
            local[data[k % n]]++;
        }

        #pragma omp critical
        {
            for (int b = 0; b < NBINS; b++) bins[b] += local[b];
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
