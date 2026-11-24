/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    int nth = omp_get_max_threads();
    /* One private 256-bin histogram per thread; all fit in L1. */
    long *priv = (long *)xmalloc((size_t)nth * NBINS * sizeof(long));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long *my = priv + (size_t)tid * NBINS;

        for (int r = 0; r < reps; r++) {
            for (int b = 0; b < NBINS; b++) my[b] = 0;

            #pragma omp for nowait schedule(static)
            for (long i = 0; i < n; i++)
                my[data[i]]++;
        }
    }

    /* Reduce the per-thread private histograms into the shared one. */
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < NBINS; b++) {
        long s = 0;
        for (int t = 0; t < nth; t++)
            s += priv[(size_t)t * NBINS + b];
        bins[b] = s;
    }

    free(priv);
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
