/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <string.h>
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    const int max_threads = omp_get_max_threads();

    /* Private histogram per thread, padded to avoid false sharing. */
    const int stride = 64; /* bytes; one cache line */
    const int bins_per_thread = ((NBINS * (int)sizeof(long) + stride - 1) / stride) * stride / (int)sizeof(long);

    long *priv = (long *)xmalloc((size_t)max_threads * bins_per_thread * sizeof(long));
    memset(priv, 0, (size_t)max_threads * bins_per_thread * sizeof(long));

    for (int r = 0; r < reps; r++) {
        long i;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long *my = priv + (size_t)tid * bins_per_thread;
            #pragma omp for schedule(static)
            for (i = 0; i < n; i++) {
                my[data[i]]++;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < NBINS; b++) {
        long s = 0;
        for (int t = 0; t < max_threads; t++)
            s += priv[(size_t)t * bins_per_thread + b];
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
