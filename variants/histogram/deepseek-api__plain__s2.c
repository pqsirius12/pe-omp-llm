/* k08_histogram: count values into 256 bins.
 * UNIFORM cost, but a CORRECTNESS TRAP: many iterations write the same bin,
 * so a naive "parallel for" races. Needs atomics, a reduction, or private bins. */
#include "common.h"
#include <omp.h>

#define NBINS 256

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const unsigned char *data, long *bins) {
    for (int b = 0; b < NBINS; b++) bins[b] = 0;

    int nthreads = omp_get_max_threads();
    long *local = (long *)xmalloc((size_t)nthreads * NBINS * sizeof(long));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long *my = local + (size_t)tid * NBINS;
        for (int b = 0; b < NBINS; b++) my[b] = 0;

        for (int r = 0; r < reps; r++) {
            #pragma omp for schedule(static)
            for (long i = 0; i < n; i++)
                my[data[i]]++;
        }
    }

    for (int t = 0; t < nthreads; t++) {
        long *my = local + (size_t)t * NBINS;
        for (int b = 0; b < NBINS; b++) bins[b] += my[b];
    }
    free(local);
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
