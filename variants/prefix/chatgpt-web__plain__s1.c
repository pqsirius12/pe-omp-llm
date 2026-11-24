#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *x, long *out) {
    if (n <= 0) return;

    int nt = omp_get_max_threads();
    long block = (n + nt - 1) / nt;

    long *sums = malloc((size_t)nt * sizeof(long));
    if (!sums) {
        /* Fallback to serial scan if allocation fails */
        for (int r = 0; r < reps; r++) {
            long run = 0;
            for (long i = 0; i < n; i++) {
                run += x[i];
                out[i] = run;
            }
        }
        return;
    }

    for (int r = 0; r < reps; r++) {
        /* Phase 1: compute local prefix sums in parallel */
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long start = (long)tid * block;
            long end = start + block;
            if (end > n) end = n;

            long local = 0;
            for (long i = start; i < end; i++) {
                local += x[i];
                out[i] = local;
            }
            sums[tid] = local;

            #pragma omp barrier

            /* Phase 2: compute each block's offset */
            long offset = 0;
            for (int t = 0; t < tid; t++)
                offset += sums[t];

            /* Phase 3: add offset to local prefix sums */
            for (long i = start; i < end; i++)
                out[i] += offset;
        }
    }

    free(sums);
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 100000000);
    int reps = (int)arg_long(argc, argv, 2, 5);
    long *x = xmalloc(n * sizeof *x), *out = xmalloc(n * sizeof *out);
    rng_seed(10);
    for (long i = 0; i < n; i++) x[i] = (long)(rng_next() % 100);

    double t0 = now_sec();
    kernel(n, reps, x, out);
    double t1 = now_sec();

    double sum = 0.0;   /* sample every 1000th entry plus the last one */
    for (long i = 0; i < n; i += 1000) sum += (double)out[i];
    sum += (double)out[n - 1];
    report("prefix", t1 - t0, sum);
    free(x); free(out);
    return 0;
}
