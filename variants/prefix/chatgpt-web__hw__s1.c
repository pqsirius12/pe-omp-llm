/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *x, long *out) {
    if (n <= 0 || reps <= 0) return;

    int nt = omp_get_max_threads();

    /* Block boundaries and partial sums */
    long *sums = xmalloc((size_t)nt * sizeof(long));
    long *offsets = xmalloc((size_t)nt * sizeof(long));

    for (int r = 0; r < reps; r++) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int team = omp_get_num_threads();

            long chunk = (n + team - 1) / team;
            long start = (long)tid * chunk;
            long end = start + chunk;
            if (end > n) end = n;

            long local = 0;

            /* Pass 1: independent prefix sums within each block */
            if (start < n) {
                for (long i = start; i < end; i++) {
                    local += x[i];
                    out[i] = local;
                }
            }
            sums[tid] = local;

            #pragma omp barrier

            /* Pass 2: calculate this block's offset */
            long offset = 0;
            for (int t = 0; t < tid; t++)
                offset += sums[t];
            offsets[tid] = offset;

            #pragma omp barrier

            /* Pass 3: add the block offset to its local prefix */
            if (offset != 0 && start < n) {
                #pragma omp simd
                for (long i = start; i < end; i++)
                    out[i] += offset;
            }
        }
    }

    free(sums);
    free(offsets);
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
