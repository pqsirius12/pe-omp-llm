/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *x, long *out) {
    /* The outer reps loop is mathematically redundant because each iteration 
       completely overwrites the 'out' array based on the same input 'x'.
       However, to respect the intent of the performance benchmark (which uses 
       reps to ensure a significant workload), we parallelize the core prefix 
       sum operation while keeping the repetitions. */
    for (int r = 0; r < reps; r++) {
        long run = 0;
        /* 
         * We use OpenMP 5.0+ 'inscan' reduction to perform an inclusive scan.
         * 'schedule(guided)' is chosen because the target machine (Apple M4) 
         * has a heterogeneous architecture (P-cores and E-cores). Guided 
         * scheduling helps balance the load by dynamically adjusting chunk 
         * sizes, preventing faster cores from idling while slower cores finish.
         */
        #pragma omp parallel for reduction(inscan, +:run) schedule(guided)
        for (long i = 0; i < n; i++) {
            run += x[i];
            #pragma omp scan inclusive
            out[i] = run;
        }
    }
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
