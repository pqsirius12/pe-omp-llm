/* k10_prefix: inclusive prefix sum, out[i] = x[0] + ... + x[i].
 * LOOP-CARRIED DEPENDENCY: each step needs the previous one. A plain
 * "parallel for" gives wrong answers; a correct version needs a two-pass
 * block scan (or OpenMP 5's scan directive). Integer data keeps it exact. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int reps, const long *x, long *out) {
    /* 
     * We use the OpenMP 5.0 'inscan' reduction to implement the prefix sum.
     * This directive tells the compiler to perform an inclusive scan.
     *
     * Performance considerations for Apple M4:
     * 1. Heterogeneous Cores: The M4 has Performance (P) and Efficiency (E) cores.
     *    Using 'schedule(guided)' allows the faster P-cores to take on more work,
     *    preventing the slower E-cores from becoming a bottleneck.
     * 2. Memory Bandwidth: Prefix sum is memory-bound. 'guided' scheduling maintains 
     *    good cache locality while balancing the load.
     * 3. Repetitions: Since 'reps' is small (default 5), the overhead of 
     *    repeatedly creating the parallel region is negligible compared to the 
     *    computation on 10^8 elements.
     */
    for (int r = 0; r < reps; r++) {
        long run = 0;
        #pragma omp parallel for schedule(guided) reduction(inscan, +:run)
        for (long i = 0; i < n; i++) {
            run += x[i];
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

