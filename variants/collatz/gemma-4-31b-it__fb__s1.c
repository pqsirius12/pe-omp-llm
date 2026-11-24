/* k12_collatz: number of Collatz steps for every start value below n.
 * IRREGULAR and UNPREDICTABLE: neighbouring start values can need
 * wildly different step counts, with no pattern a static split can use. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int *steps) {
    /* 
     * The Collatz problem has a highly irregular workload. To optimize:
     * 1. We use 'schedule(guided)' to balance the load while minimizing 
     *    the synchronization overhead.
     * 2. We use '__builtin_ctzll' (count trailing zeros) to process all 
     *    consecutive divisions by 2 in a single step, significantly 
     *    reducing the number of iterations in the inner loop.
     */
    #pragma omp parallel for schedule(guided)
    for (long i = 1; i < n; i++) {
        unsigned long v = (unsigned long)i;
        int s = 0;
        while (v > 1) {
            if (v & 1) {
                v = 3 * v + 1;
                s++;
            }
            /* v is now guaranteed to be even (either from the start or from 3v+1) */
            int tz = __builtin_ctzll(v);
            v >>= tz;
            s += tz;
        }
        steps[i] = s;
    }
}
/* === KERNEL END === */

int main(int argc, char **argv) {
    long n = arg_long(argc, argv, 1, 20000000);
    int *steps = xmalloc(n * sizeof *steps);
    steps[0] = 0;

    double t0 = now_sec();
    kernel(n, steps);
    double t1 = now_sec();

    double sum = 0.0;
    for (long i = 1; i < n; i++) sum += steps[i];
    report("collatz", t1 - t0, sum);
    free(steps);
    return 0;
}

