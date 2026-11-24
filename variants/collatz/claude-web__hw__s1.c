/* k12_collatz: number of Collatz steps for every start value below n.
 * IRREGULAR and UNPREDICTABLE: neighbouring start values can need
 * wildly different step counts, with no pattern a static split can use. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int *steps) {
    /* Base block: computed directly (small, serial). */
    long base = n < 4096 ? n : 4096;
    for (long i = 1; i < base; i++) {
        unsigned long v = (unsigned long)i;
        int s = 0;
        while (v != 1) {
            if (v & 1) {
                v = (3 * v + 1) >> 1;   /* odd step + guaranteed halving */
                s += 2;
            } else {
                int t = __builtin_ctzll(v);
                v >>= t;
                s += t;
            }
        }
        steps[i] = s;
    }

    /* Doubling ranges [m, 2m): iterate until v < m, then reuse steps[v]
     * which was fully computed in an earlier range (barrier between ranges). */
    #pragma omp parallel
    {
        for (long m = base; m < n; m *= 2) {
            long hi = (2 * m < n) ? 2 * m : n;
            unsigned long um = (unsigned long)m;
            #pragma omp for schedule(dynamic, 1024)
            for (long i = m; i < hi; i++) {
                unsigned long v = (unsigned long)i;
                int s = 0;
                while (v >= um) {
                    if (v & 1) {
                        v = (3 * v + 1) >> 1;
                        s += 2;
                    } else {
                        int t = __builtin_ctzll(v);
                        v >>= t;
                        s += t;
                    }
                }
                steps[i] = s + steps[v];   /* v in [1, m): already final */
            }
            /* implicit barrier here */
        }
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
