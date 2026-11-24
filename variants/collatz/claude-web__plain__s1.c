/* k12_collatz: number of Collatz steps for every start value below n.
 * IRREGULAR and UNPREDICTABLE: neighbouring start values can need
 * wildly different step counts, with no pattern a static split can use. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int *steps) {
    if (n <= 1) return;

    /* Small prefix computed directly (serial, cheap). */
    long base = n < 4096 ? n : 4096;
    for (long i = 1; i < base; i++) {
        unsigned long v = (unsigned long)i;
        int s = 0;
        while (v != 1) {
            if (v & 1) {
                v = 3 * v + 1;
                s++;
            } else {
                int t = __builtin_ctzl(v);
                v >>= t;
                s += t;
            }
        }
        steps[i] = s;
    }
    if (base >= n) return;

    /* Doubling phases: in phase [lo, hi), all values < lo are already known.
     * Follow each trajectory until it drops below lo, then use the table. */
    #pragma omp parallel
    {
        for (long lo = base; lo < n; lo *= 2) {
            long hi = (lo * 2 < n) ? lo * 2 : n;
            const unsigned long ulo = (unsigned long)lo;

            #pragma omp for schedule(dynamic, 1024)
            for (long i = lo; i < hi; i++) {
                unsigned long v = (unsigned long)i;
                int s = 0;
                while (v >= ulo) {
                    if (v & 1) {
                        /* odd: 3v+1 is even, so do the following halving too */
                        v = (3 * v + 1) >> 1;
                        s += 2;
                    } else {
                        int t = __builtin_ctzl(v);
                        v >>= t;
                        s += t;
                    }
                }
                steps[i] = s + steps[v];
            }
            /* implicit barrier: phase results visible before next phase */
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
