/* k12_collatz: number of Collatz steps for every start value below n.
 * IRREGULAR and UNPREDICTABLE: neighbouring start values can need
 * wildly different step counts, with no pattern a static split can use. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int *steps) {
    /* 
     * The Collatz problem is highly irregular; some values of 'i' take 
     * significantly longer to converge than others. Additionally, 
     * the target machine has a heterogeneous CPU (P-cores and E-cores).
     * 
     * 'schedule(guided)' is ideal here because it handles both issues:
     * 1. It starts with large chunks to minimize synchronization overhead.
     * 2. It dynamically reduces chunk size over time, ensuring that 
     *    faster cores (P-cores) do more work and no thread is left 
     *    stuck with a computationally heavy chunk at the end.
     */
    #pragma omp parallel for schedule(guided)
    for (long i = 1; i < n; i++) {
        unsigned long v = (unsigned long)i;
        int s = 0;
        while (v != 1) {
            v = (v & 1) ? 3 * v + 1 : v / 2;
            s++;
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
