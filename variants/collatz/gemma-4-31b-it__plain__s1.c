/* k12_collatz: number of Collatz steps for every start value below n.
 * IRREGULAR and UNPREDICTABLE: neighbouring start values can need
 * wildly different step counts, with no pattern a static split can use. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int *steps) {
    /* 
     * The Collatz sequence is highly irregular, meaning the work per iteration 
     * varies significantly and unpredictably. A 'guided' schedule is ideal here 
     * as it starts with large chunks to reduce overhead and decreases chunk size 
     * to handle load imbalance toward the end of the loop.
     */
    #pragma omp parallel for schedule(guided)
    for (long i = 1; i < n; i++) {
        unsigned long v = (unsigned long)i;
        int s = 0;
        while (v != 1) {
            if (v & 1) {
                v = 3 * v + 1;
            } else {
                v = v / 2;
            }
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
