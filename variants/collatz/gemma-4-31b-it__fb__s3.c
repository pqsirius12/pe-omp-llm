/* k12_collatz: number of Collatz steps for every start value below n.
 * IRREGULAR and UNPREDICTABLE: neighbouring start values can need
 * wildly different step counts, with no pattern a static split can use. */
#include "common.h"

/* === KERNEL BEGIN === */
#include <omp.h>

static void kernel(long n, int *steps) {
    /* 
     * The workload for Collatz is highly irregular. schedule(guided) is used 
     * to maintain load balance while reducing the scheduling overhead compared 
     * to dynamic scheduling.
     * 
     * Algorithmic Optimization:
     * In the standard Collatz sequence, if v is odd, 3v+1 is always even. 
     * We can combine these two steps (3v+1 and the subsequent division by 2) 
     * into one operation: (3v+1)/2. This reduces the number of loop iterations 
     * and condition checks without changing the final step count.
     */
    #pragma omp parallel for schedule(guided)
    for (long i = 1; i < n; i++) {
        unsigned long v = (unsigned long)i;
        int s = 0;
        while (v != 1) {
            if (v & 1) {
                // v is odd: 3v + 1 is guaranteed to be even, so we can combine steps.
                v = (3 * v + 1) >> 1;
                s += 2;
            } else {
                // v is even
                v >>= 1;
                s++;
            }
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

