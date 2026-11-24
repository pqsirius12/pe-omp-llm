/* k12_collatz: number of Collatz steps for every start value below n.
 * IRREGULAR and UNPREDICTABLE: neighbouring start values can need
 * wildly different step counts, with no pattern a static split can use. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int *steps) {
    /* 
       The Collatz sequence has highly irregular computation times per starting value.
       We use 'schedule(dynamic, 256)' to ensure load balancing across threads.
       Dynamic scheduling is generally more robust than guided when the workload 
       distribution is extremely skewed and unpredictable.
       
       The internal loop is optimized by combining the '3v + 1' step and the 
       subsequent division by 2 (which is guaranteed to happen since 3v + 1 
       is always even when v is odd). This reduces the number of loop iterations 
       and branch checks.
    */
    #pragma omp parallel for schedule(dynamic, 256)
    for (long i = 1; i < n; i++) {
        unsigned long v = (unsigned long)i;
        int s = 0;
        while (v != 1) {
            if (v & 1) {
                /* v is odd: v -> 3v + 1 -> (3v + 1)/2. Two steps in one. */
                v = (3 * v + 1) >> 1;
                s += 2;
            } else {
                /* v is even: v -> v/2. One step. */
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
