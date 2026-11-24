/* k12_collatz: number of Collatz steps for every start value below n.
 * IRREGULAR and UNPREDICTABLE: neighbouring start values can need
 * wildly different step counts, with no pattern a static split can use. */
#include "common.h"
#include <omp.h>

/* === KERNEL BEGIN === */
static void kernel(long n, int *steps) {
    /*
     * The workload of the Collatz sequence is highly irregular.
     * Because the target machine has a heterogeneous architecture (P-cores and E-cores),
     * static scheduling would lead to significant load imbalance as P-cores would 
     * finish their chunks much faster than E-cores.
     *
     * 'schedule(guided)' is used here because it dynamically balances the load by 
     * assigning larger chunks initially and smaller chunks as the loop progresses. 
     * This reduces the synchronization overhead compared to 'schedule(dynamic, 1)' 
     * while still effectively distributing work across cores of varying speeds.
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
